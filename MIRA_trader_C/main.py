"""
MIRA_trader_C – Main Entry Point

Usage
-----
    # Backtest mode (safe, no real orders)
    python main.py --mode backtest

    # Paper trading mode (no real orders, live data)
    python main.py --mode paper

    # Live futures trading (requires API credentials via env vars)
    python main.py --mode live

    # Signal scanner only (no auto-trade – just scan and alert)
    python main.py --mode scanner

    # Override config file
    python main.py --config path/to/config.yaml --mode paper
"""
from __future__ import annotations

import argparse
import asyncio
import logging
import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config, load_symbol_config
from monitoring.logger import setup_logging


logger = logging.getLogger(__name__)


# ──────────────────────────────────────────────────────────────
# Builder helpers
# ──────────────────────────────────────────────────────────────

def _build_data_provider(cfg: dict):
    provider_name = cfg.get("data", {}).get("provider", "ccxt")
    if provider_name == "csv":
        from data.csv_provider import CSVDataProvider
        return CSVDataProvider(cfg["data"].get("csv_dir", "data/csv"))
    from data.ccxt_provider import CCXTDataProvider
    return CCXTDataProvider(cfg.get("exchange", {}))


def _build_strategy(cfg: dict):
    strategy_cfg = cfg.get("strategy", {})
    active = strategy_cfg.get("active", "multi_strategy")
    if active == "multi_strategy":
        from strategies.multi_strategy import MultiStrategy
        return MultiStrategy(strategy_cfg)
    from strategies.multi_strategy import build_strategy
    return build_strategy(active, strategy_cfg.get(active, {}))


def _build_broker(cfg: dict, mode: str):
    futures_enabled = cfg.get("exchange", {}).get("futures", False)
    if mode in ("paper", "backtest", "scanner"):
        if futures_enabled:
            from execution.futures_broker import PaperFuturesBroker
            initial_capital = cfg.get("risk", {}).get("initial_capital", 1000.0)
            return PaperFuturesBroker({"USDT": initial_capital}, cfg.get("futures", {}))
        from execution.paper_broker import PaperBroker
        initial_capital = cfg.get("risk", {}).get("initial_capital", 1000.0)
        return PaperBroker({"USDT": initial_capital})
    # live mode
    if futures_enabled:
        from execution.futures_broker import FuturesBroker
        return FuturesBroker(cfg.get("exchange", {}))
    from execution.live_broker import LiveBroker
    return LiveBroker(cfg.get("exchange", {}))


def _build_notifier(cfg: dict):
    from notifications.telegram import build_notifier
    return build_notifier(cfg)


# ──────────────────────────────────────────────────────────────
# Backtest mode
# ──────────────────────────────────────────────────────────────

def run_backtest(cfg: dict) -> None:
    """Run a full backtest over historical data."""
    from backtesting.engine import BacktestEngine

    data_provider = _build_data_provider(cfg)

    data_cfg = cfg.get("data", {})
    symbols = data_cfg.get("symbols", ["BTC/USDT"])
    timeframe = data_cfg.get("timeframe", "1h")
    lookback = data_cfg.get("lookback_candles", 500)

    for symbol in symbols:
        # Load per-symbol config
        sym_cfg = load_symbol_config(symbol, cfg)
        strategy = _build_strategy(sym_cfg)

        logger.info("Running backtest for %s [%s]", symbol, timeframe)
        df = data_provider.fetch_ohlcv(symbol, timeframe, lookback)
        engine = BacktestEngine(sym_cfg, strategy)
        results = engine.run(df, symbol)
        summary = results.summary()
        logger.info("Backtest complete for %s", symbol)
        print(f"\n─── Backtest: {symbol} ───")
        for k, v in summary.items():
            print(f"  {k}: {v}")


# ──────────────────────────────────────────────────────────────
# Paper / Live trading mode (async multi-worker)
# ──────────────────────────────────────────────────────────────

async def run_trading_async(cfg: dict, mode: str) -> None:
    """Run paper or live trading with per-symbol async workers."""
    from risk.engine import RiskEngine
    from monitoring.metrics import MetricsTracker
    from bot.worker import BotWorker
    from signal_scanner.scanner import SignalScanner

    data_provider = _build_data_provider(cfg)
    broker = _build_broker(cfg, mode)
    risk_engine = RiskEngine(
        cfg.get("risk", {}),
        cfg.get("risk", {}).get("initial_capital", 1000.0),
    )
    metrics = MetricsTracker(cfg.get("monitoring", {}).get("metrics_file", "logs/metrics.json"))
    notifier = _build_notifier(cfg)

    symbols = cfg.get("data", {}).get("symbols", ["BTC/USDT"])
    logger.info("MIRA_trader_C %s trading started. Symbols: %s", mode, symbols)

    if notifier:
        notifier.send_system_status(f"Started ({mode})", f"Symbols: {', '.join(symbols)}")

    # Build per-symbol workers
    workers = [
        BotWorker(
            symbol=symbol,
            global_cfg=cfg,
            data_provider=data_provider,
            broker=broker,
            risk_engine=risk_engine,
            metrics=metrics,
            notifier=notifier,
        )
        for symbol in symbols
    ]

    # Build scanner if enabled
    scanner_tasks = []
    if cfg.get("scanner", {}).get("enabled", True):
        global_strategy = _build_strategy(cfg)
        scanner = SignalScanner(cfg, data_provider, global_strategy, notifier)
        scanner_tasks.append(asyncio.create_task(scanner.run()))

    # Run all workers concurrently
    worker_tasks = [asyncio.create_task(w.run()) for w in workers]

    try:
        await asyncio.gather(*worker_tasks, *scanner_tasks)
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass
    finally:
        for task in worker_tasks + scanner_tasks:
            task.cancel()

        m = metrics.get_metrics()
        print(f"\n─── Session Summary ───")
        print(f"Total trades : {m.total_trades}")
        print(f"Win rate     : {m.win_rate:.1%}")
        print(f"Total PnL    : ${m.total_pnl:.2f}")
        print(f"Max drawdown : {m.max_drawdown_pct:.2f}%")

        if notifier:
            notifier.send_summary({
                "total_trades": m.total_trades,
                "win_rate": f"{m.win_rate:.1%}",
                "total_pnl": f"${m.total_pnl:.2f}",
                "max_drawdown": f"{m.max_drawdown_pct:.2f}%",
            })


def run_paper(cfg: dict) -> None:
    try:
        asyncio.run(run_trading_async(cfg, "paper"))
    except KeyboardInterrupt:
        logger.info("Paper trading stopped by user.")


def run_live(cfg: dict) -> None:
    logger.warning("Live trading requires API credentials (MIRA_API_KEY, MIRA_API_SECRET).")
    try:
        asyncio.run(run_trading_async(cfg, "live"))
    except KeyboardInterrupt:
        logger.info("Live trading stopped by user.")


# ──────────────────────────────────────────────────────────────
# Scanner-only mode
# ──────────────────────────────────────────────────────────────

async def run_scanner_async(cfg: dict) -> None:
    """Run only the signal scanner – no trade execution."""
    from signal_scanner.scanner import SignalScanner

    data_provider = _build_data_provider(cfg)
    strategy = _build_strategy(cfg)
    notifier = _build_notifier(cfg)

    symbols = cfg.get("data", {}).get("symbols", ["BTC/USDT"])
    logger.info("MIRA_trader_C signal scanner started. Symbols: %s", symbols)

    if notifier:
        notifier.send_system_status("Scanner started", f"Symbols: {', '.join(symbols)}")

    scanner = SignalScanner(cfg, data_provider, strategy, notifier)
    try:
        await scanner.run()
    except (asyncio.CancelledError, KeyboardInterrupt):
        logger.info("Scanner stopped.")
        if notifier:
            notifier.send_system_status("Scanner stopped")


def run_scanner(cfg: dict) -> None:
    try:
        asyncio.run(run_scanner_async(cfg))
    except KeyboardInterrupt:
        logger.info("Scanner stopped by user.")


# ──────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────

def _timeframe_to_seconds(tf: str) -> int:
    units = {"m": 60, "h": 3600, "d": 86400}
    if tf[-1] in units:
        return int(tf[:-1]) * units[tf[-1]]
    return 3600


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="MIRA_trader_C – Crypto Futures Trading Bot")
    parser.add_argument("--config", default=None, help="Path to config YAML file")
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live", "scanner"],
        default=None,
        help="Override runtime mode from config",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.mode:
        cfg["mode"] = args.mode

    setup_logging(cfg)
    mode = cfg.get("mode", "paper")
    logger.info("MIRA_trader_C starting in mode: %s", mode)

    if mode == "backtest":
        run_backtest(cfg)
    elif mode == "paper":
        run_paper(cfg)
    elif mode == "live":
        run_live(cfg)
    elif mode == "scanner":
        run_scanner(cfg)
    else:
        logger.error("Unknown mode: %s", mode)
        sys.exit(1)


if __name__ == "__main__":
    main()

