"""
MIRA_trader_C – Main Entry Point

Usage
-----
    # Backtest mode (safe, no real orders)
    python main.py --mode backtest

    # Paper trading mode (no real orders, live data)
    python main.py --mode paper

    # Live trading (requires API credentials via env vars)
    python main.py --mode live

    # Override config file
    python main.py --config path/to/config.yaml --mode paper
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Allow running from the project root
sys.path.insert(0, str(Path(__file__).parent))

from config import load_config
from monitoring.logger import setup_logging


logger = logging.getLogger(__name__)


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
    if mode in ("paper", "backtest"):
        from execution.paper_broker import PaperBroker
        initial_capital = cfg.get("risk", {}).get("initial_capital", 1000.0)
        return PaperBroker({"USDT": initial_capital})
    from execution.live_broker import LiveBroker
    return LiveBroker(cfg.get("exchange", {}))


def run_backtest(cfg: dict) -> None:
    """Run a full backtest over historical data."""
    from backtesting.engine import BacktestEngine

    data_provider = _build_data_provider(cfg)
    strategy = _build_strategy(cfg)

    data_cfg = cfg.get("data", {})
    symbols = data_cfg.get("symbols", ["BTC/USDT"])
    timeframe = data_cfg.get("timeframe", "1h")
    lookback = data_cfg.get("lookback_candles", 500)

    for symbol in symbols:
        logger.info("Running backtest for %s [%s]", symbol, timeframe)
        df = data_provider.fetch_ohlcv(symbol, timeframe, lookback)
        engine = BacktestEngine(cfg, strategy)
        results = engine.run(df, symbol)
        summary = results.summary()
        logger.info("Backtest complete for %s", symbol)
        for k, v in summary.items():
            print(f"  {k}: {v}")


def run_paper(cfg: dict) -> None:
    """Run paper trading loop (live data, no real orders)."""
    import time
    from risk.engine import RiskEngine
    from monitoring.metrics import MetricsTracker, TradeRecord
    from utils.helpers import now_utc

    data_provider = _build_data_provider(cfg)
    strategy = _build_strategy(cfg)
    broker = _build_broker(cfg, "paper")
    risk_engine = RiskEngine(
        cfg.get("risk", {}),
        cfg.get("risk", {}).get("initial_capital", 1000.0),
    )
    metrics = MetricsTracker(cfg.get("monitoring", {}).get("metrics_file", "logs/metrics.json"))

    data_cfg = cfg.get("data", {})
    symbols = data_cfg.get("symbols", ["BTC/USDT"])
    timeframe = data_cfg.get("timeframe", "1h")
    lookback = data_cfg.get("lookback_candles", 500)

    open_positions: dict = {}
    capital = cfg.get("risk", {}).get("initial_capital", 1000.0)

    logger.info("MIRA_trader_C paper trading started. Symbols: %s", symbols)

    try:
        while True:
            for symbol in symbols:
                try:
                    df = data_provider.fetch_ohlcv(symbol, timeframe, lookback)
                    signal = strategy.generate_signal(df)
                    price = float(df["close"].iloc[-1])

                    logger.info("[%s] price=%.4f signal=%s", symbol, price, signal.action)

                    if signal.action == "buy" and symbol not in open_positions:
                        approved, reason = risk_engine.approve_trade(
                            symbol, signal, price, open_positions
                        )
                        if approved:
                            qty, sl, tp = risk_engine.compute_sizing(price)
                            order = broker.place_market_order(symbol, "buy", qty)
                            from execution.paper_broker import PaperBroker
                            if isinstance(broker, PaperBroker):
                                broker.fill_paper_order(order, price)
                            open_positions[symbol] = {
                                "entry_price": price,
                                "qty": qty,
                                "sl": sl,
                                "tp": tp,
                                "entry_time": now_utc(),
                            }
                            logger.info("[%s] OPENED paper position qty=%.6f @ %.4f", symbol, qty, price)
                        else:
                            logger.debug("[%s] Trade rejected: %s", symbol, reason)

                    elif symbol in open_positions:
                        pos = open_positions[symbol]
                        if price <= pos["sl"] or price >= pos["tp"] or signal.action == "sell":
                            pnl = (price - pos["entry_price"]) * pos["qty"]
                            pnl_pct = pnl / capital * 100
                            exit_reason = (
                                "stop_loss" if price <= pos["sl"]
                                else "take_profit" if price >= pos["tp"]
                                else "strategy_signal"
                            )
                            order = broker.place_market_order(symbol, "sell", pos["qty"])
                            from execution.paper_broker import PaperBroker
                            if isinstance(broker, PaperBroker):
                                broker.fill_paper_order(order, price)
                            risk_engine.record_trade_result(pnl_pct)
                            capital += pnl
                            risk_engine.update_capital(capital)

                            metrics.record_trade(TradeRecord(
                                symbol=symbol,
                                side="long",
                                entry_price=pos["entry_price"],
                                exit_price=price,
                                qty=pos["qty"],
                                pnl=pnl,
                                pnl_pct=pnl_pct,
                                timestamp=now_utc().isoformat(),
                                exit_reason=exit_reason,
                            ))
                            del open_positions[symbol]
                            logger.info(
                                "[%s] CLOSED position @ %.4f pnl=%.2f (%.2f%%) reason=%s",
                                symbol, price, pnl, pnl_pct, exit_reason
                            )

                except Exception as exc:
                    logger.error("Error processing %s: %s", symbol, exc, exc_info=True)

            # Sleep until next candle close (timeframe-based interval)
            tf_seconds = _timeframe_to_seconds(timeframe)
            logger.debug("Sleeping %ds until next bar...", tf_seconds)
            time.sleep(tf_seconds)

    except KeyboardInterrupt:
        logger.info("Paper trading stopped by user.")
        m = metrics.get_metrics()
        print(f"\n─── Session Summary ───")
        print(f"Total trades : {m.total_trades}")
        print(f"Win rate     : {m.win_rate:.1%}")
        print(f"Total PnL    : ${m.total_pnl:.2f}")
        print(f"Max drawdown : {m.max_drawdown_pct:.2f}%")


def _timeframe_to_seconds(tf: str) -> int:
    units = {"m": 60, "h": 3600, "d": 86400}
    if tf[-1] in units:
        return int(tf[:-1]) * units[tf[-1]]
    return 3600


def main() -> None:
    parser = argparse.ArgumentParser(description="MIRA_trader_C – Crypto Trading Bot")
    parser.add_argument("--config", default=None, help="Path to config YAML file")
    parser.add_argument(
        "--mode",
        choices=["backtest", "paper", "live"],
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
        logger.warning("Live trading requires API credentials (MIRA_API_KEY, MIRA_API_SECRET).")
        # Live uses the same loop as paper but with LiveBroker
        cfg["_broker_mode"] = "live"
        run_paper(cfg)
    else:
        logger.error("Unknown mode: %s", mode)
        sys.exit(1)


if __name__ == "__main__":
    main()
