"""
MIRA_trader_C – Web Dashboard Server (FastAPI)

Serves the web UI and exposes REST + SSE APIs so the browser can:
  • Monitor bot status, metrics, and trade history
  • Control the bot (start / stop / kill-switch)
  • Edit configuration live
  • Trigger and view backtest results
  • Stream real-time log events

Run:
    cd MIRA_trader_C
    python web/server.py
  or
    uvicorn web.server:app --host 0.0.0.0 --port 8080 --reload
"""
from __future__ import annotations

import asyncio
import json
import logging
import os
import subprocess
import sys
import threading
import time
from collections import deque
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, AsyncGenerator, Dict, List, Optional

import yaml

# ── FastAPI ────────────────────────────────────────────────────
from fastapi import BackgroundTasks, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import HTMLResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

# ── Project imports ────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from config import load_config
from monitoring.logger import setup_logging

logger = logging.getLogger(__name__)

# ─────────────────────────────────────────────────────────────
# App factory
# ─────────────────────────────────────────────────────────────
app = FastAPI(
    title="MIRA trader C – Dashboard",
    version="1.0.0",
    docs_url="/api/docs",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC_DIR = Path(__file__).parent / "static"
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ─────────────────────────────────────────────────────────────
# Global state
# ─────────────────────────────────────────────────────────────
_CONFIG_PATH = PROJECT_ROOT / "config" / "config.yaml"
_METRICS_PATH = PROJECT_ROOT / "logs" / "metrics.json"
_LOG_PATH = PROJECT_ROOT / "logs" / "mira.log"

_bot_process: Optional[subprocess.Popen] = None
_bot_start_time: Optional[float] = None
_sse_clients: List[asyncio.Queue] = []
_recent_logs: deque = deque(maxlen=200)

# ─────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────

def _cfg() -> dict:
    try:
        return load_config(_CONFIG_PATH)
    except Exception:
        return {}


def _read_metrics() -> dict:
    if not _METRICS_PATH.exists():
        return _default_metrics()
    try:
        with open(_METRICS_PATH, "r") as fh:
            return json.load(fh)
    except Exception:
        return _default_metrics()


def _default_metrics() -> dict:
    return {
        "total_trades": 0,
        "winning_trades": 0,
        "losing_trades": 0,
        "total_pnl": 0.0,
        "peak_pnl": 0.0,
        "max_drawdown_pct": 0.0,
        "current_drawdown_pct": 0.0,
        "win_rate": "0.0%",
        "profit_factor": "0.00",
        "trade_history": [],
    }


def _bot_running() -> bool:
    global _bot_process
    if _bot_process is None:
        return False
    return _bot_process.poll() is None


def _uptime_str() -> str:
    if not _bot_running() or _bot_start_time is None:
        return "00:00:00"
    elapsed = int(time.time() - _bot_start_time)
    h, rem = divmod(elapsed, 3600)
    m, s = divmod(rem, 60)
    return f"{h:02d}:{m:02d}:{s:02d}"


async def _broadcast(event: str, data: Any) -> None:
    payload = f"event: {event}\ndata: {json.dumps(data)}\n\n"
    dead = []
    for q in _sse_clients:
        try:
            q.put_nowait(payload)
        except asyncio.QueueFull:
            dead.append(q)
    for q in dead:
        if q in _sse_clients:
            _sse_clients.remove(q)


def _tail_log(n: int = 100) -> List[str]:
    if not _LOG_PATH.exists():
        return []
    try:
        lines = _LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        return lines[-n:]
    except Exception:
        return []


# ─────────────────────────────────────────────────────────────
# Background polling thread (pushes SSE updates every 5s)
# ─────────────────────────────────────────────────────────────
_loop: Optional[asyncio.AbstractEventLoop] = None


def _bg_thread() -> None:
    global _loop
    while True:
        time.sleep(5)
        if _loop and not _loop.is_closed():
            asyncio.run_coroutine_threadsafe(
                _broadcast("metrics", _read_metrics()), _loop
            )
            asyncio.run_coroutine_threadsafe(
                _broadcast("status", {
                    "running": _bot_running(),
                    "uptime": _uptime_str(),
                    "mode": _cfg().get("mode", "paper"),
                }),
                _loop,
            )


@app.on_event("startup")
async def _startup() -> None:
    global _loop
    _loop = asyncio.get_event_loop()
    t = threading.Thread(target=_bg_thread, daemon=True)
    t.start()
    setup_logging(_cfg())
    logger.info("MIRA Web Dashboard started.")


# ─────────────────────────────────────────────────────────────
# Root – serve SPA
# ─────────────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def root() -> HTMLResponse:
    html_file = STATIC_DIR / "index.html"
    return HTMLResponse(content=html_file.read_text(encoding="utf-8"))


# ─────────────────────────────────────────────────────────────
# SSE stream
# ─────────────────────────────────────────────────────────────
@app.get("/api/v1/stream")
async def sse_stream(request: Request) -> StreamingResponse:
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    _sse_clients.append(q)

    async def generator() -> AsyncGenerator[str, None]:
        try:
            # Send initial state immediately
            yield f"event: metrics\ndata: {json.dumps(_read_metrics())}\n\n"
            yield (
                f"event: status\ndata: {json.dumps({'running': _bot_running(), 'uptime': _uptime_str(), 'mode': _cfg().get('mode', 'paper')})}\n\n"
            )
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=30)
                    yield msg
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            if q in _sse_clients:
                _sse_clients.remove(q)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
            "Connection": "keep-alive",
        },
    )


# ─────────────────────────────────────────────────────────────
# Status
# ─────────────────────────────────────────────────────────────
@app.get("/api/v1/status")
async def get_status() -> dict:
    cfg = _cfg()
    metrics = _read_metrics()
    capital = cfg.get("risk", {}).get("initial_capital", 1000.0)
    return {
        "running": _bot_running(),
        "mode": cfg.get("mode", "paper"),
        "uptime": _uptime_str(),
        "capital": capital,
        "total_pnl": metrics.get("total_pnl", 0.0),
        "kill_switch": cfg.get("risk", {}).get("kill_switch", False),
        "symbols": cfg.get("data", {}).get("symbols", []),
        "timeframe": cfg.get("data", {}).get("timeframe", "1h"),
        "strategy": cfg.get("strategy", {}).get("active", "multi_strategy"),
        "timestamp": datetime.now(tz=timezone.utc).isoformat(),
    }


# ─────────────────────────────────────────────────────────────
# Metrics
# ─────────────────────────────────────────────────────────────
@app.get("/api/v1/metrics")
async def get_metrics() -> dict:
    return _read_metrics()


# ─────────────────────────────────────────────────────────────
# Equity curve (derived from trade_history)
# ─────────────────────────────────────────────────────────────
@app.get("/api/v1/equity")
async def get_equity() -> dict:
    m = _read_metrics()
    history = m.get("trade_history", [])
    cfg = _cfg()
    capital = cfg.get("risk", {}).get("initial_capital", 1000.0)

    points = []
    running_capital = capital
    for t in history:
        running_capital += t.get("pnl", 0)
        points.append({
            "ts": t.get("timestamp", ""),
            "value": round(running_capital, 2),
            "pnl": round(t.get("pnl", 0), 2),
        })

    if not points:
        # Generate a flat baseline
        now = datetime.now(tz=timezone.utc).isoformat()
        points = [{"ts": now, "value": capital, "pnl": 0}]

    return {"equity": points, "initial_capital": capital}


# ─────────────────────────────────────────────────────────────
# Trades
# ─────────────────────────────────────────────────────────────
@app.get("/api/v1/trades")
async def get_trades(limit: int = 50, offset: int = 0) -> dict:
    m = _read_metrics()
    history = m.get("trade_history", [])
    total = len(history)
    page = list(reversed(history))[offset: offset + limit]
    return {"trades": page, "total": total, "limit": limit, "offset": offset}


# ─────────────────────────────────────────────────────────────
# Logs
# ─────────────────────────────────────────────────────────────
@app.get("/api/v1/logs")
async def get_logs(n: int = 100) -> dict:
    return {"lines": _tail_log(n)}


# ─────────────────────────────────────────────────────────────
# Config
# ─────────────────────────────────────────────────────────────
@app.get("/api/v1/config")
async def get_config() -> dict:
    try:
        with open(_CONFIG_PATH, "r") as fh:
            return yaml.safe_load(fh) or {}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


class ConfigUpdate(BaseModel):
    config: Dict[str, Any]


@app.post("/api/v1/config")
async def update_config(body: ConfigUpdate) -> dict:
    try:
        # Safety: never allow writing blank API keys
        cfg = body.config
        if "exchange" in cfg:
            cfg["exchange"].pop("api_key", None)
            cfg["exchange"].pop("api_secret", None)

        with open(_CONFIG_PATH, "w") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)

        await _broadcast("config_updated", {"ok": True})
        return {"ok": True, "message": "Configuration saved."}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────
# Kill-switch toggle
# ─────────────────────────────────────────────────────────────
@app.post("/api/v1/bot/killswitch")
async def toggle_kill_switch() -> dict:
    try:
        with open(_CONFIG_PATH, "r") as fh:
            cfg = yaml.safe_load(fh)
        current = cfg.get("risk", {}).get("kill_switch", False)
        cfg.setdefault("risk", {})["kill_switch"] = not current
        with open(_CONFIG_PATH, "w") as fh:
            yaml.dump(cfg, fh, default_flow_style=False, allow_unicode=True)
        new_state = not current
        await _broadcast("kill_switch", {"active": new_state})
        return {"kill_switch": new_state}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────
# Bot start / stop
# ─────────────────────────────────────────────────────────────

# Allowlist map: user input → literal command-line value.
# Using a dict lookup (not the raw user string) breaks CodeQL taint flow
# and guarantees no command-line injection regardless of the input value.
_MODE_MAP: dict = {"paper": "paper", "live": "live", "backtest": "backtest"}


class BotStartRequest(BaseModel):
    mode: str = "paper"


@app.post("/api/v1/bot/start")
async def bot_start(body: BotStartRequest, background_tasks: BackgroundTasks) -> dict:
    global _bot_process, _bot_start_time
    if _bot_running():
        return {"ok": False, "message": "Bot is already running."}

    # Resolve user input to a fixed literal from the allowlist map.
    # This prevents command-line injection: only the dict's own string values
    # ever reach the subprocess command array.
    safe_mode: str | None = _MODE_MAP.get(body.mode)
    if safe_mode is None:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid mode. Allowed: {sorted(_MODE_MAP)}",
        )

    try:
        env = os.environ.copy()
        cmd = [sys.executable, str(PROJECT_ROOT / "main.py"), "--mode", safe_mode]
        _bot_process = subprocess.Popen(
            cmd,
            cwd=str(PROJECT_ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
        )
        _bot_start_time = time.time()
        background_tasks.add_task(_stream_subprocess_output)
        await _broadcast("bot_status", {"running": True, "mode": safe_mode})
        return {"ok": True, "message": f"Bot started in {safe_mode} mode.", "pid": _bot_process.pid}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


async def _stream_subprocess_output() -> None:
    global _bot_process
    if _bot_process and _bot_process.stdout:
        for line in _bot_process.stdout:
            stripped = line.rstrip()
            _recent_logs.append(stripped)
            await _broadcast("log", {"line": stripped})


@app.post("/api/v1/bot/stop")
async def bot_stop() -> dict:
    global _bot_process, _bot_start_time
    if not _bot_running():
        return {"ok": False, "message": "Bot is not running."}
    _bot_process.terminate()
    try:
        _bot_process.wait(timeout=5)
    except subprocess.TimeoutExpired:
        _bot_process.kill()
    _bot_start_time = None
    await _broadcast("bot_status", {"running": False})
    return {"ok": True, "message": "Bot stopped."}


# ─────────────────────────────────────────────────────────────
# Backtest
# ─────────────────────────────────────────────────────────────
class BacktestRequest(BaseModel):
    symbol: str = "BTC/USDT"
    timeframe: str = "1h"
    strategy: str = "trend_follow"
    start_date: str = "2024-01-01"
    end_date: str = "2024-12-31"
    initial_capital: float = 1000.0
    fee_rate: float = 0.001
    slippage_pct: float = 0.05


@app.post("/api/v1/backtest")
async def run_backtest(body: BacktestRequest) -> dict:
    """Run a quick backtest on the server and return results."""
    try:
        from data.csv_provider import CSVDataProvider
        from strategies.trend_follow import TrendFollowStrategy
        from strategies.mean_reversion import MeanReversionStrategy
        from strategies.breakout import BreakoutStrategy
        from strategies.multi_strategy import MultiStrategy
        from backtesting.engine import BacktestEngine

        strategy_map = {
            "trend_follow": TrendFollowStrategy,
            "mean_reversion": MeanReversionStrategy,
            "breakout": BreakoutStrategy,
        }

        cfg = _cfg()
        cfg["backtest"] = {
            "fee_rate": body.fee_rate,
            "slippage_pct": body.slippage_pct,
        }
        cfg["risk"]["initial_capital"] = body.initial_capital
        cfg["risk"]["cooldown_after_loss_trades"] = 0
        cfg["risk"]["max_daily_loss_pct"] = 100.0

        csv_dir = str(PROJECT_ROOT / "data" / "csv")
        dp = CSVDataProvider(csv_dir)

        try:
            df = dp.fetch_ohlcv(body.symbol, body.timeframe, 5000)
        except FileNotFoundError:
            return {
                "error": (
                    f"CSV file for {body.symbol} [{body.timeframe}] not found. "
                    "Place a CSV file in MIRA_trader_C/data/csv/ "
                    f"named {body.symbol.replace('/', '_')}_{body.timeframe}.csv"
                )
            }

        # Date filter
        import pandas as pd
        if body.start_date:
            df = df[df.index >= pd.Timestamp(body.start_date, tz="UTC")]
        if body.end_date:
            df = df[df.index <= pd.Timestamp(body.end_date, tz="UTC")]

        if len(df) < 50:
            return {"error": "Not enough data for the selected date range."}

        strat_params = cfg.get("strategy", {})
        if body.strategy == "multi_strategy":
            strat = MultiStrategy(strat_params)
        else:
            cls = strategy_map.get(body.strategy)
            if cls is None:
                return {"error": f"Unknown strategy: {body.strategy}"}
            strat = cls(strat_params.get(body.strategy, {}))

        engine = BacktestEngine(cfg, strat)
        results = engine.run(df, body.symbol)
        summary = results.summary()

        # Build equity curve for chart
        equity = [
            {"ts": str(ts), "value": float(v)}
            for ts, v in results.equity_curve.items()
        ]

        # Build trade list
        trades = [
            {
                "entry_time": str(t.entry_time),
                "exit_time": str(t.exit_time),
                "entry_price": round(t.entry_price, 4),
                "exit_price": round(t.exit_price, 4) if t.exit_price else None,
                "pnl": round(t.pnl, 2),
                "exit_reason": t.exit_reason,
            }
            for t in results.trades
        ]

        return {
            "summary": summary,
            "equity": equity[-500:],   # cap at 500 pts for payload size
            "trades": trades[-100:],
            "candle_count": len(df),
        }

    except Exception as exc:
        logger.exception("Backtest error")
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────
# Signal Scanner feed
# ─────────────────────────────────────────────────────────────
_SIGNAL_LOG_PATH = PROJECT_ROOT / "logs" / "signal_log.jsonl"


def _read_signal_log(limit: int = 100) -> list:
    """Read recent signals from the JSONL signal log file."""
    if not _SIGNAL_LOG_PATH.exists():
        return []
    try:
        lines = _SIGNAL_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        results = []
        for line in reversed(lines[-limit * 2:]):
            line = line.strip()
            if line:
                try:
                    results.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if len(results) >= limit:
                break
        return results
    except Exception:
        return []


@app.get("/api/v1/signals")
async def get_signals(limit: int = 50, symbol: Optional[str] = None) -> dict:
    """Return recent scanner signals from signal_log.jsonl and in-process store."""
    try:
        from signal_scanner.scanner import get_recent_signals
        signals = get_recent_signals(limit * 2)
    except Exception:
        signals = []

    # Fall back to file-based signals if in-process store is empty
    if not signals:
        signals = _read_signal_log(limit * 2)

    if symbol:
        signals = [s for s in signals if s.get("symbol") == symbol]

    return {
        "signals": signals[-limit:],
        "total": len(signals),
    }


@app.get("/api/v1/signals/stream")
async def signals_sse_stream(request: Request) -> StreamingResponse:
    """SSE stream that pushes new scanner signals in real-time."""
    q: asyncio.Queue = asyncio.Queue(maxsize=50)
    _sse_clients.append(q)

    async def generator() -> AsyncGenerator[str, None]:
        # Send last 10 signals immediately
        recent = _read_signal_log(10)
        if recent:
            yield f"event: signals\ndata: {json.dumps(recent)}\n\n"
        try:
            while True:
                if await request.is_disconnected():
                    break
                try:
                    msg = await asyncio.wait_for(q.get(), timeout=30)
                    if '"type": "signal"' in msg or "action" in msg:
                        yield msg
                    else:
                        yield ": keepalive\n\n"
                except asyncio.TimeoutError:
                    yield ": keepalive\n\n"
        finally:
            if q in _sse_clients:
                _sse_clients.remove(q)

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no", "Connection": "keep-alive"},
    )


# ─────────────────────────────────────────────────────────────
# Open Positions
# ─────────────────────────────────────────────────────────────
_open_positions_store: Dict[str, Any] = {}


def update_positions_store(positions: Dict[str, Any]) -> None:
    """Called by the bot worker to update the shared positions state."""
    _open_positions_store.clear()
    _open_positions_store.update(positions)


@app.get("/api/v1/positions")
async def get_positions() -> dict:
    """Return currently open positions for all symbols."""
    return {"positions": _open_positions_store}


# ─────────────────────────────────────────────────────────────
# Advanced Analytics
# ─────────────────────────────────────────────────────────────
@app.get("/api/v1/analytics")
async def get_analytics(timeframe: str = "1h") -> dict:
    """Return advanced performance analytics (Sharpe, Sortino, etc.)."""
    try:
        from monitoring.analytics import compute_analytics, compute_per_symbol_analytics
        from monitoring.metrics import Metrics
        import json as _json

        m = _read_metrics()
        if not m.get("trade_history"):
            from monitoring.analytics import _empty_analytics
            return {"overall": _empty_analytics(), "per_symbol": {}}

        # Reconstruct Metrics from persisted dict
        from monitoring.metrics import TradeRecord
        metrics = Metrics()
        metrics.total_trades = m.get("total_trades", 0)
        metrics.winning_trades = m.get("winning_trades", 0)
        metrics.losing_trades = m.get("losing_trades", 0)
        metrics.total_pnl = m.get("total_pnl", 0.0)
        metrics.peak_pnl = m.get("peak_pnl", 0.0)
        metrics.max_drawdown_pct = m.get("max_drawdown_pct", 0.0)

        for t in m.get("trade_history", []):
            metrics.trade_history.append(TradeRecord(
                symbol=t.get("symbol", ""),
                side=t.get("side", ""),
                entry_price=t.get("entry_price", 0.0),
                exit_price=t.get("exit_price", 0.0),
                qty=t.get("qty", 0.0),
                pnl=t.get("pnl", 0.0),
                pnl_pct=t.get("pnl_pct", 0.0),
                timestamp=t.get("timestamp", ""),
                exit_reason=t.get("exit_reason", ""),
            ))

        overall = compute_analytics(metrics, timeframe)
        per_symbol = compute_per_symbol_analytics(metrics, timeframe)

        return {"overall": overall, "per_symbol": per_symbol}
    except Exception as exc:
        logger.exception("Analytics error")
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────
# Risk log
# ─────────────────────────────────────────────────────────────
_RISK_LOG_PATH = PROJECT_ROOT / "logs" / "risk_log.jsonl"


@app.get("/api/v1/risk/events")
async def get_risk_events(limit: int = 50) -> dict:
    """Return recent risk events from risk_log.jsonl."""
    if not _RISK_LOG_PATH.exists():
        return {"events": []}
    try:
        lines = _RISK_LOG_PATH.read_text(encoding="utf-8", errors="replace").splitlines()
        events = []
        for line in reversed(lines[-limit * 2:]):
            line = line.strip()
            if line:
                try:
                    events.append(json.loads(line))
                except json.JSONDecodeError:
                    continue
            if len(events) >= limit:
                break
        return {"events": events}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


# ─────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(
        "web.server:app",
        host="0.0.0.0",
        port=8080,
        reload=True,
        reload_dirs=[str(PROJECT_ROOT)],
        app_dir=str(PROJECT_ROOT),
    )
