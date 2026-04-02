# MIRA_trader_C

A production-oriented, **free/low-cost** crypto **futures** trading bot built in Python.

---

## Features

| Layer | What it does |
|---|---|
| **Data** | Fetches free public OHLCV candles via [CCXT](https://github.com/ccxt/ccxt) (100+ exchanges) or local CSV |
| **Indicators** | EMA, SMA, RSI, MACD, Bollinger Bands, ATR, ADX – pure pandas/numpy |
| **Strategies** | Trend-following, Mean-reversion, Breakout, **Volatility (BB Squeeze)**, Multi-strategy voting |
| **Regime Detection** | `RegimeDetector` classifies market as `trending / ranging / volatile / choppy` using ADX + ATR |
| **Multi-strategy** | Regime-aware voting: preferred strategies for detected regime cast double votes |
| **Per-symbol Config** | Each trading pair can override strategy, risk, and leverage via `config/symbols/<SYM>.yaml` |
| **Futures Support** | Long/Short signals, `FuturesBroker` (leverage, isolated/cross margin, liquidation price) |
| **Signal Scanner** | Standalone scan loop – alerts without auto-trading; outputs `logs/signal_log.jsonl` |
| **Telegram Alerts** | Real-time notifications: signal, order filled, risk event, error, hourly summary |
| **Async Bot Workers** | One `BotWorker` per symbol, all running concurrently via `asyncio` |
| **Risk Engine** | Daily loss, cooldown, kill switch, leverage cap, liquidation buffer, portfolio exposure cap |
| **Futures Backtest** | Leverage × PnL, funding rate charges, per-candle liquidation check, short positions |
| **Advanced Analytics** | Sharpe, Sortino, Calmar, Profit Factor, Expectancy, per-symbol breakdown |
| **Web Dashboard** | FastAPI server with SSE stream, signals feed, positions, analytics endpoints |
| **Monitoring** | Structured JSON logging, metrics tracker, `logs/risk_log.jsonl` |

---

## Quick Start

### 1. Install dependencies

```bash
cd MIRA_trader_C
pip install -r requirements.txt
```

### 2. Configure

Edit `config/config.yaml`.  
**Never put API keys in config files.** Use environment variables:

```bash
export MIRA_API_KEY="your_exchange_key"
export MIRA_API_SECRET="your_exchange_secret"

# Optional – Telegram notifications
export MIRA_TELEGRAM_TOKEN="your_bot_token"
export MIRA_TELEGRAM_CHAT_ID="your_chat_id"
```

### 3. Run a backtest (free, no account needed)

```bash
python main.py --mode backtest
```

### 4. Run paper trading (live data, no real orders)

```bash
python main.py --mode paper
```

### 5. Run signal scanner only (no trading, just alerts)

```bash
python main.py --mode scanner
```

### 6. Launch web dashboard

```bash
python web/run.py
# Open http://localhost:8080
```

### 7. Run unit tests

```bash
pytest tests/ -v
```

---

## Project Structure

```
MIRA_trader_C/
├── config/
│   ├── config.yaml              # Global defaults + all parameters
│   ├── symbols/                 # Per-symbol overrides
│   │   ├── BTC_USDT.yaml        # Override strategy/risk/leverage for BTC/USDT
│   │   └── ETH_USDT.yaml
│   └── __init__.py              # load_config() + load_symbol_config() (deep-merge)
├── data/
│   ├── base.py                  # DataProvider interface
│   ├── ccxt_provider.py         # Live OHLCV via CCXT (free public API)
│   └── csv_provider.py          # Offline CSV data for backtesting
├── indicators/
│   └── technical.py             # EMA, SMA, RSI, MACD, Bollinger, ATR, ADX
├── strategies/
│   ├── base.py                  # BaseStrategy + Signal dataclass (long/short/close_long/close_short/hold)
│   ├── trend_follow.py          # Dual EMA crossover + RSI
│   ├── mean_reversion.py        # Bollinger Bands + RSI
│   ├── breakout.py              # ATR-filtered range breakout
│   ├── volatility.py            # Bollinger Squeeze + ATR expansion (NEW)
│   ├── regime.py                # MarketRegimeDetector: trending/ranging/volatile/choppy (NEW)
│   └── multi_strategy.py        # Regime-aware voting engine
├── signal_scanner/              # NEW
│   └── scanner.py               # SignalScanner: async scan loop → log + Telegram + /api/signals
├── bot/                         # NEW
│   └── worker.py                # BotWorker: async per-symbol trading loop
├── notifications/               # NEW
│   └── telegram.py              # TelegramNotifier: signal/order/risk/error/summary alerts
├── risk/
│   └── engine.py                # RiskEngine: sizing, futures risk, liquidation buffer
├── execution/
│   ├── base.py                  # BrokerBase interface + Order dataclass
│   ├── paper_broker.py          # Simulated spot fills
│   ├── futures_broker.py        # FuturesBroker (CCXT) + PaperFuturesBroker (NEW)
│   └── live_broker.py           # Real CCXT market orders
├── backtesting/
│   └── engine.py                # Bar-by-bar backtester with leverage, funding, liquidation
├── monitoring/
│   ├── logger.py                # Structured JSON + console logging
│   ├── metrics.py               # MetricsTracker (PnL, drawdown, win rate)
│   └── analytics.py             # Sharpe, Sortino, Calmar, Profit Factor, per-symbol (NEW)
├── web/
│   ├── server.py                # FastAPI endpoints (dashboard + bot control + new API)
│   ├── run.py                   # Uvicorn launcher
│   └── static/index.html        # SPA dashboard (signals, positions, analytics pages)
├── logs/
│   ├── signal_log.jsonl         # Every scanned signal
│   └── risk_log.jsonl           # Every risk event
├── tests/
│   ├── test_indicators.py
│   ├── test_risk_engine.py
│   ├── test_strategies.py
│   ├── test_backtesting.py
│   ├── test_regime.py           # NEW
│   ├── test_volatility.py       # NEW
│   ├── test_futures_broker.py   # NEW
│   ├── test_signal_scanner.py   # NEW
│   ├── test_analytics.py        # NEW
│   └── test_notifications.py    # NEW
├── main.py                      # CLI entry: paper / live / backtest / scanner
└── requirements.txt
```

---

## Configuration Reference

### Global config (`config/config.yaml`)

| Section | Key options |
|---|---|
| `mode` | `paper` / `backtest` / `live` / `scanner` |
| `exchange` | `name` (ccxt id), `testnet`, `futures: true` |
| `data` | `symbols`, `timeframe`, `lookback_candles`, `provider` |
| `strategy` | `active`, per-strategy params, `regime_aware: true` |
| `futures` | `leverage`, `margin_type` (`isolated`/`cross`) |
| `scanner` | `enabled`, `interval_seconds`, `min_confidence` |
| `notifications.telegram` | `enabled`, `bot_token`, `chat_id`, `batch_summary_hours` |
| `risk` | `initial_capital`, `max_risk_per_trade_pct`, `max_daily_loss_pct`, `stop_loss_pct`, `take_profit_pct`, `kill_switch`, `max_open_positions` |
| `backtest` | `start_date`, `end_date`, `fee_rate`, `slippage_pct`, `funding_rate` |
| `monitoring` | `log_level`, `log_file`, `metrics_file`, `risk_log_file` |

### Per-symbol override (`config/symbols/BTC_USDT.yaml`)

```yaml
# config/symbols/BTC_USDT.yaml
strategy:
  active: trend_follow
  trend_follow:
    fast_ema: 9
    slow_ema: 21

futures:
  leverage: 5
  margin_type: isolated

risk:
  stop_loss_pct: 1.0
  take_profit_pct: 2.5
```

Use `load_symbol_config(symbol, global_cfg)` to get the merged per-symbol config.

---

## Signal Types

Strategies emit `Signal` objects with action in:

| Action | Meaning |
|---|---|
| `long` | Enter a long position |
| `short` | Enter a short position |
| `close_long` | Close an existing long |
| `close_short` | Close an existing short |
| `hold` | No action |
| `buy` / `sell` | Legacy aliases for `long` / `short` |

---

## Web Dashboard API

| Endpoint | Description |
|---|---|
| `GET /api/v1/status` | Bot status (running, mode, uptime) |
| `GET /api/v1/metrics` | Trade metrics (PnL, win rate, drawdown) |
| `GET /api/v1/signals` | Recent scanner signals |
| `GET /api/v1/signals/stream` | SSE stream of new signals (real-time) |
| `GET /api/v1/positions` | Open positions per symbol |
| `GET /api/v1/analytics` | Sharpe, Sortino, per-symbol stats |
| `GET /api/v1/risk/events` | Recent risk log entries |
| `POST /api/v1/bot/start` | Start bot in specified mode |
| `POST /api/v1/bot/stop` | Stop bot |
| `POST /api/v1/bot/killswitch` | Toggle kill switch |
| `POST /api/v1/backtest` | Run a backtest |

---

## Risk Management

The `RiskEngine` applies these guardrails before every trade:

1. **Kill switch** – halts all trading immediately
2. **Daily loss limit** – stops trading when cumulative daily PnL exceeds `max_daily_loss_pct`
3. **Cooldown** – skips N trades after a loss
4. **Max open positions** – caps simultaneous exposure
5. **Portfolio exposure cap** – limits total capital at risk across all symbols
6. **Liquidation buffer** – closes position early when within N% of liquidation price
7. **Position sizing** – sizes each trade so max loss = `max_risk_per_trade_pct` of capital
8. **Short position risk** – applies tighter stops for short positions (unlimited loss potential)

---

## Extending the Bot

### Add a new strategy

```python
# strategies/my_strategy.py
from strategies.base import BaseStrategy, Signal
import pandas as pd

class MyStrategy(BaseStrategy):
    name = "my_strategy"

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        return Signal(action="long", confidence=0.8, strategy_name=self.name)
```

Register it in `strategies/multi_strategy.py` under `_REGISTRY`.

### Add a per-symbol override

Create `config/symbols/SOL_USDT.yaml` with only the values you want to override (deep-merged with globals).

---

## Security Notes

- **Never commit API keys** to source control.
- Use `MIRA_API_KEY`, `MIRA_API_SECRET`, `MIRA_TELEGRAM_TOKEN`, `MIRA_TELEGRAM_CHAT_ID` env vars.
- Start with `testnet: true` and `mode: paper` before risking real funds.
- Set `risk.kill_switch: true` in config or use the web dashboard to halt the bot immediately.

---

## Disclaimer

This software is provided for educational purposes. Crypto trading involves
significant financial risk. Past backtest performance does not guarantee future
results. Use at your own risk.


---

## Features

| Layer | What it does |
|---|---|
| **Data** | Fetches free public OHLCV candles via [CCXT](https://github.com/ccxt/ccxt) (100+ exchanges) or local CSV |
| **Indicators** | EMA, SMA, RSI, MACD, Bollinger Bands, ATR – pure pandas/numpy |
| **Strategies** | Trend-following, Mean-reversion, Breakout, Multi-strategy voting |
| **Risk Engine** | Max daily loss, per-trade risk sizing, max open positions, cooldown, kill switch |
| **Backtesting** | Bar-by-bar engine with realistic fee + slippage hooks |
| **Paper Trading** | Full live loop with zero real orders |
| **Live Trading** | CCXT-powered market orders (testnet or real) |
| **Monitoring** | Structured JSON logging + metrics tracker (PnL, drawdown, win rate) |
| **Config-driven** | All parameters in `config/config.yaml` – no hardcoded values |

---

## Quick Start

### 1. Install dependencies

```bash
cd MIRA_trader_C
pip install -r requirements.txt
```

### 2. Configure

Edit `config/config.yaml`.  
**Never put API keys in the file.** Use environment variables instead:

```bash
export MIRA_API_KEY="your_key"
export MIRA_API_SECRET="your_secret"
```

### 3. Run a backtest (free, no account needed)

```bash
python main.py --mode backtest
```

### 4. Run paper trading (live data, no real orders)

```bash
python main.py --mode paper
```

### 5. Run unit tests

```bash
cd MIRA_trader_C
pytest tests/ -v
```

---

## Project Structure

```
MIRA_trader_C/
├── config/
│   ├── config.yaml          # All runtime parameters
│   └── __init__.py          # Config loader (injects env-var secrets)
├── data/
│   ├── base.py              # DataProvider interface
│   ├── ccxt_provider.py     # Live data via CCXT (free public API)
│   └── csv_provider.py      # Offline/backtesting data from CSV
├── indicators/
│   └── technical.py         # EMA, SMA, RSI, MACD, Bollinger, ATR
├── strategies/
│   ├── base.py              # BaseStrategy + Signal dataclass
│   ├── trend_follow.py      # Dual EMA crossover + RSI
│   ├── mean_reversion.py    # Bollinger Bands + RSI
│   ├── breakout.py          # ATR-filtered range breakout
│   └── multi_strategy.py    # Voting engine combining strategies
├── risk/
│   └── engine.py            # RiskEngine: sizing, guardrails, kill switch
├── backtesting/
│   └── engine.py            # Bar-by-bar backtester + BacktestResults
├── execution/
│   ├── base.py              # BrokerBase interface
│   ├── paper_broker.py      # Simulated fills (paper trading)
│   └── live_broker.py       # Real CCXT market orders
├── monitoring/
│   ├── logger.py            # Structured JSON + console logging
│   └── metrics.py           # Trade metrics tracker (PnL, drawdown, etc.)
├── utils/
│   └── helpers.py           # Utility functions
├── tests/
│   ├── test_indicators.py
│   ├── test_risk_engine.py
│   ├── test_strategies.py
│   └── test_backtesting.py
├── main.py                  # CLI entry point
└── requirements.txt
```

---

## Configuration Reference

See [`config/config.yaml`](config/config.yaml) for all options with inline comments.

Key sections:

| Section | Key options |
|---|---|
| `mode` | `paper` / `backtest` / `live` |
| `exchange` | `name` (ccxt id), `testnet` |
| `data` | `symbols`, `timeframe`, `lookback_candles`, `provider` |
| `strategy` | `active` strategy + per-strategy params |
| `risk` | `initial_capital`, `max_risk_per_trade_pct`, `max_daily_loss_pct`, `stop_loss_pct`, `take_profit_pct`, `kill_switch` |
| `backtest` | `start_date`, `end_date`, `fee_rate`, `slippage_pct` |
| `monitoring` | `log_level`, `log_file`, `metrics_file` |

---

## Risk Management

The `RiskEngine` applies these guardrails before every trade:

1. **Kill switch** – halts all trading immediately when `risk.kill_switch: true`
2. **Daily loss limit** – stops trading when cumulative daily PnL reaches `max_daily_loss_pct`
3. **Cooldown** – skips N trades after a losing trade (`cooldown_after_loss_trades`)
4. **Max open positions** – caps simultaneous exposure (`max_open_positions`)
5. **Symbol exposure cap** – limits capital per symbol (`max_exposure_per_symbol_pct`)
6. **Position sizing** – sizes each trade so max loss = `max_risk_per_trade_pct` of capital
7. **Stop-loss / Take-profit** – automatically computed from config or strategy override

---

## Extending the Bot

### Add a new strategy

```python
# strategies/my_strategy.py
from strategies.base import BaseStrategy, Signal
import pandas as pd

class MyStrategy(BaseStrategy):
    name = "my_strategy"

    def generate_signal(self, df: pd.DataFrame) -> Signal:
        # Your logic here
        return Signal(action="buy", confidence=0.8)
```

Then register it in `strategies/multi_strategy.py` under `_REGISTRY`.

### Swap the data provider

Change `data.provider` in `config.yaml` to `csv` and place CSV files in `data/csv/`:

```
data/csv/BTC_USDT_1h.csv   # columns: timestamp,open,high,low,close,volume
```

---

## Security Notes

- **Never commit API keys** to source control.
- Use environment variables `MIRA_API_KEY` / `MIRA_API_SECRET`.
- Start with `testnet: true` and `mode: paper` before risking real funds.
- Use the kill switch (`risk.kill_switch: true` in config) to halt the bot immediately.

---

## Disclaimer

This software is provided for educational purposes. Crypto trading involves
significant financial risk. Past backtest performance does not guarantee future
results. Use at your own risk.
