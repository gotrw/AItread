# MIRA_trader_C

A production-oriented, **free/low-cost** crypto trading bot built in Python.

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
