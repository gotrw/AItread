from binance.client import Client
import json

def load_config():
    with open("config/config.json") as f:
        return json.load(f)

config = load_config()
client = Client(api_key=config["api_key"], api_secret=config["api_secret"])
def get_balance(asset="USDT"):
    balance = client.get_asset_balance(asset=asset)
    return float(balance["free"]) if balance else 0.0

def get_price(symbol="BTCUSDT"):
    ticker = client.get_symbol_ticker(symbol=symbol)
    return float(ticker["price"])

import pandas as pd

def get_ohlcv(symbol="BTCUSDT", interval="15m", limit=100):
    klines = client.get_klines(symbol=symbol, interval=interval, limit=limit)
    df = pd.DataFrame(klines, columns=[
        "timestamp", "open", "high", "low", "close", "volume",
        "close_time", "qav", "num_trades", "taker_base_vol", "taker_quote_vol", "ignore"
    ])
    df["close"] = df["close"].astype(float)
    return df
