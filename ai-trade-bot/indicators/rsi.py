from ta.momentum import RSIIndicator

def check_rsi(df, period=14):
    rsi = RSIIndicator(close=df["close"], window=period).rsi()
    last_rsi = rsi.iloc[-1]
    return last_rsi < 30  # Buy Signal ถ้า RSI ต่ำ
