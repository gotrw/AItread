from ta.volatility import BollingerBands

def check_bollinger(df):
    bb = BollingerBands(close=df["close"])
    lower_band = bb.bollinger_lband().iloc[-1]
    close_price = df["close"].iloc[-1]
    return close_price < lower_band  # ราคาแตะ lower = มีโอกาสเด้ง
