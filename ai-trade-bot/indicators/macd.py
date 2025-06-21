from ta.trend import MACD

def check_macd(df):
    macd = MACD(close=df["close"])
    return macd.macd_diff().iloc[-1] > 0  # Histogram บวก = Buy
