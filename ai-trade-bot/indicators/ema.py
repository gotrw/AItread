def check_ema(df, short=9, long=21):
    df["ema_short"] = df["close"].ewm(span=short).mean()
    df["ema_long"] = df["close"].ewm(span=long).mean()
    return df["ema_short"].iloc[-1] > df["ema_long"].iloc[-1]  # EMA cross up = Buy
