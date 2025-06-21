from indicators.rsi import check_rsi
from indicators.ema import check_ema
from indicators.macd import check_macd
from indicators.bollinger import check_bollinger

def vote_signal(df):
    votes = 0
    if check_rsi(df): votes += 1
    if check_ema(df): votes += 1
    if check_macd(df): votes += 1
    if check_bollinger(df): votes += 1

    return votes >= 3  # ผ่าน 3 จาก 4 = Buy
