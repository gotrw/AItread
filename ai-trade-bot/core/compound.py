import os

LOG_PATH = "logs/trade_log.csv"

def init_fund():
    if not os.path.exists(LOG_PATH):
        with open(LOG_PATH, "w") as f:
            f.write("symbol,buy_price,sell_price,pnl,status,qty,fund\n")
        return 10  # ทุนเริ่มต้น
    else:
        with open(LOG_PATH, "r") as f:
            lines = f.readlines()
            if len(lines) < 2:
                return 10
            last_line = lines[-1].strip().split(",")
            return float(last_line[-1])

def save_trade_log(symbol, buy_price, sell_price, pnl, status, qty, fund):
    with open(LOG_PATH, "a") as f:
        f.write(f"{symbol},{buy_price},{sell_price},{pnl:.2f},{status},{qty},{fund:.2f}\n")

def update_fund(current_fund, pnl_percent):
    profit = (pnl_percent / 100) * current_fund
    return current_fund + profit
