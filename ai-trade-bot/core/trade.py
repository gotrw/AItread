from core.binance_api import client
import time

def buy_market(symbol: str, quantity: float):
    try:
        order = client.order_market_buy(symbol=symbol, quantity=quantity)
        return order
    except Exception as e:
        return f"Buy Error: {e}"

def sell_market(symbol: str, quantity: float):
    try:
        order = client.order_market_sell(symbol=symbol, quantity=quantity)
        return order
    except Exception as e:
        return f"Sell Error: {e}"

def simulate_sell(symbol, buy_price, tp_percent=5, sl_percent=3):
    # ใช้สำหรับ simulation เท่านั้น (ไม่เทรดจริง)
    from core.binance_api import get_price
    sell_price = get_price(symbol)
    pnl = ((sell_price - buy_price) / buy_price) * 100
    if pnl >= tp_percent:
        status = "✅ TP"
    elif pnl <= -sl_percent:
        status = "❌ SL"
    else:
        status = "⏳ HOLD"
    return pnl, status, sell_price
