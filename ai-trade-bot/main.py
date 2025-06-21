from core.binance_api import get_ohlcv, get_price
from core.strategy_vote import vote_signal
from core.trade import buy_market, simulate_sell
from core.compound import init_fund, update_fund, save_trade_log
from core.telegram_alert import send_telegram
from core.pauser import is_paused
import time

try:
    if is_paused():
        print("⛔ Bot ถูก Pause อยู่")
        send_telegram("⛔ Bot ถูก Pause (status.txt = pause)")
        exit()

    symbol = "BTCUSDT"
    df = get_ohlcv(symbol)
    should_buy = vote_signal(df)

    if should_buy:
        fund = init_fund()
        price = get_price(symbol)
        qty = round(fund / price, 6)

        send_telegram(f"📥 เข้าไม้ {symbol}\nทุน: ${fund:.2f}\nจำนวน: {qty}\nราคา: {price:.2f}")

        time.sleep(3)
        pnl, status, sell_price = simulate_sell(symbol, price)
        new_fund = update_fund(fund, pnl)
        save_trade_log(symbol, price, sell_price, pnl, status, qty, new_fund)

        send_telegram(f"{status} {symbol}\nขายที่: {sell_price:.2f}\nPnL: {pnl:.2f}%\nทุนใหม่: ${new_fund:.2f}")

    else:
        print("❌ Vote ไม่ผ่าน ยังไม่เข้าไม้")
        send_telegram("❌ Vote ไม่ผ่าน ยังไม่เข้าไม้")

except Exception as e:
    err_msg = f"🔥 ERROR: {str(e)}"
    print(err_msg)
    send_telegram(err_msg)
