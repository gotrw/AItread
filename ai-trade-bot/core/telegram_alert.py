import requests

TOKEN = "7856530511:AAEV6P1e5mdN4tnq-SjYxIUj6oEMCVbuLjY"
CHAT_ID = "7726892356"

def send_telegram(msg):
    url = f"https://api.telegram.org/bot{TOKEN}/sendMessage"
    payload = {
        "chat_id": CHAT_ID,
        "text": msg
    }
    try:
        requests.post(url, data=payload)
    except Exception as e:
        print(f"Telegram Error: {e}")
