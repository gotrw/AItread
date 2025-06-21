import schedule
import time
import os
from datetime import datetime

def run_main():
    os.system("python main.py")
    with open("logs/run_log.txt", "a") as f:
        f.write(f"[{datetime.now()}] Bot executed.\n")

def start_scheduler(interval_min=15):
    schedule.every(interval_min).minutes.do(run_main)

    print(f"⏱ Bot จะรันทุก {interval_min} นาที")
    while True:
        schedule.run_pending()
        time.sleep(1)
