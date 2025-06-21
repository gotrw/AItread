import os

STATUS_FILE = "status.txt"

def is_paused():
    if not os.path.exists(STATUS_FILE):
        return False
    with open(STATUS_FILE, "r") as f:
        status = f.read().strip()
        return status.lower() == "pause"

def set_status(mode="pause"):
    with open(STATUS_FILE, "w") as f:
        f.write(mode.lower())
