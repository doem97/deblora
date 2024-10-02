import sys
import signal
import requests
import pdb


def send_telegram_notification(message):
    bot_token = "7254878605:AAGHLOnoaj8W3oGUl-BbWlywnuSXSOWKOb0"
    chat_id = "6148817210"
    url = f"https://api.telegram.org/bot{bot_token}/sendMessage"
    payload = {"chat_id": chat_id, "text": message}
    try:
        requests.post(url, json=payload)
    except Exception as e:
        print(f"Failed to send Telegram notification: {e}")


def excepthook(type, value, traceback):
    error_message = f"An error occurred: {value}, and entering ipdb."
    print(error_message)
    send_telegram_notification(error_message)
    pdb.post_mortem(traceback)  # WARN: remove pdb traceback


def signal_handler(signum, frame):
    print("Interrupt received, exiting...")
    sys.exit(0)


# Example usage
def setup_error_handling():
    sys.excepthook = excepthook
    signal.signal(signal.SIGINT, signal_handler)
