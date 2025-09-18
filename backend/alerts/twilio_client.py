import os
from typing import Tuple


def send_whatsapp_alert(to_phone: str, body: str) -> Tuple[bool, str]:
    # Placeholder: integrate Twilio Client when credentials are provided.
    # from twilio.rest import Client
    # account = os.environ.get("TWILIO_ACCOUNT_SID", "")
    # token = os.environ.get("TWILIO_AUTH_TOKEN", "")
    # from_phone = os.environ.get("TWILIO_WHATSAPP_FROM", "")
    # if not (account and token and from_phone and to_phone):
    #     return False, "Twilio credentials or numbers missing"

    # client = Client(account, token)
    # msg = client.messages.create(
    #     from_=f"whatsapp:{from_phone}", to=f"whatsapp:{to_phone}", body=body
    # )
    # return True, msg.sid
    return True, "mock-message-sid"


