import os
from typing import Dict, Any


def load_config() -> Dict[str, Any]:
    return {
        "ENV": os.environ.get("FLASK_ENV", "development"),
        "CORS_ORIGINS": os.environ.get("CORS_ORIGINS", "*"),
        "TWILIO_ACCOUNT_SID": os.environ.get("TWILIO_ACCOUNT_SID", ""),
        "TWILIO_AUTH_TOKEN": os.environ.get("TWILIO_AUTH_TOKEN", ""),
        "TWILIO_WHATSAPP_FROM": os.environ.get("TWILIO_WHATSAPP_FROM", ""),
        "TWILIO_WHATSAPP_TO": os.environ.get("TWILIO_WHATSAPP_TO", ""),
        "DATABASE_URL": os.environ.get("DATABASE_URL", ""),
    }


