import os

from dotenv import load_dotenv

load_dotenv()


class Config:

    EMAIL_DATA_DATABASE = "email_organizer"
    EMAIL_DATA_COLLECTION = "emails_v2"

    ENCRYPTION_KEY = os.environ["ENCRYPTION_KEY"]

    OPENAI_API_KEY = os.environ["OPENAI_API_KEY"]

    @property
    def connection_string(self) -> str:
        user = os.environ.get("DB_USER")
        password = os.environ.get("DB_PASS")
        raw_uri = os.environ.get("DB_URI")
        return raw_uri.format(user=user, password=password)
