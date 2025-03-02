import os

from dotenv import load_dotenv

from src.topic_labels import topic_labels

load_dotenv()


class Config:

    AZURE_TENANT_ID = os.environ["AZURE_TENANT_ID"]
    AZURE_CLIENT_ID = os.environ["AZURE_CLIENT_ID"]
    AZURE_CLIENT_SECRET = os.environ["AZURE_CLIENT_SECRET"]
    AZURE_GRAPH_SCOPES = os.environ["AZURE_GRAPH_SCOPES"].split(" ")

    EMAIL_DATA_DATABASE = "email_organizer"
    EMAIL_DATA_COLLECTION = "emails_v2"
    EMAIL_RUN_LOG_COLLECTION = "email_run_log"
    EMAIL_SUPERVISED_TRAINING_COLLECTION = "email_supervised_training"

    EMBEDDING_FORMAT_STRING = "Subject: {subject}\n\nBody: {body}"

    ENCRYPTION_KEY = os.environ["ENCRYPTION_KEY"]
    MODEL = "all-MiniLm-L6-V2"
    EMBEDDING_MODEL = MODEL
    LLAMA2 = "Llama2"

    HF_TOKEN = os.environ["HF_TOKEN"]

    SPACY_LIBRARY = "en_core_web_sm"

    TOPIC_LABELS = topic_labels
    LLAMA_MAX_NEW_TOKENS = 300

    @property
    def connection_string(self) -> str:
        user = os.environ.get("DB_USER")
        password = os.environ.get("DB_PASS")
        raw_uri = os.environ.get("DB_URI")
        return raw_uri.format(user=user, password=password)
