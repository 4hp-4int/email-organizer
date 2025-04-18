import os

from dotenv import load_dotenv
from transformers import AutoTokenizer

from src.topic_labels import topic_labels

load_dotenv()


class Config:

    AZURE_TENANT_ID = os.environ["AZURE_TENANT_ID"]
    AZURE_CLIENT_ID = os.environ["AZURE_CLIENT_ID"]
    AZURE_CLIENT_SECRET = os.environ["AZURE_CLIENT_SECRET"]
    AZURE_GRAPH_SCOPES = os.environ["AZURE_GRAPH_SCOPES"].split(" ")

    EMAIL_DATA_DATABASE = "email_organizer"
    EMAIL_DATA_COLLECTION = "emails_v3"
    EMAIL_RUN_LOG_COLLECTION = "email_run_log"
    EMAIL_SUPERVISED_TRAINING_COLLECTION = "email_supervised_training"

    EMBEDDING_FORMAT_STRING = "Subject: {subject}\n\nBody: {body}"

    ENCRYPTION_KEY = os.environ["ENCRYPTION_KEY"]
    MODEL = "all-MiniLm-L6-V2"
    EMBEDDING_MODEL = MODEL
    LLAMA2 = "Llama2"

    HF_TOKEN = os.environ["HF_TOKEN"]

    TOPIC_LABELS = topic_labels

    MIN_PROBABILITY = 0.52
    LLAMA_MAX_NEW_TOKENS = 300

    PROMPT = """\
    <|im_start|>system
    You are an expert topic classifier that assigns precise and concise labels to topics based on provided documents and keywords. The ultimate goal is to produce labels which result in an easy to review and organized email inbox.
    Strictly follow these rules:
    1. Your response must **only contain the label** (2-5 words).  
    2. Do **not** include explanations, confirmations, or extra phrases like "Sure! Based on the information provided...".  
    3. If the topic is unclear, return a topic label which more generally encompasses the topic.  
    4. Maintain professional, contextually accurate labels.
    <|im_end|><|im_start|>user
    Here's an Example: I have a topic that contains the following emails:
    - Subject: Your January Financial Report - Body: View your detailed financial report for January. See your spending, savings, and investment performance.
    - Subject: Bill Reminder - Electricity - Body: Your electricity bill is due on February 15th. Pay online or at any authorized payment center.
    - Subject: Credit Card Statement Available - Body: Your latest credit card statement is now available online. Check your transactions and payment details.

    The topic is described by the following keywords: 'bills, statements, financial, report, payment, credit card, electricity, due, transactions, spending'.

    Generate a concise and specific label for this topic in **2-5 words**. The label must accurately represent the documents and keywords. The ultimate goal is to produce labels which result in an easy to review and organized email inbox.
    I have a topic that contains the following emails: 
    [DOCUMENTS]
    The topic is described by the following keywords: [KEYWORDS]

    Based on the information above, extract a short topic label in the following format:
    topic: <topic label>
    <|im_end|>
    <|im_start|>assistant
    """

    @property
    def connection_string(self) -> str:
        user = os.environ.get("DB_USER")
        password = os.environ.get("DB_PASS")
        raw_uri = os.environ.get("DB_URI")
        return raw_uri.format(user=user, password=password)
