import typer
from dataclasses import asdict
from loguru import logger

# If you need these from your code:
from src.agent import EmailOrganizerAgent
from xconfig import Config
from pymongo import MongoClient

# Import your @coro if needed
from src.util import coro

config = Config()
database_client = MongoClient(config.connection_string)
email_collection = database_client.get_database(
    config.EMAIL_DATA_DATABASE
).get_collection(config.EMAIL_DATA_COLLECTION)


@coro
async def collect_and_store_email(
    user_id: str = typer.Option(
        "khalen@4hp-4int.com", help="User ID to categorize emails for."
    ),
):
    """
    Asynchronously collects emails for the given user and stores them in the database.
    """
    email_agent = EmailOrganizerAgent(name="Aloyisius")

    async for message in email_agent.get_inbox(user_id):
        result = email_collection.insert_one(asdict(message))
        if not result:
            logger.exception("Failed to write email to the database")
