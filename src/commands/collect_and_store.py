import typer
from dataclasses import asdict
from loguru import logger

# If you need these from your code:
from src.agent import EmailOrganizerAgent
from xconfig import Config
from pymongo import MongoClient, InsertOne

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
    batch_size: int = typer.Option(
        100, help="The default batch size to insert into in the database"
    ),
):
    """
    Asynchronously collects emails for the given user and stores them in the database.
    """
    email_agent = EmailOrganizerAgent(name="Aloyisius")

    bulk_insert_operations = []
    async for idx, message in email_agent.get_inbox(user_id):
        if not message:
            continue

        bulk_insert_operations.append(InsertOne(asdict(message)))

        if idx % batch_size == 0:
            result = email_collection.bulk_write(bulk_insert_operations)
            logger.info(f"{result.inserted_count} records inserted..")

            # Clear the list, grab the next batch_size
            bulk_insert_operations.clear()

        if not result:
            logger.exception("Failed to write email to the database")
