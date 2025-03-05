import typer
from loguru import logger
from voyageai import Client

from src.agent import EmailOrganizerAgent
from src.topic_model_factory import TopicModelFactory
from xconfig import Config
from pymongo import MongoClient
from src.util import coro

config = Config()
database_client = MongoClient(config.connection_string)
email_run_log_collection = database_client.get_database(
    config.EMAIL_DATA_DATABASE
).get_collection(config.EMAIL_RUN_LOG_COLLECTION)


@coro
async def review_categorize_todays_email(
    user_id: str = typer.Option(
        "khalen@4hp-4int.com", help="User ID to categorize emails for."
    ),
    config_path: str = typer.Option(
        "model_config.json", help="Path to the topic model."
    ),
    dry_run: bool = typer.Option(
        True, help="Perform a dry run without categorizing emails."
    ),
):
    """
    Review and categorize today's email for the specified user.
    """
    email_agent = EmailOrganizerAgent(name="Ao")
    factory = TopicModelFactory(config_path=config_path)

    vo_client = Client()
    topic_model = factory.load_existing_model(vo_client)

    folder_destination_ids = await email_agent.get_folder_destination_ids(user_id)
    emails = await email_agent.get_todays_unread_emails(user_id)
    logger.info(f"Found {len(emails)} emails to categorize")

    # Preprocess
    todays_emails = [email_agent.preprocess_function(e) for e in emails]

    # Infer topics
    new_topics, probs = topic_model.transform(todays_emails)
    new_labels = [
        topic_model.get_topic_info(topic).get("CustomName").values[0].strip()
        for topic in new_topics
    ]
    emails_labels = list(zip(emails, new_labels, probs))

    # Categorize if not dry run
    if not dry_run:
        operation_log = await email_agent.categorize_emails(
            user_id, emails_labels, folder_destination_ids
        )
        if operation_log:
            logger.info("Successfully categorized today's emails.")
            result = email_run_log_collection.bulk_write(operation_log)
            if not result.acknowledged:
                logger.error("Failed to store run results to database")
            else:
                logger.info("Successfully stored run results to database")

    if dry_run:
        for email, label, prob in emails_labels:
            logger.info(
                f"Email: {email.subject} - "
                f"Predicted Label: {label} - "
                f"Probability: {prob}"
            )
