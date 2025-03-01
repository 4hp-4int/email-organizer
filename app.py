"""
This module provides a command-line interface (CLI) for an email organization system.
It includes commands for collecting and storing emails, training a classifier model,
creating inbox folders, and reviewing and categorizing today's emails.

Classes:
    None

Functions:
    collect_and_store_email(user_id: str)

    train_classifier_model()

    create_inbox_folders(user_id: str)

    review_categorize_todays_email(user_id: str, model_path: str, dry_run: bool)
        Review and categorize today's email using a pre-trained topic model.

"""

import code
from dataclasses import asdict
from loguru import logger

from bertopic import BERTopic
from bson import MinKey
import numpy as np
from pymongo import MongoClient
import typer

from src.agent import EmailOrganizerAgent
from src.manager import AgentManager
from src.util import coro
from xconfig import Config


app = typer.Typer(help="Email Organizer CLI")
config = Config()

database_client = MongoClient(config.connection_string)
email_collection = database_client.get_database(
    Config.EMAIL_DATA_DATABASE
).get_collection(Config.EMAIL_DATA_COLLECTION)

email_run_log_collection = database_client.get_database(
    config.EMAIL_DATA_DATABASE
).get_collection(config.EMAIL_RUN_LOG_COLLECTION)

manager = AgentManager()
email_agent = EmailOrganizerAgent(name="Aloyisius")

manager.register_agent(email_agent)


# This method uses async because of the graph api.
@app.command("collect_and_store")
@coro
async def collect_and_store_email(
    user_id: str = typer.Option(
        "khalen@4hp-4int.com", help="User ID to categorize emails for."
    ),
):
    """
    Asynchronously collects emails for a specified user and stores them in the database.

    Args:
        user_id (str): User ID to categorize emails for. Defaults to "khalen@4hp-4int.com".

    Raises:
        Exception: If the email fails to be written to the database.
    """
    async for message in email_agent.get_inbox(user_id):
        result = email_collection.insert_one(asdict(message))
        if not result:
            logger.exception("Failed to write email to the database")


@app.command("train_classifier_model")
def train_classifier_model(
    model_path: str = typer.Option(
        "emailClassifier-35", help="Path to the topic model."
    )
):
    """
    Train a classifier model using BERTopic on email data from the database.

    This function performs the following steps:
    1. Retrieves all emails from the email collection in the database.
    2. Prepares training data by extracting embeddings and decrypted text from emails.
    3. Trains a BERTopic model on the extracted data.
    4. Visualizes the topics and saves the visualization to an HTML file.
    5. Saves the topic information to a JSON file.
    6. Prints the topics to the console.
    7. Saves the trained BERTopic model to a file.

    Logging is used to provide information about the progress of each step.
    """
    logger.info("Classifying emails using data in emails collections")

    # Grab all the emails
    topic_model = BERTopic(verbose=True, min_topic_size=10, nr_topics=13)

    email_cursor = email_collection.find({"_id": {"$gte": MinKey()}})

    raw_doc_strings = list()
    embeddings = list()

    logger.info("Preparing training data from the database.")
    for idx, email in enumerate(email_cursor, start=1):

        if idx % 10 == 0:
            logger.info(f"Processed {idx} emails for training")

        embedding = email.get("embeddings")
        doc_string = email_agent._decrypt_message(email.get("embedding_text"))

        if not embedding:
            logger.debug(f"{email['_id']} does not have embeddings")
            continue

        raw_doc_strings.append(doc_string)
        embeddings.append(embedding)

    np_embeddings = np.array(embeddings)

    logger.info("Training Bert on the Topics")
    topic_model.fit_transform(documents=raw_doc_strings, embeddings=np_embeddings)

    logger.info("Writing topics to HTML File")
    fig = topic_model.visualize_topics()
    fig.write_html("topics.html")

    # Save to JSON
    topic_model.get_topic_info().to_json(
        "topics_info.json", orient="records", lines=True
    )
    logger.debug(topic_model.get_topics())

    topic_model.save(f"{model_path}")

    code.interact()


@app.command("create_inbox_folders")
@coro
async def create_inbox_folders(
    user_id: str = typer.Option(
        "khalen@4hp-4int.com", help="User ID to categorize emails for."
    ),
):
    """
    Create folders in the user's inbox based on the topics.
    """
    result = await email_agent.prepare_inbox_folders(
        user_id, set(config.TOPIC_LABELS.values())
    )

    if result:
        logger.info("Successfully created inbox folders")


@app.command("review_and_categorize")
@coro
async def review_categorize_todays_email(
    user_id: str = typer.Option(
        "khalen@4hp-4int.com", help="User ID to categorize emails for."
    ),
    model_path: str = typer.Option(
        "emailClassifier-35", help="Path to the topic model."
    ),
    dry_run: bool = typer.Option(
        True,
        help="Perform a dry run without categorizing emails.",
    ),
):
    """
    Review and categorize today's email.
        Args:
        user_id (str): User ID to categorize emails for. Default is "khalen@4hp-4int.com".
        model_path (str): Path to the topic model. Default is "emailClassifier-35".
        dry_run (bool): Perform a dry run without categorizing emails. Default is True.


    This function loads a pre-trained topic model and uses it to categorize today's unread emails for the specified user.
    If dry_run is set to True, it will only log the categorization results without making any changes.
    Otherwise, it will categorize the emails and move them to the appropriate folders.
    """
    topic_model = BERTopic.load(model_path, embedding_model=email_agent.model)

    todays_emails = list()
    folder_destination_ids = await email_agent.get_folder_destination_ids(user_id)
    raw_emails = await email_agent.get_todays_unread_emails(user_id)
    logger.info(f"Found {len(raw_emails)} emails to categorize")

    for email in raw_emails:
        email_content = email_agent.preprocess_function(email)
        todays_emails.append(email_content)

    new_topics, _ = topic_model.transform(todays_emails)
    new_labels = [config.TOPIC_LABELS.get(topic, "Unknown") for topic in new_topics]

    emails_labels = list(zip(raw_emails, new_labels))

    if not dry_run:
        operation_log = await email_agent.categorize_emails(
            user_id, emails_labels, folder_destination_ids
        )

        if operation_log:
            logger.info(
                "Successfully categorized today's emails, storing run results to database"
            )
            result = email_run_log_collection.bulk_write(operation_log)

            if not result.acknowledged:
                logger.error("Failed to store run results to database")
            else:
                logger.info("Successfully stored run results to database")

        # Debug output
        for email_label, topic in zip(emails_labels, new_topics):
            email, label = email_label
            logger.info(
                f"Subject: {email.subject} - Label: {label} - topic_id: {topic}"
            )


if __name__ == "__main__":
    app()
