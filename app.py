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

from dataclasses import asdict
import pickle

from datasets import Dataset
from bertopic import BERTopic
from bson import MinKey
from loguru import logger
import numpy as np
from pymongo import MongoClient
from voyageai import Client
import typer

from src.agent import EmailOrganizerAgent
from src.embedders import VoyageEmbedder
from src.llama_agent import LlamaAgent
from src.topic_model_factory import TopicModelFactory
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

supervised_training_collection = database_client.get_database(
    config.EMAIL_DATA_DATABASE
).get_collection(config.EMAIL_SUPERVISED_TRAINING_COLLECTION)


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
    email_agent = EmailOrganizerAgent(name="Aloyisius")

    async for message in email_agent.get_inbox(user_id):
        result = email_collection.insert_one(asdict(message))
        if not result:
            logger.exception("Failed to write email to the database")


@app.command("create_inbox_folders")
@coro
async def create_inbox_folders(
    user_id: str = typer.Option(
        "khalen@4hp-4int.com", help="User ID to categorize emails for."
    ),
    config_path: str = typer.Option(
        "model_config.json", help="Path to the JSON config."
    ),
):
    """
    Create folders in the user's inbox based on the topics from a loaded BERTopic model.
    """
    # 1. Build the topic model factory from your config or model directory
    factory = TopicModelFactory(config_path=config_path)

    # 2. Load the existing model
    vo_client = Client()
    topic_model = factory.load_existing_model(vo_client)

    label_dict = {
        row["Topic"]: row["CustomName"].strip()
        for _, row in topic_model.get_topic_info().iterrows()
        if row["Topic"] != -1
    }

    # 4. Convert the label list into a set (folder names must be unique)
    labels_set = set(label_dict.values())

    # 5. Use your agent to create folders for these labels
    email_agent = EmailOrganizerAgent(name="Aloyisius")
    result = await email_agent.prepare_inbox_folders(user_id, labels_set)

    if result:
        logger.info("Successfully created inbox folders")
    else:
        logger.warning("No folders were created (possibly already exist).")


@app.command("review_and_categorize")
@coro
async def review_categorize_todays_email(
    user_id: str = typer.Option(
        "khalen@4hp-4int.com", help="User ID to categorize emails for."
    ),
    config_path: str = typer.Option(
        "model_config.json", help="Path to the topic model."
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


    This function loads a pre-trained topic model and uses it to categorize today's unread
    emails for the specified user.If dry_run is set to True, it will only log the
    categorization results without making any changes. Otherwise, it will categorize
    the emails and move them to the appropriate folders.
    """
    email_agent = EmailOrganizerAgent(name="Ao")
    factory = TopicModelFactory(config_path=config_path)

    # 2. Load the existing model
    vo_client = Client()
    topic_model = factory.load_existing_model(vo_client)

    todays_emails = []
    folder_destination_ids = await email_agent.get_folder_destination_ids(user_id)
    emails = await email_agent.get_todays_unread_emails(user_id)
    logger.info(f"Found {len(emails)} emails to categorize")

    for email in emails:
        email_content = email_agent.preprocess_function(email)
        todays_emails.append(email_content)

    new_topics, _ = topic_model.transform(todays_emails)

    # Prepare the labels for the emails.
    new_labels = [
        topic_model.get_topic_info(topic).get("CustomName").values[0].strip()
        for topic in new_topics
    ]

    emails_labels = list(zip(emails, new_labels))

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
        logger.info(f"Subject: {email.subject} - Label: {label} - topic_id: {topic}")


@app.command("train_classifier_model_v2")
def train_classifier_model_v2(
    config_path: str = typer.Option(
        "model_config.json", help="Path to the JSON config."
    ),
    retrain: bool = typer.Option(default=False),
):
    """
    A new, more flexible training command that uses TopicModelFactory
    to build and train a BERTopic model from a JSON config.
    """
    # 1. Build the factory from config
    factory = TopicModelFactory(config_path)
    vo_client = Client()

    # If you want to start from scratch:
    if not retrain:
        topic_model = factory.create_topic_model(vo_client)
    else:
        topic_model = factory.load_existing_model(vo_client)

    # 3. Retrieve and prepare training data
    email_agent = EmailOrganizerAgent(name="Aloyisius")
    logger.info("Collecting emails from DB")

    training_limit = factory.cfg.get("training_limit", 5000)
    email_cursor = email_collection.find().limit(training_limit)

    raw_doc_strings = []
    for idx, email in enumerate(email_cursor, start=1):
        if idx % 10 == 0:
            logger.info(f"Processed {idx} emails for training")
        doc_string = email_agent._decrypt_message(email.get("embedding_text"))
        # optional slicing to avoid extremely large docs
        raw_doc_strings.append(doc_string[:8192])

    dataset = Dataset.from_dict({"text": raw_doc_strings})

    # 4. Train
    logger.info("Fitting BERTopic model with loaded config")
    topics, probs = topic_model.fit_transform(dataset["text"])

    # If you were using Llama2 for labeling, you can do something like:
    if factory.cfg.get("use_llama2"):
        llama_agent = LlamaAgent()
        llama2_labels = llama_agent.parse_labels(
            topic_model.get_topics(full=True)["Llama2"]
        )
        logger.info(llama2_labels)
        topic_model.set_topic_labels(llama2_labels)

    # 5. Save the trained model
    model_path = factory.cfg["model_name"]
    topic_model.save(model_path, serialization="safetensors", save_ctfidf=True)

    # If needed, also save pickles or other artifacts just like in your existing code
    with open(f"{model_path}/rep_docs.pickle", "wb") as handle:
        pickle.dump(
            topic_model.representative_docs_, handle, protocol=pickle.HIGHEST_PROTOCOL
        )
    with open(f"{model_path}/embeddings.pickle", "wb") as handle:
        pickle.dump(
            topic_model.topic_embeddings_, handle, protocol=pickle.HIGHEST_PROTOCOL
        )
    with open(f"{model_path}/full_docs.pickle", "wb") as handle:
        pickle.dump(dataset["text"], handle, protocol=pickle.HIGHEST_PROTOCOL)

    topic_model.get_topic_info().to_json(
        f"{model_path}/topics_info.json", orient="records", lines=True
    )

    logger.info("Training complete!")


if __name__ == "__main__":
    app()
