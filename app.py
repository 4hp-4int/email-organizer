import code
from dataclasses import asdict
import logging

from bertopic import BERTopic
from bson import MinKey
import numpy as np
from pymongo import MongoClient
import typer

from src.agent import EmailOrganizerAgent
from src.manager import AgentManager
from src.util import coro
from xconfig import Config


logger = logging.getLogger("Email-Organizer")
logging.basicConfig(level=logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

app = typer.Typer(help="Agentic Web Scraper Cli")
config = Config()

database_client = MongoClient(config.connection_string)
email_collection = database_client.get_database(
    Config.EMAIL_DATA_DATABASE
).get_collection(Config.EMAIL_DATA_COLLECTION)


manager = AgentManager()
email_agent = EmailOrganizerAgent(name="Aloyisius")

manager.register_agent(email_agent)


# This method uses async because of the graph api.
@app.command("collect_and_store")
@coro
async def collect_and_store_email():

    async for message in email_agent.get_inbox("khalen@4hp-4int.com"):
        result = email_collection.insert_one(asdict(message))
        if not result:
            logger.exception("Failed to write email to the database")


@app.command("train_classifier_model")
def train_classifier_model():
    logging.info("Classifying emails using data in emails collections")

    # Grab all the emails
    topic_model = BERTopic(verbose=True, min_topic_size=10, nr_topics=13)

    email_cursor = email_collection.find({"_id": {"$gte": MinKey()}})

    raw_doc_strings = list()
    embeddings = list()

    logging.info("Preparing training data from the database.")
    for idx, email in enumerate(email_cursor, start=1):

        if idx % 10 == 0:
            logging.info(f"Processed {idx} emails for training")

        embedding = email.get("embeddings")
        doc_string = email_agent._decrypt_message(email.get("embedding_text"))

        if not embedding:
            logging.debug(f"{email['_id']} does not have embeddings")
            continue

        raw_doc_strings.append(doc_string)
        embeddings.append(embedding)

    np_embeddings = np.array(embeddings)

    logging.info("Training Bert on the Topics")
    topics, probs = topic_model.fit_transform(
        documents=raw_doc_strings, embeddings=np_embeddings
    )

    logging.info("Writing topics to HTML File")
    fig = topic_model.visualize_topics()
    fig.write_html("topics.html")

    # Save to JSON
    topic_model.get_topic_info().to_json(
        "topics_info.json", orient="records", lines=True
    )
    print(topic_model.get_topics())

    topic_model.save(f"emailClassifier-13")

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

    # Store the inbox destination ids.


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
        result = await email_agent.categorize_emails(
            user_id, emails_labels, folder_destination_ids
        )

        if result:
            logger.info("Successfully categorized today's emails")
        else:
            logger.error("Failed to categorize today's emails")
    else:
        for email_label, topic in zip(emails_labels, new_topics):
            email, label = email_label
            logger.info(
                f"Subject: {email.subject} - Label: {label} - topic_id: {topic}"
            )


if __name__ == "__main__":
    app()
