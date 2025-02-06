import code
from dataclasses import asdict
import logging

from bertopic import BERTopic
from bson import MinKey
import numpy as np
from pymongo import MongoClient
import typer
from sentence_transformers import SentenceTransformer
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)
from sklearn.model_selection import train_test_split

from src.agent import EmailOrganizerAgent
from src.manager import AgentManager
from src.util import coro, simplify_html
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
    topic_model = BERTopic(verbose=True, min_topic_size=10, nr_topics=50)

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

    topic_model.save(f"emailClassifier-50")

    code.interact()


if __name__ == "__main__":
    app()
