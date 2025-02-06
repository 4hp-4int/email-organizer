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


@app.command("classify_emails")
def classify_emails():
    logging.info("Classifying emails using data in emails collections")

    # Grab all the emails
    topic_model = BERTopic(verbose=True, min_topic_size=10, nr_topics=50)

    # Exclude Amazon Emails ZOMG
    email_cursor = email_collection.find({"_id": {"$gte": MinKey()}})

    raw_doc_strings = list()
    embeddings = list()

    logging.info("Preparing training data from the database.")
    for idx, email in enumerate(email_cursor, start=1):

        if idx % 10 == 0:
            logging.info(f"Processed {idx} emails for training")

        embedding = email.get("embeddings")
        doc_string = email.get("embedding_text")

        if not embedding:
            logging.debug(f"{email['_id']} does not have embeddings")
            continue

        raw_doc_strings.append(doc_string)
        embeddings.append(embedding)

    np_embeddings = np.array(embeddings)

    logging.info("Training Bert on the Topcis")
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
    print(topic_model.get_topic_info())

    code.interact()


@app.command("train_on_emails")
def train_on_emails():

    # Preprocess data
    def preprocess_function(email):
        subject = email_agent._decrypt_message(email["subject"])
        body = email_agent._decrypt_message(email["body"])

        return tokenizer(subject + " " + body, truncation=True, padding=True)

    # Grab all the emails
    emails = email_collection.find({"_id": {"$gte": MinKey()}}).limit(100).to_list()

    # Initialize the Tokenizer
    tokenizer = DistilBertTokenizer.from_pretrained("distilbert-base-uncased")
    tokenized_datasets = list(map(preprocess_function, emails))

    train_datasets, test_datasets = train_test_split(
        tokenized_datasets, test_size=0.2, random_state=42
    )

    # Load the model
    model = DistilBertForSequenceClassification.from_pretrained(
        "distilbert-base-uncased", num_labels=20
    )

    # Define training arguments
    training_args = TrainingArguments(
        output_dir="./results",
        evaluation_strategy="epoch",
        learning_rate=2e-5,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=16,
        num_train_epochs=3,
        weight_decay=0.01,
    )

    # Create Trainer
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_datasets,
        eval_dataset=test_datasets,
    )

    # Fine-tune the model
    trainer.train()


if __name__ == "__main__":
    app()
