from dataclasses import asdict
import json
import logging
from typing import List, Dict

from bson import MinKey
from pymongo import MongoClient
import typer
from transformers import (
    DistilBertForSequenceClassification,
    DistilBertTokenizer,
    Trainer,
    TrainingArguments,
)
import requests
from sklearn.model_selection import train_test_split

from src.agent import CSSSelectorAgent, EmailOrganizerAgent, EmailMessage
from src.manager import AgentManager
from src.message import Message
from src.util import simplify_html, chunk_text, process_chunk
from xconfig import Config


logger = logging.getLogger("Email-Organizer")
logger.setLevel(logging.DEBUG)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

app = typer.Typer(help="Agentic Web Scraper Cli")
config = Config()

database_client = MongoClient(config.connection_string)
email_collection = database_client.get_database(
    Config.EMAIL_DATA_DATABASE
).get_collection(Config.EMAIL_DATA_COLLECTION)


manager = AgentManager()
css_selector_agent = CSSSelectorAgent(name="CSSSelectorAgent")
email_agent = EmailOrganizerAgent(name="Aloyisius")

manager.register_agent(css_selector_agent)
manager.register_agent(email_agent)


def aggregate_chunk_results(results: List[Dict]) -> Dict:
    """
    Combine the individual chunk results into a unified representation.
    Here we simply merge the tag counts, but you could design a more complex structure.
    """
    aggregated = {"dom_map": {}, "snippets": []}
    for result in results:
        # Aggregate the tag counts
        for tag, count in result["dom_map"].items():
            aggregated["dom_map"][tag] = aggregated["dom_map"].get(tag, 0) + count
        # Save the snippet for reference
        aggregated["snippets"].append(result["snippet"])
    return aggregated


@app.command()
def generate_selectors(
    url: str, prompt: str = "Extract information for time-series data"
):

    typer.echo(f"Fetching URL: {url}")

    try:
        response = requests.get(url)
        response.raise_for_status()
    except Exception as e:
        typer.echo(f"Error loading webpage: {url} - {e}", err=True)
        raise typer.Exit(code=1)

    html_content = simplify_html(response.text)
    chunks = chunk_text(html_content, 2000)
    chunk_results = [process_chunk(chunk) for chunk in chunks]

    aggregated_dom = aggregate_chunk_results(chunk_results)
    typer.echo("Aggregated DOM representation:")
    typer.echo(json.dumps(aggregated_dom, indent=2))

    message = Message(
        sender="CLI",
        recipient=css_selector_agent.name,
        payload={"html": html_content, "prompt": prompt},
    )

    # # Send the message and get the response from the agent
    # response_message = manager.send_message(css_selector_agent.name, message)

    # if response_message:
    #     selectors = response_message.payload.get("selectors", {})
    #     typer.echo("\nExtracted CSS selectors:")
    #     typer.echo(json.dumps(selectors, indent=2))
    # else:
    #     typer.echo("No response received from the agent.")


import asyncio
from functools import wraps


def coro(f):
    @wraps(f)
    def wrapper(*args, **kwargs):
        return asyncio.run(f(*args, **kwargs))

    return wrapper


# This method uses async because of the graph api.
@app.command("collect_and_store")
@coro
async def collect_and_store_email():

    async for message in email_agent.get_inbox("khalen@4hp-4int.com"):
        result = email_collection.insert_one(asdict(message))
        if not result:
            logger.exception("Failed to write email to the database")


@app.command("train_on_emails")
def train_on_emails():

    # Preprocess data
    def preprocess_function(email):
        subject = email_agent._decrypt_message(email["subject"])
        body = email_agent._decrypt_message(email["body"])

        return tokenizer(subject + " " + body, truncation=True, padding=True)

    # Grab all the emails
    emails = email_collection.find().limit(100).to_list()

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
