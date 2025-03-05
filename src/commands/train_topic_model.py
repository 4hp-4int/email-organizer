import typer
import pickle
from datasets import Dataset
from loguru import logger
from voyageai import Client

from src.agent import EmailOrganizerAgent
from src.llama_agent import LlamaAgent
from src.topic_model_factory import TopicModelFactory
from xconfig import Config
from pymongo import MongoClient

config = Config()
database_client = MongoClient(config.connection_string)
email_collection = database_client.get_database(
    config.EMAIL_DATA_DATABASE
).get_collection(config.EMAIL_DATA_COLLECTION)


def train_topic_model(
    config_path: str = typer.Option("model_config.json", help="Path to config."),
    retrain: bool = typer.Option(
        False, help="If True, loads and retrains existing model."
    ),
):
    """
    Train or retrain a topic classifier model from config.
    """
    factory = TopicModelFactory(config_path)
    vo_client = Client()

    if not retrain:
        topic_model = factory.create_topic_model(vo_client)
    else:
        topic_model = factory.load_existing_model(vo_client)

    # Gather data
    email_agent = EmailOrganizerAgent(name="Aloyisius")
    training_limit = factory.cfg.get("training_limit", 5000)
    email_cursor = email_collection.find().limit(training_limit)

    raw_doc_strings = []
    for idx, email in enumerate(email_cursor, start=1):
        if idx % 10 == 0:
            logger.info(f"Processed {idx} emails for training")
        doc_string = email_agent._decrypt_message(email.get("embedding_text"))
        raw_doc_strings.append(doc_string)

    dataset = Dataset.from_dict({"text": raw_doc_strings})

    # Fit the model
    logger.info("Fitting BERTopic model with loaded config")
    topics, probs = topic_model.fit_transform(dataset["text"])

    # Optionally handle Llama2-based labels
    if factory.cfg.get("use_llama2"):
        llama_agent = LlamaAgent()
        llama2_labels = llama_agent.parse_labels(
            topic_model.get_topics(full=True)["Llama2"]
        )
        logger.info(llama2_labels)
        # You could do something like:
        # topic_model.set_topic_labels(llama2_labels)

    # Save the model
    model_path = factory.cfg["model_name"]
    topic_model.save(model_path, serialization="safetensors", save_ctfidf=True)
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
