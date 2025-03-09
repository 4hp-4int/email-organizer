import typer
import pickle
from typing import Optional
from datasets import Dataset
from loguru import logger
from voyageai import Client

from src.agent import EmailOrganizerAgent
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
    docs_pickle_path: Optional[str] = typer.Option(
        None, help="Path to pickled documents list."
    ),
    embeddings_pickle_path: Optional[str] = typer.Option(
        None, help="Path to pickled embeddings list."
    ),
):
    """
    Train or retrain a topic classifier model from config.
    """
    raw_doc_strings = []
    doc_embeddings = []
    factory = TopicModelFactory(config_path)
    vo_client = Client()

    if not retrain:
        topic_model = factory.create_topic_model(vo_client)
    else:
        topic_model = factory.load_existing_model(vo_client)

    if docs_pickle_path:
        logger.info("Loading documents and embeddings from provided pickle paths...")
        with open(docs_pickle_path, "rb") as f_docs:
            raw_doc_strings = pickle.load(f_docs)

    if embeddings_pickle_path:
        with open(embeddings_pickle_path, "rb") as f_embs:
            import numpy as np

            doc_embeddings = np.array(pickle.load(f_embs))

    # Gather data
    email_agent = EmailOrganizerAgent(name="Aloyisius")
    training_limit = factory.cfg.get("training_limit", 5000)
    email_cursor = email_collection.find().limit(training_limit)

    # Grab docs from the database if no pickles provided
    if not docs_pickle_path:
        for idx, email in enumerate(email_cursor, start=1):
            if idx % 10 == 0:
                logger.info(f"Processed {idx} emails for training")
            doc_string = email_agent._decrypt_message(email.get("embedding_text"))
            raw_doc_strings.append(doc_string)

    dataset = Dataset.from_dict({"text": raw_doc_strings})

    # Fit the model
    logger.info("Fitting BERTopic model with loaded config")
    if retrain:
        topics, probs = topic_model.partial_fit(dataset["text"])

    else:
        # Use stored training data
        if embeddings_pickle_path:
            topics, probs = topic_model.fit_transform(
                documents=dataset["text"], embeddings=doc_embeddings
            )
        else:
            topics, probs = topic_model.fit_transform(dataset["text"])

    print(topic_model.get_topic_info())

    # Save the model with the name of the model
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
