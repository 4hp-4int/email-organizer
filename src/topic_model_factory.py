# topic_model_factory.py
import json
from bertopic import BERTopic
from umap import UMAP
from hdbscan import HDBSCAN
from loguru import logger

from src.llama_agent import LlamaAgent
from src.embedders import VoyageEmbedder
from xconfig import Config


class TopicModelFactory:
    def __init__(self, config_path: str):
        """
        Loads the JSON config from disk and prepares everything needed
        to instantiate or update a BERTopic model.
        """
        with open(config_path, "r") as f:
            self.cfg = json.load(f)
        self.app_config = Config()

    def create_topic_model(self, voyage_client) -> BERTopic:
        """
        Create and return a BERTopic instance based on the JSON config.
        If 'use_llama2' is True, we set up the Llama2-based representation.
        Otherwise, we create a standard BERTopic with minimal parameters.
        """

        # 1. Create the embedding model from Voyage AI if desired.
        embedding_model = None
        if self.cfg.get("voyage_model_name"):
            embedding_model = VoyageEmbedder(
                voyage_client,
                model_path=self.cfg["model_name"],
                model_name=self.cfg["voyage_model_name"],
            )

        rep_model = None
        if self.cfg.get("use_llama2"):
            llama_agent = LlamaAgent()
            rep_model = {"Llama2": llama_agent.llama2}

        umap_model = UMAP(**self.cfg.get("umap_params", {}))
        hdbscan_model = HDBSCAN(**self.cfg.get("hdbscan_params", {}))

        # 3. Build the actual BERTopic
        model = BERTopic(
            verbose=True,
            nr_topics=self.cfg.get("nr_topics", None),
            min_topic_size=self.cfg.get("min_topic_size", 10),
            embedding_model=embedding_model,
            umap_model=umap_model,
            hdbscan_model=hdbscan_model,
            representation_model=rep_model,
            zeroshot_topic_list=self.cfg.get("zeroshot_topic_list", []),
        )

        return model

    def load_existing_model(self, voyage_client) -> BERTopic:
        """
        If you want iterative improvements, you can load a saved model
        and optionally re-fit it on more data.
        """
        model_path = self.cfg["model_name"]

        logger.info(f"Loading existing BERTopic model from: {model_path}")
        # If you want the same embedder or Llama2 representation, pass them in:
        embedding_model = VoyageEmbedder(voyage_client, self.cfg["voyage_model_name"])
        loaded_model = BERTopic.load(model_path, embedding_model=embedding_model)
        return loaded_model
