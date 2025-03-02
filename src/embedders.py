from pathlib import Path
import pickle

from bertopic.backend import BaseEmbedder
from loguru import logger
import numpy as np


class VoyageEmbedder(BaseEmbedder):
    """A custom BERTopic backend for Voyage AI embeddings."""

    def __init__(self, client, model_path, model_name="voyage-3", truncation=None):
        """
        Args:
            client: An instance of the VoyageAI Client class.
            model_name (str): Which Voyage AI model to use for embeddings.
            truncation (bool): Whether to enable truncation in Voyage embedding calls.
        """
        super().__init__()
        self.client = client
        self.model_name = model_name
        self.truncation = truncation
        self.batch_size = 128

        self.model_path = Path(model_path)
        self.model_path.parent.mkdir(parents=True, exist_ok=True)

    def embed(self, documents, verbose=False):
        all_embeddings = []
        # If your model or the Voyage docs impose a maximum batch size,
        # you can embed the documents in chunks of `self.batch_size`
        for start_idx in range(0, len(documents), self.batch_size):
            print(f"Starting next embedding batch! {start_idx}")
            end_idx = start_idx + self.batch_size
            chunk = documents[start_idx:end_idx]

            # Call the Voyage client
            response = self.client.embed(
                texts=chunk,
                model=self.model_name,
            )

            # response.embeddings is typically a list of vectors
            all_embeddings.extend(response.embeddings)

        numpy_embeddings = np.array(all_embeddings)

        logger.debug(f"Serializing Model Embeddings: {self.model_path}")

        with open(f"embeddings.pickle", "wb") as handle:
            pickle.dump(numpy_embeddings, handle, protocol=pickle.HIGHEST_PROTOCOL)

        return numpy_embeddings
