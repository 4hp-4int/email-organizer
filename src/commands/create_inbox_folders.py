import typer
from loguru import logger
from voyageai import Client

from src.agent import EmailOrganizerAgent
from src.topic_model_factory import TopicModelFactory
from src.util import coro
from xconfig import Config

config = Config()


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
    # Build the model from your config
    factory = TopicModelFactory(config_path=config_path)
    topic_model = factory.load_existing_model(Client())

    # Retrieve topic labels
    label_dict = {
        row["Topic"]: row["CustomName"].strip()
        for _, row in topic_model.get_topic_info().iterrows()
        if row["Topic"] != -1
    }

    labels_set = set(label_dict.values())

    # Use agent to create the folders
    email_agent = EmailOrganizerAgent(name="Aloyisius")
    result = await email_agent.prepare_inbox_folders(user_id, labels_set)

    if result:
        logger.info("Successfully created inbox folders")
    else:
        logger.warning("No folders were created (possibly already exist).")
