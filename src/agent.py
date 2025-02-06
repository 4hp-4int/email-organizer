from cryptography.fernet import Fernet
from dataclasses import dataclass, field
from datetime import date
from logging import getLogger
import json
import os

from agent_protocol import Agent
from azure.identity import ClientSecretCredential
from msgraph import GraphServiceClient
from msgraph.generated.users.users_request_builder import UsersRequestBuilder
from kiota_abstractions.base_request_configuration import RequestConfiguration
from sentence_transformers import SentenceTransformer
from openai import OpenAI
import spacy

from src.message import Message
from src.util import simplify_html
from xconfig import Config

logger = getLogger("Email-Organizer")

# Configure your OpenAI API key (or set up your Ollama client accordingly)
client = OpenAI(api_key=Config.OPENAI_API_KEY)
config = Config()


@dataclass
class EmailMessage:
    subject: str
    body: str
    sender: str
    importance: str
    message_id: str
    conversation_id: str
    received_date: date
    embeddings: list = field(default_factory=list)


class EmailOrganizerAgent(Agent):
    def __init__(self, name: str):
        super().__init__()
        self.cipher_suite = Fernet(config.ENCRYPTION_KEY)
        self.nlp = spacy.load("en_core_web_sm")
        self.model = SentenceTransformer("all-MiniLm-L6-V2")

        tenant_id = os.environ.get("AZURE_TENANT_ID")
        client_id = os.environ.get("AZURE_CLIENT_ID")
        client_secret = os.environ.get("AZURE_CLIENT_SECRET")
        graph_scopes = os.environ.get("AZURE_GRAPH_SCOPES").split(" ")

        client_secret_credential = ClientSecretCredential(
            tenant_id, client_id, client_secret
        )
        self.user_client = GraphServiceClient(
            credentials=client_secret_credential, scopes=graph_scopes
        )
        self.name = name

    def _encrypt_message(self, message):
        message_json = json.dumps(message).encode("utf-8")
        encrypted_message = self.cipher_suite.encrypt(message_json)
        return encrypted_message

    def _decrypt_message(self, encrypted_message):
        decrypted_message = self.cipher_suite.decrypt(encrypted_message)
        return json.loads(decrypted_message.decode("utf-8"))

    async def get_user(self, user_id: str):
        logger.info(f"retrieving user inbox: {user_id}")
        query_params = UsersRequestBuilder.UsersRequestBuilderGetQueryParameters(
            select=["displayName", "mail", "userPrincipalName"]
        )
        request_configuration = RequestConfiguration(query_parameters=query_params)
        user = await self.user_client.users.by_user_id(user_id).get(
            request_configuration=request_configuration
        )
        return user

    async def get_inbox(self, user_id: str):
        def preprocess_function(email):
            subject = email.subject
            body = simplify_html(email.body.content)
            combined_text = f"Subject: {subject}\n\nBody: {body}"
            doc = self.nlp(combined_text)
            filtered_words = [
                token.text for token in doc if not token.is_stop and not token.is_punct
            ]
            # Join the filtered words back into a string
            cleaned_text = " ".join(filtered_words)
            return cleaned_text

        logger.info("Grabbing emails for user.")
        query_params = {
            "$select": "from,isRead,receivedDateTime,subject,body",
            "$top": 100,
            "$orderby": "receivedDateTime DESC",
        }
        request_configuration = RequestConfiguration(query_parameters=query_params)

        while True:
            messages = await self.user_client.users.by_user_id(user_id).messages.get(
                request_configuration=request_configuration
            )
            for message in messages.value:
                email_content = preprocess_function(message)
                email_embeddings = self.model.encode(
                    email_content, show_progress_bar=False
                )
                email = EmailMessage(
                    subject=self._encrypt_message(message.subject),
                    body=self._encrypt_message(message.body.content),
                    sender=self._encrypt_message(message.sender.email_address.address),
                    importance=message.importance,
                    conversation_id=message.conversation_id,
                    received_date=message.received_date_time,
                    message_id=message.internet_message_id,
                    embeddings=email_embeddings.tolist(),
                )
                yield email

            if messages.odata_next_link:
                logger.info("Found another page!")
                query_params["$top"] += 100
                # Update the request configuration
                request_configuration = RequestConfiguration(
                    query_parameters=query_params
                )
            else:
                logger.info("No more emails found")
                break

    def on_message(self, message: Message) -> Message:
        """
        Takes in a batch of email messages, and assigns a category to each email.
        """

        return True
