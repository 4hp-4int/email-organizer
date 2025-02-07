from cryptography.fernet import Fernet
from dataclasses import dataclass, field
from datetime import date, datetime
from loguru import logger
import json
import pytz

from agent_protocol import Agent
from azure.identity import ClientSecretCredential
from msgraph import GraphServiceClient
from msgraph.generated.models.o_data_errors.o_data_error import ODataError
from msgraph.generated.models.message import Message
from msgraph.generated.users.users_request_builder import UsersRequestBuilder
from msgraph.generated.users.item.messages.item.move.move_post_request_body import (
    MovePostRequestBody,
)


from msgraph.generated.users.item.messages.messages_request_builder import (
    MessagesRequestBuilder,
)
from kiota_abstractions.base_request_configuration import RequestConfiguration
from sentence_transformers import SentenceTransformer
import spacy

from src.message import Message
from src.util import simplify_html
from xconfig import Config

# Configure your OpenAI API key (or set up your Ollama client accordingly)
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
    embedding_text: str = field(default_factory=str)


class EmailOrganizerAgent(Agent):
    """
    Agent to organize emails.
    """

    def __init__(self, name: str):
        super().__init__()
        self.cipher_suite = Fernet(config.ENCRYPTION_KEY)
        self.nlp = spacy.load(config.SPACY_LIBRARY)
        self.model = SentenceTransformer(config.MODEL)

        tenant_id = config.AZURE_TENANT_ID
        client_id = config.AZURE_CLIENT_ID
        client_secret = config.AZURE_CLIENT_SECRET
        graph_scopes = config.AZURE_GRAPH_SCOPES

        client_secret_credential = ClientSecretCredential(
            tenant_id, client_id, client_secret
        )
        self.user_client = GraphServiceClient(
            credentials=client_secret_credential,
            scopes=graph_scopes,
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
        """
        Retrieve user information from Microsoft Graph API.
        """
        logger.info(f"retrieving user inbox: {user_id}")
        query_params = UsersRequestBuilder.UsersRequestBuilderGetQueryParameters(
            select=["displayName", "mail", "userPrincipalName"]
        )
        request_configuration = RequestConfiguration(query_parameters=query_params)
        return await self.user_client.users.by_user_id(user_id).get(
            request_configuration
        )

    def preprocess_function(self, email):
        """
        Preprocess the email content for embedding.
        """
        # Extract subject and body
        if isinstance(email, dict):
            subject = email.get("subject")
            body = email.get("body")
        else:
            subject = email.subject
            body = email.body.content

        # Combine subject and body for embedding
        combined_text = config.EMBEDDING_FORMAT_STRING.format(
            subject=subject,
            body=simplify_html(body)
            .replace("{", "{{")
            .replace("}", "}}"),  ## this is for emails that have curly braces in them
        )

        # Filter out stop words and punctuation
        doc = self.nlp(combined_text)
        filtered_words = [
            token.text for token in doc if not token.is_stop and not token.is_punct
        ]
        # Join the filtered words back into a string
        cleaned_text = " ".join(filtered_words)
        return cleaned_text

    async def get_inbox(self, user_id: str):
        """
        Retrieve emails from the user's inbox.
        """
        logger.info("Grabbing emails for user.")

        query_params = MessagesRequestBuilder.MessagesRequestBuilderGetQueryParameters(
            select="from,isRead,receivedDateTime,subject,body",
            top=100,
            orderby="receivedDateTime DESC",
        )
        request_configuration = RequestConfiguration(query_parameters=query_params)

        # Grab the first page of emails
        messages = await self.user_client.users.by_user_id(user_id).messages.get(
            request_configuration=request_configuration
        )

        while True:

            for message in messages.value:

                # Preprocess the email content
                try:
                    email_content = self.preprocess_function(message)
                except KeyError:
                    logger.exception("Failed to process email")
                    continue

                # Generate embeddings
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
                    embedding_text=self._encrypt_message(email_content),
                )
                yield email

            # Check if there are more pages
            if messages.odata_next_link:
                messages = (
                    await self.user_client.users.by_user_id(user_id)
                    .messages.with_url(messages.odata_next_link)
                    .get()
                )
            else:
                logger.info("No more emails found")
                break

    async def get_todays_unread_emails(self, user_id):
        """
        Fetch unread emails received today for the given user.
        """
        todays_emails = list()
        logger.info("Fetching today's unread emails.")

        # Get today's date in ISO 8601 format (UTC)
        today = (
            datetime.now(pytz.UTC)
            .replace(hour=0, minute=0, second=0, microsecond=0)
            .isoformat()
        )

        # Query parameters to filter unread messages from today

        query_params = MessagesRequestBuilder.MessagesRequestBuilderGetQueryParameters(
            filter=f"isRead eq false and receivedDateTime ge {today}",
            top=100,
            orderby="receivedDateTime desc",
            select="from,isRead,receivedDateTime,subject,body,id,parentFolderId",
        )
        request_configuration = RequestConfiguration(query_parameters=query_params)

        while True:
            try:
                # Fetch unread messages
                messages = await self.user_client.users.by_user_id(
                    user_id
                ).messages.get(request_configuration=request_configuration)

                todays_emails.extend(messages.value)

                # Check if there are more pages
                if messages.odata_next_link:
                    messages = (
                        await self.user_client.users.by_user_id(user_id)
                        .messages.with_url(messages.odata_next_link)
                        .get()
                    )
                else:
                    logger.info("No more emails found")
                    break

            except ODataError as e:
                logger.error(f"Error fetching emails: {e}")
                raise e

        return todays_emails

    async def prepare_inbox_folders(self, user_id: str, topics: list):
        """
        We need to prepare the inbox categories for the user in Outlook.
        """
        from msgraph.generated.models.mail_folder import MailFolder

        mail_folder_models = [
            MailFolder(
                display_name=label,
                is_hidden=False,
                parent_folder_id="inbox",
            )
            for label in topics
        ]
        for folder in mail_folder_models:

            # Create the folder or skip if it already exists
            try:
                await self.user_client.users.by_user_id(user_id).mail_folders.post(
                    folder
                )
            except ODataError as e:
                logger.exception(
                    f"Failed to create folder {folder.display_name} - error: {e}"
                )
                continue

    async def get_folder_destination_ids(self, user_id: str):
        """
        Retrieve the destination folder IDs for the user's inbox.
        """
        mail_folders = list()
        folders = await self.user_client.users.by_user_id(user_id).mail_folders.get()

        while True:
            # Check if there are more pages
            if folders.odata_next_link:
                logger.info("Fetching more folders.")
                folders = (
                    await self.user_client.users.by_user_id(user_id)
                    .mail_folders.with_url(folders.odata_next_link)
                    .get()
                )
                mail_folders.extend(folders.value)
            else:
                break

        return {
            folder.display_name: folder.id
            for folder in mail_folders
            if folder.display_name in config.TOPIC_LABELS.values()
        }

    async def categorize_emails(
        self,
        user_id: str,
        messages_labels: tuple[Message, str],
        folder_destination_ids: dict[str, str],
    ):
        """
        Categorize emails based on the provided labels.
        """

        for message, label in messages_labels:
            # Skip if the label is "unknown"
            if label == "error":
                continue

            # Skip if the email is already in the correct folder
            if message.parent_folder_id == folder_destination_ids.get(label):
                logger.debug("Email already in the correct folder")
                continue

            # Skip if the label is not in the config
            if not folder_destination_ids.get(label):
                continue

            logger.info(f"Categorizing email {message.id} as {label}")
            move_request_body = MovePostRequestBody()
            move_request_body.destination_id = folder_destination_ids.get(label)
            # Categorize the email via the Graph API
            try:
                await self.user_client.users.by_user_id(user_id).messages.by_message_id(
                    message_id=message.id
                ).move.post(move_request_body)
            except ODataError as e:
                logger.exception(
                    f"Failed to categorize email {message.subject} - error: {e}"
                )
                break

        return True

    def on_message(self, message: Message) -> Message:
        """
        Takes in a batch of email messages, and assigns a category to each email.
        """

        return True
