from cryptography.fernet import Fernet
from dataclasses import dataclass, field
from datetime import date, datetime
from loguru import logger
import json
from typing import List
import pytz

from azure.identity import ClientSecretCredential
from msgraph import GraphServiceClient
from msgraph.generated.models.o_data_errors.o_data_error import ODataError, APIError
from msgraph.generated.models.message import Message
from msgraph.generated.users.users_request_builder import UsersRequestBuilder
from msgraph.generated.users.item.messages.item.move.move_post_request_body import (
    MovePostRequestBody,
)
from pymongo import InsertOne


from msgraph.generated.users.item.mail_folders.mail_folders_request_builder import (
    MailFoldersRequestBuilder,
)
from msgraph.generated.users.item.messages.messages_request_builder import (
    MessagesRequestBuilder,
)
from kiota_abstractions.base_request_configuration import RequestConfiguration
from kiota_authentication_azure.azure_identity_authentication_provider import (
    AzureIdentityAuthenticationProvider,
)

from src.message import Message
from src.request_adapter import LoggingRequestAdapter
from src.util import simplify_html
from xconfig import Config

# Configure your OpenAI API key (or set up your Ollama client accordingly)
config = Config()


@dataclass
class EmailMessage:
    message_id: str
    received_date: date
    embedding_text: str = field(default_factory=str)


class EmailOrganizerAgent:
    """
    Agent to organize emails.
    """

    def __init__(self, name: str):
        super().__init__()
        self.cipher_suite = Fernet(config.ENCRYPTION_KEY)

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
            request_adapter=LoggingRequestAdapter(
                auth_provider=AzureIdentityAuthenticationProvider(
                    client_secret_credential, scopes=graph_scopes
                )
            ),
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

        return combined_text

    def create_email_message(self, message: Message) -> EmailMessage:
        """
        Creates an EmailMessage object from a raw message retrieved
        via the Microsoft Graph API.

        Args:
            message: A Message object from the Graph API.

        Returns:
            An EmailMessage object with encrypted subject, body, sender,
            and associated embeddings.
        """
        # Preprocess the email content for embedding
        try:
            email_content = self.preprocess_function(message)
        except Exception as e:
            logger.exception(f"Failed to preprocess email: {e}")
            raise e

        try:
            # Create the EmailMessage object
            email_msg = EmailMessage(
                received_date=message.received_date_time,
                message_id=message.id,
                embedding_text=self._encrypt_message(email_content),
            )
        except AttributeError as e:
            logger.exception(f"Encountered error {e} - Skipping email for now")
            return

        return email_msg

    async def get_inbox(self, user_id: str):
        """
        Retrieve emails from the user's inbox.
        """
        logger.info("Grabbing emails for user.")

        query_params = MessagesRequestBuilder.MessagesRequestBuilderGetQueryParameters(
            select="from,isRead,receivedDateTime,subject,body",
            top=50,
            orderby="receivedDateTime DESC",
        )
        request_configuration = RequestConfiguration(query_parameters=query_params)

        # Grab the first page of emails
        messages = await self.user_client.users.by_user_id(user_id).messages.get(
            request_configuration=request_configuration
        )

        logger.info("Collecting inbox emails...")
        num_emails_collected = 0
        while True:

            for message in messages.value:

                logger.debug(f"Processing Emil: {message.id}")
                email = self.create_email_message(message)
                num_emails_collected += 1
                yield num_emails_collected, email
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
            # filter=f"isRead eq false and receivedDateTime ge {today}",
            filter=f"",
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

                if len(todays_emails) >= 100:
                    break

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
            )
            for label in topics
        ]
        for folder in mail_folder_models:

            # Create the folder or skip if it already exists
            try:
                await self.user_client.users.by_user_id(
                    user_id
                ).mail_folders.by_mail_folder_id(
                    "inbox"  # This needs to be grabbed from the api.
                ).child_folders.post(
                    folder
                )
            except (ODataError, APIError) as e:
                logger.exception(
                    f"Failed to create folder {folder.display_name} - error: {e}"
                )
                continue

        return True

    async def get_folder_destination_ids(self, user_id: str):
        query_params = (
            MailFoldersRequestBuilder.MailFoldersRequestBuilderGetQueryParameters(
                top=1000, expand=["childFolders"]
            )
        )
        request_configuration = RequestConfiguration(query_parameters=query_params)

        # This call fetches *Inbox*, then child_folders of Inbox, in one request
        inbox_child_folders_response = (
            await (
                self.user_client.users.by_user_id(user_id)
                .mail_folders
                # "Inbox" is the folder ID or well-known name
                .by_mail_folder_id("Inbox")
                .child_folders.get(request_configuration=request_configuration)
            )
        )

        mail_folders = {}
        for folder in inbox_child_folders_response.value:
            mail_folders[folder.display_name] = folder.id

        return mail_folders

    async def categorize_emails(
        self,
        user_id: str,
        messages_labels: tuple[Message, str, int],
        folder_destination_ids: dict[str, str],
    ) -> List[dict[str, str]]:
        """
        Categorize emails based on the provided labels.
        """
        operation_log = list()
        for message, label, prob in messages_labels:
            # Skip if the label is "unknown"
            if label == "Misc. / Outliers / Marketing":
                continue

            if prob < config.MIN_PROBABILITY:
                logger.debug(
                    f"Skipping email with low probability - {message.subject} - {label}"
                )
                continue

            # Skip if the email is already in the correct folder
            if message.parent_folder_id == folder_destination_ids.get(label):
                logger.info("Email already in the correct folder")
                continue

            # Skip if the label is not in the config
            if not folder_destination_ids.get(label):
                logger.info(f"{label}, {folder_destination_ids.keys()}")
                continue

            # Move the email to the appropriate folder
            logger.info(f"Categorizing email {message.id} as {label}")
            move_request_body = MovePostRequestBody()
            move_request_body.destination_id = folder_destination_ids.get(label)

            # Categorize the email via the Graph API
            try:
                await self.user_client.users.by_user_id(user_id).messages.by_message_id(
                    message_id=message.id
                ).move.post(move_request_body)

                # Log the categorization
                move_operation = InsertOne(
                    {
                        "user_id": user_id,
                        "message_id": message.id,
                        "subject": self._encrypt_message(message.subject),
                        "label": label,
                    }
                )
                operation_log.append(move_operation)

            except ODataError as e:
                logger.exception(
                    f"Failed to categorize email {message.subject} - error: {e}"
                )
                break
            except Exception as e:
                logger.exception(f"Failed to write operation to database - error: {e}")
                break

        return operation_log
