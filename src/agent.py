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
from openai import OpenAI

from src.prompts import AGENT_SCRAPER_PROMPT
from src.message import Message
from xconfig import Config

logger = getLogger("Email-Organizer")

# Configure your OpenAI API key (or set up your Ollama client accordingly)
client = OpenAI(api_key=Config.OPENAI_API_KEY)


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


class CSSSelectorAgent(Agent):

    def __init__(self, name: str):
        super().__init__()
        self.name = name

    """
    An agent that receives HTML content and an extraction prompt,
    uses an LLM to determine interesting HTML elements, and outputs
    CSS selectors for those elements.
    """

    def on_message(self, message: Message) -> Message:
        # Retrieve HTML and prompt from the incoming message payload.
        html_content = message.payload.get("html", "")
        extraction_prompt = message.payload.get("prompt", "")

        llm_prompt = AGENT_SCRAPER_PROMPT.format(
            html_content=html_content, extraction_prompt=extraction_prompt
        )

        # Use the LLM to generate a response.
        try:
            response = client.chat.completions.create(
                model="gpt-4",
                messages=[
                    {
                        "role": "system",
                        "content": "You are an expert in web scraping and HTML analysis.",
                    },
                    {"role": "user", "content": llm_prompt},
                ],
                temperature=0.2,  # Lower temperature for more deterministic output
                max_tokens=300,
            )
            # Extract the text content from the LLM response.
            text_response = response.choices[0].message.content.strip()
        except Exception as e:
            error_response = {"error": f"Error calling LLM API: {str(e)}"}
            return Message(sender=self.name, payload={"selectors": error_response})

        # Try to parse the LLM response as JSON.
        try:
            selectors = json.loads(text_response)
        except json.JSONDecodeError as e:
            # In case parsing fails, return the raw output for debugging.
            selectors = {
                "error": "Failed to parse JSON from LLM output.",
                "raw_output": text_response,
                "exception": str(e),
            }

        # Return the CSS selectors in a message.
        return Message(sender=self.name, payload={"selectors": selectors})


class EmailOrganizerAgent(Agent):
    def __init__(self, name: str):
        super().__init__()
        self.cipher_suite = Fernet(os.environ.get("ENCRYPTION_KEY"))

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
        logger.info("Grabbing emails for user.")
        query_params = {
            "$select": "from,isRead,receivedDateTime,subject,body",
            "$top": 25,
            "$orderby": "receivedDateTime DESC",
        }
        request_configuration = RequestConfiguration(query_parameters=query_params)

        while True:
            messages = await self.user_client.users.by_user_id(user_id).messages.get(
                request_configuration=request_configuration
            )
            for message in messages.value:
                email = EmailMessage(
                    subject=self._encrypt_message(message.subject),
                    body=self._encrypt_message(message.body.content),
                    sender=self._encrypt_message(message.sender.email_address.address),
                    importance=message.importance,
                    conversation_id=message.conversation_id,
                    received_date=message.received_date_time,
                    message_id=message.internet_message_id,
                )
                yield email

            if messages.odata_next_link:
                logger.info("Found another page!")
                query_params["$top"] += 25
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
