import unittest
from datetime import date, datetime
from agent import EmailMessage, EmailOrganizerAgent
from unittest.mock import patch, MagicMock
import pytz


class TestEmailMessage(unittest.TestCase):
    def test_email_message_initialization(self):
        email_message = EmailMessage(
            subject="Test Subject",
            body="Test Body",
            sender="test@example.com",
            importance="high",
            message_id="12345",
            conversation_id="67890",
            received_date=date.today(),
        )
        self.assertEqual(email_message.subject, "Test Subject")
        self.assertEqual(email_message.body, "Test Body")
        self.assertEqual(email_message.sender, "test@example.com")
        self.assertEqual(email_message.importance, "high")
        self.assertEqual(email_message.message_id, "12345")
        self.assertEqual(email_message.conversation_id, "67890")
        self.assertEqual(email_message.received_date, date.today())
        self.assertEqual(email_message.embeddings, [])
        self.assertEqual(email_message.embedding_text, "")


class TestEmailOrganizerAgent(unittest.TestCase):
    @patch("agent.Config")
    @patch("agent.Fernet")
    @patch("agent.spacy.load")
    @patch("agent.SentenceTransformer")
    @patch("agent.ClientSecretCredential")
    @patch("agent.GraphServiceClient")
    def setUp(
        self,
        MockGraphServiceClient,
        MockClientSecretCredential,
        MockSentenceTransformer,
        MockSpacyLoad,
        MockFernet,
        MockConfig,
    ):
        self.mock_config = MockConfig.return_value
        self.mock_config.ENCRYPTION_KEY = "test_key"
        self.mock_config.MODEL = "all-MiniLM-L6-v2"
        self.mock_config.AZURE_TENANT_ID = "tenant_id"
        self.mock_config.AZURE_CLIENT_ID = "client_id"
        self.mock_config.AZURE_CLIENT_SECRET = "client_secret"
        self.mock_config.AZURE_GRAPH_SCOPES = ["scope1", "scope2"]
        self.mock_config.EMBEDDING_FORMAT_STRING = "{subject} {body}"
        self.mock_config.TOPIC_LABELS = {"label1": "Label 1", "label2": "Label 2"}

        self.mock_fernet = MockFernet.return_value
        self.mock_spacy = MockSpacyLoad.return_value
        self.mock_model = MockSentenceTransformer.return_value
        self.mock_credential = MockClientSecretCredential.return_value
        self.mock_graph_client = MockGraphServiceClient.return_value

        self.agent = EmailOrganizerAgent(name="Test Agent")

    def test_encrypt_message(self):
        message = {"key": "value"}
        self.mock_fernet.encrypt.return_value = b"encrypted_message"
        encrypted_message = self.agent._encrypt_message(message)
        self.assertEqual(encrypted_message, b"encrypted_message")

    def test_decrypt_message(self):
        encrypted_message = b"encrypted_message"
        self.mock_fernet.decrypt.return_value = b'{"key": "value"}'
        decrypted_message = self.agent._decrypt_message(encrypted_message)
        self.assertEqual(decrypted_message, {"key": "value"})

    def test_preprocess_function(self):
        email = {"subject": "Test Subject", "body": "Test Body"}
        self.mock_spacy.return_value = MagicMock()
        self.mock_spacy.return_value.__iter__.return_value = [
            MagicMock(text="Test", is_stop=False, is_punct=False),
            MagicMock(text="Body", is_stop=False, is_punct=False),
        ]
        cleaned_text = self.agent.preprocess_function(email)
        self.assertEqual(cleaned_text, "Test Body")


if __name__ == "__main__":
    unittest.main()
