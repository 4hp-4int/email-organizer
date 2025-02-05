from dataclasses import dataclass, field
from typing import Dict, Any, Optional
import datetime
import json


@dataclass
class Message:
    sender: str
    payload: Dict[str, Any]
    recipient: Optional[str] = None
    timestamp: str = field(
        default_factory=lambda: datetime.datetime.utcnow().isoformat()
    )

    def to_json(self) -> str:
        """
        Serialize the Message object to a JSON string.
        """
        message_dict = {
            "sender": self.sender,
            "recipient": self.recipient,
            "payload": self.payload,
            "timestamp": self.timestamp,
        }
        return json.dumps(message_dict)

    @classmethod
    def from_json(cls, json_str: str) -> "Message":
        """
        Deserialize a JSON string to a Message object.
        """
        data = json.loads(json_str)
        return cls(
            sender=data.get("sender", ""),
            recipient=data.get("recipient"),
            payload=data.get("payload", {}),
            timestamp=data.get("timestamp", datetime.datetime.utcnow().isoformat()),
        )

    def __repr__(self) -> str:
        return f"Message(sender={self.sender}, recipient={self.recipient}, payload={self.payload}, timestamp={self.timestamp})"
