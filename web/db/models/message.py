# app/web/models/message.py

import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage

from web.db import Base, BaseMixin


class Message(Base, BaseMixin):
    __tablename__ = "messages"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_on = Column(DateTime, default=lambda: datetime.now(timezone.utc))
    role = Column(String, nullable=False)
    content = Column(String, nullable=False)

    conversation_id = Column(String, ForeignKey("conversation.id"), nullable=False)
    conversation = relationship("Conversation", back_populates="messages")

    def as_dict(self):
        return {
            "id": self.id,
            "role": self.role,
            "content": self.content,
            # "created_on": self.created_on.isoformat(),
            # "conversation_id": self.conversation_id,
        }

    def as_lc_message(self) -> HumanMessage | AIMessage | SystemMessage:
        if self.role == "human":
            return HumanMessage(content=self.content)
        elif self.role == "ai":
            return AIMessage(content=self.content)
        elif self.role == "system":
            return SystemMessage(content=self.content)
        else:
            raise ValueError(f"Unknown message role: {self.role}")
