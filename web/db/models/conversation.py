import uuid
from datetime import datetime, timezone
from sqlalchemy import Column, String, DateTime, ForeignKey, Integer
from sqlalchemy.orm import relationship

from web.db import Base, BaseMixin


class Conversation(Base, BaseMixin):
    __tablename__ = "conversation"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    created_on = Column(DateTime, default=lambda: datetime.now(timezone.utc))

    retriever = Column(String, nullable=True)
    memory = Column(String, nullable=True)
    llm = Column(String, nullable=True)

    pdf_id = Column(String, ForeignKey("pdfs.id"), nullable=False)
    pdf = relationship("Pdf", back_populates="conversations")

    # user_id = Column(Integer, ForeignKey("users.id"), nullable=False)
    # user = relationship("User", back_populates="conversation")

    messages = relationship(
        "Message", back_populates="conversation", order_by="Message.created_on", cascade="all, delete-orphan"
    )

    def as_dict(self):
        return {
            "id": self.id,
            # "created_on": self.created_on.isoformat(),
            "pdf_id": self.pdf_id,
            # "user_id": self.user_id,
            # "retriever": self.retriever,
            # "memory": self.memory,
            # "llm": self.llm,
            "messages": [m.as_dict() for m in self.messages],
        }
