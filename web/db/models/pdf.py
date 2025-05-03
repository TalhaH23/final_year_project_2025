from sqlalchemy import Column, String
from sqlalchemy.orm import relationship
from web.db import Base, BaseMixin
import uuid

class Pdf(Base, BaseMixin):
    __tablename__ = "pdfs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    
    conversations = relationship(
        "Conversation",
        back_populates="pdf",
        order_by="Conversation.created_on.desc()",
        cascade="all, delete-orphan",
    )

    def as_dict(self):
        return {"id": self.id, "name": self.name}