from sqlalchemy import Column, String, ForeignKey
from sqlalchemy.orm import relationship
from web.db import Base, BaseMixin
import uuid

class Pdf(Base, BaseMixin):
    __tablename__ = "pdfs"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    project_id = Column(String, ForeignKey("projects.id"))
    title = Column(String, nullable=True) 

    conversations = relationship(
        "Conversation",
        back_populates="pdf",
        order_by="Conversation.created_on.desc()",
        cascade="all, delete-orphan",
    )

    project = relationship("Project", back_populates="pdfs")

    def as_dict(self):
        return {"id": self.id, "name": self.name, "project_id": self.project_id, "title": self.title}
