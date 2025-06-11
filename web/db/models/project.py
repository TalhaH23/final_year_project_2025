from sqlalchemy import Column, String, Enum, Text
import uuid
from web.db import Base
from sqlalchemy.orm import relationship

class Project(Base):
    __tablename__ = "projects"

    id = Column(String, primary_key=True, default=lambda: str(uuid.uuid4()))
    name = Column(String, nullable=False)
    review_question = Column(Text, nullable=True)
    review_type = Column(Enum("intervention", "diagnostic", "prognostic", "methodological", "qualitative", name="review_type"), nullable=False)
    search_criteria = Column(String, nullable=True)
    filtered_pdf_ids = Column(Text, nullable=True)
    
    pdfs = relationship("Pdf", back_populates="project", cascade="all, delete-orphan")
