from pydantic import BaseModel, Extra


class Metadata(BaseModel, extra=Extra.allow):
    """Metadata for the chat session, including conversation and PDF ID"""
    conversation_id: str
    pdf_id: str

class ChatArgs(BaseModel, extra=Extra.allow):
    """Arguments for initiating a chat session with metadata"""
    conversation_id: str
    pdf_id: str
    metadata: Metadata
    streaming: bool
