from typing import Dict, List
from sqlalchemy.orm import Session
from langchain.schema.messages import AIMessage, HumanMessage, SystemMessage
from web.db import get_db
from web.db.models.message import Message
from web.db.models.conversation import Conversation


def get_messages_by_conversation_id(
    conversation_id: str,
    db: Session,
) -> List[AIMessage | HumanMessage | SystemMessage]:
    """
    Retrieves all messages that belong to the given conversation_id.

    :param conversation_id: The ID of the conversation.
    :param db: SQLAlchemy session (from FastAPI dependency).
    :return: List of LangChain messages.
    """
    messages = (
        db.query(Message)
        .filter(Message.conversation_id == conversation_id)
        .order_by(Message.created_on.desc())
        .all()
    )
    return [message.as_lc_message() for message in messages]


def add_message_to_conversation(
    db: Session,
    conversation_id: str,
    role: str,
    content: str,
) -> Message:
    """
    Creates and stores a new message tied to the given conversation_id
    with the provided role and content.

    :param db: SQLAlchemy session
    :param conversation_id: The id of the conversation
    :param role: The role of the message (e.g. "human", "ai", "system")
    :param content: The message content
    :return: The created Message object
    """
    message = Message(
        conversation_id=conversation_id,
        role=role,
        content=content,
    )
    db.add(message)
    db.commit()
    db.refresh(message)
    return message


def get_conversation_components(
    db: Session, conversation_id: str
) -> dict[str, str]:
    """
    Returns the components (llm, retriever, memory) used in a conversation.
    """
    conversation = db.query(Conversation).filter_by(id=conversation_id).first()
    if not conversation:
        raise ValueError(f"Conversation {conversation_id} not found.")
    
    return {
        "llm": conversation.llm,
        "retriever": conversation.retriever,
        "memory": conversation.memory,
    }


def set_conversation_components(
    db: Session, conversation_id: str, llm: str, retriever: str, memory: str
) -> None:
    """
    Sets the components used by a conversation.
    """
    conversation = db.query(Conversation).filter_by(id=conversation_id).first()
    if not conversation:
        raise ValueError(f"Conversation {conversation_id} not found.")
    
    conversation.llm = llm
    conversation.retriever = retriever
    conversation.memory = memory
    db.commit()
