from langchain.memory import ConversationBufferMemory
from langchain.schema import BaseChatMessageHistory
from web.api import get_messages_by_conversation_id, add_message_to_conversation
from sqlalchemy.orm import Session

class SQLMessageHistory(BaseChatMessageHistory):
    """SQL-based message history for LangChain conversations"""
    def __init__(self, conversation_id: str, db: Session):
        self.conversation_id = conversation_id
        self.db = db
    
    @property
    def messages(self):
        return get_messages_by_conversation_id(self.conversation_id, self.db)
    
    def add_message(self, message):
        return add_message_to_conversation(
            db=self.db,
            conversation_id=self.conversation_id,
            role=message.type,
            content=message.content
        )
        
    def clear(self):
        pass
    
def build_memory(chat_args, db: Session):
    """
    Builds a conversation memory that stores chat history in a SQL database
    """
    return ConversationBufferMemory(
        chat_memory=SQLMessageHistory(
            conversation_id=chat_args.conversation_id,
            db=db
        ),
        memory_key="chat_history",
        return_messages=True,
        output_key="answer",
    )