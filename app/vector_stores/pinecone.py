import os
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from app.embeddings.openai import embeddings
from dotenv import load_dotenv

load_dotenv()

vector_store = LangchainPinecone.from_existing_index(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings
)

def build_retriever(chat_args):
    search_kwargs = {"filter": {"pdf_id": chat_args.pdf_id}}
    return vector_store.as_retriever(
        search_kwargs=search_kwargs,
    )