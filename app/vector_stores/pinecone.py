import os
from pinecone import Pinecone
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from app.embeddings.openai import embeddings
from dotenv import load_dotenv

load_dotenv()

pc = Pinecone(
    api_key=os.getenv("PINECONE_API_KEY")
)

vector_store = LangchainPinecone.from_existing_index(
    os.getenv("PINECONE_INDEX_NAME"),
    embeddings
)
