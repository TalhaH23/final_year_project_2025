import os
import pinecone
from langchain.vectorstores.pinecone import Pinecone
from app.embeddings.openai import embeddings
from dotenv import load_dotenv

load_dotenv()


pinecone.Pinecone(
    api_key=os.getenv("PINECONE_API_KEY"),
    environment=os.getenv("PINECONE_ENV_NAME"),
)

vector_store = Pinecone.from_existing_index(os.getenv("PINECONE_INDEX_NAME"), embeddings)