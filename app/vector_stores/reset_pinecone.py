import os
import pinecone as pc
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from app.embeddings.openai import embeddings
from dotenv import load_dotenv

load_dotenv()

vector_store = LangchainPinecone.from_existing_index(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings
)

# Assuming LangchainPinecone was initialized already
pinecone_index = vector_store._index

try:
    vector_store._index.delete(delete_all=True)
    print("✅ Cleared namespace.")
except pc.core.client.exceptions.NotFoundException:
    print("⚠️ Namespace is empty or already deleted.")