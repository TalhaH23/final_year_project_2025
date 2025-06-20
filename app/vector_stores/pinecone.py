import os
import time
from langchain_community.vectorstores import Pinecone as LangchainPinecone
from langchain_core.documents import Document
from app.embeddings.openai import embeddings
from dotenv import load_dotenv

load_dotenv()

vector_store = LangchainPinecone.from_existing_index(
    index_name=os.getenv("PINECONE_INDEX_NAME"),
    embedding=embeddings
)

def build_retriever(chat_args):
    """
    Builds a retriever for the vector store based on the provided chat arguments
    """
    search_kwargs = {"filter": {"pdf_id": chat_args.pdf_id}}
    return vector_store.as_retriever(
        search_kwargs=search_kwargs,
    )
    
def process_embeddings(pdf_id: str, serialized_docs: list[dict]):
    """
    Processes and adds embeddings to the vector store for provided PDF chunks
    """
    start = time.perf_counter()
    # print(f"Creating embeddings for PDF ID {pdf_id}...")
    docs = [Document(**d) for d in serialized_docs]
    vector_store.add_documents(docs)
    end_time = time.perf_counter() - start
    # print(f"Embeddings created for PDF ID {pdf_id} in {end_time:.2f} seconds.")