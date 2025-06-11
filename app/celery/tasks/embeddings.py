from langchain_core.documents import Document
from app.vector_stores.pinecone import vector_store

def process_embeddings(pdf_id: str, serialized_docs: list[dict]):
    docs = [Document(**d) for d in serialized_docs]
    vector_store.add_documents(docs)
    print(f"✅ Embeddings created for PDF ID {pdf_id}")