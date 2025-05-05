from app.celery import celery_app
from langchain_core.documents import Document

@celery_app.task(name="process_embeddings")
def process_embeddings(pdf_id: str, serialized_docs: list[dict]):
    docs = [Document(**d) for d in serialized_docs]
    from app.vector_stores.pinecone import vector_store
    vector_store.add_documents(docs)
    print(f"Embeddings created for PDF ID {pdf_id}")
