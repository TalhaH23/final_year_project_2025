from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from app.vector_stores.pinecone import build_retriever
from langchain.memory import ConversationBufferMemory
from app.llms.chatopenai import build_llm
from app.memories.sql_memory import build_memory
from langchain_core.runnables import Runnable
from langchain_core.retrievers import BaseRetriever
from langchain_core.language_models import BaseChatModel
from app.models import ChatArgs
from sqlalchemy.orm import Session
    
def build_chat(chat_args: ChatArgs, db: Session):
    llm: BaseChatModel = build_llm(chat_args)
    retriever: BaseRetriever = build_retriever(chat_args)
    memory: ConversationBufferMemory = build_memory(chat_args, db)

    # Rewrite follow-ups
    contextualize_q_prompt = ChatPromptTemplate.from_messages([
        ("system", "Rewrite follow-up questions to be standalone. Only rewrite if needed."),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    history_aware_retriever = create_history_aware_retriever(
        llm=llm, retriever=retriever, prompt=contextualize_q_prompt
    )

    # Answering step
    qa_prompt = ChatPromptTemplate.from_messages([
        ("system", "Answer based only on the following context. Don't make things up.\n\n{context}"),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ])
    question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

    rag_chain: Runnable = create_retrieval_chain(
        retriever=history_aware_retriever,
        combine_docs_chain=question_answer_chain
    )

    return rag_chain, memory