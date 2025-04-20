import os
from dotenv import load_dotenv
from operator import itemgetter

from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.prompts import PromptTemplate
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnableLambda

# === 1. Load API key from .env ===
load_dotenv()

# === 2. Initialize the LLM ===
llm = ChatOpenAI(
    model="gpt-3.5-turbo",
    temperature=0,
    max_tokens=2000,
)

# === 3. Load PDF ===
pdf_folder_path = "PDFs"
pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
print(f"🗂️ Using PDF file: {pdf_files[0]}")
docs = PyMuPDFLoader(pdf_files[0]).load()

# === 4. Split into chunks ===
text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
split_docs = text_splitter.split_documents(docs)

# === 5. MAP Prompt ===
map_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Write a concise summary of the following:

    {text}

    SUMMARY:
    """
)
map_chain = map_prompt | llm

# === 6. REDUCE Prompt ===
reduce_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    You are an expert summarizer. Given the following partial summaries, create a cohesive and complete final summary:

    {text}

    FINAL SUMMARY:
    """
)
reduce_chain = reduce_prompt | llm

# === 7. MAP-REDUCE Chain ===

# MAP: summarize each chunk
mapped = RunnableLambda(lambda docs: [map_chain.invoke({"text": doc.page_content}).content for doc in docs])

# REDUCE: summarize the summaries
combine_summaries = RunnableLambda(lambda summaries: {"text": "\n\n".join(summaries)})
map_reduce_chain = mapped | combine_summaries | reduce_chain

# === 8. Run the full chain ===
final_summary = map_reduce_chain.invoke(split_docs).content

# === 9. Output ===
print("✅ Final MapReduce Summary:\n")
print(final_summary)
