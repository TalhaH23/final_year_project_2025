import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import pymupdf
import asyncio
import aiofiles
import time
from typing import List
from collections import defaultdict
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from app.vector_stores.pinecone import vector_store 
from app.title_extraction import chunk_document_by_titles

load_dotenv()

# pdf_folder_path = "PDFs"
# pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

light_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=4000)  
strong_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, max_tokens=4000)  

chunk_summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are summarizing a small part of a scientific paper.

Summarization Rules:
- Write 5 to 6 sentences maximum.
- Focus ONLY on the key idea of the chunk.
- Do NOT list bullet points yet.
- Ignore references, citations, numeric markers (e.g., [1], (2020)).
- Output plain text, no formatting.

Chunk:

{text}
"""
)

chunk_summary_chain = chunk_summary_prompt | light_llm

section_summary_prompt = PromptTemplate(
    input_variables=["section_title", "summaries"],
    template="""
You are summarizing an entire section of a document based on smaller summaries.

Summarization Rules:
- Summarize into 3-6 bullet points maximum.
- Focus ONLY on the major concepts, definitions, findings, or important conclusions.
- Ignore minor examples, fine-grained methods unless critical.
- Keep each bullet short and clear.
- Ignore references and citations.

Formatting Rules:
- Output only raw HTML.
- Start with <h2>{section_title}</h2>
- Then a <ul> list with 3-6 <li> points.
- NO markdown, no triple backticks, no extra commentary.

Smaller Summaries:

{summaries}
"""
)

section_summary_chain = section_summary_prompt | strong_llm

# document_reduce_prompt = PromptTemplate(
#     input_variables=["text", "main_title"],
#     template="""
# You are organizing multiple section summaries into a final structured document.

# Instructions:
# - Maintain the original order of sections.
# - Do NOT alter, merge, or invent new sections.
# - Only output raw HTML.
# - Start with <h1>{main_title}</h1>
# - Then include the section summaries below it.
# - No markdown, no triple backticks.
# - Ignore Open Access, License, or Correspondence labels unless they contain real academic content.

# Here are the section summaries:

# {text}
# """
# )

document_reduce_prompt = PromptTemplate(
    input_variables=["text", "main_title"],
    template="""
You are an expert academic summarizer. Your task is to synthesize a clear, informative, and professional narrative summary of a scientific paper based on detailed section summaries.

Instructions:
- Write a narrative summary in 4-6 HTML <p> paragraphs.
- Cover the paper's purpose, methodology, proposed concepts or frameworks, key findings, and broader implications.
- Maintain the original order of the sections, but express the ideas in cohesive, flowing prose.
- Include brief explanations or definitions of key ideas if helpful, but avoid unnecessary technical detail.
- Write in a neutral, academic tone appropriate for a scientific audience.
- Only output raw HTML.
- Begin with <h1>{main_title}</h1>
- Follow with 4-6 <p> paragraphs containing the summary.
- Do NOT use bullet points, lists, or section headers.
- Do NOT include markdown, backticks, or references.
- Ignore Open Access, License, and Correspondence notices unless they contain real academic content.

Here are the section summaries:

{text}
"""
)

document_reduce_chain = document_reduce_prompt | strong_llm

def group_doc_by_section(docs: List[Document]):
    section_map = defaultdict(list)
    for doc in docs:
        section_map[doc.metadata.get("section_title", "Unknown Section")].append(doc)
    return section_map.values()

async def llm_summary(sections: List[Document]):
    section_summaries = []

    for section_docs in group_doc_by_section(sections):
        section_title = section_docs[0].metadata["section_title"]

        chunk_tasks = [
            chunk_summary_chain.ainvoke({"text": doc.page_content})
            for doc in section_docs
        ]
        chunk_summaries = await asyncio.gather(*chunk_tasks)
        merged_chunk_text = "\n".join([r.content for r in chunk_summaries])

        section_summary = await section_summary_chain.ainvoke({
            "section_title": section_title,
            "summaries": merged_chunk_text
        })

        section_summaries.append(section_summary.content)

    final_document_html = await document_reduce_chain.ainvoke({
        "text": "\n\n".join(section_summaries),
        "main_title": sections[0].metadata.get("main_title", "Untitled Document")
    })
    
    print(f"Document summarised")

    return final_document_html.content

async def process_single_pdf(file_path: str, docs: List[Document]) -> str:
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    start_time = time.perf_counter()

    try:
        summary = await llm_summary(docs)

        duration = time.perf_counter() - start_time
        print(f"✅ {base_name} summary done in {duration:.2f}s")

        summary_path = os.path.join("summaries", f"{base_name}_summary.txt")
        async with aiofiles.open(summary_path, "w", encoding="utf-8") as f:
            await f.write(summary)

        return summary

    except Exception as e:
        print(f"❌ Failed to process {base_name}: {e}")
        return ""


async def process_pdfs(filepaths: List[str]):
    tasks = []

    for file_path in filepaths:
        docs = chunk_document_by_titles(file_path, chunk_size=500, chunk_overlap=50)
        tasks.append(llm_summary(docs))  # call llm_summary directly

    summaries = await asyncio.gather(*tasks)
    return summaries
    
async def generate_summary(pdf_id, summary_folder, docs: List[Document]):
    file_path = os.path.join("uploads", f"{pdf_id}.pdf")
    summary_text = await llm_summary(docs)

    summary_filename = f"{pdf_id}.txt"
    summary_path = os.path.join(summary_folder, summary_filename)

    async with aiofiles.open(summary_path, 'w', encoding='utf-8') as f:
        await f.write(summary_text or "No summary generated.")

    return summary_path

 