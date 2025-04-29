import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import pymupdf
from typing import List
from collections import defaultdict
from langchain_core.documents import Document
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from app.vector_stores.pinecone import vector_store 
from app.title_extraction import section_headers, chunk_document_by_titles

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

def llm_summary(sections: List[Document]):
    section_summaries = []

    for section_docs in group_doc_by_section(sections):
        section_title = section_docs[0].metadata["section_title"]

        chunk_summaries = [
            chunk_summary_chain.invoke({"text": doc.page_content}).content
            for doc in section_docs
        ]

        merged_chunk_text = "\n".join(chunk_summaries)
        section_summary = section_summary_chain.invoke({
            "section_title": section_title,
            "summaries": merged_chunk_text
        }).content

        section_summaries.append(section_summary)

    final_document_html = document_reduce_chain.invoke({
        "text": "\n\n".join(section_summaries),
        "main_title": sections[0].metadata.get("main_title", "Untitled Document")
    }).content

    return final_document_html


def process_single_pdf(file_path):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n--- Processing: {os.path.basename(file_path)} ---")

    try:
        titles = section_headers(file_path)
        # print(f"Found {len(titles)} section headers")
        print(f"Extracted Titles for {base_name}")

        chunked_docs = chunk_document_by_titles(file_path, titles, chunk_size=500, chunk_overlap=50)
        # print(f"Chunked into {len(chunked_docs)} titled sections")
        print(f"Chunked Documents for {base_name}")
        
        chunk_output_path = os.path.join("app/summaries", f"{base_name}_chunks.txt")
        os.makedirs("summaries", exist_ok=True)

        with open(chunk_output_path, "w", encoding="utf-8") as f:
            for i, doc in enumerate(chunked_docs, 1):
                f.write(f"\n\n Chunk {i}: {doc.metadata.get('section_title', 'Untitled')}\n")
                f.write("-" * 40 + "\n")
                f.write(doc.page_content.strip())
                f.write("\n" + "=" * 60)

        # vector_store.add_documents(chunked_docs)

        summary = llm_summary(chunked_docs)
        # print(f"\nFinal Summary for {os.path.basename(file_path)}:\n{summary}\n")
        print(f"Generated Summary for {base_name}")

        summary_output_path = os.path.join("app/summaries", f"{base_name}_summary.txt")
        with open(summary_output_path, "w", encoding="utf-8") as f:
            f.write(summary)
        return summary

    except Exception as e:
        print(f"Error while processing {file_path}: {str(e)}")

# Parallel processing
# with ThreadPoolExecutor(max_workers=4) as executor:
#     executor.map(process_single_pdf, pdf_files)

# process_single_pdf(pdf_files[1])
