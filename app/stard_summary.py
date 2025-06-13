import os
import time
from dotenv import load_dotenv
import asyncio
from typing import List
from collections import defaultdict
from langchain_core.documents import Document
from langchain.prompts import PromptTemplate
from app.llms.chatopenai import light_llm, strong_llm

load_dotenv()

CHECKLIST_DIR = "app/checklists"
STARD_CHECKLIST_PATH = os.path.join(CHECKLIST_DIR, "stard.md")

with open(STARD_CHECKLIST_PATH, "r", encoding="utf-8") as f:
    stard_checklist = f.read()

chunk_summary_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are summarising a part of a scientific diagnostic accuracy paper.

Summarsation Rules:
- Write 5 to 6 sentences maximum.
- Ignore references, citations, and irrelevant metadata.
- Output plain text, no formatting.

Chunk:

{text}
"""
)

chunk_summary_chain = chunk_summary_prompt | light_llm

section_summary_prompt = PromptTemplate(
    input_variables=["section_title", "summaries"],
    template="""
You are summarising a section of a diagnostic accuracy paper using pre-summarised chunks.

Summarisation Rules:
- Write 3-6 bullet points.
- Avoid deep methodology unless relevant to diagnostic evaluation.

Formatting Rules:
- Output raw HTML only.
- Begin with <h2>{section_title}</h2>
- Then use a <ul> with 3-6 concise <li> points.

Summaries:

{summaries}
"""
)

section_summary_chain = section_summary_prompt | strong_llm

document_reduce_prompt = PromptTemplate(
    input_variables=["text", "main_title", "checklist"],
    template="""
You are writing a structured narrative summary of a diagnostic accuracy paper using the STARD checklist as a guide.

Instructions:
- Use the structure of the STARD checklist provided below.
- Summarise each applicable section based on the content.
- Ignore checklist items not covered in the summaries.
- Do not repeat the checklist verbatim.
- Do not critique or evaluate — focus on reporting what is present.
- Output only valid raw HTML.

<h1>{main_title}</h1>

Checklist Structure:
{checklist}

Section Summaries:
{text}
"""
)

document_reduce_chain = document_reduce_prompt | strong_llm

def group_doc_by_section(docs: List[Document]):
    """Group documents by their section titles"""
    
    section_map = defaultdict(list)
    for doc in docs:
        section_map[doc.metadata.get("section_title", "Unknown Section")].append(doc)
    return section_map.values()

async def llm_summary(sections: List[Document]):
    """Generate a structured summary of the provided sections using LLMs"""
    
    start_time = time.perf_counter()
    print(f"Summarising {len(sections)} sections...")
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
        "main_title": sections[0].metadata.get("main_title", "Untitled Document"),
        "checklist": stard_checklist
    })
    
    print(f"Document summarised")
    end_time = time.perf_counter() - start_time
    # print(f"Summary completed in {end_time:.2f} seconds.")

    return final_document_html.content

# async def process_single_pdf(file_path: str, docs: List[Document]) -> str:
#     base_name = os.path.splitext(os.path.basename(file_path))[0]
#     start_time = time.perf_counter()

#     try:
#         summary = await llm_summary(docs)

#         duration = time.perf_counter() - start_time
#         print(f"{base_name} summary done in {duration:.2f}s")

#         summary_path = os.path.join("summaries", f"{base_name}_summary.txt")
#         async with aiofiles.open(summary_path, "w", encoding="utf-8") as f:
#             await f.write(summary)

#         return summary

#     except Exception as e:
#         print(f"Failed to process {base_name}: {e}")
#         return ""


# async def process_pdfs(filepaths: List[str]):
#     tasks = []

#     for file_path in filepaths:
#         docs = chunk_document_by_titles(file_path, chunk_size=500, chunk_overlap=50)
#         tasks.append(llm_summary(docs))

#     summaries = await asyncio.gather(*tasks)
#     return summaries
    
# async def generate_summary(pdf_id, summary_folder, docs: List[Document]):
#     file_path = os.path.join("uploads", f"{pdf_id}.pdf")
#     summary_text = await llm_summary(docs)

#     summary_filename = f"{pdf_id}.txt"
#     summary_path = os.path.join(summary_folder, summary_filename)

#     async with aiofiles.open(summary_path, 'w', encoding='utf-8') as f:
#         await f.write(summary_text or "No summary generated.")

#     return summary_path

 