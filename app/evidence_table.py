import os
import pandas as pd
import json
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain.prompts import PromptTemplate
import asyncio
import aiofiles
import time
import re
import json
from typing import List, Dict
from langchain_core.documents import Document
from app.vector_stores.pinecone import vector_store
from app.llms.chatopenai import light_llm
from app.criteria.criteria import CRITERIA_GUIDANCE
from web.db.models.pdf import Pdf

load_dotenv()

prompt_template = PromptTemplate.from_template(
    """
Study Text:
{text}

Task:
{instruction}

- If the information is clearly described, summarize it concisely in 1-3 sentences.
- If it is not present, return exactly: Not specified.

Answer:
"""
)

evidence_table_chain = prompt_template | light_llm

# # === Extract Component with Retry ===
# def extract_component(element: str, docs: List[Document], k: int = 5) -> str:
#     combined = "\n\n".join(doc.page_content for doc in docs[:k])
#     inputs = {"element": element, "text": combined}

#     try:
#         response = evidence_table_chain.invoke(inputs)
#         return response.content.strip()
#     except Exception as e:
#         print(f"⚠️ Error for '{element}': {e}")

#     return f"{element}: Extraction failed"


async def extract_component(element: str, docs: List[Document], fallback_docs: List[Document], k: int = 5) -> str:
    combined = "\n\n".join(doc.page_content for doc in docs[:k]) if docs else ""
    if not combined and fallback_docs:
        combined = "\n\n".join(doc.page_content for doc in fallback_docs)
    if not combined:
        return "Not specified."

    guidance = CRITERIA_GUIDANCE.get(element, {})
    instruction = guidance.get("instruction", f"Extract the {element} from the study.")

    try:
        response = await evidence_table_chain.ainvoke({
            "text": combined,
            "instruction": instruction,
        })
        return response.content.strip()
    except Exception as e:
        print(f"❌ Error extracting '{element}': {e}")
        return "Extraction failed."




# # === Main Function ===
# def create_evidence_table(ids: List[str], criteria: List[str], k: int = 5) -> List[dict]:
#     all_data = []

#     for id in ids:
#         print(f"Processing: {id}")
#         entry = {"Document": id}

#         for element in criteria:
#             query = f"{element} of the study"
#             print(f"🔍 Querying: {query} for doc: {id}")
            
#             docs = vector_store.similarity_search(query=query, k=k, filter={"pdf_id": id})
#             print(f"Retrieved {len(docs)} chunks")

#             if docs:
#                 print(f"Top doc preview:\n{docs[0].page_content[:300]}")
#                 summary = extract_component(element, docs, k)
#                 entry[element] = summary
#             else:
#                 entry[element] = f"{element}: Not found"

#         all_data.append(entry)

#         # Markdown export
#         doc_id = os.path.splitext(os.path.basename(id))[0]
#         md_lines = [f"# Evidence Summary: {doc_id}"] + [
#             f"**{key}**: {val}" for key, val in entry.items() if key != "Document"
#         ]
#         with open(f"markdown_summaries/{doc_id}.md", "w") as f:
#             f.write("\n\n".join(md_lines))

#     # Optional: persist raw JSON
#     with open("evidence_table.json", "w") as f:
#         json.dump(all_data, f, indent=2)
#     print("✅ Raw evidence data saved to evidence_table.json")

#     return all_data  # return JSON-like list of dicts



async def create_evidence_table(pdfs: Dict[str, Pdf], criteria: List[str], k: int = 5) -> List[dict]:
    all_data = []

    for pdf_id, pdf_obj in pdfs.items():
        print(f"\n📄 Processing: {pdf_id}")
        entry = {
            "Document": pdf_obj.title or pdf_obj.name or pdf_id
        }

        try:
            fallback_docs = vector_store.similarity_search("full text", k=100, filter={"pdf_id": pdf_id})
        except Exception as e:
            print(f"❌ Failed to load full document for {pdf_id}: {e}")
            fallback_docs = []

        # Extract each criterion sequentially (or async if needed)
        for element in criteria:
            guidance = CRITERIA_GUIDANCE.get(element, {})
            query = guidance.get("query", f"{element} of the study")
            print(f"🔍 Querying: {query}")

            try:
                docs = vector_store.similarity_search(query=query, k=k, filter={"pdf_id": pdf_id})
                print(f"✅ Retrieved {len(docs)} chunks")
            except Exception as e:
                print(f"❌ Error during similarity search: {e}")
                docs = []

            summary = await extract_component(element, docs, fallback_docs, k)
            entry[element] = summary

        all_data.append(entry)

    # Save full JSON
    with open("evidence_table.json", "w") as f:
        json.dump(all_data, f, indent=2)

    print("✅ Evidence table written to evidence_table.json")
    return all_data

        
