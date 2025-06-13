import time
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
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

- If the information is clearly described, summarise it concisely in 1-3 sentences.
- If it is not present, return exactly: Not specified.

Answer:
"""
)

evidence_table_chain = prompt_template | light_llm

async def extract_component(element: str, docs: List[Document], fallback_docs: List[Document], k: int = 5) -> str:
    """Extract a summary of a specific criteria component from the provided chunks"""
    
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
        print(f"Error extracting '{element}': {e}")
        return "Extraction failed."

async def create_evidence_table(pdfs: Dict[str, Pdf], criteria: List[str], k: int = 5) -> List[dict]:
    """Create an evidence table from the provided PDFs based on specified criteria"""
    
    all_data = []
    start_time = time.perf_counter()
    print(f"Creating evidence table for {len(pdfs)} PDFs with criteria: {criteria}")

    for pdf_id, pdf_obj in pdfs.items():
        print(f"\n Processing: {pdf_id}")
        entry = {
            "Document": pdf_obj.title or pdf_obj.name or pdf_id
        }

        try:
            fallback_docs = vector_store.similarity_search("full text", k=100, filter={"pdf_id": pdf_id})
        except Exception as e:
            print(f"Failed to load full document for {pdf_id}: {e}")
            fallback_docs = []

        for element in criteria:
            guidance = CRITERIA_GUIDANCE.get(element, {})
            query = guidance.get("query", f"{element} of the study")
            print(f"Querying: {query}")

            try:
                docs = vector_store.similarity_search(query=query, k=k, filter={"pdf_id": pdf_id})
                print(f"Retrieved {len(docs)} chunks")
            except Exception as e:
                print(f"Error during similarity search: {e}")
                docs = []

            summary = await extract_component(element, docs, fallback_docs, k)
            entry[element] = summary

        all_data.append(entry)

    # with open("evidence_table.json", "w") as f:
    #     json.dump(all_data, f, indent=2)

    # print("Evidence table written to evidence_table.json")
    end_time = time.perf_counter() - start_time
    # print(f"Evidence table created in {end_time:.2f} seconds for {len(pdfs)} PDFs.")
    return all_data

        
