import os
import asyncio
import aiofiles
import time
import json
from dotenv import load_dotenv
from langchain.prompts import PromptTemplate
from typing import List
from collections import defaultdict
from langchain_core.documents import Document
from collections import defaultdict
from app.vector_stores.pinecone import vector_store
from app.stard_summary import llm_summary, group_doc_by_section
from app.criteria.criteria import parse_llm_screening_output
from app.llms.chatopenai import light_llm, strong_llm

load_dotenv()

CHECKLIST_DIR = "app/checklists"
PRISMA_CHECKLIST_PATH = os.path.join(CHECKLIST_DIR, "prisma.md")

with open(PRISMA_CHECKLIST_PATH, "r", encoding="utf-8") as f:
    prisma_checklist = f.read()

def generate_review_prompt(criteria: list[str]) -> PromptTemplate:
    """Generates a prompt template for systematic review screening"""
    
    criteria_format = "\n".join(
        f"{c}: [Matched / Not Matched / N/A] [brief summary]" for c in criteria
    )

    template = f"""
Given the systematic review question and the summary of the document, decide if the document should be included in the systematic review. Use the following criteria to guide your decision: {", ".join(criteria)}.

Systematic Review Question:
{{review_question}}

Summary:
{{summary}}

Return your answer in the following format:
Decision: [Include / Exclude / Unclear]  
Confidence: [1 to 5]  
{criteria_format}
Rationale: [brief explanation]
"""
    return PromptTemplate(
        input_variables=["review_question", "summary"],
        template=template.strip()
    )


async def llm_screening(review_question: str | None, summary: str, criteria: list[str]):
    """Screen documents for systematic review based on the provided question and criteria"""
    
    start_time = time.perf_counter()
    print(f"Screening documents for question: {review_question} with criteria: {criteria}")
    if not review_question:
        raise ValueError("Review question is required for screening.")

    try:
        prompt = generate_review_prompt(criteria)
        print(prompt.template)
        review_question_chain = prompt | light_llm
        response = await review_question_chain.ainvoke({
            "review_question": review_question,
            "summary": summary
        })
        end_time = time.perf_counter() - start_time
        # print(f"Screening completed in {end_time:.2f} seconds.")
        return response.content
    except Exception as e:
        print(f"Error in llm_screening: {e}")
        return None
    
async def get_screening_result(
    pdf_id, review_question, summary_folder, review_result_folder,
    docs: List[Document], criteria: List[str]
):
    """Get the screening result for a PDF document based on the summary, question and criteria"""
    
    start_time = time.perf_counter()
    print(f"Getting screening result for PDF ID: {pdf_id}")
    summary_text = await llm_summary(docs)
    raw_screening = await llm_screening(review_question, summary_text, criteria)
    screening_result = parse_llm_screening_output(raw_screening, criteria)

    summary_path = os.path.join(summary_folder, f"{pdf_id}.txt")
    async with aiofiles.open(summary_path, 'w', encoding='utf-8') as f:
        await f.write(summary_text or "No summary generated.")

    review_result_path = os.path.join(review_result_folder, f"{pdf_id}_screening_result.json")
    async with aiofiles.open(review_result_path, 'w', encoding='utf-8') as f:
        await f.write(json.dumps(screening_result, indent=2))

    end_time = time.perf_counter() - start_time
    # print(f"Screening result for {pdf_id} total time taken: {end_time:.2f} seconds.")
    return summary_path


def filter_documents_by_similarity(
    query: str,
    ids: List[str],
    n: int = 10,
) -> List[str]:
    """
    Filter documents based on similarity to a query, restricting to a provided list of document IDs.
    """
    start_time = time.perf_counter()
    print(f"Filtering documents for query: {query} with IDs: {ids}")
    results = vector_store.similarity_search_with_score(query, k=100)

    doc_scores = defaultdict(list)
    for doc, score in results:
        pdf_id = doc.metadata.get("pdf_id")
        if pdf_id in ids:
            doc_scores[pdf_id].append(score)

    ranked_docs = sorted(doc_scores.items(), key=lambda x: sum(x[1]) / len(x[1]), reverse=True)
    top_docs = [pdf_id for pdf_id, _ in ranked_docs[:n]]
    end_time = time.perf_counter() - start_time
    # print(f"Filtered {len(top_docs)} documents in {end_time:.2f} seconds.")

    return top_docs

async def wait_for_embeddings(pdf_ids: List[str], timeout: int = 60, poll_interval: int = 5) -> None:
    """Wait for embeddings to be ready in the vector store for the given PDF IDs"""
    
    print(f"Waiting for embeddings for {len(pdf_ids)} PDFs...")

    start = time.time()
    remaining = set(pdf_ids)

    while time.time() - start < timeout and remaining:
        print(f"Checking Pinecone for {len(remaining)} remaining PDFs...")
        for pdf_id in list(remaining):
            try:
                results = vector_store.similarity_search("placeholder", k=1, filter={"pdf_id": pdf_id})
                if results:
                    remaining.remove(pdf_id)
            except Exception as e:
                print(f"Error querying vector store for {pdf_id}: {e}")
        if remaining:
            await asyncio.sleep(poll_interval)

    if remaining:
        print(f"Timeout. Missing embeddings for: {remaining}")
    else:
        print("All embeddings ready.")


###------------------------------------ SYSTEMATIC REVIEW EVALUATION ------------------------------------###
chunk_evaluation_prompt = PromptTemplate(
    input_variables=["chunk_text", "checklist"],
    template="""
You are assessing whether this part of a systematic review satisfies relevant items in the PRISMA 2020 checklist.

Checklist:
{checklist}

Document Chunk:
{chunk_text}

Instructions:
- Review the chunk against the checklist.
- Only comment on checklist items that this chunk addresses.
- For each applicable item, provide:
  - Rating: Yes / Partially / No
  - Justification
- Format your answer as a list of items with headings and bullet points.
"""
)

chunk_evaluation_chain = chunk_evaluation_prompt | strong_llm

doc_evaluation_prompt = PromptTemplate(
    input_variables=["chunk_evaluations", "main_title", "checklist"],
    template="""
You are an expert reviewer synthesizing a final evaluation of a systematic review using the PRISMA 2020 checklist.

Checklist:
{checklist}

Title of Systematic Review:
{main_title}

Below are evaluations of different parts of the document, each judged against the PRISMA checklist:

Chunk-Level Evaluations:
{chunk_evaluations}

Instructions:
- For each checklist item, combine the insights from the chunk evaluations.
- Decide on a final rating: Yes / Partially / No.
- Provide a single, brief justification for each item using evidence from the chunk evaluations.
- Format your final evaluation in **Markdown**, as a list of PRISMA items with:
  - Item number or name
  - Final rating
  - Final justification
"""
)


doc_evaluation_chain = doc_evaluation_prompt | strong_llm

async def llm_evaluate(sections: List[Document]) -> str:
    chunk_tasks = []

    for section_docs in group_doc_by_section(sections):
        for doc in section_docs:
            chunk_tasks.append(chunk_evaluation_chain.ainvoke({
                "chunk_text": doc.page_content,
                "checklist": prisma_checklist
            }))

    chunk_evaluations = await asyncio.gather(*chunk_tasks)

    combined_chunk_evaluations = "\n\n".join([c.content for c in chunk_evaluations])

    main_title = sections[0].metadata.get("main_title", "Untitled Review")

    final_evaluation = await doc_evaluation_chain.ainvoke({
        "chunk_evaluations": combined_chunk_evaluations,
        "main_title": main_title,
        "checklist": prisma_checklist
    })

    print("Final PRISMA evaluation complete.")
    return final_evaluation.content



# async def process_pdfs(filepaths: List[str]):
#     tasks = []

#     for file_path in filepaths:
#         docs = chunk_document_by_titles(file_path, chunk_size=500, chunk_overlap=50)
#         tasks.append(llm_evaluate(docs))

#     evaluation = await asyncio.gather(*tasks)
#     print(evaluation)
#     return evaluation

