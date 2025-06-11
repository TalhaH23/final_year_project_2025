import os
import aiofiles
import asyncio
import time
import json
import logging
from typing import List, Dict
from langchain_core.documents import Document
# What is the effectiveness of cognitive behavioral therapy (CBT) for treating depression in adolescents?

from fastapi import FastAPI, UploadFile, File, Request, Depends, Form
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, RedirectResponse, JSONResponse
from fastapi.concurrency import run_in_threadpool

# from app.chunking_test import generate_summary
from app.vector_stores.pinecone import process_embeddings
from app.title_extraction import chunk_document_by_titles, chunk_docs
from app.celery.tasks import embeddings
from web.routes.conversation_messages import router as conversation_router
from app.systematic_review import filter_documents_by_similarity, wait_for_embeddings, get_screening_result
from app.evidence_table import create_evidence_table
from app.criteria.criteria import criteria_dict

from web.db import get_db
from web.db.models.pdf import Pdf
from web.db.models.project import Project
from web.db.models.conversation import Conversation
from sqlalchemy.orm import Session
import uuid


# Configure Logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Paths
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
SUMMARY_FOLDER = os.path.join(os.getcwd(), 'summaries')
REVIEW_RESULT_FOLDER = os.path.join(os.getcwd(), 'review_results')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)
os.makedirs(REVIEW_RESULT_FOLDER, exist_ok=True)

# FastAPI Setup
app = FastAPI()
app.include_router(conversation_router)
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates") 

# Helpers
async def process_single_pdf(pdf: UploadFile, project: Project, db: Session) -> tuple[str, List[Document]]:
    if not pdf.filename.endswith('.pdf'):
        return None, []

    pdf_id = str(uuid.uuid4())
    file_path = os.path.join(UPLOAD_FOLDER, f"{pdf_id}.pdf")

    # Save file (async)
    async with aiofiles.open(file_path, 'wb') as f:
        await f.write(await pdf.read())

    logger.info(f"Uploaded {pdf.filename} to {file_path}")

    # Chunking (CPU-bound)
    chunked_docs, pdf_title = await run_in_threadpool(chunk_document_by_titles, file_path, 500, 50)

    db_pdf = Pdf(id=pdf_id, name=pdf.filename, project_id=project.id, title=pdf_title)
    db.add(db_pdf)

    serialized_docs = [
        {"page_content": doc.page_content, "metadata": {**doc.metadata, "pdf_id": pdf_id}}
        for doc in chunked_docs
    ]

    # Embedding (sync I/O-bound)
    logger.info(f"Creating embeddings for {pdf.filename}")
    try:
        await run_in_threadpool(process_embeddings, pdf_id, serialized_docs)
    except Exception as e:
        logger.error(f"Failed to process embeddings for {pdf.filename}: {e}")

    return pdf_id, chunked_docs

async def process_uploaded_pdfs(pdfs: List[UploadFile], project: Project, db: Session) -> Dict[str, List]:
    tasks = [process_single_pdf(pdf, project, db) for pdf in pdfs]
    results = await asyncio.gather(*tasks)

    all_pdf_ids = []
    chunks_dict = {}

    for pdf_id, chunks in results:
        if pdf_id:
            all_pdf_ids.append(pdf_id)
            chunks_dict[pdf_id] = chunks

    db.commit()
    return all_pdf_ids, chunks_dict

# Routes
@app.post("/projects/new", response_class=HTMLResponse)
async def create_project(
    request: Request,
    name: str = Form(...),
    review_question: str = Form(...),
    review_type: str = Form(...),
    search_criteria: str = Form(...),
    pdfs: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    project = Project(
        name=name,
        review_question=review_question,
        review_type=review_type,
        search_criteria=search_criteria,
    )
    db.add(project)
    db.commit()
    db.refresh(project)

    logger.info("Starting upload and processing")
    start = time.perf_counter()

    all_pdf_ids, chunks_dict = await process_uploaded_pdfs(pdfs, project, db)

    await wait_for_embeddings(all_pdf_ids, timeout=120, poll_interval=5)

    filtered_ids = filter_documents_by_similarity(review_question, all_pdf_ids, n=3)
    project.filtered_pdf_ids = json.dumps(filtered_ids)
    db.commit()

    tasks = [
        get_screening_result(
            pdf_id, review_question, SUMMARY_FOLDER, REVIEW_RESULT_FOLDER,
            chunks_dict[pdf_id], criteria_dict.get(project.search_criteria, [])
        )
        for pdf_id in filtered_ids
    ]
    await asyncio.gather(*tasks)

    logger.info(f"✅ Project completed in {time.perf_counter() - start:.2f}s")
    return RedirectResponse("/", status_code=303)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, db: Session = Depends(get_db)):
    projects = db.query(Project).all()
    return templates.TemplateResponse("home.html", {"request": request, "projects": projects})

@app.post("/projects/{project_id}/upload", response_class=HTMLResponse)
async def handle_upload(
    request: Request,
    project_id: str,
    pdfs: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    project = db.query(Project).filter_by(id=project_id).first()
    if not project:
        return HTMLResponse(content="Invalid project ID", status_code=400)

    all_pdf_ids, chunks_dict = await process_uploaded_pdfs(pdfs, project, db)

    # Wait for embeddings
    await wait_for_embeddings(all_pdf_ids, timeout=120, poll_interval=5)

    # Recalculate filtering
    review_question = project.review_question
    filtered_ids = json.loads(project.filtered_pdf_ids or "[]")
    new_filtered = filter_documents_by_similarity(review_question, all_pdf_ids, n=3)
    merged_filtered_ids = list(set(filtered_ids + new_filtered))
    project.filtered_pdf_ids = json.dumps(merged_filtered_ids)
    
    db_pdfs = db.query(Pdf).filter(Pdf.id.in_(merged_filtered_ids)).all()
    filtered_pdfs = {pdf.id: pdf for pdf in db_pdfs}
    db.commit()
    
    criteria = criteria_dict.get(project.search_criteria, [])

    # Run screening results for new PDFs
    tasks = [
        get_screening_result(pdf_id, review_question, SUMMARY_FOLDER, REVIEW_RESULT_FOLDER, chunks_dict[pdf_id], criteria)
        for pdf_id in new_filtered
    ]
    await asyncio.gather(*tasks)

    # # Rebuild evidence table (optional: only for new_filtered)
    # await create_evidence_table(filtered_pdfs, criteria, k=5)

    return RedirectResponse(f"/projects/{project_id}", status_code=303)


@app.get("/projects/{project_id}", response_class=HTMLResponse)
async def view_project(request: Request, project_id: str, db: Session = Depends(get_db)):
    project = db.query(Project).filter_by(id=project_id).first()
    if not project:
        return HTMLResponse(content="Project not found", status_code=404)

    pdfs = db.query(Pdf).filter_by(project_id=project_id).all()
    filtered_ids = json.loads(project.filtered_pdf_ids or "[]")
    filtered_pdfs = {pdf.id: pdf for pdf in pdfs if pdf.id in filtered_ids}
    
    # Load screening results from disk
    screening_decisions = {}
    for pdf_id in filtered_ids:
        review_path = os.path.join("review_results", f"{pdf_id}_screening_result.json")
        if os.path.exists(review_path):
            with open(review_path, "r", encoding="utf-8") as f:
                try:
                    result = json.load(f)
                    screening_decisions[pdf_id] = result.get("decision", "Unclear")
                except Exception:
                    screening_decisions[pdf_id] = "Unclear"

    return templates.TemplateResponse("project_detail.html", {
        "request": request,
        "project": project,
        "pdfs": pdfs,
        "screening_decisions": screening_decisions,
        "evidence_table": [],
    })
    
@app.post("/projects/{project_id}/evidence", response_class=HTMLResponse)
async def generate_evidence_table(
    request: Request,
    project_id: str,
    pdf_ids: List[str] = Form(...),
    db: Session = Depends(get_db)
):
    pdfs = db.query(Pdf).filter(Pdf.id.in_(pdf_ids)).all()
    pdf_dict = {pdf.id: pdf for pdf in pdfs}

    project = db.query(Project).filter_by(id=project_id).first()
    criteria = criteria_dict.get(project.search_criteria, [])

    table = await create_evidence_table(pdf_dict, criteria, k=5)

    # ✅ Save table to disk so it can be retrieved later
    cached_path = os.path.join("review_results", f"{project_id}_evidence_table.json")
    with open(cached_path, "w", encoding="utf-8") as f:
        json.dump(table, f, indent=2)

    return templates.TemplateResponse("evidence_modal.html", {
        "request": request,
        "evidence_table": table
    })

@app.get("/projects/{project_id}/evidence_cached", response_class=HTMLResponse)
async def get_cached_evidence_table(
    request: Request,
    project_id: str,
    db: Session = Depends(get_db)
):
    # Define path where evidence_table is stored
    path = os.path.join("review_results", f"{project_id}_evidence_table.json")
    
    if not os.path.exists(path):
        return HTMLResponse(content="Evidence table not found.", status_code=404)

    with open(path, "r", encoding="utf-8") as f:
        evidence_table = json.load(f)

    return templates.TemplateResponse("evidence_modal.html", {
        "request": request,
        "evidence_table": evidence_table
    })

@app.get("/view/{pdf_id}", response_class=HTMLResponse)
async def view_pdf(request: Request, pdf_id: str, db: Session = Depends(get_db)):
    summary_path = os.path.join(SUMMARY_FOLDER, f"{pdf_id}.txt")
    review_path = os.path.join(REVIEW_RESULT_FOLDER, f"{pdf_id}_screening_result.json")
    pdf = db.query(Pdf).filter_by(id=pdf_id).first()

    if not pdf:
        return {"error": "PDF not found"}

    conversation = db.query(Conversation).filter_by(pdf_id=pdf.id).order_by(Conversation.created_on.desc()).first()
    if not conversation:
        conversation = Conversation(pdf_id=pdf.id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    summary_text = "Summary not available."
    if os.path.exists(summary_path):
        async with aiofiles.open(summary_path, "r", encoding="utf-8") as f:
            summary_text = await f.read()

    screening_result = {}
    if os.path.exists(review_path):
        async with aiofiles.open(review_path, "r", encoding="utf-8") as f:
            screening_result = json.loads(await f.read())

    return templates.TemplateResponse("view.html", {
        "request": request,
        "filename": f"{pdf.id}.pdf",
        "summary_text": summary_text,
        "screening_result": screening_result,
        "display_name": pdf.name,
        "conversation_id": conversation.id,
    })

@app.get("/download_summary/{filename}")
async def download_summary(filename: str):
    path = os.path.join(SUMMARY_FOLDER, filename)
    if os.path.exists(path):
        return FileResponse(path, filename=filename, media_type='application/octet-stream')
    return {"error": "File not found"}
