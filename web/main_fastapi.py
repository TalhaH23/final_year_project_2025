import os
import aiofiles
import asyncio
import time
from typing import List

from fastapi import FastAPI, UploadFile, File, Request, Depends
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, RedirectResponse

from app.chunking_test import generate_summary
from app.celery.tasks.embeddings import process_embeddings
from app.title_extraction import chunk_document_by_titles
from app.celery.tasks import embeddings
from web.routes.conversation_messages import router as conversation_router

from web.db import get_db
from web.db.models.pdf import Pdf
from web.db.models.conversation import Conversation
from sqlalchemy.orm import Session
import uuid


# ---- Config ----
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
SUMMARY_FOLDER = os.path.join(os.getcwd(), 'summaries')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

# ---- App Setup ----
app = FastAPI()
app.include_router(conversation_router)
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates") 

# ---- Routes ----

@app.get("/debug-routes")
def debug_routes():
    return [route.name for route in app.routes]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request, db: Session = Depends(get_db)):
    pdfs = db.query(Pdf).all()
    return templates.TemplateResponse("home.html", {"request": request, "pdfs": pdfs})

@app.get("/upload", response_class=HTMLResponse, name="upload")
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse, name="handle_upload")
async def handle_upload(
    request: Request,
    pdfs: List[UploadFile] = File(...),
    db: Session = Depends(get_db),
):
    start_time = time.perf_counter()
    print(f"Upload started at {start_time:.2f}s")
    tasks = []

    for pdf in pdfs:
        if pdf.filename.endswith('.pdf'):
            pdf_id = str(uuid.uuid4())
            file_path = os.path.join(UPLOAD_FOLDER, f"{pdf_id}.pdf")

            async with aiofiles.open(file_path, 'wb') as out_file:
                content = await pdf.read()
                await out_file.write(content)

            new_pdf = Pdf(id=pdf_id, name=pdf.filename)
            db.add(new_pdf)
            
            print(f"Uploaded {pdf.filename} to {file_path}")
            chunked_docs = chunk_document_by_titles(file_path, chunk_size=500, chunk_overlap=50)
            print(f"Chunked {pdf.filename} into {len(chunked_docs)} sections")

            serialized_docs = [
                {
                    "page_content": doc.page_content,
                    "metadata": {**doc.metadata, "pdf_id": pdf_id}
                }
                for doc in chunked_docs
            ]
            
            print(f"Creating embeddings for {pdf.filename}")
            try:
                print(process_embeddings.app.conf.broker_url)
                process_embeddings.delay(pdf_id, serialized_docs)
            except Exception as e:
                print(f"Failed to queue embedding task: {e}")

            tasks.append(generate_summary(pdf_id, SUMMARY_FOLDER, chunked_docs))
            print(f"Summary generation task for {pdf.filename} added to queue")

    db.commit()
    await asyncio.gather(*tasks)

    end_time = time.perf_counter()
    print(f"Upload completed at {end_time:.2f}s, took {end_time - start_time:.2f}s")
    return RedirectResponse("/", status_code=303)

@app.get("/view/{pdf_id}", response_class=HTMLResponse)
async def view_pdf(request: Request, pdf_id: str, db: Session = Depends(get_db)):
    summary_path = os.path.join(SUMMARY_FOLDER, f"{pdf_id}.txt")
    pdf = db.query(Pdf).filter_by(id=pdf_id).first()

    if not pdf:
        return {"error": "PDF not found"}

    conversation = (
        db.query(Conversation)
        .filter_by(pdf_id=pdf.id)
        .order_by(Conversation.created_on.desc())
        .first()
    )
    if not conversation:
        conversation = Conversation(pdf_id=pdf.id)
        db.add(conversation)
        db.commit()
        db.refresh(conversation)

    if os.path.exists(summary_path):
        async with aiofiles.open(summary_path, "r", encoding="utf-8") as f:
            summary_text = await f.read()
    else:
        summary_text = "Summary not available."

    return templates.TemplateResponse("view.html", {
        "request": request,
        "filename": f"{pdf.id}.pdf",
        "summary_text": summary_text,
        "display_name": pdf.name,
        "conversation_id": conversation.id,
    })

@app.get("/download_summary/{filename}")
async def download_summary(filename: str):
    summary_path = os.path.join(SUMMARY_FOLDER, filename)
    if os.path.exists(summary_path):
        return FileResponse(summary_path, filename=filename, media_type='application/octet-stream')
    return {"error": "File not found"}
