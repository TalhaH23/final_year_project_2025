import os
import aiofiles
import asyncio
from typing import List

from fastapi import FastAPI, UploadFile, File, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import FileResponse, RedirectResponse

from app.chunking_test import process_pdfs, generate_summary



# ---- Config ----
UPLOAD_FOLDER = os.path.join(os.getcwd(), 'uploads')
SUMMARY_FOLDER = os.path.join(os.getcwd(), 'summaries')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(SUMMARY_FOLDER, exist_ok=True)

# ---- App Setup ----
app = FastAPI()
app.mount("/uploads", StaticFiles(directory=UPLOAD_FOLDER), name="uploads")
app.mount("/static", StaticFiles(directory="web/static"), name="static")
templates = Jinja2Templates(directory="web/templates") 

# ---- Routes ----

@app.get("/debug-routes")
def debug_routes():
    return [route.name for route in app.routes]

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    pdfs = os.listdir(UPLOAD_FOLDER)
    return templates.TemplateResponse("home.html", {"request": request, "pdfs": pdfs})

@app.get("/upload", response_class=HTMLResponse, name="upload")
async def upload_page(request: Request):
    return templates.TemplateResponse("upload.html", {"request": request})

@app.post("/upload", response_class=HTMLResponse, name="handle_upload")
async def handle_upload(request: Request, pdfs: List[UploadFile] = File(...)):
    filepaths = []

    for pdf in pdfs:
        filename = pdf.filename
        if filename.endswith('.pdf'):
            file_path = os.path.join(UPLOAD_FOLDER, filename)
            async with aiofiles.open(file_path, 'wb') as out_file:
                content = await pdf.read()
                await out_file.write(content)
            filepaths.append(file_path)
            
    tasks = [generate_summary(fp, SUMMARY_FOLDER) for fp in filepaths]
    await asyncio.gather(*tasks)

    # asyncio.create_task(process_pdfs(filepaths))

    return RedirectResponse("/", status_code=303)

@app.get("/view/{filename}", response_class=HTMLResponse)
async def view_pdf(request: Request, filename: str):
    base_name, _ = os.path.splitext(filename)
    summary_filename = f"{base_name}.txt"
    summary_path = os.path.join(SUMMARY_FOLDER, summary_filename)

    if os.path.exists(summary_path):
        async with aiofiles.open(summary_path, "r", encoding="utf-8") as f:
            summary_text = await f.read()
    else:
        summary_text = "Summary not available."

    return templates.TemplateResponse("view.html", {
        "request": request,
        "filename": filename,               # for iframe
        "summary_text": summary_text       # for right pane
    })

@app.get("/download_summary/{filename}")
async def download_summary(filename: str):
    summary_path = os.path.join(SUMMARY_FOLDER, filename)
    if os.path.exists(summary_path):
        return FileResponse(summary_path, filename=filename, media_type='application/octet-stream')
    return {"error": "File not found"}
