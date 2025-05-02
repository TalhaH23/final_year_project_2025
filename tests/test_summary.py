import os
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.chunking_test import process_pdfs
from app.pdf_summary import process_single_pdf

# pdf_folder_path = "uploads"
# pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

# def run_async():
#     asyncio.run(process_pdfs(pdf_files))

# def run_sync():
#     with ThreadPoolExecutor(max_workers=4) as executor:
#         executor.map(process_single_pdf, pdf_files)

# if __name__ == "__main__":
#     mode = sys.argv[1] if len(sys.argv) > 1 else "async"  # default is async

#     if mode == "async":
#         print("Running async processing...")
#         run_async()
#     elif mode == "sync":
#         print("Running sync/threaded processing...")
#         run_sync()
#     else:
#         print(f"Unknown mode: {mode}. Use 'async' or 'sync'.")

from celery import Celery
app = Celery(broker="redis://127.0.0.1:6379/0")
print(app.control.ping())
