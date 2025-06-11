import os
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.systematic_review import process_pdfs


pdf_folder_path = "app/PDFs"
pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

def run_async():
    asyncio.run(process_pdfs(pdf_files))

def run_sync():
    with ThreadPoolExecutor(max_workers=4) as executor:
        executor.map(process_pdfs, pdf_files)

if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else "async" 

    if mode == "async":
        print("Running async processing...")
        run_async()
    elif mode == "sync":
        print("Running sync/threaded processing...")
        run_sync()
    else:
        print(f"Unknown mode: {mode}. Use 'async' or 'sync'.")
