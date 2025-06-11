import os
import sys
import asyncio
from concurrent.futures import ThreadPoolExecutor
from app.evidence_table import create_evidence_table
from app.title_extraction import extract_data_from_pdf, clean_text, get_partitioned_elements

def extract_and_process_pdf(file_path):
    """
    Extracts data from a PDF file and processes it to create an evidence table.
    """
    elements = get_partitioned_elements(file_path)
    full_text = extract_data_from_pdf(elements)
    cleaned_text = clean_text(full_text)
    
    # Here you would typically call your LLM or processing function
    # For demonstration, we just print the cleaned text
    
    print(f"Processed {file_path}:\n{cleaned_text}\n")

pdf_folder_path = "app/PDFs"
pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

# def run_async():
#     asyncio.run(process_pdfs(pdf_files))

# def run_sync():
#     with ThreadPoolExecutor(max_workers=4) as executor:
#         executor.map(process_pdfs, pdf_files)

if __name__ == "__main__":
    # mode = sys.argv[1] if len(sys.argv) > 1 else "async" 

    # if mode == "async":
    #     print("Running async processing...")
    #     run_async()
    # elif mode == "sync":
    #     print("Running sync/threaded processing...")
    #     run_sync()
    # else:
    #     print(f"Unknown mode: {mode}. Use 'async' or 'sync'.")
    create_evidence_table(pdf_files[0])

    
