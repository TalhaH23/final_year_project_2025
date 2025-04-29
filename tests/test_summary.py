import os
from app.chunking_test import process_single_pdf
import pymupdf

pdf_folder_path = "uploads"
pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

summary_text = process_single_pdf(pdf_files[0])

# doc = pymupdf.open(pdf_files[0])
# metadata = doc.metadata

# print(metadata)

# # Print the summary result
# print("\n Generated Summary:\n")
# print(summary_text)

### python3 -m tests.test_summary.py