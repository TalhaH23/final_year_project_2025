import os
from app.pdf_summary import process_single_pdf

pdf_folder_path = "uploads"
pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

summary_text = process_single_pdf(pdf_files[0])

# # Print the summary result
# print("\n Generated Summary:\n")
# print(summary_text)