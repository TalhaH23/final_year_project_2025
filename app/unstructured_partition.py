import os
from langchain_community.document_loaders import PyMuPDFLoader
from unstructured.partition.pdf import partition_pdf

pdf_folder_path = "PDFs"
pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

# docs = PyMuPDFLoader(pdf_files[0]).load()

# Process the PDF
elements = partition_pdf(filename=pdf_files[1], strategy="hi_res")

# Print each element's category and preview of its text
for i, el in enumerate(elements):
    if el.category == "Title":
        print(f"\n--- Element #{i + 1} ---")
        print(f"Title: {el.text.strip()}")
    # print(f"Category: {el.category}")
    # print(f"Text: {el.text.strip()[:300]}")  # Print only first 300 characters for brevity
