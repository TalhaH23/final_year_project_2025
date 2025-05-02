import os
from langchain_community.document_loaders import PyMuPDFLoader
from unstructured.partition.pdf import partition_pdf

pdf_folder_path = "PDFs"
pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

# docs = PyMuPDFLoader(pdf_files[0]).load()

elements = partition_pdf(filename=pdf_files[0], strategy="hi_res")

for i, el in enumerate(elements):
    if el.category == "Table":
        print(f"Table {i}:")
        print(el.text.strip())
    if el.category == "ListItem":
        print(f"ListItem {i}:")
        print(el.text.strip())
    if el.category == "Heading":
        print(f"Heading {i}:")
        print(el.text.strip())
    if el.category == "Title":
        print(f"Title {i}:")
        print(el.text.strip())
    # print(f"Category: {el.category}")
    # print(f"Text: {el.text.strip()[:300]}")
