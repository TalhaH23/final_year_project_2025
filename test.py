import os
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.schema import Document
import re

# Set up paths
pdf_folder_path = "PDFs"
pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

docs = PyMuPDFLoader(pdf_files[0]).load()

text_splitter = RecursiveCharacterTextSplitter(
    # Set a really small chunk size, just to show.
    chunk_size = 256,
    chunk_overlap  = 20
)

# print(docs[0].page_content)

def smart_section_chunker(file_path):
    """Improved section-based chunker for general PDFs. Avoids tables, references, etc."""

    # Step 1: Load text
    raw_docs = PyMuPDFLoader(file_path).load()
    # print(raw_docs[0].page_content)  # Print first 100 characters of the first document
    full_text = "\n".join(doc.page_content for doc in raw_docs)

    # Step 2: Split into lines
    lines = full_text.split("\n")
    # print(lines[:10])  # Print first 10 lines for debugging

    # Step 3: Identify headings
    section_indices = []
    for i, line in enumerate(lines):
        stripped = line.strip()

        if len(stripped) < 5 or len(stripped.split()) > 10:
            continue  # too short or too long to be a heading

        # Heading-like patterns
        if re.match(r"^\d{1,2}(\.\d{1,2})?\.?\s+[A-Z]", stripped) or re.match(r"^[A-Z][A-Za-z\s]{3,60}$", stripped):
            section_indices.append(i)
            # print(f"Heading found: {stripped} at line {i}")

    # Step 4: Chunk based on headings
    chunks = []
    for i in range(len(section_indices)):
        start = section_indices[i]
        end = section_indices[i + 1] if i + 1 < len(section_indices) else len(lines)
        section_lines = lines[start:end]
        section_text = "\n".join(section_lines).strip()

        if len(section_text.split()) > 30:  # skip accidental junk "headings"
            heading = lines[start].strip()
            chunks.append(Document(page_content=section_text, metadata={"section": heading}))

    return chunks

chunked_docs = smart_section_chunker(pdf_files[0])

print(f"Chunked into {len(chunked_docs)} sections")
for doc in chunked_docs:
    print('\n')
    print(f"Section: {doc.metadata['section']}")
    print('\n')
    print(doc.page_content[:100] + "...")  # Print first 100 characters of each section