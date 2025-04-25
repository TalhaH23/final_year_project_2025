import os
import fitz
import pdfplumber
from typing import List
from langchain_core.documents import Document
from collections import Counter
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title
from langchain.text_splitter import RecursiveCharacterTextSplitter

# ---------- Step 1: PyMuPDF Header Extraction ----------
def extract_mupdf_titles(file_path):
    doc = fitz.open(file_path)
    font_sizes = []

    headers = set()
    max_font_text = ""
    max_font_size = 0

    for page in doc:
        for block in page.get_text("dict")["blocks"]:
            for line in block.get("lines", []):
                line_text = ""
                max_size_in_line = 0
                bold_count = 0

                for span in line.get("spans", []):
                    text = span["text"].strip()
                    if not text or len(text) > 100:
                        continue

                    font_size = round(span["size"], 1)
                    font = span.get("font", "")
                    is_bold = "Bold" in font or font.endswith(".B")

                    if is_bold:
                        bold_count += 1
                        line_text += " " + text
                        if font_size > max_size_in_line:
                            max_size_in_line = font_size

                    # Track largest font in the document
                    if font_size > max_font_size and len(text.split()) > 4:
                        max_font_size = font_size
                        max_font_text = text

                # If most of the line is bolded and reasonably short, consider it a header
                if bold_count > 0 and len(line_text.split()) <= 20:
                    headers.add(line_text.strip())

    # Also include the largest font text if it looks like a title
    if max_font_text and max_font_text not in headers:
        headers.add(max_font_text)

    return headers


# ---------- Step 2: Unstructured Title Extraction ----------
def extract_unstructured_titles(file_path):
    elements = partition_pdf(filename=file_path, strategy="hi_res")
    titles = []
    for el in elements:
        if isinstance(el, Title):
            titles.append(el.text.strip())
    return titles  # Keep order

# ---------- Step 3: Compare and Print ----------
pdf_folder = "PDFs"
pdf_files = [os.path.join(pdf_folder, f) for f in os.listdir(pdf_folder) if f.endswith(".pdf")]
file_path = pdf_files[1]


def section_headers(file_path):
    mupdf_titles = extract_mupdf_titles(file_path)
    unstructured_titles = extract_unstructured_titles(file_path)

    # Intersect: retain unstructured order
    intersecting_titles = [title for title in unstructured_titles if title in mupdf_titles]

    print("✅ Titles detected by BOTH methods (in document order):\n")
    for title in intersecting_titles:
        print(f"📌 {title}")
        
    return intersecting_titles

# section_headers(file_path)

def get_title_positions_by_lines(full_text: str, titles: List[str]) -> List[tuple]:
    positions = []
    lower_text = full_text.lower()
    lines = full_text.splitlines()

    running_idx = 0
    for line in lines:
        line_clean = line.strip()
        for title in titles:
            if line_clean == title:
                pos = lower_text.find(line_clean.lower(), running_idx)
                if pos != -1:
                    positions.append((title, pos))
        running_idx += len(line) + 1  # +1 for newline

    return positions

titles = section_headers(file_path)

def chunk_document_by_titles(file_path, titles: List[str], chunk_size=800, chunk_overlap=100) -> List[Document]:

    # doc = fitz.open(file_path)
    # full_text = ""
    # for page in doc:
    #     full_text += page.get_text()
    
    full_text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            page_text = page.extract_text()
            if page_text:
                full_text += page_text + "\n"

    full_text = full_text.strip()
    
    print(full_text[:5000])  # Print first 1000 characters for debugging

    title_positions = get_title_positions_by_lines(full_text, titles)

    title_positions.sort(key=lambda x: x[1])
    print("Title positions:", title_positions)

    splitter = RecursiveCharacterTextSplitter(chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []

    for i, (title, start_idx) in enumerate(title_positions):
        context = full_text[start_idx:start_idx+100]  # show 100 chars from the found position
        print(f"\n🟡 Title: {title}")
        print(f"Position: {start_idx}")
        print(f"Context snippet: {context}")
        end_idx = title_positions[i + 1][1] if i + 1 < len(title_positions) else len(full_text)
        section_text = full_text[start_idx:end_idx].strip()

        # if section_text:
        #     chunks = splitter.split_text(section_text)
        #     for chunk in chunks:
        #         all_chunks.append(
        #             Document(
        #                 page_content=chunk,
        #                 metadata={"source": file_path, "title": title}
        #             )
        #         )
        
        all_chunks.append(
            Document(
                page_content=section_text,
                metadata={"source": file_path, "title": title}
            )
        )

    return all_chunks

chunk_document_by_titles(file_path,titles)