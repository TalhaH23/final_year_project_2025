import os
import fitz
import pdfplumber
import re
import pymupdf
import difflib
from typing import List, Tuple
from langchain_core.documents import Document
from collections import Counter
from unstructured.partition.pdf import partition_pdf
from langchain_community.document_loaders import PyMuPDFLoader
from unstructured.documents.elements import Title
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.documents.elements import Title, NarrativeText, ListItem, Text, Element, Table, Header
from tiktoken import encoding_for_model
import json

enc = encoding_for_model("gpt-4")
def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def extract_main_title(doc: open) -> str:
    # doc = fitz.open(file_path)
    first_page = doc[0]
    blocks = first_page.get_text("dict")["blocks"]

    title_spans = []

    # Gather all spans with position and size
    for block in blocks:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                size = span["size"]
                y = span["bbox"][1]  # vertical position (top)
                if len(text) > 5 and not text.lower().startswith("doi"):
                    title_spans.append({"text": text, "size": size, "y": y})

    if not title_spans:
        return "Untitled Document"

    # Sort by font size descending, then vertical position (y)
    title_spans.sort(key=lambda x: (-x["size"], x["y"]))

    # Take top 3 spans within the largest font size range (e.g., 1 pt tolerance)
    top_size = title_spans[0]["size"]
    candidate_lines = [span for span in title_spans if abs(span["size"] - top_size) < 1.0]

    # Sort those lines top-to-bottom (based on y position)
    candidate_lines.sort(key=lambda x: x["y"])

    # Join text lines to form the full title
    full_title = " ".join(span["text"] for span in candidate_lines)
    return full_title.strip()

def extract_mupdf_titles(file_path: str) -> Tuple[set, str]:
    doc = fitz.open(file_path)
    main_title = extract_main_title(doc)
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

                    if font_size > max_font_size and len(text.split()) > 4:
                        max_font_size = font_size
                        max_font_text = text

                if bold_count > 0 and len(line_text.split()) <= 20:
                    headers.add(line_text.strip())

    if max_font_text and max_font_text not in headers:
        headers.add(max_font_text)

    return headers, main_title

# def extract_mupdf_titles(file_path: str) -> set:
#     doc = fitz.open(file_path)
#     headers = set()

#     for page in doc:
#         for block in page.get_text("dict")["blocks"]:
#             for line in block.get("lines", []):
#                 line_text = ""
#                 font_sizes = []
#                 bold_flags = []

#                 for span in line.get("spans", []):
#                     text = span.get("text", "").strip()
#                     if not text or len(text) > 100:
#                         continue

#                     font_size = round(span.get("size", 0), 1)
#                     font_name = span.get("font", "")
#                     is_bold = "Bold" in font_name or ".B" in font_name

#                     line_text += " " + text
#                     font_sizes.append(font_size)
#                     bold_flags.append(is_bold)

#                 line_text = line_text.strip()
#                 if not line_text or len(line_text.split()) > 20:
#                     continue

#                 avg_font_size = sum(font_sizes) / len(font_sizes) if font_sizes else 0
#                 bold_ratio = sum(bold_flags) / len(bold_flags) if bold_flags else 0

#                 # Heuristic: large or mostly bold lines are likely headers
#                 if avg_font_size >= 10 and (bold_ratio >= 0.5 or avg_font_size > 11):
#                     headers.add(line_text)

#     doc.close()
#     return headers


def get_partitioned_elements(file_path: str) -> list[Element]:
    return partition_pdf(filename=file_path, strategy="hi_res")

def extract_titles_from_elements(elements: list[Element]) -> list[str]:
    return [el.text.strip() for el in elements if isinstance(el, Title)]

def get_intersecting_titles(file_path: str, elements: list[Element]) -> Tuple[list[str], str]:
    fitz_titles, main_title = extract_mupdf_titles(file_path)
    unstructured_titles = extract_titles_from_elements(elements)
    
    # print("\n🧩 Titles from `unstructured`:")
    # for t in unstructured_titles:
    #     print(f" - {t}")

    # print("\n🔡 Titles from `fitz`:")
    # for t in fitz_titles:
    #     print(f" - {t}")
    return [title for title in unstructured_titles if title in fitz_titles], main_title

def clean_text(text: str) -> str:
    match = re.search(r"(Bibliography|References|Acknowledgements|Funding|Abbreviations)", text, re.IGNORECASE)
    return text[:match.start()] if match else text

def extract_cleaned_text(elements: list[Element]) -> str:
    raw_text = "\n".join(
        el.text.strip()
        for el in elements
        if el.text and isinstance(el, (Title, NarrativeText, ListItem, Text))
    )
    return clean_text(raw_text.strip())

def extract_data_from_pdf(elements: list[Element]) -> str:
    raw_text = "\n".join(
        el.text.strip()
        for el in elements
        if el.text and isinstance(el, (Title, Header, Table, ListItem))
    )
    return clean_text(raw_text.strip())

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
        running_idx += len(line) + 1

    return positions


# def get_title_positions_by_lines(full_text: str, titles: List[str]) -> List[tuple]:
#     positions = []
#     lines = full_text.splitlines()
#     full_text_lower = full_text.lower()

#     for title in titles:
#         # Try to locate the best match in the full text using fuzzy matching
#         match = difflib.get_close_matches(title.lower(), lines, n=1, cutoff=0.8)
#         if match:
#             matched_line = match[0]
#             position = full_text_lower.find(matched_line.lower())
#             if position != -1:
#                 positions.append((title, position))
#             else:
#                 print(f"⚠️ Match found but position not located: {matched_line}")
#         else:
#             print(f"❌ No match found for title: '{title}'")

#     return positions


# def extract_main_title(file_path: str) -> str:
#     doc = pymupdf.open(file_path)
#     metadata = doc.metadata
#     main_title = doc.metadata.get("title", "Untitled Document")
#     doc.close()
    
#     print("PDF Metadata:")
#     for key, value in metadata.items():
#         print(f"{key}: {value}")    
    
#     return main_title




def chunk_document_by_titles(file_path: str, chunk_size: int, chunk_overlap: int) -> Tuple[List[Document], str]:
    MIN_TOKEN_THRESHOLD = 500
    output_dir = "app/chunks"
    os.makedirs(output_dir, exist_ok=True)

    elements = get_partitioned_elements(file_path)
    print(f"Extracted {len(elements)} elements from {file_path}")
    # for el in elements:
    #     print(f"Element type: {type(el).__name__}, text: {el.text[:50]}...")
        
    titles, main_title = get_intersecting_titles(file_path, elements)
    print("main_title:", main_title)
    
    full_text = extract_cleaned_text(elements)
    
    # print(f"\n🔍 Extracted {len(titles)} titles:")
    # for t in titles:
    #     print(f" - '{t}'")
    # print(f"\n📄 Full text has {len(full_text.splitlines())} lines.")
    
    title_positions = get_title_positions_by_lines(full_text, titles)
    title_positions.sort(key=lambda x: x[1])
    print(f"📌 Found {len(title_positions)} matched title positions.")
    

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
        model_name="gpt-4",
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    all_chunks = []
    doc_id = os.path.splitext(os.path.basename(file_path))[0]
    chunk_index = 0
    
    if not title_positions:
        print("⚠️ No title positions found. Chunking full document instead.")
        chunks = splitter.split_text(full_text)
        for i, chunk_text in enumerate(chunks):
            chunk = Document(
                page_content=chunk_text,
                metadata={"source": file_path, "main_title": main_title, "section_title": "Full Document", "table": False}
            )
            all_chunks.append(chunk)
            _write_chunk_to_file(output_dir, doc_id, i, chunk)
        return all_chunks

    for i, (title, start_idx) in enumerate(title_positions):
        end_idx = title_positions[i + 1][1] if i + 1 < len(title_positions) else len(full_text)
        section_text = full_text[start_idx:end_idx].strip()

        if not section_text:
            continue

        if count_tokens(section_text) < MIN_TOKEN_THRESHOLD:
            chunk = Document(
                page_content=section_text,
                metadata={
                    "source": file_path,
                    "main_title": main_title,
                    "section_title": title,
                    "table": False
                }
            )
            all_chunks.append(chunk)
            _write_chunk_to_file(output_dir, doc_id, chunk_index, chunk)
            chunk_index += 1
        else:
            sub_chunks = splitter.split_text(section_text)
            for chunk_text in sub_chunks:
                chunk = Document(
                    page_content=chunk_text,
                    metadata={
                        "source": file_path,
                        "main_title": main_title,
                        "section_title": title,
                        "table": False
                    }
                )
                all_chunks.append(chunk)
                # _write_chunk_to_file(output_dir, doc_id, chunk_index, chunk)
                # chunk_index += 1

    return all_chunks, main_title


def _write_chunk_to_file(output_dir: str, doc_id: str, chunk_index: int, doc: Document):
    chunk_filename = os.path.join(output_dir, f"{doc_id}_chunk_{chunk_index}.json")
    with open(chunk_filename, "w", encoding="utf-8") as f:
        json.dump({
            "page_content": doc.page_content,
            "metadata": doc.metadata
        }, f, indent=2, ensure_ascii=False)

def chunk_docs(file_path: str, chunk_size: int, chunk_overlap: int) -> List[Document]:
    docs = PyMuPDFLoader(file_path).load()

    full_text = "\n".join([doc.page_content for doc in docs])
    cleaned_text = clean_text(full_text)
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
    )
    chunks = text_splitter.split_text(cleaned_text)
    
    return [Document(page_content=chunk, metadata={"table": True}) for chunk in chunks]
