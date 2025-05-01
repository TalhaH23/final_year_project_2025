import os
import fitz
import pdfplumber
import re
import pymupdf
from typing import List
from langchain_core.documents import Document
from collections import Counter
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title
from langchain.text_splitter import RecursiveCharacterTextSplitter
from unstructured.documents.elements import Title, NarrativeText, ListItem, Text, Element
from tiktoken import encoding_for_model

enc = encoding_for_model("gpt-4")
def count_tokens(text: str) -> int:
    return len(enc.encode(text))

def extract_mupdf_titles(file_path: str) -> set:
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

                    if font_size > max_font_size and len(text.split()) > 4:
                        max_font_size = font_size
                        max_font_text = text

                if bold_count > 0 and len(line_text.split()) <= 20:
                    headers.add(line_text.strip())

    if max_font_text and max_font_text not in headers:
        headers.add(max_font_text)

    return headers


def get_partitioned_elements(file_path: str) -> list[Element]:
    return partition_pdf(filename=file_path, strategy="hi_res")

def extract_titles_from_elements(elements: list[Element]) -> list[str]:
    return [el.text.strip() for el in elements if isinstance(el, Title)]

def get_intersecting_titles(file_path: str, elements: list[Element]) -> list[str]:
    fitz_titles = extract_mupdf_titles(file_path)
    unstructured_titles = extract_titles_from_elements(elements)
    return [title for title in unstructured_titles if title in fitz_titles]

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

def extract_main_title(file_path: str) -> str:
    doc = pymupdf.open(file_path)
    main_title = doc.metadata.get("title", "Untitled Document")
    doc.close()
    return main_title


def chunk_document_by_titles(file_path, chunk_size: int, chunk_overlap: int) -> List[Document]:
    MIN_TOKEN_THRESHOLD = 500

    # doc = fitz.open(file_path)
    # full_text = ""
    # for page in doc:
    #     full_text += page.get_text()
    
    # full_text = ""
    # with pdfplumber.open(file_path) as pdf:
    #     for page in pdf.pages:
    #         page_text = page.extract_text()
    #         if page_text:
    #             full_text += page_text + "\n"
    
    elements = get_partitioned_elements(file_path)
    titles = get_intersecting_titles(file_path, elements)
    full_text = extract_cleaned_text(elements)

    title_positions = get_title_positions_by_lines(full_text, titles)
    title_positions.sort(key=lambda x: x[1])
    
    main_title = extract_main_title(file_path)

    splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(model_name="gpt-4", chunk_size=chunk_size, chunk_overlap=chunk_overlap)
    all_chunks = []

    for i, (title, start_idx) in enumerate(title_positions):
        # context = full_text[start_idx:start_idx+100]
        # print(f"\n Title: {title}")
        # print(f"Position: {start_idx}")
        # print(f"Context snippet: {context}")
        end_idx = title_positions[i + 1][1] if i + 1 < len(title_positions) else len(full_text)
        section_text = full_text[start_idx:end_idx].strip()

        if section_text:
            if count_tokens(section_text) < MIN_TOKEN_THRESHOLD:  # characters, or later token count
                all_chunks.append(
                    Document(
                        page_content=section_text,
                        metadata={"source": file_path, "main_title": main_title, "section_title": title}
                    )
                )
            else:
                chunks = splitter.split_text(section_text)
                for chunk in chunks:
                    all_chunks.append(
                        Document(
                            page_content=chunk,
                            metadata={"source": file_path, "main_title": main_title, "section_title": title}
                        )
                    )

        
        # all_chunks.append(
        #     Document(
        #         page_content=section_text,
        #         metadata={"source": file_path, "title": title}
        #     )
        # )

    return all_chunks
