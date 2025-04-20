import os
import fitz
from collections import Counter
from unstructured.partition.pdf import partition_pdf
from unstructured.documents.elements import Title

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
file_path = pdf_files[0]


def section_headers(file_path):
    mupdf_titles = extract_mupdf_titles(file_path)
    unstructured_titles = extract_unstructured_titles(file_path)

    # Intersect: retain unstructured order
    intersecting_titles = [title for title in unstructured_titles if title in mupdf_titles]

    print("✅ Titles detected by BOTH methods (in document order):\n")
    for title in intersecting_titles:
        print(f"📌 {title}")
        
    return intersecting_titles

section_headers(file_path)

