import fitz
import os
from collections import Counter

def int_to_rgb(color_int):
    r = (color_int >> 16) & 255
    g = (color_int >> 8) & 255
    b = color_int & 255
    return (r, g, b)

def is_bold(font_name):
    return "Bold" in font_name or "bold" in font_name or "BD" in font_name or font_name.endswith(".B")

pdf_folder_path = "PDFs"
pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]
doc = fitz.open(pdf_files[0])

font_sizes = set()
for page in doc:
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                font_sizes.add(round(span["size"], 1))

common_size = Counter(font_sizes).most_common(1)[0][0]
print(f"📐 Most common font size (body text): {common_size}")

top_5_largest = sorted(font_sizes, reverse=True)[:5]

print("🔠 Top 5 Largest Font Sizes in the Document:")
for size in top_5_largest:
    print(f"• {size}")

print("\n📄 Detected Headers:\n")

for page_num, page in enumerate(doc, start=1):
    for block in page.get_text("dict")["blocks"]:
        for line in block.get("lines", []):
            for span in line.get("spans", []):
                text = span["text"].strip()
                if not text:
                    continue
                
                font = span.get("font", "")
                size = round(span["size"], 1)
                color = int_to_rgb(span["color"])
                bold = is_bold(font)

                word_count = len(text.split())

                if text.lower().strip() == "diagnostic test accuracy reviews":
                    print(f"\n🔎 Found Target Text on Page {page_num}: '{text}'")
                    print(f" ↳ size: {size}, font: {font}, bold: {bold}, color: {color}")
                    
                # 🎯 Detect section headers
                if bold and size > common_size and word_count <= 10:
                    print(f"[Page {page_num}] 📘 SECTION: {text}")
                    print(f" ↳ size: {size}, font: {font}, color: {color}")

                # 🎯 Detect subsection headers
                elif bold and size == common_size and word_count <= 12:
                    if len(text) < 80:  # limit based on character length too
                        print(f"[Page {page_num}] 📗 SUBSECTION: {text}")
                        print(f" ↳ size: {size}, font: {font}, color: {color}")
