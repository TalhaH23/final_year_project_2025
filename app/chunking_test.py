import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from vector_stores.pinecone import vector_store 
from title_extraction import section_headers, chunk_document_by_titles

load_dotenv()

pdf_folder_path = "PDFs"
pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, max_tokens=2000)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

map_prompt = PromptTemplate(
    input_variables=["title", "text"],
    template="""
You are a professional summarizer. Your task is to summarize the following document section **only if it contains meaningful academic or scientific content** (such as Introduction, Methods, Results, Discussion, Review Typologies, etc.).

---
Summarize if:
- The section has actual scientific discussion, research methodology, results, arguments, or structured narrative text.
- The section includes tables or data — in that case, summarize the table into a compact text or table format.

---
Do NOT summarize if:
- The section is administrative, legal, or irrelevant (e.g., "Correspondence", "Author Contributions", "Open Access", "License", "Competing Interests", "Ethics Approval").
- The section is extremely short (less than a few meaningful sentences).
- The section only repeats boilerplate text.

If the section is irrelevant or empty, respond exactly with:

**No summary needed.**

---
Special Instructions:
- Ignore references, footnotes, citations, and in-text citation numbers (e.g., "[1]", "(2020)").
- Focus only on the main ideas and key points.
- If the section is data or a table, provide a structured short table-style summary.
- Ignore Open Access, License, or Correspondence labels unless they contain real academic content.

---

Return your answer in the following format:
Section Title: {title}

Summary:
- Bullet point 1
- Bullet point 2
- Bullet point 3
- ...

Here is the section content to summarize:

{text}
"""
)

reduce_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are given structured summaries. Organize them into a nested bullet list:
- Have the main document title at the top. This is the title of the entire document. This is not always in the section titles. Please deduce the main title from the text.
- Main sections at top level.
- Use the Section Titles provided, and ensure the summaries are concise.
- Some Section Titles may be empty or irrelevant; ignore them.
- Subsections nested under them when appropriate.
- Use bullet points for clarity.
- Avoid excessive detail, but ensure the main ideas are captured.
- If the section is a table or data, summarize it in a table format.

Here are the summaries:

{text}
"""
)

map_chain = map_prompt | llm
reduce_chain = reduce_prompt | llm
map_reduce_chain = (
    RunnableLambda(
        lambda docs: [
            map_chain.invoke({
                "title": doc.metadata.get("title", "Untitled"),
                "text": doc.page_content
            }).content
            for doc in docs
        ]
    )
    | RunnableLambda(lambda summaries: {"text": "\n\n".join(summaries)})
    | reduce_chain
)


def process_single_pdf(file_path):
    print(f"\n--- Processing: {os.path.basename(file_path)} ---")

    try:
        titles = section_headers(file_path)
        print(f"Found {len(titles)} section headers")

        chunked_docs = chunk_document_by_titles(file_path, titles, chunk_size=1000, chunk_overlap=100)
        print(f"Chunked into {len(chunked_docs)} titled sections")

        base_name = os.path.splitext(os.path.basename(file_path))[0]
        chunk_output_path = os.path.join("summaries", f"{base_name}_chunks.txt")
        os.makedirs("summaries", exist_ok=True)

        with open(chunk_output_path, "w", encoding="utf-8") as f:
            for i, doc in enumerate(chunked_docs, 1):
                f.write(f"\n\n🔹 Chunk {i}: {doc.metadata.get('title', 'Untitled')}\n")
                f.write("-" * 40 + "\n")
                f.write(doc.page_content.strip())
                f.write("\n" + "=" * 60)

        vector_store.add_documents(chunked_docs)

        summary = map_reduce_chain.invoke(chunked_docs).content
        print(f"\nFinal Summary for {os.path.basename(file_path)}:\n{summary}\n")

        summary_output_path = os.path.join("summaries", f"{base_name}_summary.txt")
        with open(summary_output_path, "w", encoding="utf-8") as f:
            f.write(summary)

    except Exception as e:
        print(f"Error while processing {file_path}: {str(e)}")

# Parallel processing
# with ThreadPoolExecutor(max_workers=4) as executor:
#     executor.map(process_single_pdf, pdf_files)

process_single_pdf(pdf_files[1])
