import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from app.vector_stores.pinecone import vector_store 
from app.title_extraction import section_headers, chunk_document_by_titles

load_dotenv()

# pdf_folder_path = "PDFs"
# pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, max_tokens=4000)

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

If the section is irrelevant or empty, do not include it in the summary.

---
Special Instructions:
- Ignore references, footnotes, citations, and in-text citation numbers (e.g., "[1]", "(2020)").
- Focus only on the main ideas and key points.
- If the section is data or a table, provide a structured short table-style summary.
- Ignore Open Access, License, or Correspondence labels unless they contain real academic content.

---
Return your answer in HTML format like this:

<h2>{title}</h2>
<ul>
  <li>Bullet point 1</li>
  <li>Bullet point 2</li>
  <li>Bullet point 3</li>
</ul>

IMPORTANT: Do not include Markdown code fences like ```html. Return only the pure HTML content.

Here is the section content to summarize:

{text}
"""
)

reduce_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
You are given multiple HTML-formatted summaries.  
Organize them into a nested HTML bullet list, grouped based on section titles::

- At the top, display the main document title inside `<h1>`.
- Use `<h2>` for main sections.
- Use `<ul><li>` for bullet points under each section.
- Nest subsections inside the appropriate parent section, if needed.
- Discard any sections that say "**No summary needed.**"
- Keep the final output fully in valid HTML format, ready to be rendered in a webpage.
- IMPORTANT: Do not include Markdown code fences like ```html. Return only the pure HTML content.

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
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n--- Processing: {os.path.basename(file_path)} ---")

    try:
        titles = section_headers(file_path)
        # print(f"Found {len(titles)} section headers")
        print(f"Extracted Titles for {base_name}")

        chunked_docs = chunk_document_by_titles(file_path, titles, chunk_size=800, chunk_overlap=100)
        # print(f"Chunked into {len(chunked_docs)} titled sections")
        print(f"Chunked Documents for {base_name}")
        
        chunk_output_path = os.path.join("app/summaries", f"{base_name}_chunks.txt")
        os.makedirs("summaries", exist_ok=True)

        with open(chunk_output_path, "w", encoding="utf-8") as f:
            for i, doc in enumerate(chunked_docs, 1):
                f.write(f"\n\n🔹 Chunk {i}: {doc.metadata.get('title', 'Untitled')}\n")
                f.write("-" * 40 + "\n")
                f.write(doc.page_content.strip())
                f.write("\n" + "=" * 60)

        vector_store.add_documents(chunked_docs)

        summary = map_reduce_chain.invoke(chunked_docs).content
        # print(f"\nFinal Summary for {os.path.basename(file_path)}:\n{summary}\n")
        print(f"Generated Summary for {base_name}")

        summary_output_path = os.path.join("app/summaries", f"{base_name}_summary.txt")
        with open(summary_output_path, "w", encoding="utf-8") as f:
            f.write(summary)
        return summary

    except Exception as e:
        print(f"Error while processing {file_path}: {str(e)}")

# Parallel processing
# with ThreadPoolExecutor(max_workers=4) as executor:
#     executor.map(process_single_pdf, pdf_files)

# process_single_pdf(pdf_files[1])
