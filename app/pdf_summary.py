import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
import pymupdf
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

map_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0, max_tokens=4000)
reduce_llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, max_tokens=4000)

text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

map_prompt = PromptTemplate(
    input_variables=["section_title", "text"],
    template="""
You are a professional academic summarizer.

Summarize the following document section ONLY if it contains meaningful academic or scientific content.

Summarization rules:
- Limit to 3-6 bullet points maximum, no matter how long the section is.
- Focus ONLY on the main ideas and key takeaways.
- Do NOT list specific study examples, minor variations, or detailed methodologies unless crucial.
- Prioritize general principles, definitions, findings, or conclusions.
- Ignore references, citations, and numeric citation markers (e.g., [1], (2020)).

Formatting rules:
- Output ONLY raw HTML tags and content.
- DO NOT use Markdown, DO NOT use triple backticks (```), DO NOT wrap content in code fences.
- Your output must start directly with a <h2> tag.
- Then use a <ul> list with 3-6 <li> bullet points under each heading.

Here is the section title and content:

Section title: {section_title}

Content:

{text}
"""
)


reduce_prompt = PromptTemplate(
    input_variables=["text", "main_title"],
    template="""
You are a document organizer. You are given multiple HTML-formatted summaries of a document's sections.

Instructions:

- Group the summaries based on section titles.
- Maintain the original order of sections based on their appearance.
- Do not alter, merge, or invent new section titles.
- Only output raw HTML tags and content.
- DO NOT use Markdown, DO NOT use triple backticks (```), DO NOT wrap content in code fences.
- Display the main document title {main_title} at the top inside a <h1> tag.
- Use <h2> for each section title.
- Under each <h2> heading, use a <ul> with <li> bullet points.
- Output should be fully valid HTML, ready to render directly in a web page.
- No additional commentary, explanations, or wrapping in markdown.

Output structure:

<h1>{main_title}</h1>

<h2>First Section Title</h2>
<ul>
  <li>Point 1</li>
  <li>Point 2</li>
</ul>

<h2>Second Section Title</h2>
<ul>
  <li>Point 1</li>
  <li>Point 2</li>
</ul>

Summaries:

{text}
"""
)



map_chain = map_prompt | map_llm
reduce_chain = reduce_prompt | reduce_llm

map_reduce_chain = (
    RunnableLambda(
        lambda docs: {
            "summaries": [
                map_chain.invoke({
                    "section_title": doc.metadata.get("section_title", "Untitled Section"),
                    "text": doc.page_content
                }).content
                for doc in docs
            ],
            "main_title": docs[0].metadata.get("main_title", "Untitled Document")
        }
    )
    | RunnableLambda(lambda inputs: {
        "text": "\n\n".join(inputs["summaries"]),
        "main_title": inputs["main_title"]
    })
    | reduce_chain
)


def process_single_pdf(file_path):
    base_name = os.path.splitext(os.path.basename(file_path))[0]
    print(f"\n--- Processing: {os.path.basename(file_path)} ---")

    try:
        titles = section_headers(file_path)
        # print(f"Found {len(titles)} section headers")
        print(f"Extracted Titles for {base_name}")

        chunked_docs = chunk_document_by_titles(file_path, titles, chunk_size=500, chunk_overlap=50)
        # print(f"Chunked into {len(chunked_docs)} titled sections")
        print(f"Chunked Documents for {base_name}")
        
        chunk_output_path = os.path.join("app/summaries", f"{base_name}_chunks.txt")
        os.makedirs("summaries", exist_ok=True)

        with open(chunk_output_path, "w", encoding="utf-8") as f:
            for i, doc in enumerate(chunked_docs, 1):
                f.write(f"\n\n Chunk {i}: {doc.metadata.get('section_title', 'Untitled')}\n")
                f.write("-" * 40 + "\n")
                f.write(doc.page_content.strip())
                f.write("\n" + "=" * 60)

        # vector_store.add_documents(chunked_docs)

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
