import os
from dotenv import load_dotenv
from concurrent.futures import ThreadPoolExecutor
from langchain_openai import ChatOpenAI
from langchain_community.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.prompts import PromptTemplate
from langchain_core.runnables import RunnableLambda
from vector_stores.pinecone import vector_store 

# Load env
load_dotenv()

# Set up paths
pdf_folder_path = "PDFs"
pdf_files = [os.path.join(pdf_folder_path, f) for f in os.listdir(pdf_folder_path) if f.endswith('.pdf')]

# LLM setup
llm = ChatOpenAI(model="gpt-4-turbo", temperature=0, max_tokens=2000)

# Text splitter
text_splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)

# Prompts
map_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Summarize the following document section:

    {text}

    SUMMARY:
    """
)

reduce_prompt = PromptTemplate(
    input_variables=["text"],
    template="""
    Given these partial summaries, write a final comprehensive summary:

    {text}

    FINAL SUMMARY:
    """
)

# Map-Reduce Chain
map_chain = map_prompt | llm
reduce_chain = reduce_prompt | llm
map_reduce_chain = (
    RunnableLambda(lambda docs: [map_chain.invoke({"text": doc.page_content}).content for doc in docs])
    | RunnableLambda(lambda summaries: {"text": "\n\n".join(summaries)})
    | reduce_chain
)


# Process function
def process_single_pdf(file_path):
    print(f"\n--- Processing: {os.path.basename(file_path)} ---")

    try:
        docs = PyMuPDFLoader(file_path).load()
        chunked_docs = text_splitter.split_documents(docs)
        print(f"Chunked into {len(chunked_docs)} sections")

        # Optional: Add to vector store
        vector_store.add_documents(chunked_docs)

        # Summarize
        summary = map_reduce_chain.invoke(chunked_docs).content # Map-Reduce
        # summary = refine_summary(chunked_docs)  # Refine summary
        print(f"\nFinal Summary for {os.path.basename(file_path)}:\n{summary}\n")

        # Save summary to file
        output_filename = os.path.splitext(os.path.basename(file_path))[0] + "_summary.txt"
        output_path = os.path.join("summaries", output_filename)
        os.makedirs("summaries", exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(summary)

    except Exception as e:
        print(f"Error while processing {file_path}: {str(e)}")

# Parallel processing
with ThreadPoolExecutor(max_workers=4) as executor:
    executor.map(process_single_pdf, pdf_files)
