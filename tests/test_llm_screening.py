import sys
import types
from dataclasses import dataclass
import pytest
from unittest.mock import patch
import app.systematic_review as sr

# Create dummy external modules that are missing in the test environment
for name in ["fitz", "pdfplumber", "pymupdf"]:
    sys.modules.setdefault(name, types.ModuleType(name))

# Minimal unstructured package
unstructured = types.ModuleType("unstructured")
partition = types.ModuleType("unstructured.partition")
pdf_mod = types.ModuleType("unstructured.partition.pdf")

def partition_pdf(filename=None, strategy=None):
    return []

pdf_mod.partition_pdf = partition_pdf
partition.pdf = pdf_mod
unstructured.partition = partition

documents = types.ModuleType("unstructured.documents")
elements = types.ModuleType("unstructured.documents.elements")
class Title:
    def __init__(self, text=""):
        self.text = text
class NarrativeText: pass
class ListItem: pass
class Text: pass
class Element: pass
class Table: pass
class Header: pass

for cls in [Title, NarrativeText, ListItem, Text, Element, Table, Header]:
    setattr(elements, cls.__name__, cls)

documents.elements = elements
unstructured.documents = documents

sys.modules.setdefault("unstructured", unstructured)
sys.modules.setdefault("unstructured.partition", partition)
sys.modules.setdefault("unstructured.partition.pdf", pdf_mod)
sys.modules.setdefault("unstructured.documents", documents)
sys.modules.setdefault("unstructured.documents.elements", elements)

# Minimal langchain_core Document and text splitter
langchain_core = types.ModuleType("langchain_core")
langchain_documents = types.ModuleType("langchain_core.documents")
@dataclass
class Document:
    page_content: str
    metadata: dict
langchain_documents.Document = Document
langchain_core.documents = langchain_documents
sys.modules.setdefault("langchain_core", langchain_core)
sys.modules.setdefault("langchain_core.documents", langchain_documents)

splitter_mod = types.ModuleType("langchain.text_splitter")
class DummySplitter:
    def __init__(self, chunk_size=0, chunk_overlap=0):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
    @classmethod
    def from_tiktoken_encoder(cls, model_name="gpt-4", chunk_size=0, chunk_overlap=0):
        return cls(chunk_size, chunk_overlap)
    def split_text(self, text):
        return [text]
splitter_mod.RecursiveCharacterTextSplitter = DummySplitter
sys.modules.setdefault("langchain.text_splitter", splitter_mod)

# Stub for langchain_community.document_loaders
community_mod = types.ModuleType("langchain_community")
loaders_mod = types.ModuleType("langchain_community.document_loaders")
class PyMuPDFLoader:
    def __init__(self, *args, **kwargs):
        pass
    def load(self):
        return []
loaders_mod.PyMuPDFLoader = PyMuPDFLoader
community_mod.document_loaders = loaders_mod
sys.modules.setdefault("langchain_community", community_mod)
sys.modules.setdefault("langchain_community.document_loaders", loaders_mod)

# Minimal tiktoken encoder
tiktoken_mod = types.ModuleType("tiktoken")
class DummyEncoding:
    def encode(self, text):
        return text.split()

def encoding_for_model(model_name):
    return DummyEncoding()

tiktoken_mod.encoding_for_model = encoding_for_model
sys.modules.setdefault("tiktoken", tiktoken_mod)

# Fixture to set asyncio backend for anyio
@pytest.fixture
def anyio_backend():
    return "asyncio"

# Stub missing external modules
dotenv = types.ModuleType("dotenv")
def load_dotenv():
    pass
dotenv.load_dotenv = load_dotenv
sys.modules.setdefault("dotenv", dotenv)

openai_mod = types.ModuleType("langchain_openai")
class ChatOpenAI:
    def __init__(self, *args, **kwargs):
        pass
    async def ainvoke(self, *args, **kwargs):
        return types.SimpleNamespace(content="")
openai_mod.ChatOpenAI = ChatOpenAI
sys.modules.setdefault("langchain_openai", openai_mod)

prompts_mod = types.ModuleType("langchain.prompts")
class PromptTemplate:
    def __init__(self, input_variables=None, template=""):
        self.input_variables = input_variables
        self.template = template
    def __or__(self, other):
        return other
prompts_mod.PromptTemplate = PromptTemplate
sys.modules.setdefault("langchain.prompts", prompts_mod)

aiofiles_mod = types.ModuleType("aiofiles")
class DummyFile:
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        pass
    async def read(self):
        return ""
    async def write(self, data):
        pass
async def aio_open(*args, **kwargs):
    return DummyFile()
aiofiles_mod.open = aio_open
sys.modules.setdefault("aiofiles", aiofiles_mod)

pinecone_mod = types.ModuleType("app.vector_stores.pinecone")
class DummyVectorStore:
    def similarity_search_with_score(self, query, k=10):
        return []
pinecone_mod.vector_store = DummyVectorStore()
sys.modules.setdefault("app.vector_stores.pinecone", pinecone_mod)

stard_mod = types.ModuleType("app.stard_summary")
async def llm_summary(docs):
    return ""
def group_doc_by_section(sections):
    return [sections]
stard_mod.llm_summary = llm_summary
stard_mod.group_doc_by_section = group_doc_by_section
sys.modules.setdefault("app.stard_summary", stard_mod)

chat_mod = types.ModuleType("app.llms.chatopenai")
class DummyLLM:
    async def ainvoke(self, *args, **kwargs):
        return types.SimpleNamespace(content="")
chat_mod.light_llm = DummyLLM()
chat_mod.strong_llm = DummyLLM()
sys.modules.setdefault("app.llms.chatopenai", chat_mod)

@pytest.mark.anyio
async def test_llm_screening_requires_question(anyio_backend):
    with pytest.raises(ValueError):
        await sr.llm_screening(None, "summary", ["crit"])

@pytest.mark.anyio
async def test_llm_screening_returns_llm_output(anyio_backend):
    expected = "ok"

    class DummyPrompt:
        def __init__(self):
            self.template = "dummy"
        def __or__(self, other):
            return other

    class DummyOutputLLM:
        async def ainvoke(self, inputs):
            return types.SimpleNamespace(content=expected)

    dummy_llm = DummyOutputLLM()
    with patch.object(sr, "generate_review_prompt", return_value=DummyPrompt()), \
         patch.object(sr, "light_llm", dummy_llm):
        result = await sr.llm_screening("question", "summary", ["crit"])
    assert result == expected