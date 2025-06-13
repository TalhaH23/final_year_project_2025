import importlib
import sys
import types
from dataclasses import dataclass
from unittest.mock import AsyncMock
import asyncio
import types as pytypes

# Stub dotenv
import types as _types;
sys.modules.setdefault("dotenv", _types.ModuleType("dotenv")).load_dotenv=lambda: None

# Dummy modules for external dependencies used during import
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

# Minimal langchain_core Document
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

# Stub for langchain_openai
langchain_openai = types.ModuleType("langchain_openai")
class ChatOpenAI:
    def __init__(self, *args, **kwargs):
        pass
langchain_openai.ChatOpenAI = ChatOpenAI
sys.modules.setdefault("langchain_openai", langchain_openai)

# Stub for langchain.text_splitter
splitter_mod = types.ModuleType("langchain.text_splitter")
class DummySplitter:
    def __init__(self, *args, **kwargs):
        pass
    @classmethod
    def from_tiktoken_encoder(cls, *args, **kwargs):
        return cls()
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

# Stub for langchain.prompts
prompts_mod = types.ModuleType("langchain.prompts")
class PromptTemplate:
    def __init__(self, *args, **kwargs):
        pass
    def __or__(self, other):
        return DummyChain()
prompts_mod.PromptTemplate = PromptTemplate
sys.modules.setdefault("langchain.prompts", prompts_mod)

# Dummy chain object
class DummyChain:
    def __init__(self):
        pass
    async def ainvoke(self, *args, **kwargs):
        return types.SimpleNamespace(content="")

# Stub for langchain_core.runnables
runnables_mod = types.ModuleType("langchain_core.runnables")
class RunnableLambda:
    def __init__(self, func):
        self.func = func
    def __call__(self, *args, **kwargs):
        return self.func(*args, **kwargs)
runnables_mod.RunnableLambda = RunnableLambda
sys.modules.setdefault("langchain_core.runnables", runnables_mod)

# Stub for aiofiles
sys.modules.setdefault("aiofiles", types.ModuleType("aiofiles"))

# Stub vector store and llm modules
vector_pkg = types.ModuleType("app.vector_stores")
vector_mod = types.ModuleType("app.vector_stores.pinecone")
vector_mod.vector_store = None
vector_pkg.pinecone = vector_mod
sys.modules.setdefault("app.vector_stores", vector_pkg)
sys.modules.setdefault("app.vector_stores.pinecone", vector_mod)

chat_mod = types.ModuleType("app.llms.chatopenai")
class DummyLLM:
    pass
chat_mod.light_llm = DummyLLM()
chat_mod.strong_llm = DummyLLM()
sys.modules.setdefault("app.llms.chatopenai", chat_mod)

stard_summary = importlib.import_module("app.stard_summary")

def test_llm_summary_basic(monkeypatch):
    docs = [
        stard_summary.Document(page_content="a", metadata={"section_title": "Intro", "main_title": "Doc"}),
        stard_summary.Document(page_content="b", metadata={"section_title": "Intro", "main_title": "Doc"}),
        stard_summary.Document(page_content="c", metadata={"section_title": "Methods", "main_title": "Doc"}),
    ]

    async def fake_chunk(args):
        return pytypes.SimpleNamespace(content=f"chunk:{args['text']}")

    async def fake_section(args):
        return pytypes.SimpleNamespace(content=f"sec:{args['section_title']}:{args['summaries']}")

    async def fake_doc(args):
        return pytypes.SimpleNamespace(content=f"doc:{args['main_title']}:{args['text']}")

    monkeypatch.setattr(stard_summary, "chunk_summary_chain", pytypes.SimpleNamespace(ainvoke=AsyncMock(side_effect=fake_chunk)))
    monkeypatch.setattr(stard_summary, "section_summary_chain", pytypes.SimpleNamespace(ainvoke=AsyncMock(side_effect=fake_section)))
    monkeypatch.setattr(stard_summary, "document_reduce_chain", pytypes.SimpleNamespace(ainvoke=AsyncMock(side_effect=fake_doc)))
    monkeypatch.setattr(stard_summary, "stard_checklist", "check")

    result = asyncio.run(stard_summary.llm_summary(docs))

    expected = "doc:Doc:sec:Intro:chunk:a\nchunk:b\n\nsec:Methods:chunk:c"
    assert result == expected