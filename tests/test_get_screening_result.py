import sys
import types
from dataclasses import dataclass
import json
import pytest
import asyncio
from unittest.mock import patch
import app.systematic_review as sr

# Basic stubs for modules required by app.title_extraction
for name in ["fitz", "pdfplumber", "pymupdf"]:
    sys.modules.setdefault(name, types.ModuleType(name))

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
class Title: pass
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

# dotenv.load_dotenv
dotenv_mod = types.ModuleType('dotenv')
def noop(*args, **kwargs):
    pass
dotenv_mod.load_dotenv = noop
sys.modules.setdefault('dotenv', dotenv_mod)

# langchain_openai.ChatOpenAI
lc_openai = types.ModuleType('langchain_openai')
class DummyChat:
    async def ainvoke(self, *args, **kwargs):
        return types.SimpleNamespace(content="")
lc_openai.ChatOpenAI = DummyChat
sys.modules.setdefault('langchain_openai', lc_openai)

# langchain.prompts.PromptTemplate
lc_prompts = types.ModuleType('langchain.prompts')
class PromptTemplate:
    def __init__(self, input_variables=None, template=""):
        self.input_variables = input_variables
        self.template = template
    def __or__(self, other):
        return self
    async def ainvoke(self, data):
        return types.SimpleNamespace(content="")
lc_prompts.PromptTemplate = PromptTemplate
sys.modules.setdefault('langchain.prompts', lc_prompts)

# aiofiles with async open/write
aiofiles_mod = types.ModuleType('aiofiles')
class AsyncFile:
    def __init__(self, path, mode='r', encoding=None):
        self._f = open(path, mode, encoding=encoding)
    async def __aenter__(self):
        return self
    async def __aexit__(self, exc_type, exc, tb):
        self._f.close()
    async def write(self, data):
        self._f.write(data)


def aio_open(path, mode='r', encoding=None):
    return AsyncFile(path, mode, encoding)

aiofiles_mod.open = aio_open
sys.modules.setdefault('aiofiles', aiofiles_mod)

# langchain_core.documents.Document
lc_core = types.ModuleType('langchain_core')
lc_docs = types.ModuleType('langchain_core.documents')
@dataclass
class Document:
    page_content: str
    metadata: dict
lc_docs.Document = Document
lc_core.documents = lc_docs
sys.modules.setdefault('langchain_core', lc_core)
sys.modules.setdefault('langchain_core.documents', lc_docs)

splitter_mod = types.ModuleType('langchain.text_splitter')
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
sys.modules.setdefault('langchain.text_splitter', splitter_mod)

community_mod = types.ModuleType('langchain_community')
loaders_mod = types.ModuleType('langchain_community.document_loaders')
class PyMuPDFLoader:
    def __init__(self, *args, **kwargs):
        pass
    def load(self):
        return []
loaders_mod.PyMuPDFLoader = PyMuPDFLoader
community_mod.document_loaders = loaders_mod
sys.modules.setdefault('langchain_community', community_mod)
sys.modules.setdefault('langchain_community.document_loaders', loaders_mod)

tiktoken_mod = types.ModuleType('tiktoken')
class DummyEncoding:
    def encode(self, text):
        return text.split()
def encoding_for_model(model_name):
    return DummyEncoding()
tiktoken_mod.encoding_for_model = encoding_for_model
sys.modules.setdefault('tiktoken', tiktoken_mod)

# app.vector_stores.pinecone placeholder
pine_mod = types.ModuleType('app.vector_stores.pinecone')
pine_mod.vector_store = None
sys.modules.setdefault('app.vector_stores.pinecone', pine_mod)

# app.llms.chatopenai placeholder
llms_mod = types.ModuleType('app.llms.chatopenai')
llms_mod.light_llm = DummyChat()
llms_mod.strong_llm = DummyChat()
sys.modules.setdefault('app.llms.chatopenai', llms_mod)

# app.stard_summary placeholder (patched later)
stard_mod = types.ModuleType('app.stard_summary')
async def dummy_summary(docs):
    return ""
stard_mod.llm_summary = dummy_summary
stard_mod.group_doc_by_section = lambda docs: [docs]
sys.modules.setdefault('app.stard_summary', stard_mod)

def test_get_screening_result(tmp_path):
    async def run_test():
        summary_folder = tmp_path / "summ"
        review_folder = tmp_path / "rev"
        summary_folder.mkdir()
        review_folder.mkdir()

        docs = [sr.Document(page_content="dummy", metadata={"section_title": "Sec", "main_title": "Main"})]
        criteria = ["Population"]

        async def fake_summary(d):
            assert d == docs
            return "summary text"

        async def fake_screening(q, s, c):
            assert q == "question"
            assert s == "summary text"
            assert c == criteria
            return "Decision: Include\nConfidence: 5\nRationale: Good"

        parsed = {"decision": "Include"}

        with patch.object(sr, "llm_summary", side_effect=fake_summary) as psum, \
             patch.object(sr, "llm_screening", side_effect=fake_screening) as pscr, \
             patch.object(sr, "parse_llm_screening_output", return_value=parsed) as pparse:
            result_path = await sr.get_screening_result(
                "id1", "question", str(summary_folder), str(review_folder), docs, criteria
            )

        assert result_path == str(summary_folder / "id1.txt")
        assert (summary_folder / "id1.txt").read_text() == "summary text"
        written = json.loads((review_folder / "id1_screening_result.json").read_text())
        assert written == parsed
        assert psum.called
        assert pscr.called
        assert pparse.called

    asyncio.run(run_test())