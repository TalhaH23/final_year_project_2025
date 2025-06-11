import sys
import types
from dataclasses import dataclass

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

# With dummy modules in place we can import the module under test
import app.title_extraction as te
from unittest.mock import patch


def test_chunk_document_by_titles_with_titles(tmp_path):
    dummy_elements = [elements.Element()]
    text = "Introduction\nIntro text\nMethods\nMethods text"
    with patch.object(te, "get_partitioned_elements", return_value=dummy_elements), \
         patch.object(te, "get_intersecting_titles", return_value=(["Introduction", "Methods"], "Mock Title")), \
         patch.object(te, "extract_cleaned_text", return_value=text), \
         patch.object(te, "_write_chunk_to_file"):
        docs, main_title = te.chunk_document_by_titles("dummy.pdf", chunk_size=1000, chunk_overlap=0)

    assert main_title == "Mock Title"
    assert len(docs) == 2
    assert docs[0].metadata == {
        "source": "dummy.pdf",
        "main_title": "Mock Title",
        "section_title": "Introduction",
        "table": False,
    }
    assert docs[1].metadata["section_title"] == "Methods"


def test_chunk_document_by_titles_no_titles():
    dummy_elements = [elements.Element()]
    text = "Only text without titles"
    with patch.object(te, "get_partitioned_elements", return_value=dummy_elements), \
         patch.object(te, "get_intersecting_titles", return_value=([], "Mock Title")), \
         patch.object(te, "extract_cleaned_text", return_value=text), \
         patch.object(te, "_write_chunk_to_file"):
        docs = te.chunk_document_by_titles("dummy.pdf", chunk_size=1000, chunk_overlap=0)

    assert len(docs) == 1
    assert docs[0].metadata == {
        "source": "dummy.pdf",
        "main_title": "Mock Title",
        "section_title": "Full Document",
        "table": False,
    }