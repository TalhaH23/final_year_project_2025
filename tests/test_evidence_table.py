import sys
import types
from dataclasses import dataclass
import json
import importlib
import pytest
import asyncio

# Provide dummy modules required for import
sys.modules.setdefault("pandas", types.ModuleType("pandas"))
dotenv_mod = types.ModuleType("dotenv")
dotenv_mod.load_dotenv = lambda *args, **kwargs: None
sys.modules.setdefault("dotenv", dotenv_mod)
sys.modules.setdefault("aiofiles", types.ModuleType("aiofiles"))

# langchain_core.documents.Document
@dataclass
class Document:
    page_content: str
    metadata: dict

doc_mod = types.ModuleType("langchain_core.documents")
doc_mod.Document = Document
sys.modules.setdefault("langchain_core.documents", doc_mod)
core_mod = sys.modules.setdefault("langchain_core", types.ModuleType("langchain_core"))
core_mod.documents = doc_mod

# langchain_openai.ChatOpenAI
class DummyChatOpenAI:
    def __init__(self, *args, **kwargs):
        pass
    async def ainvoke(self, inputs):
        return types.SimpleNamespace(content="")

openai_mod = types.ModuleType("langchain_openai")
openai_mod.ChatOpenAI = DummyChatOpenAI
sys.modules.setdefault("langchain_openai", openai_mod)

# langchain.prompts.PromptTemplate
class DummyChain:
    async def ainvoke(self, inputs):
        return types.SimpleNamespace(content="")

class DummyPromptTemplate:
    @classmethod
    def from_template(cls, template):
        return cls()
    def __or__(self, other):
        return DummyChain()

prompts_mod = types.ModuleType("langchain.prompts")
prompts_mod.PromptTemplate = DummyPromptTemplate
sys.modules.setdefault("langchain.prompts", prompts_mod)

# app.vector_stores.pinecone.vector_store
class DummyVectorStore:
    def similarity_search(self, query=None, k=5, filter=None):
        return [Document(page_content=f"content for {filter['pdf_id']}", metadata={"source": filter['pdf_id']})]

vector_mod = types.ModuleType("app.vector_stores.pinecone")
vector_mod.vector_store = DummyVectorStore()
sys.modules.setdefault("app.vector_stores.pinecone", vector_mod)

# app.llms.chatopenai.light_llm
class DummyLLM:
    async def ainvoke(self, inputs):
        return types.SimpleNamespace(content="")

llm_mod = types.ModuleType("app.llms.chatopenai")
llm_mod.light_llm = DummyLLM()
sys.modules.setdefault("app.llms.chatopenai", llm_mod)

# web.db.models.pdf.Pdf
@dataclass
class Pdf:
    id: str
    name: str = ""
    title: str | None = None

pdf_mod = types.ModuleType("web.db.models.pdf")
pdf_mod.Pdf = Pdf
sys.modules.setdefault("web.db.models.pdf", pdf_mod)

# Import the module under test
et = importlib.import_module("app.evidence_table")

def test_create_evidence_table(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    async def fake_extract(element, docs, fallback_docs, k=5):
        return f"{element}_summary"

    monkeypatch.setattr(et, "extract_component", fake_extract)

    pdfs = {
        "1": Pdf(id="1", title="Title1", name="name1"),
        "2": Pdf(id="2", name="name2"),
    }
    criteria = ["Population", "Intervention"]

    table = asyncio.run(et.create_evidence_table(pdfs, criteria, k=1))

    expected = [
        {"Document": "Title1", "Population": "Population_summary", "Intervention": "Intervention_summary"},
        {"Document": "name2", "Population": "Population_summary", "Intervention": "Intervention_summary"},
    ]

    assert table == expected