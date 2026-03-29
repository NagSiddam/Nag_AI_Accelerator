"""Unit tests for RAG chatbot ingestion utilities."""

import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ingest import chunk_documents, CHUNK_SIZE, CHUNK_OVERLAP
from langchain_core.documents import Document


def _make_doc(content: str, source: str = "test.txt") -> Document:
    return Document(page_content=content, metadata={"source": source})


def test_chunk_documents_respects_chunk_size():
    """Chunks should not exceed CHUNK_SIZE characters (with some splitter tolerance)."""
    long_text = "word " * 600  # ~3000 characters
    docs = [_make_doc(long_text)]
    chunks = chunk_documents(docs)
    for chunk in chunks:
        assert len(chunk.page_content) <= CHUNK_SIZE + CHUNK_OVERLAP


def test_chunk_documents_preserves_metadata():
    """Each chunk should carry the original source metadata."""
    docs = [_make_doc("Short text about AI.", source="ai_doc.txt")]
    chunks = chunk_documents(docs)
    assert all(c.metadata.get("source") == "ai_doc.txt" for c in chunks)


def test_chunk_documents_returns_list():
    docs = [_make_doc("Hello world.")]
    result = chunk_documents(docs)
    assert isinstance(result, list)
    assert len(result) >= 1


def test_load_documents_empty_dir():
    """load_documents should return an empty list for an empty directory."""
    from ingest import load_documents

    with tempfile.TemporaryDirectory() as tmpdir:
        docs = load_documents(tmpdir)
        assert docs == []
