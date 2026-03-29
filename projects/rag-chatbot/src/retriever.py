"""Vector store retrieval utilities for the RAG chatbot."""

import os

from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings


def load_index(index_dir: str) -> FAISS:
    """Load a persisted FAISS vector store from disk.

    Note: ``allow_dangerous_deserialization=True`` is required by LangChain's
    FAISS wrapper when loading a locally saved index. Only load indexes from
    trusted, controlled sources (e.g., files you generated yourself).
    """
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.load_local(
        index_dir, embeddings, allow_dangerous_deserialization=True
    )
    return vector_store


def get_retriever(index_dir: str, k: int = 4):
    """Return a retriever that fetches the top-k most similar chunks."""
    vector_store = load_index(index_dir)
    return vector_store.as_retriever(search_kwargs={"k": k})
