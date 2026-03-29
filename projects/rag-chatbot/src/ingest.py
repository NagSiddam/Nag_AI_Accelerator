"""Document ingestion: load, chunk, embed, and index documents with FAISS."""

import argparse
import os
from pathlib import Path
from typing import List

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import DirectoryLoader, PyPDFLoader, TextLoader
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "200"))


def load_documents(docs_dir: str) -> List:
    """Load all PDF and TXT files from a directory."""
    path = Path(docs_dir)
    docs = []

    # Load PDFs
    pdf_loader = DirectoryLoader(str(path), glob="**/*.pdf", loader_cls=PyPDFLoader)
    docs.extend(pdf_loader.load())

    # Load plain text files
    txt_loader = DirectoryLoader(str(path), glob="**/*.txt", loader_cls=TextLoader)
    docs.extend(txt_loader.load())

    print(f"Loaded {len(docs)} document pages from {docs_dir}")
    return docs


def chunk_documents(docs: List) -> List:
    """Split documents into overlapping chunks for indexing."""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )
    chunks = splitter.split_documents(docs)
    print(f"Split into {len(chunks)} chunks")
    return chunks


def build_index(chunks: List, index_dir: str) -> FAISS:
    """Embed chunks and persist a FAISS vector store."""
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(chunks, embeddings)
    os.makedirs(index_dir, exist_ok=True)
    vector_store.save_local(index_dir)
    print(f"Index saved to {index_dir}")
    return vector_store


def ingest(docs_dir: str, index_dir: str) -> None:
    docs = load_documents(docs_dir)
    if not docs:
        print("No documents found. Place .pdf or .txt files in the docs directory.")
        return
    chunks = chunk_documents(docs)
    build_index(chunks, index_dir)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest documents into a FAISS vector store")
    parser.add_argument("--docs_dir", default="data/documents", help="Directory with PDF/TXT files")
    parser.add_argument("--index_dir", default="data/index", help="Directory to save FAISS index")
    args = parser.parse_args()
    ingest(args.docs_dir, args.index_dir)
