# RAG Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that grounds LLM responses in your own document corpus. Ingest PDF or plain-text files, index them into a local FAISS vector store, and query the chatbot to get accurate, cited answers.

## Features

- Document ingestion: PDF and `.txt` files
- Text chunking with configurable overlap
- Embedding and indexing via FAISS (local, no external vector DB required)
- Query interface using LangChain and OpenAI chat models
- Source citations returned with every answer

## Project Structure

```
rag-chatbot/
├── data/
│   └── documents/      # Place your PDF/TXT documents here
├── src/
│   ├── ingest.py        # Document loading, chunking, and indexing
│   ├── retriever.py     # Vector store retrieval utilities
│   └── chatbot.py       # RAG query interface (CLI)
├── tests/
│   └── test_ingest.py   # Unit tests for ingestion utilities
├── requirements.txt
└── README.md
```

## Quick Start

```bash
pip install -r requirements.txt
export OPENAI_API_KEY=your_key_here

# Step 1: Ingest documents and build the vector store index
python src/ingest.py --docs_dir data/documents --index_dir data/index

# Step 2: Chat with your documents
python src/chatbot.py --index_dir data/index
```

## Environment Variables

| Variable | Description |
|---|---|
| `OPENAI_API_KEY` | Your OpenAI API key (required) |
| `OPENAI_MODEL` | Chat model to use (default: `gpt-4o-mini`) |
| `CHUNK_SIZE` | Characters per text chunk (default: `1000`) |
| `CHUNK_OVERLAP` | Overlap between chunks (default: `200`) |
