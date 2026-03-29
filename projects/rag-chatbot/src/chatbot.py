"""RAG chatbot CLI powered by LangChain and OpenAI."""

import argparse
import os

from dotenv import load_dotenv
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_openai import ChatOpenAI

from retriever import get_retriever

load_dotenv()

DEFAULT_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")


def build_chain(index_dir: str, model: str = DEFAULT_MODEL):
    """Construct the RAG chain with source citations."""
    retriever = get_retriever(index_dir)
    llm = ChatOpenAI(model=model, temperature=0)
    chain = RetrievalQAWithSourcesChain.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
    )
    return chain


def chat(index_dir: str, model: str = DEFAULT_MODEL) -> None:
    """Interactive CLI chat loop."""
    print(f"RAG Chatbot ready (model={model}, index={index_dir})")
    print("Type 'exit' or 'quit' to stop.\n")
    chain = build_chain(index_dir, model)

    while True:
        query = input("You: ").strip()
        if not query:
            continue
        if query.lower() in {"exit", "quit"}:
            break

        result = chain.invoke({"question": query})
        answer = result.get("answer", "No answer found.")
        sources = result.get("source_documents", [])

        print(f"\nAssistant: {answer}")
        if sources:
            print("Sources:")
            for doc in sources:
                source = doc.metadata.get("source", "unknown")
                print(f"  - {source}")
        print()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="RAG Chatbot CLI")
    parser.add_argument("--index_dir", default="data/index", help="Path to the FAISS index")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="OpenAI chat model name")
    args = parser.parse_args()
    chat(args.index_dir, args.model)
