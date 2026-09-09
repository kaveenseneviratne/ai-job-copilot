"""
Phase 2: Retrieval-augmented Q&A.

Takes a question, retrieves relevant chunks from BOTH the profile and
job-postings collections, and asks an LLM to answer strictly from that
retrieved context -- explicitly instructed to say so when the context
doesn't contain an answer, rather than guessing.

This is deliberately kept separate from any "agent" logic (Phase 3):
this module answers one question at a time from retrieved context. The
agent layer will call this repeatedly / route between profile and job
context on its own.

Run:
    python -m src.ask "Does Kaveen have Docker experience?"
"""
from __future__ import annotations

import sys
from dataclasses import dataclass

from langchain_groq import ChatGroq
from langchain_core.messages import HumanMessage, SystemMessage

from src.config import (
    CV_COLLECTION,
    GROQ_MODEL,
    JOBS_COLLECTION,
    LLM_TEMPERATURE,
    TOP_K_PER_COLLECTION,
    require_groq_key,
)
from src.ingest import _get_chroma_client, _get_embedding_fn

SYSTEM_PROMPT = """You are a factual assistant answering questions about a \
candidate's profile (CV, recommendation letter) and job postings they are \
considering, using ONLY the context chunks provided below.

Rules:
- Answer strictly from the provided context. Do not use outside knowledge \
about the candidate, the companies, or general assumptions.
- If the context does not contain enough information to answer, say so \
explicitly (e.g. "The profile does not mention X") rather than guessing \
or inferring something not stated.
- Be concise and direct.
- When relevant, note which document a piece of information came from.
"""


@dataclass
class RetrievedChunk:
    text: str
    source: str
    distance: float
    collection: str


def _retrieve(query: str, collection_name: str, embedding_fn) -> list[RetrievedChunk]:
    client = _get_chroma_client()
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_fn
    )
    results = collection.query(query_texts=[query], n_results=TOP_K_PER_COLLECTION)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    return [
        RetrievedChunk(text=doc, source=meta.get("source", "?"), distance=dist,
                        collection=collection_name)
        for doc, meta, dist in zip(docs, metas, dists)
    ]


def retrieve_context(query: str) -> list[RetrievedChunk]:
    """Retrieve relevant chunks from both the profile and job-posting collections."""
    embedding_fn = _get_embedding_fn()
    profile_chunks = _retrieve(query, CV_COLLECTION, embedding_fn)
    job_chunks = _retrieve(query, JOBS_COLLECTION, embedding_fn)
    return profile_chunks + job_chunks


def _format_context(chunks: list[RetrievedChunk]) -> str:
    lines = []
    for i, c in enumerate(chunks, 1):
        origin = "PROFILE" if c.collection == CV_COLLECTION else "JOB POSTING"
        lines.append(f"[{i}] ({origin} — {c.source})\n{c.text}")
    return "\n\n".join(lines)


def answer_question(query: str, llm: ChatGroq | None = None) -> tuple[str, list[RetrievedChunk]]:
    """Retrieve context for `query` and generate a grounded answer.

    `llm` can be injected for testing; defaults to a real ChatGroq client.
    """
    chunks = retrieve_context(query)
    context_str = _format_context(chunks)

    user_prompt = f"Context:\n\n{context_str}\n\nQuestion: {query}"

    if llm is None:
        require_groq_key()
        llm = ChatGroq(model=GROQ_MODEL, temperature=LLM_TEMPERATURE)

    response = llm.invoke([
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])
    return response.content, chunks


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m src.ask "your question here"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])

    print(f"Question: {query}\n")
    answer, chunks = answer_question(query)

    print("Answer:")
    print(answer)

    print(f"\n(Based on {len(chunks)} retrieved chunks: "
          + ", ".join(sorted({c.source for c in chunks})) + ")")


if __name__ == "__main__":
    main()
