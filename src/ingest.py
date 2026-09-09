"""
Phase 1: Ingestion pipeline.

Loads every .txt/.pdf file under data/documents/cv and
data/documents/job_postings, splits them into overlapping chunks,
embeds them with a local sentence-transformers model, and persists
them into two separate local Chroma collections (profile vs jobs).

Run:
    python -m src.ingest
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import chromadb
from chromadb.utils import embedding_functions

from src.config import (
    CHROMA_DIR,
    CHUNK_OVERLAP,
    CHUNK_SIZE,
    CV_COLLECTION,
    CV_DIR,
    EMBEDDING_MODEL,
    JOB_POSTINGS_DIR,
    JOBS_COLLECTION,
)


@dataclass
class Chunk:
    text: str
    source: str      # filename this chunk came from
    chunk_index: int


def _read_text(path: Path) -> str:
    if path.suffix.lower() == ".txt":
        return path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".pdf":
        try:
            from pypdf import PdfReader
        except ImportError as e:
            raise ImportError(
                "pypdf is required to read PDF files: pip install pypdf"
            ) from e
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)
    raise ValueError(f"Unsupported file type: {path.suffix}")


def _chunk_text(text: str, source: str) -> list[Chunk]:
    """Simple recursive-ish character chunker with overlap.

    Good enough for CV-length and JD-length documents. If you later
    ingest much longer documents, swap this for a token-aware splitter.
    """
    text = " ".join(text.split())  # normalize whitespace
    if not text:
        return []

    chunks: list[Chunk] = []
    start = 0
    idx = 0
    while start < len(text):
        end = min(start + CHUNK_SIZE, len(text))
        # try to break on a sentence/space boundary rather than mid-word
        if end < len(text):
            last_space = text.rfind(" ", start, end)
            if last_space > start:
                end = last_space
        chunk_text = text[start:end].strip()
        if chunk_text:
            chunks.append(Chunk(text=chunk_text, source=source, chunk_index=idx))
            idx += 1
        if end == len(text):
            break
        start = max(end - CHUNK_OVERLAP, start + 1)
    return chunks


def _load_chunks_from_dir(directory: Path) -> list[Chunk]:
    all_chunks: list[Chunk] = []
    if not directory.exists():
        return all_chunks
    for path in sorted(directory.glob("**/*")):
        if path.is_file() and path.suffix.lower() in (".txt", ".pdf"):
            text = _read_text(path)
            all_chunks.extend(_chunk_text(text, source=path.name))
    return all_chunks


def _get_chroma_client() -> chromadb.ClientAPI:
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(CHROMA_DIR))


def _get_embedding_fn():
    # Requires internet on first run to download the model weights.
    return embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name=EMBEDDING_MODEL
    )


def _upsert_chunks(
    client: chromadb.ClientAPI,
    collection_name: str,
    chunks: Iterable[Chunk],
    embedding_fn,
) -> int:
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_fn
    )
    chunks = list(chunks)
    if not chunks:
        return 0

    ids = [f"{c.source}::{c.chunk_index}" for c in chunks]
    documents = [c.text for c in chunks]
    metadatas = [{"source": c.source, "chunk_index": c.chunk_index} for c in chunks]

    collection.upsert(ids=ids, documents=documents, metadatas=metadatas)
    return len(chunks)


def main() -> None:
    print("Loading and chunking documents...")
    cv_chunks = _load_chunks_from_dir(CV_DIR)
    job_chunks = _load_chunks_from_dir(JOB_POSTINGS_DIR)
    print(f"  Profile documents: {len(cv_chunks)} chunks")
    print(f"  Job postings:      {len(job_chunks)} chunks")

    if not cv_chunks and not job_chunks:
        print("No documents found. Add files under data/documents/cv "
              "or data/documents/job_postings and re-run.")
        sys.exit(1)

    print(f"Loading embedding model '{EMBEDDING_MODEL}' "
          f"(downloads on first run, needs internet)...")
    embedding_fn = _get_embedding_fn()

    client = _get_chroma_client()

    n_cv = _upsert_chunks(client, CV_COLLECTION, cv_chunks, embedding_fn)
    n_jobs = _upsert_chunks(client, JOBS_COLLECTION, job_chunks, embedding_fn)

    print(f"Stored {n_cv} profile chunks in collection '{CV_COLLECTION}'")
    print(f"Stored {n_jobs} job-posting chunks in collection '{JOBS_COLLECTION}'")
    print(f"Chroma DB persisted at: {CHROMA_DIR}")


if __name__ == "__main__":
    main()
