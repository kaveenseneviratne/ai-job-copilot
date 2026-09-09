"""
Phase 1 sanity check: query the vector store directly and inspect
what comes back, before any agent logic sits on top of it.

Run:
    python -m src.query "does Kaveen have Docker experience?"
"""
from __future__ import annotations

import sys

from src.ingest import _get_chroma_client, _get_embedding_fn
from src.config import CV_COLLECTION, JOBS_COLLECTION


def search(query: str, collection_name: str, n_results: int = 3):
    client = _get_chroma_client()
    embedding_fn = _get_embedding_fn()
    collection = client.get_or_create_collection(
        name=collection_name, embedding_function=embedding_fn
    )
    results = collection.query(query_texts=[query], n_results=n_results)
    return results


def print_results(results, label: str):
    print(f"\n--- {label} ---")
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]
    dists = results.get("distances", [[]])[0]
    if not docs:
        print("  (no results — did you run `python -m src.ingest` first?)")
        return
    for doc, meta, dist in zip(docs, metas, dists):
        print(f"  [{meta.get('source')} #{meta.get('chunk_index')}] "
              f"(distance={dist:.3f})")
        print(f"    {doc[:200]}{'...' if len(doc) > 200 else ''}")


def main() -> None:
    if len(sys.argv) < 2:
        print('Usage: python -m src.query "your question here"')
        sys.exit(1)
    query = " ".join(sys.argv[1:])

    profile_results = search(query, CV_COLLECTION)
    job_results = search(query, JOBS_COLLECTION)

    print_results(profile_results, "Profile matches (CV / recommendation letter)")
    print_results(job_results, "Job posting matches")


if __name__ == "__main__":
    main()
