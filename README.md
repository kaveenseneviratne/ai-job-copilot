# AI Job-Search Copilot

An agentic RAG system that ingests your CV and job postings, then
answers questions and flags skill gaps between them automatically.

Built as a portfolio project targeting AI Engineer roles: retrieval,
vector search, agentic orchestration, an evaluation harness, and a
served API + UI.

## Status: Phase 1 complete (ingestion)

- [x] **Phase 1 — Ingestion**: load CV/letter/job-posting documents,
      chunk them, embed them, store in a local Chroma vector DB.
- [ ] Phase 2 — Retrieval + Q&A over the store
- [ ] Phase 3 — Agent layer: compare a JD against your profile, flag gaps
- [ ] Phase 4 — FastAPI backend + Streamlit frontend
- [ ] Phase 5 — Evaluation harness (retrieval + answer quality)

## Setup

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Add your documents

- `data/documents/cv/` — your CV and recommendation letter (.txt or .pdf)
  already contains your current CV and Prof. Iliev's letter as a start.
- `data/documents/job_postings/` — job postings you're considering
  (.txt or .pdf) — already contains the DKSR posting as a test case.

Add more job postings here any time — just save the posting text as a
new `.txt` file in that folder and re-run ingestion.

## Run ingestion

```bash
python -m src.ingest
```

This downloads a small local embedding model (`all-MiniLM-L6-v2`, ~90MB,
first run only, needs internet) and builds two local Chroma collections:
one for your profile documents, one for job postings.

## Try retrieval directly

```bash
python -m src.query "does Kaveen have Docker experience?"
python -m src.query "what does the DKSR posting require for orchestration?"
```

This shows you the raw chunks retrieved for a query — useful for
sanity-checking retrieval quality before we add the agent layer on top
in Phase 2.

## Why two separate collections?

Keeping your profile and job postings in separate Chroma collections
(rather than one mixed collection) means a query like "what does this
JD require" won't accidentally pull back CV chunks, and vice versa.
The Phase 3 agent layer queries both and compares them explicitly.

## Notes on chunking

Chunks are ~500 characters with 80-character overlap, splitting on
word boundaries. This is intentionally simple and works well for
CV/letter/JD-length documents. If you later ingest much longer
documents (e.g. full company handbooks), consider swapping in a
token-aware splitter instead.
