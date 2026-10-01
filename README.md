# AI Job-Search Copilot

An agentic RAG system that ingests your CV and job postings, then
answers questions and flags skill gaps between them automatically.

Built as a portfolio project targeting AI Engineer roles: retrieval,
vector search, agentic orchestration, an evaluation harness, and a
served API + UI.

## Status: Phase 4 complete (API + frontend)

- [x] **Phase 1 — Ingestion**: load CV/letter/job-posting documents,
      chunk them, embed them, store in a local Chroma vector DB.
- [x] **Phase 2 — Retrieval + Q&A**: grounded question-answering over
      the ingested documents via Groq.
- [x] **Phase 3 — Agent layer**: LangGraph agent that extracts
      structured requirements from a job posting and checks each
      against your profile, producing a gap report.
- [x] **Phase 4 — API + frontend**: FastAPI backend exposing `/ask`
      and `/analyze`; Streamlit UI on top of it.
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
```

Shows the raw retrieved chunks for a query — useful for sanity-checking
retrieval quality on its own, before any LLM sits on top of it.

## Ask a grounded question (Phase 2)

Requires a `GROQ_API_KEY` in a local `.env` file (see `.env.example`).

```bash
python -m src.ask "Does Kaveen have Docker experience?"
```

## Run a full gap analysis (Phase 3)

```bash
python -m src.agent data/documents/job_postings/dksr_data_engineer.txt
```

Extracts structured requirements from the posting, checks each against
your profile via retrieval, and prints a formatted gap report.

## Run the full app: API + Streamlit UI (Phase 4)

Two terminals, both with the venv activated:

```bash
# Terminal 1: the API
uvicorn src.api:app --reload --port 8000

# Terminal 2: the UI
streamlit run streamlit_app.py
```

Streamlit opens automatically in your browser (usually
`http://localhost:8501`). The API also has interactive docs at
`http://localhost:8000/docs` if you want to poke at it directly.

The UI has two tabs: ask a free-form question about your profile, or
paste a full job posting for a structured gap analysis.

## Why two separate collections?

Keeping your profile and job postings in separate Chroma collections
(rather than one mixed collection) means a query like "what does this
JD require" won't accidentally pull back CV chunks, and vice versa.
The Phase 3 agent queries the profile collection specifically when
checking each requirement.

## Notes on chunking

Chunks are ~500 characters with 80-character overlap, splitting on
word boundaries. This is intentionally simple and works well for
CV/letter/JD-length documents. If you later ingest much longer
documents (e.g. full company handbooks), consider swapping in a
token-aware splitter instead.
