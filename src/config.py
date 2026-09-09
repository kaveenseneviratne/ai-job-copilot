"""Shared configuration and paths for the AI Job-Search Copilot."""
import os
from pathlib import Path

from dotenv import load_dotenv

# --- Paths ---
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Load variables from a local .env file (e.g. GROQ_API_KEY) into the
# environment. Safe to call even if .env doesn't exist yet.
load_dotenv(PROJECT_ROOT / ".env")
DATA_DIR = PROJECT_ROOT / "data" / "documents"
CV_DIR = DATA_DIR / "cv"
JOB_POSTINGS_DIR = DATA_DIR / "job_postings"
CHROMA_DIR = PROJECT_ROOT / "chroma_db"

# --- Chroma collections ---
# We keep two separate collections so retrieval can be scoped:
# querying "what does my profile say about Docker" should search
# CV_COLLECTION only; querying "what does this JD require" should
# search JOBS_COLLECTION only. The agent layer combines both.
CV_COLLECTION = "kaveen_profile"
JOBS_COLLECTION = "job_postings"

# --- Chunking ---
CHUNK_SIZE = 500       # characters per chunk
CHUNK_OVERLAP = 80     # overlap between consecutive chunks

# --- Embeddings ---
# sentence-transformers model, downloaded on first run (requires internet).
EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# --- Retrieval (Phase 2+) ---
TOP_K_PER_COLLECTION = 4  # how many chunks to pull from each collection per query

# --- LLM (used in Phase 2+: Q&A, agent, generation) ---
# Uses Groq's OpenAI-compatible API, matching the stack from your
# cold-email-generator project. Set GROQ_API_KEY in your environment
# or a .env file before running phases 2+.
# NOTE: llama-3.3-70b-versatile moved to Groq's Enterprise-only tier
# (contact-sales pricing) as of late 2026, so it's not available on
# free/developer accounts anymore. openai/gpt-oss-120b is the current
# generally-available flagship model on Groq's free/developer tier.
GROQ_MODEL = "openai/gpt-oss-120b"
GROQ_API_KEY_ENV_VAR = "GROQ_API_KEY"
LLM_TEMPERATURE = 0.1  # low: we want grounded, consistent answers, not creative ones


def require_groq_key() -> str:
    key = os.environ.get(GROQ_API_KEY_ENV_VAR)
    if not key:
        raise RuntimeError(
            f"{GROQ_API_KEY_ENV_VAR} is not set. Create a .env file "
            f"(see .env.example) or export it in your shell before running."
        )
    return key
