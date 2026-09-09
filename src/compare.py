"""
Phase 3, step 2: for each extracted requirement, retrieve relevant
profile context and ask the LLM to judge match / partial / gap --
strictly from that retrieved context, same grounding discipline as
Phase 2's Q&A.
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.config import CV_COLLECTION, TOP_K_PER_COLLECTION
from src.extract import _strip_code_fences
from src.ingest import _get_chroma_client, _get_embedding_fn
from src.schema import Requirement, RequirementAssessment

ASSESS_SYSTEM_PROMPT = """You judge whether a candidate's profile \
supports a specific job requirement, using ONLY the retrieved profile \
excerpts provided -- not outside knowledge or assumptions.

Classify as:
- "match": the profile clearly demonstrates this requirement
- "partial": the profile shows related/adjacent experience, but not a \
direct, confident match
- "gap": the profile shows no meaningful evidence of this requirement

Respond with ONLY valid JSON, no markdown fences, no commentary, in \
exactly this shape:
{"verdict": "match", "evidence": "one or two sentence justification, \
quoting or closely paraphrasing the relevant profile excerpt"}

Do not be generous -- if the excerpts don't clearly support the \
requirement, say so as "gap" rather than inferring a match from \
loosely related experience.
"""


def _retrieve_profile_context(query: str, embedding_fn) -> tuple[str, list[str]]:
    client = _get_chroma_client()
    collection = client.get_or_create_collection(
        name=CV_COLLECTION, embedding_function=embedding_fn
    )
    results = collection.query(query_texts=[query], n_results=TOP_K_PER_COLLECTION)
    docs = results.get("documents", [[]])[0]
    metas = results.get("metadatas", [[]])[0]

    sources = sorted({m.get("source", "?") for m in metas})
    context_str = "\n\n".join(
        f"[{m.get('source')}] {doc}" for doc, m in zip(docs, metas)
    )
    return context_str, sources


def assess_requirement(
    requirement: Requirement, llm, embedding_fn=None
) -> RequirementAssessment:
    if embedding_fn is None:
        embedding_fn = _get_embedding_fn()

    context_str, sources = _retrieve_profile_context(requirement.text, embedding_fn)

    user_prompt = (
        f"Requirement: {requirement.text}\n\n"
        f"Retrieved profile excerpts:\n\n{context_str if context_str else '(none found)'}"
    )

    response = llm.invoke([
        SystemMessage(content=ASSESS_SYSTEM_PROMPT),
        HumanMessage(content=user_prompt),
    ])

    raw = _strip_code_fences(response.content)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON for requirement assessment.\n"
            f"Requirement: {requirement.text}\n"
            f"Raw output was:\n{response.content}"
        ) from e

    return RequirementAssessment(
        requirement=requirement.text,
        importance=requirement.importance,
        verdict=data["verdict"],
        evidence=data["evidence"],
        sources=sources,
    )
