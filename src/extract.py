"""
Phase 3, step 1: extract structured requirements from a raw job posting.

The LLM reads the full job posting text (not chunked -- postings are
short enough to fit in context whole) and returns a JSON list of
requirements, each tagged must_have or nice_to_have. We validate that
JSON against the Requirement schema so malformed output fails loudly
here rather than corrupting the rest of the pipeline.
"""
from __future__ import annotations

import json
import re

from langchain_core.messages import HumanMessage, SystemMessage

from src.schema import RequirementList

EXTRACT_SYSTEM_PROMPT = """You extract structured requirements from job \
postings.

Read the job posting below and list every distinct requirement or \
expected skill, tagging each as "must_have" or "nice_to_have" based on \
how the posting frames it (e.g. "must-have", "required" -> must_have; \
"nice to have", "bonus", "preferred" -> nice_to_have; unlabeled core \
responsibilities -> must_have).

Respond with ONLY valid JSON, no markdown fences, no commentary, in \
exactly this shape:
{"requirements": [{"text": "...", "importance": "must_have"}, ...]}

Keep each requirement text short and specific (e.g. "Hands-on Apache \
Airflow experience", not a whole paragraph). Do not merge unrelated \
requirements into one entry.
"""


def _strip_code_fences(text: str) -> str:
    """LLMs sometimes wrap JSON in ```json ... ``` despite instructions
    not to. Strip that defensively rather than failing on it."""
    text = text.strip()
    match = re.match(r"^```(?:json)?\s*(.*?)\s*```$", text, re.DOTALL)
    return match.group(1) if match else text


def extract_requirements(job_text: str, llm) -> RequirementList:
    response = llm.invoke([
        SystemMessage(content=EXTRACT_SYSTEM_PROMPT),
        HumanMessage(content=f"Job posting:\n\n{job_text}"),
    ])

    raw = _strip_code_fences(response.content)
    try:
        data = json.loads(raw)
    except json.JSONDecodeError as e:
        raise ValueError(
            f"Model did not return valid JSON for requirement extraction.\n"
            f"Raw output was:\n{response.content}"
        ) from e

    return RequirementList.model_validate(data)
