"""
Phase 3: Structured data models for the agent layer.

Using pydantic here (rather than plain dataclasses like Phase 1/2) because
these objects get built FROM LLM output -- pydantic gives us validation
and clear errors when the model returns something malformed, instead of
silently propagating bad data downstream.
"""
from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

Importance = Literal["must_have", "nice_to_have"]
Verdict = Literal["match", "partial", "gap"]


class Requirement(BaseModel):
    text: str
    importance: Importance


class RequirementList(BaseModel):
    requirements: list[Requirement]


class RequirementAssessment(BaseModel):
    requirement: str
    importance: Importance
    verdict: Verdict
    evidence: str
    sources: list[str] = Field(default_factory=list)


class GapReport(BaseModel):
    company_or_role: str
    assessments: list[RequirementAssessment]

    @property
    def must_have_assessments(self) -> list[RequirementAssessment]:
        return [a for a in self.assessments if a.importance == "must_have"]

    @property
    def nice_to_have_assessments(self) -> list[RequirementAssessment]:
        return [a for a in self.assessments if a.importance == "nice_to_have"]

    @property
    def must_have_match_rate(self) -> float:
        must_haves = self.must_have_assessments
        if not must_haves:
            return 1.0
        matches = sum(1 for a in must_haves if a.verdict == "match")
        return matches / len(must_haves)
