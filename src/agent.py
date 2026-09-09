"""
Phase 3: the agent layer.

Orchestrates the full gap-analysis pipeline as a LangGraph graph:

    extract_requirements -> assess_all_requirements -> build_report

Each node reads/writes a shared state dict. This is a deliberately
simple linear graph -- the LangGraph value here isn't complex branching,
it's having an explicit, inspectable state machine instead of a bare
function call chain, which is what the rest of this project (and a
future FastAPI layer) can hook into, inspect, or extend (e.g. adding a
conditional retry node if extraction fails validation).

Run:
    python -m src.agent data/documents/job_postings/dksr_data_engineer.txt
"""
from __future__ import annotations

import sys
from pathlib import Path
from typing import TypedDict

from langchain_groq import ChatGroq
from langgraph.graph import END, StateGraph

from src.compare import assess_requirement
from src.config import GROQ_MODEL, LLM_TEMPERATURE, require_groq_key
from src.extract import extract_requirements
from src.ingest import _get_embedding_fn
from src.schema import GapReport, Requirement, RequirementAssessment


class AgentState(TypedDict, total=False):
    job_text: str
    company_or_role: str
    llm: ChatGroq
    embedding_fn: object
    requirements: list[Requirement]
    assessments: list[RequirementAssessment]
    report: GapReport


def _extract_node(state: AgentState) -> AgentState:
    result = extract_requirements(state["job_text"], state["llm"])
    return {"requirements": result.requirements}


def _assess_node(state: AgentState) -> AgentState:
    assessments = [
        assess_requirement(req, state["llm"], state["embedding_fn"])
        for req in state["requirements"]
    ]
    return {"assessments": assessments}


def _report_node(state: AgentState) -> AgentState:
    report = GapReport(
        company_or_role=state.get("company_or_role", "this role"),
        assessments=state["assessments"],
    )
    return {"report": report}


def build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("extract", _extract_node)
    graph.add_node("assess", _assess_node)
    graph.add_node("report", _report_node)

    graph.set_entry_point("extract")
    graph.add_edge("extract", "assess")
    graph.add_edge("assess", "report")
    graph.add_edge("report", END)

    return graph.compile()


def run_gap_analysis(
    job_text: str,
    company_or_role: str = "this role",
    llm=None,
    embedding_fn=None,
) -> GapReport:
    if llm is None:
        require_groq_key()
        llm = ChatGroq(model=GROQ_MODEL, temperature=LLM_TEMPERATURE)
    if embedding_fn is None:
        embedding_fn = _get_embedding_fn()

    app = build_graph()
    final_state = app.invoke({
        "job_text": job_text,
        "company_or_role": company_or_role,
        "llm": llm,
        "embedding_fn": embedding_fn,
    })
    return final_state["report"]


VERDICT_LABELS = {"match": "✅ Match", "partial": "⚠️  Partial", "gap": "❌ Gap"}


def format_report(report: GapReport) -> str:
    lines = [f"# Gap Analysis: {report.company_or_role}\n"]

    must_haves = report.must_have_assessments
    nice_to_haves = report.nice_to_have_assessments

    if must_haves:
        met = sum(1 for a in must_haves if a.verdict == "match")
        lines.append(
            f"## Must-haves: {met}/{len(must_haves)} matched "
            f"({report.must_have_match_rate:.0%})\n"
        )
        for a in must_haves:
            lines.append(f"**{VERDICT_LABELS[a.verdict]} — {a.requirement}**")
            lines.append(f"  {a.evidence}")
            if a.sources:
                lines.append(f"  _Sources: {', '.join(a.sources)}_")
            lines.append("")

    if nice_to_haves:
        lines.append("## Nice-to-haves\n")
        for a in nice_to_haves:
            lines.append(f"**{VERDICT_LABELS[a.verdict]} — {a.requirement}**")
            lines.append(f"  {a.evidence}")
            if a.sources:
                lines.append(f"  _Sources: {', '.join(a.sources)}_")
            lines.append("")

    gaps = [a.requirement for a in must_haves if a.verdict == "gap"]
    if gaps:
        lines.append("## Priority gaps to address")
        for g in gaps:
            lines.append(f"- {g}")

    return "\n".join(lines)


def main() -> None:
    if len(sys.argv) < 2:
        print("Usage: python -m src.agent <path_to_job_posting.txt>")
        sys.exit(1)

    path = Path(sys.argv[1])
    job_text = path.read_text(encoding="utf-8")

    print(f"Analyzing {path.name}...\n")
    report = run_gap_analysis(job_text, company_or_role=path.stem)
    print(format_report(report))


if __name__ == "__main__":
    main()
