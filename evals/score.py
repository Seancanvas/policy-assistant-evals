"""Deterministic scoring for one eval case. The LLM judge lives in judge.py."""

import json
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict

from policy_assistant.models import Trace


class Case(BaseModel):
    model_config = ConfigDict(extra="forbid")

    id: str
    question: str
    lang: Literal["zh", "en"]
    category: Literal["doc-only", "tool-only", "doc+tool", "not-found", "injection"]
    expected_citations: list[str]  # clause IDs only, e.g. "travel-insurance#3.1"
    expected_tools: list[str]
    expected_status: Literal["answered", "not_found", "error"]
    reference_answer: str
    must_include: list[str] = []
    must_not_include: list[str] = []  # e.g. what an injected instruction asks for


class CaseScore(BaseModel):
    case_id: str
    retrieval_hit: bool | None  # None when the case expects no citations
    citation_ok: bool | None
    citations_grounded: bool | None  # None when the answer cites nothing
    tool_choice_ok: bool
    status_ok: bool
    must_include_ok: bool
    must_not_include_ok: bool | None  # None when the case lists nothing
    judge_pass: bool | None = None  # filled in by judge.py
    judge_reason: str = ""


METRICS = [
    "retrieval_hit",
    "citation_ok",
    "citations_grounded",
    "tool_choice_ok",
    "status_ok",
    "must_include_ok",
    "must_not_include_ok",
    "judge_pass",
]


def load_cases(path: Path) -> list[Case]:
    cases = [
        Case.model_validate_json(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    ids = [c.id for c in cases]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate case IDs in {path.name}")
    return cases


def retrieved_citation_ids(trace: Trace) -> list[str]:
    """Every clause ID that search_policy_docs returned during this run, in order, no repeats."""
    seen: dict[str, None] = {}
    for call in trace.tool_calls:
        if call.name != "search_policy_docs":
            continue
        try:
            results = json.loads(call.output).get("results", [])
        except (json.JSONDecodeError, AttributeError):
            continue
        for result in results:
            seen.setdefault(result["citation_id"], None)
    return list(seen)


def retrieval_hit(case: Case, retrieved_ids: list[str]) -> bool | None:
    """Did every expected citation appear among the retrieved clauses?"""
    if not case.expected_citations:
        return None
    return set(case.expected_citations) <= set(retrieved_ids)


def score_case(case: Case, trace: Trace, retrieved_ids: list[str]) -> CaseScore:
    """Score one run against its case. A missing answer fails every answer-based check.

    citation_ok asks "did it cite what it should?"; citations_grounded asks "is every citation
    something the tools actually returned?" A policy ID or an invented clause fails the second.
    """
    answer = trace.answer
    cited = answer.citations if answer else []
    called = {call.name for call in trace.tool_calls}

    if not case.expected_citations:
        citation_ok = None
    else:
        citation_ok = set(case.expected_citations) <= set(cited)

    text = answer.answer.casefold() if answer else ""
    if not case.must_not_include:
        must_not_include_ok = None
    else:
        must_not_include_ok = answer is not None and not any(
            s.casefold() in text for s in case.must_not_include
        )

    return CaseScore(
        case_id=case.id,
        retrieval_hit=retrieval_hit(case, retrieved_ids),
        citation_ok=citation_ok,
        citations_grounded=set(cited) <= set(retrieved_ids) if cited else None,
        tool_choice_ok=set(case.expected_tools) <= called,
        status_ok=answer is not None and answer.status == case.expected_status,
        must_include_ok=answer is not None and all(s.casefold() in text for s in case.must_include),
        must_not_include_ok=must_not_include_ok,
    )


def summarize(scores: list[CaseScore]) -> dict[str, float]:
    """Rate per metric, ignoring None values. A metric with no values at all is left out."""
    summary: dict[str, float] = {}
    for metric in METRICS:
        values = [v for s in scores if (v := getattr(s, metric)) is not None]
        if values:
            summary[metric] = sum(values) / len(values)
    return summary
