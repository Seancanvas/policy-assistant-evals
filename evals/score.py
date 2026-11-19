"""Deterministic scoring for one eval case. The LLM judge lives in judge.py."""

from pydantic import BaseModel

from policy_assistant.models import Trace


class Case(BaseModel):
    id: str
    question: str
    lang: str
    category: str  # doc-only | tool-only | doc+tool | not-found | injection
    expected_citations: list[str]
    expected_tools: list[str]
    expected_status: str
    reference_answer: str
    must_include: list[str] = []


class CaseScore(BaseModel):
    case_id: str
    retrieval_hit: bool | None  # None when the case expects no citations
    citation_ok: bool | None
    tool_choice_ok: bool
    status_ok: bool
    must_include_ok: bool
    judge_pass: bool | None = None  # filled in by judge.py
    judge_reason: str = ""


def retrieval_hit(case: Case, retrieved_ids: list[str]) -> bool | None:
    """Did every expected citation appear among the retrieved clauses?"""
    raise NotImplementedError("TODO day 2")


def score_case(case: Case, trace: Trace, retrieved_ids: list[str]) -> CaseScore:
    """TODO day 2:
    - citation_ok: the answer cites every expected citation (None if none expected).
    - tool_choice_ok: every expected tool was called at least once.
    - status_ok: answer.status == expected_status; a missing answer counts as False.
    - must_include_ok: every must_include string appears in answer.answer.
    """
    raise NotImplementedError("TODO day 2")


def summarize(scores: list[CaseScore]) -> dict[str, float]:
    """Rate per metric, ignoring None values. For example {"tool_choice_ok": 0.875, ...}."""
    raise NotImplementedError("TODO day 2")
