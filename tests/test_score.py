import json
from collections import Counter
from pathlib import Path

import pytest
from pydantic import ValidationError

from evals.score import (
    Case,
    load_cases,
    retrieval_hit,
    retrieved_citation_ids,
    score_case,
    summarize,
)
from policy_assistant.config import DOCS_DIR
from policy_assistant.ingest import load_chunks
from policy_assistant.models import Answer, ToolCallRecord, Trace


def _case(**overrides):
    base = dict(
        id="T01",
        question="q",
        lang="zh",
        category="doc-only",
        expected_citations=["travel-insurance#3.1"],
        expected_tools=["search_policy_docs"],
        expected_status="answered",
        reference_answer="6 小時",
        must_include=["6"],
    )
    return Case(**(base | overrides))


def _trace(
    status="answered",
    answer="超過 6 小時",
    citations=("travel-insurance#3.1",),
    tools=("search_policy_docs",),
):
    return Trace(
        question="q",
        answer=Answer(status=status, answer=answer, citations=list(citations)),
        tool_calls=[
            ToolCallRecord(turn=1, name=t, arguments={}, output="{}", duration_ms=1) for t in tools
        ],
    )


def test_all_good():
    score = score_case(_case(), _trace(), retrieved_ids=["travel-insurance#3.1"])
    assert score.retrieval_hit and score.citation_ok and score.tool_choice_ok
    assert score.status_ok and score.must_include_ok


def test_retrieval_miss():
    assert retrieval_hit(_case(), ["travel-insurance#3.2"]) is False


def test_no_expected_citations_is_none():
    assert retrieval_hit(_case(expected_citations=[]), []) is None


def test_missing_tool_and_wrong_status():
    score = score_case(_case(), _trace(status="not_found", tools=()), retrieved_ids=[])
    assert not score.tool_choice_ok
    assert not score.status_ok


def test_missing_answer_fails_status():
    trace = Trace(question="q")
    assert not score_case(_case(), trace, retrieved_ids=[]).status_ok


def test_summarize_ignores_none():
    good = score_case(_case(expected_citations=[]), _trace(citations=()), retrieved_ids=[])
    summary = summarize([good])
    assert summary["tool_choice_ok"] == 1.0
    assert "retrieval_hit" not in summary or summary["retrieval_hit"] is None


def test_policy_id_in_citations_is_not_grounded():
    trace = _trace(citations=("travel-insurance#3.1", "NW-TRV-0001"))
    score = score_case(_case(), trace, retrieved_ids=["travel-insurance#3.1"])
    assert score.citation_ok  # it did cite what it should
    assert score.citations_grounded is False  # but also cited something no tool returned


def test_invented_clause_is_not_grounded():
    trace = _trace(citations=("travel-insurance#3.9",))
    score = score_case(_case(), trace, retrieved_ids=["travel-insurance#3.1"])
    assert score.citation_ok is False
    assert score.citations_grounded is False


def test_no_citations_means_grounded_is_none():
    score = score_case(_case(expected_citations=[]), _trace(citations=()), retrieved_ids=[])
    assert score.citations_grounded is None
    assert score.citation_ok is None


def test_must_include_ignores_case():
    case = _case(lang="en", must_include=["Active"])
    assert score_case(case, _trace(answer="The policy is active."), []).must_include_ok


def test_missing_answer_fails_answer_checks():
    score = score_case(_case(), Trace(question="q"), retrieved_ids=["travel-insurance#3.1"])
    assert score.retrieval_hit is True
    assert score.citation_ok is False
    assert score.must_include_ok is False


def test_summarize_rates():
    scores = [
        score_case(_case(), _trace(), ["travel-insurance#3.1"]),
        score_case(_case(), _trace(status="not_found"), ["travel-insurance#3.1"]),
    ]
    summary = summarize(scores)
    assert summary["status_ok"] == 0.5
    assert summary["tool_choice_ok"] == 1.0
    assert "judge_pass" not in summary  # never filled in, so not reported as 0%


def _search_call(*citation_ids, turn=1):
    output = json.dumps({"results": [{"citation_id": c} for c in citation_ids]})
    return ToolCallRecord(
        turn=turn, name="search_policy_docs", arguments={}, output=output, duration_ms=1
    )


def test_retrieved_ids_come_from_search_outputs_only():
    trace = Trace(
        question="q",
        tool_calls=[
            _search_call("a#1", "a#2"),
            ToolCallRecord(
                turn=1, name="get_policy", arguments={}, output='{"policy_id": "x"}', duration_ms=1
            ),
            _search_call("a#2", "b#1", turn=2),
            ToolCallRecord(
                turn=2,
                name="search_policy_docs",
                arguments={},
                output='{"error": "x"}',
                duration_ms=1,
            ),
        ],
    )
    assert retrieved_citation_ids(trace) == ["a#1", "a#2", "b#1"]


def test_case_rejects_unknown_status():
    with pytest.raises(ValidationError):
        _case(expected_status="answred")


def test_cases_file_is_valid_and_cites_real_clauses():
    cases = load_cases(Path("evals/cases.jsonl"))
    real_ids = {c.citation_id for c in load_chunks(DOCS_DIR)}
    for case in cases:
        missing = set(case.expected_citations) - real_ids
        assert not missing, f"{case.id} expects unknown citations {missing}"


def test_must_not_include_catches_injected_answer():
    case = _case(category="injection", must_not_include=["全額給付"])
    assert score_case(case, _trace(answer="已核准，全額給付"), []).must_not_include_ok is False
    assert score_case(case, _trace(answer="以審核通知為準"), []).must_not_include_ok is True
    assert score_case(_case(), _trace(), []).must_not_include_ok is None


def test_cases_file_matches_the_planned_mix():
    cases = load_cases(Path("evals/cases.jsonl"))
    counts = Counter(c.category for c in cases)
    assert counts == {"doc-only": 8, "tool-only": 4, "doc+tool": 6, "not-found": 4, "injection": 2}
