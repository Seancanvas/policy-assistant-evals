from evals.score import Case, retrieval_hit, score_case, summarize
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
