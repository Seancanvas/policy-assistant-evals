import json
from datetime import datetime
from types import SimpleNamespace

from evals.rejudge import rejudge_report
from evals.run import CaseResult, RunReport
from evals.score import Case, CaseScore
from policy_assistant.models import Answer, Trace


class FakeJudgeClient:
    def __init__(self, verdicts):
        self.calls = []
        self._verdicts = list(verdicts)
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        passed, reason = self._verdicts.pop(0)
        return SimpleNamespace(output_text=json.dumps({"passed": passed, "reason": reason}))


def _result(case_id, status="answered", judge_pass=False):
    case = Case(
        id=case_id,
        question="q",
        lang="en",
        category="doc-only",
        expected_citations=[],
        expected_tools=[],
        expected_status="answered",
        reference_answer="r",
    )
    trace = Trace(question="q", answer=Answer(status=status, answer="a", citations=[]))
    score = CaseScore(
        case_id=case_id,
        retrieval_hit=None,
        citation_ok=None,
        citations_grounded=None,
        tool_choice_ok=True,
        status_ok=status == "answered",
        must_include_ok=True,
        must_not_include_ok=None,
        judge_pass=judge_pass,
        judge_reason="old reason",
    )
    return CaseResult(case=case, trace=trace, retrieved_ids=[], score=score)


def _report(results):
    return RunReport(
        label="baseline",
        created_at="2026-11-23T14:31:00",
        git_commit="0a9e090",
        model="m",
        judge_model="old-judge",
        embedding_model="e",
        top_k=4,
        max_turns=5,
        summary={"judge_pass": 0.0},
        stats={"latency_ms_mean": 1.0},
        results=results,
    )


def _rejudge(report, verdicts):
    client = FakeJudgeClient(verdicts)
    new = rejudge_report(
        report, client, "new-judge", "baseline-rejudged", "base.json", datetime(2026, 11, 23)
    )
    return new, client


def test_only_the_judge_fields_change():
    original = _report([_result("T01")])
    new, _ = _rejudge(original, [(True, "supported by evidence")])

    old_r, new_r = original.results[0], new.results[0]
    assert new_r.score.judge_pass is True
    assert new_r.score.judge_reason == "supported by evidence"
    assert new_r.trace == old_r.trace  # same answers
    assert new_r.score.model_dump(exclude={"judge_pass", "judge_reason"}) == old_r.score.model_dump(
        exclude={"judge_pass", "judge_reason"}
    )
    assert original.results[0].score.judge_pass is False  # the input report is not modified


def test_report_records_where_it_came_from_and_rescores():
    new, _ = _rejudge(_report([_result("T01"), _result("T02")]), [(True, "x"), (False, "y")])
    assert new.label == "baseline-rejudged"
    assert new.rejudged_from == "base.json"
    assert new.judge_model == "new-judge"
    assert new.summary["judge_pass"] == 0.5
    assert new.stats == {"latency_ms_mean": 1.0}  # agent stats are kept


def test_error_answers_are_not_sent_to_the_judge():
    new, client = _rejudge(_report([_result("T01", status="error")]), [])
    assert client.calls == []
    assert new.results[0].score.judge_reason == "No answer to judge."
