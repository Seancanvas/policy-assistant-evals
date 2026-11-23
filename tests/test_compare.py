from evals.compare import (
    Tally,
    case_changes,
    case_tallies,
    config_changes,
    format_comparison,
    load_run,
)
from evals.run import CaseResult, RunReport
from evals.score import Case, CaseScore
from policy_assistant.models import Trace


def _result(case_id, attempt=1, judge=True, retrieval=None, reason=""):
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
    score = CaseScore(
        case_id=case_id,
        retrieval_hit=retrieval,
        citation_ok=None,
        citations_grounded=None,
        tool_choice_ok=True,
        status_ok=True,
        must_include_ok=True,
        must_not_include_ok=None,
        judge_pass=judge,
        judge_reason=reason,
    )
    return CaseResult(
        case=case, trace=Trace(question="q"), retrieved_ids=[], score=score, attempt=attempt
    )


def _report(results, summary=None, **config):
    base = dict(
        label="x",
        created_at="2026-11-23T00:00:00",
        git_commit="aaa",
        model="m",
        judge_model="m",
        embedding_model="e",
        top_k=4,
        max_turns=5,
        summary=summary or {},
        stats={},
        results=results,
    )
    return RunReport(**(base | config))


def test_tallies_add_up_repeats_and_skip_none():
    report = _report([_result("T01", 1, judge=True), _result("T01", 2, judge=False)])
    tallies = case_tallies(report)["T01"]
    assert tallies["judge_pass"] == Tally(1, 2)
    assert "retrieval_hit" not in tallies  # None in every run


def test_changes_split_worse_and_better_with_reason():
    before = _report([_result("T01", judge=True), _result("T02", retrieval=False)])
    after = _report(
        [_result("T01", judge=False, reason="contradicts"), _result("T02", retrieval=True)]
    )
    changes = {(c.case_id, c.metric): c for c in case_changes(before, after)}
    worse = changes[("T01", "judge_pass")]
    assert worse.after.rate < worse.before.rate and worse.reason == "contradicts"
    assert changes[("T02", "retrieval_hit")].after == Tally(1, 1)
    assert len(changes) == 2  # unchanged metrics are not listed


def test_repeated_runs_compare_by_rate():
    before = _report([_result("T05", 1, retrieval=True), _result("T05", 2, retrieval=False)])
    after = _report([_result("T05", i, retrieval=True) for i in range(1, 6)])
    (change,) = case_changes(before, after)
    assert (str(change.before), str(change.after)) == ("1/2", "5/5")


def test_config_changes_show_only_what_differs():
    before = _report([], git_commit="aaa", top_k=4)
    after = _report([], git_commit="bbb", top_k=4, repeat=5)
    assert config_changes(before, after) == [("git_commit", "aaa", "bbb"), ("repeat", 1, 5)]


def test_format_lists_worse_before_better_and_missing_cases():
    before = _report(
        [_result("T01", judge=True), _result("T02", judge=False), _result("T03")],
        summary={"judge_pass": 0.5},
    )
    after = _report(
        [_result("T01", judge=False), _result("T02", judge=True)], summary={"judge_pass": 0.5}
    )
    text = format_comparison(before, after)
    assert text.index("Worse (pass -> fail): 1") < text.index("Better (fail -> pass): 1")
    assert "Only in before: T03" in text
    assert "+0.0pp" in text
    assert "(none)" in text  # no setup change


def test_load_run_round_trips(tmp_path):
    report = _report([_result("T01")], summary={"judge_pass": 1.0})
    path = tmp_path / "run.json"
    path.write_text(report.model_dump_json(), encoding="utf-8")
    assert load_run(path) == report
