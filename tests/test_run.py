import json
from datetime import datetime
from types import SimpleNamespace

import openai
import pytest

from evals.run import (
    COLUMNS,
    choose_cases,
    evaluate_case,
    format_table,
    output_path,
    run_all,
    run_stats,
    save_report,
)
from evals.score import METRICS, Case, summarize


def _usage():
    return SimpleNamespace(input_tokens=10, output_tokens=5)


def _tool_call(name, arguments, call_id="call_1"):
    call = SimpleNamespace(
        type="function_call", name=name, arguments=json.dumps(arguments), call_id=call_id
    )
    return SimpleNamespace(output=[call], output_text="", usage=_usage())


def _final(text):
    return SimpleNamespace(
        output=[SimpleNamespace(type="message")], output_text=text, usage=_usage()
    )


def _answer(status="answered", answer="Active.", citations=()):
    return _final(json.dumps({"status": status, "answer": answer, "citations": list(citations)}))


def _verdict(passed=True, reason="ok"):
    return SimpleNamespace(output_text=json.dumps({"passed": passed, "reason": reason}))


def _api_error():
    # Built without a request object, so the test does not depend on the SDK's HTTP library.
    error = openai.APIConnectionError.__new__(openai.APIConnectionError)
    Exception.__init__(error, "Connection error.")
    return error


class FakeClient:
    """Answers agent and judge calls from one script, in order. Raises if called too often."""

    def __init__(self, scripted):
        self.calls = []
        self._scripted = list(scripted)
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        item = self._scripted.pop(0)
        if isinstance(item, Exception):
            raise item
        return item


class FakeRunner:
    def run(self, name, arguments_json):
        if name == "search_policy_docs":
            return json.dumps({"results": [{"citation_id": "travel-insurance#3.1"}]})
        return json.dumps({"status": "active"})


def _case(**overrides):
    base = dict(
        id="T01",
        question="q",
        lang="en",
        category="tool-only",
        expected_citations=[],
        expected_tools=["get_policy"],
        expected_status="answered",
        reference_answer="Active.",
    )
    return Case(**(base | overrides))


def _evaluate(case, scripted):
    client = FakeClient(scripted)
    result = evaluate_case(case, client, FakeRunner(), "m", max_turns=5, judge_model="j")
    return result, client


def test_runs_scores_and_judges_one_case():
    result, client = _evaluate(
        _case(),
        [_tool_call("get_policy", {"policy_id": "NW-TRV-0001"}), _answer(), _verdict()],
    )
    assert result.score.tool_choice_ok and result.score.status_ok
    assert result.score.judge_pass is True
    assert client.calls[-1]["model"] == "j"  # the judge used the judge model


def test_retrieved_ids_come_from_the_trace():
    case = _case(
        category="doc-only",
        expected_citations=["travel-insurance#3.1"],
        expected_tools=["search_policy_docs"],
    )
    result, _ = _evaluate(
        case,
        [
            _tool_call("search_policy_docs", {"query": "delay"}),
            _answer(citations=["travel-insurance#3.1"]),
            _verdict(),
        ],
    )
    assert result.retrieved_ids == ["travel-insurance#3.1"]
    assert result.score.retrieval_hit and result.score.citation_ok
    assert result.score.citations_grounded


def test_error_answer_is_not_sent_to_the_judge():
    # Only one scripted response: calling the judge would pop from an empty list.
    result, client = _evaluate(_case(), [_final("not json")])
    assert len(client.calls) == 1
    assert result.score.judge_pass is False
    assert result.score.judge_reason == "No answer to judge."


def test_api_error_fails_the_case_not_the_run():
    error = _api_error()
    result, _ = _evaluate(_case(), [error])
    assert result.trace.answer.status == "error"
    assert result.trace.answer.answer.startswith("API error")
    assert result.score.status_ok is False


def test_judge_api_error_leaves_judge_pass_unknown():
    error = _api_error()
    result, _ = _evaluate(_case(), [_answer(), error])
    assert result.score.judge_pass is None  # unknown, so it does not count as a fail
    assert result.score.judge_reason.startswith("Judge API error")


def test_choose_cases_keeps_file_order_and_rejects_unknown_ids():
    cases = [_case(id="T01"), _case(id="T02"), _case(id="T03")]
    assert [c.id for c in choose_cases(cases, ["T03", "T01"])] == ["T01", "T03"]
    assert choose_cases(cases, None) == cases
    with pytest.raises(ValueError, match="T99"):
        choose_cases(cases, ["T99"])


def test_run_stats():
    a, _ = _evaluate(_case(), [_answer(), _verdict()])
    b, _ = _evaluate(_case(), [_tool_call("get_policy", {}), _answer(), _verdict()])
    stats = run_stats([a, b])
    assert stats["turns_mean"] == 1.5
    assert stats["tool_calls_mean"] == 0.5
    assert stats["input_tokens_total"] == 30


def test_table_covers_every_metric():
    assert set(COLUMNS.values()) == set(METRICS)


def test_format_table_marks_none_with_dash():
    result, _ = _evaluate(_case(), [_answer(), _verdict()])
    table = format_table([result], summarize([result.score]))
    row = table.splitlines()[1].split()
    assert row[:2] == ["T01", "tool-only"]
    assert row[2] == "-"  # retrieval_hit: the case expects no citations
    assert "100%" in table.splitlines()[-1]


def test_output_path_has_time_and_label():
    path = output_path("baseline", datetime(2026, 11, 23, 9, 5))
    assert path.name == "2026-11-23_0905_baseline.json"


def test_run_all_repeats_each_case_and_numbers_attempts():
    seen = []

    def evaluate(case):
        seen.append(case.id)
        result, _ = _evaluate(case, [_answer(), _verdict()])
        return result

    results = run_all([_case(id="T01"), _case(id="T02")], repeat=3, evaluate=evaluate)
    assert seen == ["T01"] * 3 + ["T02"] * 3
    assert [r.attempt for r in results] == [1, 2, 3, 1, 2, 3]


def test_table_shows_pass_counts_for_repeated_runs():
    passed, _ = _evaluate(_case(), [_answer(), _verdict(passed=True)])
    failed, _ = _evaluate(_case(), [_answer(), _verdict(passed=False)])
    table = format_table([passed, failed], summarize([passed.score, failed.score]))
    lines = table.splitlines()
    assert len(lines) == 3  # header, one row for T01, rate
    row = lines[1].split()
    assert row[-1] == "1/2"  # judge
    assert row[2] == "-"  # retrieval_hit is None in both runs
    assert "50%" in lines[-1]


def test_save_report_never_overwrites(tmp_path):
    path = tmp_path / "runs" / "x.json"
    save_report(path, "first")
    with pytest.raises(FileExistsError):
        save_report(path, "second")
    assert path.read_text(encoding="utf-8") == "first"
