import json
from types import SimpleNamespace

from evals.judge import evidence_from_trace, judge
from policy_assistant.models import ToolCallRecord, Trace


class FakeClient:
    def __init__(self, output_text):
        self.calls = []
        self._text = output_text
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(output_text=self._text)


def test_returns_the_verdict():
    client = FakeClient(json.dumps({"passed": True, "reason": "Same facts."}))
    assert judge(client, "m", "q", "ref", "ans", "ev") == (True, "Same facts.")


def test_sends_all_three_parts_with_a_strict_schema():
    client = FakeClient(json.dumps({"passed": False, "reason": "x"}))
    judge(client, "judge-m", "Q?", "REF", "ANS", "EV")
    call = client.calls[0]
    assert call["model"] == "judge-m"
    for part in ("<question>\nQ?", "<reference>\nREF", "<evidence>\nEV", "<answer>\nANS"):
        assert part in call["input"]
    assert call["text"]["format"]["strict"] is True


def test_invalid_verdict_counts_as_fail_with_reason():
    passed, reason = judge(FakeClient("not json"), "m", "q", "ref", "ans", "ev")
    assert passed is False
    assert "not a valid verdict" in reason


def _call(name, output, turn=1):
    return ToolCallRecord(
        turn=turn, name=name, arguments={}, output=json.dumps(output), duration_ms=1
    )


def test_evidence_lists_clauses_and_policy_records_once():
    clause = {"citation_id": "claims-process#6.4", "title": "高額覆核", "text": "另加 5 個工作天"}
    trace = Trace(
        question="q",
        tool_calls=[
            _call("search_policy_docs", {"results": [clause]}),
            _call("get_policy", {"policy_id": "NW-TRV-0001", "status": "active"}),
            _call("search_policy_docs", {"results": [clause]}, turn=2),
            _call("get_policy", {"error": "Policy NW-TRV-9999 not found."}, turn=2),
        ],
    )
    evidence = evidence_from_trace(trace)
    assert evidence.count("[claims-process#6.4] 高額覆核\n另加 5 個工作天") == 1
    assert '"status": "active"' in evidence
    assert "not found" not in evidence  # errors are not evidence


def test_no_tool_calls_means_no_evidence():
    assert evidence_from_trace(Trace(question="q")) == "(no evidence)"
