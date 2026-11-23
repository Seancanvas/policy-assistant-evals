import json
from types import SimpleNamespace

from evals.judge import judge


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
    assert judge(client, "m", "q", "ref", "ans") == (True, "Same facts.")


def test_sends_all_three_parts_with_a_strict_schema():
    client = FakeClient(json.dumps({"passed": False, "reason": "x"}))
    judge(client, "judge-m", "Q?", "REF", "ANS")
    call = client.calls[0]
    assert call["model"] == "judge-m"
    for part in ("<question>\nQ?", "<reference>\nREF", "<answer>\nANS"):
        assert part in call["input"]
    assert call["text"]["format"]["strict"] is True


def test_invalid_verdict_counts_as_fail_with_reason():
    passed, reason = judge(FakeClient("not json"), "m", "q", "ref", "ans")
    assert passed is False
    assert "not a valid verdict" in reason
