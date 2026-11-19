"""Agent loop tests with a scripted fake client: no network, no cost."""

import json
from types import SimpleNamespace

from policy_assistant.agent import run_agent


def _usage():
    return SimpleNamespace(input_tokens=10, output_tokens=5)


def _tool_call_response(name, arguments, call_id="call_1"):
    call = SimpleNamespace(type="function_call", name=name, arguments=json.dumps(arguments),
                           call_id=call_id)
    return SimpleNamespace(output=[call], output_text="", usage=_usage())


def _final_response(status="answered", answer="ok", citations=()):
    text = json.dumps({"status": status, "answer": answer, "citations": list(citations)})
    message = SimpleNamespace(type="message")
    return SimpleNamespace(output=[message], output_text=text, usage=_usage())


class FakeClient:
    def __init__(self, scripted):
        self.calls = []
        self._scripted = list(scripted)
        self.responses = SimpleNamespace(create=self._create)

    def _create(self, **kwargs):
        self.calls.append(kwargs)
        return self._scripted.pop(0)


class FakeRunner:
    def __init__(self):
        self.calls = []

    def run(self, name, arguments_json):
        self.calls.append((name, json.loads(arguments_json)))
        return json.dumps({"status": "active"})


def test_runs_tool_and_sends_output_back():
    client = FakeClient([
        _tool_call_response("get_policy", {"policy_id": "NW-TRV-0001"}),
        _final_response(answer="Active."),
    ])
    runner = FakeRunner()
    trace = run_agent("Is NW-TRV-0001 active?", client, runner, model="m", max_turns=5)

    assert runner.calls == [("get_policy", {"policy_id": "NW-TRV-0001"})]
    second_input = client.calls[1]["input"]
    outputs = [i for i in second_input if isinstance(i, dict)
               and i.get("type") == "function_call_output"]
    assert outputs and outputs[0]["call_id"] == "call_1"
    assert trace.answer.status == "answered"
    assert len(trace.tool_calls) == 1
    assert trace.input_tokens == 20


def test_turn_limit_gives_honest_error():
    client = FakeClient([_tool_call_response("get_policy", {"policy_id": "NW-TRV-0001"})] * 3)
    trace = run_agent("loop forever", client, FakeRunner(), model="m", max_turns=3)
    assert trace.answer.status == "error"


def test_invalid_final_answer_gives_error():
    bad = SimpleNamespace(output=[SimpleNamespace(type="message")], output_text="not json",
                          usage=_usage())
    trace = run_agent("q", FakeClient([bad]), FakeRunner(), model="m", max_turns=3)
    assert trace.answer.status == "error"
