"""Agent loop tests with a scripted fake client: no network, no cost."""

import json
from types import SimpleNamespace

from policy_assistant.agent import run_agent


def _usage():
    return SimpleNamespace(input_tokens=10, output_tokens=5)


def _tool_call_response(name, arguments, call_id="call_1"):
    call = SimpleNamespace(
        type="function_call", name=name, arguments=json.dumps(arguments), call_id=call_id
    )
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
    client = FakeClient(
        [
            _tool_call_response("get_policy", {"policy_id": "NW-TRV-0001"}),
            _final_response(answer="Active."),
        ]
    )
    runner = FakeRunner()
    trace = run_agent("Is NW-TRV-0001 active?", client, runner, model="m", max_turns=5)

    assert runner.calls == [("get_policy", {"policy_id": "NW-TRV-0001"})]
    second_input = client.calls[1]["input"]
    outputs = [
        i for i in second_input if isinstance(i, dict) and i.get("type") == "function_call_output"
    ]
    assert outputs and outputs[0]["call_id"] == "call_1"
    assert trace.answer.status == "answered"
    assert len(trace.tool_calls) == 1
    assert trace.input_tokens == 20


def test_turn_limit_gives_honest_error():
    client = FakeClient([_tool_call_response("get_policy", {"policy_id": "NW-TRV-0001"})] * 3)
    trace = run_agent("loop forever", client, FakeRunner(), model="m", max_turns=3)
    assert trace.answer.status == "error"


def test_invalid_final_answer_gives_error():
    bad = SimpleNamespace(
        output=[SimpleNamespace(type="message")], output_text="not json", usage=_usage()
    )
    trace = run_agent("q", FakeClient([bad]), FakeRunner(), model="m", max_turns=3)
    assert trace.answer.status == "error"


def test_sends_the_real_question_and_answer_schema():
    client = FakeClient([_final_response()])
    run_agent("NW-HLT-0001 可以理賠嗎？", client, FakeRunner(), model="m", max_turns=3)

    first = client.calls[0]
    assert first["input"][0] == {"role": "user", "content": "NW-HLT-0001 可以理賠嗎？"}
    fmt = first["text"]["format"]
    assert fmt["type"] == "json_schema" and fmt["strict"] is True
    assert fmt["schema"]["additionalProperties"] is False


def test_each_parallel_call_gets_its_own_output():
    calls = [
        SimpleNamespace(
            type="function_call",
            name="get_policy",
            arguments=json.dumps({"policy_id": "NW-TRV-0001"}),
            call_id="call_a",
        ),
        SimpleNamespace(
            type="function_call",
            name="search_policy_docs",
            arguments=json.dumps({"query": "overseas medical"}),
            call_id="call_b",
        ),
    ]
    first = SimpleNamespace(output=calls, output_text="", usage=_usage())
    client = FakeClient([first, _final_response()])
    trace = run_agent("q", client, FakeRunner(), model="m", max_turns=5)

    second_input = client.calls[1]["input"]
    outputs = [
        i["call_id"]
        for i in second_input
        if isinstance(i, dict) and i.get("type") == "function_call_output"
    ]
    assert outputs == ["call_a", "call_b"]
    # The model's own function_call items come before their outputs.
    assert second_input[1:3] == calls
    assert [r.name for r in trace.tool_calls] == ["get_policy", "search_policy_docs"]


def test_bad_tool_arguments_are_recorded_not_raised():
    call = SimpleNamespace(
        type="function_call", name="get_policy", arguments="{oops", call_id="call_1"
    )
    first = SimpleNamespace(output=[call], output_text="", usage=_usage())

    class ErrorRunner:
        def run(self, name, arguments_json):
            return json.dumps({"error": "Invalid JSON arguments."})

    trace = run_agent("q", FakeClient([first, _final_response()]), ErrorRunner(), "m", 5)
    assert trace.tool_calls[0].arguments == {"_raw": "{oops"}
    assert trace.answer.status == "answered"


def test_turn_limit_stops_calling_the_model():
    client = FakeClient([_tool_call_response("get_policy", {"policy_id": "NW-TRV-0001"})] * 3)
    trace = run_agent("q", client, FakeRunner(), model="m", max_turns=3)
    assert len(client.calls) == 3
    assert trace.turns == 3
    assert "3 turns" in trace.answer.answer
    assert trace.answer.citations == []


def test_answer_with_extra_field_is_rejected():
    text = json.dumps({"status": "answered", "answer": "ok", "citations": [], "note": "x"})
    bad = SimpleNamespace(
        output=[SimpleNamespace(type="message")], output_text=text, usage=_usage()
    )
    trace = run_agent("q", FakeClient([bad]), FakeRunner(), model="m", max_turns=3)
    assert trace.answer.status == "error"


def test_tool_calls_record_their_turn():
    parallel = [
        SimpleNamespace(
            type="function_call",
            name="get_policy",
            arguments=json.dumps({"policy_id": "NW-TRV-0003"}),
            call_id=f"call_{i}",
        )
        for i in ("a", "b")
    ]
    client = FakeClient(
        [
            SimpleNamespace(output=parallel, output_text="", usage=_usage()),
            _tool_call_response("search_policy_docs", {"query": "rider"}, call_id="call_c"),
            _final_response(),
        ]
    )
    trace = run_agent("q", client, FakeRunner(), model="m", max_turns=5)
    assert [r.turn for r in trace.tool_calls] == [1, 1, 2]
    assert trace.turns == 3


def test_instructions_ask_for_bilingual_search_queries():
    # Guards the cross-lingual retrieval fix; the eval measures whether it works.
    client = FakeClient([_final_response()])
    run_agent("q", client, FakeRunner(), model="m", max_turns=3)
    assert "in both languages" in client.calls[0]["instructions"]
