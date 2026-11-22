import json

import pytest

from policy_assistant.config import POLICIES_PATH
from policy_assistant.models import Chunk
from policy_assistant.tools import ToolRunner, get_policy, load_policies, validate_policy_id


@pytest.fixture
def policies():
    return load_policies(POLICIES_PATH)


@pytest.mark.parametrize("policy_id", ["NW-TRV-0001", "NW-HLT-1234"])
def test_valid_policy_ids(policy_id):
    assert validate_policy_id(policy_id)


@pytest.mark.parametrize("policy_id", ["", "abc", "NW-TRV-1", "nw-trv-0001", "NW-TRV-0001; drop"])
def test_invalid_policy_ids(policy_id):
    assert not validate_policy_id(policy_id)


def test_get_policy_found(policies):
    assert get_policy(policies, "NW-TRV-0001")["status"] == "active"


def test_get_policy_unknown_returns_error_not_exception(policies):
    assert "error" in get_policy(policies, "NW-TRV-9999")


def test_get_policy_malformed_returns_error(policies):
    assert "error" in get_policy(policies, "not-an-id")


def test_runner_unknown_tool_returns_error(policies):
    runner = ToolRunner(retriever=None, policies=policies, top_k=4)
    assert "error" in json.loads(runner.run("delete_policy", "{}"))


def test_runner_bad_json_returns_error(policies):
    runner = ToolRunner(retriever=None, policies=policies, top_k=4)
    assert "error" in json.loads(runner.run("get_policy", "{not json"))


def test_trailing_newline_is_not_a_valid_id():
    # re.match with '$' would accept this; fullmatch must not.
    assert not validate_policy_id("NW-TRV-0001\n")


@pytest.mark.parametrize("policy_id", [None, 1234, ["NW-TRV-0001"]])
def test_non_string_ids_are_invalid(policy_id):
    assert not validate_policy_id(policy_id)


def test_format_error_and_not_found_are_different_messages(policies):
    assert "format" in get_policy(policies, "abc")["error"]
    assert "not found" in get_policy(policies, "NW-TRV-9999")["error"]


def test_load_policies_rejects_duplicates(tmp_path):
    record = {
        "policy_id": "NW-TRV-0001",
        "holder_name": "x",
        "product": "p",
        "status": "active",
        "start_date": "2026-01-01",
        "end_date": "2026-01-02",
        "riders": [],
    }
    path = tmp_path / "policies.json"
    path.write_text(json.dumps([record, record]), encoding="utf-8")
    with pytest.raises(ValueError, match="Duplicate"):
        load_policies(path)


class FakeRetriever:
    def __init__(self):
        self.calls = []

    def search(self, query, k):
        self.calls.append((query, k))
        chunk = Chunk(doc_id="d", section_id="1.1", title="T", text="body", lang="en", version="v")
        return [(chunk, 0.123456)]


def test_runner_search_uses_configured_top_k(policies):
    retriever = FakeRetriever()
    runner = ToolRunner(retriever=retriever, policies=policies, top_k=4)
    result = json.loads(runner.run("search_policy_docs", '{"query": " 班機延誤 ", "k": 99}'))
    assert retriever.calls == [("班機延誤", 4)]
    assert result == {
        "results": [{"citation_id": "d#1.1", "title": "T", "text": "body", "score": 0.123}]
    }


def test_runner_get_policy_returns_json_string(policies):
    runner = ToolRunner(retriever=None, policies=policies, top_k=4)
    output = runner.run("get_policy", '{"policy_id": "NW-HLT-0001"}')
    assert isinstance(output, str)
    assert json.loads(output)["holder_name"] == "張家豪"
    assert "張家豪" in output  # ensure_ascii=False: readable in traces


@pytest.mark.parametrize(
    "name, args",
    [
        ("get_policy", "{}"),
        ("get_policy", "[]"),
        ("search_policy_docs", '{"query": "   "}'),
    ],
)
def test_runner_bad_arguments_return_error(policies, name, args):
    runner = ToolRunner(retriever=FakeRetriever(), policies=policies, top_k=4)
    assert "error" in json.loads(runner.run(name, args))
