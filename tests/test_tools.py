import json

import pytest

from policy_assistant.config import POLICIES_PATH
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
