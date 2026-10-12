"""Lossless long-text transport, field roles, refs and live completion guards."""

import asyncio
import copy
import json

import httpx
import pytest
from mobilerun.agent.decision.autoglm import AutoGLMFallback, _ledger_evidence
from mobilerun.agent.decision.contracts import ContractError
from mobilerun.agent.decision.native_context import (
    ALIASES_FIELD,
    compact_native_values,
    expand_native_values,
)
from mobilerun.agent.decision.task_state import TaskState
from mobilerun.config_manager.config_manager import JevConfig
from test_decision_visual_guards import _image, _runtime, _ui
from test_task_evidence import review, task


@pytest.mark.parametrize(
    "body", ["正文\n\n原文必须保留。" * 100, "Text with punctuation.\n" * 80]
)
def test_long_duplicate_values_roundtrip_with_all_field_roles_and_identity(body):
    original = {
        "ref": "O3:n9",
        "index": 12,
        "path": [1, 3],
        "resourceId": "body",
        "text": body,
        "rawText": body,
        "inputValue": body,
        "checked": False,
        "maxTextLength": 0,
    }
    before = copy.deepcopy(original)
    compact = compact_native_values(original)
    assert compact[ALIASES_FIELD] == {"text": "rawText", "inputValue": "rawText"}
    assert expand_native_values(compact) == original
    assert original == before
    assert (
        len(json.dumps(compact, ensure_ascii=False))
        < len(json.dumps(original, ensure_ascii=False)) / 2
    )


def test_distinct_values_empty_input_and_short_values_are_retained():
    original = {
        "rawText": "a" * 100,
        "text": "b" * 100,
        "inputValue": "",
        "contentDescription": "Title",
        "stateDescription": "Title",
    }
    assert compact_native_values(original) == original
    assert expand_native_values(compact_native_values(original)) == original


@pytest.mark.parametrize(
    "node",
    [
        {"rawText": "a", ALIASES_FIELD: {}},
        {"rawText": "a", ALIASES_FIELD: []},
        {"rawText": "a", ALIASES_FIELD: {"inputValue": "O2:n0"}},
        {
            "rawText": "a",
            ALIASES_FIELD: {"text": "inputValue", "inputValue": "rawText"},
        },
        {"rawText": "a", "text": "b", ALIASES_FIELD: {"text": "rawText"}},
        {"rawText": "a", ALIASES_FIELD: {"ref": "rawText"}},
        {"rawText": 1, ALIASES_FIELD: {"text": "rawText"}},
        {"rawText": "", ALIASES_FIELD: {"text": "rawText"}},
        {"rawText": "a", ALIASES_FIELD: {"text": None}},
        {"rawText": "a", ALIASES_FIELD: {"text": ["rawText"]}},
    ],
)
def test_aliases_cannot_override_identity_facts_chain_or_cross_nodes(node):
    with pytest.raises(ContractError, match="context_alias_invalid"):
        expand_native_values(node)


def test_reserved_field_cannot_be_silently_reencoded():
    with pytest.raises(ContractError, match="context_alias_invalid"):
        compact_native_values({"rawText": "a", ALIASES_FIELD: {"text": "rawText"}})


def test_projection_retains_long_native_value_under_same_budget_without_mutating_ledger():
    body = "A long captured paragraph, including all punctuation.\n" * 18
    state = TaskState()
    state.observe(
        _ui([{"text": body, "rawText": body, "inputValue": body, "resourceId": "body"}])
    )
    before = copy.deepcopy(state.observations)
    ordinary = state.context(max_chars=2500)
    shared = state.context(max_chars=2500, share_native_values=True)
    assert ordinary["observations"][0]["nodes_budget_omitted"] == 1
    assert shared["observations"][0]["nodes_budget_omitted"] == 0
    node = shared["observations"][0]["nodes"][0]
    assert node["rawText"] == body
    restored = _ledger_evidence(shared)
    assert restored[node["ref"]]["inputValue"] == body
    assert restored["O1"]["nodes"][0]["inputValue"] == body
    assert state.observations == before
    assert state.evidence()[node["ref"]]["inputValue"] == body


def test_omission_counts_remain_explicit_when_unique_values_exceed_budget():
    state = TaskState()
    body = "unique long text " * 1000
    state.observe(_ui([{"text": body, "rawText": body, "inputValue": body}]))
    shared = state.context(max_chars=2500, share_native_values=True)
    assert shared["observations"][0]["nodes"] == []
    assert shared["observations"][0]["nodes_budget_omitted"] == 1


def test_only_capable_glm_adapter_receives_shared_values_and_jev_retains_ordinary_fields():
    body = "captured value " * 40
    adapter = AutoGLMFallback(
        model="glm-5.3-flash",
        reasoning_model="glm-5.3-flash",
        base_url="https://example.invalid/v1",
        api_key_env="GLM_API_KEY",
    )
    runner, _, provider = _runtime(
        _ui([{"text": body, "rawText": body, "inputValue": body}]), visual=adapter
    )
    runner.task_state.observe(provider.ui)
    runner.task_state.set_contract(task("Keep the saved record visible", "final_state"))
    runner.config.share_native_values = True
    shared = runner.task_context(purpose="review")["ledger"]["observations"][0][
        "nodes"
    ][0]
    assert ALIASES_FIELD in shared
    ordinary = runner.task_context(purpose="jev")["ledger"]["observations"][0]["nodes"][
        0
    ]
    assert ALIASES_FIELD not in ordinary
    runner.visual_fallback = object()
    ordinary = runner.task_context(purpose="review")["ledger"]["observations"][0][
        "nodes"
    ][0]
    assert ALIASES_FIELD not in ordinary


def test_equal_values_in_different_objects_do_not_merge_identity_or_refs():
    body = "The same full text appears in separate records. " * 60
    state = TaskState()
    state.observe(
        _ui(
            [
                {
                    "text": body,
                    "rawText": body,
                    "inputValue": body,
                    "resourceId": "first",
                },
                {
                    "text": body,
                    "rawText": body,
                    "inputValue": body,
                    "resourceId": "second",
                },
            ]
        )
    )
    ledger = state.context(share_native_values=True)
    nodes = ledger["observations"][0]["nodes"]
    assert len(nodes) == 2
    restored = _ledger_evidence(ledger)
    assert restored["O1:n0"]["resourceId"] == "first"
    assert restored["O1:n1"]["resourceId"] == "second"
    assert restored["O1:n0"]["inputValue"] == restored["O1:n1"]["inputValue"] == body


@pytest.mark.parametrize("value", [1, None, "true"])
def test_shared_values_configuration_requires_boolean(value):
    with pytest.raises(ValueError, match="share_native_values"):
        JevConfig(share_native_values=value)


@pytest.mark.parametrize("changed", ["pixels", "property"])
def test_one_http_review_expands_values_and_still_rejects_live_changes(
    monkeypatch, changed
):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-key")
    requests, holders = [], []
    body = "A captured record that must remain visible. " * 70
    goal = "Keep the saved record visible"
    contract = task(goal, "final_state")

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        context = json.loads(payload["messages"][-1]["content"][0]["text"])
        observed = context["task_state"]["ledger"]["observations"][-1]
        assert ALIASES_FIELD in observed["nodes"][0]
        current = holders[0][0].task_state.current_id
        refs = [current + ":n0", context["screenshot_evidence"]["ref"]]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": review(refs, answer=None)}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = AutoGLMFallback(
                model="glm-5.3-flash",
                reasoning_model="glm-5.3-flash",
                base_url="https://example.invalid/v1",
                api_key_env="GLM_API_KEY",
                client=client,
            )
            adapter._goal, adapter._completion_contract = goal, contract
            runner, driver, provider = _runtime(
                _ui(
                    [
                        {
                            "text": body,
                            "rawText": body,
                            "inputValue": body,
                            "resourceId": "status",
                        }
                    ]
                ),
                visual=adapter,
            )
            runner.config.share_native_values = True
            runner.config.derive_review_dependencies = True
            runner.goal = goal
            runner.task_state.set_contract(contract)
            holders.append((runner, driver, provider))
            selected = await runner._visual(provider.ui, completion_review=True)
            assert selected.calls[0].name == "complete"
            assert selected.completion_proofs
            if changed == "pixels":
                driver.image = _image("blue")
            else:
                provider.ui = _ui([{"text": "Changed", "resourceId": "status"}])
            _, reason = await runner.validate_before_execute()
            assert reason == "completion_unverified"
            assert runner.history == []

    asyncio.run(run())
    assert len(requests) == 1
