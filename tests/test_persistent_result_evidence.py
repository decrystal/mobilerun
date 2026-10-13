"""Saved results need observations beyond an uncommitted input value."""

import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from test_decision_visual_guards import _image, _runtime
from test_task_evidence import review, task, ui

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.decision.contracts import (
    ContractError,
    parse_grounded_review,
    parse_task_contract,
)
from mobilerun.agent.decision.runtime import DecisionRuntime
from mobilerun.agent.decision.task_state import TaskState
from mobilerun.agent.decision.visual_fallback import VisualFallback
from mobilerun.agent.fast_agent.xml_parser import ToolCall
from mobilerun.config_manager.config_manager import JevConfig

GOAL = "Save the requested record"


def contract(kind="state", scope="persistent"):
    result = task(GOAL, kind)
    result["requirements"][0]["effect_scope"] = scope
    return result


def field(value="Written", **extra):
    return {
        "index": 1,
        "className": "EditText",
        "resourceId": "body",
        "inputValue": value,
        "rawText": value,
        "text": value,
        "bounds": "0,0,400,100",
        "editable": True,
        **extra,
    }


def assessed(
    state,
    refs,
    *,
    results=None,
    trace=None,
    kind="state",
    scope="persistent",
    complete=True,
    status="supported",
    uploaded=None,
):
    data = json.loads(review(refs, answer=None, complete=complete, status=status))
    if results is not None:
        data["assessments"][0]["result_refs"] = results
    if trace is not None:
        data["assessments"][0]["reopen_trace"] = trace
    return parse_grounded_review(
        json.dumps(data),
        contract(kind, scope),
        state.evidence(),
        state.current_id,
        uploaded_images=uploaded or {},
        derive_review_dependencies=True,
    )


@pytest.mark.parametrize(
    "node",
    [
        field(),
        {"index": 1, "className": "TextField", "rawText": "Written"},
        {"index": 1, "className": "View", "inputValue": "Written"},
        {"index": 1, "className": "View", "isEditable": True, "rawText": "Written"},
    ],
)
def test_editor_only_evidence_cannot_certify_a_persistent_result(node):
    state = TaskState()
    state.observe(ui([node]))
    with pytest.raises(ContractError) as error:
        assessed(state, ["O1:n0"])
    assert (
        error.value.code == "effect_unverified"
        and error.value.field == "persistent_result"
    )


@pytest.mark.parametrize("kind", ["state", "collection", "artifact"])
@pytest.mark.parametrize("scope", ["visible", "persistent"])
def test_typed_result_scope_is_retained_from_the_goal_contract(kind, scope):
    data = {
        "requirements": [
            {
                "id": 1,
                "description": GOAL,
                "quote": GOAL,
                "kind": kind,
                "effect_scope": scope,
            }
        ],
        "answer_required": False,
    }
    parsed = parse_task_contract(json.dumps(data), GOAL, require_effect_scopes=True)
    assert parsed["requirements"][0]["effect_scope"] == scope


@pytest.mark.parametrize("kind", ["state", "collection", "artifact"])
def test_enabled_goal_compiler_cannot_silently_omit_result_scope(kind):
    data = {
        "requirements": [{"id": 1, "description": GOAL, "quote": GOAL, "kind": kind}],
        "answer_required": False,
    }
    with pytest.raises(ContractError) as error:
        parse_task_contract(json.dumps(data), GOAL, require_effect_scopes=True)
    assert (
        error.value.code == "contract_invalid" and error.value.field == "effect_scope"
    )


@pytest.mark.parametrize("scope", [None, True, "saved", {}, ["persistent"]])
def test_invalid_result_scope_is_not_a_model_escape_hatch(scope):
    data = {
        "requirements": [
            {
                "id": 1,
                "description": GOAL,
                "quote": GOAL,
                "kind": "state",
                "effect_scope": scope,
            }
        ],
        "answer_required": False,
    }
    with pytest.raises(ContractError):
        parse_task_contract(json.dumps(data), GOAL, require_effect_scopes=True)


@pytest.mark.parametrize("kind", ["query", "process", "final_state"])
def test_result_scope_cannot_be_added_to_a_query_or_process_or_final_position(kind):
    data = {
        "requirements": [
            {
                "id": 1,
                "description": GOAL,
                "quote": GOAL,
                "kind": kind,
                "effect_scope": "persistent",
            }
        ],
        "answer_required": kind == "query",
    }
    with pytest.raises(ContractError):
        parse_task_contract(json.dumps(data), GOAL, require_effect_scopes=True)


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_effect_scope_switch_is_a_strict_boolean(value):
    with pytest.raises(ValueError, match="effect_scopes"):
        JevConfig(effect_scopes=value)
    with pytest.raises(ValueError, match="effect_scopes"):
        VisualFallback(effect_scopes=value)


@pytest.mark.parametrize("kind", ["state", "collection", "artifact"])
def test_a_specific_saved_result_can_be_reviewed_after_leaving_its_page(kind):
    state = TaskState()
    state.observe(ui([field()]))
    state.record_action(
        ToolCall("click", {"index": 2}), ActionResult(True, "Dispatched")
    )
    state.observe(
        ui(
            [
                {
                    "index": 2,
                    "className": "TextView",
                    "rawText": "Saved record",
                    "resourceId": "record",
                }
            ]
        )
    )
    state.record_action(
        ToolCall("system_button", {"button": "Back"}), ActionResult(True, "Dispatched")
    )
    state.observe(ui([{"rawText": "Another page"}]))
    parsed = assessed(state, ["A1", "A2", "O2:n0"], results=["O2:n0"], kind=kind)
    assert parsed["status"] == "done" and parsed["assessments"][0]["result_refs"] == [
        "O2:n0"
    ]


def test_a_current_save_indicator_can_support_persistence_without_forcing_reopen():
    state = TaskState()
    state.observe(
        ui([field(), {"index": 2, "rawText": "Saved", "className": "TextView"}])
    )
    assert assessed(state, ["O1:n0", "O1:n1"], results=["O1:n1"])["status"] == "done"


def test_editor_descendants_are_not_read_only_saved_result_controls():
    state = TaskState()
    node = field()
    node["children"] = [{"index": 2, "className": "TextView", "rawText": "Written"}]
    state.observe(ui([node]))
    with pytest.raises(ContractError, match="effect_unverified"):
        assessed(state, ["O1:n1"], results=["O1:n1"])


@pytest.mark.parametrize("uploaded", [False, True])
def test_pixel_result_needs_an_image_actually_uploaded_to_this_review(uploaded):
    state = TaskState()
    state.observe(ui(), screenshot=_image("green"))
    evidence = state.evidence()
    images = {"O1:image1": evidence["O1:image1"]["sha256"]} if uploaded else {}
    if uploaded:
        assert (
            assessed(state, ["O1:image1"], results=["O1:image1"], uploaded=images)[
                "status"
            ]
            == "done"
        )
    else:
        with pytest.raises(ContractError, match="effect_unverified"):
            assessed(state, ["O1:image1"], results=["O1:image1"])


def reopened(value="First line\n第二行 🌿"):
    state = TaskState()
    state.observe(ui([field(value)]))
    state.record_action(
        ToolCall("system_button", {"button": "Back"}), ActionResult(True, "Dispatched")
    )
    state.observe(ui([{"index": 2, "className": "TextView", "rawText": "Record list"}]))
    state.record_action(
        ToolCall("click", {"index": 2}), ActionResult(True, "Dispatched")
    )
    state.observe(ui([field(value)]))
    refs = ["O1", "A1", "O2", "A2", "O3", "O3:n0"]
    trace = {"before": "O1", "away": "O2", "returned": "O3", "actions": ["A1", "A2"]}
    return state, refs, trace


def test_reopened_editor_proves_an_exact_reread_through_an_observed_navigation_gap():
    state, refs, trace = reopened()
    assert assessed(state, refs, results=["O3:n0"], trace=trace)["status"] == "done"


@pytest.mark.parametrize(
    "change",
    [
        "different_value",
        "same_field_away",
        "unknown_dispatch",
        "missing_action",
        "retype",
        "wrong_package",
        "ambiguous_field",
        "no_resource",
        "wrong_order",
    ],
)
def test_reopen_trace_does_not_turn_unknown_or_retyped_content_into_saved_evidence(
    change,
):
    state, refs, trace = reopened()
    if change == "different_value":
        state.observations[-1]["nodes"][0]["inputValue"] = "Different content"
    elif change == "same_field_away":
        state.observations[1]["nodes"] = copy.deepcopy(state.observations[0]["nodes"])
    elif change == "unknown_dispatch":
        state.actions[0]["dispatch_status"] = "unknown"
    elif change == "missing_action":
        state.actions[1]["ref"] = "A3"
        refs[3], trace["actions"][1] = "A3", "A3"
    elif change == "retype":
        state.actions[1]["tool"] = "type"
    elif change == "wrong_package":
        state.observations[-1]["phone"]["packageName"] = "com.other.records"
    elif change == "ambiguous_field":
        duplicate = copy.deepcopy(state.observations[0]["nodes"][0])
        duplicate.update(path=1, ref="O1:n1")
        state.observations[0]["nodes"].append(duplicate)
    elif change == "no_resource":
        for observation in (state.observations[0], state.observations[-1]):
            observation["nodes"][0]["resourceId"] = ""
    else:
        trace["before"], trace["away"] = trace["away"], trace["before"]
    with pytest.raises(ContractError, match="effect_unverified"):
        assessed(state, refs, results=["O3:n0"], trace=trace)


def test_result_sources_must_be_part_of_that_requirement_evidence():
    state = TaskState()
    state.observe(ui([{"rawText": "Saved"}, {"rawText": "Other"}]))
    with pytest.raises(ContractError, match="evidence_unknown"):
        assessed(state, ["O1:n0"], results=["O1:n1"])


def test_persistent_result_cannot_use_a_pre_action_witness_plus_an_uncommitted_post_action_input():
    state = TaskState()
    state.observe(ui([{"rawText": "Old saved record"}]))
    state.record_action(
        ToolCall("type", {"text": "Written", "clear": True}),
        ActionResult(True, "Dispatched"),
    )
    state.observe(ui([field()]))
    with pytest.raises(ContractError, match="effect_unverified"):
        assessed(state, ["O1:n0", "A1", "O2:n0"], results=["O1:n0"])


def test_visible_input_requirement_keeps_the_original_input_value_evidence():
    state = TaskState()
    state.observe(ui([field()]))
    assert assessed(state, ["O1:n0"], scope="visible")["status"] == "done"


def test_unknown_persistence_remains_incomplete_and_does_not_claim_a_supported_subgoal():
    state = TaskState()
    state.observe(ui([field()]))
    parsed = assessed(state, ["O1:n0"], results=[], complete=False, status="unknown")
    assert (
        parsed["status"] == "incomplete" and not parsed["assessments"][0]["satisfied"]
    )
    with pytest.raises(ContractError, match="effect_unverified"):
        assessed(state, ["O1:n0"], results=[], complete=False)


def test_real_runtime_configuration_enables_goal_effect_scopes():
    runner, driver, provider = _runtime()
    config = JevConfig(fallback="autoglm")
    configured = DecisionRuntime(
        config,
        SimpleNamespace(driver=driver, ui=provider.ui),
        runner.registry,
        provider,
        policy=runner.policy,
    )
    assert configured.visual_fallback.effect_scopes is True


def test_goal_scope_compilation_uses_one_existing_request_and_preserves_the_goal(
    monkeypatch,
):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-key")
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["messages"][1]["content"] == GOAL
        assert "effect_scope" in payload["messages"][0]["content"]
        body = {
            "requirements": [
                {
                    "id": 1,
                    "description": GOAL,
                    "quote": GOAL,
                    "kind": "state",
                    "effect_scope": "persistent",
                }
            ],
            "answer_required": False,
        }
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(body)}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = VisualFallback(
                model="glm-5.3-flash",
                reasoning_model="glm-5.3-flash",
                client=client,
                base_url="http://example.invalid/v1",
                api_key_env="GLM_API_KEY",
                effect_scopes=True,
            )
            assert await adapter.prepare_task(goal=GOAL) is None
            assert (
                adapter.completion_contract["requirements"][0]["effect_scope"]
                == "persistent"
            )
            assert await adapter.prepare_task(goal=GOAL) is None

    asyncio.run(run())
    assert len(requests) == 1


@pytest.mark.parametrize("position", [0, 1, 2])
def test_reopen_evidence_with_omitted_nodes_cannot_prove_a_field_disappeared(position):
    state, refs, trace = reopened()
    state.observations[position]["nodes_budget_omitted"] = 1
    with pytest.raises(ContractError, match="effect_unverified"):
        assessed(state, refs, results=["O3:n0"], trace=trace)


@pytest.mark.parametrize("change", ["missing_parent", "cyclic_parent"])
def test_result_node_with_unresolvable_ancestry_cannot_escape_an_editable_parent(
    change,
):
    state = TaskState()
    node = field()
    node["children"] = [{"index": 2, "className": "TextView", "rawText": "Written"}]
    state.observe(ui([node]))
    state.observations[0]["nodes"][1]["parent_path"] = (
        999 if change == "missing_parent" else 1
    )
    with pytest.raises(ContractError, match="effect_unverified"):
        assessed(state, ["O1:n1"], results=["O1:n1"])


def test_review_http_requires_result_refs_without_an_additional_request_stage(
    monkeypatch,
):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-key")
    requests = []
    state = TaskState()
    page = ui([field(), {"index": 2, "className": "TextView", "rawText": "Saved"}])
    pixels = _image("green")
    state.observe(page, screenshot=pixels)

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert payload["model"] == "glm-5.3-flash"
        assert "result_refs" in payload["messages"][0]["content"]
        data = json.loads(review(["O1:n0", "O1:n1"], answer=None))
        data["assessments"][0]["result_refs"] = ["O1:n1"]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(data)}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = VisualFallback(
                model="glm-5.3-flash",
                reasoning_model="glm-5.3-flash",
                client=client,
                base_url="http://example.invalid/v1",
                api_key_env="GLM_API_KEY",
                effect_scopes=True,
            )
            adapter._goal, adapter._completion_contract = GOAL, contract()
            result = await adapter.decide(
                goal=GOAL,
                ui_state=page,
                screenshot=pixels,
                available_tools={"complete"},
                completion_review=True,
                task_state={"ledger": state.context()},
                derive_review_dependencies=True,
            )
            assert result.status == "done" and result.assessments[0]["result_refs"] == [
                "O1:n1"
            ]

    asyncio.run(run())
    assert len(requests) == 1


def test_missing_scope_retries_only_the_existing_goal_compilation_budget(monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-key")
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        body = {
            "requirements": [
                {"id": 1, "description": GOAL, "quote": GOAL, "kind": "state"}
            ],
            "answer_required": False,
        }
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(body)}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = VisualFallback(
                reasoning_model="glm-5.3-flash",
                client=client,
                base_url="http://example.invalid/v1",
                api_key_env="GLM_API_KEY",
                effect_scopes=True,
            )
            result = await adapter.prepare_task(goal=GOAL)
            assert result.status == "error" and result.error_field == "effect_scope"
            assert adapter.completion_contract is None

    asyncio.run(run())
    assert len(requests) == 2
