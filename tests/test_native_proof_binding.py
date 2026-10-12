"""Refreshing proofs cannot merge objects or move properties between cards."""

import asyncio
import copy
import json

import httpx
import pytest
from test_decision_visual_guards import _runtime, _ui
from test_task_evidence import task

from mobilerun.agent.decision.autoglm import AutoGLMFallback
from mobilerun.agent.decision.native_proof_binding import (
    NativeProofBindingError,
    bind_native_proofs,
    corroborate_native_binding,
)
from mobilerun.agent.decision.task_state import (
    TaskState,
    corroborate_nodes,
    observed_nodes,
)
from mobilerun.config_manager.config_manager import JevConfig

GOAL = "Keep the requested records and their saved properties"


def leaf(text, resource="status", bounds="0,0,300,100"):
    return {
        "rawText": text,
        "text": text,
        "className": "TextView",
        "resourceId": resource,
        "bounds": bounds,
    }


def capture(page, labels):
    state = TaskState(GOAL)
    state.set_contract(task(GOAL, "state"))
    observation = state.observe(page)
    remaining = list(labels)
    proofs = []
    for node in observation["nodes"]:
        if node.get("rawText") in remaining:
            proofs.append(copy.deepcopy(node))
            remaining.remove(node["rawText"])
    assert remaining == []
    return (
        state,
        observation,
        proofs,
        bind_native_proofs(proofs, observation, goal=GOAL),
    )


def fresh(binding, proofs, observation, page):
    return corroborate_native_binding(
        binding,
        proofs,
        observation,
        observed_nodes(page),
        page.phone_state["packageName"],
        goal=GOAL,
    )


def cards(*, move_saved=False, title="Record A", ancestor_label=None):
    first_children = [leaf(title, "title", "0,0,300,100")]
    second_children = [leaf("Record B", "title", "0,300,300,400")]
    (second_children if move_saved else first_children).append(
        leaf("Saved", "status", "0,100,300,200")
    )
    first = {"className": "CardView", "resourceId": "row", "children": first_children}
    if ancestor_label is not None:
        first["rawText"] = ancestor_label
    return _ui(
        [
            {
                "className": "FrameLayout",
                "resourceId": "screen",
                "children": [
                    first,
                    {
                        "className": "CardView",
                        "resourceId": "row",
                        "children": second_children,
                    },
                ],
            }
        ]
    )


def test_two_different_reviewed_records_cannot_collapse_into_one_matching_control():
    page = _ui([leaf("Saved"), leaf("Saved", bounds="0,300,300,400")])
    _, observation, proofs, binding = capture(page, ["Saved", "Saved"])
    later = _ui([leaf("Saved")])
    assert corroborate_nodes(proofs, later)  # Reproduces the old per-control gap.
    assert not fresh(binding, proofs, observation, later)
    assert fresh(binding, proofs, observation, page)


def test_title_and_state_cannot_be_refreshed_from_different_cards():
    page = cards()
    _, observation, proofs, binding = capture(page, ["Record A", "Saved"])
    later = cards(move_saved=True)
    assert corroborate_nodes(proofs, later)
    assert not fresh(binding, proofs, observation, later)
    assert fresh(binding, proofs, observation, page)


def test_one_property_cannot_move_to_a_different_native_parent_identity():
    page = cards(ancestor_label="Object A")
    _, observation, proofs, binding = capture(page, ["Saved"])
    later = cards(ancestor_label="Object B")
    assert corroborate_nodes(proofs, later)
    assert not fresh(binding, proofs, observation, later)


def test_a_reviewed_parent_still_owns_its_reviewed_child():
    page = cards()
    page.elements[0]["children"][0]["rawText"] = "Card"
    _, observation, proofs, binding = capture(page, ["Card", "Saved"])
    later = copy.deepcopy(page)
    node = later.elements[0]["children"][0]["children"].pop()
    later.elements[0]["children"][1]["children"].append(node)
    assert corroborate_nodes(proofs, later)
    assert not fresh(binding, proofs, observation, later)


@pytest.mark.parametrize(
    "change",
    ["unrelated_label", "insert_sibling", "unique_bounds", "new_image", "row_position"],
)
def test_unrelated_changes_do_not_invalidate_owned_properties(change):
    page = cards()
    page.elements[0]["children"].append(leaf("12:00", "clock"))
    _, observation, proofs, binding = capture(page, ["Record A", "Saved"])
    later = copy.deepcopy(page)
    if change == "unrelated_label":
        later.elements[0]["children"][-1]["rawText"] = "12:01"
        later.elements[0]["children"][-1]["text"] = "12:01"
    elif change == "insert_sibling":
        later.elements[0]["children"].insert(0, leaf("New unrelated control", "other"))
    elif change == "unique_bounds":
        later.elements[0]["children"][0]["children"][1]["bounds"] = "20,200,350,350"
    elif change == "new_image":
        observation["images"] = [
            {"ref": "O1:image1", "sha256": "new-image", "source": "screenshot"}
        ]
    else:
        page.elements[0]["children"][0]["collectionItemInfo"] = {"rowIndex": 0}
        _, observation, proofs, binding = capture(page, ["Record A", "Saved"])
        later = copy.deepcopy(page)
        later.elements[0]["children"][0]["collectionItemInfo"]["rowIndex"] = 7
    assert corroborate_nodes(proofs, later)
    assert fresh(binding, proofs, observation, later)


@pytest.mark.parametrize(
    "change",
    [
        "value",
        "duplicate",
        "missing_ref",
        "cycle",
        "missing_parent",
        "duplicate_path",
        "package",
        "source",
        "too_many",
        "phone",
        "empty_goal",
    ],
)
def test_unknown_or_ambiguous_original_bindings_are_rejected(change):
    _, observation, proofs, _ = capture(cards(), ["Record A", "Saved"])
    goal = GOAL
    if change == "value":
        proofs[0]["rawText"] = "Other"
    elif change == "duplicate":
        proofs.append(proofs[0])
    elif change == "missing_ref":
        proofs[0]["ref"] = "O9:n1"
    elif change == "cycle":
        observation["nodes"][0]["parent_path"] = observation["nodes"][0]["path"]
    elif change == "missing_parent":
        observation["nodes"][0]["parent_path"] = 9999
    elif change == "duplicate_path":
        observation["nodes"][-1]["path"] = 0
    elif change == "package":
        observation["phone"]["packageName"] = ""
    elif change == "source":
        observation["source"] = "model_claim"
    elif change == "phone":
        observation["phone"] = "not a native phone state"
    elif change == "empty_goal":
        goal = ""
    else:
        proofs = proofs * 33
    with pytest.raises((NativeProofBindingError, ValueError)):
        bind_native_proofs(proofs, observation, goal=goal)


@pytest.mark.parametrize(
    "change",
    [
        "binding_version_bool",
        "binding_path_bool",
        "binding_package",
        "extra_binding_key",
        "source_value",
        "proof_value",
        "fresh_package",
        "fresh_cycle",
    ],
)
def test_captured_binding_source_proof_and_current_graph_are_rechecked(change):
    page = cards()
    _, observation, proofs, binding = capture(page, ["Record A", "Saved"])
    later = copy.deepcopy(page)
    if change == "binding_version_bool":
        binding["schema_version"] = True
    elif change == "binding_path_bool":
        binding["proofs"][0]["path"] = True
    elif change == "binding_package":
        binding["package"] = "other"
    elif change == "extra_binding_key":
        binding["ignored"] = True
    elif change == "source_value":
        observation["nodes"][proofs[0]["path"]]["rawText"] = "Other"
    elif change == "proof_value":
        proofs[0]["rawText"] = "Other"
    elif change == "fresh_package":
        later.phone_state["packageName"] = "org.example.other"
    else:
        nodes = observed_nodes(later)
        nodes[0]["parent_path"] = 0
        assert not corroborate_native_binding(
            binding, proofs, observation, nodes, binding["package"], goal=GOAL
        )
        return
    assert not fresh(binding, proofs, observation, later)


@pytest.mark.parametrize(
    "scenario",
    [
        "unchanged",
        "alias",
        "card",
        "ancestor",
        "unrelated",
        "tamper",
        "source",
        "goal",
        "state_goal",
        "missing_binding",
        "lost_flag_and_binding",
    ],
)
def test_one_mock_http_review_cannot_dispatch_complete_from_changed_object_ownership(
    monkeypatch, scenario
):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-key")
    page = (
        _ui([leaf("Saved"), leaf("Saved", bounds="0,300,300,400")])
        if scenario == "alias"
        else cards()
    )
    labels = ["Saved", "Saved"] if scenario == "alias" else ["Record A", "Saved"]
    requests, holders = [], []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        runner = holders[0]
        refs, needed = [], list(labels)
        for node in runner.task_state.observations[-1]["nodes"]:
            if node.get("rawText") in needed:
                refs.append(node["ref"])
                needed.remove(node["rawText"])
        assert not needed
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": json.dumps(
                                {
                                    "complete": True,
                                    "evidence": "Requested native values observed",
                                    "checks": {},
                                    "final_answer": None,
                                    "verification_refs": refs,
                                    "assessments": [
                                        {
                                            "id": 1,
                                            "status": "supported",
                                            "evidence": "Observed native values",
                                            "evidence_refs": refs,
                                        }
                                    ],
                                }
                            )
                        }
                    }
                ]
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = AutoGLMFallback(
                reasoning_model="glm-5.3-flash",
                api_key_env="GLM_API_KEY",
                thinking_mode="disabled",
                client=client,
            )
            adapter._goal, adapter._completion_contract = GOAL, task(GOAL, "state")
            runner, _, provider = _runtime(page, visual=adapter)
            holders.append(runner)
            runner.goal = GOAL
            runner.task_state = TaskState(GOAL)
            runner.task_state.set_contract(adapter._completion_contract)
            runner.config.native_proof_bindings = True
            selected = await runner._visual(page, completion_review=True)
            assert (
                selected.calls[0].name == "complete"
                and selected.completion_native_binding
            )
            if scenario == "alias":
                provider.ui = _ui([leaf("Saved")])
            elif scenario == "card":
                provider.ui = cards(move_saved=True)
            elif scenario == "ancestor":
                provider.ui = cards()
                provider.ui.elements[0]["resourceId"] = "other_screen"
            elif scenario == "unrelated":
                provider.ui = copy.deepcopy(page)
                provider.ui.elements[0]["children"].append(leaf("12:01", "clock"))
            elif scenario == "tamper":
                selected.completion_native_binding["schema_version"] = True
            elif scenario == "source":
                runner.task_state.observations[-1]["nodes"][0][
                    "resourceId"
                ] = "changed_source"
            elif scenario == "goal":
                runner.goal = "Another task"
            elif scenario == "state_goal":
                runner.task_state.goal = "Another task"
            elif scenario == "missing_binding":
                selected.completion_native_binding = None
            elif scenario == "lost_flag_and_binding":
                selected.completion_native_binding = None
                selected.completion_native_binding_required = False
            assert corroborate_nodes(selected.completion_proofs, provider.ui)
            _, reason = await runner.validate_before_execute()
            assert reason == (
                None
                if scenario in {"unchanged", "unrelated"}
                else "completion_unverified"
            )
            assert runner.history == []
            if reason:
                assert runner.last_completion_failure == "native_evidence"
                assert runner.recover_completion_validation(reason)
                assert runner.pending is None and runner.review_rejections == 1
                assert runner.feedback.startswith("Completion was NOT accepted")

    asyncio.run(run())
    assert len(requests) == 1


@pytest.mark.parametrize("enabled,capable", [(False, True), (True, False)])
def test_disabled_and_older_adapters_keep_their_existing_binding_protocol(
    monkeypatch, enabled, capable
):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-key")

    async def run():
        state, _, proofs, _ = capture(cards(), ["Saved"])
        content = json.dumps(
            {
                "complete": True,
                "evidence": "Saved",
                "checks": {},
                "final_answer": None,
                "assessments": [
                    {
                        "id": 1,
                        "status": "supported",
                        "evidence": "Saved",
                        "evidence_refs": [proofs[0]["ref"]],
                    }
                ],
            }
        )
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200, json={"choices": [{"message": {"content": content}}]}
                )
            )
        ) as client:
            adapter = AutoGLMFallback(
                reasoning_model="glm-5.3-flash",
                api_key_env="GLM_API_KEY",
                client=client,
            )
            adapter.supports_native_proof_bindings = capable
            adapter._goal, adapter._completion_contract = GOAL, state.contract
            runner, _, provider = _runtime(cards(), visual=adapter)
            runner.goal, runner.task_state = GOAL, state
            runner.config.native_proof_bindings = enabled
            selected = await runner._visual(provider.ui, completion_review=True)
            assert selected.completion_native_binding is None
            assert (await runner.validate_before_execute())[1] is None

    asyncio.run(run())


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_binding_configuration_requires_a_boolean(value):
    with pytest.raises(ValueError, match="native_proof_bindings"):
        JevConfig(native_proof_bindings=value)
