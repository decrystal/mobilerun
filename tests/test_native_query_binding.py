"""Local query completion must retain its actually reviewed native scope."""

import asyncio
import copy
import json
from dataclasses import asdict

import httpx
import pytest
from test_collection_query import page
from test_decision_visual_guards import _runtime
from test_task_evidence import task, ui

from mobilerun.agent.decision.native_query_binding import (
    NativeQueryBindingError,
    bind_native_queries,
    corroborate_native_queries,
)
from mobilerun.agent.decision.task_state import TaskState, observed_nodes
from mobilerun.agent.decision.visual_fallback import VisualFallback
from mobilerun.config_manager.config_manager import JevConfig

GOAL = "Count the records in the selected list"


def collection(rows=None, total=3, filter_text="All records"):
    nodes = page(
        "O1", rows if rows is not None else [(0, "A"), (1, "B"), (2, "C")], total=total
    )["nodes"]
    nodes.append(
        {
            "index": len(nodes) + 1,
            "resourceId": "filter",
            "className": "TextView",
            "rawText": filter_text,
        }
    )
    return ui(nodes)


async def reviewed_query(
    monkeypatch,
    change=None,
    *,
    enabled=True,
    capable=True,
    historical=False,
    mutate=None,
):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-secret")
    before = collection()
    state = TaskState(GOAL)
    state.set_contract(task(GOAL))
    state.observe(before)
    scope = next(iter(state.collections.scopes))
    state.query({"scope": scope, "generation": 1, "operation": "count"})
    if historical:
        before = ui(
            [
                {
                    "index": 1,
                    "resourceId": "detail",
                    "rawText": "An unrelated detail page",
                }
            ]
        )
        state.observe(before)
    if mutate in {
        "result",
        "source",
        "lost_source",
        "heading_before_review",
        "evidence",
    }:
        if mutate == "result":
            state.query_results[0]["value"] = 99
        elif mutate == "source":
            state._query_native_sources["Q1"]["source_sha256"] = "0" * 64
        elif mutate == "lost_source":
            state._query_native_sources.clear()
        elif mutate == "evidence":
            state.observations[0]["nodes"][0]["collectionInfo"]["rowCount"] = 99
        else:
            state.collections.scopes[scope].rows[0]["heading"] = True
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        response = {
            "complete": True,
            "evidence": "The selected native list has three observed records",
            "checks": {},
            "final_answer": "3",
            "verification_refs": ["Q1"],
            "assessments": [
                {
                    "id": 1,
                    "status": "supported",
                    "evidence": "Calculated the complete list",
                    "evidence_refs": ["Q1"],
                }
            ],
        }
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(response)}}]}
        )

    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        adapter = VisualFallback(
            model="glm-5.3-flash",
            reasoning_model="glm-5.3-flash",
            thinking_mode="disabled",
            api_key_env="GLM_API_KEY",
            base_url="https://example.invalid/v1",
            client=client,
        )
        adapter._goal, adapter._completion_contract = GOAL, state.contract
        adapter.supports_native_query_bindings = capable
        runner, _, provider = _runtime(before, visual=adapter)
        runner.config.native_query_bindings = enabled
        runner.goal, runner.task_state = GOAL, state
        selected = await runner._visual(
            before, completion_review=True, native_only_review=True
        )
        assert not selected.stop and selected.calls[0].name == "complete"
        after = copy.deepcopy(before)
        if change == "count":
            after.elements[0]["collectionInfo"]["rowCount"] = 5
        elif change == "row_value":
            after.elements[2]["rawText"] = "Another record"
        elif change == "filter":
            after.elements[-1]["rawText"] = "Archived records"
        elif change == "ambiguous_container":
            after.elements.append(
                {
                    "index": 100,
                    "resourceId": "records",
                    "className": "RecyclerView",
                    "collectionInfo": {"rowCount": 3, "columnCount": 1},
                }
            )
        elif change == "heading":
            after.elements[1]["collectionItemInfo"]["isHeading"] = True
        elif change == "missing_metadata":
            del after.elements[0]["collectionInfo"]
        elif change == "layout":
            for node in after.elements:
                node["bounds"] = "20,40,300,100"
        if mutate == "binding":
            selected.completion_query_binding["queries"][0]["scope_sha256"] = "0" * 64
        elif mutate == "lost_binding":
            selected.completion_query_binding = None
        elif mutate in {
            "lost_flag_and_binding",
            "lost_guards_and_result",
            "lost_guards_and_generation",
        }:
            selected.completion_query_binding_required = False
            selected.completion_query_binding = None
            if mutate == "lost_guards_and_result":
                state.query_results.clear()
            elif mutate == "lost_guards_and_generation":
                state.collections.scopes[scope].generation += 1
        elif mutate == "goal":
            runner.goal = "Another task"
        elif mutate == "state_goal":
            state.goal = "Another task"
        before_validation = copy.deepcopy(asdict(state.collections.scopes[scope]))
        provider.ui = after
        _, reason = await runner.validate_before_execute()
        assert len(requests) == 1 and runner.history == []
        assert asdict(state.collections.scopes[scope]) == before_validation
        serialized_request = json.dumps(requests)
        assert "_query_native_sources" not in serialized_request
        assert "completion_query_binding" not in serialized_request
        return runner, selected, reason


@pytest.mark.parametrize(
    "change", ["count", "row_value", "filter", "ambiguous_container"]
)
def test_completed_query_is_rejected_if_its_current_native_scope_changes(
    monkeypatch, change
):
    runner, _, reason = asyncio.run(reviewed_query(monkeypatch, change))
    assert reason == "completion_unverified"
    assert runner.last_completion_failure == "native_evidence"


@pytest.mark.parametrize("change", [None, "layout"])
def test_unchanged_native_query_values_and_scope_survive_layout_changes(
    monkeypatch, change
):
    runner, selected, reason = asyncio.run(reviewed_query(monkeypatch, change))
    assert reason is None
    assert selected.completion_query_binding_required
    assert selected.completion_query_binding["queries"][0]["current"] is True
    assert runner.task_state.query_results[0]["value"] == 3


@pytest.mark.parametrize("change", ["heading", "missing_metadata"])
def test_query_coverage_roles_and_missing_metadata_are_rechecked(monkeypatch, change):
    runner, _, reason = asyncio.run(reviewed_query(monkeypatch, change))
    assert (
        reason == "completion_unverified"
        and runner.last_completion_failure == "native_evidence"
    )


@pytest.mark.parametrize(
    "mutate",
    [
        "result",
        "source",
        "lost_source",
        "heading_before_review",
        "evidence",
        "binding",
        "lost_binding",
        "lost_flag_and_binding",
        "lost_guards_and_result",
        "lost_guards_and_generation",
        "goal",
        "state_goal",
    ],
)
def test_query_capture_and_guard_cannot_be_modified_or_lost_to_bypass_check(
    monkeypatch, mutate
):
    runner, _, reason = asyncio.run(reviewed_query(monkeypatch, mutate=mutate))
    assert (
        reason == "completion_unverified"
        and runner.last_completion_failure == "native_evidence"
    )


def test_historical_query_scope_need_not_be_reopened_on_the_final_page(monkeypatch):
    _, selected, reason = asyncio.run(reviewed_query(monkeypatch, historical=True))
    assert reason is None
    assert selected.completion_query_binding["queries"][0]["current"] is False


@pytest.mark.parametrize(
    "enabled,capable", [(False, True), (True, False), (False, False)]
)
def test_disabled_or_legacy_adapter_retains_original_query_protocol(
    monkeypatch, enabled, capable
):
    _, selected, reason = asyncio.run(
        reviewed_query(monkeypatch, "count", enabled=enabled, capable=capable)
    )
    assert reason is None
    assert (
        selected.completion_query_binding is None
        and not selected.completion_query_binding_required
    )


@pytest.mark.parametrize("value", [1, 0, None, "true"])
def test_native_query_binding_config_is_a_separate_strict_bool(value):
    with pytest.raises(ValueError, match="native_query_bindings"):
        JevConfig(native_query_bindings=value)


def query_state():
    state = TaskState(GOAL)
    state.set_contract(task(GOAL))
    screen = collection()
    state.observe(screen)
    scope = next(iter(state.collections.scopes))
    state.query({"scope": scope, "generation": 1, "operation": "count"})
    return state, scope, screen


def fresh(screen):
    nodes = observed_nodes(screen)
    for node in nodes:
        node["ref"] = "V:n" + str(node["path"])
    return {
        "id": "V",
        "source": "accessibility",
        "phone": screen.phone_state,
        "nodes": nodes,
    }


def test_zero_result_scope_becoming_nonempty_is_rejected():
    state = TaskState(GOAL)
    state.set_contract(task(GOAL))
    state.observe(collection([], total=0))
    scope = next(iter(state.collections.scopes))
    state.query({"scope": scope, "generation": 1, "operation": "count"})
    binding = bind_native_queries(["Q1"], state, state.current_id, goal=GOAL)
    assert corroborate_native_queries(
        binding, state, state.current_id, fresh(collection([], total=0)), goal=GOAL
    )
    assert not corroborate_native_queries(
        binding,
        state,
        state.current_id,
        fresh(collection([(0, "New")], total=1)),
        goal=GOAL,
    )


def test_query_scope_refresh_preserves_old_result_rejection_and_accepts_new_result():
    state, scope, _ = query_state()
    state.observe(collection([(0, "New"), (1, "B"), (2, "C")]))
    with pytest.raises(NativeQueryBindingError):
        bind_native_queries(["Q1"], state, state.current_id, goal=GOAL)
    state.query({"scope": scope, "generation": 2, "operation": "count"})
    binding = bind_native_queries(["Q2"], state, state.current_id, goal=GOAL)
    assert corroborate_native_queries(
        binding,
        state,
        state.current_id,
        fresh(collection([(0, "New"), (1, "B"), (2, "C")])),
        goal=GOAL,
    )


def test_additional_grounded_facts_do_not_invalidate_an_earlier_count_query():
    state, scope, screen = query_state()
    for index, row in state.collections.scopes[scope].rows.items():
        node = row["nodes"][-1]
        state.collections.add_native_fact(scope, index, "name", node["ref"], "rawText")
    state.query({"scope": scope, "generation": 1, "operation": "list", "field": "name"})
    binding = bind_native_queries(["Q1", "Q2"], state, state.current_id, goal=GOAL)
    assert corroborate_native_queries(
        binding, state, state.current_id, fresh(screen), goal=GOAL
    )


def test_new_observation_of_unchanged_data_keeps_the_original_query_sources():
    state, _, screen = query_state()
    moved = copy.deepcopy(screen)
    moved.elements[0]["bounds"] = "20,30,300,400"
    state.observe(moved)
    assert state.current_id == "O2"
    binding = bind_native_queries(["Q1"], state, state.current_id, goal=GOAL)
    assert binding["queries"][0]["current"] is True
    assert corroborate_native_queries(
        binding, state, state.current_id, fresh(moved), goal=GOAL
    )


@pytest.mark.parametrize("refs", [[], ["Q1", "Q1"], ["Q999"], ["O1:n0"], [True]])
def test_only_real_distinct_query_sources_can_be_bound(refs):
    state, _, _ = query_state()
    with pytest.raises(NativeQueryBindingError):
        bind_native_queries(refs, state, state.current_id, goal=GOAL)


def test_query_private_sources_are_reset_and_do_not_enter_evidence_or_context():
    state, _, _ = query_state()
    serialized = json.dumps({"context": state.context(), "evidence": state.evidence()})
    assert "_query_native_sources" not in serialized
    assert "scope_sha256" not in serialized and "source_sha256" not in serialized
    assert set(state._query_native_sources) == {"Q1"}
    state.reset("Another task")
    assert state._query_native_sources == {}


def test_rejected_query_scope_uses_existing_one_completion_recovery_budget(monkeypatch):
    runner, _, reason = asyncio.run(reviewed_query(monkeypatch, "count"))
    assert reason == "completion_unverified"
    assert runner.recover_completion_validation(reason)
    assert runner.review_rejections == 1
    # A second independent rejection cannot obtain a third completion review.
    runner.pending = asyncio.run(reviewed_query(monkeypatch, "count"))[1]
    runner.last_completion_failure = "native_evidence"
    assert not runner.recover_completion_validation("completion_unverified")
    assert runner.review_rejections == 2
