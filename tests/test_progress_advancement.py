"""Progress is new grounded information, not lost evidence or rendering churn."""

import asyncio
import copy
import json
from dataclasses import replace

import pytest
from test_collection_query import page
from test_decision_visual_guards import _Policy, _runtime
from test_task_evidence import task, ui

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.decision.input_verification import capture_input_target
from mobilerun.agent.decision.jev import JevDecision
from mobilerun.agent.decision.runtime import Selection
from mobilerun.agent.decision.task_state import TaskState
from mobilerun.agent.fast_agent.xml_parser import ToolCall


def state(kind="state"):
    result = TaskState("Inspect and update the records")
    result.set_contract(task(result.goal, kind))
    return result


def wait(result, screen):
    result.record_action(
        ToolCall("wait", {"duration": 0.5}), ActionResult(True, "waited")
    )
    result.observe(screen)


def test_losing_and_reacquiring_old_requirement_support_is_not_new_progress():
    result = state()
    screen = ui()
    result.observe(screen)
    result.assessments[1] = {"status": "supported"}
    result.progress.update(result)
    wait(result, screen)
    assert result.progress.stalled_actions == 1
    result.assessments[1] = {"status": "unknown"}
    wait(result, screen)
    assert result.progress.stalled_actions == 2
    result.assessments[1] = {"status": "supported"}
    wait(result, screen)
    assert result.progress.stalled_actions == 3


def test_collection_invalidation_and_smaller_coverage_do_not_reset_stalls():
    result = state("query")
    screen = ui(page("O1", [(0, "A"), (1, "B"), (2, "C")])["nodes"])
    result.observe(screen)
    wait(result, screen)
    assert result.progress.stalled_actions == 1
    changed = ui(page("O2", [(0, "Changed")])["nodes"])
    wait(result, changed)
    assert next(iter(result.collections.scopes.values())).generation == 2
    assert result.progress.stalled_actions == 2


def test_invalidating_a_query_result_is_not_progress():
    result = state("query")
    screen = ui(page("O1", [(0, "A"), (1, "B"), (2, "C")])["nodes"])
    result.observe(screen)
    scope = next(iter(result.collections.scopes))
    answer = result.query({"scope": scope, "generation": 1, "operation": "count"})
    wait(result, screen)
    assert result.progress.stalled_actions == 1
    changed = ui(page("O2", [(0, "Different"), (1, "B"), (2, "C")])["nodes"])
    wait(result, changed)
    assert answer["ref"] not in result.evidence()
    assert result.progress.stalled_actions == 2


def form(values=None, focused="title"):
    values = values or {}
    return ui(
        [
            {
                "index": index + 1,
                "resourceId": name,
                "className": "EditText",
                "editable": True,
                "focused": name == focused,
                "inputValue": values.get(name, ""),
                "bounds": f"0,{index * 100},300,{index * 100 + 80}",
            }
            for index, name in enumerate(("title", "description"))
        ]
    )


def test_runtime_same_text_in_different_native_fields_is_separate_progress():
    async def run():
        runner, _, provider = _runtime(form())
        runner.goal = "Fill both fields"
        runner.task_state = TaskState(runner.goal)
        runner.task_state.set_contract(task(runner.goal, "state"))
        values = {}
        for index, name in enumerate(("title", "description", "title")):
            before = form(values, focused=name)
            provider.ui = before
            runner.task_state.observe(before)
            call = ToolCall("type", {"text": "same value", "clear": True})
            runner.pending = Selection("test-native", [call], snapshot=before)
            current, reason = await runner.validate_before_execute()
            assert reason is None
            runner.record_execution(call, ActionResult(True, "dispatched"), current)
            values[name] = "same value"
            provider.ui = copy.deepcopy(form(values, focused=name))
            assert await runner.verify_input(call)
            assert runner.task_state.actions[-1]["effect_status"] == "confirmed"
            assert runner.task_state.progress.stalled_actions == (
                1 if index == 2 else 0
            )
        assert runner.task_state.assessments == {}

    asyncio.run(run())


def test_new_requirement_support_advances_once_and_updates_do_not_count_as_actions():
    result = state()
    screen = ui()
    result.observe(screen)
    for _ in range(3):
        wait(result, screen)
    result.assessments[1] = {"status": "supported"}
    result.progress.update(result)
    assert result.progress.stalled_actions == 0
    wait(result, screen)
    result.progress.update(result)
    result.progress.update(result)
    assert result.progress.stalled_actions == 1


def test_same_generation_new_rows_advance_once_but_revisiting_rows_does_not():
    result = state("query")
    first = ui(page("O1", [(0, "A")])["nodes"])
    result.observe(first)
    wait(result, first)
    second = ui(page("O2", [(1, "B")])["nodes"])
    wait(result, second)
    assert result.progress.stalled_actions == 0
    wait(result, first)
    wait(result, second)
    assert result.progress.stalled_actions == 2


def test_refreshed_generation_rebuilds_coverage_and_can_produce_new_query_evidence():
    result = state("query")
    result.observe(ui(page("O1", [(0, "A"), (1, "B"), (2, "C")])["nodes"]))
    scope = next(iter(result.collections.scopes))
    old = result.query({"scope": scope, "generation": 1, "operation": "count"})
    changed = ui(page("O2", [(0, "New")])["nodes"])
    wait(result, changed)
    assert result.progress.stalled_actions == 1
    wait(result, ui(page("O3", [(1, "B"), (2, "C")])["nodes"]))
    assert result.progress.stalled_actions == 0
    wait(result, changed)
    assert result.progress.stalled_actions == 1
    fresh = result.query({"scope": scope, "generation": 2, "operation": "count"})
    assert old["value"] == fresh["value"] == 3
    assert old["ref"] not in result.evidence() and fresh["ref"] in result.evidence()
    assert result.progress.stalled_actions == 0
    wait(result, changed)
    result.query({"scope": scope, "generation": 2, "operation": "count"})
    assert result.progress.stalled_actions == 1


@pytest.mark.parametrize(
    "rows,total", [([(0, "New")], 3), ([(0, "New"), (1, "B"), (2, "C")], 3), ([], 0)]
)
def test_generation_or_row_count_changes_alone_are_not_progress(rows, total):
    result = state("query")
    result.observe(ui(page("O1", [(0, "A"), (1, "B"), (2, "C")])["nodes"]))
    wait(result, ui(page("O2", rows, total=total)["nodes"]))
    assert result.progress.stalled_actions == 1
    assert next(iter(result.collections.scopes.values())).generation == 2


def test_repeated_generation_churn_never_resets_stalls_without_new_facts():
    result = state("query")
    result.observe(ui(page("O1", [(0, "A")])["nodes"]))
    for index in range(6):
        wait(result, ui(page(f"O{index + 2}", [(0, f"Changed {index}")])["nodes"]))
        assert result.progress.stalled_actions == index + 1
    assert result.progress.request_recovery(result)


def test_first_new_scope_with_rows_advances_but_revisiting_known_scope_does_not():
    result = state("query")
    result.observe(ui())
    wait(result, ui())
    rows = page("O1", [(0, "A")])["nodes"]
    first = ui(rows)
    wait(result, first)
    assert result.progress.stalled_actions == 0
    other = copy.deepcopy(rows)
    other[0]["resourceId"] = "other_records"
    wait(result, ui(other))
    assert result.progress.stalled_actions == 0
    wait(result, first)
    assert result.progress.stalled_actions == 1


def test_observed_empty_scope_requires_a_computed_zero_answer_for_query_progress():
    result = state("query")
    result.observe(ui())
    empty = ui(page("O1", [], total=0)["nodes"])
    wait(result, empty)
    assert result.progress.stalled_actions == 1
    scope = next(iter(result.collections.scopes))
    assert (
        result.query({"scope": scope, "generation": 1, "operation": "count"})["value"]
        == 0
    )
    assert result.progress.stalled_actions == 0
    wait(result, empty)
    result.query({"scope": scope, "generation": 1, "operation": "count"})
    assert result.progress.stalled_actions == 1 and result.assessments == {}


@pytest.mark.parametrize("kind", ["state", "process", "final_state"])
def test_collection_rows_alone_do_not_advance_a_nonquery_task(kind):
    result = state(kind)
    result.observe(ui())
    wait(result, ui(page("O1", [(0, "Unrelated")])["nodes"]))
    assert result.progress.stalled_actions == 1


def test_query_memory_disabled_does_not_collect_row_progress():
    result = TaskState("Count the records", query_memory=False)
    result.set_contract(task(result.goal, "query"))
    result.observe(ui())
    wait(result, ui(page("O1", [(0, "A")])["nodes"]))
    assert result.collections.scopes == {} and result.progress.stalled_actions == 1


def test_stale_query_cannot_be_reintroduced_as_progress_or_completion_evidence():
    result = state("query")
    result.observe(ui(page("O1", [(0, "A")], total=1)["nodes"]))
    scope = next(iter(result.collections.scopes))
    answer = result.query({"scope": scope, "generation": 1, "operation": "count"})
    wait(result, ui(page("O2", [(0, "B")], total=1)["nodes"]))
    result.query_results.append(copy.deepcopy(answer))
    result.progress.update(result)
    assert (
        result.progress.stalled_actions == 1 and answer["ref"] not in result.evidence()
    )


def native_target():
    return asyncio.run(capture_input_target(object(), form()))


def confirmed(result, target, *, text="same value", status="confirmed", **arguments):
    call = ToolCall("type", {"text": text, "clear": True, **arguments})
    result.record_action(call, ActionResult(True, "dispatched"), input_target=target)
    result.observe(form())
    result.record_input_effect(status)


@pytest.mark.parametrize(
    "change,advance",
    [
        ({"resource_id": "description"}, True),
        ({"package_name": "org.example.other"}, True),
        ({"window_id": 2}, True),
        ({"page_context": (("record_title", "Another record"),)}, True),
        ({"bounds": (0, 20, 300, 100)}, False),
        ({"parent": ("container", "FrameLayout", (0, 0, 400, 800))}, False),
        ({"require_focused": False}, False),
    ],
)
def test_native_field_binding_distinguishes_roles_and_ignores_unique_field_motion(
    change, advance
):
    result = state()
    result.observe(form())
    target = native_target()
    assert target.unique_resource_id
    confirmed(result, target)
    confirmed(result, replace(target, **change))
    assert result.progress.stalled_actions == (0 if advance else 1)
    assert result.assessments == {}


@pytest.mark.parametrize(
    "unique_hint,change,advance",
    [
        (False, {"bounds": (0, 100, 300, 180)}, True),
        (False, {"bounds": (0, 0, 300, 80)}, False),
        (True, {"bounds": (0, 100, 300, 180)}, False),
        (True, {"hint": "Description"}, True),
    ],
)
def test_duplicate_resources_use_existing_hint_or_position_binding(
    unique_hint, change, advance
):
    result = state()
    result.observe(form())
    target = replace(
        native_target(),
        unique_resource_id=False,
        unique_hint=unique_hint,
        hint="Title",
        window_id=1,
    )
    confirmed(result, target)
    confirmed(result, replace(target, **change))
    assert result.progress.stalled_actions == (0 if advance else 1)


def test_equivalent_parent_position_numbers_do_not_create_new_field_progress():
    result = state()
    result.observe(form())
    target = replace(
        native_target(),
        unique_resource_id=False,
        parent=("editor", "FrameLayout", (0.0, 0.0, 400.0, 800.0)),
    )
    confirmed(result, target)
    confirmed(
        result, replace(target, parent=("editor", "FrameLayout", (0, 0, 400, 800)))
    )
    assert result.progress.stalled_actions == 1


@pytest.mark.parametrize(
    "bounds", [(0, 0, 300), (True, 0, 300, 80), (0, 0, float("nan"), 80)]
)
def test_invalid_explicit_native_position_is_not_input_progress(bounds):
    result = state()
    result.observe(form())
    confirmed(result, replace(native_target(), unique_resource_id=False, bounds=bounds))
    assert result.progress.stalled_actions == 1


def test_runtime_nonreplacing_type_does_not_inherit_a_previous_field_binding():
    runner, _, _ = _runtime(form())
    runner.goal = "Enter the text"
    runner.task_state = TaskState(runner.goal)
    runner.task_state.set_contract(task(runner.goal, "state"))
    runner.task_state.observe(form())
    runner.input_target = native_target()
    call = ToolCall("type", {"text": "more", "clear": False})
    runner.pending = Selection("test-native", [call], snapshot=runner.ctx.ui)
    runner.record_execution(call, ActionResult(True, "dispatched"), runner.ctx.ui)
    assert runner.task_state._input_progress_targets == {}
    assert runner.task_state.actions[-1]["effect_status"] == "unknown"


def test_different_call_metadata_cannot_make_the_same_native_input_new_progress():
    result = state()
    result.observe(form())
    target = native_target()
    confirmed(result, target, index=1)
    confirmed(result, target, index=99)
    assert result.progress.stalled_actions == 1
    confirmed(result, target, text="another confirmed value")
    assert result.progress.stalled_actions == 0


@pytest.mark.parametrize("status", ["unknown", "mismatch"])
def test_unconfirmed_bound_input_does_not_reset_stalls(status):
    result = state()
    result.observe(form())
    confirmed(result, native_target(), status=status)
    assert result.progress.stalled_actions == 1


def test_invalid_explicit_binding_cannot_fall_back_to_unbound_progress():
    result = state()
    result.observe(form())
    confirmed(result, {"resource_id": "model-written-field"})
    assert result.progress.stalled_actions == 1


def test_private_target_and_page_context_are_not_added_to_model_actions_or_evidence():
    result = state()
    result.observe(form())
    target = replace(
        native_target(), page_context=(("record", "PRIVATE TARGET LABEL"),)
    )
    confirmed(result, target)
    serialized = json.dumps(
        {"context": result.context(), "evidence": result.evidence()}
    )
    assert "PRIVATE TARGET LABEL" not in serialized
    assert (
        "_input_progress_targets" not in serialized and "input_target" not in serialized
    )
    assert all(len(value) == 64 for value in result._input_progress_targets.values())


def test_reset_removes_prior_field_bindings_and_progress_novelty():
    result = state()
    result.observe(form())
    target = native_target()
    confirmed(result, target)
    result.reset("Fill another form")
    assert result._input_progress_targets == {} and result.progress.signature is None
    assert result.progress.seen_inputs == set() and result.progress.seen_rows == {}
    result.set_contract(task(result.goal, "state"))
    result.observe(form())
    wait(result, form())
    confirmed(result, target)
    assert result.progress.stalled_actions == 0


def test_runtime_recovery_still_uses_four_actions_and_two_attempts_per_requirement():
    async def run():
        screen = ui(
            [
                {
                    "index": 1,
                    "resourceId": "inspect",
                    "className": "Button",
                    "clickable": True,
                }
            ]
        )
        policy = _Policy(
            [
                JevDecision(
                    "action", tool_call=ToolCall("click", {"index": 1}), operation="TAP"
                )
            ]
        )
        runner, _, _ = _runtime(screen, policy=policy)
        runner.goal = "Inspect the result"
        runner.task_state = TaskState(runner.goal)
        runner.task_state.set_contract(task(runner.goal, "state"))
        runner.task_state.observe(screen)
        for attempt in range(3):
            for _ in range(4):
                wait(runner.task_state, screen)
            await runner.select(runner.goal, screen)
            recoveries = [
                item
                for item in runner.trace
                if item["event"] == "recovery" and item["source"] == "semantic_progress"
            ]
            assert len(recoveries) == min(attempt + 1, 2)
            assert runner.visual_fallback.calls == min(attempt + 1, 2)
        assert len(runner.history) == 0 and runner.task_state.assessments == {}

    asyncio.run(run())
