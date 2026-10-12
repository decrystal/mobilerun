"""A record control stays bound to its observed row and distinguishing values."""

import asyncio
import copy

import pytest
from test_decision_visual_guards import _runtime
from test_jev_decision import run_decision, state

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.decision.jev import _label, _nodes, coordinate_fingerprint
from mobilerun.agent.decision.runtime import Selection
from mobilerun.agent.decision.task_state import TaskState
from mobilerun.agent.fast_agent.xml_parser import ToolCall


def records():
    def row(index, number, date, top):
        return {
            "index": index,
            "className": "LinearLayout",
            "resourceId": "record",
            "bounds": f"0,{top},400,{top + 140}",
            "clickable": False,
            "collectionItemInfo": {"rowIndex": number, "columnIndex": 0},
            "children": [
                {
                    "index": index + 1,
                    "className": "TextView",
                    "resourceId": "title",
                    "text": "Report",
                    "rawText": "Report",
                    "bounds": f"10,{top + 10},250,{top + 45}",
                    "clickable": False,
                },
                {
                    "index": index + 2,
                    "className": "TextView",
                    "resourceId": "date",
                    "text": date,
                    "rawText": date,
                    "bounds": f"10,{top + 50},250,{top + 80}",
                    "clickable": False,
                },
                {
                    "index": index + 3,
                    "className": "CheckBox",
                    "resourceId": "select",
                    "text": "Select",
                    "bounds": f"320,{top + 40},380,{top + 100}",
                    "clickable": True,
                    "checked": False,
                },
            ],
        }

    return state(
        [
            {
                "index": 0,
                "className": "RecyclerView",
                "resourceId": "records",
                "bounds": "0,0,400,600",
                "scrollable": True,
                "clickable": False,
                "rawText": "All records",
                "collectionInfo": {"rowCount": 2, "columnCount": 1},
                "children": [row(1, 0, "2026-05-01", 40), row(5, 1, "2026-05-02", 220)],
            }
        ]
    )


@pytest.mark.parametrize("change", ["date", "scope"])
def test_changed_record_owner_rejects_a_control_that_itself_has_not_changed(change):
    page = records()
    selected, _ = run_decision(page, {"operation": "TAP", "tap_target": "4"})
    changed = copy.deepcopy(page)
    if change == "date":
        changed.elements[0]["children"][0]["children"][1]["rawText"] = "2026-05-02"
    else:
        changed.elements[0]["rawText"] = "Archived records"
    assert selected.validate(changed) == "target_owner_changed"


def test_native_point_guard_includes_the_record_name_and_date_outside_the_point():
    page = records()
    original = coordinate_fingerprint(page, 350, 100)
    page.elements[0]["children"][0]["children"][1]["rawText"] = "2026-05-02"
    assert coordinate_fingerprint(page, 350, 100) != original


def test_control_candidate_includes_its_owning_record_qualifier():
    nodes = _nodes(records())
    selected = next(node for node in nodes if node["index"] == 4)
    label = _label(selected, nodes)
    assert "Report" in label
    assert "2026-05-01" in label


@pytest.mark.parametrize("change", ["other_row", "count", "bounds"])
def test_unrelated_records_and_target_motion_do_not_change_its_owner(change):
    page = records()
    selected, _ = run_decision(page, {"operation": "TAP", "tap_target": "4"})
    if change == "other_row":
        page.elements[0]["children"][1]["children"][1]["rawText"] = "2026-06-01"
    elif change == "count":
        page.elements[0]["collectionInfo"]["rowCount"] = 3
    else:
        page.elements[0]["children"][0]["children"][2]["bounds"] = "320,100,380,160"
    assert selected.validate(page) is None


def test_same_name_with_different_dates_keeps_distinct_control_candidates():
    page = records()
    _, requests = run_decision(page, {"operation": "TAP", "tap_target": "4"})
    labels = requests[0]["questions"]["tap_target"]["criteria"]
    assert labels["4"] != labels["8"]
    assert "2026-05-01" in labels["4"] and "2026-05-02" in labels["8"]


def test_same_owner_point_guard_ignores_other_row_changes():
    page = records()
    original = coordinate_fingerprint(page, 350, 100)
    page.elements[0]["children"][1]["children"][1]["rawText"] = "2026-06-01"
    assert coordinate_fingerprint(page, 350, 100) == original


@pytest.mark.parametrize("source", ["jev", "autoglm", "llm"])
def test_runtime_blocks_changed_owner_before_any_action(source):
    page = records()
    runner, _, provider = _runtime(page)
    runner.pending = Selection(source, [ToolCall("click", {"index": 4})], snapshot=page)
    provider.ui = copy.deepcopy(page)
    provider.ui.elements[0]["children"][0]["children"][1]["rawText"] = "2026-05-02"
    _, reason = asyncio.run(runner.validate_before_execute())
    assert reason == "native_target_owner_changed"
    assert runner.history == [] and runner.executed == set()


@pytest.mark.parametrize("tool", ["click", "click_at", "long_press_at"])
def test_actual_action_evidence_retains_captured_native_target_owner(tool):
    page = records()
    runner, _, _ = _runtime(page)
    runner.task_state.observe(page)
    arguments = {"index": 4} if tool == "click" else {"x": 350, "y": 100}
    call = ToolCall(tool, arguments)
    runner.pending = Selection(
        "jev" if tool == "click" else "autoglm", [call], snapshot=page
    )
    binding = copy.deepcopy(runner.pending.native_target_binding)
    assert binding["owner_index"] == 1
    runner.record_execution(call, ActionResult(True, "Dispatched"), page)
    observed_binding = runner.history[-1]["native_target_binding"]
    assert {key: observed_binding[key] for key in binding} == binding
    assert observed_binding["owner_ref"] == "O1:n1"
    assert (
        runner.task_state.evidence()["A1"]["native_target_binding"] == observed_binding
    )
    assert runner.task_state.evidence()["A1"]["effect_status"] == "unknown"
    runner.pending.native_target_binding["owner_index"] = 99
    assert (
        runner.task_state.evidence()["A1"]["native_target_binding"] == observed_binding
    )


def test_owner_source_is_retained_in_review_memory_after_navigation():
    page = records()
    runner, _, _ = _runtime(page)
    runner.task_state.observe(page)
    call = ToolCall("click", {"index": 4})
    runner.pending = Selection("jev", [call], snapshot=page)
    runner.record_execution(call, ActionResult(True, "Dispatched"), page)
    for label in ("One", "Two", "Three"):
        runner.task_state.observe(
            state([{"index": 1, "text": label, "rawText": label}])
        )
    context = runner.task_state.context(max_observations=1, retain_proofs=True)
    assert "O1" in {observation["id"] for observation in context["observations"]}
    assert context["actions"][0]["native_target_binding"]["owner_index"] == 1


@pytest.mark.parametrize("removed", ["binding", "item_role"])
def test_missing_captured_owner_or_changed_row_role_cannot_drop_the_check(removed):
    page = records()
    runner, _, provider = _runtime(page)
    runner.pending = Selection(
        "autoglm", [ToolCall("click", {"index": 4})], snapshot=page
    )
    if removed == "binding":
        runner.pending.native_target_binding = None
    else:
        provider.ui = copy.deepcopy(page)
        provider.ui.elements[0]["children"][0].pop("collectionItemInfo")
    _, reason = asyncio.run(runner.validate_before_execute())
    assert reason == "native_target_owner_changed"


def test_plain_controls_keep_legacy_behavior_and_do_not_claim_a_record_owner():
    page = state(
        [{"index": 1, "text": "Open", "bounds": "0,0,200,80", "clickable": True}]
    )
    selected, _ = run_decision(page, {"operation": "TAP", "tap_target": "1"})
    assert selected.validate(page) is None
    selection = Selection("jev", [selected.tool_call], snapshot=page)
    assert selection.native_target_binding is None


def test_reset_does_not_retain_action_owner_evidence_from_another_goal():
    state_memory = TaskState("First goal")
    state_memory.observe(records())
    state_memory.record_action(
        ToolCall("click", {"index": 4}),
        ActionResult(True, "Dispatched"),
        native_target_binding={"owner_index": 1},
    )
    state_memory.reset("Second goal")
    assert state_memory.actions == [] and state_memory.context()["actions"] == []


def test_unknown_owner_reference_is_not_guessed_from_an_index_alone():
    memory = TaskState()
    memory.observe(state([{"index": 1, "text": "Open"}]))
    action = memory.record_action(
        ToolCall("click", {"index": 1}),
        ActionResult(True, "Dispatched"),
        native_target_binding={"owner_index": 1, "owner_sha256": "capture"},
    )
    assert "owner_ref" not in action["native_target_binding"]
    assert action["effect_status"] == "unknown"


@pytest.mark.parametrize("change", ["position", "container_resource", "cyclic_parent"])
def test_changed_structural_owner_binding_does_not_reuse_a_record_control(change):
    page = records()
    runner, _, provider = _runtime(page)
    runner.pending = Selection(
        "autoglm", [ToolCall("click", {"index": 4})], snapshot=page
    )
    provider.ui = copy.deepcopy(page)
    root, first = provider.ui.elements[0], provider.ui.elements[0]["children"][0]
    if change == "position":
        first["collectionItemInfo"]["rowIndex"] = 1
    elif change == "container_resource":
        root["resourceId"] = "another_scope"
    else:
        root["parentIndex"] = 4
    _, reason = asyncio.run(runner.validate_before_execute())
    assert reason == "native_target_owner_changed"
    assert runner.history == []
