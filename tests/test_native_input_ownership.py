"""Typing and readback stay attached to the observed native record."""

import asyncio
import copy
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_decision_visual_guards import _runtime
from test_native_action_ownership import records

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.decision.input_verification import (
    capture_input_target,
    read_input_value,
)
from mobilerun.agent.decision.jev import focused_input_observation
from mobilerun.agent.decision.progress import input_target_progress_key
from mobilerun.agent.decision.runtime import Selection
from mobilerun.agent.fast_agent.xml_parser import ToolCall
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.ui.state import UIState


def editors():
    page = records()
    for number, row in enumerate(page.elements[0]["children"]):
        field = row["children"][-1]
        field.update(
            className="EditText",
            resourceId="note",
            editable=True,
            focused=number == 0,
            inputValue="old",
            rawText="old",
            text="old",
        )
        field.pop("checked")
    return page


def change_owner(page, change):
    root = page.elements[0]
    row = root["children"][0]
    if change == "date":
        row["children"][1]["rawText"] = "2026-05-02"
    elif change == "scope":
        root["rawText"] = "Archived records"
    elif change == "position":
        row["collectionItemInfo"]["rowIndex"] = 1
    elif change == "missing_role":
        row.pop("collectionItemInfo")
    elif change == "title":
        row["children"][0]["rawText"] = "Another report"
    elif change == "parent":
        root["resourceId"] = "another_scope"


def raw_page(page):
    def native(node):
        left, top, right, bottom = (int(value) for value in node["bounds"].split(","))
        result = {
            key: copy.deepcopy(value)
            for key, value in node.items()
            if key
            not in {
                "index",
                "parentIndex",
                "children",
                "bounds",
                "text",
                "rawText",
                "inputValue",
            }
        }
        result.update(
            className="android.widget." + node["className"],
            boundsInScreen=dict(left=left, top=top, right=right, bottom=bottom),
            text=node.get("inputValue", node.get("rawText", node.get("text", ""))),
        )
        result["children"] = [native(child) for child in node.get("children", [])]
        return result

    return {
        "phone_state": copy.deepcopy(page.phone_state),
        "a11y_tree": native(page.elements[0]),
    }


def capture(page, *, driver=None, index=None):
    return asyncio.run(capture_input_target(driver or SimpleNamespace(), page, index))


def readback(page, target, *, driver=None, text="written"):
    return asyncio.run(
        read_input_value(driver or SimpleNamespace(), page, target, text)
    )


@pytest.mark.parametrize("source", ["jev", "autoglm", "llm"])
@pytest.mark.parametrize("change", ["date", "scope", "position"])
def test_fallback_type_rejects_changed_record_before_dispatch(source, change):
    page = editors()
    runner, _, provider = _runtime(page)
    call = ToolCall("type", {"text": "written", "clear": True})
    runner.pending = Selection(
        source, [call], snapshot=page, input_guard=focused_input_observation(page)
    )
    provider.ui = copy.deepcopy(page)
    change_owner(provider.ui, change)
    _, reason = asyncio.run(runner.validate_before_execute())
    assert reason == "native_target_owner_changed"
    assert (
        runner.input_target is None
        and runner.history == []
        and runner.executed == set()
    )


@pytest.mark.parametrize(
    "change", ["date", "scope", "position", "missing_role", "title", "parent"]
)
@pytest.mark.parametrize("value", ["written", "wrong"])
def test_record_change_is_unknown_even_when_reused_field_has_the_expected_value(
    change, value
):
    before = editors()
    target = capture(before)
    assert target is not None
    after = copy.deepcopy(before)
    change_owner(after, change)
    after.elements[0]["children"][0]["children"][-1]["inputValue"] = value
    result = readback(after, target)
    assert result.status == "unknown" and result.reason == "input_owner_changed"


@pytest.mark.parametrize("change", ["date", "scope", "missing_role"])
def test_native_read_during_capture_cannot_switch_the_selected_record(change):
    before = editors()
    changed = copy.deepcopy(before)
    change_owner(changed, change)
    driver = SimpleNamespace(get_ui_tree=AsyncMock(return_value=raw_page(changed)))
    assert capture(before, driver=driver) is None
    assert driver.get_ui_tree.await_count == 1


@pytest.mark.parametrize("source", ["jev", "autoglm", "llm"])
@pytest.mark.parametrize("indexed", [False, True])
def test_dispatched_type_keeps_record_evidence_without_claiming_a_saved_effect(
    source, indexed
):
    page = editors()
    runner, _, _ = _runtime(page)
    runner.task_state.observe(page)
    parameters = {"text": "written", "clear": True}
    if indexed:
        parameters["index"] = 4
    call = ToolCall("type", parameters)
    runner.pending = Selection(
        source, [call], snapshot=page, input_guard=focused_input_observation(page)
    )
    current, reason = asyncio.run(runner.validate_before_execute())
    assert reason is None
    runner.record_execution(call, ActionResult(True, "Dispatched"), current)
    binding = runner.task_state.evidence()["A1"].get("native_target_binding")
    assert (
        binding is not None
        and binding["owner_index"] == 1
        and binding["owner_ref"] == "O1:n1"
    )
    assert runner.task_state.evidence()["A1"]["effect_status"] == "unknown"
    assert runner.history[-1]["native_target_binding"] == binding


@pytest.mark.parametrize(
    "change", ["value", "focus", "layout", "other_record", "count"]
)
def test_same_record_readback_survives_the_intended_input_and_unrelated_changes(change):
    before = editors()
    target = capture(before)
    after = copy.deepcopy(before)
    field = after.elements[0]["children"][0]["children"][-1]
    field.update(inputValue="written", rawText="written", text="written")
    if change == "focus":
        field.update(focused=False, textSelectionStart=7, textSelectionEnd=7)
        after.elements[0]["children"][1]["children"][-1]["focused"] = True
    elif change == "layout":
        before = editors()
        before.elements[0]["children"][1]["children"][-1]["resourceId"] = "other_note"
        target = capture(before)
        after = copy.deepcopy(before)
        field = after.elements[0]["children"][0]["children"][-1]
        field.update(
            inputValue="written",
            rawText="written",
            text="written",
            bounds="320,120,380,180",
        )
    elif change == "other_record":
        after.elements[0]["children"][1]["children"][1]["rawText"] = "2026-08-01"
    elif change == "count":
        after.elements[0]["collectionInfo"]["rowCount"] = 3
    assert readback(after, target).status == "confirmed"


def test_same_record_actual_mismatch_keeps_the_existing_bounded_repair_path():
    page = editors()
    target = capture(page)
    result = readback(page, target)
    assert result.status == "mismatch" and result.reason == "different_value"


def test_missing_input_owner_receipt_cannot_confirm_a_record_input():
    before = editors()
    target = capture(before)
    assert getattr(target, "native_owner_sha256", None) is not None
    target = replace(target, native_owner_sha256=None)
    before.elements[0]["children"][0]["children"][-1]["inputValue"] = "written"
    assert readback(before, target).status == "unknown"


def test_missing_selection_owner_receipt_cannot_skip_type_dispatch_validation():
    page = editors()
    runner, _, _ = _runtime(page)
    call = ToolCall("type", {"text": "written", "clear": True})
    runner.pending = Selection(
        "autoglm", [call], snapshot=page, input_guard=focused_input_observation(page)
    )
    runner.pending.native_target_binding = None
    _, reason = asyncio.run(runner.validate_before_execute())
    assert reason == "native_target_owner_changed"


def test_raw_and_formatted_record_binding_agree_without_native_indices():
    payload = raw_page(editors())
    _, _, elements, phone = IndexedFormatter().format(
        payload["a11y_tree"], payload["phone_state"]
    )
    page = UIState(elements, "", "", phone, screen_width=400, screen_height=800)
    driver = SimpleNamespace(get_ui_tree=AsyncMock(return_value=payload))
    target = capture(page, driver=driver)
    assert (
        target is not None and getattr(target, "native_owner_sha256", None) is not None
    )
    field = payload["a11y_tree"]["children"][0]["children"][-1]
    field.update(text="written", focused=False)
    assert readback(page, target, driver=driver).status == "confirmed"
    assert driver.get_ui_tree.await_count == 2


def test_screenshot_only_capture_and_readback_bind_the_native_focused_record():
    before = editors()
    payload = raw_page(before)
    page = copy.deepcopy(before)
    page.elements = []
    driver = SimpleNamespace(get_ui_tree=AsyncMock(return_value=payload))
    target = capture(page, driver=driver)
    assert target is not None
    field = payload["a11y_tree"]["children"][0]["children"][-1]
    field["text"] = "written"
    payload["a11y_tree"]["children"][0]["children"][1]["text"] = "2026-05-02"
    assert readback(page, target, driver=driver).status == "unknown"


def test_goal_local_input_progress_distinguishes_different_records_in_a_reused_editor():
    first = editors()
    second = copy.deepcopy(first)
    change_owner(second, "date")
    left, right = capture(first), capture(second)
    assert input_target_progress_key(left) != input_target_progress_key(right)


def test_input_progress_identity_ignores_the_changed_value_in_the_same_record():
    page = editors()
    target = capture(page)
    page.elements[0]["children"][0]["children"][-1].update(
        inputValue="new", rawText="new", text="new"
    )
    assert input_target_progress_key(target) == input_target_progress_key(capture(page))


def test_unowned_plain_input_keeps_readback_without_inventing_a_record():
    page = editors()
    page.elements = [page.elements[0]["children"][0]["children"][-1]]
    target = capture(page)
    assert target is not None and getattr(target, "native_owner_sha256", None) is None
    page.elements[0]["inputValue"] = "written"
    assert readback(page, target).status == "confirmed"


@pytest.mark.parametrize(
    "change", ["cycle", "missing_parent", "invalid_parent", "invalid_role"]
)
def test_unresolvable_native_input_ancestry_does_not_authorize_typing(change):
    page = editors()
    root = page.elements[0]
    row = root["children"][0]
    if change == "cycle":
        root["parentIndex"] = 4
    elif change == "missing_parent":
        row["parentIndex"] = 999
    elif change == "invalid_parent":
        row["parentIndex"] = "0"
    else:
        row["collectionItemInfo"] = "unobserved role"
    assert capture(page) is None


def test_generated_formatter_labels_for_missing_native_text_do_not_change_the_owner():
    payload = raw_page(editors())
    root = payload["a11y_tree"]
    root.pop("text")
    root["resourceId"] = ""
    for row in root["children"]:
        row.pop("text")
    _, _, elements, phone = IndexedFormatter().format(root, payload["phone_state"])
    page = UIState(elements, "", "", phone, screen_width=400, screen_height=800)
    driver = SimpleNamespace(get_ui_tree=AsyncMock(return_value=payload))
    assert capture(page, driver=driver) is not None


def test_input_owner_survives_global_index_reassignment_in_the_same_record():
    page = editors()
    target = capture(page)
    root = page.elements[0]
    for node in [
        root,
        *root["children"],
        *(child for row in root["children"] for child in row["children"]),
    ]:
        node["index"] += 100
    root["children"][0]["children"][-1].update(
        inputValue="written", rawText="written", text="written"
    )
    assert readback(page, target).status == "confirmed"


@pytest.mark.parametrize("changed", [False, True])
def test_nested_record_input_is_bound_to_its_outer_record_context(changed):
    page = editors()
    inner = page.elements[0]["children"][0]
    page.elements[0]["children"][0] = {
        "index": 20,
        "className": "LinearLayout",
        "resourceId": "outer_record",
        "collectionItemInfo": {"rowIndex": 0},
        "rawText": "Parent A",
        "bounds": "0,30,400,200",
        "children": [inner],
    }
    target = capture(page)
    assert target is not None
    inner["children"][-1].update(
        inputValue="written", rawText="written", text="written"
    )
    if changed:
        page.elements[0]["children"][0]["rawText"] = "Parent B"
    assert readback(page, target).status == ("unknown" if changed else "confirmed")


@pytest.mark.parametrize("changed", [False, True])
def test_input_repair_only_proposes_replacement_in_the_original_record(changed):
    page = editors()
    target = capture(page)
    call = ToolCall("type", {"text": "written", "clear": True})
    runner, _, _ = _runtime(page)
    goal = "Edit the first report note"
    runner.goal = goal
    runner.task_state.reset(goal)
    runner.input_retry = (call, target)
    runner.input_retry_activated = True
    if changed:
        change_owner(page, "date")
    selected = asyncio.run(runner.select(goal, page))
    if changed:
        assert selected.stop and selected.calls == []
    else:
        assert (
            not selected.stop
            and selected.source == "input_recovery"
            and selected.calls == [call]
        )
    assert runner.history == []


def test_native_phone_format_equivalence_still_requires_the_original_record():
    page = editors()
    field = page.elements[0]["children"][0]["children"][-1]
    field["inputType"] = 3
    target = capture(page)
    field.update(
        inputValue="+44 (1234) 567-890",
        rawText="+44 (1234) 567-890",
        text="+44 (1234) 567-890",
    )
    assert readback(page, target, text="+441234567890").status == "confirmed"
    change_owner(page, "date")
    assert readback(page, target, text="+441234567890").status == "unknown"


def test_current_native_record_overrides_a_parsed_ui_claim_of_matching_input():
    before = editors()
    driver = SimpleNamespace(get_ui_tree=AsyncMock(return_value=raw_page(before)))
    target = capture(before, driver=driver)
    assert target is not None
    before.elements[0]["children"][0]["children"][-1]["inputValue"] = "written"
    changed = copy.deepcopy(before)
    change_owner(changed, "date")
    driver.get_ui_tree.return_value = raw_page(changed)
    result = readback(before, target, driver=driver)
    assert result.status == "unknown" and result.reason == "input_owner_changed"
    assert driver.get_ui_tree.await_count == 2
