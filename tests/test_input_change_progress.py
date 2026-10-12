"""A successful readback may confirm an unchanged value, not an input change."""

import asyncio
import copy
import hashlib
import json
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_decision_input_verification import _raw_input
from test_decision_visual_guards import _runtime
from test_task_evidence import task, ui

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.decision.input_verification import (
    capture_input_target,
    read_input_value,
)
from mobilerun.agent.decision.runtime import Selection
from mobilerun.agent.decision.task_state import TaskState
from mobilerun.agent.fast_agent.xml_parser import ToolCall


def editor(value, *, native_type=1):
    return ui(
        [
            {
                "index": 1,
                "resourceId": "body",
                "className": "EditText",
                "editable": True,
                "focused": True,
                "inputValue": value,
                "inputType": native_type,
                "bounds": "0,0,300,100",
            }
        ]
    )


async def type_and_readback(before, text, after, *, raw_before=None, raw_after=None):
    runner, driver, provider = _runtime(before)
    runner.goal = "Fill the field and inspect the result"
    runner.task_state = TaskState(runner.goal)
    runner.task_state.set_contract(task(runner.goal, "state"))
    runner.task_state.observe(before)
    if raw_before is not None:
        driver.get_ui_tree = AsyncMock(side_effect=[raw_before, raw_after])
    call = ToolCall("type", {"text": text, "clear": True})
    runner.pending = Selection("test-native", [call], snapshot=before)
    current, reason = await runner.validate_before_execute()
    assert reason is None
    runner.record_execution(call, ActionResult(True, "dispatched"), current)
    provider.ui = copy.deepcopy(after)
    assert await runner.verify_input(call)
    assert runner.task_state.actions[-1]["effect_status"] == "confirmed"
    assert runner.task_state.assessments == {}
    return runner, driver, provider


def test_first_confirmed_type_of_an_already_present_value_is_not_change_progress():
    before = editor("Already present")
    runner, *_ = asyncio.run(type_and_readback(before, "Already present", before))
    assert runner.task_state.progress.stalled_actions == 1


def test_exact_unicode_and_multiline_noop_is_confirmed_without_change_progress():
    value = " 第一行 🌿\n\n第二行\t "
    before = editor(value)
    runner, *_ = asyncio.run(type_and_readback(before, value, before))
    assert runner.task_state.progress.stalled_actions == 1


def test_native_phone_display_equivalence_is_not_a_new_input_change():
    before = editor("+44 (1234) 567-890", native_type=3)
    runner, *_ = asyncio.run(type_and_readback(before, "+441234567890", before))
    assert runner.last_input_readback.reason == "native_phone_format"
    assert runner.task_state.progress.stalled_actions == 1


@pytest.mark.parametrize(
    "before,text",
    [
        ("", "New value"),
        ("old", "old "),
        ("e" + chr(0x301), "é"),
        ("line one" + chr(10) + "line two", "line one" + chr(10) * 2 + "line two"),
    ],
)
def test_confirmed_exact_value_change_advances_without_normalizing_text(before, text):
    runner, *_ = asyncio.run(type_and_readback(editor(before), text, editor(text)))
    assert runner.task_state.progress.stalled_actions == 0
    assert runner.task_state._input_progress_changes == {"A1": True}


@pytest.mark.parametrize(
    "before,text",
    [
        ("020 1234", "20 1234"),
        ("+44 1234", "0044 1234"),
        ("1234", "1235"),
        ("12-34", "12 35"),
    ],
)
def test_native_phone_real_digit_or_prefix_changes_still_advance(before, text):
    runner, *_ = asyncio.run(
        type_and_readback(
            editor(before, native_type=3), text, editor(text, native_type=3)
        )
    )
    assert runner.task_state.progress.stalled_actions == 0


def test_phone_like_text_field_does_not_enable_number_equivalence():
    before, after = editor("12-34"), editor("1234")
    before.elements[0]["hint"] = "Phone"
    after.elements[0]["hint"] = "Phone"
    runner, *_ = asyncio.run(type_and_readback(before, "1234", after))
    assert runner.task_state.progress.stalled_actions == 0


def test_hint_text_is_not_the_captured_original_input_value():
    before = editor("Search")
    before.elements[0]["isShowingHintText"] = True
    runner, *_ = asyncio.run(type_and_readback(before, "Search", editor("Search")))
    assert runner.task_state.progress.stalled_actions == 0
    assert runner.input_target.before_value_sha256 == hashlib.sha256(b"").hexdigest()


def test_raw_native_before_value_is_used_without_an_extra_snapshot_request():
    before = editor("")
    raw = {
        "phone_state": before.phone_state,
        "a11y_tree": [_raw_input("Already present", resource="body")],
    }
    runner, driver, _ = asyncio.run(
        type_and_readback(
            before,
            "Already present",
            editor("Already present"),
            raw_before=raw,
            raw_after=raw,
        )
    )
    assert (
        driver.get_ui_tree.await_count == 2
    )  # Existing capture and existing readback.
    assert runner.visual_fallback.calls == 0
    assert runner.task_state.progress.stalled_actions == 1
    assert (
        runner.input_target.before_value_sha256
        == hashlib.sha256(b"Already present").hexdigest()
    )


def test_screenshot_only_ui_uses_the_same_native_capture_for_unchanged_input():
    before = editor("Already present")
    raw = {
        "phone_state": before.phone_state,
        "a11y_tree": [_raw_input("Already present", resource="body")],
    }
    before.elements = []
    runner, driver, _ = asyncio.run(
        type_and_readback(
            before, "Already present", before, raw_before=raw, raw_after=raw
        )
    )
    assert driver.get_ui_tree.await_count == 2
    assert runner.task_state.progress.stalled_actions == 1


def test_captured_before_value_hash_and_change_flags_remain_private():
    before = editor("")
    private_value = "PRIVATE RAW ORIGINAL VALUE"
    raw_before = {
        "phone_state": before.phone_state,
        "a11y_tree": [_raw_input(private_value, resource="body")],
    }
    raw_after = {
        "phone_state": before.phone_state,
        "a11y_tree": [_raw_input("New value", resource="body")],
    }
    runner, *_ = asyncio.run(
        type_and_readback(
            before,
            "New value",
            editor("New value"),
            raw_before=raw_before,
            raw_after=raw_after,
        )
    )
    serialized = json.dumps(
        {
            "context": runner.task_state.context(),
            "evidence": runner.task_state.evidence(),
            "trace": runner.trace,
        }
    )
    assert private_value not in serialized
    assert runner.input_target.before_value_sha256 not in serialized
    assert (
        "before_value_sha256" not in serialized
        and "before_phone_sha256" not in serialized
    )
    assert "_input_progress_changes" not in serialized


def record_trusted_confirmation(target):
    state = TaskState("Fill the field")
    state.set_contract(task(state.goal, "state"))
    state.observe(editor("old"))
    state.record_action(
        ToolCall("type", {"text": "new", "clear": True}),
        ActionResult(True, "dispatched"),
        input_target=target,
    )
    state.observe(editor("new"))
    state.record_input_effect("confirmed")
    return state


@pytest.mark.parametrize("before_hash", [None, "", "f" * 63, "z" * 64, 123, True])
def test_missing_or_invalid_explicit_before_receipt_cannot_prove_input_change(
    before_hash,
):
    target = asyncio.run(capture_input_target(SimpleNamespace(), editor("old")))
    state = record_trusted_confirmation(
        replace(target, before_value_sha256=before_hash)
    )
    assert state.progress.stalled_actions == 1
    assert state._input_progress_changes == {"A1": None}
    assert state.actions[-1]["effect_status"] == "confirmed"


def test_legacy_target_without_before_hash_still_verifies_value_but_not_change():
    target = asyncio.run(capture_input_target(SimpleNamespace(), editor("old")))
    legacy = replace(target, before_value_sha256=None, before_phone_sha256=None)
    assert (
        asyncio.run(
            read_input_value(SimpleNamespace(), editor("new"), legacy, "new")
        ).status
        == "confirmed"
    )
    assert record_trusted_confirmation(legacy).progress.stalled_actions == 1


@pytest.mark.parametrize("status", ["unknown", "mismatch"])
def test_captured_difference_without_independent_confirmation_is_not_progress(status):
    target = asyncio.run(capture_input_target(SimpleNamespace(), editor("old")))
    state = TaskState("Fill the field")
    state.set_contract(task(state.goal, "state"))
    state.observe(editor("old"))
    state.record_action(
        ToolCall("type", {"text": "new", "clear": True}),
        ActionResult(True, "dispatched"),
        input_target=target,
    )
    state.observe(editor("new"))
    state.record_input_effect(status)
    assert state._input_progress_changes == {"A1": True}
    assert state.progress.stalled_actions == 1 and state.assessments == {}


def test_goal_reset_discards_the_before_value_change_receipts():
    target = asyncio.run(capture_input_target(SimpleNamespace(), editor("old")))
    state = record_trusted_confirmation(target)
    assert state.progress.stalled_actions == 0 and state._input_progress_changes
    state.reset("Inspect another field")
    assert state._input_progress_changes == {} and state._input_progress_targets == {}


def test_recovery_still_triggers_for_four_noop_inputs_without_marking_completion():
    async def run():
        before = editor("Already present")
        runner, _, provider = await type_and_readback(before, "Already present", before)
        for _ in range(3):
            call = ToolCall("type", {"text": "Already present", "clear": True})
            runner.pending = Selection("test-native", [call], snapshot=provider.ui)
            current, reason = await runner.validate_before_execute()
            assert reason is None
            runner.record_execution(call, ActionResult(True, "dispatched"), current)
            assert await runner.verify_input(call)
        assert runner.task_state.progress.stalled_actions == 4
        assert runner.task_state.progress.request_recovery(runner.task_state)
        assert runner.task_state.progress.recoveries == {1: 1}
        assert runner.task_state.assessments == {} and runner.input_retry is None

    asyncio.run(run())
