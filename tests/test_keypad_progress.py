"""Native intermediate input evidence never substitutes for completion."""

import copy

import pytest
from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.decision.task_state import TaskState
from mobilerun.agent.fast_agent.xml_parser import ToolCall
from test_task_evidence import task, ui


def screen(value="0", *, package="com.example.form", change=None):
    keys = [
        {
            "index": index + 1,
            "resourceId": f"digit_{index}",
            "className": "Button",
            "rawText": str(index),
            "text": str(index),
            "isEnabled": True,
        }
        for index in range(10)
    ]
    display = {
        "index": 11,
        "resourceId": "entry_value",
        "className": "TextView",
        "rawText": value,
    }
    nodes = keys + [display]
    if change == "missing_key":
        nodes.pop(9)
    elif change == "duplicate_key":
        nodes.append({**keys[2], "index": 15})
    elif change == "ambiguous_display":
        nodes.append({**display, "index": 16})
    elif change == "changed_identity":
        display["resourceId"] = "unrelated_value"
    elif change == "hint":
        display["isShowingHintText"] = True
    elif change == "checkbox":
        display["checkable"] = True
    elif change == "range":
        display["rangeInfo"] = {"current": 1}
    return ui(nodes, package=package)


def state(goal="Enter 150543 into the form", kind="state"):
    result = TaskState(goal)
    result.set_contract(task(goal, kind))
    return result


def press(result, digit, after, *, success=True, tool="click"):
    result.record_action(
        ToolCall(tool, {"index": int(digit) + 1}), ActionResult(success, "dispatched")
    )
    result.observe(after)
    return result.actions[-1]


def test_each_exact_keypad_readback_is_progress_without_supporting_the_requirement():
    result = state()
    result.observe(screen("00h 00m 00s"))
    digits = ""
    for digit in "150543":
        digits += digit
        padded = digits.zfill(6)
        value = f"{padded[:2]}h {padded[2:4]}m {padded[4:]}s"
        action = press(result, digit, screen(value))
        effect = action["intermediate_effect"]
        assert effect["after_value"] == value and effect["after_digits"] == str(
            int(digits)
        )
        assert (
            effect["requirement_id"] == 1 and effect["requirement_satisfied"] is False
        )
        assert (
            effect["before_ref"] in result.evidence()
            and effect["after_ref"] in result.evidence()
        )
        assert result.progress.stalled_actions == 0 and result.assessments == {}
    assert not result.progress.request_recovery(result)


def test_backtracking_to_an_already_observed_input_cannot_reset_stalls():
    result = state()
    result.observe(screen())
    press(result, "1", screen("1"))
    result.record_action(
        ToolCall("click", {"index": 99}), ActionResult(True, "cleared")
    )
    result.observe(screen())
    press(result, "1", screen("1"))
    assert result.progress.stalled_actions == 2 and result.assessments == {}


@pytest.mark.parametrize(
    "change",
    ["missing_key", "duplicate_key", "ambiguous_display", "hint", "checkbox", "range"],
)
def test_unbound_or_unreliable_native_keypad_evidence_is_not_progress(change):
    result = state()
    result.observe(screen(change=change))
    action = press(result, "1", screen("1", change=change))
    assert "intermediate_effect" not in action and result.progress.stalled_actions == 1


@pytest.mark.parametrize(
    "change",
    [
        "wrong_value",
        "unchanged",
        "changed_identity",
        "new_package",
        "unknown_dispatch",
        "other_tool",
    ],
)
def test_dispatch_or_arbitrary_property_change_does_not_prove_an_input_effect(change):
    result = state()
    result.observe(screen("1"))
    after = screen("12")
    if change == "wrong_value":
        after = screen("3")
    elif change == "unchanged":
        after = screen("1")
    elif change == "changed_identity":
        after = screen("12", change=change)
    elif change == "new_package":
        after = screen("12", package="com.example.other")
    action = press(
        result,
        "2",
        after,
        success=change != "unknown_dispatch",
        tool="wait" if change == "other_tool" else "click",
    )
    assert "intermediate_effect" not in action and result.progress.stalled_actions == 1


@pytest.mark.parametrize(
    "kind,goal",
    [
        ("query", "Count 15 items"),
        ("process", "Tap 15 times"),
        ("state", "Open the form"),
    ],
)
def test_only_an_unresolved_numeric_state_requirement_can_bind_input_progress(
    kind, goal
):
    result = state(goal, kind)
    result.observe(screen())
    action = press(result, "1", screen("1"))
    assert "intermediate_effect" not in action


def test_supported_requirement_does_not_bind_unrelated_followup_input():
    result = state()
    result.assessments[1] = {"status": "supported"}
    result.observe(screen())
    assert "intermediate_effect" not in press(result, "1", screen("1"))


def test_known_autonomous_readout_does_not_become_progress_on_an_accidental_match():
    result = state()
    result.observe(screen("1"))
    result.record_action(
        ToolCall("wait", {"duration": 0.5}), ActionResult(True, "waited")
    )
    result.observe(screen("12"))
    action = press(result, "4", screen("124"))
    assert "intermediate_effect" not in action and result.progress.stalled_actions == 2


def test_two_exact_candidate_readbacks_are_ambiguous_and_not_progress():
    result = state()
    before, after = screen(), screen("1")
    before.elements.append(
        {**copy.deepcopy(before.elements[-1]), "index": 17, "resourceId": "other"}
    )
    after.elements.append(
        {**copy.deepcopy(after.elements[-1]), "index": 17, "resourceId": "other"}
    )
    result.observe(before)
    assert "intermediate_effect" not in press(result, "1", after)


def test_new_goal_clears_intermediate_readback_history():
    result = state()
    result.observe(screen())
    press(result, "1", screen("1"))
    result.reset("Another numeric task 123")
    assert result.actions == [] and result.progress.signature is None
