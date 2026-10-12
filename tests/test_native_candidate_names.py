"""Full native names remain visible and bound when rendered labels truncate."""

import asyncio
import copy
import json

import httpx
import pytest
from test_jev_decision import response_for, run_decision, state

from mobilerun.agent.decision.jev import (
    JevDecisionPolicy,
    _label,
    control_observations,
    coordinate_fingerprint,
)


def row(*, name="monthly_report_final.md", parent=False):
    label = {
        "index": 2 if parent else 1,
        "className": "TextView",
        "bounds": "20,80,280,140",
        "text": "monthly_report…",
        "rawText": name,
        "clickable": not parent,
    }
    if parent:
        return {
            "index": 1,
            "className": "LinearLayout",
            "bounds": "0,60,300,160",
            "clickable": True,
            "children": [label],
        }
    return label


@pytest.mark.parametrize("parent", [False, True])
def test_changed_full_name_with_same_rendered_label_invalidates_the_selected_target(
    parent,
):
    page = state([row(parent=parent)])
    selected, requests = run_decision(page, {"operation": "TAP", "tap_target": "1"})
    assert len(requests) == 1
    changed = copy.deepcopy(page)
    label = changed.elements[0]["children"][0] if parent else changed.elements[0]
    label["rawText"] = "monthly_report_final_backup.md"
    assert selected.validate(changed) == "target_changed"


def test_changed_full_name_invalidates_a_native_coordinate_region():
    page = state([row()])
    original = coordinate_fingerprint(page, 100, 100)
    page.elements[0]["rawText"] = "monthly_report_final_backup.md"
    assert coordinate_fingerprint(page, 100, 100) != original


def test_candidate_uses_observed_full_name_when_rendered_name_is_truncated():
    node = row()
    assert node["rawText"] in _label(node, [node])


@pytest.mark.parametrize("suffix", ["_backup.md", "_old.md", ".md.zip"])
def test_raw_names_with_same_rendered_text_keep_distinct_candidate_labels(suffix):
    one, two = row(), row(name="monthly_report_final" + suffix)
    assert _label(one, [one]) != _label(two, [two])


@pytest.mark.parametrize("length", [400, 2000])
def test_long_names_preserve_tail_and_explicit_goal_span_inside_bounded_label(length):
    name = "research_" + "x" * length + "_final.md"
    goal = "Open the file named " + name + "."
    node = row(name=name)
    label = _label(node, [node], goal=goal)
    assert len(label) <= 360
    assert "_final.md" in label
    assert "goal_rawText_span=" in label
    marker = label.split("goal_rawText_span=", 1)[1].split(";", 1)[0]
    start, end = [int(part) for part in marker.split(":")]
    assert goal[start:end] == name
    assert "elided" in label


def test_long_native_text_not_in_goal_gets_no_literal_match_claim():
    node = row(name="a" * 1000 + "_backup.md")
    label = _label(node, [node], goal="Open the requested file")
    assert len(label) <= 360
    assert "_backup.md" in label
    assert "goal_rawText_span" not in label


def test_matching_text_beyond_first_six_labels_is_not_silently_certified():
    parent = {"index": 1, "className": "LinearLayout", "bounds": "0,0,300,600"}
    nodes = (
        [parent]
        + [{"index": i + 2, "parentIndex": 1, "text": str(i) * 80} for i in range(6)]
        + [{"index": 8, "parentIndex": 1, "rawText": "hidden_name.md"}]
    )
    assert "goal_rawText_span" not in _label(parent, nodes, goal="Open hidden_name.md")


@pytest.mark.parametrize("flag", ["password", "isPassword"])
def test_password_raw_names_are_not_advertised_or_matched_to_goal(flag):
    node = row(name="private-" + "x" * 1000)
    node[flag] = True
    label = _label(node, [node], goal="Open " + node["rawText"])
    assert "private-" not in label
    assert "goal_rawText_span" not in label
    assert node["rawText"] not in str(control_observations(state([node])))


@pytest.mark.parametrize("raw", [None, "", True, 7, {"text": "unknown"}])
def test_non_text_raw_values_do_not_become_object_names(raw):
    node = row()
    node["rawText"] = raw
    label = _label(node, [node], goal="Open unknown")
    assert label == "monthly_report…"


def test_full_raw_native_control_text_preserves_the_rendered_label_too():
    page = state([row()])
    text = control_observations(page)["controls"][0]["text"]
    assert "monthly_report_final.md" in text
    assert "monthly_report…" in text


def test_native_candidate_information_is_bounded_and_transmitted_in_one_jev_request():
    name = "archive_" + "x" * 1000 + "_final.md"
    goal = "Open the file named " + name
    requests = []

    async def run():
        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            labels = payload["questions"]["tap_target"]["criteria"]
            assert "_final.md" in labels["1"]
            assert "goal_rawText_span=" in labels["1"]
            assert all(len(label) <= 360 for label in labels.values())
            return httpx.Response(
                200, json=response_for(payload, {"operation": "TAP", "tap_target": "1"})
            )

        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            policy = JevDecisionPolicy(api_key="offline-test-key", client=client)
            return await policy.decide(goal=goal, ui_state=state([row(name=name)]))

    selected = asyncio.run(run())
    assert selected.status == "action"
    assert len(requests) == 1
    assert selected.tool_call.parameters == {"index": 1}


def test_unique_target_motion_with_same_full_name_still_validates():
    page = state([row()])
    selected, _ = run_decision(page, {"operation": "TAP", "tap_target": "1"})
    page.elements[0]["bounds"] = "20,100,280,160"
    assert selected.validate(page) is None
