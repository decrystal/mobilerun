"""Offline behavior checks: Jev cannot invent executable device arguments."""

import asyncio
import copy
import json
import time

import httpx
import pytest

from mobilerun.agent.decision.jev import JevDecisionPolicy, snapshot_fingerprint
from mobilerun.tools.ui.state import UIState


def state(elements=None, **phone):
    return UIState(
        elements=(
            elements
            if elements is not None
            else [
                {
                    "index": 1,
                    "className": "LinearLayout",
                    "text": "",
                    "bounds": "0,50,300,150",
                    "clickable": True,
                    "children": [
                        {
                            "index": 2,
                            "className": "TextView",
                            "text": "网络和互联网",
                            "bounds": "10,60,290,140",
                            "clickable": False,
                        },
                    ],
                },
            ]
        ),
        formatted_text="",
        focused_text="",
        phone_state={
            "packageName": "com.android.settings",
            "currentApp": "Settings",
            **phone,
        },
        screen_width=400,
        screen_height=800,
    )


def choice(criteria, selected=None, confidence=0.99):
    selected = selected or next(iter(criteria))
    return {
        "type": "choice",
        "choice": selected,
        "confidence": confidence,
        "probabilities": {key: float(key == selected) for key in criteria},
    }


def test_screenshot_only_routes_to_vision_without_blind_app_decision():
    async def run():
        def reject_request(request):
            raise AssertionError("Jev cannot read a screenshot")

        async with httpx.AsyncClient(
            transport=httpx.MockTransport(reject_request)
        ) as client:
            policy = JevDecisionPolicy(api_key="offline", client=client)
            return await policy.decide(
                goal="打开设置",
                ui_state=state([], observationMode="screenshot_only"),
                apps=[{"packageName": "com.android.settings", "label": "设置"}],
            )

    decision = asyncio.run(run())
    assert decision.status == "fallback"
    assert decision.reason == "visual_observation_requires_vlm"
    assert decision.api_called is False


def response_for(body, selections=None):
    selections = selections or {"operation": "TAP"}
    return {
        "model": "jev-1.13.0",
        "answers": {
            name: choice(q["criteria"], selections.get(name))
            for name, q in body["questions"].items()
        },
        "usage": {"input_tokens": 100, "output_tokens": 20},
    }


def run_decision(ui=None, selections=None, mutate=None, **kwargs):
    requests = []

    async def run():
        def handler(request):
            body = json.loads(request.content)
            requests.append(body)
            result = response_for(body, selections)
            if mutate:
                mutate(result)
            return httpx.Response(200, json=result)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            policy = JevDecisionPolicy(api_key="test-secret", client=client)
            result = await policy.decide(
                goal="进入网络设置", ui_state=ui or state(), **kwargs
            )
            await policy.aclose()
            assert not client.is_closed  # The caller owns injected clients.
            return result

    return asyncio.run(run()), requests


def test_long_press_uses_observed_bounds_and_rejects_a_moved_target():
    ui = state(
        [
            {
                "index": 1,
                "className": "ImageView",
                "text": "Photo",
                "bounds": "20,100,220,300",
                "clickable": False,
            }
        ]
    )
    decision, requests = run_decision(
        ui, {"operation": "LONG_PRESS", "long_press_target": "1"}
    )
    assert decision.tool_call.name == "long_press_at"
    assert decision.tool_call.parameters == {"x": 120, "y": 200}
    assert decision.validate(ui) is None
    ui.elements[0]["bounds"] = "20,120,220,320"
    assert decision.validate(ui) == "target_bounds_changed"


def test_long_press_respects_resized_screenshot_coordinate_contract():
    ui = state([{"index": 1, "className": "ImageView", "bounds": "20,100,220,300"}])
    ui.coordinate_scale_x = 2
    ui.coordinate_scale_y = 2
    decision, _ = run_decision(
        ui, {"operation": "LONG_PRESS", "long_press_target": "1"}
    )
    assert decision.tool_call.parameters == {"x": 60, "y": 100}


def test_unavailable_long_press_is_not_offered():
    _, requests = run_decision(available_tools={"click"})
    assert "LONG_PRESS" not in requests[0]["questions"]["operation"]["criteria"]


def test_natural_language_input_can_request_model_resolution_without_text_flag():
    ui = state(
        [{"index": 1, "className": "EditText", "bounds": "0,0,300,60", "focused": True}]
    )
    decision, requests = run_decision(
        ui,
        {"operation": "RESOLVE_INPUT"},
        guidance="Enter the user's requested value",
        task_state={"progress_note": "Search stage"},
    )
    assert decision.status == "fallback"
    assert decision.reason == "text_value_required"
    assert decision.tool_call is None
    assert requests[0]["state"]["taskState"]["progress_note"] == "Search stage"
    assert (
        "Enter the user's requested value"
        in requests[0]["questions"]["operation"]["instructions"]["rules"]
    )


def test_uncertain_input_resolution_still_delegates_without_executing():
    def low_confidence(response):
        response["answers"]["operation"]["confidence"] = 0.3

    decision, _ = run_decision(
        input_state(), {"operation": "RESOLVE_INPUT"}, mutate=low_confidence
    )
    assert decision.status == "fallback"
    assert decision.reason == "text_value_required"
    assert decision.tool_call is None


def test_text_long_press_requires_glyph_grounding_instead_of_field_center():
    decision, _ = run_decision(
        input_state(), {"operation": "LONG_PRESS", "long_press_target": "7"}
    )
    assert decision.status == "fallback"
    assert decision.reason == "text_long_press_requires_visual_grounding"
    assert decision.tool_call is None
    assert decision.target_guard.index == 7


def test_one_request_selects_operation_and_parent_with_child_label():
    decision, requests = run_decision()
    assert len(requests) == 1
    assert requests[0]["questions"]["tap_target"]["criteria"] == {"1": "网络和互联网"}
    assert decision.status == "action"
    assert decision.tool_call.name == "click"
    assert decision.tool_call.parameters == {"index": 1}
    assert decision.validate(state()) is None
    assert requests[0]["state"]["elements"][0]["bounds"] == "0,50,300,150"
    assert "test-secret" not in repr(decision)


@pytest.mark.parametrize(
    "mutate",
    [
        lambda a: a.update(choice="invented"),
        lambda a: a.update(confidence=float("nan")),
        lambda a: a.update(confidence=True),
        lambda a: a.update(probabilities={"TAP": 1}),
        lambda a: a["probabilities"].update(invented=0),
        lambda a: a["probabilities"].update(TAP=-0.1),
        lambda a: a["probabilities"].update(TAP=0.8),
        lambda a: a.update(choice="WAIT"),
        lambda a: a.update(type="noul"),
    ],
)
def test_invalid_operation_distributions_never_produce_a_tool(mutate):
    def corrupt(result):
        mutate(result["answers"]["operation"])

    decision, _ = run_decision(mutate=corrupt)
    assert decision.status == "fallback"
    assert decision.tool_call is None
    assert decision.reason in {"invalid_response", "invalid_json"}


def test_missing_selected_target_rejected_but_unused_heads_not_consumed():
    decision, _ = run_decision(mutate=lambda r: r["answers"].pop("tap_target"))
    assert decision.reason == "invalid_response"
    decision, _ = run_decision(
        selections={"operation": "WAIT"},
        mutate=lambda r: r["answers"].update(tap_target={"choice": "evil"}),
    )
    assert decision.tool_call.name == "wait"


@pytest.mark.parametrize(
    "head,reason",
    [
        ("operation", "low_operation_confidence"),
        ("tap_target", "low_target_confidence"),
    ],
)
def test_low_confidence_is_an_observable_fallback(head, reason):
    decision, _ = run_decision(
        mutate=lambda r: r["answers"][head].update(confidence=0.1)
    )
    assert decision.reason == reason
    assert decision.tool_call is None


def test_empty_or_disabled_candidate_space_does_not_call_the_api():
    for ui, kwargs in [
        (state([]), {}),
        (state(), {"available_tools": []}),
        (state([{"index": 1, "bounds": "0,0,100,100", "enabled": False}]), {}),
    ]:
        decision, requests = run_decision(ui=ui, **kwargs)
        assert decision.reason == "no_candidates"
        assert requests == []


def test_candidate_overflow_is_not_silently_truncated():
    ui = state(
        [
            {"index": i, "bounds": "0,0,20,20", "text": str(i), "clickable": True}
            for i in range(256)
        ]
    )
    decision, requests = run_decision(ui)
    assert decision.reason == "too_many_candidates"
    assert requests == []


def input_state(**extra):
    node = {
        "index": 7,
        "className": "EditText",
        "text": "旧值",
        "bounds": "20,100,300,150",
        "editable": True,
        "focused": True,
        **extra,
    }
    return state([node], isEditable=True)


def test_chinese_input_uses_only_explicit_exact_value_and_clear():
    decision, requests = run_decision(
        input_state(),
        texts=["北京南站"],
        selections={"operation": "TYPE_TEXT", "text_value": "0"},
    )
    assert requests[0]["questions"]["text_value"]["criteria"]["0"] == "北京南站"
    assert decision.tool_call.name == "type"
    assert decision.tool_call.parameters == {"text": "北京南站", "clear": True}
    assert decision.target_guard.index == 7
    assert decision.validate(input_state()) is None


def test_web_container_labels_do_not_repeat_the_whole_page_in_candidates():
    from mobilerun.agent.decision.jev import _label

    nodes = [{"index": 1, "text": "Web page"}]
    nodes.extend(
        {"index": i + 2, "parentIndex": 1, "text": str(i) * 80} for i in range(50)
    )
    label = _label(nodes[0], nodes)
    assert label.startswith("Web page / 000")
    assert len(label) <= 360
    assert "49" * 80 not in label


def test_already_focused_input_cannot_be_refocused_instead_of_typing():
    ui = input_state(inputValue="old value")
    ui.elements.append(
        {"index": 8, "text": "Search", "clickable": True, "bounds": "300,100,390,150"}
    )
    decision, requests = run_decision(ui, selections={"operation": "RESOLVE_INPUT"})
    assert decision.reason == "text_value_required"
    body = requests[0]
    assert set(body["questions"]["tap_target"]["criteria"]) == {"8"}
    assert body["state"]["focusedInput"] == 7
    assert body["state"]["focusedInputValue"] == "old value"
    assert body["state"]["screen"]["width"] == 400
    assert body["state"]["elements"][0]["bounds"] == "20,100,300,150"
    assert body["state"]["elements"][0]["focused"] is True


def test_no_auto_extraction_of_chinese_goal_or_password_entry():
    for ui, texts in [
        (input_state(), []),
        (input_state(password=True), ["秘密"]),
        (state(), ["北京南站"]),
    ]:
        _, requests = run_decision(ui, texts=texts, selections={"operation": "BACK"})
        assert "TYPE_TEXT" not in requests[0]["questions"]["operation"]["criteria"]
        assert "text_value" not in requests[0]["questions"]


def test_none_text_and_blocked_missing_text_have_distinct_reasons():
    decision, _ = run_decision(
        input_state(),
        texts=["北京南站"],
        selections={"operation": "TYPE_TEXT", "text_value": "NONE"},
    )
    assert decision.reason == "no_matching_explicit_text"
    decision, _ = run_decision(input_state(), selections={"operation": "BLOCKED"})
    assert decision.reason == "missing_explicit_text"


def test_explicit_text_parameter_is_a_sequence_of_complete_values():
    decision, requests = run_decision(input_state(), texts="北京南站")
    assert decision.reason == "invalid_explicit_texts"
    assert requests == []
    decision, requests = run_decision(input_state(), texts=[42])
    assert decision.reason == "invalid_explicit_texts"
    assert requests == []


def test_open_app_uses_installed_package_and_no_app_opener_llm():
    decision, _ = run_decision(
        state([]),
        apps=[{"packageName": "com.example.map", "label": "地图"}],
        selections={"operation": "OPEN_APP", "app_target": "com.example.map"},
    )
    assert decision.tool_call.name == "launch_app"
    assert decision.tool_call.parameters == {"package_name": "com.example.map"}


def test_done_never_invokes_complete():
    decision, _ = run_decision(selections={"operation": "DONE"})
    assert decision.status == "done"
    assert decision.tool_call is None
    assert decision.reason == "model_requested_completion_verification"


def test_reused_index_and_changed_child_content_are_stale():
    decision, _ = run_decision()
    changed = state()
    changed.elements[0]["children"][0]["text"] = "恢复出厂设置"
    assert decision.validate(changed) == "target_changed"
    assert decision.validate(state([])) == "target_missing"
    assert decision.validate(state(packageName="com.other.app")) == "page_changed"
    decision.observed_at = time.monotonic() - 31
    assert decision.validate(state()) == "snapshot_expired"


def test_click_can_follow_same_target_moving_but_fingerprint_changes():
    ui = state()
    decision, _ = run_decision(ui)
    moved = copy.deepcopy(ui)
    moved.elements[0]["bounds"] = "0,200,300,300"
    assert snapshot_fingerprint(ui) != snapshot_fingerprint(moved)
    assert decision.validate(moved) is None


def test_flat_parent_indices_merge_children_and_guard_child_reuse():
    ui = state(
        [
            {
                "index": 1,
                "bounds": "0,0,300,200",
                "clickable": True,
                "text": "",
                "parentIndex": None,
            },
            {
                "index": 2,
                "bounds": "0,0,300,100",
                "clickable": False,
                "text": "无线网络",
                "parentIndex": 1,
            },
        ]
    )
    decision, requests = run_decision(ui)
    assert requests[0]["questions"]["tap_target"]["criteria"]["1"] == "无线网络"
    ui.elements[1]["text"] = "删除网络"
    assert decision.validate(ui) == "target_changed"


def test_scroll_coordinates_come_from_observed_bounds_and_cannot_be_model_generated():
    ui = state(
        [
            {
                "index": 3,
                "bounds": "0,100,400,700",
                "scrollable": True,
                "clickable": False,
            }
        ]
    )
    decision, _ = run_decision(
        ui,
        selections={"operation": "SCROLL_DOWN"},
        mutate=lambda r: r["answers"]["scroll_target"].update(coordinate=[999, 999]),
    )
    assert decision.tool_call.name == "swipe"
    assert decision.tool_call.parameters == {
        "coordinate": [200, 580],
        "coordinate2": [200, 220],
        "duration": 0.3,
    }
    ui.elements[0]["bounds"] = "0,150,400,750"
    assert decision.validate(ui) == "target_bounds_changed"


def test_scroll_uses_display_coordinate_contract():
    ui = state(
        [
            {
                "index": 3,
                "bounds": "0,100,400,700",
                "scrollable": True,
                "clickable": False,
            }
        ]
    )
    ui.coordinate_scale_x = ui.coordinate_scale_y = 2.0
    decision, _ = run_decision(ui, selections={"operation": "SCROLL_DOWN"})
    assert decision.tool_call.parameters["coordinate"] == [100, 290]
    assert decision.tool_call.parameters["coordinate2"] == [100, 110]


def test_input_guard_checks_focus_not_only_index():
    ui = input_state()
    decision, _ = run_decision(
        ui, texts=["北京南站"], selections={"operation": "TYPE_TEXT"}
    )
    ui.elements[0]["focused"] = False
    ui.phone_state["isEditable"] = False
    assert decision.validate(ui) == "target_changed"


def test_request_error_never_echoes_response_body_or_key(monkeypatch):
    monkeypatch.setenv("TYPESAFE_API_KEY", "environment-secret")

    async def run():
        seen = []

        def handler(request):
            seen.append(request.headers["Authorization"])
            return httpx.Response(401, text="environment-secret leaked by remote error")

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            policy = JevDecisionPolicy(client=client)
            result = await policy.decide(goal="打开设置", ui_state=state())
        assert seen == ["Bearer environment-secret"]
        assert result.reason == "http_status_401"
        assert "environment-secret" not in repr(result)

    asyncio.run(run())


def test_total_deadline_and_cancellation_are_different():
    async def run(cancel):
        entered = asyncio.Event()

        async def handler(request):
            entered.set()
            await asyncio.sleep(10)
            return httpx.Response(200)

        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            policy = JevDecisionPolicy(
                api_key="test", client=client, timeout=0.01 if not cancel else 10
            )
            task = asyncio.create_task(policy.decide(goal="打开设置", ui_state=state()))
            await entered.wait()
            if cancel:
                task.cancel()
                with pytest.raises(asyncio.CancelledError):
                    await task
            else:
                assert (await task).reason == "request_timeout"

    asyncio.run(run(False))
    asyncio.run(run(True))
