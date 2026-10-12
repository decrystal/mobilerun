"""Offline integration checks for Jev-first routing and independent verification."""

import asyncio
import copy
import hashlib
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import httpx
import pytest
from llama_index.core.base.llms.types import ChatMessage, ChatResponse

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.decision.autoglm import (
    AutoGLMDecision,
    AutoGLMToolCall,
    parse_autoglm_decision,
)
from mobilerun.agent.decision.jev import JevDecision, JevDecisionPolicy
from mobilerun.agent.decision.runtime import (
    DecisionRuntime,
    Selection,
    completion_matches,
    launch_app,
    register_decision_tools,
    validate_completion_criteria,
)
from mobilerun.agent.droid.state import MobileAgentState
from mobilerun.agent.fast_agent.fast_agent import FastAgent
from mobilerun.agent.fast_agent.xml_parser import ToolCall
from mobilerun.agent.tool_registry import ToolRegistry
from mobilerun.tools.ui.state import UIState


def ui_state(nodes=None, **phone):
    return UIState(
        elements=(
            nodes
            if nodes is not None
            else [
                {
                    "index": 1,
                    "text": "设置",
                    "bounds": "0,0,400,80",
                    "clickable": False,
                },
                {
                    "index": 2,
                    "text": "WLAN",
                    "bounds": "0,80,400,160",
                    "clickable": True,
                },
            ]
        ),
        formatted_text="设置 / WLAN",
        focused_text="",
        phone_state={
            "packageName": "com.android.settings",
            "currentApp": "Settings.Home",
            **phone,
        },
        screen_width=400,
        screen_height=800,
    )


def config(**overrides):
    return SimpleNamespace(
        **{
            "completion": {},
            "model": "jev-1.13.0",
            "timeout": 1,
            "confidence_threshold": 0.5,
            "fallback": "autoglm",
            "fallback_model": "autoglm-phone",
            "reasoning_model": "glm-4.6v",
            "fallback_base_url": "https://open.bigmodel.cn/api/paas/v4",
            "fallback_api_key_env": "BIGMODEL_API_KEY",
            "fallback_thinking": None,
            "fallback_timeout": 1,
            "text_values": [],
            **overrides,
        }
    )


class Driver:
    def __init__(self, events, apps=None):
        self.events = events
        self.apps = (
            apps
            if apps is not None
            else [{"package_name": "com.android.settings", "label": "设置"}]
        )
        self.started = []

    async def get_apps(self, **kwargs):
        self.events.append("inventory")
        return self.apps

    async def screenshot(self):
        self.events.append("screenshot")
        return b"mock-screenshot"

    async def start_app(self, package):
        self.started.append(package)
        return "launched"


class Provider:
    requires_coordinate_tools = False

    def __init__(self, ui):
        self.ui = ui

    async def get_state(self):
        return self.ui


class Policy:
    def __init__(self, events, result=None):
        self.events = events
        self.requests = []
        self.result = result or JevDecision(
            "action",
            tool_call=ToolCall("click", {"index": 2}),
            operation="TAP",
            confidence=0.99,
        )

    async def decide(self, **kwargs):
        self.events.append("jev")
        self.requests.append(kwargs)
        return self.result


class Visual:
    def __init__(self, events, result=None):
        self.events = events
        self.requests = []
        self.result = result or AutoGLMDecision(
            "action", calls=(AutoGLMToolCall("click_at", {"x": 100, "y": 200}),)
        )

    async def decide(self, **kwargs):
        self.events.append("autoglm")
        self.requests.append(kwargs)
        return self.result


def runtime(
    ui=None, settings=None, policy_result=None, visual_result=None, verifier=None
):
    ui = ui or ui_state()
    events = []
    policy = Policy(events, policy_result)
    visual = Visual(events, visual_result)
    ctx = SimpleNamespace(driver=Driver(events), ui=ui)
    registry = SimpleNamespace(
        tools={
            name: object()
            for name in (
                "click",
                "click_at",
                "type",
                "wait",
                "system_button",
                "launch_app",
                "complete",
            )
        }
    )
    provider = Provider(ui)
    runner = DecisionRuntime(
        settings or config(),
        ctx,
        registry,
        provider,
        policy=policy,
        visual_fallback=visual,
        verifier=verifier,
    )
    return runner, events, policy, visual, provider


@pytest.mark.parametrize(
    "criteria",
    [
        {"unknown": "x"},
        {"package": ""},
        {"text_all": []},
        {"text_any": "WLAN"},
        {"checked": {"蓝牙": 1}},
        {"input_text": False},
        {"checked": {}},
    ],
)
def test_invalid_completion_contract_is_rejected(criteria):
    with pytest.raises(ValueError):
        validate_completion_criteria(criteria)


def test_completion_requires_all_explicit_postconditions():
    ui = ui_state(
        [
            {"index": 1, "text": "无线网络设置", "bounds": "0,0,200,80"},
            {"index": 2, "text": "使用WLAN", "checked": True, "bounds": "0,80,200,160"},
            {
                "index": 3,
                "text": "北京南站",
                "editable": True,
                "bounds": "0,160,200,220",
            },
        ],
        currentApp="Settings.WifiSettings",
    )
    criteria = {
        "package": "com.android.settings",
        "activity_contains": "WifiSettings",
        "text_all": ["无线网络", "使用WLAN"],
        "text_any": ["网络设置", "蓝牙设置"],
        "input_text": "北京南站",
        "checked": {"使用WLAN": True},
    }
    assert completion_matches(ui, criteria)
    assert not completion_matches(ui, {})
    assert not completion_matches(ui, {**criteria, "package": "com.example.map"})
    assert not completion_matches(ui, {**criteria, "input_text": "北京"})
    assert not completion_matches(ui, {**criteria, "checked": {"使用WLAN": False}})


def test_checked_verifier_understands_existing_formatter_and_rejects_ambiguity():
    node = {
        "index": 1,
        "text": "使用WLAN",
        "checkedState": "isChecked=True",
        "bounds": "0,0,200,80",
    }
    assert completion_matches(ui_state([node]), {"checked": {"使用WLAN": True}})
    assert not completion_matches(
        ui_state([node, {**node, "index": 2}]), {"checked": {"使用WLAN": True}}
    )


def test_initial_matching_state_requires_model_done_before_completion():
    runner, events, policy, visual, _ = runtime(
        settings=config(completion={"package": "com.android.settings"}),
        policy_result=JevDecision("done", operation="DONE", confidence=1),
        visual_result=AutoGLMDecision("done", message="Settings is visible."),
    )
    selection = asyncio.run(runner.select("打开设置", runner.ctx.ui))
    assert selection.source == "verifier"
    assert selection.calls[0].name == "complete"
    assert selection.calls[0].parameters["success"] is True
    assert events == ["inventory", "jev", "screenshot", "autoglm"]
    assert len(policy.requests) == 1
    assert visual.requests[0]["completion_review"] is True


@pytest.mark.parametrize("use_external_verifier", [False, True])
@pytest.mark.parametrize("fallback", [False, True])
def test_app_match_does_not_skip_search_and_save_actions(
    fallback, use_external_verifier
):
    goal = "打开小红书，然后搜索anna 田中杏奈壁纸，保存前10篇帖子的图片"
    runner, events, policy, visual, _ = runtime(
        ui=ui_state(packageName="com.xingin.xhs"),
        settings=config(completion={"package": "com.xingin.xhs"}),
        policy_result=(
            JevDecision("fallback", reason="request_timeout") if fallback else None
        ),
        verifier=(lambda _: True) if use_external_verifier else None,
    )
    selection = asyncio.run(runner.select(goal, runner.ctx.ui))
    assert selection.source == ("autoglm" if fallback else "jev")
    assert selection.calls[0].name == ("click_at" if fallback else "click")
    assert policy.requests[0]["goal"] == goal
    assert not selection.stop
    assert not any(row["event"] == "verification" for row in runner.trace)
    if fallback:
        assert visual.requests[0]["goal"] == goal
        assert events.index("jev") < events.index("autoglm")


def test_jev_done_is_checked_after_inference_and_rechecked_before_execution():
    runner, _, policy, _, provider = runtime(
        ui=ui_state(packageName="com.android.launcher"),
        settings=config(completion={"package": "com.xingin.xhs"}),
        visual_result=AutoGLMDecision("done", message="The app is open."),
    )

    async def finish(**kwargs):
        provider.ui = ui_state(packageName="com.xingin.xhs")
        return JevDecision("done", operation="DONE", confidence=1)

    policy.decide = finish

    async def run():
        selection = await runner.select("打开小红书", runner.ctx.ui)
        assert selection.calls[0].name == "complete"
        assert selection.snapshot is provider.ui
        provider.ui = ui_state(packageName="com.android.launcher")
        _, reason = await runner.validate_before_execute()
        assert reason == "completion_unverified"

    asyncio.run(run())


def test_jev_action_is_first_and_never_calls_fallback_or_screenshot():
    runner, events, policy, visual, _ = runtime(
        settings=config(text_values=["北京南站"])
    )

    async def run():
        first = await runner.select("打开WLAN", runner.ctx.ui)
        await runner.select("打开WLAN", runner.ctx.ui)
        return first

    selection = asyncio.run(run())
    assert selection.source == "jev"
    assert selection.calls == [ToolCall("click", {"index": 2})]
    assert events == [
        "inventory",
        "jev",
        "jev",
    ]  # Inventory is reused; zero fallback calls.
    assert len(policy.requests) == 2
    assert visual.requests == []
    assert policy.requests[0]["texts"] == ["北京南站"]


def test_jev_failure_then_autoglm_order_and_feedback_are_explicit():
    runner, events, _, visual, _ = runtime(
        policy_result=JevDecision("fallback", reason="request_timeout")
    )
    selection = asyncio.run(runner.select("打开WLAN", runner.ctx.ui))
    assert events == ["inventory", "jev", "screenshot", "autoglm"]
    assert selection.source == "autoglm"
    assert selection.calls == [ToolCall("click_at", {"x": 100, "y": 200})]
    assert "request_timeout" in visual.requests[0]["feedback"]
    assert [row["source"] for row in runner.trace if row["event"] == "decision"] == [
        "jev",
        "autoglm",
    ]


@pytest.mark.parametrize("change", ["meaning", "bounds", "overlay", "package"])
def test_native_fallback_reference_is_rechecked_before_device_input(change):
    from mobilerun.agent.decision.autoglm import parse_autoglm_decision

    state = ui_state()
    proposed = parse_autoglm_decision(
        '{"action":"Tap","index":2}',
        ui_state=state,
        screenshot_size=(400, 800),
        available_tools={"click"},
        structured=True,
    )
    runner, _, _, _, provider = runtime(
        ui=state,
        policy_result=JevDecision("fallback", reason="uncertain"),
        visual_result=proposed,
    )

    async def run():
        selected = await runner.select("Open network settings", state)
        assert selected.calls == [ToolCall("click", {"index": 2})]
        provider.ui = copy.deepcopy(state)
        if change == "meaning":
            provider.ui.elements[1]["text"] = "Delete"
        elif change == "bounds":
            provider.ui.elements[1]["bounds"] = "0,300,400,380"
        elif change == "overlay":
            provider.ui.elements.append(
                {"index": 3, "text": "Confirm", "bounds": "0,80,400,160"}
            )
        else:
            provider.ui.phone_state["packageName"] = "another.app"
        _, reason = await runner.validate_before_execute()
        assert reason == "stale_fallback_state"
        assert not any(event["event"] == "action" for event in runner.trace)

    asyncio.run(run())


def test_explicit_assertions_still_reject_incorrect_model_completion():
    runner, events, _, _, _ = runtime(
        settings=config(completion={"text_all": ["not on this screen"]}),
        policy_result=JevDecision("done", operation="DONE", confidence=1),
        visual_result=AutoGLMDecision("done", message="完成了"),
    )
    selection = asyncio.run(runner.select("WLAN已打开", runner.ctx.ui))
    assert selection.stop is True
    assert selection.calls == []
    assert selection.reason == "completion_unverified"
    assert events.index("jev") < events.index("autoglm")


@pytest.mark.parametrize("with_assertions", [False, True])
def test_verified_final_answer_reaches_completion_tool_and_shared_state(
    with_assertions,
):
    from mobilerun.agent.droid.state import MobileAgentState
    from mobilerun.agent.utils.signatures import build_tool_registry

    runner, _, _, _, _ = runtime(
        settings=config(
            completion={"package": "com.android.settings"} if with_assertions else {}
        ),
        policy_result=JevDecision("done", operation="DONE", confidence=1),
        visual_result=AutoGLMDecision(
            "done", message="Verified from device", final_answer="14:00"
        ),
    )
    runner.ctx.shared_state = MobileAgentState()

    async def run():
        selection = await runner.select("What time is the meeting?", runner.ctx.ui)
        assert selection.calls[0].parameters["final_answer"] == "14:00"
        assert (await runner.validate_before_execute())[1] is None
        registry, _ = await build_tool_registry()
        assert "final_answer" in registry.tools["complete"].params
        call = selection.calls[0]
        result = await registry.execute(call.name, call.parameters, runner.ctx)
        assert result.success
        assert runner.ctx.shared_state.final_answer == "14:00"
        assert runner.ctx.shared_state.answer == "Verified from device"
        await runner.ctx.shared_state.complete(False, reason="Later task failed")
        assert runner.ctx.shared_state.final_answer is None

    asyncio.run(run())


def test_agent_can_finish_without_caller_assertions_after_visual_review():
    assessments = (
        {
            "id": 1,
            "requirement": "Open WLAN",
            "satisfied": True,
            "evidence": "WLAN visible",
        },
    )
    runner, events, policy, visual, provider = runtime(
        policy_result=JevDecision("done", operation="DONE", confidence=1),
        visual_result=AutoGLMDecision(
            "done",
            message="The requested WLAN page is visible.",
            assessments=assessments,
        ),
    )
    selection = asyncio.run(runner.select("Open the WLAN page", runner.ctx.ui))
    assert selection.source == "autoglm"
    assert selection.calls[0].name == "complete"
    assert events.index("jev") < events.index("autoglm")
    assert "EVERY requirement" in visual.requests[0]["feedback"]
    assert asyncio.run(runner.validate_before_execute())[1] is None
    assert (
        next(e for e in runner.trace if e["event"] == "completion")["basis"]
        == "agent_assessment"
    )
    assert (
        next(e for e in runner.trace if e["event"] == "completion_assessment")[
            "requirements"
        ]
        == assessments
    )
    provider.ui = ui_state(packageName="com.example.other")
    assert asyncio.run(runner.validate_before_execute())[1] == "stale_fallback_state"


@pytest.mark.parametrize("checks", [{"text_all": ["Downloads"]}, {}])
def test_hallucinated_completion_evidence_requires_correction(checks):
    runner, _, _, visual, _ = runtime(
        policy_result=JevDecision("done", operation="DONE")
    )
    visual.decide = AsyncMock(
        side_effect=[
            AutoGLMDecision(
                "done", message="Downloads is visible", completion_checks=checks
            ),
            AutoGLMDecision(
                "action", calls=(AutoGLMToolCall("click_at", {"x": 100, "y": 200}),)
            ),
        ]
    )
    selection = asyncio.run(
        runner.select("Keep the Downloads menu visible", runner.ctx.ui)
    )
    assert selection.calls[0].name == "click_at"
    assert (
        next(e for e in runner.trace if e["event"] == "evidence_check")["success"]
        is False
    )
    assert not any(e["event"] == "completion" for e in runner.trace)


def test_generated_completion_checks_are_corroborated_without_user_flags():
    runner, _, _, _, _ = runtime(
        policy_result=JevDecision("done", operation="DONE"),
        visual_result=AutoGLMDecision(
            "done", message="WLAN is visible", completion_checks={"text_all": ["WLAN"]}
        ),
    )
    selection = asyncio.run(runner.select("Open WLAN", runner.ctx.ui))
    assert selection.calls[0].name == "complete"
    assert (
        next(e for e in runner.trace if e["event"] == "evidence_check")["success"]
        is True
    )


def test_text_grounding_allows_only_a_long_press():
    runner, _, _, visual, _ = runtime(
        policy_result=JevDecision(
            "fallback",
            reason="text_long_press_requires_visual_grounding",
            operation="LONG_PRESS",
        ),
        visual_result=AutoGLMDecision(
            "action", calls=(AutoGLMToolCall("long_press_at", {"x": 100, "y": 200}),)
        ),
    )
    runner.registry.tools["long_press_at"] = object()
    selected = asyncio.run(runner.select("Long press the entered text", runner.ctx.ui))
    assert selected.calls[0].name == "long_press_at"
    assert visual.requests[0]["available_tools"] == {"long_press_at"}
    assert "visible text glyphs" in visual.requests[0]["feedback"]


def test_uncertain_done_requests_review_instead_of_another_device_action():
    runner, _, _, visual, _ = runtime(
        policy_result=JevDecision(
            "fallback",
            reason="low_operation_confidence",
            operation="DONE",
            confidence=0.3,
        ),
        visual_result=AutoGLMDecision(
            "done", message="WLAN visible", completion_checks={"text_all": ["WLAN"]}
        ),
    )
    selected = asyncio.run(runner.select("Open WLAN", runner.ctx.ui))
    assert selected.calls[0].name == "complete"
    assert visual.requests[0]["completion_review"] is True


def test_invalid_review_check_schema_retries_review_without_device_input():
    runner, _, _, visual, _ = runtime(
        policy_result=JevDecision("done", operation="DONE")
    )
    visual.decide = AsyncMock(
        side_effect=[
            AutoGLMDecision(
                "done", message="Menu visible", completion_checks={"menu_visible": True}
            ),
            AutoGLMDecision(
                "done", message="WLAN visible", completion_checks={"text_all": ["WLAN"]}
            ),
        ]
    )
    selected = asyncio.run(runner.select("Open WLAN", runner.ctx.ui))
    assert selected.calls[0].name == "complete"
    assert all(
        call.kwargs["completion_review"] for call in visual.decide.await_args_list
    )
    assert any(
        e["event"] == "format_retry" and not e["action_executed"] for e in runner.trace
    )


@pytest.mark.parametrize("corrected", [True, False])
def test_action_only_state_evidence_gets_requirement_feedback_and_one_fresh_retry(
    corrected,
):
    from mobilerun.agent.decision.contracts import ContractError, parse_grounded_review
    from mobilerun.agent.decision.task_state import TaskState

    # One requirement has observation evidence; the other has only a dispatched
    # input. The real checker must reject the unsupported individual property.
    state = TaskState()
    state.observe(ui_state())
    state.record_action(
        ToolCall("type", {"text": "private value"}), ActionResult(True, "typed")
    )
    review = {
        "complete": True,
        "evidence": "One observed property and one attempted input",
        "checks": {},
        "final_answer": None,
        "verification_refs": ["O1:n0"],
        "assessments": [
            {
                "id": 1,
                "status": "supported",
                "evidence": "Page observed",
                "evidence_refs": ["O1:n0"],
            },
            {
                "id": 2,
                "status": "supported",
                "evidence": "Input dispatched",
                "evidence_refs": ["A1"],
            },
        ],
    }
    contract = {
        "requirements": [
            {"id": 1, "kind": "state", "description": "Observe the page"},
            {"id": 2, "kind": "state", "description": "Save a field value"},
        ],
        "answer_required": False,
        "checks": {},
    }
    with pytest.raises(ContractError) as rejected:
        parse_grounded_review(json.dumps(review), contract, state.evidence(), "O1")
    assert rejected.value.code == "effect_unverified"
    assert rejected.value.field == "evidence_refs"
    error = AutoGLMDecision(
        "error",
        message="Completion reviewer returned invalid evidence.",
        error_code=rejected.value.code,
        error_field=rejected.value.field,
    )
    runner, events, _, visual, provider = runtime(
        policy_result=JevDecision("done", operation="DONE")
    )
    before = provider.ui
    after = ui_state(currentApp="Settings.Reobserved")

    async def decide(**kwargs):
        visual.requests.append(kwargs)
        if len(visual.requests) == 1:
            provider.ui = after
            return error
        return (
            AutoGLMDecision(
                "done",
                message="The WLAN page was observed",
                completion_checks={"text_all": ["WLAN"]},
            )
            if corrected
            else error
        )

    visual.decide = decide
    selected = asyncio.run(runner.select("Open WLAN", before))
    assert len(visual.requests) == 2
    assert all(r["completion_review"] for r in visual.requests)
    assert visual.requests[1]["ui_state"] is after
    feedback = visual.requests[1]["review_feedback"]
    assert "Check every supported non-process requirement separately" in feedback
    assert "A# action references alone cannot prove" in feedback
    assert "complete=false" in feedback
    assert "private value" not in feedback
    assert runner.history == []
    assert not any(e["event"] == "action" for e in runner.trace)
    assert sum(e["event"] == "format_retry" for e in runner.trace) == 1
    assert selected.stop is (not corrected)


@pytest.mark.parametrize("corrected", [True, False])
def test_contradictory_package_retries_review_once_without_device_actions(corrected):
    runner, _, _, visual, _ = runtime(
        policy_result=JevDecision("done", operation="DONE")
    )
    error = AutoGLMDecision(
        "error",
        message="Completion reviewer contradicted the observed foreground package.",
    )
    visual.decide = AsyncMock(
        side_effect=[
            error,
            (
                AutoGLMDecision(
                    "done",
                    message="WLAN visible",
                    completion_checks={"text_all": ["WLAN"]},
                )
                if corrected
                else error
            ),
        ]
    )
    selected = asyncio.run(runner.select("Open WLAN", runner.ctx.ui))
    requests = visual.decide.await_args_list
    assert len(requests) == 2 and all(r.kwargs["completion_review"] for r in requests)
    assert "com.android.settings" in requests[1].kwargs["review_feedback"]
    assert runner.history == []
    if corrected:
        assert [call.name for call in selected.calls] == ["complete"]
    else:
        assert selected.stop and not selected.calls


def test_completion_checks_use_actual_input_and_exclude_hidden_text():
    ui = ui_state(
        [
            {"index": 1, "text": "Downloads", "visible": False},
            {"index": 2, "text": "Hint", "inputValue": "example.com", "editable": True},
        ]
    )
    assert not completion_matches(ui, {"text_all": ["Downloads"]})
    assert not completion_matches(ui, {"input_text": "Hint"})
    assert completion_matches(ui, {"input_text": "example.com"})


def test_application_guidance_reaches_both_models_without_manager_inference():
    runner, _, policy, visual, _ = runtime(policy_result=JevDecision("fallback"))
    runner.app_card_provider = SimpleNamespace(
        load_app_card=AsyncMock(return_value="Use the overflow menu.")
    )
    asyncio.run(runner.select("Find the menu", runner.ctx.ui))
    assert "Use the overflow menu." in policy.requests[0]["guidance"]
    assert "Use the overflow menu." in visual.requests[0]["guidance"]
    assert "ONE action" in visual.requests[0]["guidance"]
    runner.app_card_provider.load_app_card.assert_awaited_with("com.android.settings")


def test_final_review_can_continue_after_a_premature_visual_completion_claim():
    runner, _, _, visual, _ = runtime(policy_result=JevDecision("fallback"))
    visual.decide = AsyncMock(
        side_effect=[
            AutoGLMDecision("done", message="The menu was open earlier."),
            AutoGLMDecision(
                "action", calls=(AutoGLMToolCall("click_at", {"x": 100, "y": 200}),)
            ),
        ]
    )
    selected = asyncio.run(
        runner.select("Keep the navigation menu open", runner.ctx.ui)
    )
    assert selected.calls[0].name == "click_at"
    assert visual.decide.await_count == 2
    assert "Final review" in visual.decide.await_args.kwargs["feedback"]
    assert not any(row["event"] == "completion" for row in runner.trace)


@pytest.mark.parametrize("with_assertions", [False, True])
def test_rejected_semantic_review_requests_correction_without_completing(
    with_assertions,
):
    runner, _, _, visual, _ = runtime(
        settings=config(
            completion={"package": "com.android.settings"} if with_assertions else {}
        ),
        policy_result=JevDecision("done", operation="DONE"),
    )
    visual.decide = AsyncMock(
        side_effect=[
            AutoGLMDecision(
                "incomplete", message="App info is not visible", model="review-model"
            ),
            AutoGLMDecision(
                "action", calls=(AutoGLMToolCall("click_at", {"x": 100, "y": 200}),)
            ),
        ]
    )
    selected = asyncio.run(runner.select("Open the app info menu", runner.ctx.ui))
    assert selected.calls[0].name == "click_at"
    calls = visual.decide.await_args_list
    assert calls[0].kwargs["completion_review"] is True
    assert calls[1].kwargs["completion_review"] is False
    assert calls[0].kwargs["recovery"] is False
    assert calls[1].kwargs["recovery"] is True
    assert "App info is not visible" in calls[1].kwargs["feedback"]
    assert not any(row["event"] == "completion" for row in runner.trace)


def test_repeated_completion_claims_after_rejection_stop_without_actions():
    runner, _, _, visual, _ = runtime(
        policy_result=JevDecision("done", operation="DONE")
    )
    visual.decide = AsyncMock(
        side_effect=[
            AutoGLMDecision("incomplete", message="Still a folder"),
            AutoGLMDecision("done", message="Complete"),
            AutoGLMDecision("incomplete", message="Still a folder"),
        ]
    )
    selected = asyncio.run(runner.select("Open the app info menu", runner.ctx.ui))
    assert selected.stop and selected.calls == []
    assert selected.reason.startswith("completion_unverified:")
    assert visual.decide.await_count == 3
    assert runner.history == []


def test_matching_package_assertion_cannot_accept_a_page_changed_during_review():
    runner, _, _, visual, provider = runtime(
        settings=config(completion={"package": "com.android.settings"}),
        policy_result=JevDecision("done", operation="DONE"),
    )

    async def review(**kwargs):
        provider.ui = ui_state(
            [{"index": 1, "text": "Different page", "bounds": "0,0,400,80"}]
        )
        return AutoGLMDecision(
            "done", message="The previously observed WLAN page was visible."
        )

    visual.decide = review
    selection = asyncio.run(runner.select("Show the WLAN page", runner.ctx.ui))
    assert selection.calls[0].name == "complete"
    assert asyncio.run(runner.validate_before_execute())[1] == "stale_fallback_state"


def test_progress_records_executed_actions_and_next_observation():
    runner, _, _, _, provider = runtime(
        policy_result=JevDecision("fallback"),
        visual_result=AutoGLMDecision(
            "action",
            calls=(AutoGLMToolCall("click_at", {"x": 100, "y": 100}),),
            progress="Inspect the network settings, then return home.",
        ),
    )
    selection = asyncio.run(runner.select("Inspect networks", runner.ctx.ui))
    assert runner.progress_note == ""
    runner.record_execution(
        selection.calls[0],
        ActionResult(success=True, summary="Tap sent"),
        runner.ctx.ui,
    )
    assert runner.progress_note.startswith("Inspect")
    assert "after" not in runner.history[0]
    provider.ui = ui_state(
        [{"index": 1, "text": "Network details", "bounds": "0,0,400,80"}]
    )
    runner.observe_result(provider.ui)
    assert runner.task_context()["execution_log"][0]["after"]["visible_text"] == [
        "Network details"
    ]
    assert runner.task_context()["executed_actions"] == 1


def test_cross_page_repeated_action_is_bounded_even_when_recent_page_changed():
    runner, _, _, _, _ = runtime()
    call = ToolCall("click_at", {"x": 100, "y": 100})
    runner.action_signatures = [runner.signature(call, runner.ctx.ui)] * 2
    selection = runner.accept_fallback([call], runner.ctx.ui)
    assert selection.stop
    assert "Repeated action" in selection.reason


def test_generic_fallback_can_finish_autonomously_with_current_evidence():
    runner, _, _, _, _ = runtime(settings=config(fallback="llm"))
    runner.pending = runner.accept_fallback(
        [
            ToolCall(
                "complete",
                {"success": True, "message": "The requested screen is visible"},
            )
        ],
        runner.ctx.ui,
    )
    assert asyncio.run(runner.validate_before_execute())[1] is None


def test_generic_fallback_completion_is_rechecked_against_fresh_state():
    runner, _, _, _, provider = runtime(
        settings=config(fallback="llm", completion={"text_all": ["已连接"]})
    )
    ui = runner.ctx.ui
    runner.pending = runner.accept_fallback(
        [ToolCall("complete", {"success": True})], ui
    )
    provider.ui = ui_state()
    _, reason = asyncio.run(runner.validate_before_execute())
    assert reason == "completion_unverified"
    guard = runner.trace[-1]
    assert guard["event"] == "completion_guard"
    assert guard["failed_check"] == "explicit_postconditions"
    assert guard["explicit_checks"] == {"text_all": ["已连接"]}
    assert guard["observed_image_sha256"] is None


@pytest.mark.parametrize("changed", ["unchanged", "package", "native", "image"])
def test_completion_guard_retains_actual_binding_and_fresh_rejection_inputs(changed):
    ui = ui_state(
        [
            {
                "index": 1,
                "className": "TextView",
                "resourceId": "result",
                "rawText": "Saved",
            },
            {
                "index": 2,
                "className": "EditText",
                "password": True,
                "inputValue": "private-value",
            },
        ]
    )
    runner, _, _, _, provider = runtime(ui=ui)
    runner.task_state.observe(ui, screenshot=b"reviewed-image")
    runner.pending = runner.accept_fallback(
        [ToolCall("complete", {"success": True, "message": "The result is saved"})],
        ui,
    )
    runner.bind_completion_evidence(
        AutoGLMDecision(
            "done",
            evidence_refs=("O1:n0", "O1:image1"),
            verification_refs=("O1:n0", "O1:image1"),
        )
    )
    binding = runner.trace[-1]
    assert binding["event"] == "completion_guard_binding"
    assert binding["reviewed_observation_id"] == "O1"
    assert (
        binding["image_proofs"][0]["sha256"]
        == hashlib.sha256(b"reviewed-image").hexdigest()
    )
    assert binding["native_proofs"][0]["rawText"] == "Saved"

    provider.ui = copy.deepcopy(ui)
    if changed == "package":
        provider.ui.phone_state["packageName"] = "another.app"
    elif changed == "native":
        provider.ui.elements[0]["rawText"] = "Unsaved"
    image = b"fresh-image" if changed == "image" else b"reviewed-image"
    runner.ctx.driver.screenshot = AsyncMock(return_value=image)
    _, reason = asyncio.run(runner.validate_before_execute())
    guard = runner.trace[-1]
    assert reason == (None if changed == "unchanged" else "completion_unverified")
    assert guard["event"] == "completion_guard"
    assert guard["success"] is (changed == "unchanged")
    assert (
        guard["failed_check"]
        == {
            "unchanged": None,
            "package": "foreground_package",
            "native": "native_evidence",
            "image": "current_image",
        }[changed]
    )
    assert guard["expected_image_sha256"] == binding["image_proofs"][0]["sha256"]
    assert guard["observed_image_sha256"] == (
        None if changed in {"package", "native"} else hashlib.sha256(image).hexdigest()
    )
    assert runner.ctx.driver.screenshot.await_count == (
        0 if changed in {"package", "native"} else 1
    )
    assert guard["observed_package"] == provider.ui.phone_state["packageName"]
    assert guard["observed_native_nodes"][0]["rawText"] == (
        "Unsaved" if changed == "native" else "Saved"
    )
    assert guard["reviewed_native_proofs"][0]["rawText"] == "Saved"
    assert "private-value" not in json.dumps(guard)
    assert runner.history == [] and runner.ctx.driver.started == []

    # Later mutable UI/evidence cannot rewrite the exact failed-call capture.
    runner.pending.completion_proofs[0]["rawText"] = "later proof"
    provider.ui.elements[0]["rawText"] = "later screen"
    assert binding["native_proofs"][0]["rawText"] == "Saved"
    assert guard["reviewed_native_proofs"][0]["rawText"] == "Saved"
    assert guard["observed_native_nodes"][0]["rawText"] != "later screen"


@pytest.mark.parametrize("changed", ["readout", "stopped", "package", "assertion"])
def test_reviewed_completion_rechecks_outcome_without_freezing_live_readouts(changed):
    ui = ui_state(
        [
            {"index": 1, "text": "00:01", "bounds": "0,0,400,80"},
            {"index": 2, "text": "Pause", "bounds": "0,80,400,160"},
        ]
    )
    runner, _, _, _, provider = runtime(
        ui=ui,
        settings=config(
            completion={"text_all": ["00:01"]} if changed == "assertion" else {}
        ),
        policy_result=JevDecision("done"),
        visual_result=AutoGLMDecision(
            "done",
            message="Pause is visible",
            completion_checks={"text_all": ["Pause"]},
        ),
    )

    async def run():
        await runner.select("Start the activity", ui)
        fresh = copy.deepcopy(ui)
        fresh.elements[0]["text"] = "00:02"
        if changed == "stopped":
            fresh.elements[1]["text"] = "Start"
        elif changed == "package":
            fresh.phone_state["packageName"] = "another.app"
        provider.ui = fresh
        return await runner.validate_before_execute()

    _, reason = asyncio.run(run())
    assert reason == (None if changed == "readout" else "completion_unverified")
    assert runner.history == []
    guard = runner.trace[-1]
    assert guard["event"] == "completion_guard"
    assert (
        guard["failed_check"]
        == {
            "readout": None,
            "stopped": "reviewed_postconditions",
            "package": "foreground_package",
            "assertion": "explicit_postconditions",
        }[changed]
    )


def test_verifier_truthy_strings_do_not_count_as_verified():
    runner, _, _, _, _ = runtime(verifier=lambda _: "true")
    assert asyncio.run(runner.verified(runner.ctx.ui)) is False


async def real_jev_action(ui, operation="TAP", texts=()):
    def handler(request):
        payload = json.loads(request.content)
        answers = {}
        for key, q in payload["questions"].items():
            selected = operation if key == "operation" else next(iter(q["criteria"]))
            answers[key] = {
                "type": "choice",
                "choice": selected,
                "confidence": 1.0,
                "probabilities": {
                    value: float(value == selected) for value in q["criteria"]
                },
            }
        return httpx.Response(200, json={"answers": answers})

    async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
        policy = JevDecisionPolicy(api_key="offline", client=client)
        return await policy.decide(
            goal="输入北京南站" if texts else "打开WLAN", ui_state=ui, texts=texts
        )


def test_stale_jev_target_is_rejected_before_any_execution():
    ui = ui_state()
    decision = asyncio.run(real_jev_action(ui))
    runner, _, _, _, provider = runtime(ui=ui, policy_result=decision)

    async def run():
        await runner.select("打开WLAN", ui)
        changed = copy.deepcopy(ui)
        changed.elements[1]["text"] = "恢复出厂设置"
        provider.ui = changed
        return await runner.validate_before_execute()

    _, reason = asyncio.run(run())
    assert reason == "target_changed"
    assert runner.history == []
    assert runner.ctx.driver.started == []
    assert runner.trace[-1]["event"] == "stale"


def test_repeated_jev_action_routes_to_fallback_and_retains_failed_execution():
    runner, events, _, visual, _ = runtime()

    async def run():
        selected = await runner.select("打开WLAN", runner.ctx.ui)
        runner.record_execution(
            selected.calls[0], ActionResult(False, "uncertain execution"), runner.ctx.ui
        )
        return await runner.select("打开WLAN", runner.ctx.ui)

    selection = asyncio.run(run())
    assert selection.source == "autoglm"
    assert runner.history[0]["success"] is False
    assert "same action" in visual.requests[0]["feedback"]
    assert visual.requests[0]["recovery"] is True
    assert events == ["inventory", "jev", "jev", "screenshot", "autoglm"]


def test_fallback_cannot_dispatch_multiple_or_unknown_tools():
    runner, _, _, _, _ = runtime()
    for calls in (
        [ToolCall("click", {"index": 2}), ToolCall("wait")],
        [ToolCall("shell")],
        [],
    ):
        selected = runner.accept_fallback(calls, runner.ctx.ui)
        assert selected.stop
        assert selected.calls == []


def test_chinese_input_is_read_back_completely():
    ui = ui_state(
        [
            {
                "index": 7,
                "text": "搜索目的地",
                "inputValue": "",
                "className": "EditText",
                "resourceId": "com.android.settings:id/search",
                "editable": True,
                "focused": True,
                "bounds": "0,80,400,140",
            }
        ],
        isEditable=True,
    )
    runner, _, _, _, provider = runtime(ui=ui)
    call = ToolCall("type", {"text": "北京南站", "clear": True})
    runner.pending = Selection("jev", [call], snapshot=ui)

    async def run():
        _, reason = await runner.validate_before_execute()
        assert reason is None
        assert runner.input_target is not None
        after = copy.deepcopy(ui)
        after.elements[0].update(text="北京南站", inputValue="北京南站")
        provider.ui = after
        return await runner.verify_input(call)

    assert asyncio.run(run()) is True
    assert runner.trace[-1] == {"event": "input_verification", "success": True}


@pytest.mark.parametrize("clear", [True, False])
@pytest.mark.parametrize("restoration", [True, False, "error"])
def test_input_restores_temporary_keyboard_before_any_later_action(clear, restoration):
    ui = ui_state(
        [
            {
                "index": 7,
                "className": "EditText",
                "resourceId": "search",
                "inputValue": "",
                "editable": True,
                "focused": True,
                "bounds": "0,80,400,140",
            }
        ],
        isEditable=True,
    )
    runner, _, _, _, provider = runtime(ui=ui)
    call = ToolCall("type", {"text": "蓝牙", "clear": clear})
    runner.pending = Selection("jev", [call], snapshot=ui)

    async def restore():
        if clear:
            assert runner.trace[-1] == {"event": "input_verification", "success": True}
        if restoration == "error":
            raise RuntimeError("private restoration details")
        return restoration

    runner.ctx.driver.restore_input_method = AsyncMock(side_effect=restore)

    async def run():
        _, reason = await runner.validate_before_execute()
        assert reason is None
        provider.ui = copy.deepcopy(ui)
        provider.ui.elements[0]["inputValue"] = "蓝牙"
        return await runner.verify_input(call)

    assert asyncio.run(run()) is (restoration is True)
    runner.ctx.driver.restore_input_method.assert_awaited_once_with()
    assert runner.trace[-1] == {
        "event": "input_method_restoration",
        "success": restoration is True,
    }
    assert "private restoration details" not in str(runner.trace)


def test_non_input_actions_do_not_restore_the_keyboard():
    runner, _, _, _, _ = runtime()
    runner.ctx.driver.restore_input_method = AsyncMock()
    assert asyncio.run(
        runner.verify_input(ToolCall("long_press_at", {"x": 30, "y": 20}))
    )
    runner.ctx.driver.restore_input_method.assert_not_awaited()


def test_wrong_field_cannot_satisfy_input_readback(monkeypatch):
    before = ui_state(
        [
            {
                "index": 7,
                "text": "旧值",
                "inputValue": "旧值",
                "className": "EditText",
                "resourceId": "com.android.settings:id/origin",
                "editable": True,
                "focused": True,
                "bounds": "0,80,400,140",
            },
            {
                "index": 8,
                "text": "北京南站",
                "inputValue": "北京南站",
                "className": "EditText",
                "resourceId": "com.android.settings:id/destination",
                "editable": True,
                "focused": False,
                "bounds": "0,180,400,240",
            },
        ],
        isEditable=True,
    )
    decision = asyncio.run(
        real_jev_action(before, operation="TYPE_TEXT", texts=["北京南站"])
    )
    runner, _, _, _, _ = runtime(ui=before, policy_result=decision)
    runner.pending = Selection(
        "jev", [decision.tool_call], guard=decision, snapshot=before
    )
    _, reason = asyncio.run(runner.validate_before_execute())
    assert reason is None
    assert runner.input_target.resource_id == "com.android.settings:id/origin"
    ticks = iter([0.0, 4.0, 8.0])
    monkeypatch.setattr(
        "mobilerun.agent.decision.runtime.time",
        SimpleNamespace(monotonic=lambda: next(ticks)),
    )
    assert asyncio.run(runner.verify_input(decision.tool_call)) is False


def test_resolve_input_limits_fallback_to_text_when_fresh_field_is_focused():
    ui = ui_state(
        [
            {
                "index": 2,
                "className": "EditText",
                "focused": True,
                "bounds": "0,80,400,160",
                "inputValue": "",
            }
        ]
    )
    runner, _, _, visual, _ = runtime(
        ui=ui,
        policy_result=JevDecision("fallback", reason="text_value_required"),
        visual_result=AutoGLMDecision(
            "action",
            calls=(AutoGLMToolCall("type", {"text": "example.com", "clear": True}),),
        ),
    )
    selected = asyncio.run(runner.select("Enter example.com", ui))
    assert selected.calls[0].name == "type"
    assert visual.requests[0]["available_tools"] == {"type"}


def test_input_readback_follows_bound_resource_when_ui_indices_change():
    before = ui_state(
        [
            {
                "index": 7,
                "text": "搜索目的地",
                "inputValue": "",
                "className": "EditText",
                "resourceId": "com.android.settings:id/destination",
                "editable": True,
                "focused": True,
                "bounds": "0,80,400,140",
            }
        ],
        isEditable=True,
    )
    decision = asyncio.run(
        real_jev_action(before, operation="TYPE_TEXT", texts=["北京南站"])
    )
    runner, _, _, _, provider = runtime(ui=before, policy_result=decision)
    runner.pending = Selection(
        "jev", [decision.tool_call], guard=decision, snapshot=before
    )

    async def run():
        _, reason = await runner.validate_before_execute()
        assert reason is None
        target = runner.input_target
        assert target.resource_id == "com.android.settings:id/destination"
        # The former index is now a different field; the destination has moved and
        # been renumbered after typing changed the accessibility tree.
        provider.ui = ui_state(
            [
                {
                    "index": 7,
                    "text": "错误地点",
                    "inputValue": "错误地点",
                    "className": "EditText",
                    "resourceId": "com.android.settings:id/origin",
                    "editable": True,
                    "focused": False,
                    "bounds": "0,80,400,140",
                },
                {
                    "index": 23,
                    "text": "北京南站",
                    "inputValue": "北京南站",
                    "className": "EditText",
                    "resourceId": "com.android.settings:id/destination",
                    "editable": True,
                    "focused": True,
                    "bounds": "0,180,400,240",
                },
            ],
            isEditable=True,
        )
        matched = await runner.verify_input(decision.tool_call)
        assert runner.input_target is target  # Readback cannot bind a different input.
        return matched

    assert asyncio.run(run()) is True


def test_input_readback_without_prebinding_cannot_claim_success():
    ui = ui_state(
        [
            {
                "index": 7,
                "text": "北京南站",
                "inputValue": "北京南站",
                "className": "EditText",
                "resourceId": "com.android.settings:id/destination",
                "editable": True,
                "focused": True,
                "bounds": "0,80,400,140",
            },
        ],
        isEditable=True,
    )
    runner, _, _, _, _ = runtime(ui=ui)
    call = ToolCall("type", {"text": "北京南站", "clear": True})
    runner.pending = Selection("jev", [call], snapshot=ui)
    assert runner.input_target is None
    assert asyncio.run(runner.verify_input(call)) is False


@pytest.mark.parametrize(
    "inventory_key", ["package_name", "package", "packageName", "app_id"]
)
def test_launch_only_allows_current_installed_inventory(inventory_key):
    driver = Driver([], apps=[{inventory_key: "com.example.maps", "label": "地图"}])
    ctx = SimpleNamespace(driver=driver)

    async def run():
        bad = await launch_app("com.evil.not_installed", ctx=ctx)
        assert not bad.success
        assert driver.started == []
        good = await launch_app("com.example.maps", ctx=ctx)
        assert good.success
        assert driver.started == ["com.example.maps"]

    asyncio.run(run())


def test_registry_registration_uses_the_actual_registry_contract():
    registry = ToolRegistry()
    registry.register("open_app", lambda: None, {}, "Would call another LLM")
    register_decision_tools(registry)
    assert "open_app" not in registry.tools
    assert registry.tools["launch_app"].deps == {"start_app", "get_apps"}


class PromptResolver:
    def get_prompt(self, name):
        return {
            "fast_agent_system": "Use the registered tools: {{ tool_descriptions }}",
            "fast_agent_user": "Goal: {{ goal }}",
        }.get(name)


def run_workflow(mode="jev", stale=False):
    async def run():
        events = []
        shared = MobileAgentState(telemetry_config_enabled=False)
        initial = ui_state()
        completed = ui_state([{"index": 10, "text": "已连接", "bounds": "0,0,300,80"}])

        class WorkflowProvider(Provider):
            def __init__(self):
                super().__init__(initial)
                self.reads = 0

            async def get_state(self):
                self.reads += 1
                events.append("observe")
                if stale and self.reads % 2 == 0:
                    changed = copy.deepcopy(self.ui)
                    changed.elements[1]["text"] = "恢复出厂设置"
                    return changed
                return self.ui

        provider = WorkflowProvider()

        class WorkflowPolicy:
            async def decide(self, **kwargs):
                events.append("jev")
                if kwargs["ui_state"] is completed:
                    return JevDecision("done", operation="DONE", confidence=1)
                if mode in {
                    "fallback",
                    "llm",
                    "llm_done",
                    "format_recovery",
                    "format_repeat",
                    "network_error",
                }:
                    return JevDecision("fallback", reason="request_timeout")
                if mode == "done":
                    return JevDecision("done", operation="DONE", confidence=1)
                return await real_jev_action(kwargs["ui_state"])

        driver = Driver(events)
        context = SimpleNamespace(
            driver=driver,
            ui=initial,
            shared_state=shared,
            credential_manager=None,
            state_provider=provider,
        )
        registry = ToolRegistry()

        async def click(index, *, ctx):
            events.append("execute:click")
            assert index == 2
            if mode == "action_failed":
                return ActionResult(False, "input may have been applied")
            provider.ui = completed
            return ActionResult(True, "Clicked WLAN")

        async def click_at(x, y, *, ctx):
            events.append("execute:click_at")
            assert (x, y) == (100, 200)
            provider.ui = completed
            return ActionResult(True, "Tapped WLAN")

        async def complete(success, message="", *, ctx):
            events.append("execute:complete")
            await shared.complete(success, message=message)
            return ActionResult(True, message)

        registry.register(
            "click", click, {"index": {"type": "number"}}, "Click an element"
        )
        registry.register(
            "click_at",
            click_at,
            {"x": {"type": "number"}, "y": {"type": "number"}},
            "Tap a point",
        )
        registry.register(
            "complete",
            complete,
            {"success": {"type": "boolean"}, "message": {"type": "string"}},
            "Complete",
        )

        class WorkflowVisual(Visual):
            async def decide(self, **kwargs):
                if kwargs.get("completion_review") and kwargs["ui_state"] is completed:
                    self.events.append("autoglm")
                    self.requests.append(kwargs)
                    return AutoGLMDecision(
                        "done", message="The requested connected state is visible."
                    )
                return await super().decide(**kwargs)

        visual = WorkflowVisual(
            events, AutoGLMDecision("done", message="完成") if mode == "done" else None
        )
        if mode in {"format_recovery", "format_repeat", "network_error"}:
            errors = {
                "format_recovery": "AutoGLM did not return a complete action.",
                "format_repeat": "AutoGLM returned an invalid or unavailable action.",
                "network_error": "AutoGLM network request failed; no action was dispatched.",
            }

            class RetryingVisual(WorkflowVisual):
                async def decide(self, **kwargs):
                    if kwargs.get("completion_review"):
                        return await super().decide(**kwargs)
                    events.append("autoglm")
                    self.requests.append(kwargs)
                    if mode != "format_recovery" or len(self.requests) == 1:
                        return AutoGLMDecision("error", message=errors[mode])
                    assert not any(e.startswith("execute:") for e in events)
                    return self.result

            visual = RetryingVisual(events)
        settings = SimpleNamespace(
            jev=config(
                enabled=True,
                completion={"text_all": ["已连接"]},
                fallback="llm" if mode.startswith("llm") else "autoglm",
            ),
            fast_agent=SimpleNamespace(
                vision=not mode.startswith("llm"), parallel_tools=False
            ),
            max_steps=5,
            streaming=False,
            after_sleep_action=0,
        )
        agent = FastAgent(
            llm=SimpleNamespace(),
            agent_config=settings,
            registry=registry,
            action_ctx=context,
            state_provider=provider,
            shared_state=shared,
            prompt_resolver=PromptResolver(),
            decision_policy=WorkflowPolicy(),
            visual_fallback=visual,
        )

        async def generic_response(*args, **kwargs):
            events.append("llm")
            assert "jev" in events
            invocation = (
                '<invoke name="complete"><parameter name="success">true</parameter></invoke>'
                if mode == "llm_done"
                else '<invoke name="click"><parameter name="index">2</parameter></invoke>'
            )
            return ChatResponse(
                message=ChatMessage(
                    role="assistant",
                    content=(
                        "Perform the next step.\n<function_calls>"
                        + invocation
                        + "</function_calls>"
                    ),
                )
            )

        generic = AsyncMock(
            side_effect=(
                generic_response
                if mode.startswith("llm")
                else AssertionError("Generic LLM must not run before Jev")
            )
        )
        with patch("mobilerun.agent.fast_agent.fast_agent.acall_with_retries", generic):
            result = await agent.run(input="打开WLAN并连接")
        return result, events, generic.await_count, agent

    return asyncio.run(run())


def test_real_fast_agent_workflow_jev_executes_before_final_review_without_generic_llm():
    result, events, generic_calls, _ = run_workflow()
    assert result["success"] is True
    assert generic_calls == 0
    assert events.count("jev") == 2
    assert events.count("autoglm") == 1
    assert (
        events.index("execute:click")
        < events.index("screenshot")
        < events.index("autoglm")
    )
    assert events.index("jev") < events.index("execute:click")
    assert [e for e in events if e.startswith("execute:")] == [
        "execute:click",
        "execute:complete",
    ]


def test_real_fast_agent_workflow_jev_error_then_visual_fallback_without_generic_llm():
    result, events, generic_calls, _ = run_workflow(mode="fallback")
    assert result["success"] is True
    assert generic_calls == 0
    assert events.index("jev") < events.index("screenshot") < events.index("autoglm")
    assert [e for e in events if e.startswith("execute:")] == [
        "execute:click_at",
        "execute:complete",
    ]


def test_real_fast_agent_workflow_unverified_done_stops_without_input():
    result, events, generic_calls, _ = run_workflow(mode="done")
    assert result["success"] is False
    assert result["reason"] == "completion_unverified"
    assert generic_calls == 0
    assert not any(e.startswith("execute:") for e in events)


def test_real_fast_agent_workflow_rejects_stale_targets_and_stops_after_three():
    result, events, generic_calls, agent = run_workflow(stale=True)
    assert result["success"] is False
    assert result["reason"] == "target_changed"
    assert generic_calls == 0
    assert events.count("jev") == 3
    assert agent.decision_runtime.stale_count == 3
    assert not any(e.startswith("execute:") for e in events)


def test_real_fast_agent_workflow_configured_llm_runs_only_after_jev_failure():
    result, events, generic_calls, _ = run_workflow(mode="llm")
    assert result["success"] is True
    assert generic_calls == 1
    assert events.index("jev") < events.index("llm") < events.index("execute:click")
    assert "autoglm" not in events


def test_real_fast_agent_workflow_generic_llm_cannot_self_certify_completion():
    result, events, generic_calls, _ = run_workflow(mode="llm_done")
    assert result["success"] is False
    assert result["reason"] == "completion_unverified"
    assert generic_calls == 1
    assert not any(e.startswith("execute:") for e in events)


def test_real_fast_agent_workflow_uncertain_execution_is_not_replayed():
    result, events, generic_calls, _ = run_workflow(mode="action_failed")
    assert result["success"] is False
    assert "stopped without replay" in result["reason"]
    assert events.count("jev") == 1
    assert events.count("execute:click") == 1
    assert generic_calls == 0
    assert "autoglm" not in events


class FrameDriver(Driver):
    """A screenshot provider with raw Portal phone state, but no actionable UI tree."""

    def __init__(self, events):
        super().__init__(events)
        self.frame = b"search-screen-cursor-visible"
        self.phone = {
            "packageName": "com.android.settings",
            "currentApp": "Settings.Search",
            "isEditable": True,
            "keyboardVisible": True,
        }
        self.tree = {"text": "搜索设置", "resourceId": "title", "children": []}

    async def screenshot(self):
        self.events.append("screenshot")
        return self.frame

    async def get_ui_tree(self):
        self.events.append("raw_phone")
        return {"phone_state": dict(self.phone), "a11y_tree": copy.deepcopy(self.tree)}


def visual_navigation_runtime(tool_name="system_button", arguments=None):
    action = AutoGLMDecision(
        "action",
        calls=(
            AutoGLMToolCall(
                tool_name, arguments if arguments is not None else {"button": "back"}
            ),
        ),
    )
    runner, events, policy, visual, provider = runtime(
        ui=ui_state([]),
        policy_result=JevDecision("fallback", reason="no_candidates"),
        visual_result=action,
    )
    runner.ctx.driver = FrameDriver(events)
    return runner, events, policy, visual, provider


def test_visual_back_allows_cursor_pixels_to_change_with_same_raw_navigation_state():
    runner, events, _, _, _ = visual_navigation_runtime()

    async def run():
        selection = await runner.select("退出搜索", runner.ctx.ui)
        assert all(
            selection.navigation_context[key] == value
            for key, value in runner.ctx.driver.phone.items()
        )
        assert selection.visual_fingerprint
        runner.ctx.driver.frame = b"search-screen-cursor-invisible"
        return await runner.validate_before_execute()

    _, reason = asyncio.run(run())
    assert reason is None
    assert runner.pending.calls == [ToolCall("system_button", {"button": "back"})]
    assert events.count("screenshot") == 1  # The navigation guard uses raw phone state.
    assert events.count("raw_phone") == 2
    assert runner.history == []


@pytest.mark.parametrize(
    "changed",
    [
        {"packageName": "com.example.other"},
        {"currentApp": "Settings.Home"},
        {"isEditable": False},
        {"keyboardVisible": False},
    ],
)
def test_visual_back_rejects_changed_raw_app_activity_or_edit_mode(changed):
    runner, _, _, _, _ = visual_navigation_runtime()

    async def run():
        await runner.select("退出搜索", runner.ctx.ui)
        runner.ctx.driver.phone.update(changed)
        return await runner.validate_before_execute()

    _, reason = asyncio.run(run())
    assert reason == "stale_navigation_context"
    assert runner.history == []
    assert runner.ctx.driver.started == []


def test_visual_back_rejects_a_different_page_inside_the_same_app():
    runner, _, _, _, _ = visual_navigation_runtime()

    async def run():
        await runner.select("退出搜索", runner.ctx.ui)
        runner.ctx.driver.tree["text"] = "恢复出厂设置"
        return await runner.validate_before_execute()

    _, reason = asyncio.run(run())
    assert reason == "stale_navigation_context"
    assert runner.history == []


def test_visual_coordinate_action_still_requires_the_same_pixels():
    runner, events, _, _, _ = visual_navigation_runtime(
        "click_at", {"x": 100, "y": 200}
    )

    async def run():
        await runner.select("点击搜索", runner.ctx.ui)
        runner.ctx.driver.frame = b"search-screen-cursor-invisible"
        return await runner.validate_before_execute()

    _, reason = asyncio.run(run())
    assert reason == "stale_visual_frame"
    assert events.count("screenshot") == 2
    assert runner.history == []


def test_stale_feedback_reaches_next_visual_call_without_inventing_execution_history():
    runner, _, _, visual, _ = visual_navigation_runtime(
        "click_at", {"x": 100, "y": 200}
    )

    async def run():
        await runner.select("点击搜索", runner.ctx.ui)
        runner.ctx.driver.frame = b"changed-before-tap"
        _, reason = await runner.validate_before_execute()
        assert reason == "stale_visual_frame"
        return await runner.select("点击搜索", runner.ctx.ui)

    selected = asyncio.run(run())
    assert not selected.stop
    assert len(visual.requests) == 2
    assert "NOT executed" in visual.requests[1]["feedback"]
    assert "observation changed" in visual.requests[1]["feedback"]
    assert visual.requests[1]["recent_actions"] == []
    assert not any(row["event"] == "action" for row in runner.trace)


def test_format_retry_observes_new_state_and_image_before_asking_again():
    runner, events, _, _, provider = visual_navigation_runtime(
        "click_at", {"x": 100, "y": 200}
    )
    before = provider.ui
    after = ui_state([], currentApp="Settings.SearchUpdated")

    class ReobservingVisual(Visual):
        async def decide(self, **kwargs):
            events.append("autoglm")
            self.requests.append(kwargs)
            if len(self.requests) == 1:
                provider.ui = after
                runner.ctx.driver.frame = b"new-image-after-invalid-response"
                return AutoGLMDecision(
                    "error",
                    message="AutoGLM returned an invalid or unavailable action.",
                )
            return self.result

    visual = ReobservingVisual(events)
    runner.visual_fallback = visual
    selected = asyncio.run(runner.select("点击搜索", before))
    assert not selected.stop
    assert selected.snapshot is after
    assert visual.requests[0]["ui_state"] is before
    assert visual.requests[1]["ui_state"] is after
    assert visual.requests[1]["screenshot"] == b"new-image-after-invalid-response"
    assert visual.requests[0]["recovery"] is False
    assert visual.requests[1]["recovery"] is True
    assert "JSON action object" in visual.requests[1]["feedback"]
    assert "rejected before any device input" in visual.requests[1]["feedback"]
    assert len(visual.requests) == 2
    assert runner.history == []
    assert [
        row["action_executed"] for row in runner.trace if row["event"] == "format_retry"
    ] == [False]


@pytest.mark.parametrize("corrected", [True, False])
def test_rejected_type_target_feedback_uses_fresh_state_without_repairing_or_extra_retries(
    corrected,
):
    runner, events, _, _, provider = visual_navigation_runtime()
    before = provider.ui
    after = ui_state([], currentApp="Settings.OtherEditor")
    rejected = parse_autoglm_decision(
        '{"action":"Type","text":"private payload","index":7}',
        ui_state=SimpleNamespace(),
        screenshot_size=(0, 0),
        available_tools={"type"},
        structured=True,
    )

    class CorrectingVisual(Visual):
        async def decide(self, **kwargs):
            self.requests.append(kwargs)
            events.append("autoglm")
            if len(self.requests) == 1:
                provider.ui = after
                runner.ctx.driver.frame = b"fresh-target-choice"
                return rejected
            # Only the model's new proposal can replace its rejected Type.
            return self.result if corrected else rejected

    visual = CorrectingVisual(events)
    runner.visual_fallback = visual
    selection = asyncio.run(runner.select("Edit a chosen field", before))
    assert len(visual.requests) == 2
    assert visual.requests[1]["ui_state"] is after
    assert visual.requests[1]["screenshot"] == b"fresh-target-choice"
    feedback = visual.requests[1]["feedback"]
    assert '"rejected_known_fields": ["index"]' in feedback
    assert '"allowed_arguments": ["text"]' in feedback
    assert "Do not delete a target or constraint" in feedback
    assert "private payload" not in feedback
    assert runner.history == []
    assert not any(e["event"] == "action" for e in runner.trace)
    assert sum(e["event"] == "format_retry" for e in runner.trace) == 1
    if corrected:
        assert not selection.stop
        assert selection.calls[0].name == "click_at"
    else:
        assert selection.stop and not selection.calls


def test_untrusted_argument_diagnostics_are_filtered_before_trace_and_retry_feedback():
    runner, _, _, visual, _ = runtime(
        policy_result=JevDecision("fallback", reason="no_candidates"),
    )
    error = AutoGLMDecision(
        "error",
        message="AutoGLM returned an invalid or unavailable action.",
        error_code="unknown_argument",
        error_field="arguments",
        error_argument_fields=(
            "index",
            "secret diagnostic content",
            {"text": "secret"},
        ),
        error_allowed_arguments=("text", "secret allowed content"),
    )
    visual.decide = AsyncMock(side_effect=[error, error])
    selected = asyncio.run(runner.select("Edit field", runner.ctx.ui))
    assert selected.stop and not selected.calls
    assert visual.decide.await_count == 2
    feedback = visual.decide.await_args_list[1].kwargs["feedback"]
    assert '"rejected_known_fields": ["index"]' in feedback
    assert "secret" not in feedback and "secret" not in json.dumps(runner.trace)


def test_real_workflow_format_retry_executes_only_one_successful_proposal():
    result, events, generic_calls, agent = run_workflow(mode="format_recovery")
    assert result["success"] is True
    assert events.count("jev") == 2
    assert events.count("autoglm") == 3
    assert generic_calls == 0
    assert events.count("execute:click_at") == 1
    model_calls = [i for i, event in enumerate(events) if event == "autoglm"]
    assert model_calls[1] < events.index("execute:click_at") < model_calls[2]
    assert [row["event"] for row in agent.decision_runtime.trace].count(
        "format_retry"
    ) == 1


def test_real_workflow_repeated_format_failure_stops_after_one_retry_without_input():
    result, events, generic_calls, _ = run_workflow(mode="format_repeat")
    assert result["success"] is False
    assert events.count("autoglm") == 2
    assert events.count("jev") == 1
    assert generic_calls == 0
    assert not any(event.startswith("execute:") for event in events)


@pytest.mark.parametrize(
    "message",
    [
        "AutoGLM network request failed; no action was dispatched.",
        "AutoGLM request timed out; no action was dispatched.",
        "AutoGLM request failed (HTTP 429).",
        "AutoGLM request failed (HTTP 503).",
        "AutoGLM returned an invalid API response.",
    ],
)
def test_visual_transport_and_api_errors_do_not_use_format_retry(message):
    runner, events, _, visual, _ = runtime(
        policy_result=JevDecision("fallback", reason="request_timeout"),
        visual_result=AutoGLMDecision("error", message=message),
    )
    selected = asyncio.run(runner.select("打开WLAN", runner.ctx.ui))
    assert selected.stop
    assert selected.reason == message
    assert len(visual.requests) == 1
    assert events.count("autoglm") == 1
    assert not any(row["event"] in {"format_retry", "action"} for row in runner.trace)


def test_real_workflow_network_error_stops_without_model_or_input_retry():
    result, events, generic_calls, _ = run_workflow(mode="network_error")
    assert result["success"] is False
    assert events.count("autoglm") == 1
    assert events.count("jev") == 1
    assert generic_calls == 0
    assert not any(event.startswith("execute:") for event in events)


def test_runtime_does_not_replay_unexecuted_visual_model_history():
    events = []
    ctx = SimpleNamespace(driver=Driver(events), ui=ui_state())
    runner = DecisionRuntime(
        config(),
        ctx,
        SimpleNamespace(tools={}),
        Provider(ctx.ui),
        policy=Policy(events),
    )
    assert runner.visual_fallback.max_history_turns == 0
