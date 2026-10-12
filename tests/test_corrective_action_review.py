"""An unmet grounded requirement must lead to an action or a stated blocker."""

import asyncio
import json
from unittest.mock import AsyncMock

import httpx
import pytest
from test_decision_visual_guards import _Policy, _runtime, _ui
from test_task_evidence import task

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.decision.autoglm import (
    AutoGLMDecision,
    AutoGLMFallback,
    AutoGLMToolCall,
)
from mobilerun.agent.decision.jev import JevDecision
from mobilerun.agent.decision.task_state import TaskState
from mobilerun.agent.fast_agent.xml_parser import ToolCall


@pytest.mark.parametrize(
    "outcome", ["correct", "repeat_finish", "blocked", "invalid_target", "stale_target"]
)
def test_actual_request_cannot_repeat_completion_instead_of_correcting(
    monkeypatch, outcome
):
    monkeypatch.setenv("GLM_API_KEY", "offline-secret")
    goal = "Keep the saved record visible"
    contract = task(goal, "final_state")
    requests, holders = [], []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        runner, provider = holders[0]
        if len(requests) == 1 or len(requests) == 4:
            context = json.loads(payload["messages"][-1]["content"][0]["text"])
            ref = context["task_state"]["ledger"]["current_observation"] + ":n0"
            complete = len(requests) == 4
            body = {
                "complete": complete,
                "evidence": (
                    "Saved status observed" if complete else "Record is unsaved"
                ),
                "checks": {},
                "final_answer": None,
                "verification_refs": [ref] if complete else [],
                "assessments": [
                    {
                        "id": 1,
                        "status": "supported" if complete else "unknown",
                        "evidence": (
                            "Saved status observed"
                            if complete
                            else "Save effect is unconfirmed"
                        ),
                        "evidence_refs": [ref] if complete else [],
                    }
                ],
            }
        else:
            assert "Finish和finish不可用" in payload["messages"][0]["content"]
            assert payload["model"] == "glm-5.3-flash"
            assert payload["thinking"] == {"type": "disabled"}
            assert any(
                part["type"] == "image_url"
                for part in payload["messages"][-1]["content"]
            )
            if len(requests) == 2 or outcome == "repeat_finish":
                body = {"action": "Finish", "message": "Everything is already done"}
            elif outcome == "blocked":
                body = {
                    "action": "Take_over",
                    "message": "Save control cannot be operated",
                }
            elif outcome == "invalid_target":
                body = {"action": "Tap", "index": 999}
            else:
                body = {"action": "Tap", "index": 1}
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(body)}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = AutoGLMFallback(
                model="glm-5.3-flash",
                reasoning_model="glm-5.3-flash",
                base_url="https://example.invalid/v1",
                api_key_env="GLM_API_KEY",
                thinking_mode="disabled",
                client=client,
            )
            adapter._goal, adapter._completion_contract = goal, contract
            runner, _, provider = _runtime(
                _ui(
                    [
                        {
                            "index": 1,
                            "resourceId": "save",
                            "text": "Unsaved",
                            "className": "Button",
                            "clickable": True,
                            "bounds": "0,0,120,100",
                        },
                    ]
                ),
                visual=adapter,
            )
            runner.config.reasoning_model = "glm-5.3-flash"
            runner.goal, runner.task_state = goal, TaskState(goal)
            runner.task_state.set_contract(contract)
            holders.append((runner, provider))
            selected = await runner._visual(provider.ui, completion_review=True)
            assert len(requests) == 3 and runner.review_rejections == 1
            assert runner.history == []
            assert not any(item["event"] == "completion" for item in runner.trace)
            errors = [
                item
                for item in runner.trace
                if item["event"] == "decision" and item["status"] == "error"
            ]
            assert errors[0]["error_code"] == "completion_disallowed"
            assert sum(item["event"] == "format_retry" for item in runner.trace) == 1
            corrections = [
                item
                for item in runner.trace
                if item["event"] == "fallback_context" and item.get("corrective_only")
            ]
            assert len(corrections) == 2 and all(
                not item["completion_review"] for item in corrections
            )
            assert not any(
                "Everything is already done" in str(item) for item in adapter._history
            )
            if outcome in {"repeat_finish", "blocked", "invalid_target"}:
                assert selected.stop and selected.calls == []
                if outcome == "blocked":
                    assert selected.reason == "Save control cannot be operated"
                return
            assert not selected.stop and selected.calls[0].name == "click"
            if outcome == "stale_target":
                provider.ui = _ui(
                    [
                        {
                            "index": 1,
                            "resourceId": "delete",
                            "text": "Delete",
                            "className": "Button",
                            "clickable": True,
                            "bounds": "0,0,120,100",
                        }
                    ]
                )
                assert (await runner.validate_before_execute())[
                    1
                ] == "stale_fallback_state"
                assert runner.history == []
                return
            assert (await runner.validate_before_execute())[1] is None
            runner.record_execution(
                selected.calls[0],
                ActionResult(success=True, summary="Save clicked"),
                provider.ui,
            )
            provider.ui = _ui(
                [
                    {
                        "index": 1,
                        "resourceId": "save",
                        "text": "Saved",
                        "className": "Button",
                        "clickable": True,
                        "bounds": "0,0,120,100",
                    }
                ]
            )
            runner.observe_result(provider.ui)
            runner.policy = _Policy(
                [
                    JevDecision(
                        "action",
                        tool_call=ToolCall("click", {"index": 1}),
                        operation="TAP",
                    )
                ]
            )
            next_action = await runner.select(goal, provider.ui)
            assert next_action.source == "jev" and len(requests) == 3
            complete = await runner._visual(provider.ui, completion_review=True)
            assert complete.calls[0].name == "complete" and not complete.stop
            # The existing runtime resets rejection accounting after an actual
            # successful tool execution; this change must preserve that rule.
            assert len(requests) == 4 and runner.review_rejections == 0
            assert (await runner.validate_before_execute())[1] is None
            assert len(runner.history) == 1

    asyncio.run(run())


@pytest.mark.parametrize("returned", ["done", "complete_tool"])
def test_runtime_rejects_completion_even_if_adapter_ignores_corrective_mode(returned):
    async def run():
        runner, _, provider = _runtime(_ui([{"index": 1, "text": "Unsaved"}]))
        goal = "Keep the saved record visible"
        runner.goal, runner.task_state = goal, TaskState(goal)
        runner.task_state.set_contract(task(goal, "final_state"))
        runner.config.reasoning_model = "glm-5.3-flash"
        visual = runner.visual_fallback
        visual.supports_corrective_actions = True
        illegal = (
            AutoGLMDecision("done", message="Done")
            if returned == "done"
            else AutoGLMDecision(
                "action",
                calls=(
                    AutoGLMToolCall("complete", {"success": True, "message": "Done"}),
                ),
            )
        )
        visual.decide = AsyncMock(
            side_effect=[
                AutoGLMDecision("incomplete", message="Save effect is unknown"),
                illegal,
                AutoGLMDecision(
                    "action", calls=(AutoGLMToolCall("click", {"index": 1}),)
                ),
            ]
        )
        selected = await runner._visual(provider.ui, completion_review=True)
        assert selected.calls[0].name == "click" and not selected.stop
        assert visual.decide.await_count == 3 and runner.review_rejections == 1
        assert all(
            call.kwargs.get("corrective_only")
            for call in visual.decide.await_args_list[1:]
        )
        assert runner.history == [] and not any(
            item["event"] == "completion" for item in runner.trace
        )

    asyncio.run(run())


@pytest.mark.parametrize(
    "mode", ["review", "no_recovery", "native", "ungrounded", "no_reasoner"]
)
def test_invalid_corrective_mode_never_calls_api(monkeypatch, mode):
    monkeypatch.setenv("GLM_API_KEY", "offline-secret")
    goal = "Keep the saved record visible"

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: pytest.fail("Invalid corrective request called the API"),
            )
        ) as client:
            adapter = AutoGLMFallback(
                model="glm-5.3-flash",
                reasoning_model=None if mode == "no_reasoner" else "glm-5.3-flash",
                api_key_env="GLM_API_KEY",
                client=client,
            )
            adapter._goal, adapter._completion_contract = goal, (
                None if mode == "ungrounded" else task(goal, "final_state")
            )
            runner, driver, provider = _runtime(_ui([{"index": 1, "text": "Unsaved"}]))
            result = await adapter.decide(
                goal=goal,
                screenshot=await driver.screenshot(),
                ui_state=provider.ui,
                available_tools=runner.registry.tools,
                corrective_only=True,
                recovery=mode != "no_recovery",
                completion_review=mode == "review",
                native_only_review=mode == "native",
            )
            assert result.status == "error" and not result.calls

    asyncio.run(run())


@pytest.mark.parametrize("disabled", ["semantic_recovery", "capability", "budget"])
def test_disabled_recovery_and_existing_rejection_limit_are_preserved(
    monkeypatch, disabled
):
    monkeypatch.setenv("GLM_API_KEY", "offline-secret")
    goal = "Keep the saved record visible"
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert "Finish和finish不可用" not in payload["messages"][0]["content"]
        if len(requests) == 2:
            body = {"action": "Finish", "message": "Already done"}
        else:
            body = {
                "complete": False,
                "evidence": "Save effect unknown",
                "checks": {},
                "final_answer": None,
                "assessments": [
                    {
                        "id": 1,
                        "status": "unknown",
                        "evidence": "Unsaved",
                        "evidence_refs": [],
                    }
                ],
                "verification_refs": [],
            }
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(body)}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = AutoGLMFallback(
                model="glm-5.3-flash",
                reasoning_model="glm-5.3-flash",
                api_key_env="GLM_API_KEY",
                client=client,
            )
            adapter._goal, adapter._completion_contract = goal, task(
                goal, "final_state"
            )
            runner, _, provider = _runtime(
                _ui([{"index": 1, "text": "Unsaved"}]), visual=adapter
            )
            runner.config.reasoning_model = "glm-5.3-flash"
            runner.goal, runner.task_state = goal, TaskState(goal)
            runner.task_state.set_contract(adapter.completion_contract)
            if disabled == "semantic_recovery":
                runner.config.semantic_recovery = False
            elif disabled == "capability":
                adapter.supports_corrective_actions = False
            else:
                runner.review_rejections = 1
            selected = await runner._visual(provider.ui, completion_review=True)
            assert selected.stop and selected.calls == [] and runner.history == []
            assert runner.review_rejections == 2
            assert len(requests) == (1 if disabled == "budget" else 3)
            assert not any(item.get("corrective_only") for item in runner.trace)

    asyncio.run(run())
