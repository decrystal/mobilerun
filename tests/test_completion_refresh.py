"""Fresh completion reviews keep live evidence and existing recovery budgets."""

import asyncio
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest
from test_jev_first_runtime import config, runtime, ui_state
from test_task_evidence import task

from mobilerun.agent.decision.contracts import parse_grounded_review
from mobilerun.agent.decision.jev import JevDecision
from mobilerun.agent.decision.visual_fallback import VisualDecision, VisualToolCall
from mobilerun.agent.fast_agent.events import FastAgentOutputEvent
from mobilerun.agent.fast_agent.fast_agent import FastAgent
from mobilerun.agent.fast_agent.xml_parser import ToolCall


def setup_review(*, recovery=True):
    screen = ui_state(
        [
            {
                "index": 1,
                "resourceId": "record_status",
                "className": "TextView",
                "rawText": "Saved",
                "text": "Saved",
                "bounds": "0,0,400,80",
            },
        ]
    )
    runner, events, policy, visual, provider = runtime(
        ui=screen,
        settings=config(semantic_recovery=recovery),
        policy_result=JevDecision("done", operation="DONE"),
    )
    runner.goal = "Save this record"
    runner.task_state.reset(runner.goal)
    runner.task_state.set_contract(task(runner.goal, "state"))
    driver = SimpleNamespace(frame=b"frame-a", get_apps=AsyncMock(return_value=[]))

    async def screenshot():
        events.append("screenshot")
        return driver.frame

    driver.screenshot = screenshot
    runner.ctx.driver = driver

    def response(*, image=True):
        observation = runner.task_state.observations[-1]
        refs = [observation["nodes"][0]["ref"]]
        if image:
            refs.append(observation["images"][-1]["ref"])
        data = {
            "complete": True,
            "evidence": "The record is saved",
            "checks": {},
            "final_answer": None,
            "verification_refs": refs,
            "assessments": [
                {
                    "id": 1,
                    "status": "supported",
                    "evidence": "Saved status",
                    "evidence_refs": refs,
                }
            ],
        }
        parsed = parse_grounded_review(
            json.dumps(data),
            runner.task_state.contract,
            runner.task_state.evidence(),
            runner.task_state.current_id,
        )
        return VisualDecision(**{k: v for k, v in parsed.items() if k != "missing"})

    visual.decide = AsyncMock(side_effect=lambda **_: response())
    return runner, events, policy, visual, provider, driver, response


def test_changed_image_requests_fresh_review_without_accepting_old_image():
    async def run():
        runner, events, policy, visual, provider, driver, response = setup_review()
        original = await runner._visual(provider.ui, completion_review=True)
        driver.frame = b"frame-b"
        _, reason = await runner.validate_before_execute()
        assert reason == "completion_unverified"
        assert runner.recover_completion_validation(reason)
        assert runner.pending is None
        visual.decide = AsyncMock(side_effect=lambda **_: response(image=False))
        refreshed = await runner.select(runner.goal, provider.ui)
        assert refreshed is not original and refreshed.calls[0].name == "complete"
        assert refreshed.completion_image_guard is None
        assert refreshed.completion_proofs[0]["rawText"] == "Saved"
        assert (await runner.validate_before_execute())[1] is None
        assert visual.decide.await_args.kwargs["completion_review"] is True
        assert visual.decide.await_args.kwargs["screenshot"] == b"frame-b"
        assert "NOT accepted" in visual.decide.await_args.kwargs["feedback"]
        assert policy.requests == [] and runner.history == []
        assert events.count("screenshot") == 3
        assert runner.review_rejections == 1

    asyncio.run(run())


def test_persistent_visual_dependency_stops_at_existing_rejection_limit():
    async def run():
        runner, _, _, _, provider, driver, _ = setup_review()
        await runner._visual(provider.ui, completion_review=True)
        driver.frame = b"frame-b"
        _, reason = await runner.validate_before_execute()
        assert runner.recover_completion_validation(reason)
        refreshed = await runner.select(runner.goal, provider.ui)
        assert refreshed.completion_image_guard is not None
        driver.frame = b"frame-c"
        _, reason = await runner.validate_before_execute()
        assert reason == "completion_unverified"
        assert not runner.recover_completion_validation(reason)
        assert runner.review_rejections == 2 and runner.history == []

    asyncio.run(run())


def test_changed_native_property_can_request_a_corrective_action_after_fresh_review():
    async def run():
        runner, _, _, visual, provider, _, _ = setup_review()
        await runner._visual(provider.ui, completion_review=True)
        provider.ui.elements[0].update(rawText="Unsaved", text="Unsaved")
        _, reason = await runner.validate_before_execute()
        assert (
            reason == "completion_unverified"
            and runner.last_completion_failure == "native_evidence"
        )
        assert runner.recover_completion_validation(reason)
        visual.decide = AsyncMock(
            return_value=VisualDecision(
                "action",
                calls=(VisualToolCall("click", {"index": 1}),),
            )
        )
        selected = await runner.select(runner.goal, provider.ui)
        assert selected.calls[0].name == "click"
        assert selected.snapshot.elements[0]["rawText"] == "Unsaved"
        assert not selected.stop and runner.history == []
        assert (await runner.validate_before_execute())[1] is None

    asyncio.run(run())


@pytest.mark.parametrize(
    "mode", ["disabled", "already_rejected", "ungrounded", "wrong_reason"]
)
def test_refresh_cannot_bypass_disabled_recovery_or_existing_budgets(mode):
    async def run():
        runner, _, _, _, provider, driver, _ = setup_review(recovery=mode != "disabled")
        await runner._visual(provider.ui, completion_review=True)
        driver.frame = b"frame-b"
        _, reason = await runner.validate_before_execute()
        if mode == "already_rejected":
            runner.review_rejections = 1
        elif mode == "ungrounded":
            runner.pending.completion_proofs = None
        elif mode == "wrong_reason":
            reason = "target_changed"
        assert not runner.recover_completion_validation(reason)
        assert not runner.completion_refresh_pending

    asyncio.run(run())


def test_new_goal_drops_pending_completion_refresh_and_old_evidence():
    async def run():
        runner, _, policy, _, provider, driver, _ = setup_review()
        await runner._visual(provider.ui, completion_review=True)
        driver.frame = b"frame-b"
        _, reason = await runner.validate_before_execute()
        assert runner.recover_completion_validation(reason)
        policy.result = JevDecision(
            "action", tool_call=ToolCall("click", {"index": 1}), operation="TAP"
        )
        await runner.select("Inspect another record", provider.ui)
        assert not runner.completion_refresh_pending and runner.review_rejections == 0
        assert runner.last_completion_failure is None and len(policy.requests) == 1

    asyncio.run(run())


def test_fast_agent_consumes_failed_completion_and_returns_to_ordinary_step_loop():
    async def run():
        runner, _, _, _, provider, driver, _ = setup_review()
        selection = await runner._visual(provider.ui, completion_review=True)
        driver.frame = b"frame-b"
        pending = {"pending_tool_calls": selection.calls}

        async def get(key, default):
            return pending.get(key, default)

        async def put(key, value):
            pending[key] = value

        ctx = SimpleNamespace(
            store=SimpleNamespace(get=get, set=put),
            write_event_to_stream=lambda _: None,
        )
        agent = object.__new__(FastAgent)
        agent.decision_runtime = runner
        agent.action_ctx = runner.ctx
        agent.tool_call_counter = 0
        event = await agent.execute_code(ctx, None)
        assert isinstance(event, FastAgentOutputEvent)
        assert pending["pending_tool_calls"] == []
        assert runner.completion_refresh_pending and agent.tool_call_counter == 0
        assert runner.history == []

    asyncio.run(run())
