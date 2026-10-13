"""Real HTTP payloads cannot use hidden images to certify a native-only review."""

import asyncio
import copy
import json

import httpx
import pytest
from test_decision_visual_guards import _image, _runtime, _ui
from test_task_evidence import task

from mobilerun.agent.decision.task_state import TaskState
from mobilerun.agent.decision.visual_fallback import VisualFallback


@pytest.mark.parametrize(
    "reply",
    [
        "native",
        "hidden_image",
        "missing_property",
        "unknown_visual",
        "changed_native",
        "whole_observation",
    ],
)
def test_guard_refresh_uses_actual_native_evidence_and_rejects_hidden_images(
    monkeypatch, reply
):
    monkeypatch.setenv("GLM_API_KEY", "offline-secret")
    goal = "Keep the saved record visible"
    contract = task(goal, "final_state")
    requests, holders = [], []

    def respond(request):
        payload = json.loads(request.content)
        content = payload["messages"][-1]["content"]
        context = json.loads(content[0]["text"])
        requests.append(context)
        assert payload["model"] == "glm-5.3-flash"
        assert payload["thinking"] == {"type": "disabled"}
        runner, driver, provider = holders[0]
        current = runner.task_state.current_id
        native_ref = current + ":n0"
        refs = [native_ref]
        complete, status = True, "supported"
        if len(requests) == 1:
            assert any(part["type"] == "image_url" for part in content)
            refs.append(context["screenshot_evidence"]["ref"])
        else:
            assert context["evidence_mode"] == "native"
            assert context["screenshot_evidence"] is None
            assert context["uploaded_screenshot_evidence"] == []
            assert all(part["type"] == "text" for part in content)
            assert not any(
                ":image" in ref for ref in context["available_evidence_refs"]
            )
            assert all(
                "images" not in page
                for page in context["task_state"]["ledger"]["observations"]
            )
            assert "没有上传任何当前或历史图片" in payload["messages"][0]["content"]
            if reply == "hidden_image":
                refs.append(runner.task_state.observations[-1]["images"][-1]["ref"])
            elif reply == "missing_property":
                refs = []
            elif reply == "whole_observation":
                refs = [current]
            elif reply == "unknown_visual":
                refs, complete, status = [], False, "unknown"
            elif reply == "changed_native":
                provider.ui = _ui(
                    [{"index": 1, "resourceId": "status", "text": "Unsaved"}]
                )
            driver.image = _image("blue")
        body = {
            "complete": complete,
            "evidence": "Saved status or insufficient visual evidence",
            "checks": {},
            "final_answer": None,
            "verification_refs": refs,
            "assessments": [
                {
                    "id": 1,
                    "status": status,
                    "evidence": "Observed native property",
                    "evidence_refs": refs,
                }
            ],
        }
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(body)}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = VisualFallback(
                model="glm-5.3-flash",
                reasoning_model="glm-5.3-flash",
                base_url="https://example.invalid/v1",
                api_key_env="GLM_API_KEY",
                thinking_mode="disabled",
                client=client,
            )
            adapter._goal, adapter._completion_contract = goal, contract
            runner, driver, provider = _runtime(
                _ui(
                    [
                        {"index": 1, "resourceId": "status", "text": "Saved"},
                    ]
                ),
                visual=adapter,
            )
            holders.append((runner, driver, provider))
            runner.goal = goal
            runner.task_state = TaskState(goal)
            runner.task_state.set_contract(contract)
            await runner._visual(provider.ui, completion_review=True)
            driver.image = _image("green")
            _, reason = await runner.validate_before_execute()
            assert reason == "completion_unverified"
            assert runner.recover_completion_validation(reason)
            assert runner.completion_refresh_native
            selected = await runner.select(goal, provider.ui)
            # The local ledger still retains actual images; the projected request
            # and its parser cannot use them as unuploaded proof.
            assert runner.task_state.observations[0]["images"]
            assert runner.history == []
            if reply in {"native", "changed_native"}:
                assert not selected.stop and selected.completion_image_guard is None
                _, reason = await runner.validate_before_execute()
                assert reason == (
                    None if reply == "native" else "completion_unverified"
                )
                if reason:
                    assert runner.last_completion_failure == "native_evidence"
                    assert not runner.recover_completion_validation(reason)
                assert len(requests) == 2
            else:
                assert selected.stop and selected.calls == []
                assert len(requests) == (2 if reply == "unknown_visual" else 3)
                if reply == "hidden_image":
                    errors = [
                        row
                        for row in runner.trace
                        if row["event"] == "decision" and row.get("status") == "error"
                    ]
                    assert errors[-1]["error_code"] == "evidence_unknown"

    asyncio.run(run())


def test_native_review_projection_does_not_mutate_caller_ledger(monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "offline-secret")
    goal = "Leave the record open"
    state = TaskState(goal)
    state.set_contract(task(goal, "final_state"))
    screen = _ui([{"text": "Saved"}])
    image = _image("red")
    state.observe(screen, screenshot=image)
    context = {"ledger": state.context()}
    original = copy.deepcopy(context)

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(503, json={"error": "offline failure"}),
            )
        ) as client:
            adapter = VisualFallback(
                reasoning_model="glm-5.3-flash",
                api_key_env="GLM_API_KEY",
                base_url="https://example.invalid/v1",
                client=client,
            )
            adapter._goal, adapter._completion_contract = goal, state.contract
            result = await adapter.decide(
                goal=goal,
                screenshot=image,
                ui_state=screen,
                available_tools={"complete"},
                task_state=context,
                completion_review=True,
                native_only_review=True,
            )
            assert result.status == "error"
            assert context == original

    asyncio.run(run())


@pytest.mark.parametrize("reason", ["not_review", "no_contract", "empty_controls"])
def test_native_mode_cannot_silently_change_an_action_request_or_ungrounded_review(
    monkeypatch, reason
):
    monkeypatch.setenv("GLM_API_KEY", "offline-secret")

    def forbidden(_):
        pytest.fail("Invalid native-only mode must not call the API")

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(forbidden)
        ) as client:
            goal = "Leave the record open"
            adapter = VisualFallback(
                reasoning_model="glm-5.3-flash",
                api_key_env="GLM_API_KEY",
                client=client,
            )
            adapter._goal = goal
            adapter._completion_contract = (
                None if reason == "no_contract" else task(goal, "final_state")
            )
            result = await adapter.decide(
                goal=goal,
                screenshot=_image("red"),
                ui_state=_ui([] if reason == "empty_controls" else [{"text": "Saved"}]),
                available_tools={"complete"},
                native_only_review=True,
                completion_review=reason != "not_review",
            )
            assert result.status == "error"

    asyncio.run(run())
