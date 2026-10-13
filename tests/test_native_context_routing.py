"""Transport sharing preserves provider routing and strict action parsing."""

import asyncio
import copy
import json

import httpx
import pytest
from test_decision_visual_guards import _image, _ui
from test_task_evidence import task

from mobilerun.agent.decision.native_context import ALIASES_FIELD, compact_native_values
from mobilerun.agent.decision.task_state import TaskState
from mobilerun.agent.decision.visual_fallback import VisualFallback, _ledger_evidence


@pytest.mark.parametrize("body_chars", [600, 2400, 9000])
@pytest.mark.parametrize("response_kind", ["valid", "multiple_actions"])
@pytest.mark.parametrize("source", ["observation", "collection"])
def test_same_evidence_keeps_model_protocol_and_validation_after_sharing(
    monkeypatch,
    body_chars,
    response_kind,
    source,
):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-key")
    goal = "Open the next record"
    contract = task(goal, "final_state")
    body = ("Captured 正文.\n" * body_chars)[:body_chars]
    current = _ui(
        [
            {
                "index": 1,
                "text": "Next",
                "className": "Button",
                "bounds": "0,0,200,100",
                "resourceId": "next",
                "clickable": True,
            }
        ]
    )
    state = TaskState()
    state.observe(
        _ui(
            [
                {
                    "index": 1,
                    "text": body,
                    "rawText": body,
                    "inputValue": body,
                    "resourceId": "source",
                }
            ]
        )
    )
    state.observe(current)
    ordinary = state.context(max_chars=100_000)
    shared = state.context(max_chars=100_000, share_native_values=True)
    if source == "collection":
        for ledger in (ordinary, shared):
            node = ledger["observations"][0]["nodes"][0]
            ledger["observations"][0]["nodes"] = []
            ledger["collections"] = [{"rows": [{"nodes": [node]}]}]
    assert _ledger_evidence(ordinary) == _ledger_evidence(shared)
    before = copy.deepcopy((ordinary, shared))
    requests, results = [], []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        content = (
            json.dumps({"action": "Tap", "index": 1})
            if payload["model"] == "reasoner"
            else 'do(action="Tap", element=[250, 250])'
        )
        if response_kind == "multiple_actions":
            content += "\n" + content
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            for ledger in (ordinary, shared):
                adapter = VisualFallback(
                    model="autoglm-phone",
                    reasoning_model="reasoner",
                    base_url="https://example.invalid/v1",
                    api_key_env="GLM_API_KEY",
                    client=client,
                )
                adapter._goal, adapter._completion_contract = goal, contract
                results.append(
                    await adapter.decide(
                        goal=goal,
                        screenshot=_image("red"),
                        ui_state=current,
                        available_tools={"click", "click_at"},
                        task_state={"ledger": ledger},
                    )
                )

    asyncio.run(run())
    assert len(requests) == 2
    assert (ordinary, shared) == before
    assert requests[0]["model"] == requests[1]["model"]
    assert requests[0]["max_tokens"] == requests[1]["max_tokens"]
    assert requests[0]["messages"][0] == requests[1]["messages"][0]
    assert results[0].status == results[1].status
    assert results[0].calls == results[1].calls
    expected_model = "autoglm-phone" if body_chars == 600 else "reasoner"
    assert requests[0]["model"] == expected_model
    if response_kind == "valid":
        assert results[0].status == "action"
        assert len(results[0].calls) == 1
    else:
        assert results[0].status != "action"
        assert results[0].calls == ()
    ordinary_text, shared_text = [
        payload["messages"][-1]["content"][0]["text"] for payload in requests
    ]
    assert ALIASES_FIELD not in ordinary_text
    assert ALIASES_FIELD in shared_text
    assert len(shared_text) < len(ordinary_text)
    if body_chars == 2400:
        assert len(ordinary_text) > 8000
        assert len(shared_text) < 8000


def test_untyped_context_keeps_phone_protocol_even_with_long_shared_values(monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-key")
    body = "Captured text " * 900
    node = {"ref": "O1:n0", "text": body, "rawText": body, "inputValue": body}
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": 'do(action="Tap", element=[250, 250])',
                        }
                    }
                ]
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = VisualFallback(
                reasoning_model="reasoner",
                base_url="https://example.invalid/v1",
                api_key_env="GLM_API_KEY",
                client=client,
            )
            result = await adapter.decide(
                goal="Open the next record",
                screenshot=_image("red"),
                ui_state=_ui(),
                available_tools={"click_at"},
                task_state={
                    "ledger": {
                        "observations": [
                            {
                                "id": "O1",
                                "nodes": [compact_native_values(node)],
                            }
                        ]
                    }
                },
            )
            assert result.status == "action"

    asyncio.run(run())
    assert len(requests) == 1
    assert requests[0]["model"] == "autoglm-phone"
