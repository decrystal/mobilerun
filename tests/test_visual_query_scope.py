"""Scoped visual answers retain uploaded-image and final-frame requirements."""

import asyncio
import json

import httpx
import pytest
from test_decision_visual_guards import _image, _runtime
from test_query_semantic_scope import scoped_contract
from test_task_evidence import ui

from mobilerun.agent.decision.contracts import ContractError, parse_grounded_review
from mobilerun.agent.decision.task_state import TaskState
from mobilerun.agent.decision.visual_fallback import VisualDecision, VisualFallback
from mobilerun.config_manager.config_manager import JevConfig

GOAL = "What balance is currently displayed for the Travel account?"


def contract(reference="current"):
    result = scoped_contract(reference)
    item = result["requirements"][0]
    item.update(
        description=GOAL, quote=GOAL, attribute={"kind": "value", "label": "balance"}
    )
    item["query_scope"] = {
        "reference": reference,
        "quote": GOAL,
        "constraints": ["Travel account"],
    }
    return result


def fixture(reference="current"):
    state = TaskState(GOAL)
    state.set_contract(contract(reference))
    state.observe(ui(), screenshot=_image("red"))
    return state


def response(ref="O1:image1", *, missing_membership=False):
    return {
        "complete": True,
        "checks": {},
        "evidence": "Read the account and balance from the uploaded frame",
        "final_answer": "247.60 EUR",
        "assessments": [
            {
                "id": 1,
                "status": "supported",
                "evidence": "The selected account and displayed amount are visible",
                "evidence_refs": [] if missing_membership else [ref],
                "scope_refs": [ref],
                "constraint_evidence": [
                    {"quote": "Travel account", "evidence_refs": [ref]}
                ],
            }
        ],
    }


def parsed(state, data, uploaded=None):
    return parse_grounded_review(
        json.dumps(data),
        state.contract,
        state.evidence(),
        state.current_id,
        uploaded_images=uploaded,
        derive_review_dependencies=True,
    )


def test_direct_current_visual_answer_can_use_actual_uploaded_scope_and_value():
    state = fixture()
    result = parsed(state, response())
    assert result["status"] == "done" and result["verification_refs"] == ("O1:image1",)
    assert state.collections.scopes == {} and state.query_results == []


@pytest.mark.parametrize("uploaded", [True, False])
def test_current_question_cannot_borrow_an_old_visual_scope_even_if_uploaded(uploaded):
    state = fixture()
    old = state.evidence()["O1:image1"]["sha256"]
    state.observe(
        ui([{"index": 1, "rawText": "Another page"}]), screenshot=_image("blue")
    )
    with pytest.raises(ContractError, match="query_scope_unverified"):
        parsed(state, response(), uploaded={"O1:image1": old} if uploaded else None)


def test_explicit_observed_visual_answer_can_cite_a_reuploaded_historical_frame():
    state = fixture("observed")
    old = state.evidence()["O1:image1"]["sha256"]
    state.observe(
        ui([{"index": 1, "rawText": "Unrelated details"}]), screenshot=_image("blue")
    )
    result = parsed(state, response(), uploaded={"O1:image1": old})
    assert result["status"] == "done" and result["verification_refs"] == ()


@pytest.mark.parametrize("wrong", [False, True])
def test_historical_visual_range_still_requires_the_exact_uploaded_image_hash(wrong):
    state = fixture("observed")
    state.observe(
        ui([{"index": 1, "rawText": "Another page"}]), screenshot=_image("blue")
    )
    with pytest.raises(ContractError, match="evidence_stale"):
        parsed(state, response(), uploaded={"O1:image1": "0" * 64} if wrong else None)


def test_scope_fields_cannot_remove_their_image_from_own_assessment_membership():
    state = fixture()
    with pytest.raises(ContractError, match="query_scope_unverified"):
        parsed(state, response(missing_membership=True))


def test_scope_image_from_an_older_frame_of_the_same_observation_is_stale():
    state = fixture()
    state.observe(ui(), screenshot=_image("blue"))
    with pytest.raises(ContractError, match="evidence_stale"):
        parsed(state, response())
    assert parsed(state, response("O1:image2"))["status"] == "done"


@pytest.mark.parametrize("changed", [False, True])
def test_runtime_uses_exact_visual_frame_guard_for_scoped_direct_answers(
    monkeypatch, changed
):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-secret")
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(response())}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = VisualFallback(
                model="glm-5.3-flash",
                reasoning_model="glm-5.3-flash",
                query_scopes=True,
                api_key_env="GLM_API_KEY",
                base_url="https://example.invalid/v1",
                client=client,
            )
            state = fixture()
            adapter._goal, adapter._completion_contract = GOAL, state.contract
            runner, driver, _ = _runtime(ui(), visual=adapter)
            runner.config = JevConfig(fallback="autoglm")
            runner.goal, runner.task_state = GOAL, state
            selected = await runner._visual(runner.ctx.ui, completion_review=True)
            assert (
                not selected.stop
                and selected.calls[0].parameters["final_answer"] == "247.60 EUR"
            )
            assert (
                selected.completion_image_guard
                and not selected.completion_query_binding_required
            )
            if changed:
                driver.image = _image("blue")
            _, reason = await runner.validate_before_execute()
            assert (reason is not None) is changed
            assert runner.history == [] and not runner.executed

    asyncio.run(run())
    assert len(requests) == 1
    assert (
        sum(
            part["type"] == "image_url"
            for part in requests[0]["messages"][-1]["content"]
        )
        == 1
    )


def test_scope_retry_keeps_a_direct_visual_path_when_no_native_query_exists():
    class RejectedVisual:
        supports_derived_review_dependencies = True

        def __init__(self):
            self.requests = []

        async def decide(self, **kwargs):
            self.requests.append(kwargs)
            return VisualDecision(
                "error",
                message="Completion reviewer returned invalid evidence.", format_error=True,
                error_code="query_scope_unverified",
                error_field="query_scope",
            )

    async def run():
        adapter = RejectedVisual()
        runner, _, _ = _runtime(ui(), visual=adapter)
        runner.config = JevConfig(fallback="autoglm")
        runner.goal, runner.task_state = GOAL, fixture()
        selected = await runner._visual(runner.ctx.ui, completion_review=True)
        assert len(adapter.requests) == 2 and selected.stop and selected.calls == []
        feedback = adapter.requests[1]["review_feedback"]
        assert "without a Q#" in feedback and "uploaded O#:image#" in feedback
        assert "Do not invent a native" in feedback
        assert runner.history == [] and not runner.executed

    asyncio.run(run())


def test_visual_review_prompt_distinguishes_query_sources_from_direct_images(
    monkeypatch,
):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-secret")
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(response())}}]}
        )

    async def run():
        state = fixture()
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = VisualFallback(
                model="glm-5.3-flash",
                reasoning_model="glm-5.3-flash",
                query_scopes=True,
                api_key_env="GLM_API_KEY",
                base_url="https://example.invalid/v1",
                client=client,
            )
            adapter._goal, adapter._completion_contract = GOAL, state.contract
            result = await adapter.decide(
                goal=GOAL,
                ui_state=ui(),
                screenshot=_image("red"),
                available_tools={"complete"},
                task_state={"ledger": state.context()},
                completion_review=True,
                derive_review_dependencies=True,
            )
            assert result.status == "done"

    asyncio.run(run())
    guidance = requests[0]["messages"][0]["content"]
    assert "没有使用Q#计算、直接读取截图时" in guidance
    assert "实际上传的O#:image#" in guidance
    assert "不要编造原生控件" in guidance
