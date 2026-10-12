"""A redundant model selector cannot weaken a grounded live completion guard."""

import asyncio
import json

import httpx
import pytest
from mobilerun.agent.decision.autoglm import (
    _GROUNDED_REVIEW_PROMPT,
    AutoGLMFallback,
    _grounded_review_prompt,
)
from mobilerun.agent.decision.contracts import ContractError, parse_grounded_review
from mobilerun.agent.decision.task_state import TaskState
from mobilerun.config_manager.config_manager import JevConfig
from test_decision_visual_guards import _image, _runtime, _ui
from test_task_evidence import review, task, ui


@pytest.mark.parametrize("selector", [[], ["O1:n0"], ["O1:image1"]])
def test_valid_assessments_keep_every_current_dependency_despite_selector(selector):
    state = TaskState()
    state.observe(
        ui([{"text": "Saved", "resourceId": "status"}]), screenshot=b"current"
    )
    data = json.loads(review(["O1:n0", "O1:image1"], answer=None))
    data["verification_refs"] = selector
    parsed = parse_grounded_review(
        json.dumps(data),
        task("Keep the saved record visible", "final_state"),
        state.evidence(),
        "O1",
        derive_review_dependencies=True,
    )
    assert parsed["verification_refs"] == ("O1:n0", "O1:image1")
    assert (
        parsed["assessments"][0]["evidence_refs"]
        == data["assessments"][0]["evidence_refs"]
    )


@pytest.mark.parametrize("selector", ["O1:n0", ["O99:n0"], [3]])
def test_unsolicited_invalid_selector_still_rejected(selector):
    state = TaskState()
    state.observe(ui([{"text": "Saved"}]))
    data = json.loads(review(["O1:n0"], answer=None))
    data["verification_refs"] = selector
    with pytest.raises(ContractError, match="evidence_unknown"):
        parse_grounded_review(
            json.dumps(data),
            task("Keep this record visible", "final_state"),
            state.evidence(),
            "O1",
            derive_review_dependencies=True,
        )


def test_unknown_assessment_reference_cannot_be_repaired_by_derivation():
    with pytest.raises(ContractError, match="evidence_unknown"):
        parse_grounded_review(
            review(["O9:n3"]),
            task(),
            {},
            "O9",
            derive_review_dependencies=True,
        )


def test_unsupported_requirement_cannot_become_complete():
    with pytest.raises(ContractError, match="review_contradiction"):
        parse_grounded_review(
            review([], status="unknown"),
            task(),
            {},
            "O1",
            derive_review_dependencies=True,
        )


def test_older_current_image_stays_stale_even_if_uploaded():
    state = TaskState()
    for pixels in (b"first", b"second"):
        state.observe(ui(), screenshot=pixels)
    evidence = state.evidence()
    with pytest.raises(ContractError, match="evidence_stale"):
        parse_grounded_review(
            review(["O1:image1"]),
            task(),
            evidence,
            "O1",
            uploaded_images={"O1:image1": evidence["O1:image1"]["sha256"]},
            derive_review_dependencies=True,
        )


@pytest.mark.parametrize("uploaded", [False, True])
def test_historical_pixels_require_actual_upload_and_cannot_prove_final_page(uploaded):
    state = TaskState()
    state.observe(ui([{"text": "First record"}]), screenshot=b"first")
    state.observe(ui([{"text": "Second record"}]), screenshot=b"second")
    evidence = state.evidence()
    images = {"O1:image1": evidence["O1:image1"]["sha256"]} if uploaded else {}
    args = (review(["O1:image1", "O2:image1"]), task(), evidence, "O2")
    if not uploaded:
        with pytest.raises(ContractError, match="evidence_stale"):
            parse_grounded_review(
                *args, uploaded_images=images, derive_review_dependencies=True
            )
        return
    parsed = parse_grounded_review(
        *args, uploaded_images=images, derive_review_dependencies=True
    )
    assert parsed["evidence_refs"] == ("O1:image1", "O2:image1")
    assert parsed["verification_refs"] == ("O2:image1",)
    with pytest.raises(ContractError, match="evidence_stale"):
        parse_grounded_review(
            review(["O1:image1"], answer=None),
            task("Keep this page visible", "final_state"),
            evidence,
            "O2",
            uploaded_images=images,
            derive_review_dependencies=True,
        )


@pytest.mark.parametrize("exact", [False, True])
def test_temporal_exception_retains_explicit_requested_value(exact):
    state = TaskState()
    for value in ("12", "13"):
        state.observe(ui([{"resourceId": "readout", "text": value}, {"text": "Pause"}]))
    data = json.loads(review(["O2:n0", "O2:n1"], answer=None))
    data["verification_refs"] = []
    parsed = parse_grounded_review(
        json.dumps(data),
        task("Stop the timer at 13" if exact else "Keep the timer running", "state"),
        state.evidence(),
        "O2",
        derive_review_dependencies=True,
    )
    assert parsed["verification_refs"] == (("O2:n0", "O2:n1") if exact else ("O2:n1",))


def test_only_temporal_readout_cannot_discard_its_only_live_witness():
    state = TaskState()
    for value in ("12", "13"):
        state.observe(ui([{"resourceId": "readout", "text": value}]))
    parsed = parse_grounded_review(
        review(["O2:n0"], answer=None),
        task("Keep the timer running", "state"),
        state.evidence(),
        "O2",
        derive_review_dependencies=True,
    )
    assert parsed["verification_refs"] == ("O2:n0",)


@pytest.mark.parametrize(
    "field",
    [{"editable": True}, {"inputValue": "13"}, {"rangeInfo": {"min": 0, "max": 20}}],
)
def test_numeric_input_and_range_are_mandatory_even_when_changing(field):
    state = TaskState()
    for value in ("12", "13"):
        state.observe(
            ui([{"resourceId": "value", "text": value, **field}, {"text": "Active"}])
        )
    parsed = parse_grounded_review(
        review(["O2:n0", "O2:n1"], answer=None),
        task("Keep the requested state", "state"),
        state.evidence(),
        "O2",
        derive_review_dependencies=True,
    )
    assert parsed["verification_refs"] == ("O2:n0", "O2:n1")


def test_legacy_parser_keeps_explicit_omission_refusal():
    state = TaskState()
    state.observe(ui([{"text": "Saved"}]), screenshot=b"current")
    data = json.loads(review(["O1:n0", "O1:image1"], answer=None))
    data["verification_refs"] = ["O1:n0"]
    with pytest.raises(ContractError, match="evidence_missing"):
        parse_grounded_review(
            json.dumps(data),
            task("Keep this page visible", "final_state"),
            state.evidence(),
            "O1",
        )


def test_dependency_switch_requires_boolean_and_prompt_keeps_grounding_rules():
    with pytest.raises(ValueError, match="must be a boolean"):
        JevConfig(derive_review_dependencies=1)
    assert JevConfig().derive_review_dependencies is True
    assert _grounded_review_prompt(False) == _GROUNDED_REVIEW_PROMPT
    prompt = _grounded_review_prompt(True)
    assert "最外层还输出verification_refs数组" not in prompt
    assert "客户端根据已校验的逐项证据构建完整实时核验依赖" in prompt
    assert "不能证明当前页面" in prompt
    assert "动作已派发不是效果确认" in prompt


@pytest.mark.parametrize("changed", ["pixels", "property"])
def test_derived_http_review_is_one_call_and_fresh_guard_still_rejects_changes(
    monkeypatch, changed
):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-key")
    requests, holders = [], []
    goal = "Keep the saved record visible"
    contract = task(goal, "final_state")

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        context = json.loads(payload["messages"][-1]["content"][0]["text"])
        assert (
            "最外层还输出verification_refs数组" not in payload["messages"][0]["content"]
        )
        current = holders[0][0].task_state.current_id
        refs = [current + ":n0", context["screenshot_evidence"]["ref"]]
        body = json.loads(review(refs, answer=None))
        body["verification_refs"] = refs[:1]
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
                client=client,
            )
            adapter._goal, adapter._completion_contract = goal, contract
            runner, driver, provider = _runtime(
                _ui([{"text": "Saved", "resourceId": "status"}]),
                visual=adapter,
            )
            runner.config.derive_review_dependencies = True
            runner.goal = goal
            runner.task_state.set_contract(contract)
            holders.append((runner, driver, provider))
            selected = await runner._visual(provider.ui, completion_review=True)
            assert selected.calls[0].name == "complete"
            assert selected.completion_proofs
            if changed == "pixels":
                driver.image = _image("blue")
            else:
                provider.ui = _ui([{"text": "Unsaved", "resourceId": "status"}])
            _, reason = await runner.validate_before_execute()
            assert reason == "completion_unverified"
            assert runner.history == []

    asyncio.run(run())
    assert len(requests) == 1
