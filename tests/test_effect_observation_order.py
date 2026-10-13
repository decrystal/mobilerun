"""A dispatch and a pre-dispatch page are not an observed resulting effect."""

import asyncio
import copy
import json

import httpx
import pytest
from test_decision_visual_guards import _image, _runtime
from test_task_evidence import review, task, ui

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.decision.contracts import ContractError, parse_grounded_review
from mobilerun.agent.decision.task_state import TaskState
from mobilerun.agent.decision.visual_fallback import VisualFallback
from mobilerun.agent.fast_agent.xml_parser import ToolCall


def dispatch(state):
    state.record_action(
        ToolCall("click", {"index": 1}), ActionResult(True, "Dispatched")
    )


def assess(state, refs, *, kind="state", complete=True, strict=True):
    return parse_grounded_review(
        review(refs, answer=None, complete=complete),
        task("Save the requested record", kind),
        state.evidence(),
        state.current_id,
        derive_review_dependencies=strict,
    )


@pytest.mark.parametrize(
    "kind,capture_after",
    [("state", False), ("artifact", False), ("collection", False), ("state", True)],
)
def test_stale_cited_action_cannot_use_pre_dispatch_content(kind, capture_after):
    state = TaskState()
    state.observe(ui([{"text": "Record", "resourceId": "record"}]))
    dispatch(state)
    if capture_after:
        state.observe(ui([{"text": "Saving", "resourceId": "status"}]))
    with pytest.raises(ContractError) as error:
        assess(state, ["A1", "O1:n0"], kind=kind)
    assert error.value.code == "effect_unverified"
    assert error.value.field == "post_action_observation"


@pytest.mark.parametrize("kind", ["state", "artifact", "collection", "final_state"])
def test_result_read_after_dispatch_supports_the_effect(kind):
    state = TaskState()
    state.observe(ui([{"text": "Draft"}]))
    dispatch(state)
    state.observe(ui([{"text": "Saved record"}]))
    assert assess(state, ["A1", "O2:n0"], kind=kind)["status"] == "done"


def test_before_and_after_comparison_keeps_both_witnesses():
    state = TaskState()
    state.observe(ui([{"text": "Before"}]))
    dispatch(state)
    state.observe(ui([{"text": "After"}]))
    refs = ["A1", "O1:n0", "O2:n0"]
    parsed = assess(state, refs)
    assert parsed["assessments"][0]["evidence_refs"] == refs


def test_result_observed_after_dispatch_remains_valid_after_navigation():
    state = TaskState()
    state.observe(ui([{"text": "Draft"}]))
    dispatch(state)
    state.observe(ui([{"text": "Saved record"}]))
    dispatch(state)
    state.observe(ui([{"text": "Another page"}]))
    assert assess(state, ["A1", "A2", "O2:n0"])["status"] == "done"


def test_unchanged_native_page_recaptured_after_dispatch_is_a_fresh_read():
    state = TaskState()
    page = ui([{"text": "Requested value"}])
    state.observe(page)
    dispatch(state)
    state.observe(page)
    assert len(state.observations) == 1
    assert assess(state, ["A1", "O1:n0"])["status"] == "done"


@pytest.mark.parametrize("kind", ["query", "process"])
def test_reading_historical_information_or_proving_a_process_is_not_an_effect(kind):
    state = TaskState()
    state.observe(ui([{"text": "Result"}]))
    dispatch(state)
    parsed = parse_grounded_review(
        review(["A1", "O1:n0"], answer="Result" if kind == "query" else None),
        task("Read the result" if kind == "query" else "Press the button", kind),
        state.evidence(),
        state.current_id,
        derive_review_dependencies=True,
    )
    assert parsed["status"] == "done"


def test_no_action_citation_still_accepts_an_already_observed_result():
    state = TaskState()
    state.observe(ui([{"text": "Saved record"}]))
    assert assess(state, ["O1:n0"])["status"] == "done"


def test_incomplete_review_can_preserve_supported_result_while_waiting_for_observation():
    state = TaskState()
    state.observe(ui([{"text": "Draft"}]))
    dispatch(state)
    assert assess(state, ["A1", "O1:n0"], complete=False)["status"] == "incomplete"


def test_legacy_dependency_mode_keeps_its_existing_protocol():
    state = TaskState()
    state.observe(ui([{"text": "Draft"}]))
    dispatch(state)
    assert assess(state, ["A1", "O1:n0"], strict=False)["status"] == "done"


@pytest.mark.parametrize("after", [None, "O0", "O1:n0", "O99", True])
def test_missing_or_impossible_action_readback_does_not_prove_a_result(after):
    state = TaskState()
    state.observe(ui([{"text": "Draft"}]))
    dispatch(state)
    state.observe(ui([{"text": "Saved record"}]))
    state.actions[0]["after"] = after
    with pytest.raises(ContractError, match="effect_unverified"):
        assess(state, ["A1", "O2:n0"])


@pytest.mark.parametrize("fresh", [False, True])
def test_calculated_collection_requires_all_its_native_sources_after_the_cited_action(
    fresh,
):
    state = TaskState()
    state.observe(ui([{"text": "Old row"}]))
    dispatch(state)
    state.observe(ui([{"text": "New row"}]))
    evidence = state.evidence()
    evidence["Q1"] = {
        "source": "deterministic_query",
        "coverage": {"complete": True},
        "evidence_refs": ["O2:n0"] if fresh else ["O1:n0", "O2:n0"],
    }
    args = (
        review(["A1", "Q1"], answer=None),
        task("Process the collection", "collection"),
        evidence,
        state.current_id,
    )
    if fresh:
        assert (
            parse_grounded_review(*args, derive_review_dependencies=True)["status"]
            == "done"
        )
    else:
        with pytest.raises(ContractError, match="effect_unverified"):
            parse_grounded_review(*args, derive_review_dependencies=True)


@pytest.mark.parametrize("fresh", [False, True])
def test_uploaded_screenshot_result_must_follow_the_action(fresh):
    state = TaskState()
    state.observe(ui([{"text": "Draft"}]), screenshot=b"before")
    dispatch(state)
    state.observe(ui([{"text": "Saved record"}]), screenshot=b"after")
    evidence = state.evidence()
    ref = "O2:image1" if fresh else "O1:image1"
    args = (
        review(["A1", ref], answer=None),
        task("Save the record", "state"),
        evidence,
        state.current_id,
    )
    uploaded = {ref: evidence[ref]["sha256"]}
    if fresh:
        assert (
            parse_grounded_review(
                *args,
                uploaded_images=uploaded,
                derive_review_dependencies=True,
            )["status"]
            == "done"
        )
    else:
        with pytest.raises(ContractError, match="effect_unverified"):
            parse_grounded_review(
                *args, uploaded_images=uploaded, derive_review_dependencies=True
            )


def test_image_from_before_dispatch_is_not_a_post_action_image_under_reused_native_id():
    state = TaskState()
    page = ui([{"text": "Record"}])
    state.observe(page, screenshot=b"before")
    dispatch(state)
    state.observe(page, screenshot=b"after")
    assert state.actions[0]["after"] == "O1"
    state.observe(ui([{"text": "Another page"}]), screenshot=b"other")
    evidence = state.evidence()
    with pytest.raises(ContractError, match="effect_unverified"):
        parse_grounded_review(
            review(["A1", "O1:image1"], answer=None),
            task("Save the record", "state"),
            evidence,
            state.current_id,
            uploaded_images={"O1:image1": evidence["O1:image1"]["sha256"]},
            derive_review_dependencies=True,
        )


@pytest.mark.parametrize("fresh", [False, True])
def test_http_reviewer_reports_temporal_gap_without_an_additional_request(
    monkeypatch, fresh
):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-key")
    state = TaskState()
    state.observe(ui([{"text": "Draft"}]))
    dispatch(state)
    state.observe(ui([{"text": "Saved record"}]))
    goal = "Save the requested record"
    contract = task(goal, "state")
    state.set_contract(contract)
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        refs = ["A1", "O2:n0" if fresh else "O1:n0"]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": review(refs, answer=None)}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = VisualFallback(
                model="glm-5.3-flash",
                reasoning_model="glm-5.3-flash",
                base_url="https://example.invalid/v1",
                api_key_env="GLM_API_KEY",
                client=client,
            )
            adapter._goal, adapter._completion_contract = goal, contract
            ledger = copy.deepcopy(state.context())
            return await adapter.decide(
                goal=goal,
                screenshot=_image("red"),
                ui_state=ui([{"text": "Saved record"}]),
                available_tools={"complete"},
                task_state={"ledger": ledger},
                completion_review=True,
                native_only_review=True,
                derive_review_dependencies=True,
            )

    decision = asyncio.run(run())
    assert len(requests) == 1
    if fresh:
        assert decision.status == "done"
    else:
        assert decision.status == "error"
        assert decision.error_code == "effect_unverified"
        assert decision.error_field == "post_action_observation"


def test_runtime_uses_existing_single_review_retry_and_preserves_no_replay(monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-key")
    requests = []
    goal = "Save the requested record"
    contract = task(goal, "state")

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if len(requests) == 2:
            context = json.loads(payload["messages"][-1]["content"][0]["text"])
            assert "post_action_observation" in context["format_feedback"]
            assert "instead of replaying" in context["format_feedback"]
        refs = ["A1", "O1:n0" if len(requests) == 1 else "O2:n0"]
        return httpx.Response(
            200, json={"choices": [{"message": {"content": review(refs, answer=None)}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = VisualFallback(
                model="glm-5.3-flash",
                reasoning_model="glm-5.3-flash",
                base_url="https://example.invalid/v1",
                api_key_env="GLM_API_KEY",
                client=client,
            )
            adapter._goal, adapter._completion_contract = goal, contract
            runner, _, provider = _runtime(
                ui([{"text": "Saved record"}]), visual=adapter
            )
            runner.config.derive_review_dependencies = True
            runner.goal = goal
            runner.task_state = TaskState(goal)
            runner.task_state.set_contract(contract)
            runner.task_state.observe(ui([{"text": "Draft"}]))
            dispatch(runner.task_state)
            runner.task_state.observe(provider.ui)
            selected = await runner._visual(
                provider.ui, completion_review=True, native_only_review=True
            )
            assert selected.calls[0].name == "complete"
            assert runner.history == []
            assert runner.executed == set()
            assert sum(event["event"] == "format_retry" for event in runner.trace) == 1

    asyncio.run(run())
    assert len(requests) == 2
