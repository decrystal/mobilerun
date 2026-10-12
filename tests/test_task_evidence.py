"""Behavioral regressions for evidence-based goals, not named benchmark tasks."""

import asyncio
import base64
import copy
import hashlib
import json
from io import BytesIO
from types import SimpleNamespace

import httpx
import pytest
from PIL import Image

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.decision.autoglm import AutoGLMDecision, AutoGLMFallback
from mobilerun.agent.decision.contracts import (
    ContractError,
    parse_grounded_review,
    parse_task_contract,
)
from mobilerun.agent.decision.input_verification import (
    capture_input_target,
    read_input_value,
)
from mobilerun.agent.decision.task_state import TaskState, corroborate_nodes
from mobilerun.agent.fast_agent.xml_parser import ToolCall
from mobilerun.tools.ui.state import UIState


def ui(nodes=(), package="com.example.records"):
    return UIState(list(nodes), "", "", {"packageName": package}, 400, 800)


def task(goal="Is this item completed?", kind="query"):
    return parse_task_contract(
        json.dumps(
            {
                "requirements": [
                    {"id": 1, "description": goal, "kind": kind, "quote": goal}
                ],
                "answer_required": kind == "query",
            }
        ),
        goal,
    )


def review(refs, *, answer="False", status="supported", complete=True):
    return json.dumps(
        {
            "complete": complete,
            "evidence": "Inspected the identified record",
            "checks": {},
            "final_answer": answer,
            "assessments": [
                {
                    "id": 1,
                    "status": status,
                    "evidence": "The observed property answers the question",
                    "evidence_refs": refs,
                }
            ],
        }
    )


@pytest.mark.parametrize("answer", ["False", "0", "True", "2 cups", "None found"])
def test_query_success_is_independent_of_answer_truth_value(answer):
    state = TaskState()
    state.observe(ui([{"index": 1, "text": "Property", "isChecked": False}]))
    result = parse_grounded_review(
        review(["O1:n0"], answer=answer), task(), state.evidence(), "O1"
    )
    assert result["status"] == "done" and result["final_answer"] == answer


@pytest.mark.parametrize("status", ["unknown", "contradicted"])
def test_unknown_and_contradicted_requirements_cannot_be_certified(status):
    with pytest.raises(ContractError, match="review_contradiction"):
        parse_grounded_review(review([], status=status), task(), {}, "O1")
    result = parse_grounded_review(
        review([], status=status, complete=False), task(), {}, "O1"
    )
    assert result["status"] == "incomplete" and result["final_answer"] is None


@pytest.mark.parametrize(
    "refs,code",
    [
        ([], "evidence_missing"),
        (["O9:n0"], "evidence_unknown"),
        (["A1"], "effect_unverified"),
        (["O1"], "effect_unverified"),
    ],
)
def test_queries_require_actual_content_evidence(refs, code):
    state = TaskState()
    state.observe(ui([{"text": "Result"}]))
    state.record_action(ToolCall("click", {"index": 1}), ActionResult(True, "clicked"))
    with pytest.raises(ContractError, match=code):
        parse_grounded_review(review(refs), task(), state.evidence(), "O1")


def test_saved_result_can_be_proven_before_leaving_editor_but_final_position_cannot():
    state = TaskState()
    state.observe(ui([{"text": "Saved record"}]))
    state.observe(ui([{"text": "Other page"}]))
    result = parse_grounded_review(
        review(["O1:n0"], answer=None),
        task("Save the record", "state"),
        state.evidence(),
        "O2",
    )
    assert result["status"] == "done"
    with pytest.raises(ContractError, match="evidence_stale"):
        parse_grounded_review(
            review(["O1:n0"], answer=None),
            task("Leave this page open", "final_state"),
            state.evidence(),
            "O2",
        )


def test_observation_values_are_immutable_complete_and_keep_parent_relationships():
    page = ui(
        [
            {
                "index": 8,
                "text": "card",
                "children": [
                    {"index": 9, "inputValue": "x" * 1000, "isChecked": False},
                    {"index": 10, "password": True, "text": "private"},
                ],
            }
        ]
    )
    state = TaskState()
    first = state.observe(page)
    page.elements[0]["children"][0]["inputValue"] = "changed"
    first["nodes"][1]["inputValue"] = "also changed"
    assert state.evidence()["O1:n1"]["inputValue"] == "x" * 1000
    assert state.evidence()["O1:n1"]["parent_path"] == 0
    assert "private" not in str(state.context())


def test_repeated_observations_dont_inflate_evidence_or_silently_replace_images():
    state = TaskState()
    state.observe(ui(), screenshot=b"first")
    state.observe(ui(), screenshot=b"second")
    assert len(state.observations) == 1
    assert len(state.observations[0]["images"]) == 2
    with pytest.raises(ContractError, match="evidence_stale"):
        parse_grounded_review(review(["O1:image1"]), task(), state.evidence(), "O1")
    assert (
        parse_grounded_review(review(["O1:image2"]), task(), state.evidence(), "O1")[
            "status"
        ]
        == "done"
    )


def test_an_old_screenshot_hash_is_not_visual_evidence_of_a_new_page():
    state = TaskState()
    state.observe(ui(), screenshot=b"first")
    state.observe(ui([{"text": "Changed"}]), screenshot=b"second")
    with pytest.raises(ContractError, match="evidence_stale"):
        parse_grounded_review(review(["O1:image1"]), task(), state.evidence(), "O2")


def test_history_image_is_query_evidence_only_when_exact_pixels_are_uploaded():
    state = TaskState()
    state.observe(ui([{"text": "First record"}]), screenshot=b"first")
    state.observe(ui([{"text": "Second record"}]), screenshot=b"second")
    known = state.evidence()
    refs = ["O1:image1", "O2:image1"]
    uploaded = {ref: known[ref]["sha256"] for ref in refs}
    result = parse_grounded_review(
        review(refs, answer="2"), task(), known, "O2", uploaded_images=uploaded
    )
    assert result["status"] == "done"
    assert result["verification_refs"] == ("O2:image1",)
    for invalid in [{}, {"O1:image1": "wrong hash"}]:
        with pytest.raises(ContractError, match="evidence_stale"):
            parse_grounded_review(
                review(refs), task(), known, "O2", uploaded_images=invalid
            )
    with pytest.raises(ContractError, match="evidence_stale"):
        parse_grounded_review(
            review(refs, answer=None),
            task("Leave this page open", "final_state"),
            known,
            "O2",
            uploaded_images=uploaded,
        )
    final = json.loads(review(refs))
    final["verification_refs"] = ["O1:image1"]
    with pytest.raises(ContractError, match="evidence_missing") as rejected:
        parse_grounded_review(
            json.dumps(final), task(), known, "O2", uploaded_images=uploaded
        )
    assert rejected.value.field == "verification_refs"
    final["verification_refs"] = refs
    projected = parse_grounded_review(
        json.dumps(final), task(), known, "O2", uploaded_images=uploaded
    )
    assert projected["evidence_refs"] == tuple(refs)
    assert projected["verification_refs"] == ("O2:image1",)


def test_visual_final_state_cannot_omit_its_live_image_guard():
    state = TaskState()
    state.observe(ui(), screenshot=b"current")
    final = json.loads(review(["O1:image1"], answer=None))
    final["verification_refs"] = []
    with pytest.raises(ContractError, match="evidence_missing") as rejected:
        parse_grounded_review(
            json.dumps(final),
            task("Leave this page open", "final_state"),
            state.evidence(),
            "O1",
        )
    assert rejected.value.field == "verification_refs"
    assert rejected.value.refs == ("O1:image1",)


@pytest.mark.parametrize("kind", ["state", "final_state"])
def test_review_reports_omitted_current_image_without_accepting_completion(kind):
    state = TaskState()
    state.observe(ui([{"text": "Saved"}]), screenshot=b"current")
    data = json.loads(review(["O1:n0", "O1:image1"], answer=None))
    data["verification_refs"] = ["O1:n0"]
    contract = task("Show the saved view", kind)
    with pytest.raises(ContractError, match="evidence_missing") as rejected:
        parse_grounded_review(json.dumps(data), contract, state.evidence(), "O1")
    assert rejected.value.field == "omitted_state_property"
    assert rejected.value.refs == ("O1:image1",)
    data["verification_refs"].append("O1:image1")
    result = parse_grounded_review(json.dumps(data), contract, state.evidence(), "O1")
    assert result["verification_refs"] == ("O1:n0", "O1:image1")
    assert result["status"] == "done"


def test_diagnostic_references_exclude_observed_values_and_unstructured_model_text():
    invalid = (None, 7, "private text", "O1", "A1", "O2:image1\nprivate text")
    valid = tuple(f"O2:n{i}" for i in range(20))
    error = ContractError(
        "evidence_missing", "omitted_state_property", refs=invalid + valid + valid
    )
    assert error.refs == valid[:16]
    assert str(error) == "evidence_missing"
    assert "private text" not in str(error.__dict__)


def test_each_state_requirement_reports_its_own_missing_guard():
    goal = "Save the item and leave the confirmation view open"
    contract = parse_task_contract(
        json.dumps(
            {
                "requirements": [
                    {
                        "id": 1,
                        "description": "Save the item",
                        "quote": "Save the item",
                        "kind": "state",
                    },
                    {
                        "id": 2,
                        "description": "leave the confirmation view open",
                        "quote": "leave the confirmation view open",
                        "kind": "final_state",
                    },
                ],
                "answer_required": False,
            }
        ),
        goal,
    )
    state = TaskState()
    state.observe(ui([{"text": "Saved"}]), screenshot=b"confirmation")
    data = json.loads(review(["O1:n0"], answer=None))
    data["assessments"].append(
        {
            "id": 2,
            "status": "supported",
            "evidence": "Visible confirmation view",
            "evidence_refs": ["O1:image1"],
        }
    )
    data["verification_refs"] = ["O1:n0"]
    with pytest.raises(ContractError, match="evidence_missing") as rejected:
        parse_grounded_review(json.dumps(data), contract, state.evidence(), "O1")
    assert rejected.value.field == "state_verification_refs"
    assert rejected.value.refs == ("O1:image1",)


@pytest.mark.parametrize("correct_on_retry", [True, False])
def test_real_review_retry_reports_missing_image_and_requires_recaptured_reference(
    monkeypatch,
    correct_on_retry,
):
    from test_decision_visual_guards import _image, _runtime, _ui

    monkeypatch.setenv("BIGMODEL_API_KEY", "test-secret")
    goal = "Leave the completed view open"
    contract = task(goal, "final_state")
    requests = []

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = AutoGLMFallback(
                model="glm-5.3-flash",
                reasoning_model="glm-5.3-flash",
                client=client,
                base_url="https://example.invalid/v1",
            )
            adapter._goal = goal
            adapter._completion_contract = contract
            runner, driver, _ = _runtime(_ui([{"text": "Saved"}]), visual=adapter)
            driver_holder.append(driver)
            runner.goal = goal
            runner.task_state = TaskState(goal)
            runner.task_state.set_contract(contract)
            selection = await runner._visual(runner.ctx.ui, completion_review=True)
            decisions = [row for row in runner.trace if row["event"] == "decision"]
            assert decisions[0]["error_field"] == "omitted_state_property"
            assert decisions[0]["error_refs"] == ["O1:image1"]
            assert decisions[1]["status"] == ("done" if correct_on_retry else "error")
            if correct_on_retry:
                assert not selection.stop
                assert [call.name for call in selection.calls] == ["complete"]
                assert (
                    selection.completion_image_guard
                    == hashlib.sha256(driver.image).hexdigest()
                )
                _, reason = await runner.validate_before_execute()
                assert reason is None
            else:
                assert selection.stop and selection.calls == []
                assert decisions[1]["error_refs"] == ["O1:image2"]
            assert runner.history == []
            assert not any(row["event"] == "action" for row in runner.trace)
            retries = [row for row in runner.trace if row["event"] == "format_retry"]
            assert len(retries) == 1 and not retries[0]["action_executed"]

    def respond(request):
        payload = json.loads(request.content)
        context = json.loads(payload["messages"][-1]["content"][0]["text"])
        requests.append(context)
        current = context["screenshot_evidence"]
        image_ref = current["ref"]
        assert payload["model"] == "glm-5.3-flash"
        assert image_ref == f"O1:image{len(requests)}"
        if len(requests) == 1:
            driver_holder[0].image = _image("green")
        else:
            feedback = context["format_feedback"]
            assert "Actual references omitted from the previous reply" in feedback
            assert "O1:image1" in feedback
            assert "newly uploaded screenshot_evidence.ref" in feedback
            assert "test-secret" not in feedback
            image = next(
                item
                for item in payload["messages"][-1]["content"]
                if item["type"] == "image_url"
            )
            pixels = base64.b64decode(image["image_url"]["url"].split(",", 1)[1])
            assert hashlib.sha256(pixels).hexdigest() == current["sha256"]
        response = json.loads(review(["O1:n0", image_ref], answer=None))
        response["verification_refs"] = ["O1:n0"]
        if len(requests) == 2 and correct_on_retry:
            response["verification_refs"].append(image_ref)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(response)}}]}
        )

    driver_holder = []
    asyncio.run(run())
    assert len(requests) == 2


@pytest.mark.parametrize("invalid_answer", ["", 0, False, None])
@pytest.mark.parametrize(
    "retry_outcome", ["corrected", "invalid", "changed_after_review"]
)
def test_real_answer_format_retry_preserves_negative_answer_and_live_evidence(
    monkeypatch, invalid_answer, retry_outcome
):
    from test_decision_visual_guards import _image, _runtime, _ui

    monkeypatch.setenv("BIGMODEL_API_KEY", "test-secret")
    goal = "What count is displayed?"
    contract = task(goal)
    requests = []
    drivers = []

    def respond(request):
        payload = json.loads(request.content)
        context = json.loads(payload["messages"][-1]["content"][0]["text"])
        requests.append(context)
        assert context["task_state"]["ledger"]["contract"] == contract
        if len(requests) == 1:
            drivers[0].image = _image("green")
        else:
            assert "non-empty string" in context["format_feedback"]
            assert "Do not guess an answer" in context["format_feedback"]
        answer = (
            "0" if len(requests) == 2 and retry_outcome != "invalid" else invalid_answer
        )
        return httpx.Response(
            200,
            json={
                "choices": [{"message": {"content": review(["O1:n0"], answer=answer)}}]
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = AutoGLMFallback(
                model="glm-5.3-flash",
                reasoning_model="glm-5.3-flash",
                client=client,
                base_url="https://example.invalid/v1",
            )
            adapter._goal = goal
            adapter._completion_contract = contract
            runner, driver, provider = _runtime(
                _ui([{"index": 1, "text": "0"}]), visual=adapter
            )
            drivers.append(driver)
            runner.goal = goal
            runner.task_state.set_contract(contract)
            selected = await runner._visual(runner.ctx.ui, completion_review=True)
            decisions = [row for row in runner.trace if row["event"] == "decision"]
            assert decisions[0]["status"] == "error"
            assert decisions[0]["error_field"] == "final_answer"
            assert decisions[0]["error_code"] == (
                "answer_missing" if invalid_answer is None else "protocol_error"
            )
            assert len(requests) == 2
            if retry_outcome == "invalid":
                assert selected.stop and selected.calls == []
                assert decisions[1]["status"] == "error"
            else:
                assert not selected.stop and [call.name for call in selected.calls] == [
                    "complete"
                ]
                assert selected.calls[0].parameters["final_answer"] == "0"
                if retry_outcome == "changed_after_review":
                    provider.ui = _ui([{"index": 1, "text": "1"}])
                _, reason = await runner.validate_before_execute()
                assert reason == (
                    "completion_unverified"
                    if retry_outcome == "changed_after_review"
                    else None
                )
                guards = [
                    row for row in runner.trace if row["event"] == "completion_guard"
                ]
                assert guards[-1]["failed_check"] == (
                    "native_evidence"
                    if retry_outcome == "changed_after_review"
                    else None
                )
                assert guards[-1]["success"] is (
                    retry_outcome != "changed_after_review"
                )
            assert runner.history == []
            assert not any(row["event"] == "action" for row in runner.trace)
            retries = [row for row in runner.trace if row["event"] == "format_retry"]
            assert len(retries) == 1 and not retries[0]["action_executed"]

    asyncio.run(run())


def test_retry_diagnostics_drop_unknown_whole_observation_and_action_references():
    from test_decision_visual_guards import _runtime, _ui, _Visual

    requests = []

    class InvalidReview(_Visual):
        async def decide(self, **kwargs):
            requests.append(kwargs)
            return AutoGLMDecision(
                "error",
                message="Completion reviewer returned invalid evidence.",
                error_code="evidence_missing",
                error_field="omitted_state_property",
                error_refs=("O1:n0", "O9:image1", "private text", "O1", "A1", None, 8),
            )

    runner, _, _ = _runtime(_ui([{"text": "private value"}]), visual=InvalidReview())
    runner.goal = "Leave this page open"
    runner.task_state.set_contract(task(runner.goal, "final_state"))
    selection = asyncio.run(runner._visual(runner.ctx.ui, completion_review=True))
    assert selection.stop and selection.calls == []
    assert len(requests) == 2
    decisions = [row for row in runner.trace if row["event"] == "decision"]
    assert [row["error_refs"] for row in decisions] == [["O1:n0"], ["O1:n0"]]
    feedback = requests[1]["review_feedback"]
    assert "O1:n0" in feedback
    assert all(
        value not in feedback
        for value in ("O9:image1", "private text", "private value", "A1")
    )
    assert runner.history == []


def test_historical_upload_cannot_certify_old_pixels_of_the_current_observation():
    state = TaskState()
    for screenshot in [b"first", b"second"]:
        state.observe(ui(), screenshot=screenshot)
    known = state.evidence()
    with pytest.raises(ContractError, match="evidence_stale"):
        parse_grounded_review(
            review(["O1:image1"]),
            task(),
            known,
            "O1",
            uploaded_images={"O1:image1": known["O1:image1"]["sha256"]},
        )


@pytest.mark.parametrize("cache_limit", [2, 8])
@pytest.mark.parametrize("history_enabled", [True, False])
def test_multiscreen_review_uploads_bounded_actual_images_and_resets_between_tasks(
    monkeypatch, cache_limit, history_enabled
):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-secret")
    monkeypatch.setattr(
        "mobilerun.agent.decision.autoglm._MAX_CACHED_IMAGES", cache_limit
    )
    goal = "Count the observed entries"
    contract = task(goal)
    state = TaskState(goal)
    state.set_contract(contract)
    requests = []
    pictures = []
    for color in ["red", "green", "blue", "white"]:
        stream = BytesIO()
        Image.new("RGB", (400, 800), color).save(stream, format="PNG")
        pictures.append(stream.getvalue())

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        content = payload["messages"][-1]["content"]
        context = json.loads(content[0]["text"])
        if "uploaded_screenshot_evidence" in context:
            metadata = context["uploaded_screenshot_evidence"]
            images = [
                item["image_url"]["url"]
                for item in content
                if item["type"] == "image_url"
            ]
            count = min(cache_limit, 3) if history_enabled else 1
            assert [item["ref"] for item in metadata] == [
                f"O{4 - index}:image1" for index in range(count)
            ]
            assert [item["current"] for item in metadata] == [True] + [False] * (
                count - 1
            )
            assert len(images) == count
            for item, url, pixels in zip(
                metadata, images, list(reversed(pictures))[:count], strict=True
            ):
                actual = base64.b64decode(url.split(",", 1)[1])
                assert (
                    actual == pixels
                    and hashlib.sha256(actual).hexdigest() == item["sha256"]
                )
            result = json.loads(review([item["ref"] for item in metadata], answer="3"))
            result["verification_refs"] = ["O4:image1"]
            output = json.dumps(result)
        else:
            output = '{"action":"Wait","seconds":1}'
        return httpx.Response(200, json={"choices": [{"message": {"content": output}}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = AutoGLMFallback(reasoning_model="reasoner", client=client)
            adapter._goal = goal
            adapter._completion_contract = contract
            for index, picture in enumerate(pictures):
                page = ui([{"index": 1, "text": f"Record {index}"}])
                state.observe(page, screenshot=picture)
                result = await adapter.decide(
                    goal=goal,
                    screenshot=picture,
                    ui_state=page,
                    available_tools={"wait", "complete"},
                    recovery=True,
                    task_state={"ledger": state.context()},
                    completion_review=index == 3,
                    review_history_images=history_enabled,
                )
            assert result.status == "done" and result.final_answer == "3"
            adapter.reset()
            assert adapter._review_images == {}

    asyncio.run(run())
    assert len(requests) == 4


def test_visual_change_then_return_gets_new_current_proof_without_inflating_identical_captures():
    state = TaskState()
    for value in [b"first", b"second", b"first", b"first"]:
        state.observe(ui(), screenshot=value)
    images = state.observations[0]["images"]
    assert len(images) == 3 and images[0]["sha256"] == images[2]["sha256"]
    assert (
        parse_grounded_review(review(["O1:image3"]), task(), state.evidence(), "O1")[
            "status"
        ]
        == "done"
    )
    for old in ["O1:image1", "O1:image2"]:
        with pytest.raises(ContractError, match="evidence_stale"):
            parse_grounded_review(review([old]), task(), state.evidence(), "O1")


@pytest.mark.parametrize("uploaded_current", [True, False])
def test_reviewer_gets_exact_image_reference_and_available_ids_not_a_guessed_number(
    monkeypatch, uploaded_current
):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-secret")
    pictures = []
    for color in ["white", "black"]:
        stream = BytesIO()
        Image.new("RGB", (400, 800), color).save(stream, format="PNG")
        pictures.append(stream.getvalue())
    goal = "Inspect the displayed state"
    contract = task(goal, "state")
    state = TaskState(goal)
    state.set_contract(contract)
    for picture in pictures:
        state.observe(ui(), screenshot=picture)

    def respond(request):
        payload = json.loads(request.content)
        context = json.loads(payload["messages"][-1]["content"][0]["text"])
        assert context["available_evidence_refs"] == ["O1", "O1:image1", "O1:image2"]
        if uploaded_current:
            assert context["screenshot_evidence"]["ref"] == "O1:image2"
            content = review([context["screenshot_evidence"]["ref"]], answer=None)
        else:
            assert context["screenshot_evidence"] is None
            content = review(["O1:image1"], answer=None)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = AutoGLMFallback(reasoning_model="reasoner", client=client)
            adapter._goal = goal
            adapter._completion_contract = contract
            return await adapter.decide(
                goal=goal,
                screenshot=pictures[1 if uploaded_current else 0],
                ui_state=ui(),
                available_tools={"complete"},
                task_state={"ledger": state.context()},
                completion_review=True,
            )

    result = asyncio.run(run())
    assert result.status == ("done" if uploaded_current else "error")
    if not uploaded_current:
        assert result.error_code == "evidence_stale"


def test_omitted_pages_are_explicit_and_accepted_evidence_is_retained():
    state = TaskState()
    for index in range(12):
        state.observe(ui([{"text": f"Record {index}"}]))
    state.assessments[1] = {"status": "supported", "evidence_refs": ["O1:n0"]}
    context = state.context(max_observations=2)
    assert [item["id"] for item in context["observations"]] == ["O1", "O11", "O12"]
    assert context["observations_omitted"] == 9


def test_typed_reviewer_receives_current_native_boundary_provenance(monkeypatch):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-secret")
    stream = BytesIO()
    Image.new("RGB", (400, 800), "white").save(stream, format="PNG")
    screenshot = stream.getvalue()
    goal = "Inspect the displayed entries"
    contract = task(goal, "state")
    page = ui(
        [
            {
                "index": 1,
                "className": "ScrollView",
                "bounds": "0,0,400,800",
                "scrollable": True,
                "childCount": 2,
                "children": [
                    {"index": 2, "text": "First", "bounds": "0,10,400,200"},
                    {"index": 3, "text": "Second", "bounds": "0,250,400,400"},
                ],
            }
        ]
    )
    state = TaskState(goal)
    state.set_contract(contract)
    state.observe(page, screenshot=screenshot)

    def respond(request):
        payload = json.loads(request.content)
        context = json.loads(payload["messages"][-1]["content"][0]["text"])
        observed = context["task_state"]["ledger"]["observations"][0]
        container = observed["containers"][0]
        assert container["children_total"] == container["fully_visible_children"] == 2
        assert container["child_refs"] == ["O1:n1", "O1:n2"]
        assert set([container["ref"], *container["child_refs"]]) <= set(
            context["available_evidence_refs"]
        )
        assert context["controls"] == []
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": review(
                                [], answer=None, status="unknown", complete=False
                            )
                        }
                    }
                ]
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = AutoGLMFallback(reasoning_model="reasoner", client=client)
            adapter._goal = goal
            adapter._completion_contract = contract
            return await adapter.decide(
                goal=goal,
                screenshot=screenshot,
                ui_state=page,
                available_tools={"complete"},
                task_state={"ledger": state.context()},
                completion_review=True,
            )

    assert asyncio.run(run()).status == "incomplete"


def test_contract_cannot_add_requirements_with_fabricated_goal_quotes():
    data = task()
    data["requirements"][0]["quote"] = "Save ten files"
    with pytest.raises(ContractError, match="contract_invalid"):
        parse_task_contract(json.dumps(data), "Is this item completed?")


def test_input_diagnostics_distinguish_missing_field_and_observed_wrong_value():
    before = ui(
        [
            {
                "index": 1,
                "className": "EditText",
                "editable": True,
                "focused": True,
                "resourceId": "field",
                "inputValue": "",
                "bounds": "0,0,200,60",
            }
        ]
    )
    target = asyncio.run(capture_input_target(object(), before))
    wrong = copy.deepcopy(before)
    wrong.elements[0]["inputValue"] = "wrong"
    assert (
        asyncio.run(read_input_value(object(), wrong, target, "right")).status
        == "mismatch"
    )
    assert (
        asyncio.run(read_input_value(object(), ui(), target, "right")).status
        == "unknown"
    )


def test_real_adapter_prepares_once_and_reviews_historical_send_evidence(monkeypatch):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-secret")
    requests = []
    goal = "Send the message hello"
    contract = task(goal, "state")
    state = TaskState(goal)
    state.set_contract(contract)
    state.observe(ui([{"text": "hello", "className": "Message"}]))
    page = ui([{"text": "Inbox", "className": "EditText", "inputValue": ""}])
    state.observe(page)
    buf = BytesIO()
    Image.new("RGB", (400, 800)).save(buf, format="PNG")
    state.observe(page, screenshot=buf.getvalue())

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        user = payload["messages"][-1]["content"]
        content = json.dumps(
            {"requirements": contract["requirements"], "answer_required": False}
        )
        if isinstance(user, list):
            context = json.loads(user[0]["text"])
            assert context["completion_contract"]["schema_version"] == 2
            assert "test-secret" not in str(context)
            content = review(["O1:n0"], answer=None)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = AutoGLMFallback(reasoning_model="reasoner", client=client)
            assert await adapter.prepare_task(goal=goal) is None
            assert await adapter.prepare_task(goal=goal) is None
            result = await adapter.decide(
                goal=goal,
                screenshot=buf.getvalue(),
                ui_state=page,
                available_tools={"complete"},
                task_state={"ledger": state.context()},
                completion_review=True,
            )
            assert result.status == "done" and result.completion_checks == {}

    asyncio.run(run())
    assert len(requests) == 2
    assert requests[0]["messages"][-1]["content"] == goal
    assert "image_url" not in str(requests[0])


@pytest.mark.parametrize("repair_enabled", [True, False])
def test_real_runtime_allows_only_one_observed_same_field_input_repair(
    monkeypatch, repair_enabled
):
    from test_jev_first_runtime import runtime

    page = ui(
        [
            {
                "index": 1,
                "className": "EditText",
                "editable": True,
                "focused": True,
                "resourceId": "field",
                "inputValue": "",
                "bounds": "0,0,200,60",
            }
        ]
    )
    runner, _, _, _, provider = runtime(ui=page)
    runner.config.input_repair = repair_enabled
    runner.task_state.set_contract(task("Enter right", "process"))
    runner.goal = runner.task_state.goal = "Enter right"
    call = ToolCall("type", {"text": "right", "clear": True})
    runner.input_target = asyncio.run(capture_input_target(runner.ctx.driver, page))
    ticks = iter([0, 4, 8, 12])
    monkeypatch.setattr(
        "mobilerun.agent.decision.runtime.time",
        SimpleNamespace(monotonic=lambda: next(ticks)),
    )
    assert asyncio.run(runner.verify_input(call)) is False
    if not repair_enabled:
        assert runner.input_retry is None
        assert runner.last_input_readback.status == "mismatch"
        return
    assert runner.input_retry is not None
    selection = asyncio.run(runner.select("Enter right", page))
    assert selection.source == "input_activation" and selection.calls == [
        ToolCall("click", {"index": 1})
    ]
    selection = asyncio.run(runner.select("Enter right", page))
    assert selection.source == "input_recovery" and selection.calls == [call]
    # A second failure never queues another replacement.
    assert asyncio.run(runner.verify_input(call)) is False
    assert runner.input_retry is None


def test_real_runtime_native_phone_format_confirms_without_a_repair_or_model_call():
    from test_jev_first_runtime import runtime

    page = ui(
        [
            {
                "index": 1,
                "className": "EditText",
                "editable": True,
                "focused": True,
                "resourceId": "field",
                "inputValue": "",
                "inputType": 3,
                "bounds": "0,0,200,60",
            }
        ]
    )
    runner, events, policy, visual, provider = runtime(ui=page)
    runner.goal = runner.task_state.goal = "Enter the phone number"
    runner.task_state.set_contract(task(runner.goal, "process"))
    runner.input_target = asyncio.run(capture_input_target(runner.ctx.driver, page))
    after = copy.deepcopy(page)
    after.elements[0]["inputValue"] = "+44 1234-567890"
    after.elements[0]["focused"] = False
    provider.ui = after
    call = ToolCall("type", {"text": "+441234567890", "clear": True})
    assert asyncio.run(runner.verify_input(call)) is True
    assert runner.last_input_readback.reason == "native_phone_format"
    assert runner.input_retry is None and runner.input_retry_counts == {}
    assert runner.history == [] and events == []
    assert policy.requests == visual.requests == []


def test_changed_input_focus_blocks_replacement_after_a_failed_attempt(monkeypatch):
    from test_jev_first_runtime import runtime

    page = ui(
        [
            {
                "index": 1,
                "className": "EditText",
                "editable": True,
                "focused": True,
                "resourceId": "field",
                "inputValue": "",
                "bounds": "0,0,200,60",
            }
        ]
    )
    runner, _, _, _, _ = runtime(ui=page)
    runner.goal = runner.task_state.goal = "Enter right"
    runner.task_state.set_contract(task("Enter right", "process"))
    call = ToolCall("type", {"text": "right", "clear": True})
    target = asyncio.run(capture_input_target(runner.ctx.driver, page))
    runner.input_retry = (call, target)
    changed = copy.deepcopy(page)
    changed.elements[0]["resourceId"] = "different-field"
    selection = asyncio.run(runner.select("Enter right", changed))
    assert selection.stop and not selection.calls


def test_completion_corroborates_the_state_control_and_ignores_an_unrelated_dynamic_readout():
    state = TaskState()
    page = ui(
        [
            {"text": "01", "className": "TextView", "resourceId": "readout"},
            {"text": "Pause", "className": "Button", "resourceId": "toggle"},
        ]
    )
    state.observe(page)
    proof = state.evidence()["O1:n1"]
    later = copy.deepcopy(page)
    later.elements[0]["text"] = "02"
    assert corroborate_nodes([proof], later)
    later.elements[1]["text"] = "Start"
    assert not corroborate_nodes([proof], later)


def test_model_context_is_bounded_without_destroying_original_evidence():
    state = TaskState()
    for number in range(8):
        state.observe(ui([{"text": f"{number}-{row}" * 40} for row in range(100)]))
    context = state.context(max_chars=4000)
    assert len(json.dumps(context)) < 8000
    assert len(state.observations) == 8 and len(state.evidence()) == 808
    assert sum(item["nodes_omitted"] for item in context["observations"]) > 0


def test_accessible_type_description_is_distinct_from_unchecked_completion():
    state = TaskState()
    state.observe(
        ui(
            [
                {
                    "className": "CheckBox",
                    "contentDescription": "urgent: Item",
                    "checked": False,
                }
            ]
        )
    )
    contract = task("Is Item urgent?")
    contract["requirements"][0]["attribute"] = {"kind": "type", "label": "urgent"}
    with pytest.raises(ContractError, match="query_property_mismatch"):
        parse_grounded_review(
            review(["O1:n0"], answer="False"), contract, state.evidence(), "O1"
        )
    assert (
        parse_grounded_review(
            review(["O1:n0"], answer="True"), contract, state.evidence(), "O1"
        )["status"]
        == "done"
    )


def test_markdown_closing_delimiter_can_be_removed_but_extra_objects_are_rejected():
    state = TaskState()
    state.observe(ui([{"text": "Property"}]))
    assert (
        parse_grounded_review(review(["O1:n0"]) + "`", task(), state.evidence(), "O1")[
            "status"
        ]
        == "done"
    )
    with pytest.raises(ContractError, match="protocol_error"):
        parse_grounded_review(review(["O1:n0"]) + "{}", task(), state.evidence(), "O1")


def test_image_freshness_uses_numeric_order_after_ten_frames():
    state = TaskState()
    for index in range(11):
        state.observe(ui(), screenshot=str(index).encode())
    with pytest.raises(ContractError, match="evidence_stale"):
        parse_grounded_review(review(["O1:image9"]), task(), state.evidence(), "O1")
    assert (
        parse_grounded_review(review(["O1:image11"]), task(), state.evidence(), "O1")[
            "status"
        ]
        == "done"
    )


def test_verification_cannot_omit_requested_exact_state_or_all_state_witnesses():
    state = TaskState()
    state.observe(ui([{"text": "Saved"}, {"text": "7", "rangeInfo": {"current": 7}}]))
    data = json.loads(review(["O1:n0", "O1:n1"], answer=None))
    data["verification_refs"] = ["O1:n0"]
    with pytest.raises(ContractError, match="evidence_missing") as rejected:
        parse_grounded_review(
            json.dumps(data),
            task("Set value to seven", "state"),
            state.evidence(),
            "O1",
        )
    assert rejected.value.field == "omitted_state_property"
    assert rejected.value.refs == ("O1:n1",)
    data["verification_refs"] = []
    with pytest.raises(ContractError, match="evidence_missing") as rejected:
        parse_grounded_review(
            json.dumps(data),
            task("Set value to seven", "state"),
            state.evidence(),
            "O1",
        )
    assert rejected.value.field == "verification_refs"
    assert rejected.value.refs == ("O1:n0", "O1:n1")


def test_changing_readout_can_use_stable_state_witness_but_exact_request_cannot():
    state = TaskState()
    for value in ("12", "13"):
        state.observe(ui([{"resourceId": "readout", "text": value}, {"text": "Pause"}]))
    data = json.loads(review(["O2:n0", "O2:n1"], answer=None))
    data["verification_refs"] = ["O2:n1"]
    assert (
        parse_grounded_review(
            json.dumps(data),
            task("Keep the timer running", "state"),
            state.evidence(),
            "O2",
        )["status"]
        == "done"
    )
    with pytest.raises(ContractError, match="evidence_missing") as rejected:
        parse_grounded_review(
            json.dumps(data),
            task("Stop the timer at 13", "state"),
            state.evidence(),
            "O2",
        )
    assert rejected.value.field == "omitted_state_property"
    assert rejected.value.refs == ("O2:n0",)


def test_type_hint_matches_generic_object_suffix():
    state = TaskState()
    state.observe(ui([{"contentDescription": "reminder: Meeting", "checked": False}]))
    contract = task("Is Meeting a reminder item?")
    contract["requirements"][0]["attribute"] = {
        "kind": "type",
        "label": "reminder item",
    }
    with pytest.raises(ContractError, match="query_property_mismatch"):
        parse_grounded_review(
            review(["O1:n0"], answer="False"), contract, state.evidence(), "O1"
        )


def test_prepared_visual_actions_route_large_context_to_reasoner_without_dispatch(
    monkeypatch,
):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-secret")
    requests = []
    goal = "Tap the visible control"
    contract = task(goal, "state")

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        content = (
            json.dumps(
                {"requirements": contract["requirements"], "answer_required": False}
            )
            if len(requests) == 1
            else '{"action":"Tap","element":[500,500]}'
        )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = AutoGLMFallback(reasoning_model="reasoner", client=client)
            assert await adapter.prepare_task(goal=goal) is None
            buf = BytesIO()
            Image.new("RGB", (400, 800)).save(buf, format="PNG")
            result = await adapter.decide(
                goal=goal,
                screenshot=buf.getvalue(),
                ui_state=ui(),
                available_tools={"click_at"},
                task_state={"progress_note": "x" * 9000},
            )
            assert result.status == "action" and result.model == "reasoner"

    asyncio.run(run())
    assert requests[-1]["model"] == "reasoner" and "response_format" not in requests[-1]


def test_pure_operation_contract_cannot_require_query_answer():
    goal = "Create a new record"
    contract = task(goal, "state")
    contract["answer_required"] = True
    with pytest.raises(ContractError, match="contract_invalid"):
        parse_task_contract(json.dumps(contract), goal)


def test_reused_editor_id_on_another_object_cannot_trigger_replacement():
    editor = {
        "index": 2,
        "resourceId": "editor",
        "className": "EditText",
        "focused": True,
        "editable": True,
        "inputValue": "",
        "bounds": "0,100,200,500",
    }
    before = ui(
        [
            {
                "index": 1,
                "className": "TextView",
                "resourceId": "title",
                "rawText": "Object A",
            },
            editor,
        ]
    )
    target = asyncio.run(capture_input_target(object(), before))
    assert target is not None and target.page_context
    after = copy.deepcopy(before)
    after.elements[0]["rawText"] = "Object B"
    result = asyncio.run(read_input_value(object(), after, target, "expected value"))
    assert (
        result.status == "unknown" and result.reason == "edited_object_context_changed"
    )


def test_memory_preserves_actual_empty_input_and_hint_flag():
    state = TaskState()
    state.observe(
        ui(
            [
                {
                    "className": "EditText",
                    "inputValue": "",
                    "rawText": "Enter a name",
                    "isShowingHintText": True,
                }
            ]
        )
    )
    node = state.context()["observations"][0]["nodes"][0]
    assert node["inputValue"] == "" and node["isShowingHintText"] is True


def test_focus_and_numeric_animation_do_not_evict_other_pages_from_memory():
    state = TaskState()
    state.observe(ui([{"rawText": "Earlier useful page"}]))
    for index in range(12):
        state.observe(ui([{"rawText": "Current page"}, {"rawText": str(index)}]))
    context = state.context(max_observations=2)
    assert context["observations"][0]["id"] == "O1"
    assert context["current_observation"] == "O13"
