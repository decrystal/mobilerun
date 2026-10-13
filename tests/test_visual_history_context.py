"""Empty-native visual history stays usable within existing context/image limits."""

import asyncio
import copy
import json

import httpx
import pytest
from test_decision_visual_guards import _image, _runtime
from test_task_evidence import ui
from test_visual_query_scope import contract, response

from mobilerun.agent.decision.task_state import TaskState
from mobilerun.agent.decision.visual_fallback import VisualFallback
from mobilerun.config_manager.config_manager import JevConfig

GOAL = "What balance was shown for the Travel account in the earlier captured screen?"
COLORS = ("red", "blue", "green", "yellow", "purple", "orange")


def frame_state(colors=COLORS[:3], *, reference="observed", enabled=True):
    state = TaskState(GOAL, visual_frame_observations=enabled)
    task = contract(reference)
    item = task["requirements"][0]
    item.update(description=GOAL, quote=GOAL)
    item["query_scope"]["quote"] = GOAL
    state.set_contract(task)
    for color in colors:
        state.observe(ui(), screenshot=_image(color))
    return state


def observation_ids(context):
    return [item["id"] for item in context["observations"]]


@pytest.mark.parametrize("count", [3, 4])
def test_distinct_visual_frames_survive_the_existing_recent_page_window(count):
    state = frame_state(COLORS[:count])
    context = state.context()
    assert observation_ids(context) == [f"O{i}" for i in range(1, count + 1)]
    assert context["observations_omitted"] == 0
    assert all(item["nodes"] == [] for item in context["observations"])


@pytest.mark.parametrize("retain_proofs", [False, True])
def test_distinct_frame_window_is_bounded_and_omissions_remain_explicit(retain_proofs):
    state = frame_state()
    for color in COLORS[3:]:
        state.observe(ui(), screenshot=_image(color))
    context = state.context(max_observations=2, retain_proofs=retain_proofs)
    assert observation_ids(context) == ["O5", "O6"]
    assert context["observations_omitted"] == 4
    assert len(state.observations) == 6 and "O1:image1" in state.evidence()


@pytest.mark.parametrize(
    "retain_proofs,expected", [(True, ["O1", "O3"]), (False, ["O3"])]
)
def test_returned_identical_pixels_share_a_frame_group_without_merging_capture_refs(
    retain_proofs, expected
):
    state = frame_state(("red", "blue", "red"))
    context = state.context(max_observations=1, retain_proofs=retain_proofs)
    assert observation_ids(context) == expected
    assert state.current_id == "O3" and len(state.observations) == 3


def test_current_only_action_context_keeps_one_actual_frame():
    state = frame_state()
    context = state.context(
        max_observations=1, max_chars=2500, max_actions=4, retain_proofs=False
    )
    assert observation_ids(context) == ["O3"] and context["observations_omitted"] == 2


def test_same_pixels_in_different_packages_do_not_collapse_their_groups():
    state = frame_state(("red",))
    state.observe(ui(package="com.example.other"), screenshot=_image("red"))
    context = state.context(max_observations=1)
    assert observation_ids(context) == ["O2"] and context["observations_omitted"] == 1


def test_native_numeric_updates_keep_the_original_two_capture_group():
    state = frame_state(())
    for number, color in enumerate(COLORS[:3]):
        state.observe(
            ui([{"index": 1, "resourceId": "amount", "rawText": str(number)}]),
            screenshot=_image(color),
        )
    assert observation_ids(state.context()) == ["O2", "O3"]
    assert len(state.observations) == 3 and all(
        item["nodes"] for item in state.observations
    )


def test_node_only_empty_observations_do_not_fabricate_visual_groups():
    state = frame_state(())
    for activity in ("First", "Second", "Third"):
        page = ui()
        page.phone_state["currentApp"] = activity
        state.observe(page)
    assert observation_ids(state.context()) == ["O2", "O3"]
    assert all("images" not in item for item in state.observations)


def test_legacy_mode_keeps_one_observation_with_its_original_image_sequence():
    state = frame_state(enabled=False)
    context = state.context()
    assert observation_ids(context) == ["O1"]
    assert [item["ref"] for item in context["observations"][0]["images"]] == [
        "O1:image1",
        "O1:image2",
        "O1:image3",
    ]


def test_context_is_a_copy_and_does_not_destroy_omitted_local_pixels():
    state = frame_state()
    before = copy.deepcopy(state.observations)
    context = state.context(max_observations=1)
    context["observations"][0]["images"][0]["sha256"] = "changed-model-view"
    assert state.observations == before and "O1:image1" in state.evidence()


def test_existing_accepted_proof_can_pin_a_frame_outside_the_recent_window():
    state = frame_state()
    state.assessments[1] = {
        "id": 1,
        "status": "supported",
        "evidence_refs": ["O1:image1"],
    }
    assert observation_ids(state.context(max_observations=1)) == ["O1", "O3"]
    assert observation_ids(state.context(max_observations=1, retain_proofs=False)) == [
        "O3"
    ]


@pytest.mark.parametrize(
    "mode",
    [
        "success",
        "missing_cache",
        "mismatched_cache",
        "four_frames",
        "current_scope",
        "changed_after_review",
        "historical_after_navigation",
    ],
)
def test_actual_runtime_review_obeys_visible_history_exact_uploads_and_original_guards(
    monkeypatch, mode
):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-secret")
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        context = json.loads(payload["messages"][-1]["content"][0]["text"])
        requests.append((payload, context))
        result = response()
        if mode == "changed_after_review":
            result["assessments"][0]["evidence_refs"].append("O3:image1")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(result)}}]}
        )

    async def run():
        count = 4 if mode == "four_frames" else 3
        state = frame_state(
            (), reference="current" if mode == "current_scope" else "observed"
        )
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
            for number, color in enumerate(COLORS[:count], 1):
                state.observe(ui(), screenshot=_image(color))
                if number < count and not (mode == "missing_cache" and number == 1):
                    adapter._cache_review_image(
                        state.evidence(), state.current_id, _image(color), "image/png"
                    )
            if mode == "mismatched_cache":
                _, mime, pixels = adapter._review_images["O1:image1"]
                adapter._review_images["O1:image1"] = ("0" * 64, mime, pixels)
            runner, driver, _ = _runtime(ui(), visual=adapter)
            runner.config = JevConfig(fallback="autoglm")
            runner.goal, runner.task_state = GOAL, state
            driver.image = _image(COLORS[count - 1])
            selected = await runner._visual(runner.ctx.ui, completion_review=True)
            allowed = mode in {
                "success",
                "changed_after_review",
                "historical_after_navigation",
            }
            assert selected.stop is not allowed
            if allowed:
                assert (
                    len(requests) == 1
                    and selected.calls[0].parameters["final_answer"] == "247.60 EUR"
                )
                assert bool(selected.completion_image_guard) is (
                    mode == "changed_after_review"
                )
                assert not selected.completion_query_binding_required
                if mode in {"changed_after_review", "historical_after_navigation"}:
                    driver.image = _image("purple")
                _, reason = await runner.validate_before_execute()
                assert (reason is not None) is (mode == "changed_after_review")
            else:
                assert len(requests) == 2 and selected.calls == []
            assert runner.history == [] and not runner.executed

    asyncio.run(run())
    first_payload, first_context = requests[0]
    assert "O1:image1" in first_context["available_evidence_refs"]
    uploaded = [item["ref"] for item in first_context["uploaded_screenshot_evidence"]]
    expected = (
        ["O4:image1", "O3:image1", "O2:image1"]
        if mode == "four_frames"
        else (
            ["O3:image1", "O2:image1"]
            if mode in {"missing_cache", "mismatched_cache"}
            else ["O3:image1", "O2:image1", "O1:image1"]
        )
    )
    assert uploaded == expected
    assert sum(
        part["type"] == "image_url" for part in first_payload["messages"][-1]["content"]
    ) == len(expected)
