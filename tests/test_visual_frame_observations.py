"""Pure screenshot frames keep actual order without changing native snapshots."""

import asyncio
import copy

import pytest
from test_decision_visual_guards import _image, _runtime
from test_task_evidence import ui
from test_visual_query_scope import GOAL, contract, parsed, response

from mobilerun.agent.decision.autoglm import AutoGLMFallback
from mobilerun.agent.decision.effect_order import has_post_action_result
from mobilerun.agent.decision.runtime import DecisionRuntime
from mobilerun.agent.decision.task_state import TaskState
from mobilerun.config_manager.config_manager import JevConfig


def enabled_state(reference="observed"):
    # Also lets the immutable old-source reproduction run before the new
    # constructor flag exists. The old observer ignores this opt-in field.
    state = TaskState(GOAL)
    state.visual_frame_observations = True
    state.set_contract(contract(reference))
    return state


def test_changed_empty_tree_screenshot_gets_a_distinct_actual_observation():
    state = enabled_state()
    first = state.observe(ui(), screenshot=_image("red"))
    second = state.observe(ui(), screenshot=_image("blue"))
    assert first["id"] == "O1" and second["id"] == "O2"
    assert len(state.observations) == 2
    assert first["nodes"] == second["nodes"] == []


def test_visual_page_history_can_be_reuploaded_without_fabricating_a_native_change():
    state = enabled_state()
    state.observe(ui(), screenshot=_image("red"))
    old = state.evidence()["O1:image1"]["sha256"]
    state.observe(ui(), screenshot=_image("blue"))
    result = parsed(state, response(), uploaded={"O1:image1": old})
    assert result["status"] == "done" and result["verification_refs"] == ()


def test_returning_to_a_previous_visual_page_preserves_three_actual_capture_epochs():
    state = enabled_state()
    for color in ("red", "blue", "red"):
        state.observe(ui(), screenshot=_image(color))
    assert [item["id"] for item in state.observations] == ["O1", "O2", "O3"]
    assert (
        state.observations[0]["images"][0]["sha256"]
        == state.observations[2]["images"][0]["sha256"]
    )


def test_consecutive_identical_empty_tree_frames_are_still_deduplicated():
    state = enabled_state()
    state.observe(ui(), screenshot=_image("red"))
    before = copy.deepcopy(state.observations)
    state.observe(ui(), screenshot=_image("red"))
    assert state.observations == before and state.current_id == "O1"


def test_node_only_refresh_does_not_fabricate_a_visual_capture():
    state = enabled_state()
    state.observe(ui(), screenshot=_image("red"))
    before = copy.deepcopy(state.observations)
    state.observe(ui())
    assert state.observations == before
    state.observe(ui(), screenshot=_image("blue"))
    assert state.current_id == "O2"


def test_native_tree_keeps_existing_stable_observation_and_frame_sequence():
    state = enabled_state()
    native = ui([{"index": 1, "resourceId": "amount", "rawText": "247.60"}])
    for color in ("red", "blue", "red"):
        state.observe(native, screenshot=_image(color))
    assert len(state.observations) == 1
    assert [item["ref"] for item in state.observations[0]["images"]] == [
        "O1:image1",
        "O1:image2",
        "O1:image3",
    ]


def test_unavailable_hidden_nodes_do_not_create_native_identity_for_pixel_frames():
    state = enabled_state()
    hidden = ui([{"index": 1, "rawText": "Secret", "isVisibleToUser": False}])
    state.observe(hidden, screenshot=_image("red"))
    state.observe(hidden, screenshot=_image("blue"))
    assert state.current_id == "O2" and all(
        not item["nodes"] for item in state.observations
    )


def test_device_context_change_and_new_image_create_only_one_new_observation():
    state = enabled_state()
    state.observe(ui(), screenshot=_image("red"))
    other = ui(package="com.example.other")
    state.observe(other, screenshot=_image("blue"))
    assert [item["id"] for item in state.observations] == ["O1", "O2"]


def test_explicit_legacy_state_mode_keeps_its_original_frame_numbering():
    state = TaskState(GOAL, visual_frame_observations=False)
    state.observe(ui(), screenshot=_image("red"))
    state.observe(ui(), screenshot=_image("blue"))
    assert state.current_id == "O1" and len(state.observations[0]["images"]) == 2


def test_reset_retains_the_configured_frame_mode_without_retaining_old_images():
    state = TaskState(GOAL, visual_frame_observations=True)
    state.observe(ui(), screenshot=_image("red"))
    state.reset("Another goal")
    assert state.visual_frame_observations is True and state.observations == []
    state.observe(ui(), screenshot=_image("blue"))
    assert state.current_id == "O1"


@pytest.mark.parametrize("switch", [1, "yes", None])
def test_frame_mode_switch_requires_an_actual_boolean(switch):
    with pytest.raises(ValueError):
        TaskState(visual_frame_observations=switch)
    with pytest.raises(ValueError):
        JevConfig(visual_frame_observations=switch)


def test_default_agent_runtime_enables_frame_observations():
    runner = DecisionRuntime(JevConfig(fallback="autoglm"), None, None, None)
    assert runner.config.visual_frame_observations is True
    assert runner.task_state.visual_frame_observations is True
    assert runner.visual_fallback.model == "glm-5.3-flash"


def test_changed_visual_frame_after_a_real_recorded_action_has_a_later_boundary():
    state = enabled_state()
    state.observe(ui(), screenshot=_image("red"))
    # Existing action fields suffice; no new receipt or invented timestamp.
    state.actions.append(
        {
            "ref": "A1",
            "before": "O1",
            "tool": "system_button",
            "dispatch_status": "dispatched",
        }
    )
    state.observe(ui())
    state.observe(ui(), screenshot=_image("blue"))
    evidence = state.evidence()
    assert not has_post_action_result(["A1", "O1:image1"], evidence, state.current_id)
    assert has_post_action_result(["A1", "O2:image1"], evidence, state.current_id)


def test_actual_previous_frame_cache_survives_next_frame_and_uploads_with_its_old_ref(
    monkeypatch,
):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-secret")
    state = enabled_state()
    state.observe(ui(), screenshot=_image("red"))
    adapter = AutoGLMFallback()
    old = adapter._cache_review_image(
        state.evidence(), state.current_id, _image("red"), "image/png"
    )
    state.observe(ui(), screenshot=_image("blue"))
    current = adapter._cache_review_image(
        state.evidence(), state.current_id, _image("blue"), "image/png"
    )
    assert old["ref"] == "O1:image1" and current["ref"] == "O2:image1"
    assert set(adapter._review_images) == {"O1:image1", "O2:image1"}


def test_runtime_node_and_pixel_refreshes_share_one_new_frame_without_extra_captures():
    async def run():
        runner, driver, provider = _runtime(ui())
        runner.goal, runner.task_state = GOAL, enabled_state()
        runner.observe_result(provider.ui)
        runner.task_state.observe(provider.ui, screenshot=await driver.screenshot())
        driver.image = _image("blue")
        runner.observe_result(provider.ui)
        runner.task_state.observe(provider.ui, screenshot=await driver.screenshot())
        assert (
            runner.task_state.current_id == "O2"
            and len(runner.task_state.observations) == 2
        )
        assert runner.history == [] and not runner.executed

    asyncio.run(run())
