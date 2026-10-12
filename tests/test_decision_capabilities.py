"""Controlled capability removal preserves grounded verification and real input evidence."""

import asyncio
from unittest.mock import Mock

import pytest
from test_collection_query import page
from test_jev_first_runtime import config, runtime
from test_task_evidence import task, ui

from mobilerun.agent.decision.task_state import TaskState
from mobilerun.config_manager.config_manager import JevConfig


@pytest.mark.parametrize("enabled", [True, False])
def test_native_query_memory_can_be_removed_without_discarding_raw_evidence(enabled):
    state = TaskState("Count entries", query_memory=enabled)
    observed = state.observe(ui(page("O1", [(0, "A"), (1, "B")], total=2)["nodes"]))
    assert len(observed["nodes"]) == 5
    assert bool(state.collections.scopes) is enabled
    assert bool(state.context()["collections"]) is enabled
    state.reset("New query")
    assert state.query_memory is enabled and state.collections.scopes == {}


@pytest.mark.parametrize("enabled", [True, False])
def test_semantic_recovery_switch_changes_routing_without_weakening_completion(enabled):
    runner, events, policy, visual, _ = runtime(
        settings=config(semantic_recovery=enabled)
    )
    goal = "Inspect the requested property"
    runner.goal = runner.task_state.goal = goal
    runner.task_state.set_contract(task(goal))
    recover = Mock(return_value=True)
    runner.task_state.progress.request_recovery = recover
    asyncio.run(runner.select(goal, runner.ctx.ui))
    assert ("autoglm" in events) is enabled
    assert ("jev" in events) is not enabled
    if enabled:
        recover.assert_called_once()
        assert (
            visual.requests[0]["task_state"]["ledger"]["contract"]["schema_version"]
            == 2
        )
    else:
        recover.assert_not_called()
        assert "semantic_progress" not in policy.requests[0]["task_state"]["ledger"]
    assert runner.task_state.contract["answer_required"] is True


def test_runtime_propagates_disabled_review_images_to_an_injected_adapter():
    runner, _, _, visual, _ = runtime(settings=config(review_history_images=False))
    asyncio.run(runner._visual(runner.ctx.ui))
    assert visual.requests[0]["review_history_images"] is False


@pytest.mark.parametrize(
    "name",
    ["query_memory", "input_repair", "semantic_recovery", "review_history_images"],
)
def test_capability_flags_require_boolean_values(name):
    with pytest.raises(ValueError, match="must be a boolean"):
        JevConfig(**{name: 1})
    assert getattr(JevConfig(), name) is True
