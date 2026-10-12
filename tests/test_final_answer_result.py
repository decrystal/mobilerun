"""Public agent results carry answers separately from completion explanations."""

import asyncio

import pytest
from test_bounded_finalization import _agent, _finalize


@pytest.mark.parametrize("success", [True, False])
def test_public_result_preserves_answer_only_for_success(monkeypatch, success):
    agent = _agent(monkeypatch)
    agent.config.agent.manager.vision = False
    agent.shared_state.final_answer = "14:00"
    result, _ = asyncio.run(
        _finalize(agent, success=success, reason="Verified explanation")
    )
    assert result.reason == "Verified explanation"
    assert result.model_dump()["final_answer"] == ("14:00" if success else None)
