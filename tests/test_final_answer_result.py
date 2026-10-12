"""Public agent results carry answers separately from completion explanations."""

import asyncio
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from mobilerun.agent.droid import droid_agent
from mobilerun.agent.droid.events import FinalizeEvent
from mobilerun.agent.droid.state import MobileAgentState


@pytest.mark.parametrize("success", [True, False])
def test_public_result_preserves_answer_only_for_success(monkeypatch, success):
    monkeypatch.setattr(droid_agent, "capture", Mock())
    monkeypatch.setattr(droid_agent, "flush", AsyncMock())
    state = MobileAgentState(final_answer="14:00")
    no_vision = SimpleNamespace(vision=False)
    agent = SimpleNamespace(
        shared_state=state,
        user_id=None,
        output_model=None,
        _stream_screenshots=False,
        mcp_manager=None,
        config=SimpleNamespace(
            agent=SimpleNamespace(
                manager=no_vision, executor=no_vision, fast_agent=no_vision
            ),
            logging=SimpleNamespace(save_trajectory="none"),
        ),
    )
    result = asyncio.run(
        droid_agent.MobileAgent.finalize(
            agent,
            SimpleNamespace(write_event_to_stream=Mock()),
            FinalizeEvent(success=success, reason="Verified explanation"),
        )
    )
    assert result.reason == "Verified explanation"
    assert result.model_dump()["final_answer"] == ("14:00" if success else None)
