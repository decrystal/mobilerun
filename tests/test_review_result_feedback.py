"""Rejected persistence evidence must get precise feedback without extra retries."""

import asyncio
from unittest.mock import AsyncMock

import pytest
from test_decision_visual_guards import _runtime
from test_persistent_result_evidence import assessed, contract, reopened

from mobilerun.agent.decision.contracts import ContractError
from mobilerun.agent.decision.visual_fallback import VisualDecision


@pytest.mark.parametrize("field", ["result_refs", "reopen_trace", "persistent_result"])
def test_existing_bounded_retry_explains_result_and_trace_reference_membership(field):
    async def run():
        error = VisualDecision(
            "error",
            message="Completion reviewer returned invalid evidence.", format_error=True,
            error_code=(
                "effect_unverified"
                if field == "persistent_result"
                else "evidence_unknown"
            ),
            error_field=field,
        )
        visual = type("Visual", (), {"decide": AsyncMock(side_effect=[error, error])})()
        runner, _, provider = _runtime(visual=visual)
        runner.task_state.set_contract(contract())
        selected = await runner._visual(provider.ui, completion_review=True)
        assert visual.decide.await_count == 2
        second = visual.decide.call_args_list[1].kwargs
        assert second["completion_review"] and not second.get("corrective_only")
        feedback = second["review_feedback"]
        assert "result_refs" in feedback and "evidence_refs" in feedback
        assert "each trace reference" in feedback
        assert "returned" in feedback and "before" in feedback
        assert "Do not replay" in feedback
        assert sum(event["event"] == "format_retry" for event in runner.trace) == 1
        assert selected.calls == [] and selected.stop
        assert runner.history == [] and not runner.executed

    asyncio.run(run())


@pytest.mark.parametrize("removed", ["O1", "O2", "O3", "A1", "A2"])
def test_trace_reference_not_in_its_own_assessment_still_fails_strictly(removed):
    state, refs, trace = reopened()
    refs.remove(removed)
    with pytest.raises(ContractError) as error:
        assessed(state, refs, results=["O3:n0"], trace=trace)
    assert (
        error.value.code == "evidence_unknown" and error.value.field == "reopen_trace"
    )


def test_history_editor_is_still_not_a_returned_saved_result_source():
    state, refs, trace = reopened()
    refs.append("O1:n0")
    with pytest.raises(ContractError) as error:
        assessed(state, refs, results=["O1:n0"], trace=trace)
    assert (
        error.value.code == "effect_unverified"
        and error.value.field == "persistent_result"
    )


def test_complete_real_trace_and_returned_field_still_pass_without_auto_repair():
    state, refs, trace = reopened()
    result = assessed(state, refs, results=["O3:n0"], trace=trace)
    assert result["status"] == "done" and result["assessments"][0]["satisfied"]
