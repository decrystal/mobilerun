import json

from test_task_evidence import task, ui

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.decision.task_state import TaskState
from mobilerun.agent.fast_agent.xml_parser import ToolCall


def test_animation_does_not_reset_stall_and_recovery_is_bounded():
    state = TaskState("Update the record")
    state.set_contract(task(state.goal, "state"))
    state.observe(ui([{"resourceId": "clock", "rawText": "12"}]))
    for index in range(4):
        state.record_action(
            ToolCall("wait", {"duration": 0.5}), ActionResult(True, "waited")
        )
        state.observe(ui([{"resourceId": "clock", "rawText": str(13 + index)}]))
    assert state.progress.stalled_actions == 4
    assert state.progress.request_recovery(state)
    state.progress.stalled_actions = 4
    assert state.progress.request_recovery(state)
    state.progress.stalled_actions = 4
    assert not state.progress.request_recovery(state)


def test_input_dispatch_is_not_progress_and_duplicate_confirmed_input_is_not_new_progress():
    state = TaskState("Fill the form")
    state.set_contract(task(state.goal, "state"))
    state.observe(ui())
    state.record_action(ToolCall("click", {"index": 1}), ActionResult(True, "clicked"))
    state.observe(ui())
    for attempt in range(2):
        state.record_action(
            ToolCall("type", {"text": "new value", "clear": True}),
            ActionResult(True, "dispatched"),
        )
        state.observe(ui())
        state.record_input_effect("confirmed")
        assert state.progress.stalled_actions == attempt


def test_new_goal_clears_pending_input_recovery_and_previous_evidence():
    import asyncio

    from test_jev_first_runtime import runtime

    runner, *_ = runtime()
    runner.goal = "Old task"
    runner.task_state.reset("Old task")
    runner.task_state.set_contract(task("Old task", "state"))
    runner.input_retry = (ToolCall("type", {"text": "OLD SECRET", "clear": True}), None)
    runner.history.append({"tool": "type", "arguments": {"text": "OLD SECRET"}})
    runner.task_state.observe(ui())
    asyncio.run(runner.select("New task", ui()))
    assert runner.input_retry is None
    assert "OLD SECRET" not in json.dumps(runner.task_context())


def test_only_pure_local_query_errors_have_bounded_recovery():
    from test_jev_first_runtime import runtime

    runner, *_ = runtime()
    failure = ActionResult(False, "Local query rejected (query_scope_stale)")
    assert not runner.recover_local_failure(ToolCall("click", {"index": 1}), failure)
    query = ToolCall("query_collection", {"plan": {}})
    assert runner.recover_local_failure(query, failure)
    assert runner.recover_local_failure(query, failure)
    assert not runner.recover_local_failure(query, failure)


def test_jev_can_request_query_grounding_when_native_scope_is_available():
    from test_jev_decision import run_decision

    decision, requests = run_decision(
        selections={"operation": "ANALYZE_COLLECTION"},
        available_tools={"click", "query_collection"},
        task_state={
            "ledger": {
                "collections": [{"scope": "actual", "generation": 1}],
                "contract": task("Count all records"),
            }
        },
    )
    assert decision.reason == "query_requires_grounding"
    assert "ANALYZE_COLLECTION" in requests[0]["questions"]["operation"]["criteria"]
