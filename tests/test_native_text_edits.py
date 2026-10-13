"""Prefix/suffix edits preserve complete input values and existing runtime guards."""

import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from test_decision_visual_guards import _runtime
from test_native_text_references import BODY, destination
from test_task_evidence import task

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.decision.native_edit import resolve_native_text_edit
from mobilerun.agent.decision.native_text import (
    text_digest,
    validate_native_text_receipt,
)
from mobilerun.agent.decision.task_state import TaskState
from mobilerun.agent.decision.visual_fallback import VisualFallback, parse_visual_decision
from mobilerun.config_manager.config_manager import JevConfig

GOAL = "Preserve the existing body and add the requested text"
ADDED = "\n\n 新增段落 🌿 \n"


def state_and_page(body=BODY):
    page = destination(body)
    state = TaskState(GOAL)
    state.observe(page)
    return state, page


def action(position="append", **extra):
    return {
        "action": "Type",
        "text_edit": {
            "ref": "O1:n0",
            "field": "inputValue",
            "position": position,
            "text": ADDED,
        },
        **extra,
    }


def parse(
    value, *, state=None, page=None, structured=True, enabled=True, current_id=None
):
    default_state, default_page = state_and_page()
    state, page = state or default_state, page or default_page
    content = (
        json.dumps(value)
        if structured
        else 'do(action="Type", text_edit=' + repr(value["text_edit"]) + ")"
    )
    return parse_visual_decision(
        content,
        ui_state=page,
        screenshot_size=(400, 800),
        available_tools={"type"},
        structured=structured,
        native_text_sources=state.native_text_snapshot()["private"],
        native_text_edits=enabled,
        current_observation_id=state.current_id if current_id is None else current_id,
        goal=GOAL,
    )


@pytest.mark.parametrize("position", ["append", "prepend"])
@pytest.mark.parametrize("structured", [True, False])
def test_exact_before_value_and_added_whitespace_are_combined_as_one_normal_type(
    position, structured
):
    state, page = state_and_page()
    result = parse(action(position), state=state, page=page, structured=structured)
    expected = BODY + ADDED if position == "append" else ADDED + BODY
    assert result.status == "action" and len(result.calls) == 1
    assert result.calls[0].arguments == {"text": expected, "clear": True}
    assert result.input_source["sha256"] == text_digest(BODY)
    assert result.input_source["result_sha256"] == text_digest(expected)
    assert BODY not in json.dumps(result.input_source, ensure_ascii=False)
    assert validate_native_text_receipt(
        state.observations, GOAL, result.input_source, expected
    )
    assert not validate_native_text_receipt(
        state.observations, GOAL, result.input_source, ADDED
    )


@pytest.mark.parametrize(
    "change",
    [
        "old_page",
        "different_package",
        "different_field",
        "blurred",
        "changed_value",
        "unknown_value",
    ],
)
def test_a_captured_body_cannot_authorize_editing_a_different_or_unknown_current_input(
    change,
):
    state, page = state_and_page()
    current_id = state.current_id
    if change == "old_page":
        current_id = "O2"
    elif change == "different_package":
        page.phone_state["packageName"] = "org.example.other"
    elif change == "different_field":
        page.elements[0]["resourceId"] = "another_field"
    elif change == "blurred":
        page.elements[0]["focused"] = False
    elif change == "changed_value":
        page.elements[0]["inputValue"] = "Changed contents"
    else:
        del page.elements[0]["inputValue"]
    result = parse(action(), state=state, page=page, current_id=current_id)
    assert result.status == "error" and result.calls == ()


@pytest.mark.parametrize(
    "change",
    [
        "rawText",
        "replace",
        "position_array",
        "empty_addition",
        "span",
        "text_override",
        "source_override",
        "disabled",
        "numeric_enabled",
    ],
)
def test_unsupported_or_ambiguous_edit_requests_have_no_action(change):
    value = action()
    enabled = True
    if change == "rawText":
        value["text_edit"]["field"] = "rawText"
    elif change == "replace":
        value["text_edit"]["position"] = "replace"
    elif change == "position_array":
        value["text_edit"]["position"] = ["append"]
    elif change == "empty_addition":
        value["text_edit"]["text"] = ""
    elif change == "span":
        value["text_edit"]["span"] = [0, 1]
    elif change == "text_override":
        value["text"] = "override"
    elif change == "source_override":
        value["text_source"] = {"ref": "O1:n0", "field": "inputValue"}
    elif change == "disabled":
        enabled = False
    else:
        enabled = 1
    result = parse(value, enabled=enabled)
    assert result.status == "error" and result.calls == ()


@pytest.mark.parametrize(
    "content",
    [
        '{"action":"Type","text_edit":{"ref":"O1:n0","field":"inputValue","position":"append","position":"prepend","text":"x"}}',
        'do(action="Type", text_edit={"ref":"O1:n0","field":"inputValue","position":"append","text":str("x")})',
        'do(action="Type", text_edit={"ref":"O1:n0","field":"inputValue","position":"append","text":"x","text":"y"})',
    ],
)
def test_duplicate_keys_and_executable_phone_edit_values_are_refused(content):
    state, page = state_and_page()
    result = parse_visual_decision(
        content,
        ui_state=page,
        screenshot_size=(400, 800),
        available_tools={"type"},
        structured=content.startswith("{"),
        native_text_sources=state.native_text_snapshot()["private"],
        native_text_edits=True,
        current_observation_id=state.current_id,
        goal=GOAL,
    )
    assert result.status == "error" and result.calls == ()


@pytest.mark.parametrize(
    "body,added,limit", [("x" * 16_384, "y", None), ("old", "🌱", 4)]
)
def test_full_result_limits_are_checked_without_clipping_original_or_addition(
    body, added, limit
):
    state, page = state_and_page(body)
    if limit is not None:
        page.elements[0]["maxTextLength"] = limit
        state = TaskState(GOAL)
        state.observe(page)
    value = action()
    value["text_edit"]["text"] = added
    result = parse(value, state=state, page=page)
    assert result.status == "error" and result.calls == ()
    assert state.observations[0]["nodes"][0]["inputValue"] == body


@pytest.mark.parametrize(
    "changed",
    [
        None,
        "source",
        "target",
        "argument",
        "goal",
        "position",
        "receipt_target",
        "receipt_target_index",
    ],
)
def test_one_http_edit_binds_the_private_source_index_and_current_preexecution_guards(
    monkeypatch, changed
):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-key")
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        assert "text_edit" in payload["messages"][0]["content"]
        variants = (
            BODY,
            json.dumps(BODY, ensure_ascii=False)[1:-1],
            json.dumps(BODY, ensure_ascii=True)[1:-1],
        )

        def inspect(value):
            if isinstance(value, str):
                assert all(body not in value for body in variants)
            elif isinstance(value, dict):
                for item in value.values():
                    inspect(item)
            elif isinstance(value, list):
                for item in value:
                    inspect(item)

        # The explicit source index stays bounded. Current native controls may
        # still contain the visible value needed for semantic content decisions.
        prompt = payload["messages"][-1]["content"][0]["text"]
        index = json.loads(
            prompt.split("已捕获的原生文字来源（preview不是完整值）：", 1)[1]
        )
        inspect(index)
        assert all("node" not in item for item in index["sources"])
        return httpx.Response(
            200,
            json={"choices": [{"message": {"content": json.dumps(action("prepend"))}}]},
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = VisualFallback(
                reasoning_model="reasoner", api_key_env="GLM_API_KEY", client=client
            )
            adapter._goal, adapter._completion_contract = GOAL, task(GOAL, "process")
            state, page = state_and_page()
            state.set_contract(adapter._completion_contract)
            runner, _, provider = _runtime(page, visual=adapter)
            runner.registry.tools["type"] = object()
            runner.config.native_text_references = runner.config.native_text_edits = (
                True
            )
            runner.goal, runner.task_state, runner.feedback = (
                GOAL,
                state,
                "text_value_required",
            )
            selected = await runner._visual(page)
            assert selected.input_source["kind"] == "native_text_edit"
            expected = ADDED + BODY
            assert selected.calls[0].parameters == {"text": expected, "clear": True}
            if changed == "source":
                state.observations[0]["nodes"][0]["inputValue"] = "Changed original"
            elif changed == "target":
                provider.ui = destination(BODY, resource="another_field")
            elif changed == "argument":
                selected.calls[0].parameters["text"] = ADDED
            elif changed == "goal":
                runner.goal = "Another task"
            elif changed == "position":
                selected.input_source["position"] = "append"
            elif changed == "receipt_target":
                selected.input_source["target"]["resourceId"] = "another_field"
            elif changed == "receipt_target_index":
                selected.input_source["target"]["index"] = True
            current, reason = await runner.validate_before_execute()
            if changed:
                assert reason == (
                    "input_focus_or_value_changed"
                    if changed == "target"
                    else "native_text_source_changed"
                )
                assert runner.history == []
                return
            assert reason is None
            runner.record_execution(
                selected.calls[0],
                ActionResult(True, "Combined input dispatched"),
                current,
            )
            saved = state.actions[-1]
            assert (
                saved["effect_status"] == "unknown"
                and saved["input_source"] == selected.input_source
            )
            provider.ui = destination(expected)
            assert await runner.verify_input(selected.calls[0])
            assert saved["effect_status"] == "confirmed"

    asyncio.run(run())
    assert len(requests) == 1


def test_edit_repair_replaces_the_same_full_result_once_without_appending_twice(
    monkeypatch,
):
    class Adapter:
        supports_native_text_references = supports_native_text_edits = True

        async def decide(self, **kwargs):
            value, receipt = resolve_native_text_edit(
                action()["text_edit"],
                kwargs["native_text_sources"],
                GOAL,
                kwargs["ui_state"],
                kwargs["task_state"]["ledger"]["current_observation"],
            )
            from mobilerun.agent.decision.visual_fallback import (
                VisualDecision,
                VisualToolCall,
            )

            return VisualDecision(
                "action",
                (VisualToolCall("type", {"text": value, "clear": True}),),
                input_source=receipt,
            )

    async def expire(runner, call):
        ticks = iter([0, 4])
        with monkeypatch.context() as local:
            local.setattr(
                "mobilerun.agent.decision.runtime.time",
                SimpleNamespace(monotonic=lambda: next(ticks)),
            )
            assert not await runner.verify_input(call)

    async def run():
        state, page = state_and_page()
        state.set_contract(task(GOAL, "process"))
        runner, _, provider = _runtime(page, visual=Adapter())
        runner.registry.tools["type"] = object()
        runner.config.native_text_references = runner.config.native_text_edits = True
        runner.goal, runner.task_state, runner.feedback = (
            GOAL,
            state,
            "text_value_required",
        )
        selected = await runner._visual(page)
        current, reason = await runner.validate_before_execute()
        assert reason is None
        original = selected.calls[0]
        receipt = copy.deepcopy(selected.input_source)
        runner.record_execution(
            original, ActionResult(True, "Input dispatched"), current
        )
        await expire(runner, original)
        activation = await runner.select(GOAL, page)
        assert activation.source == "input_activation"
        runner.record_execution(
            activation.calls[0], ActionResult(True, "Activated editor"), page
        )
        replacement = await runner.select(GOAL, page)
        assert (
            replacement.input_source == receipt
            and replacement.calls[0].parameters == original.parameters
        )
        current, reason = await runner.validate_before_execute()
        assert reason is None
        runner.record_execution(
            replacement.calls[0], ActionResult(True, "Replacement dispatched"), current
        )
        await expire(runner, replacement.calls[0])
        assert runner.input_retry is None
        provider.ui = destination(BODY + ADDED)
        assert await runner.verify_input(replacement.calls[0])
        assert state.actions[-1]["effect_status"] == "confirmed"
        assert state.actions[-1]["arguments"]["text"].count(ADDED) == 1

    asyncio.run(run())


@pytest.mark.parametrize("enabled,capable", [(False, True), (True, False)])
def test_old_or_disabled_adapter_receives_only_the_existing_source_capability(
    enabled, capable
):
    from mobilerun.agent.decision.visual_fallback import VisualDecision, VisualToolCall

    class Adapter:
        supports_native_text_references = True
        supports_native_text_edits = capable
        kwargs = None

        async def decide(self, **kwargs):
            self.kwargs = kwargs
            return VisualDecision(
                "action", (VisualToolCall("type", {"text": "literal", "clear": True}),)
            )

    state, page = state_and_page()
    state.set_contract(task(GOAL, "process"))
    adapter = Adapter()
    runner, _, _ = _runtime(page, visual=adapter)
    runner.registry.tools["type"] = object()
    runner.config.native_text_references, runner.config.native_text_edits = (
        True,
        enabled,
    )
    runner.goal, runner.task_state, runner.feedback = GOAL, state, "text_value_required"
    asyncio.run(runner._visual(page))
    assert "native_text_sources" in adapter.kwargs
    assert "native_text_edits" not in adapter.kwargs
    assert "native_text_sources" not in runner.task_context(purpose="jev")


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_native_edit_configuration_requires_a_boolean(value):
    with pytest.raises(ValueError, match="native_text_edits"):
        JevConfig(native_text_edits=value)
