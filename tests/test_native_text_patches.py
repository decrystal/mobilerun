"""Anchored edits preserve all other text and retain execution/readback guards."""

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
from mobilerun.agent.decision.native_patch import resolve_native_text_patch
from mobilerun.agent.decision.native_text import (
    text_digest,
    validate_native_text_receipt,
)
from mobilerun.agent.decision.task_state import TaskState
from mobilerun.agent.decision.visual_fallback import (
    VisualDecision,
    VisualFallback,
    VisualToolCall,
    parse_visual_decision,
)
from mobilerun.config_manager.config_manager import JevConfig

GOAL = "Change only the requested fragment and preserve the rest of the original body"
OLD = "唯一旧片段e\u0301 🌿"
NEW = " 新段落🌳\n "
HEAD, TAIL = (
    BODY[: len(BODY) // 2] + "\n  ",
    "\n\n" + BODY[len(BODY) // 2 :] + "  End\n",
)
ORIGINAL, EXPECTED = HEAD + OLD + TAIL, HEAD + NEW + TAIL


def state_and_page(body=ORIGINAL):
    page = destination(body)
    state = TaskState(GOAL)
    state.observe(page)
    return state, page


def action(match=OLD, replacement=NEW, before="", after="", **extra):
    return {
        "action": "Type",
        "text_patch": {
            "ref": "O1:n0",
            "field": "inputValue",
            "before": before,
            "match": match,
            "after": after,
            "replacement": replacement,
        },
        **extra,
    }


def parse(
    value,
    *,
    state=None,
    page=None,
    structured=True,
    enabled=True,
    current_id=None,
    sources=True,
):
    default_state, default_page = state_and_page()
    state, page = state or default_state, page or default_page
    content = (
        json.dumps(value)
        if structured
        else 'do(action="Type", text_patch=' + repr(value["text_patch"]) + ")"
    )
    return parse_visual_decision(
        content,
        ui_state=page,
        screenshot_size=(400, 800),
        available_tools={"type"},
        structured=structured,
        native_text_sources=(
            state.native_text_snapshot()["private"] if sources else None
        ),
        native_text_patches=enabled,
        current_observation_id=current_id or state.current_id,
        goal=GOAL,
    )


@pytest.mark.parametrize("structured", [True, False])
@pytest.mark.parametrize(
    "body,value,expected",
    [
        (ORIGINAL, action(), EXPECTED),
        (ORIGINAL, action(replacement=""), HEAD + TAIL),
        (ORIGINAL, action(match="", before=OLD), HEAD + OLD + NEW + TAIL),
        (
            "one: todo\ntwo: todo\n",
            action(match="todo", before="two: ", after="\n", replacement="done"),
            "one: todo\ntwo: done\n",
        ),
        (
            "a👨‍👩‍👧‍👦e\u0301\n\n z",
            action(match="👨‍👩‍👧‍👦", replacement="🌳"),
            "a🌳e\u0301\n\n z",
        ),
        (
            "header🌱\n body  ",
            action(match="", after="header🌱", replacement="new\n"),
            "new\nheader🌱\n body  ",
        ),
        ("entire", action(match="entire", replacement=""), ""),
    ],
    ids=[
        "replace_long",
        "delete_long",
        "insert_middle",
        "disambiguate",
        "unicode",
        "insert_start",
        "delete_all",
    ],
)
def test_replace_delete_and_insert_at_unique_boundaries_as_one_normal_type(
    structured, body, value, expected
):
    state, page = state_and_page(body)
    result = parse(value, state=state, page=page, structured=structured)
    assert result.status == "action" and len(result.calls) == 1
    assert result.calls[0].arguments == {"text": expected, "clear": True}
    assert result.input_source["sha256"] == text_digest(body)
    assert result.input_source["result_sha256"] == text_digest(expected)
    assert validate_native_text_receipt(
        state.observations, GOAL, result.input_source, expected
    )
    assert not validate_native_text_receipt(
        state.observations, GOAL, result.input_source, body
    )
    assert page.elements[0]["inputValue"] == body


@pytest.mark.parametrize(
    "body,value,field",
    [
        ("banana", action(match="ana", replacement="x"), "anchor_ambiguous"),
        ("ab ab", action(match="ab", replacement="x"), "anchor_ambiguous"),
        ("ab ab", action(match="", after="ab", replacement="x"), "anchor_ambiguous"),
        ("ab", action(match="", replacement="x"), "empty_anchor_or_no_change"),
        ("ab", action(match="ab", replacement="ab"), "empty_anchor_or_no_change"),
        ("ab", action(match="missing", replacement="x"), "anchor_not_found"),
        (
            "beforeabafter",
            action(match="ab", before="before ", after="after", replacement="x"),
            "anchor_not_found",
        ),
        (
            "beforeabafter",
            action(match="ab", before="before", after=" after", replacement="x"),
            "anchor_not_found",
        ),
        ("café", action(match="cafe\u0301", replacement="x"), "anchor_not_found"),
    ],
)
def test_ambiguous_overlapping_missing_or_normalized_anchors_cannot_propose_an_action(
    body, value, field
):
    state, page = state_and_page(body)
    result = parse(value, state=state, page=page)
    assert result.status == "error" and result.calls == ()
    assert result.error_code == "text_patch_invalid" and result.error_field == field


@pytest.mark.parametrize(
    "change",
    [
        "missing_context",
        "extra_offset",
        "batch",
        "numeric_match",
        "null_before",
        "rawText",
        "text_override",
        "source_override",
        "edit_override",
        "disabled",
        "numeric_enabled",
        "no_sources",
    ],
)
def test_patch_is_explicit_typed_and_mutually_exclusive_with_other_input_parameters(
    change,
):
    value, enabled, sources = action(), True, True
    if change == "missing_context":
        value["text_patch"].pop("before")
    elif change == "extra_offset":
        value["text_patch"]["start"] = 1
    elif change == "batch":
        value["text_patch"] = [value["text_patch"]]
    elif change == "numeric_match":
        value["text_patch"]["match"] = 1
    elif change == "null_before":
        value["text_patch"]["before"] = None
    elif change == "rawText":
        value["text_patch"]["field"] = "rawText"
    elif change == "text_override":
        value["text"] = "override"
    elif change == "source_override":
        value["text_source"] = {"ref": "O1:n0", "field": "inputValue"}
    elif change == "edit_override":
        value["text_edit"] = {
            "ref": "O1:n0",
            "field": "inputValue",
            "position": "append",
            "text": "x",
        }
    elif change == "disabled":
        enabled = False
    elif change == "numeric_enabled":
        enabled = 1
    else:
        sources = False
    result = parse(value, enabled=enabled, sources=sources)
    assert result.status == "error" and result.calls == ()


@pytest.mark.parametrize("structured", [True, False])
@pytest.mark.parametrize("change", ["duplicate", "expression", "expansion"])
def test_nested_duplicate_keys_and_executable_literal_values_are_rejected(
    structured, change
):
    state, page = state_and_page()
    spec = action()["text_patch"]
    if structured:
        content = json.dumps(action())
        if change == "duplicate":
            content = content.replace('"before": ""', '"before": "", "before": "x"')
        elif change == "expression":
            content = content.replace('"before": ""', '"before": {}')
        else:
            content = content.replace('"before": ""', '"before": [""]')
    else:
        literal = repr(spec)
        if change == "duplicate":
            literal = literal.replace("'before': ''", "'before': '', 'before': 'x'")
        elif change == "expression":
            literal = literal.replace("'before': ''", "'before': str('')")
        else:
            literal = literal[:-1] + ", **{}}"
        content = 'do(action="Type", text_patch=' + literal + ")"
    result = parse_visual_decision(
        content,
        ui_state=page,
        screenshot_size=(400, 800),
        available_tools={"type"},
        structured=structured,
        native_text_sources=state.native_text_snapshot()["private"],
        native_text_patches=True,
        current_observation_id=state.current_id,
        goal=GOAL,
    )
    assert result.status == "error" and result.calls == ()


@pytest.mark.parametrize(
    "change",
    ["old_observation", "package", "field", "blurred", "body", "unknown_value"],
)
def test_current_full_value_and_input_identity_are_required(change):
    state, page = state_and_page()
    current_id = state.current_id
    if change == "old_observation":
        current_id = "O2"
    elif change == "package":
        page.phone_state["packageName"] = "org.example.other"
    elif change == "field":
        page.elements[0]["resourceId"] = "other"
    elif change == "blurred":
        page.elements[0]["focused"] = False
    elif change == "body":
        page.elements[0]["inputValue"] += "changed"
    else:
        del page.elements[0]["inputValue"]
    result = parse(action(), state=state, page=page, current_id=current_id)
    assert result.status == "error" and result.calls == ()


@pytest.mark.parametrize(
    "body,value,limit",
    [
        ("x" * 16_384, action(match="", before="x" * 16_384, replacement="y"), None),
        ("old", action(match="old", replacement="🌱🌿🌳"), 5),
        ("old", action(match="old", replacement=""), 0),
    ],
)
def test_complete_output_respects_character_and_android_utf16_limits(
    body, value, limit
):
    state, page = state_and_page(body)
    if limit is not None:
        page.elements[0]["maxTextLength"] = limit
        state = TaskState(GOAL)
        state.observe(page)
    result = parse(value, state=state, page=page)
    assert result.status == ("action" if limit == 0 else "error")
    assert state.observations[0]["nodes"][0]["inputValue"] == body


@pytest.mark.parametrize(
    "changed",
    [
        None,
        "source",
        "target",
        "argument",
        "goal",
        "replacement",
        "start",
        "end_bool",
        "target_index_bool",
        "context",
    ],
)
def test_one_mock_http_patch_retains_source_execution_and_readback_guards(
    monkeypatch, changed
):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-key")
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        prompt = payload["messages"][0]["content"]
        assert "text_patch" in prompt and "含重叠匹配" in prompt
        index_text = payload["messages"][-1]["content"][0]["text"].split(
            "已捕获的原生文字来源（preview不是完整值）：", 1
        )[1]
        index = json.loads(index_text)
        assert ORIGINAL not in json.dumps(index, ensure_ascii=False)
        assert all("node" not in item for item in index["sources"])
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(action())}}]}
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
            runner.config.native_text_references = runner.config.native_text_patches = (
                True
            )
            runner.goal, runner.task_state, runner.feedback = (
                GOAL,
                state,
                "text_value_required",
            )
            selected = await runner._visual(page)
            assert selected.calls[0].parameters == {"text": EXPECTED, "clear": True}
            assert selected.input_source["kind"] == "native_text_patch"
            if changed == "source":
                state.observations[0]["nodes"][0]["inputValue"] += "changed"
            elif changed == "target":
                provider.ui = destination(ORIGINAL, resource="another_field")
            elif changed == "argument":
                selected.calls[0].parameters["text"] = NEW
            elif changed == "goal":
                runner.goal = "Another task"
            elif changed == "replacement":
                selected.input_source["replacement"] = "different"
            elif changed == "start":
                selected.input_source["start"] += 1
            elif changed == "end_bool":
                selected.input_source["end"] = True
            elif changed == "target_index_bool":
                selected.input_source["target"]["index"] = True
            elif changed == "context":
                selected.input_source["before"] = "changed"
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
                selected.calls[0], ActionResult(True, "Input dispatched"), current
            )
            saved = state.actions[-1]
            assert (
                saved["effect_status"] == "unknown"
                and saved["input_source"] == selected.input_source
            )
            provider.ui = destination(EXPECTED)
            assert await runner.verify_input(selected.calls[0])
            assert saved["effect_status"] == "confirmed"
            assert state.assessments == {}

    asyncio.run(run())
    assert len(requests) == 1


def test_one_observed_input_mismatch_repair_reuses_full_result_without_reapplying_patch(
    monkeypatch,
):
    class Adapter:
        supports_native_text_references = supports_native_text_patches = True

        async def decide(self, **kwargs):
            text, receipt = resolve_native_text_patch(
                action(match="", before=OLD)["text_patch"],
                kwargs["native_text_sources"],
                GOAL,
                kwargs["ui_state"],
                kwargs["task_state"]["ledger"]["current_observation"],
            )
            return VisualDecision(
                "action",
                (VisualToolCall("type", {"text": text, "clear": True}),),
                input_source=receipt,
            )

    async def run():
        state, page = state_and_page()
        state.set_contract(task(GOAL, "process"))
        runner, _, provider = _runtime(page, visual=Adapter())
        runner.registry.tools["type"] = object()
        runner.config.native_text_references = runner.config.native_text_patches = True
        runner.goal, runner.task_state, runner.feedback = (
            GOAL,
            state,
            "text_value_required",
        )
        selected = await runner._visual(page)
        original, receipt = selected.calls[0], copy.deepcopy(selected.input_source)
        current, reason = await runner.validate_before_execute()
        assert reason is None
        runner.record_execution(original, ActionResult(True, "Dispatched"), current)
        ticks = iter([0, 4])
        with monkeypatch.context() as local:
            local.setattr(
                "mobilerun.agent.decision.runtime.time",
                SimpleNamespace(monotonic=lambda: next(ticks)),
            )
            assert not await runner.verify_input(original)
        activation = await runner.select(GOAL, page)
        assert activation.source == "input_activation"
        runner.record_execution(
            activation.calls[0], ActionResult(True, "Activated editor"), page
        )
        replacement = await runner.select(GOAL, page)
        assert (
            replacement.calls[0].parameters == original.parameters
            and replacement.input_source == receipt
        )
        current, reason = await runner.validate_before_execute()
        assert reason is None
        runner.record_execution(
            replacement.calls[0], ActionResult(True, "Replaced input"), current
        )
        provider.ui = destination(HEAD + OLD + NEW + TAIL)
        assert await runner.verify_input(replacement.calls[0])
        assert state.actions[-1]["effect_status"] == "confirmed"
        assert state.actions[-1]["arguments"]["text"].count(NEW) == 1

    asyncio.run(run())


@pytest.mark.parametrize(
    "enabled,capable,references,schema",
    [
        (False, True, True, 2),
        (True, False, True, 2),
        (True, True, False, 2),
        (True, True, True, 1),
    ],
)
def test_disabled_old_and_incompatible_adapters_do_not_receive_patch_capability(
    enabled, capable, references, schema
):
    class Adapter:
        supports_native_text_references = True
        supports_native_text_patches = capable
        kwargs = None

        async def decide(self, **kwargs):
            self.kwargs = kwargs
            return VisualDecision(
                "action", (VisualToolCall("type", {"text": "literal", "clear": True}),)
            )

    state, page = state_and_page()
    contract = task(GOAL, "process")
    contract["schema_version"] = schema
    state.set_contract(contract)
    adapter = Adapter()
    runner, _, _ = _runtime(page, visual=adapter)
    runner.registry.tools["type"] = object()
    runner.config.native_text_references, runner.config.native_text_patches = (
        references,
        enabled,
    )
    runner.goal, runner.task_state, runner.feedback = GOAL, state, "text_value_required"
    asyncio.run(runner._visual(page))
    assert "native_text_patches" not in adapter.kwargs
    assert "native_text_sources" not in runner.task_context(purpose="jev")


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_patch_configuration_requires_a_boolean(value):
    with pytest.raises(ValueError, match="native_text_patches"):
        JevConfig(native_text_patches=value)
