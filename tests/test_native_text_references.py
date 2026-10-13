"""Captured text survives navigation and stays bound through action validation."""

import asyncio
import copy
import json
from types import SimpleNamespace

import httpx
import pytest
from test_decision_visual_guards import _runtime, _ui
from test_task_evidence import task

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.decision.contracts import ContractError
from mobilerun.agent.decision.native_text import (
    advertised_native_sources,
    native_text_snapshot,
    resolve_native_text,
    validate_native_text_receipt,
)
from mobilerun.agent.decision.task_state import TaskState
from mobilerun.agent.decision.visual_fallback import VisualFallback, parse_visual_decision
from mobilerun.config_manager.config_manager import JevConfig

GOAL = "Copy the captured body into the destination"
BODY = "  保留全部正文、标点和空行。\n\nCaptured original 🌱.\n" * 180


def captured_state():
    state = TaskState(GOAL)
    source_ui = _ui(
        [
            {"index": 1, "rawText": "Record A", "resourceId": "title"},
            {
                "index": 2,
                "rawText": BODY,
                "inputValue": BODY,
                "text": BODY,
                "className": "EditText",
                "resourceId": "body",
            },
        ],
        page="Source",
    )
    source_ui.phone_state["packageName"] = "org.example.source"
    state.observe(source_ui)
    for number in range(6):
        state.observe(
            _ui(
                [{"index": 1, "rawText": f"Other page {number}"}], page=f"Page {number}"
            )
        )
    return state


def destination(value="", *, resource="destination", focused=True):
    result = _ui(
        [
            {
                "index": 1,
                "rawText": value,
                "inputValue": value,
                "text": value,
                "className": "EditText",
                "resourceId": resource,
                "focused": focused,
                "isMultiLine": True,
                "bounds": "0,0,350,200",
            }
        ],
        page="Destination",
    )
    result.phone_state["packageName"] = "org.example.destination"
    return result


def test_old_full_value_survives_compact_context_with_explicit_source_and_omissions():
    state = captured_state()
    before = copy.deepcopy(state.observations)
    ordinary = state.context(max_observations=1, max_chars=2500, retain_proofs=False)
    assert "O1" not in {item["id"] for item in ordinary["observations"]}
    snapshot = state.native_text_snapshot()
    assert "O1:n1" in snapshot["private"]
    assert BODY not in json.dumps(snapshot["index"], ensure_ascii=False)
    source = snapshot["private"]["O1:n1"]
    assert source["node"]["inputValue"] == BODY
    assert source["metadata"]["page_labels"] == [
        {"ref": "O1:n0", "rawText": "Record A"}
    ]
    bounded = native_text_snapshot(state.observations, GOAL, max_sources=1)
    assert len(bounded["private"]) == 1 and bounded["index"]["sources_omitted"] > 0
    assert state.observations == before


@pytest.mark.parametrize("structured", [True, False])
@pytest.mark.parametrize("field", ["rawText", "inputValue"])
def test_exact_source_action_preserves_unicode_whitespace_and_one_normal_type(
    structured, field
):
    snapshot = captured_state().native_text_snapshot()
    spec = {"ref": "O1:n1", "field": field}
    content = (
        json.dumps({"action": "Type", "text_source": spec})
        if structured
        else 'do(action="Type", text_source=' + repr(spec) + ")"
    )
    result = parse_visual_decision(
        content,
        ui_state=destination(),
        screenshot_size=(400, 800),
        available_tools={"type"},
        structured=structured,
        native_text_sources=snapshot["private"],
        goal=GOAL,
    )
    assert result.status == "action" and len(result.calls) == 1
    assert result.calls[0].arguments == {"text": BODY, "clear": True}
    assert (
        result.input_source["ref"] == "O1:n1" and result.input_source["field"] == field
    )
    assert result.input_source["value_chars"] == len(BODY)


@pytest.mark.parametrize(
    "spec",
    [
        {"ref": "O99:n0", "field": "rawText"},
        {"ref": "O1:n1", "field": "text"},
        {"ref": "O1:image1", "field": "rawText"},
        {"ref": "O1:n1", "field": "hint"},
        {"ref": "O1:n1", "field": "rawText", "span": [0, 2]},
        {"ref": "C:/private.txt", "field": "rawText"},
        {"ref": "O1:n1", "field": ["rawText"]},
        {"ref": 1, "field": "rawText"},
    ],
)
def test_invented_fields_refs_transformations_and_paths_cannot_be_input_sources(spec):
    snapshot = captured_state().native_text_snapshot()
    with pytest.raises(ContractError):
        resolve_native_text(spec, snapshot["private"], GOAL)
    result = parse_visual_decision(
        json.dumps({"action": "Type", "text_source": spec}),
        ui_state=destination(),
        screenshot_size=(400, 800),
        available_tools={"type"},
        structured=True,
        native_text_sources=snapshot["private"],
        goal=GOAL,
    )
    assert result.status == "error" and result.calls == ()


@pytest.mark.parametrize("extra", [{"text": "override"}, {"clear": False}])
def test_model_cannot_override_resolved_value_or_device_parameters(extra):
    snapshot = captured_state().native_text_snapshot()
    result = parse_visual_decision(
        json.dumps(
            {
                "action": "Type",
                "text_source": {"ref": "O1:n1", "field": "rawText"},
                **extra,
            }
        ),
        ui_state=destination(),
        screenshot_size=(400, 800),
        available_tools={"type"},
        structured=True,
        native_text_sources=snapshot["private"],
        goal=GOAL,
    )
    assert result.status == "error" and result.calls == ()


def test_unknown_goal_unadvertised_source_and_disabled_protocol_are_rejected():
    snapshot = captured_state().native_text_snapshot()
    with pytest.raises(ContractError):
        resolve_native_text(
            {"ref": "O1:n1", "field": "rawText"}, snapshot["private"], "Another goal"
        )
    metadata = copy.deepcopy(snapshot["index"])
    metadata["sources"][0]["preview"] = "different source description"
    with pytest.raises(ContractError):
        advertised_native_sources(metadata, snapshot["private"], GOAL)
    result = parse_visual_decision(
        '{"action":"Type","text_source":{"ref":"O1:n1","field":"rawText"}}',
        ui_state=destination(),
        screenshot_size=(400, 800),
        available_tools={"type"},
        structured=True,
    )
    assert result.status == "error" and result.calls == ()


@pytest.mark.parametrize(
    "flags",
    [
        {"password": True},
        {"isPassword": True},
        {"isShowingHintText": True},
        {"visible": False},
        {"inputType": 129},
        {"inputType": 145},
        {"inputType": 225},
        {"inputType": 18},
        {"inputType": "textPassword"},
    ],
)
def test_password_hint_and_hidden_fields_are_never_advertised(flags):
    observation = {
        "id": "O1",
        "source": "accessibility",
        "nodes": [
            {"ref": "O1:n0", "rawText": "sensitive value", **flags},
        ],
    }
    snapshot = native_text_snapshot([observation], GOAL)
    assert snapshot["index"]["sources"] == [] and snapshot["private"] == {}


def test_equal_text_in_different_objects_preserves_two_distinct_sources():
    observation = {
        "id": "O1",
        "source": "accessibility",
        "nodes": [
            {"ref": "O1:n0", "rawText": "same text", "resourceId": "first"},
            {"ref": "O1:n1", "rawText": "same text", "resourceId": "second"},
        ],
    }
    snapshot = native_text_snapshot([observation], GOAL)
    assert set(snapshot["private"]) == {"O1:n0", "O1:n1"}
    assert snapshot["private"]["O1:n0"]["metadata"]["resourceId"] == "first"
    assert snapshot["private"]["O1:n1"]["metadata"]["resourceId"] == "second"


@pytest.mark.parametrize("changed", [None, "source", "target", "argument", "goal"])
def test_one_http_reference_resolves_old_body_and_preserves_preexecution_guards(
    monkeypatch, changed
):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-key")
    requests = []

    def respond(request):
        payload = json.loads(request.content)
        requests.append(payload)
        prompt = payload["messages"][-1]["content"][0]["text"]
        assert len(prompt) < len(BODY)
        index = json.loads(
            prompt.split("已捕获的原生文字来源（preview不是完整值）：", 1)[1]
        )
        assert set(index) == {"sources", "sources_omitted"}
        assert all("node" not in item for item in index["sources"])
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": '{"action":"Type","text_source":{"ref":"O1:n1","field":"rawText"}}',
                        }
                    }
                ]
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = VisualFallback(
                reasoning_model="reasoner",
                api_key_env="GLM_API_KEY",
                base_url="https://example.invalid/v1",
                client=client,
            )
            adapter._goal, adapter._completion_contract = GOAL, task(GOAL, "process")
            runner, _, provider = _runtime(destination(), visual=adapter)
            runner.registry.tools["type"] = object()
            runner.config.native_text_references = True
            runner.goal = GOAL
            runner.task_state = captured_state()
            runner.task_state.set_contract(adapter._completion_contract)
            runner.feedback = "text_value_required"
            selected = await runner._visual(provider.ui)
            assert selected.input_source["ref"] == "O1:n1"
            assert selected.calls[0].parameters == {"text": BODY, "clear": True}
            if changed == "source":
                runner.task_state.observations[0]["nodes"][1][
                    "rawText"
                ] = "Changed source"
            elif changed == "target":
                provider.ui = destination(resource="another_field")
            elif changed == "argument":
                selected.calls[0].parameters["text"] = "model override"
            elif changed == "goal":
                runner.goal = "Another task"
            current, reason = await runner.validate_before_execute()
            if changed:
                assert reason == (
                    "input_focus_or_value_changed"
                    if changed == "target"
                    else "native_text_source_changed"
                )
                assert runner.history == []
            else:
                assert reason is None
                call = selected.calls[0]
                runner.record_execution(
                    call, ActionResult(True, "captured text dispatched"), current
                )
                action = runner.task_state.actions[-1]
                assert action["input_source"] == selected.input_source
                assert action["effect_status"] == "unknown"
                assert "O1" in {
                    item["id"] for item in runner.task_state.context()["observations"]
                }
                provider.ui = destination(BODY)
                assert await runner.verify_input(call)
                assert action["effect_status"] == "confirmed"

    asyncio.run(run())
    assert len(requests) == 1
    assert [item["type"] for item in requests[0]["messages"][-1]["content"]] == ["text"]
    assert "text_source" in requests[0]["messages"][0]["content"]


def test_goal_reset_cannot_reuse_an_old_captured_value_or_receipt():
    state = captured_state()
    source = state.native_text_snapshot()["private"]
    value, receipt = resolve_native_text(
        {"ref": "O1:n1", "field": "rawText"}, source, GOAL
    )
    state.reset("Another task")
    state.observe(_ui([{"index": 1, "rawText": "new value"}]))
    assert not validate_native_text_receipt(
        state.observations, state.goal, receipt, value
    )


@pytest.mark.parametrize(
    "response,structured",
    [
        (
            '{"action":"Type","text_source":{"ref":"O1:n1","ref":"O1:n0","field":"rawText"}}',
            True,
        ),
        (
            '{"action":"Type","action":"Back","text_source":{"ref":"O1:n1","field":"rawText"}}',
            True,
        ),
        (
            'do(action="Type", text_source={"ref":"O1:n1","ref":"O1:n0","field":"rawText"})',
            False,
        ),
        (
            'do(action="Type", text_source={"ref":str("O1:n1"),"field":"rawText"})',
            False,
        ),
    ],
)
def test_ambiguous_or_executable_reference_syntax_is_rejected(response, structured):
    result = parse_visual_decision(
        response,
        ui_state=destination(),
        screenshot_size=(400, 800),
        available_tools={"type"},
        structured=structured,
        native_text_sources=captured_state().native_text_snapshot()["private"],
        goal=GOAL,
    )
    assert result.status == "error" and result.calls == ()


@pytest.mark.parametrize("enabled,capable", [(False, True), (True, False)])
def test_disabled_or_unsupported_adapter_keeps_original_kwargs_and_jev_context(
    enabled, capable
):
    from mobilerun.agent.decision.visual_fallback import VisualDecision, VisualToolCall

    class Adapter:
        supports_native_text_references = capable
        kwargs = None

        async def decide(self, **kwargs):
            self.kwargs = kwargs
            return VisualDecision(
                "action", (VisualToolCall("type", {"text": "literal", "clear": True}),)
            )

    adapter = Adapter()
    runner, _, provider = _runtime(destination(), visual=adapter)
    runner.registry.tools["type"] = object()
    runner.config.native_text_references = enabled
    runner.goal, runner.task_state = GOAL, captured_state()
    runner.task_state.set_contract(task(GOAL, "process"))
    runner.feedback = "text_value_required"
    asyncio.run(runner._visual(provider.ui))
    assert "native_text_sources" not in adapter.kwargs
    assert "native_text_sources" not in adapter.kwargs["task_state"]
    assert "native_text_sources" not in runner.task_context(purpose="jev")


@pytest.mark.parametrize("value", [None, 1, "true"])
def test_reference_capability_configuration_requires_boolean(value):
    with pytest.raises(ValueError, match="native_text_references"):
        JevConfig(native_text_references=value)


@pytest.mark.parametrize(
    "changed",
    [
        None,
        "source_before_activation",
        "source_after_activation",
        "source_before_type",
        "argument",
        "target",
    ],
)
def test_reference_survives_explicit_activation_and_only_one_bound_input_repair(
    monkeypatch, changed
):
    from mobilerun.agent.decision.visual_fallback import VisualDecision

    class Adapter:
        supports_native_text_references = True

        async def decide(self, **kwargs):
            return parse_visual_decision(
                '{"action":"Type","text_source":{"ref":"O1:n1","field":"rawText"}}',
                ui_state=kwargs["ui_state"],
                screenshot_size=(400, 800),
                available_tools={"type"},
                structured=True,
                native_text_sources=kwargs["native_text_sources"],
                goal=kwargs["goal"],
            )

    async def expire_input_readback(runner, call):
        ticks = iter([0, 4])
        with monkeypatch.context() as local:
            local.setattr(
                "mobilerun.agent.decision.runtime.time",
                SimpleNamespace(monotonic=lambda: next(ticks)),
            )
            assert not await runner.verify_input(call)

    async def run():
        runner, _, provider = _runtime(destination(), visual=Adapter())
        runner.registry.tools["type"] = object()
        runner.config.native_text_references = True
        runner.goal, runner.task_state = GOAL, captured_state()
        runner.task_state.set_contract(task(GOAL, "process"))
        runner.feedback = "text_value_required"
        selected = await runner._visual(provider.ui)
        assert not isinstance(selected, VisualDecision) and selected.input_source
        current, reason = await runner.validate_before_execute()
        assert reason is None
        original = selected.calls[0]
        receipt = copy.deepcopy(selected.input_source)
        runner.record_execution(
            original, ActionResult(True, "Type dispatched"), current
        )
        await expire_input_readback(runner, original)
        assert runner.input_retry_source == receipt
        assert runner.task_state.actions[-1]["effect_status"] == "mismatch"
        if changed == "source_before_activation":
            runner.task_state.observations[0]["nodes"][1]["rawText"] = "Changed source"
        activation = await runner.select(GOAL, provider.ui)
        if changed == "source_before_activation":
            assert activation.stop and activation.calls == []
            assert len(runner.history) == 1 and runner.input_retry_source is None
            return
        assert (
            activation.source == "input_activation"
            and activation.calls[0].name == "click"
        )
        assert activation.input_source is None and runner.input_retry_source == receipt
        current, reason = await runner.validate_before_execute()
        assert reason is None
        runner.record_execution(
            activation.calls[0], ActionResult(True, "Editor activated"), current
        )
        if changed == "source_after_activation":
            runner.task_state.observations[0]["nodes"][1]["rawText"] = "Changed source"
        replacement = await runner.select(GOAL, provider.ui)
        if changed == "source_after_activation":
            assert replacement.stop and replacement.calls == []
            assert len(runner.history) == 2 and runner.input_retry_source is None
            return
        assert (
            replacement.source == "input_recovery"
            and replacement.input_source == receipt
        )
        assert replacement.calls[0].parameters == original.parameters
        if changed == "source_before_type":
            runner.task_state.observations[0]["nodes"][1]["rawText"] = "Changed source"
        elif changed == "argument":
            replacement.calls[0].parameters["text"] = "modified replacement"
        elif changed == "target":
            provider.ui = destination(resource="another_field")
        current, reason = await runner.validate_before_execute()
        if changed:
            assert reason == (
                "input_focus_or_value_changed"
                if changed == "target"
                else "native_text_source_changed"
            )
            assert len(runner.history) == 2
            return
        assert reason is None
        runner.record_execution(
            replacement.calls[0], ActionResult(True, "Replacement dispatched"), current
        )
        action = runner.task_state.actions[-1]
        assert (
            action["input_source"] == receipt and action["effect_status"] == "unknown"
        )
        await expire_input_readback(runner, replacement.calls[0])
        assert runner.input_retry is None and action["effect_status"] == "mismatch"
        provider.ui = destination(BODY)
        assert await runner.verify_input(replacement.calls[0])
        assert action["effect_status"] == "confirmed"
        assert [item["tool"] for item in runner.history] == ["type", "click", "type"]
        assert all(
            item["input_source"] == receipt
            for item in runner.history
            if item["tool"] == "type"
        )

    asyncio.run(run())


@pytest.mark.parametrize(
    "index",
    [
        None,
        [],
        {"sources": None, "sources_omitted": 0},
        {"sources": [None], "sources_omitted": 0},
        {"sources": [{"ref": []}], "sources_omitted": 0},
        {"sources": [], "sources_omitted": True},
        {"sources": [], "sources_omitted": -1},
    ],
)
def test_malformed_source_index_is_rejected_before_http(monkeypatch, index):
    from test_decision_visual_guards import _image

    monkeypatch.setenv("GLM_API_KEY", "offline-test-key")
    requests = []

    def respond(request):
        requests.append(request)
        raise AssertionError("Invalid source context must not reach HTTP")

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = VisualFallback(
                reasoning_model="reasoner", api_key_env="GLM_API_KEY", client=client
            )
            adapter._goal, adapter._completion_contract = GOAL, task(GOAL, "process")
            result = await adapter.decide(
                goal=GOAL,
                screenshot=_image("red"),
                ui_state=destination(),
                available_tools={"type"},
                task_state={"native_text_sources": index},
                native_text_sources=captured_state().native_text_snapshot()["private"],
            )
            assert (
                result.status == "error"
                and result.error_code == "text_source_invalid"
                and result.calls == ()
            )

    asyncio.run(run())
    assert requests == []
