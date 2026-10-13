"""Checks at the untrusted model/API and device coordinate boundaries."""

import asyncio
import base64
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image

from mobilerun.agent.decision.jev import control_observations
from mobilerun.agent.decision.visual_fallback import (
    BIGMODEL_CHAT_URL,
    VisualDecision,
    VisualFallback,
    _merge_required_checks,
    action_argument_diagnostic,
    parse_completion_review,
    parse_visual_decision,
)
from mobilerun.agent.utils.signatures import build_tool_registry
from mobilerun.tools.ui.state import UIState

TOOLS = {
    "click_at",
    "swipe",
    "long_press_at",
    "system_button",
    "type",
    "wait",
    "launch_app",
}


@pytest.mark.parametrize("alias", ["progress_message", "progress_text"])
@pytest.mark.parametrize("with_canonical", [False, True])
def test_rejected_progress_fields_explain_schema_without_becoming_actions(
    alias, with_canonical
):
    action = {"action": "Tap", "index": 2, alias: "untrusted note"}
    if with_canonical:
        action["progress"] = "different note"
    decision = parse_visual_decision(
        json.dumps(action),
        ui_state=SimpleNamespace(),
        screenshot_size=(0, 0),
        available_tools={"click"},
        structured=True,
    )
    assert decision.status == "error" and not decision.calls
    assert decision.error_code == "unknown_argument"
    assert action_argument_diagnostic(decision) == {
        "rejected_known_fields": [alias],
        "allowed_arguments": ["index", "message"],
    }
    assert "untrusted note" not in repr(decision)
    assert "different note" not in repr(decision)


@pytest.mark.parametrize("structured", [False, True])
def test_type_target_is_rejected_and_not_repaired_by_argument_diagnostics(structured):
    content = (
        '{"action":"Type","text":"private value","index":2}'
        if structured
        else 'do(action="Type", text="private value", index=2)'
    )
    decision = parse_visual_decision(
        content,
        ui_state=SimpleNamespace(),
        screenshot_size=(0, 0),
        available_tools={"type"},
        structured=structured,
    )
    assert decision.status == "error" and not decision.calls
    assert action_argument_diagnostic(decision) == {
        "rejected_known_fields": ["index"],
        "allowed_arguments": ["text"],
    }
    assert "private value" not in repr(decision)


def test_unknown_fields_and_values_do_not_escape_into_protocol_diagnostics():
    secret = "sk-private-secret"
    decision = parse_visual_decision(
        json.dumps({"action": "Back", secret: {"index": 2, "text": secret}}),
        ui_state=SimpleNamespace(),
        screenshot_size=(0, 0),
        available_tools={"system_button"},
        structured=True,
    )
    assert decision.status == "error" and not decision.calls
    assert action_argument_diagnostic(decision) == {
        "rejected_known_fields": [],
        "allowed_arguments": [],
    }
    assert secret not in repr(decision)


def test_argument_diagnostics_filter_injected_schema_metadata_and_legacy_results():
    decision = VisualDecision(
        "error",
        error_code="unknown_argument",
        error_field="arguments",
        error_argument_fields=(
            "index",
            "index",
            "model instruction",
            {"text": "secret"},
        ),
        error_allowed_arguments=(
            "text",
            "text",
            "progress_message",
            "sk-private",
            None,
        ),
    )
    assert action_argument_diagnostic(decision) == {
        "rejected_known_fields": ["index"],
        "allowed_arguments": ["text"],
    }
    assert action_argument_diagnostic(VisualDecision("error")) is None
    assert (
        action_argument_diagnostic(SimpleNamespace(error_code="unknown_argument"))
        is None
    )


@pytest.mark.parametrize("wrapped", [False, True])
def test_structured_progress_alias_keeps_exact_type_and_raw_response(wrapped):
    action = {
        "action": "Type",
        "text": "Coupon code: OFF50",
        "progress_note": "Input, then copy",
    }
    content = json.dumps([action] if wrapped else action)
    decision = parse_visual_decision(
        content,
        ui_state=_ui(),
        screenshot_size=(1080, 2400),
        available_tools=TOOLS,
        structured=True,
    )
    assert decision.status == "action" and len(decision.calls) == 1
    assert decision.calls[0].name == "type"
    assert decision.calls[0].arguments == {"text": "Coupon code: OFF50", "clear": True}
    assert decision.progress == "Input, then copy" and decision.raw_action == content


@pytest.mark.parametrize(
    "action",
    [
        {"action": "Back", "progress": "one", "progress_note": "two"},
        {"action": "Back", "progress_note": {"action": "Home"}},
        {"action": "Back", "progress_note": "x" * 2001},
        {"action": "Back", "progress_note": "ok", "arguments": {}},
        {
            "action": "Swipe",
            "start": [540, 1900],
            "end": [540, 700],
            "progress_note": "ok",
        },
        {"action": "Tap", "index": 999, "progress_note": "ok"},
    ],
)
def test_progress_alias_does_not_accept_ambiguity_bad_targets_or_pixel_units(action):
    decision = parse_visual_decision(
        json.dumps(action),
        ui_state=_ui(),
        screenshot_size=(1080, 2400),
        available_tools=TOOLS | {"click"},
        structured=True,
    )
    assert decision.status == "error" and not decision.calls


@pytest.mark.parametrize("language", ["json", ""])
def test_single_fenced_swipe_preserves_explicit_pixels_and_one_action(language):
    content = '[{"action":"Swipe","coordinate_space":"screenshot_pixels","start":[540,1900],"end":[540,700],"progress":"Scroll to check list coverage"}]'
    ui = _ui(
        coordinate_scale_x=1080 / 922,
        coordinate_scale_y=2400 / 2048,
        model_screenshot_width=922,
        model_screenshot_height=2048,
    )
    decision = parse_visual_decision(
        f"```{language}\n{content}\n```",
        ui_state=ui,
        screenshot_size=(922, 2048),
        available_tools={"swipe"},
        structured=True,
    )
    assert decision.status == "action"
    assert len(decision.calls) == 1
    assert decision.calls[0].name == "swipe"
    assert decision.calls[0].arguments == {
        "coordinate": [540, 1900],
        "coordinate2": [540, 700],
        "duration": 1.0,
    }
    assert decision.progress == "Scroll to check list coverage"


@pytest.mark.parametrize(
    "content",
    [
        'Explanation\n```json\n{"action":"Back"}\n```',
        '```json\n{"action":"Back"}\n```\nExplanation',
        '```json\n{"action":"Back"}\n```\n```json\n{"action":"Home"}\n```',
        '```json\n{"action":"Back"}',
        '```python\n{"action":"Back"}\n```',
        '```json\n[{"action":"Back"},{"action":"Home"}]\n```',
        '```json\ndo(action="Back")\n```',
        '```json\n{"action":"Swipe","start":[540,1900],"end":[540,700]}\n```',
    ],
)
def test_fence_support_rejects_ambiguity_batches_code_and_unspecified_pixel_units(
    content,
):
    decision = parse_visual_decision(
        content,
        ui_state=_ui(),
        screenshot_size=(1080, 2400),
        available_tools=TOOLS,
        structured=True,
    )
    assert decision.status == "error"
    assert not decision.calls


def _ui(**kwargs):
    values = dict(
        elements=[],
        formatted_text="",
        focused_text="",
        phone_state={},
        screen_width=1080,
        screen_height=2400,
    )
    values.update(kwargs)
    return UIState(**values)


@pytest.mark.parametrize("recovery,json_mode", [(True, True), (False, False)])
def test_only_json_action_requests_constrain_the_response_format(
    monkeypatch, recovery, json_mode
):
    monkeypatch.setenv("GLM_API_KEY", "test-key")
    requests = []
    reply = '{"action":"Back"}' if json_mode else 'do(action="Back")'

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await VisualFallback(
                client=client, api_key_env="GLM_API_KEY", reasoning_model="glm-5.3-flash"
            ).decide(
                goal="Go back",
                screenshot=_png(),
                recovery=recovery,
                ui_state=_ui(),
                available_tools={"system_button"},
            )

    decision = asyncio.run(run())
    assert decision.status == "action"
    # do(...) replies are plain text, so only JSON-mode requests may set the flag.
    assert ("response_format" in requests[0]) is json_mode
    if json_mode:
        assert requests[0]["response_format"] == {"type": "json_object"}


def test_endpoint_rejecting_json_mode_falls_back_to_prompt_only(monkeypatch):
    monkeypatch.setenv("GLM_API_KEY", "test-key")
    requests = []

    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        if "response_format" in payload:
            return httpx.Response(400, json={"error": {"message": "unknown field"}})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"action":"Back"}'}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            fallback = VisualFallback(
                client=client, api_key_env="GLM_API_KEY", reasoning_model="glm-5.3-flash"
            )
            for _ in range(2):
                decision = await fallback.decide(
                    goal="Go back",
                    screenshot=_png(),
                    recovery=True,
                    ui_state=_ui(),
                    available_tools={"system_button"},
                )
                assert decision.status == "action"

    asyncio.run(run())
    # One rejected attempt, its plain retry, then later requests skip JSON mode.
    assert ["response_format" in r for r in requests] == [True, False, False]


@pytest.mark.parametrize("query_available", [False, True])
def test_request_advertises_only_available_query_capability_and_keeps_native_scroll_data(
    monkeypatch, query_available
):
    monkeypatch.setenv("GLM_API_KEY", "test-key")
    requests = []

    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        return httpx.Response(
            200, json={"choices": [{"message": {"content": '{"action":"Back"}'}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            tools = {"system_button"} | (
                {"query_collection"} if query_available else set()
            )
            return await VisualFallback(
                client=client,
                api_key_env="GLM_API_KEY",
                reasoning_model="glm-5.3-flash",
            ).decide(
                goal="Count entries in the selected folder",
                screenshot=_png(),
                recovery=True,
                ui_state=_ui(
                    elements=[
                        {
                            "index": 1,
                            "className": "ScrollView",
                            "bounds": "0,0,100,200",
                            "childCount": 8,
                            "scrollable": True,
                            "actionList": [{"id": 4096, "name": "SCROLL_FORWARD"}],
                        }
                    ]
                ),
                available_tools=tools,
            )

    decision = asyncio.run(run())
    assert decision.status == "action" and decision.calls[0].name == "system_button"
    system = requests[0]["messages"][0]["content"]
    text = requests[0]["messages"][-1]["content"][0]["text"]
    assert ('"action": "Query"' in system) is query_available
    assert ("Query为本地确定性计算" in system) is query_available
    assert ('"children_total": 8' in text) and "SCROLL_FORWARD" in text
    if not query_available:
        assert "禁止输出Query" in system
        invalid = parse_visual_decision(
            '{"action":"Query","plan":{"scope":"fake","generation":1,"operation":"count"}}',
            ui_state=_ui(),
            screenshot_size=(1080, 2400),
            available_tools={"system_button"},
            structured=True,
        )
        assert invalid.status == "error" and not invalid.calls


def _parse(content, *, ui=None, size=(1080, 2400), tools=TOOLS, apps=()):
    return parse_visual_decision(
        content,
        ui_state=ui or _ui(),
        screenshot_size=size,
        available_tools=tools,
        apps=apps,
    )


def _png(size=(108, 240)):
    output = BytesIO()
    Image.new("RGB", size, "white").save(output, format="PNG")
    return output.getvalue()


@pytest.mark.parametrize(
    "content",
    [
        'do(action="Tap", element=[-1, 500])',
        'do(action="Tap", element=[1000, 500])',
        'do(action="Tap", element=[True, 500])',
        'do(action="Tap", element=[50.5, 500])',
        'do(action="Tap", element=[500])',
        'do(action="Tap", element="[500,500]")',
        'do(action="Tap", element=[500, 500], surprise="anything")',
        'do(action="Type", text=__import__("os").getcwd())',
        'do(action="Type", text=f"{1+1}")',
        'do(action="Tap", element=[n for n in [1, 2]])',
        'do(**{"action": "Back"})',
        'do(action="Back", action="Home")',
        'do("Back")',
        'danger.do(action="Back")',
        'do(action="Back"); do(action="Home")',
        '<answer>do(action="Back")</answer><answer>finish(message="done")</answer>',
        '<answer>do(action="Back")',
        'do(action="Call_API", instruction="execute arbitrary code")',
        'finish(message="done", success=True)',
        'do(action="Wait", duration="NaN seconds")',
        'do(action="Wait", duration=1e309)',
        'do(action="Wait", duration="60 seconds")',
    ],
)
def test_untrusted_or_out_of_bounds_actions_are_never_executable(content):
    decision = _parse(content)
    assert decision.status == "error"
    assert not decision.calls


def test_explicit_answer_wins_over_action_examples_inside_thinking():
    decision = _parse(
        '<think>Do not use finish(message="done") yet.</think>'
        '<answer>do(action="Tap", element=[500,750])</answer>'
    )
    assert decision.status == "action"
    assert decision.calls[0].arguments == {"x": 540, "y": 1800}


def test_type_preserves_escaped_user_text_and_replaces_the_focused_input():
    text = '他说："你好"\n路径 C:\\notes\\todo.txt\nfinish(message="not an action")'
    decision = _parse(f'do(action="Type", text={text!r})')
    assert decision.status == "action"
    assert decision.calls[0].arguments == {"text": text, "clear": True}
    assert decision.calls[0].name == "type"


@pytest.mark.parametrize("screenshot_only", [False, True])
def test_type_uses_the_real_registry_contract_in_both_observation_modes(
    screenshot_only,
):
    async def run():
        registry, _ = await build_tool_registry(screenshot_only=screenshot_only)
        decision = _parse('do(action="Type", text="中文输入")', tools=registry.tools)
        assert decision.status == "action"
        call = decision.calls[0]
        assert call.name == "type"
        driver = SimpleNamespace(input_text=AsyncMock(return_value=True))
        context = SimpleNamespace(driver=driver, ui=_ui(), macro_recorder=None)
        result = await registry.execute(call.name, call.arguments, context)
        assert result.success
        driver.input_text.assert_awaited_once_with("中文输入", True)

    asyncio.run(run())


@pytest.mark.parametrize(
    "action",
    [
        'do(action="Take_over", message="请完成验证码")',
        'do(action="Interact")',
        'do(action="Tap", element=[500,500], message="请确认付款")',
    ],
)
def test_requests_for_human_help_do_not_dispatch_actions(action):
    result = _parse(action)
    assert result.status == "needs_input"
    assert result.message
    assert not result.calls


def test_endpoint_multiline_finish_is_opaque_text_and_never_a_device_action():
    message = (
        'Search completed.\nThe field contains "Bluetooth" and results are visible.'
    )
    result = _parse('finish(message="' + message + '")')
    assert result.status == "done"
    assert result.message == message
    assert not result.calls
    invalid = _parse('finish(message="done\n"); do(action="Back")')
    assert invalid.status == "error"


def test_optional_progress_is_metadata_not_a_device_argument():
    result = _parse(
        'do(action="Back", progress="Item one inspected; next inspect item two")'
    )
    assert result.status == "action"
    assert result.progress.startswith("Item one")
    assert result.calls[0].arguments == {"button": "back"}
    assert _parse('do(action="Back", progress=42)').status == "error"


def test_model_finish_is_only_a_completion_claim_not_a_complete_tool_call():
    result = _parse('finish(message="已打开设置")')
    assert result.status == "done"
    assert result.message == "已打开设置"
    assert not result.calls


def test_launch_resolves_installed_apps_only_and_rejects_ambiguous_labels():
    apps = [
        {"label": "设置", "package_name": "com.android.settings"},
        {"label": "相册", "package": "com.example.photos"},
        {"label": "相册", "package_name": "com.other.photos"},
    ]
    result = _parse('do(action="Launch", app="设置")', apps=apps)
    assert result.calls[0].arguments == {"package_name": "com.android.settings"}
    assert _parse('do(action="Launch", app="相册")', apps=apps).status == "error"
    assert (
        _parse('do(action="Launch", app="com.uninstalled.app")', apps=apps).status
        == "error"
    )
    exact = _parse('do(action="Launch", app="com.other.photos")', apps=apps)
    assert exact.calls[0].arguments == {"package_name": "com.other.photos"}


def test_double_tap_requires_an_atomic_tool_instead_of_two_independent_mutations():
    response = 'do(action="Double Tap", element=[250,750])'
    assert _parse(response).status == "error"
    result = _parse(response, tools={"double_tap_at"})
    assert len(result.calls) == 1
    assert result.calls[0].name == "double_tap_at"


@pytest.mark.parametrize("requested_app", ["设置", "com.android.settings"])
def test_launch_accepts_portal_camelcase_inventory_without_allowing_unknown_packages(
    requested_app,
):
    # Portal's get_apps response uses these names, unlike some driver adapters.
    apps = [
        {"packageName": "com.android.settings", "appName": "设置", "isSystem": True}
    ]
    result = _parse(f'do(action="Launch", app={requested_app!r})', apps=apps)
    assert result.status == "action"
    assert result.calls[0].name == "launch_app"
    assert result.calls[0].arguments == {"package_name": "com.android.settings"}
    assert (
        _parse('do(action="Launch", app="com.unknown.app")', apps=apps).status
        == "error"
    )


@pytest.mark.parametrize(
    "requested_app,label,package",
    [
        ("系统设置", "设置", "com.android.settings"),
        ("手机相机", "相机", "com.android.camera"),
        ("Android Settings", "Settings", "com.android.settings"),
    ],
)
def test_launch_removes_only_known_device_qualifiers(requested_app, label, package):
    apps = [{"packageName": package, "appName": label}]
    result = _parse(f'do(action="Launch", app={requested_app!r})', apps=apps)
    assert result.status == "action"
    assert result.calls[0].arguments == {"package_name": package}


def test_launch_exact_label_precedes_normalization_but_ambiguous_alias_is_rejected():
    apps = [
        {"packageName": "com.android.camera", "appName": "相机"},
        {"packageName": "com.other.camera", "appName": "系统相机"},
    ]
    exact = _parse('do(action="Launch", app="系统相机")', apps=apps)
    assert exact.calls[0].arguments == {"package_name": "com.other.camera"}
    assert _parse('do(action="Launch", app="手机相机")', apps=apps).status == "error"


@pytest.mark.parametrize(
    "requested_app",
    ["恶意设置", "管理员设置", "Android恶意设置", "系统设置;evil()", "系统系统设置"],
)
def test_launch_does_not_fuzzy_match_arbitrary_or_repeated_prefixes(requested_app):
    apps = [{"packageName": "com.android.settings", "appName": "设置"}]
    assert (
        _parse(f'do(action="Launch", app={requested_app!r})', apps=apps).status
        == "error"
    )


def test_disabled_coordinate_tool_is_not_reintroduced_by_the_fallback():
    result = _parse('do(action="Tap", element=[500,500])', tools={"click"})
    assert result.status == "error"
    assert not result.calls


def test_android_resized_contract_scales_exactly_once_at_execution():
    ui = _ui(
        coordinate_scale_x=2.0,
        coordinate_scale_y=2.0,
        coordinate_contract_active=True,
        model_screenshot_width=540,
        model_screenshot_height=1200,
    )
    result = _parse('do(action="Tap", element=[500,750])', ui=ui, size=(540, 1200))
    args = result.calls[0].parameters
    assert args == {"x": 270, "y": 900}
    assert ui.convert_point(args["x"], args["y"]) == (540, 1800)


def test_raw_full_screenshot_and_resized_screenshot_select_the_same_device_point():
    ui = _ui(
        coordinate_scale_x=2.0,
        coordinate_scale_y=2.0,
        coordinate_contract_active=True,
        model_screenshot_width=540,
        model_screenshot_height=1200,
    )
    content = 'do(action="Tap", element=[300,600])'
    raw = _parse(content, ui=ui)
    resized = _parse(content, ui=ui, size=(540, 1200))
    assert raw.calls == resized.calls


def test_screenshot_only_screen_size_is_display_size_not_native_size():
    ui = _ui(
        screen_width=540,
        screen_height=1200,
        phone_state={"observationMode": "screenshot_only"},
        coordinate_scale_x=2.0,
        coordinate_scale_y=2.0,
        model_screenshot_width=540,
        model_screenshot_height=1200,
    )
    result = _parse('do(action="Swipe", start=[100,900], end=[100,100])', ui=ui)
    args = result.calls[0].arguments
    assert ui.convert_point(*args["coordinate"]) == (108, 2160)
    assert ui.convert_point(*args["coordinate2"]) == (108, 240)


def test_normalized_ui_contract_keeps_coordinates_normalized():
    ui = _ui(use_normalized=True)
    result = _parse('do(action="Tap", element=[999,999])', ui=ui)
    args = result.calls[0].arguments
    assert args == {"x": 999, "y": 999}
    assert ui.convert_point(**args) == (1078, 2397)


def test_ios_rounding_cannot_push_the_last_display_pixel_outside_input_bounds():
    ui = _ui(
        screen_width=400,
        screen_height=800,
        coordinate_scale_x=0.4,
        coordinate_scale_y=0.4,
        coordinate_contract_active=True,
        model_screenshot_width=1000,
        model_screenshot_height=2000,
    )
    result = _parse('do(action="Tap", element=[999,999])', ui=ui, size=(1000, 2000))
    assert result.status == "action"
    assert ui.convert_point(**result.calls[0].arguments) == (399, 799)


@pytest.mark.parametrize(
    "ui,size",
    [
        (_ui(screen_width=0), (1080, 2400)),
        (_ui(), (2400, 1080)),
        (_ui(coordinate_scale_x=float("nan")), (1080, 2400)),
        (_ui(coordinate_scale_x=0), (1080, 2400)),
        (_ui(model_screenshot_width=540), (1080, 2400)),
        (_ui(model_screenshot_width=540, model_screenshot_height=1200), (1080, 2400)),
    ],
)
def test_stale_dimensions_rotation_and_broken_scaling_reject_coordinate_actions(
    ui, size
):
    result = _parse('do(action="Tap", element=[500,500])', ui=ui, size=size)
    assert result.status == "error"
    assert not result.calls


def test_http_request_uses_official_endpoint_latest_image_and_bounded_text_history(
    monkeypatch,
):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-primary-key")
    monkeypatch.setenv("ZAI_API_KEY", "test-secondary-key")
    requests = []
    shot = _png()

    def handle(request):
        assert str(request.url) == BIGMODEL_CHAT_URL
        assert request.headers["Authorization"] == "Bearer test-primary-key"
        requests.append(json.loads(request.content))
        return httpx.Response(
            200, json={"choices": [{"message": {"content": 'do(action="Back")'}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            fallback = VisualFallback(client=client, max_history_turns=1)
            for _ in range(3):
                decision = await fallback.decide(
                    goal="返回设置",
                    screenshot=shot,
                    ui_state=_ui(),
                    available_tools=TOOLS,
                    recent_actions=[{"name": "click_at", "success": True}],
                    feedback="已进入子页面",
                )
                assert decision.status == "action"
            await fallback.decide(
                goal="新任务", screenshot=shot, ui_state=_ui(), available_tools=TOOLS
            )

    asyncio.run(run())
    assert [len(r["messages"]) for r in requests] == [2, 4, 4, 2]
    assert all(r["stream"] is False for r in requests)
    image_url = requests[2]["messages"][-1]["content"][1]["image_url"]["url"]
    assert base64.b64decode(image_url.split(",", 1)[1]) == shot
    assert all(isinstance(m["content"], str) for m in requests[2]["messages"][:-1])
    observation = requests[2]["messages"][-1]["content"][0]["text"]
    assert "click_at" in observation and "已进入子页面" in observation


@pytest.mark.parametrize(
    "url", ["https://relay.example/v1", "https://relay.example/v1/chat/completions/"]
)
@pytest.mark.parametrize("thinking", [None, "enabled", "disabled"])
def test_custom_provider_routes_preparation_and_action_with_its_own_key(
    monkeypatch, url, thinking
):
    monkeypatch.setenv("BIGMODEL_API_KEY", "official-key-must-not-be-sent")
    monkeypatch.setenv("GLM_API_KEY", "relay-test-key")
    requests = []

    def handle(request):
        assert str(request.url) == "https://relay.example/v1/chat/completions"
        assert request.headers["Authorization"] == "Bearer relay-test-key"
        payload = json.loads(request.content)
        assert payload["model"] == "glm-5.3-flash"
        if thinking is None:
            assert "thinking" not in payload
        else:
            assert payload["thinking"] == {"type": thinking}
        requests.append(payload)
        reply = (
            json.dumps(
                {
                    "requirements": [
                        {
                            "id": 1,
                            "kind": "state",
                            "description": "Go back",
                            "quote": "Go back",
                        }
                    ],
                    "answer_required": False,
                }
            )
            if len(requests) == 1
            # A non-AutoGLM action model answers in JSON, not do(...).
            else '{"action":"Back"}'
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            fallback = VisualFallback(
                client=client,
                base_url=url,
                api_key_env="GLM_API_KEY",
                model="glm-5.3-flash",
                reasoning_model="glm-5.3-flash",
                thinking_mode=thinking,
            )
            assert await fallback.prepare_task(goal="Go back") is None
            result = await fallback.decide(
                goal="Go back", screenshot=_png(), ui_state=_ui(), available_tools=TOOLS
            )
            assert result.status == "action"

    asyncio.run(run())
    assert len(requests) == 2
    assert requests[1]["response_format"] == {"type": "json_object"}


def test_missing_relay_key_does_not_fall_back_to_official_credentials(monkeypatch):
    monkeypatch.setenv("BIGMODEL_API_KEY", "official-key")
    monkeypatch.setenv("ZAI_API_KEY", "other-official-key")
    monkeypatch.delenv("GLM_API_KEY", raising=False)

    def reject(request):
        raise AssertionError(
            "No request is authorized with the wrong provider credential"
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(reject)) as client:
            fallback = VisualFallback(
                client=client,
                base_url="https://relay.example/v1",
                api_key_env="GLM_API_KEY",
                reasoning_model="glm-5.3-flash",
            )
            prepared = await fallback.prepare_task(goal="Go back")
            action = await fallback.decide(
                goal="Go back", screenshot=_png(), ui_state=_ui(), available_tools=TOOLS
            )
            assert prepared.error_code == action.error_code == "credentials_missing"
            assert "GLM_API_KEY" in prepared.message

    asyncio.run(run())


@pytest.mark.parametrize(
    "url",
    [
        "ftp://relay.example/v1",
        "https://key@relay.example/v1",
        "https://relay.example/v1?api_key=secret",
        "https://relay.example/v1#secret",
        "/v1",
    ],
)
def test_provider_url_rejects_embedded_secrets_and_non_http_urls(url):
    with pytest.raises(ValueError):
        VisualFallback(base_url=url)


@pytest.mark.parametrize("typing_only", [False, True])
def test_request_includes_observed_input_focus_despite_stale_phone_metadata(
    monkeypatch, typing_only
):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-key")
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": 'do(action="Type", text="example.com")'}}
                ]
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result = await VisualFallback(client=client).decide(
                goal="Enter example.com without submitting",
                screenshot=_png(),
                ui_state=_ui(
                    phone_state={"isEditable": False},
                    elements=[
                        {
                            "index": 1,
                            "className": "EditText",
                            "focused": True,
                            "bounds": "0,0,1000,100",
                            "resourceId": "browser:id/url",
                            "inputValue": "",
                        }
                    ],
                ),
                available_tools={"type"} if typing_only else TOOLS,
            )
            assert result.calls[0].arguments == {"text": "example.com", "clear": True}

    asyncio.run(run())
    observation = requests[0]["messages"][-1]["content"][0]["text"]
    assert "browser:id/url" in observation and '"focused": true' in observation
    content = requests[0]["messages"][-1]["content"]
    if typing_only:
        assert [item["type"] for item in content] == ["text"]
        assert "禁止Tap" in requests[0]["messages"][0]["content"]
    else:
        assert '"controls"' in observation
        assert content[1]["type"] == "image_url"


def test_control_evidence_is_bounded_visible_and_password_redacted():
    nodes = [
        {
            "index": i,
            "className": "TextView",
            "text": "label " + str(i),
            "bounds": "0,0,100,100",
        }
        for i in range(82)
    ]
    nodes[0].update(
        password=True,
        text="private password",
        className="EditText",
        inputValue="private password",
    )
    nodes[1].update(visible=False, text="hidden label")
    evidence = control_observations(_ui(elements=nodes))
    assert len(evidence["controls"]) == 80 and evidence["omitted"] == 1
    assert "[password]" in str(evidence)
    assert "private password" not in str(evidence)
    assert evidence["controls"][0]["input_value"] == "[password]"
    assert evidence["controls"][0]["bounds"] == (0, 0, 100, 100)
    assert "hidden label" not in str(evidence)


def test_text_argument_resolution_cannot_finish_the_whole_task(monkeypatch):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-key")

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(
                lambda _: httpx.Response(
                    200,
                    json={
                        "choices": [{"message": {"content": 'finish(message="done")'}}]
                    },
                )
            )
        ) as client:
            return await VisualFallback(client=client).decide(
                goal="Enter example.com",
                screenshot=_png(),
                ui_state=_ui(
                    elements=[
                        {
                            "index": 1,
                            "className": "EditText",
                            "bounds": "0,0,1000,100",
                            "focused": True,
                            "inputValue": "",
                        }
                    ]
                ),
                available_tools={"type"},
            )

    result = asyncio.run(run())
    assert result.status == "error" and not result.calls


def test_review_normalizes_only_empty_optional_check_containers():
    result = parse_completion_review(
        json.dumps(
            {
                "complete": True,
                "evidence": "Current input is cat",
                "missing": [],
                "checks": {
                    "input_text": "cat",
                    "text_all": [],
                    "text_any": [],
                    "checked": {},
                },
            }
        )
    )
    assert result.status == "done"
    assert result.completion_checks == {"input_text": "cat"}


@pytest.mark.parametrize(
    "checks",
    [
        {"text_all": ["Copy"], "checked": {"Wi-Fi": False}},
        {"unknown_condition": []},
        {"input_text": ""},
        {"text_any": [""]},
    ],
)
def test_review_keeps_substantive_and_invalid_fields_for_runtime_validation(checks):
    result = parse_completion_review(
        json.dumps(
            {
                "complete": True,
                "evidence": "Observed page",
                "missing": [],
                "checks": checks,
            }
        )
    )
    assert result.completion_checks == checks


def test_review_with_only_empty_checks_cannot_certify_an_accessible_page(monkeypatch):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-key")
    content = json.dumps(
        {
            "complete": True,
            "evidence": "Everything is done",
            "missing": [],
            "checks": {"text_all": [], "text_any": [], "checked": {}},
            "assessments": [
                {"id": 1, "satisfied": True, "evidence": "Menu is visible"}
            ],
        }
    )

    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        reply = (
            json.dumps({"requirements": ["Open the Copy menu"], "checks": {}})
            if len(requests) == 1
            else content
        )
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await VisualFallback(
                client=client, reasoning_model="review-model"
            ).decide(
                goal="Open the Copy menu",
                screenshot=_png(),
                ui_state=_ui(
                    elements=[{"index": 1, "text": "Home", "bounds": "0,0,100,100"}]
                ),
                available_tools=TOOLS,
                completion_review=True,
            )

    result = asyncio.run(run())
    assert result.status == "error" and not result.calls
    assert len(requests) == 2
    context = json.loads(requests[1]["messages"][-1]["content"][0]["text"])
    assert context["native_checks_required"] is True


@pytest.mark.parametrize(
    "answer,complete,required,status",
    [
        ("14:00, 田中杏奈", True, True, "done"),
        ("0", True, True, "done"),
        (None, True, False, "done"),
        (None, True, True, "error"),
        (" ", True, True, "error"),
        (0, True, True, "error"),
        (["14:00"], True, True, "error"),
        ("14:00", False, True, "incomplete"),
    ],
)
def test_final_answer_is_explicit_typed_and_only_released_after_review(
    answer, complete, required, status
):
    review = {
        "complete": complete,
        "evidence": "The event detail is visible",
        "missing": [] if complete else ["Date unverified"],
        "checks": {},
        "final_answer": answer,
    }
    result = parse_completion_review(json.dumps(review), answer_required=required)
    assert result.status == status
    assert result.final_answer == (answer if status == "done" else None)


def test_query_contract_requires_answer_on_every_review(monkeypatch):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-key")
    requests = []

    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        content = payload["messages"][-1]["content"]
        if isinstance(content, str):
            assert content == "What time is the meeting?"
            reply = {
                "requirements": ["Report meeting time"],
                "checks": {},
                "answer_required": True,
            }
        else:
            observation = json.loads(content[0]["text"])
            assert observation["completion_contract"]["answer_required"] is True
            reply = {
                "complete": True,
                "evidence": "Event detail shows 14:00",
                "missing": [],
                "checks": {"text_all": ["14:00"]},
                "assessments": [
                    {"id": 1, "satisfied": True, "evidence": "14:00 visible"}
                ],
            }
            if len(requests) == 3:
                reply["final_answer"] = "14:00"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(reply)}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            fallback = VisualFallback(client=client, reasoning_model="glm-4.6v")
            args = dict(
                goal="What time is the meeting?",
                screenshot=_png(),
                ui_state=_ui(
                    elements=[{"index": 1, "text": "14:00", "bounds": "0,0,100,100"}]
                ),
                available_tools=TOOLS,
                completion_review=True,
            )
            missing = await fallback.decide(**args)
            assert missing.status == "error" and missing.final_answer is None
            valid = await fallback.decide(**args)
            assert valid.status == "done" and valid.final_answer == "14:00"

    asyncio.run(run())
    assert len(requests) == 3


@pytest.mark.parametrize("complete", [True, False])
def test_completion_review_requires_explicit_structured_evidence(complete):
    result = parse_completion_review(
        json.dumps(
            {
                "complete": complete,
                "evidence": "The Files drawer is visible.",
                "missing": [] if complete else ["App info is not visible"],
                "checks": {"text_all": ["Downloads", "Pictures"]},
            }
        )
    )
    assert result.status == ("done" if complete else "incomplete")
    assert not result.calls
    assert result.completion_checks == {"text_all": ["Downloads", "Pictures"]}
    if not complete:
        assert "App info is not visible" in result.message


@pytest.mark.parametrize(
    "content",
    [
        'finish(message="done")',
        "[]",
        "null",
        '{"complete":"true","evidence":"visible","missing":[],"checks":{}}',
        '{"complete":true,"evidence":"visible","missing":["not downloaded"],"checks":{}}',
        '{"complete":true,"evidence":"","missing":[],"checks":{}}',
        '{"complete":true,"evidence":"visible","missing":"none","checks":{}}',
        '{"complete":true,"evidence":"visible","missing":[]}',
        '{"complete":true,"evidence":"visible","missing":[],"checks":[]}',
    ],
)
def test_invalid_or_contradictory_completion_review_cannot_finish(content):
    result = parse_completion_review(content)
    assert result.status == "error" and not result.calls


@pytest.mark.parametrize(
    "assessments",
    [
        None,
        [],
        [{"id": 1, "satisfied": True, "evidence": "cat is entered"}],
        [
            {"id": 1, "satisfied": True, "evidence": "cat is entered"},
            {"id": 1, "satisfied": True, "evidence": "Same input again"},
        ],
        [
            {"id": True, "satisfied": True, "evidence": "cat is entered"},
            {"id": 2, "satisfied": True, "evidence": "Copy is visible"},
        ],
        [
            {"id": 1, "satisfied": True, "evidence": "cat is entered"},
            {"id": 2, "satisfied": False, "evidence": "Copy is absent"},
        ],
        [
            {"id": 1, "satisfied": True, "evidence": "cat is entered"},
            {"id": 3, "satisfied": True, "evidence": "Different requirement"},
        ],
        [
            {"id": 1, "satisfied": True, "evidence": "cat is entered"},
            {"id": 2, "satisfied": "true", "evidence": "Copy is visible"},
        ],
        [
            {"id": 1, "satisfied": True, "evidence": "cat is entered"},
            {"id": 2, "satisfied": True, "evidence": " "},
        ],
    ],
)
def test_input_evidence_cannot_silently_omit_the_required_menu(assessments):
    review = {
        "complete": True,
        "evidence": "cat is entered",
        "checks": {"input_text": "cat"},
        "missing": [],
    }
    if assessments is not None:
        review["assessments"] = assessments
    result = parse_completion_review(
        json.dumps(review), requirements=["Enter cat", "Leave the Copy menu visible"]
    )
    assert result.status == "error" and not result.calls


@pytest.mark.parametrize("menu_visible", [True, False])
def test_each_requirement_keeps_its_evidence_and_unmet_item(menu_visible):
    result = parse_completion_review(
        json.dumps(
            {
                "complete": menu_visible,
                "evidence": "Reviewed input and menu separately",
                "checks": {"input_text": "cat", "text_all": ["Copy"]},
                "missing": [],
                "assessments": [
                    {"id": 2, "satisfied": menu_visible, "evidence": "Menu inspected"},
                    {
                        "id": 1,
                        "satisfied": True,
                        "evidence": "Input field contains cat",
                    },
                ],
            }
        ),
        requirements=["Enter cat", "Leave the Copy menu visible"],
    )
    assert result.status == ("done" if menu_visible else "incomplete")
    assert [item["requirement"] for item in result.assessments] == [
        "Enter cat",
        "Leave the Copy menu visible",
    ]
    assert result.assessments[0]["evidence"] == "Input field contains cat"
    if not menu_visible:
        assert "Leave the Copy menu visible" in result.message


@pytest.mark.parametrize("recovery", [True, False])
def test_semantic_recovery_uses_reasoner_only_when_requested(monkeypatch, recovery):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-key")
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "message": {
                            "content": (
                                json.dumps(
                                    {
                                        "action": "Long Press",
                                        "element": [500, 500],
                                        "progress": "Input ready; open selection menu",
                                    }
                                )
                                if recovery
                                else 'do(action="Long Press", element=[500,500], progress="Input ready; open selection menu")'
                            )
                        }
                    }
                ]
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            fallback = VisualFallback(client=client, reasoning_model="reasoner")
            result = await fallback.decide(
                goal="Select the entered text",
                screenshot=_png(),
                ui_state=_ui(),
                available_tools=TOOLS,
                recovery=recovery,
            )
            assert result.status == "action"
            assert result.calls[0].name == "long_press_at"
            assert result.progress == "Input ready; open selection menu"

    asyncio.run(run())
    assert requests[0]["model"] == ("reasoner" if recovery else "autoglm-phone")
    # JSON-mode (structured) requests constrain the format; do(...) requests cannot.
    assert ("response_format" in requests[0]) is recovery


@pytest.mark.parametrize(
    "action",
    [
        {"action": "Tap", "element": [1000, 50]},
        {"action": "Long Press", "element": [True, 500]},
        {"action": "Back", "arguments": {}},
        {"action": "system_button", "button": "back"},
        {"action": "Type", "text": {"code": "anything"}},
        [{"action": "Back"}, {"action": "Home"}],
    ],
)
def test_structured_recovery_preserves_action_validation(action):
    result = parse_visual_decision(
        json.dumps(action),
        ui_state=_ui(),
        screenshot_size=(1080, 2400),
        available_tools=TOOLS,
        structured=True,
    )
    assert result.status == "error" and not result.calls


def test_structured_recovery_cannot_use_unavailable_tools():
    result = parse_visual_decision(
        '{"action":"Long Press","element":[500,500]}',
        ui_state=_ui(),
        screenshot_size=(1080, 2400),
        available_tools={"type"},
        structured=True,
    )
    assert result.status == "error" and not result.calls


@pytest.mark.parametrize("review", [True, False])
@pytest.mark.parametrize(
    "model,token_limit",
    [("review-model", 3000), ("glm-4.6v", 3000), ("glm-4v-flash", 1024)],
)
def test_reasoning_model_handles_review_and_text_resolution_without_executor_history(
    monkeypatch, review, model, token_limit
):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-key")
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        content = (
            json.dumps(
                {
                    "complete": False,
                    "evidence": "Folder is visible",
                    "missing": ["App info menu"],
                    "checks": {"text_all": ["App info"]},
                    "assessments": [
                        {"id": 1, "satisfied": False, "evidence": "Input is empty"}
                    ],
                }
            )
            if review
            else json.dumps({"action": "Type", "text": "example.com"})
        )
        if review and len(requests) == 1:
            content = json.dumps(
                {
                    "requirements": ["Enter example.com"],
                    "checks": {"input_text": "example.com"},
                }
            )
        content = "<|begin_of_box|>" + content + "<|end_of_box|>"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            fallback = VisualFallback(client=client, reasoning_model=model)
            result = await fallback.decide(
                goal="Enter example.com",
                screenshot=_png(),
                ui_state=_ui(
                    elements=[
                        {
                            "index": 1,
                            "className": "EditText",
                            "bounds": "0,0,1000,100",
                            "focused": True,
                            "inputValue": "",
                        }
                    ]
                ),
                available_tools={"type"},
                completion_review=review,
                review_feedback="Return all required checks",
                feedback="executor-claims-everything-is-done",
                task_state={"execution_log": []},
            )
            assert result.model == model
            assert result.status == ("incomplete" if review else "action")

    asyncio.run(run())
    request = requests[-1]
    assert request["model"] == model
    assert all(r["max_tokens"] == token_limit for r in requests)
    # Text resolution is a JSON-mode request; the reviewer path is not.
    assert all(("response_format" in r) is (not review) for r in requests)
    content = request["messages"][-1]["content"]
    if review:
        assert len(requests) == 2
        assert requests[0]["messages"][-1] == {
            "role": "user",
            "content": "Enter example.com",
        }
        assert "executor-claims-everything-is-done" not in str(request)
        assert [part["type"] for part in content] == ["text", "image_url"]
        observation = json.loads(content[0]["text"])
        assert observation["completion_contract"]["requirements"] == [
            {"id": 1, "requirement": "Enter example.com"}
        ]
        assert observation["format_feedback"] == "Return all required checks"
        assert observation["current_state"]["isEditable"] is True
        assert observation["current_state"]["focusedElement"]["index"] == 1
    else:
        assert len(requests) == 1
        assert [part["type"] for part in content] == ["text"]


def test_goal_only_contract_survives_omitted_review_checks_and_resets_on_new_goal(
    monkeypatch,
):
    from mobilerun.agent.decision.runtime import completion_matches

    monkeypatch.setenv("BIGMODEL_API_KEY", "test-key")
    requests = []
    goal = "Open Files and leave the drawer showing Downloads and Pictures"
    required = {"text_all": ["Downloads", "Pictures"]}
    closed = _ui(
        phone_state={"packageName": "com.android.documentsui"},
        elements=[{"index": 1, "text": "Recent", "bounds": "0,0,100,100"}],
    )

    def handle(request):
        payload = json.loads(request.content)
        requests.append(payload)
        user = payload["messages"][-1]["content"]
        if isinstance(user, str):
            assert user in {goal, "New task"}
            reply = {"requirements": [user], "checks": required}
        else:
            context = json.loads(user[0]["text"])
            assert context["completion_contract"]["checks"] == required
            reply = {
                "complete": True,
                "evidence": "Files is open",
                "missing": [],
                "checks": {"package": "com.android.documentsui"},
                "assessments": [
                    {"id": 1, "satisfied": True, "evidence": "Claimed drawer visible"}
                ],
            }
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(reply)}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            fallback = VisualFallback(client=client, reasoning_model="review-model")
            for task in (goal, goal, "New task"):
                result = await fallback.decide(
                    goal=task,
                    screenshot=_png(),
                    ui_state=closed,
                    available_tools=TOOLS,
                    task_state={
                        "execution_log": ["screen-history-must-not-reach-compiler"]
                    },
                    completion_review=True,
                )
                assert result.completion_checks["text_all"] == required["text_all"]
                assert not completion_matches(closed, result.completion_checks)

    asyncio.run(run())
    contract_requests = [
        r for r in requests if isinstance(r["messages"][-1]["content"], str)
    ]
    assert len(requests) == 5 and len(contract_requests) == 2
    assert "screen-history-must-not-reach-compiler" not in str(contract_requests)
    assert "image_url" not in str(contract_requests)


@pytest.mark.parametrize(
    "observed,required",
    [
        ({"input_text": "dog"}, {"input_text": "cat"}),
        ({"text_any": ["Pictures"]}, {"text_any": ["Downloads"]}),
        ({"checked": {"Wi-Fi": True}}, {"checked": {"Wi-Fi": False}}),
    ],
)
def test_review_cannot_replace_required_conditions_with_conflicting_alternatives(
    observed, required
):
    with pytest.raises(ValueError, match="contradicts"):
        _merge_required_checks(observed, required)


@pytest.mark.parametrize("explicit_package", [True, False])
def test_claimed_package_must_come_from_goal_contract_or_observation(
    monkeypatch, explicit_package
):
    from mobilerun.agent.decision.runtime import completion_matches

    monkeypatch.setenv("BIGMODEL_API_KEY", "test-key")
    observed = _ui(phone_state={"packageName": "com.android.settings.intelligence"})
    requested = "com.android.settings"

    def handle(request):
        user = json.loads(request.content)["messages"][-1]["content"]
        reply = (
            {
                "requirements": ["Stay on the requested page"],
                "checks": {"package": requested} if explicit_package else {},
            }
            if isinstance(user, str)
            else {
                "complete": True,
                "evidence": "The original settings app is foreground",
                "missing": [],
                "checks": {"package": requested},
                "assessments": [
                    {"id": 1, "satisfied": True, "evidence": "Claimed package visible"}
                ],
            }
        )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(reply)}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await VisualFallback(
                client=client, reasoning_model="review-model"
            ).decide(
                goal="Stay on the requested page",
                screenshot=_png(),
                ui_state=observed,
                available_tools=TOOLS,
                completion_review=True,
            )

    result = asyncio.run(run())
    assert not result.calls
    if explicit_package:
        # A genuine goal mismatch remains a required check, never rewritten to pass.
        assert result.completion_checks == {"package": requested}
        assert not completion_matches(observed, result.completion_checks)
    else:
        assert result.status == "error"
        assert (
            result.message
            == "Completion reviewer contradicted the observed foreground package."
        )


def test_required_text_and_checkbox_conditions_accumulate_without_mutating_contract():
    required = {"text_all": ["Downloads", "Pictures"], "checked": {"Wi-Fi": False}}
    result = _merge_required_checks(
        {"text_all": ["Downloads", "Recent"], "checked": {"Bluetooth": True}}, required
    )
    assert result["text_all"] == ["Downloads", "Recent", "Pictures"]
    assert result["checked"] == {"Bluetooth": True, "Wi-Fi": False}
    assert required == {
        "text_all": ["Downloads", "Pictures"],
        "checked": {"Wi-Fi": False},
    }


@pytest.mark.parametrize("second_status", [200, 429])
def test_busy_response_retries_once_without_dispatching_device_actions(
    monkeypatch, second_status
):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-key")
    requests, sleeps = [], []

    async def sleep(delay):
        sleeps.append(delay)

    monkeypatch.setattr("mobilerun.agent.decision.visual_fallback.asyncio.sleep", sleep)

    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1 or second_status == 429:
            return httpx.Response(429, json={"error": {"code": "1305"}})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": 'do(action="Back")'}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            result = await VisualFallback(client=client).decide(
                goal="Go back", screenshot=_png(), ui_state=_ui(), available_tools=TOOLS
            )
            assert not client.is_closed
            return result

    result = asyncio.run(run())
    assert len(requests) == 2 and requests[0] == requests[1]
    assert sleeps == [2]
    assert result.status == ("action" if second_status == 200 else "error")
    assert len(result.calls) == (1 if second_status == 200 else 0)


@pytest.mark.parametrize("body", [{"error": {"code": "1113"}}, {"error": None}, [], {}])
def test_other_rate_limit_errors_are_not_retried(monkeypatch, body):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-key")
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(429, json=body)

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await VisualFallback(client=client).decide(
                goal="Go back", screenshot=_png(), ui_state=_ui(), available_tools=TOOLS
            )

    result = asyncio.run(run())
    assert result.status == "error" and not result.calls
    assert len(requests) == 1


def test_busy_backoff_cannot_exceed_total_request_deadline(monkeypatch):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-key")
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(429, json={"error": {"code": 1305}})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await VisualFallback(client=client, timeout=0.05).decide(
                goal="Go back", screenshot=_png(), ui_state=_ui(), available_tools=TOOLS
            )

    result = asyncio.run(run())
    assert result.status == "error" and not result.calls
    assert "timed out" in result.message
    assert len(requests) == 1


@pytest.mark.parametrize("status", [500, 502, 503, 504])
@pytest.mark.parametrize("recovers", [True, False])
def test_typed_inference_transient_errors_retry_once_and_return_only_one_action(
    monkeypatch, status, recovers
):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-key")
    requests = []

    async def sleep(delay):
        assert delay == 0.25

    monkeypatch.setattr("mobilerun.agent.decision.visual_fallback.asyncio.sleep", sleep)

    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1 or not recovers:
            return httpx.Response(status, json={"error": {"code": "upstream"}})
        return httpx.Response(
            200, json={"choices": [{"message": {"content": 'do(action="Back")'}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            fallback = VisualFallback(client=client)
            fallback._goal = "Go back"
            fallback._completion_contract = {
                "schema_version": 2,
                "checks": {},
                "answer_required": False,
                "requirements": [
                    {
                        "id": 1,
                        "kind": "state",
                        "description": "Return to the previous page",
                        "quote": "Go back",
                    }
                ],
            }
            return await fallback.decide(
                goal="Go back", screenshot=_png(), ui_state=_ui(), available_tools=TOOLS
            )

    result = asyncio.run(run())
    assert len(requests) == 2 and requests[0] == requests[1]
    assert result.status == ("action" if recovers else "error")
    assert len(result.calls) == (1 if recovers else 0)


def test_task_preparation_transient_retry_is_bounded_before_a_contract_exists(
    monkeypatch,
):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-key")
    requests = []

    async def sleep(delay):
        assert delay == 0.25

    monkeypatch.setattr("mobilerun.agent.decision.visual_fallback.asyncio.sleep", sleep)

    def handle(request):
        requests.append(json.loads(request.content))
        if len(requests) == 1:
            return httpx.Response(500, json={"error": {"code": "upstream"}})
        content = json.dumps(
            {
                "requirements": [
                    {
                        "id": 1,
                        "kind": "state",
                        "description": "Return to the previous page",
                        "quote": "Go back",
                    }
                ],
                "answer_required": False,
            }
        )
        return httpx.Response(
            200, json={"choices": [{"message": {"content": content}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            fallback = VisualFallback(client=client, reasoning_model="glm-4.6v")
            result = await fallback.prepare_task(goal="Go back")
            assert fallback.completion_contract is not None
            return result

    result = asyncio.run(run())
    assert len(requests) == 2 and requests[0] == requests[1]
    assert result is None  # A cached validated contract is preparation success.


def test_transient_retry_does_not_extend_total_deadline(monkeypatch):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-key")
    requests = []

    def handle(request):
        requests.append(request)
        return httpx.Response(500, json={"error": {"code": "upstream"}})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await VisualFallback(
                client=client, reasoning_model="glm-4.6v", timeout=0.05
            ).prepare_task(goal="Go back")

    result = asyncio.run(run())
    assert result.status == "error" and not result.calls
    assert len(requests) == 1
    assert result.error_code == "inference_timeout"


@pytest.mark.parametrize("fixed_component", [".md", "$", "+86"])
def test_text_resolution_receives_other_current_form_components_without_an_image(
    monkeypatch, fixed_component
):
    monkeypatch.setenv("GLM_API_KEY", "test-key")
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {"message": {"content": '{"action":"Type","text":"field part"}'}}
                ]
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            ui = _ui(
                elements=[
                    {
                        "index": 1,
                        "className": "EditText",
                        "focused": True,
                        "resourceId": "generic:id/part",
                        "bounds": "0,0,300,100",
                        "inputValue": "",
                    },
                    {
                        "index": 2,
                        "className": "EditText",
                        "resourceId": "generic:id/fixed",
                        "bounds": "300,0,450,100",
                        "inputValue": fixed_component,
                        "isEnabled": False,
                    },
                ]
            )
            fallback = VisualFallback(
                client=client,
                base_url="https://relay.example/v1",
                api_key_env="GLM_API_KEY",
                reasoning_model="glm-5.3-flash",
                thinking_mode="disabled",
            )
            return await fallback.decide(
                goal="Fill the composite value",
                screenshot=_png(),
                ui_state=ui,
                available_tools={"type"},
            )

    decision = asyncio.run(run())
    assert decision.status == "action" and decision.calls[0].arguments == {
        "text": "field part",
        "clear": True,
    }
    content = requests[0]["messages"][-1]["content"]
    assert [part["type"] for part in content] == ["text"]
    assert (
        fixed_component in content[0]["text"]
        and '"enabled": false' in content[0]["text"]
    )


@pytest.mark.parametrize(
    "failure",
    [
        "http",
        "timeout",
        "connection",
        "bad_json",
        "empty_choices",
        "bad_choice",
        "length",
    ],
)
def test_api_failures_do_not_leak_response_secrets_or_retry(failure, monkeypatch):
    secret = "sensitive-key-that-must-not-escape"
    monkeypatch.setenv("BIGMODEL_API_KEY", secret)
    calls = []

    def handle(request):
        calls.append(request)
        if failure == "http":
            return httpx.Response(401, text=secret)
        if failure == "timeout":
            raise httpx.ReadTimeout(secret, request=request)
        if failure == "connection":
            raise httpx.ConnectError(secret, request=request)
        if failure == "bad_json":
            return httpx.Response(200, text=secret)
        if failure == "empty_choices":
            return httpx.Response(200, json={"choices": []})
        if failure == "bad_choice":
            return httpx.Response(200, json={"choices": ["not a message"]})
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "length",
                        "message": {"content": 'do(action="Back")'},
                    }
                ]
            },
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await VisualFallback(client=client).decide(
                goal="返回",
                screenshot=_png(),
                ui_state=_ui(),
                available_tools=TOOLS,
            )

    result = asyncio.run(run())
    assert result.status == "error" and not result.calls
    assert secret not in repr(result)
    assert len(calls) == 1


def test_missing_key_or_corrupt_screenshot_fails_before_network(monkeypatch):
    monkeypatch.delenv("BIGMODEL_API_KEY", raising=False)
    monkeypatch.delenv("ZAI_API_KEY", raising=False)

    def unexpected_request(request):
        pytest.fail(
            "No request should be sent without credentials and a valid screenshot"
        )

    async def run():
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(unexpected_request)
        ) as client:
            fallback = VisualFallback(client=client)
            result = await fallback.decide(
                goal="返回", screenshot=_png(), ui_state=_ui(), available_tools=TOOLS
            )
            assert result.status == "error"
            monkeypatch.setenv("ZAI_API_KEY", "test-key")
            return await fallback.decide(
                goal="返回",
                screenshot=b"not-an-image",
                ui_state=_ui(),
                available_tools=TOOLS,
            )

    assert asyncio.run(run()).status == "error"


@pytest.mark.parametrize("model,json_mode", [("autoglm-phone", False), ("qwen-vl-max", True)])
def test_only_autoglm_action_models_use_the_do_protocol(monkeypatch, model, json_mode):
    monkeypatch.setenv("GLM_API_KEY", "test-key")
    requests = []
    reply = '{"action":"Back"}' if json_mode else 'do(action="Back")'

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(200, json={"choices": [{"message": {"content": reply}}]})

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await VisualFallback(
                client=client, api_key_env="GLM_API_KEY", model=model
            ).decide(
                goal="Go back",
                screenshot=_png(),
                ui_state=_ui(),
                available_tools={"system_button"},
            )

    assert asyncio.run(run()).status == "action"
    assert requests[0]["model"] == model
    assert ("response_format" in requests[0]) is json_mode
