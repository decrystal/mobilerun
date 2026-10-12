"""Exercise declared model coordinates through the real registry conversion."""

import asyncio
import json
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from PIL import Image

from mobilerun.agent.decision.autoglm import AutoGLMFallback, parse_autoglm_decision
from mobilerun.agent.utils.signatures import build_tool_registry
from mobilerun.tools.ui.state import UIState


def ui(**overrides):
    values = {
        "elements": [],
        "formatted_text": "",
        "focused_text": "",
        "phone_state": {},
        "screen_width": 1080,
        "screen_height": 2400,
    }
    return UIState(**{**values, **overrides})


def parse(
    action,
    state=None,
    size=(1080, 2400),
    tools=("click_at", "long_press_at", "swipe", "click"),
):
    return parse_autoglm_decision(
        json.dumps(action),
        ui_state=state or ui(),
        screenshot_size=size,
        available_tools=tools,
        structured=True,
    )


@pytest.mark.parametrize(
    "state,size,point,expected,screenshot_only",
    [
        (ui(), (1080, 2400), [540, 1800], (540, 1800), False),
        (
            ui(
                coordinate_scale_x=2,
                coordinate_scale_y=2,
                coordinate_contract_active=True,
                model_screenshot_width=540,
                model_screenshot_height=1200,
            ),
            (540, 1200),
            [270, 900],
            (540, 1800),
            False,
        ),
        (
            ui(
                coordinate_scale_x=2,
                coordinate_scale_y=2,
                coordinate_contract_active=True,
                model_screenshot_width=540,
                model_screenshot_height=1200,
            ),
            (1080, 2400),
            [540, 1800],
            (540, 1800),
            False,
        ),
        (
            ui(
                screen_width=540,
                screen_height=1200,
                coordinate_scale_x=2,
                coordinate_scale_y=2,
                model_screenshot_width=540,
                model_screenshot_height=1200,
                phone_state={"observationMode": "screenshot_only"},
            ),
            (540, 1200),
            [270, 900],
            (540, 1800),
            True,
        ),
        (ui(use_normalized=True), (1080, 2400), [540, 1800], (540, 1800), False),
        (
            ui(
                screen_width=400,
                screen_height=800,
                coordinate_scale_x=0.4,
                coordinate_scale_y=0.4,
                coordinate_contract_active=True,
                model_screenshot_width=1000,
                model_screenshot_height=2000,
            ),
            (1000, 2000),
            [999, 1999],
            (399, 799),
            False,
        ),
    ],
)
def test_declared_pixels_reach_the_correct_device_point_once(
    state, size, point, expected, screenshot_only
):
    async def run():
        registry, _ = await build_tool_registry(screenshot_only=screenshot_only)
        result = parse(
            {
                "action": "Tap",
                "coordinate_space": "screenshot_pixels",
                "element": point,
            },
            state,
            size,
            registry.tools,
        )
        assert result.status == "action"
        driver = SimpleNamespace(tap=AsyncMock())
        ctx = SimpleNamespace(
            driver=driver,
            ui=state,
            macro_recorder=None,
            state_provider=SimpleNamespace(requires_coordinate_tools=screenshot_only),
        )
        call = result.calls[0]
        executed = await registry.execute(call.name, call.arguments, ctx)
        assert executed.success
        driver.tap.assert_awaited_once_with(*expected)

    asyncio.run(run())


def test_declared_swipe_uses_one_coordinate_space_for_both_endpoints():
    result = parse(
        {
            "action": "Swipe",
            "coordinate_space": "screenshot_pixels",
            "start": [200, 1800],
            "end": [200, 400],
        }
    )
    assert result.calls[0].arguments == {
        "coordinate": [200, 1800],
        "coordinate2": [200, 400],
        "duration": 1.0,
    }


@pytest.mark.parametrize(
    "action,size",
    [
        ({"action": "Tap", "element": [705, 1408]}, (1080, 2400)),
        (
            {
                "action": "Tap",
                "coordinate_space": "native_pixels",
                "element": [100, 200],
            },
            (1080, 2400),
        ),
        (
            {
                "action": "Tap",
                "coordinate_space": "screenshot_pixels",
                "element": [1080, 500],
            },
            (1080, 2400),
        ),
        (
            {
                "action": "Tap",
                "coordinate_space": "screenshot_pixels",
                "element": [50.5, 100],
            },
            (1080, 2400),
        ),
        (
            {
                "action": "Tap",
                "coordinate_space": "screenshot_pixels",
                "element": [True, 100],
            },
            (1080, 2400),
        ),
        (
            {
                "action": "Tap",
                "coordinate_space": "screenshot_pixels",
                "element": [-1, 100],
            },
            (1080, 2400),
        ),
        (
            {
                "action": "Tap",
                "coordinate_space": "screenshot_pixels",
                "element": [100, 200],
            },
            (2400, 1080),
        ),
        (
            {
                "action": "Type",
                "coordinate_space": "screenshot_pixels",
                "text": "hello",
            },
            (1080, 2400),
        ),
    ],
)
def test_ambiguous_units_invalid_points_and_rotated_images_never_dispatch(action, size):
    result = parse(action, size=size)
    assert result.status == "error" and not result.calls


def test_a_single_wrapper_preserves_the_action_but_not_a_batch_or_nested_wrapper():
    action = {
        "action": "Long Press",
        "coordinate_space": "screenshot_pixels",
        "element": [705, 1408],
    }
    assert parse([action]).calls == parse(action).calls
    for wrapper in ([], [action, {"action": "Back"}], [[action]], [None]):
        result = parse(wrapper)
        assert result.status == "error" and not result.calls


def test_native_index_executes_the_current_control_instead_of_a_guessed_pixel():
    async def run():
        state = ui(
            elements=[
                {
                    "index": 7,
                    "text": "Save",
                    "bounds": "610,1300,800,1516",
                    "clickable": True,
                }
            ]
        )
        registry, _ = await build_tool_registry()
        result = parse({"action": "Tap", "index": 7}, state, tools=registry.tools)
        assert result.status == "action"
        call = result.calls[0]
        assert call.name == "click" and call.arguments == {"index": 7}
        driver = SimpleNamespace(tap=AsyncMock())
        ctx = SimpleNamespace(driver=driver, ui=state, macro_recorder=None)
        executed = await registry.execute(call.name, call.arguments, ctx)
        assert executed.success
        driver.tap.assert_awaited_once_with(705, 1408)

    asyncio.run(run())


@pytest.mark.parametrize(
    "changes,index",
    [
        ({}, True),
        ({}, -1),
        ({}, 9),
        ({"visible": False}, 7),
        ({"enabled": False}, 7),
        ({"isEnabled": False}, 7),
        ({"bounds": "1100,2400,1200,2600"}, 7),
        ({"bounds": "not coordinates"}, 7),
    ],
)
def test_native_targets_must_be_current_visible_enabled_and_locatable(changes, index):
    state = ui(
        elements=[
            {"index": 7, "text": "Save", "bounds": "610,1300,800,1516", **changes}
        ]
    )
    result = parse({"action": "Tap", "index": index}, state)
    assert result.status == "error" and not result.calls


def test_native_target_requires_one_unambiguous_representation_and_available_tool():
    node = {"index": 7, "text": "Save", "bounds": "610,1300,800,1516"}
    state = ui(elements=[node])
    for action, tools in [
        ({"action": "Tap", "index": 7, "element": [700, 1400]}, ("click", "click_at")),
        ({"action": "Tap", "index": 7}, ("click_at",)),
        ({"action": "Long Press", "index": 7}, ("long_press_at",)),
    ]:
        result = parse(action, state, tools=tools)
        assert result.status == "error" and not result.calls
    assert (
        parse({"action": "Tap", "index": 7}, ui(elements=[node, node])).status
        == "error"
    )
    assert (
        parse(
            {"action": "Tap", "index": 7}, ui(elements=[node], use_normalized=True)
        ).status
        == "error"
    )


def test_native_tap_with_confirmation_still_never_dispatches():
    state = ui(elements=[{"index": 7, "text": "Submit", "bounds": "610,1300,800,1516"}])
    result = parse({"action": "Tap", "index": 7, "message": "Please confirm"}, state)
    assert result.status == "needs_input" and not result.calls


@pytest.mark.parametrize(
    "action,expected",
    [
        ({"action": "Tap", "index": 7}, ("click", {"index": 7})),
        (
            {
                "action": "Long Press",
                "coordinate_space": "screenshot_pixels",
                "element": [705, 1408],
            },
            ("long_press_at", {"x": 705, "y": 1408}),
        ),
    ],
)
def test_reasoning_request_declares_the_actual_frame_and_parses_one_wrapped_action(
    monkeypatch, action, expected
):
    monkeypatch.setenv("BIGMODEL_API_KEY", "test-key")
    requests = []

    def handle(request):
        requests.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {"content": json.dumps([action])},
                    }
                ]
            },
        )

    async def run():
        image = BytesIO()
        Image.new("RGB", (1080, 2400), "white").save(image, format="PNG")
        state = ui(
            elements=[{"index": 7, "text": "Save", "bounds": "610,1300,800,1516"}]
        )
        async with httpx.AsyncClient(transport=httpx.MockTransport(handle)) as client:
            return await AutoGLMFallback(
                client=client, reasoning_model="glm-4.6v"
            ).decide(
                goal="Save the current record",
                screenshot=image.getvalue(),
                ui_state=state,
                available_tools={"click", "long_press_at"},
                recovery=True,
            )

    result = asyncio.run(run())
    assert result.status == "action" and len(result.calls) == 1
    assert (result.calls[0].name, result.calls[0].arguments) == expected
    assert len(requests) == 1
    payload = requests[0]
    assert payload["model"] == "glm-4.6v"
    assert "screenshot_pixels" in payload["messages"][0]["content"]
    observation = payload["messages"][-1]["content"][0]["text"]
    contract = json.loads(observation.rsplit("动作坐标契约：", 1)[1])
    assert contract["screenshot_size"] == [1080, 2400]
    assert contract["native_index_available"] is True
    assert contract["default_if_omitted"] == "normalized_1000"
