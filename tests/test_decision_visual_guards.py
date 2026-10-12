"""Visual freshness and progress guards use observations, not model claims."""

import asyncio
from io import BytesIO
from types import SimpleNamespace
from unittest.mock import AsyncMock

from PIL import Image

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.decision.autoglm import AutoGLMDecision, AutoGLMToolCall
from mobilerun.agent.decision.jev import JevDecision
from mobilerun.agent.decision.runtime import DecisionRuntime
from mobilerun.agent.fast_agent.xml_parser import ToolCall
from mobilerun.tools.ui.state import UIState


def _image(color):
    buffer = BytesIO()
    Image.new("RGB", (400, 800), color).save(buffer, format="PNG")
    return buffer.getvalue()


def _ui(nodes=(), *, page="Home", complete=False):
    return UIState(
        elements=list(nodes),
        formatted_text=page,
        focused_text="",
        phone_state={
            "packageName": "com.android.settings",
            "currentApp": page,
            "test_complete": complete,
        },
        screen_width=400,
        screen_height=800,
    )


class _Driver:
    def __init__(self):
        self.image = _image("red")

    async def get_apps(self, **_):
        return []

    async def screenshot(self):
        return self.image


class _Provider:
    def __init__(self, ui):
        self.ui = ui

    async def get_state(self):
        return self.ui


class _Policy:
    def __init__(self, results=()):
        self.results = list(results)

    async def decide(self, **_):
        return (
            self.results.pop(0)
            if self.results
            else JevDecision("fallback", reason="no_candidates")
        )


class _Visual:
    def __init__(self, result=None):
        self.result = result or AutoGLMDecision(
            "action", calls=(AutoGLMToolCall("click_at", {"x": 100, "y": 200}),)
        )
        self.calls = 0

    async def decide(self, **_):
        self.calls += 1
        return self.result


def _runtime(ui=None, *, policy=None, visual=None, verifier=None, provider=None):
    ui = ui or _ui()
    driver = _Driver()
    provider = provider or _Provider(ui)
    config = SimpleNamespace(
        completion={},
        model="jev-1.13.0",
        timeout=1,
        confidence_threshold=0.5,
        fallback="autoglm",
        fallback_model="autoglm-phone",
        fallback_timeout=1,
        text_values=[],
    )
    context = SimpleNamespace(driver=driver, ui=ui)
    registry = SimpleNamespace(
        tools={
            name: object()
            for name in ("click", "click_at", "system_button", "complete")
        }
    )
    runtime = DecisionRuntime(
        config,
        context,
        registry,
        provider,
        policy=policy or _Policy(),
        visual_fallback=visual or _Visual(),
        verifier=verifier,
    )
    return runtime, driver, provider


def test_changed_visual_frame_rejects_coordinates_even_when_ui_tree_is_empty():
    async def run():
        runtime, driver, _ = _runtime()
        selected = await runtime.select("打开显示页面", runtime.ctx.ui)
        assert selected.source == "autoglm"
        assert selected.visual_fingerprint
        driver.image = _image("blue")

        _, reason = await runtime.validate_before_execute()

        assert reason == "stale_visual_frame"
        assert runtime.history == []
        assert not runtime.executed
        assert runtime.trace[-1]["event"] == "stale"

    asyncio.run(run())


def test_unchanged_visual_frame_allows_the_selected_action():
    async def run():
        runtime, _, _ = _runtime()
        await runtime.select("打开显示页面", runtime.ctx.ui)
        _, reason = await runtime.validate_before_execute()
        assert reason is None

    asyncio.run(run())


def test_cached_native_state_cannot_relax_visual_navigation_freshness():
    runtime, driver, _ = _runtime()
    raw = {"phone_state": {"packageName": "com.android.settings"}, "a11y_tree": {}}
    driver.get_ui_tree = AsyncMock(return_value=raw)
    assert asyncio.run(runtime.navigation_context()) is not None
    raw["degraded"] = True
    assert asyncio.run(runtime.navigation_context()) is None


def test_coordinate_action_ignores_remote_text_but_rejects_target_changes_and_overlays():
    async def run():
        target = {
            "index": 1,
            "text": "Browser",
            "bounds": "50,100,200,300",
            "clickable": True,
        }
        banner = {"index": 2, "text": "Ad A", "bounds": "0,0,400,80"}
        initial = _ui([target, banner])
        runtime, _, provider = _runtime(initial)
        selected = await runtime.select("Open the browser", initial)
        assert selected.coordinate_guard

        provider.ui = _ui([target, {**banner, "text": "Ad B"}])
        _, reason = await runtime.validate_before_execute()
        assert reason is None

        for nodes in (
            [{**target, "text": "Delete"}, banner],
            [{**target, "bounds": "50,300,200,500"}, banner],
            [target, banner, {"index": 3, "text": "Confirm", "bounds": "0,0,400,800"}],
        ):
            provider.ui = _ui(nodes)
            _, reason = await runtime.validate_before_execute()
            assert reason == "stale_fallback_target"
        assert runtime.history == []

    asyncio.run(run())


def test_same_coordinates_are_reusable_after_visual_page_changes():
    async def run():
        runtime, driver, _ = _runtime()
        first = await runtime.select("先进入显示，再进入亮度", runtime.ctx.ui)
        runtime.record_execution(
            first.calls[0], ActionResult(True, "Opened display"), runtime.ctx.ui
        )
        assert runtime.executed

        driver.image = _image("blue")
        next_selection = await runtime.select("先进入显示，再进入亮度", runtime.ctx.ui)

        assert next_selection.calls == first.calls
        assert not next_selection.stop
        assert not runtime.executed

    asyncio.run(run())


def test_swipe_region_ignores_remote_banner_but_rejects_overlay_across_path():
    async def run():
        container = {"index": 1, "className": "FrameLayout", "bounds": "0,0,400,800"}
        banner = {"index": 2, "text": "Ad A", "bounds": "0,0,400,80"}
        initial = _ui([container, banner])
        visual = _Visual(
            AutoGLMDecision(
                "action",
                calls=(
                    AutoGLMToolCall(
                        "swipe", {"coordinate": [350, 400], "coordinate2": [50, 400]}
                    ),
                ),
            )
        )
        runtime, _, provider = _runtime(initial, visual=visual)
        runtime.registry.tools["swipe"] = object()
        selected = await runtime.select("Swipe left", initial)
        assert selected.coordinate_guard
        provider.ui = _ui([container, {**banner, "text": "Ad B"}])
        _, reason = await runtime.validate_before_execute()
        assert reason is None
        provider.ui.elements.append(
            {"index": 3, "text": "Dialog", "bounds": "150,300,250,500"}
        )
        _, reason = await runtime.validate_before_execute()
        assert reason == "stale_fallback_target"
        assert runtime.history == []

    asyncio.run(run())


def test_typing_ignores_remote_ads_but_rejects_changed_input_value():
    async def run():
        field = {
            "index": 1,
            "className": "EditText",
            "resourceId": "app:id/url",
            "bounds": "0,100,400,200",
            "focused": True,
            "inputValue": "old",
        }
        banner = {"index": 2, "text": "Ad A", "bounds": "0,300,400,400"}
        initial = _ui([field, banner])
        runtime, _, provider = _runtime(
            initial,
            visual=_Visual(
                AutoGLMDecision(
                    "action",
                    calls=(AutoGLMToolCall("type", {"text": "new", "clear": True}),),
                )
            ),
        )
        runtime.registry.tools["type"] = object()
        selected = await runtime.select("Type new in the address field", initial)
        assert selected.input_guard
        provider.ui = _ui([field, {**banner, "text": "Ad B"}])
        assert (await runtime.validate_before_execute())[1] is None
        provider.ui = _ui([{**field, "inputValue": "user edit"}, banner])
        assert (await runtime.validate_before_execute())[
            1
        ] == "input_focus_or_value_changed"

    asyncio.run(run())


def test_same_coordinates_without_visual_progress_are_not_replayed():
    async def run():
        runtime, _, _ = _runtime()
        first = await runtime.select("打开显示页面", runtime.ctx.ui)
        runtime.record_execution(
            first.calls[0], ActionResult(True, "Tap sent"), runtime.ctx.ui
        )

        next_selection = await runtime.select("打开显示页面", runtime.ctx.ui)

        assert next_selection.stop
        assert next_selection.calls == []
        assert next_selection.reason == "Repeated action without observable progress"

    asyncio.run(run())


def test_successful_tree_navigation_can_return_and_open_the_same_control_again():
    async def run():
        home = _ui(
            [{"index": 2, "text": "显示", "bounds": "0,80,400,160", "clickable": True}]
        )
        display = _ui(
            [{"index": 2, "text": "亮度", "bounds": "0,80,400,160"}], page="Display"
        )
        policy = _Policy(
            [
                JevDecision("action", tool_call=ToolCall("click", {"index": 2})),
                JevDecision(
                    "action", tool_call=ToolCall("system_button", {"button": "back"})
                ),
                JevDecision("action", tool_call=ToolCall("click", {"index": 2})),
            ]
        )
        visual = _Visual()
        runtime, _, _ = _runtime(home, policy=policy, visual=visual)
        first = await runtime.select("进入显示、返回，再进入显示", home)
        runtime.record_execution(
            first.calls[0], ActionResult(True, "Opened display"), home
        )
        back = await runtime.select("进入显示、返回，再进入显示", display)
        runtime.record_execution(
            back.calls[0], ActionResult(True, "Returned home"), display
        )

        repeat = await runtime.select("进入显示、返回，再进入显示", home)

        assert repeat.source == "jev"
        assert repeat.calls == first.calls
        assert not repeat.stop
        assert visual.calls == 0

    asyncio.run(run())


def test_visual_done_rechecks_completion_after_inference():
    async def run():
        initial = _ui()
        completed = _ui(complete=True)

        class AdvancingProvider(_Provider):
            async def get_state(self):
                observed = self.ui
                self.ui = completed
                return observed

        checked = []

        def verifier(ui):
            checked.append(ui.phone_state["test_complete"])
            return checked[-1]

        runtime, _, _ = _runtime(
            initial,
            provider=AdvancingProvider(initial),
            verifier=verifier,
            visual=_Visual(AutoGLMDecision("done", message="页面已加载")),
        )

        selection = await runtime.select("等待页面加载完成", initial)

        assert checked == [True]
        assert selection.source == "verifier"
        assert not selection.stop
        assert selection.calls[0].name == "complete"
        assert selection.calls[0].parameters["success"] is True

    asyncio.run(run())


def test_changed_goal_cannot_complete_against_the_previous_goal_contract():
    async def run():
        before = _ui()
        old_goal_complete = _ui(complete=True)
        verification_calls = []

        def verifier(ui):
            verification_calls.append(ui)
            return ui.phone_state["test_complete"]

        runtime, _, _ = _runtime(before, verifier=verifier)
        initial = await runtime.select("打开WLAN", before)
        assert not initial.stop
        count_before_update = len(verification_calls)

        selected = await runtime.select(
            "打开WLAN\nUser update:\n再返回设置首页", old_goal_complete
        )

        assert selected.stop
        assert selected.calls == []
        assert selected.reason.startswith("completion_contract_changed")
        assert len(verification_calls) == count_before_update
        assert runtime.trace[-1] == {
            "event": "verification",
            "success": False,
            "reason": "completion_contract_changed",
        }

    asyncio.run(run())
