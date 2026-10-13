"""The everyday runner accepts a goal and preserves the current device state."""

import asyncio
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

spec = importlib.util.spec_from_file_location(
    "jev_phone_example", Path(__file__).parents[1] / "examples" / "jev_phone.py"
)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def test_goal_only_is_a_valid_custom_task(tmp_path, monkeypatch):
    args = runner.build_parser().parse_args(
        ["--goal", "Open settings", "--output", str(tmp_path)]
    )
    execute = AsyncMock(return_value={"success": True})
    monkeypatch.setattr(runner, "execute_case", execute)
    assert asyncio.run(runner.main(args)) == 0
    assert execute.await_args.args[2:5] == ("Open settings", {}, [])


def test_project_defaults_select_relay_for_both_model_roles():
    from mobilerun.config_manager.config_manager import JevConfig

    config = JevConfig()
    assert config.fallback_model == config.reasoning_model == "glm-5.3-flash"
    assert config.fallback_api_key_env == "GLM_API_KEY"
    assert config.fallback_base_url == "http://66.154.100.178:8080/v1"
    assert config.fallback_thinking == "disabled"


def test_provider_overrides_reach_cli_configuration(tmp_path, monkeypatch):
    from mobilerun.config_manager import MobileConfig
    from mobilerun.tools.driver.adb_keyboard import AdbKeyboardAndroidDriver

    args = runner.build_parser().parse_args(
        [
            "--goal",
            "Open settings",
            "--fallback-base-url",
            "https://provider.example/v1",
            "--fallback-api-key-env",
            "CUSTOM_KEY",
            "--fallback-model",
            "action-model",
            "--reasoning-model",
            "review-model",
            "--disable-capability",
            "query_memory",
            "--disable-capability",
            "review_history_images",
        ]
    )
    monkeypatch.setattr(AdbKeyboardAndroidDriver, "connect", AsyncMock())
    monkeypatch.setattr(
        AdbKeyboardAndroidDriver, "screenshot", AsyncMock(return_value=b"image")
    )

    def capture(config):
        selected = config["agent"]["jev"]
        assert selected["fallback_base_url"] == "https://provider.example/v1"
        assert selected["fallback_api_key_env"] == "CUSTOM_KEY"
        assert selected["fallback_model"] == "action-model"
        assert selected["reasoning_model"] == "review-model"
        assert (
            selected["query_memory"] is False
            and selected["review_history_images"] is False
        )
        raise RuntimeError("configuration captured")

    monkeypatch.setattr(MobileConfig, "from_dict", capture)
    with pytest.raises(RuntimeError, match="configuration captured"):
        asyncio.run(
            runner.execute_case(args, "custom", args.goal, {}, [], tmp_path / "run")
        )


def test_custom_task_does_not_reset_apps_or_require_a_baseline_assertion(
    tmp_path, monkeypatch
):
    from mobilerun.tools.driver.adb_keyboard import AdbKeyboardAndroidDriver

    args = runner.build_parser().parse_args(["--goal", "Open settings"])

    def reject_reset(*args, **kwargs):
        raise AssertionError("Custom task must start from the user's existing screen")

    monkeypatch.setattr(runner, "adb_command", reject_reset)
    monkeypatch.setattr(
        AdbKeyboardAndroidDriver,
        "connect",
        AsyncMock(side_effect=RuntimeError("reached driver connection")),
    )
    with pytest.raises(RuntimeError, match="reached driver connection"):
        asyncio.run(
            runner.execute_case(args, "custom", args.goal, {}, [], tmp_path / "run")
        )


def test_progress_snapshot_is_persisted_alongside_trace(tmp_path):
    trace = runner.DurableTrace(tmp_path / "trace.jsonl")
    trace.append(
        {"event": "progress", "executed_actions": 2, "progress_note": "Continue search"}
    )
    trace.append(
        {"event": "progress", "executed_actions": 3, "progress_note": "Results visible"}
    )
    assert json.loads((tmp_path / "progress.json").read_text())["executed_actions"] == 3
    assert len((tmp_path / "trace.jsonl").read_text().splitlines()) == 2
    assert not (tmp_path / "progress.tmp").exists()


@pytest.mark.parametrize("model", [None, "configured-review-model"])
def test_cli_reasoning_model_reaches_agent_configuration(tmp_path, monkeypatch, model):
    from mobilerun.config_manager import MobileConfig
    from mobilerun.tools.driver.adb_keyboard import AdbKeyboardAndroidDriver

    argv = ["--goal", "Open settings"]
    if model:
        argv.extend(["--reasoning-model", model])
    args = runner.build_parser().parse_args(argv)
    monkeypatch.setattr(AdbKeyboardAndroidDriver, "connect", AsyncMock())
    monkeypatch.setattr(
        AdbKeyboardAndroidDriver, "screenshot", AsyncMock(return_value=b"image")
    )

    def capture(config):
        selected = config["agent"]["jev"]
        if model:
            assert selected["reasoning_model"] == model
        else:
            assert "reasoning_model" not in selected
        raise RuntimeError("configuration captured")

    monkeypatch.setattr(MobileConfig, "from_dict", capture)
    with pytest.raises(RuntimeError, match="configuration captured"):
        asyncio.run(
            runner.execute_case(args, "custom", args.goal, {}, [], tmp_path / "run")
        )


@pytest.mark.parametrize("capture_fails", [False, True])
@pytest.mark.parametrize("skip_xml", [False, True])
def test_completion_snapshot_precedes_cleanup_and_capture_failure_still_restores_ime(
    tmp_path, monkeypatch, capture_fails, skip_xml
):
    import mobilerun.agent.droid.droid_agent as agent_module
    import mobilerun.tools.driver.adb_keyboard as driver_module

    restored = False
    calls = []

    async def snapshot():
        calls.append("after" if restored else "completion")
        if capture_fails and not restored:
            raise RuntimeError("snapshot failed")
        return {"a11y_tree": {"text": "After cleanup" if restored else "Copy"}}

    async def restore():
        nonlocal restored
        calls.append("restore")
        restored = True
        return True

    driver = SimpleNamespace(
        connect=AsyncMock(),
        get_ui_tree=snapshot,
        screenshot=AsyncMock(return_value=b"screenshot"),
        restore_input_method=restore,
    )
    from mobilerun.agent.droid.events import ResultEvent

    result = ResultEvent(success=True, reason="Verified", steps=1, final_answer="14:00")
    agent = SimpleNamespace(run=AsyncMock(return_value=result))
    monkeypatch.setenv("GLM_API_KEY", "test-key")
    monkeypatch.setattr(
        driver_module, "AdbKeyboardAndroidDriver", lambda **kwargs: driver
    )
    monkeypatch.setattr(agent_module, "MobileAgent", lambda *args, **kwargs: agent)
    xml_capture = Mock(return_value=("<hierarchy/>", []))
    monkeypatch.setattr(runner, "read_xml", xml_capture)
    args = runner.build_parser().parse_args(["--goal", "Open the Copy menu"])
    args.skip_xml_artifact = skip_xml
    destination = tmp_path / "run"

    report = asyncio.run(
        runner.execute_case(args, "custom", args.goal, {}, [], destination)
    )

    assert report["model_result"]["final_answer"] == "14:00"
    assert (
        json.loads((destination / "result.json").read_text())["model_result"][
            "final_answer"
        ]
        == "14:00"
    )
    assert calls == ["completion", "restore", "after"]
    assert report["input_method_restored"] is True
    assert report["xml_artifact_skipped"] is skip_xml
    assert xml_capture.called is not skip_xml
    assert (
        json.loads((destination / "after-ui.json").read_text())["a11y_tree"]["text"]
        == "After cleanup"
    )
    if capture_fails:
        assert report["artifact_errors"] == ["completion_portal_state: RuntimeError"]
    else:
        assert (
            json.loads((destination / "completion-ui.json").read_text())["a11y_tree"][
                "text"
            ]
            == "Copy"
        )
        assert report["artifact_errors"] == []


def test_degraded_snapshot_is_preserved_but_reported_as_an_evidence_error(tmp_path):
    raw = {"degraded": True, "snapshotAgeMs": 10000, "a11y_tree": {"text": "Copy"}}
    driver = SimpleNamespace(
        get_ui_tree=AsyncMock(return_value=raw),
        screenshot=AsyncMock(return_value=b"current screenshot"),
    )
    errors = asyncio.run(runner.capture_snapshot(driver, tmp_path, "completion"))
    assert errors == ["completion_portal_state: StaleObservationError"]
    assert json.loads((tmp_path / "completion-ui.json").read_text()) == raw
    assert (tmp_path / "completion.png").read_bytes() == b"current screenshot"


def test_case_summary_lists_outcome_time_steps_and_effects():
    report = {
        "success": False,
        "error": None,
        "model_result": {"reason": "E2 could not be verified\nSide effects: ..."},
        "elapsed_seconds": 75.25,
        "api_calls_by_model": {"jev-1.13.0": 4, "glm-5.3-flash": 5},
    }
    actions = [
        {"source": "jev", "tool": "launch_app", "success": True, "summary": "App started"},
        {"source": "autoglm", "tool": "click", "success": False, "summary": "Clicked  儲存"},
    ]
    trace = [
        {"event": "side_effect", "id": "E1", "expected_effect": "保存图片", "status": "pending"},
        {"event": "effect_verification", "effect": "E1", "status": "probing"},
        {"event": "effect_verification", "effect": "E1", "status": "confirmed"},
    ]
    text = runner.format_summary(report, actions, trace)
    assert "结果：失败（E2 could not be verified）" in text
    assert "总耗时：75.2 秒" in text or "总耗时：75.3 秒" in text
    assert "执行步骤（2 步）" in text
    assert "   1. [jev] launch_app  App started" in text
    assert "   2. [visual] click ✗  Clicked 儲存" in text
    assert "E1 已确认  保存图片" in text
    assert "模型调用：jev-1.13.0 4 次，glm-5.3-flash 5 次" in text
