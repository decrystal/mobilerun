"""Run Mobilerun's Jev-first FastAgent on a local Android device.

Examples (environment keys or hidden interactive prompts):
  python examples/jev_phone.py --e2e
  python examples/jev_phone.py --goal "打开设置"
The E2E scenarios navigate Settings only; they do not toggle settings or send messages.
"""

from __future__ import annotations

import argparse
import asyncio
import getpass
import json
import os
import subprocess
import sys
import time
import uuid
import xml.etree.ElementTree as ET
from collections import Counter
from datetime import datetime
from pathlib import Path


class DurableTrace(list):
    """Preserve decisions even if device capture or the process later fails."""

    def __init__(self, path):
        super().__init__()
        self.path = path

    def append(self, entry):
        super().append(entry)
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(entry, ensure_ascii=False) + "\n")
        if entry.get("event") == "progress":
            temporary = self.path.with_name("progress.tmp")
            temporary.write_text(
                json.dumps(entry, ensure_ascii=False, indent=2), encoding="utf-8"
            )
            temporary.replace(self.path.with_name("progress.json"))


def adb_command(args, *command):
    result = subprocess.run(
        [args.adb, "-s", args.serial, *command], capture_output=True, timeout=30
    )
    if result.returncode:
        raise RuntimeError(
            "ADB command failed: " + result.stderr.decode("utf-8", errors="replace")
        )
    return result.stdout


def read_xml(args):
    remote = "/sdcard/mobilerun-jev-" + uuid.uuid4().hex + ".xml"
    try:
        adb_command(args, "shell", "uiautomator", "dump", remote)
        raw = adb_command(args, "exec-out", "cat", remote).decode(
            "utf-8", errors="replace"
        )
    finally:
        adb_command(args, "shell", "rm", "-f", remote)
    start = raw.find("<?xml")
    if start < 0:
        start = raw.find("<hierarchy")
    end = raw.find("</hierarchy>")
    if start < 0 or end < 0:
        raise RuntimeError("ADB did not return a complete independent UI hierarchy")
    xml = raw[start : end + len("</hierarchy>")]
    return xml, list(ET.fromstring(xml).iter("node"))


def verify_xml(nodes, criteria):
    packages = {node.get("package") for node in nodes}
    texts = [node.get(attr, "") for node in nodes for attr in ("text", "content-desc")]
    if criteria.get("package") and criteria["package"] not in packages:
        return False
    if not all(
        any(expected in value for value in texts)
        for expected in criteria.get("text_all", [])
    ):
        return False
    if criteria.get("input_text") and not any(
        node.get("class", "").split(".")[-1]
        in {"EditText", "AutoCompleteTextView", "MultiAutoCompleteTextView"}
        and node.get("text") == criteria["input_text"]
        for node in nodes
    ):
        return False
    return bool(criteria)


async def capture_snapshot(driver, directory, prefix):
    """Capture native state before screenshot/UiAutomation can affect windows."""
    from mobilerun.tools.ui.provider import require_fresh_observation

    errors = []
    try:
        raw_ui = await asyncio.wait_for(driver.get_ui_tree(), timeout=10)
        (directory / f"{prefix}-ui.json").write_text(
            json.dumps(raw_ui, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        require_fresh_observation(raw_ui)
    except Exception as exc:
        errors.append(f"{prefix}_portal_state: " + type(exc).__name__)
    try:
        (directory / f"{prefix}.png").write_bytes(
            await asyncio.wait_for(driver.screenshot(), timeout=10)
        )
    except Exception as exc:
        errors.append(f"{prefix}_screenshot: " + type(exc).__name__)
    return errors


async def execute_case(
    args, name, goal, criteria, texts, directory, *, vision_only=False
):
    import httpx
    from llama_index.llms.openai_like import OpenAILike

    from mobilerun.agent.decision.autoglm import AutoGLMFallback
    from mobilerun.agent.droid.droid_agent import MobileAgent
    from mobilerun.config_manager import MobileConfig
    from mobilerun.tools.driver.adb_keyboard import AdbKeyboardAndroidDriver

    skip_xml = getattr(args, "skip_xml_artifact", False)
    if skip_xml and (args.e2e or criteria):
        raise ValueError(
            "Cannot skip XML when E2E or independent success assertions require it"
        )

    directory.mkdir(parents=True, exist_ok=False)
    if args.e2e:
        # Fixture resets belong only to explicit E2E runs.
        adb_command(args, "shell", "am", "force-stop", "com.android.settings")
        adb_command(
            args, "shell", "am", "force-stop", "com.android.settings.intelligence"
        )
        adb_command(args, "shell", "input", "keyevent", "3")
        if vision_only:
            adb_command(
                args, "shell", "am", "start", "-n", "com.android.settings/.Settings"
            )
        await asyncio.sleep(0.8)
        initial_xml, initial_nodes = await asyncio.to_thread(read_xml, args)
        (directory / "before.xml").write_text(initial_xml, encoding="utf-8")
        if verify_xml(initial_nodes, criteria):
            raise RuntimeError(
                "E2E baseline already satisfies the goal; refusing a no-op success"
            )

    driver = AdbKeyboardAndroidDriver(serial=args.serial, portal_mode="required")
    await driver.connect()
    (directory / "before.png").write_bytes(await driver.screenshot())
    config = MobileConfig.from_dict(
        {
            "agent": {
                "reasoning": False,
                "max_steps": args.steps,
                "streaming": False,
                "vision_only": vision_only,
                "after_sleep_action": 0.15,
                "fast_agent": {"vision": True, "parallel_tools": False},
                "app_cards": {"enabled": True},
                "jev": {
                    "enabled": True,
                    "model": "jev-1.13.0",
                    "fallback": "autoglm",
                    "confidence_threshold": args.confidence,
                    "text_values": texts,
                    "completion": {} if vision_only else criteria,
                    **{name: False for name in (args.disable_capability or [])},
                    **(
                        {"reasoning_model": args.reasoning_model}
                        if args.reasoning_model
                        else {}
                    ),
                    **(
                        {
                            "fallback_thinking": (
                                None if args.thinking == "default" else args.thinking
                            )
                        }
                        if args.thinking is not None
                        else {}
                    ),
                    **(
                        {"fallback_model": args.fallback_model}
                        if args.fallback_model
                        else {}
                    ),
                    **(
                        {"fallback_base_url": args.fallback_base_url}
                        if args.fallback_base_url
                        else {}
                    ),
                    **(
                        {"fallback_api_key_env": args.fallback_api_key_env}
                        if args.fallback_api_key_env
                        else {}
                    ),
                },
            },
            "device": {
                "serial": args.serial,
                "auto_setup": False,
                "portal_mode": "required",
            },
            "logging": {"save_trajectory": "none"},
            "tracing": {"enabled": False},
            "telemetry": {"enabled": False},
            "tools": {"disabled_tools": []},
        }
    )
    # Constructing the existing LLM wrapper performs no network call. In autoglm
    # fallback mode the dedicated adapter converts the phone action protocol.
    llm = OpenAILike(
        model=config.agent.jev.fallback_model,
        api_base=config.agent.jev.fallback_base_url.removesuffix(
            "/chat/completions"
        ).rstrip("/"),
        api_key=os.environ[config.agent.jev.fallback_api_key_env],
        is_chat_model=True,
    )

    async def external_verifier(_):
        _, nodes = await asyncio.to_thread(read_xml, args)
        return verify_xml(nodes, criteria)

    response_count = 0
    calls_by_model = Counter()

    async def record_model_request(request):
        payload = json.loads(request.content)
        calls_by_model[payload["model"]] += 1
        text_chars = sum(
            (
                len(message["content"])
                if isinstance(message.get("content"), str)
                else sum(
                    len(part.get("text", ""))
                    for part in message.get("content", [])
                    if isinstance(part, dict)
                )
            )
            for message in payload.get("messages", [])
        )
        agent.decision_trace.append(
            {
                "event": "model_request",
                "model": payload["model"],
                "thinking": payload.get("thinking"),
                "text_chars": text_chars,
                "images": sum(
                    part.get("type") == "image_url"
                    for message in payload.get("messages", [])
                    for part in (
                        message.get("content", [])
                        if isinstance(message.get("content"), list)
                        else []
                    )
                    if isinstance(part, dict)
                ),
            }
        )

    async def record_model_response(response):
        nonlocal response_count
        await response.aread()
        response_count += 1
        body = response.text
        for name in (config.agent.jev.fallback_api_key_env, "TYPESAFE_API_KEY"):
            secret = os.environ.get(name)
            if secret:
                body = body.replace(secret, "[REDACTED]")
        (directory / f"autoglm-response-{response_count:03d}.json").write_text(
            body, encoding="utf-8"
        )

    fallback_client = httpx.AsyncClient(
        event_hooks={
            "request": [record_model_request],
            "response": [record_model_response],
        }
    )

    agent = MobileAgent(
        goal,
        config=config,
        llms=llm,
        driver=driver,
        completion_verifier=external_verifier if vision_only and criteria else None,
        visual_fallback=AutoGLMFallback(
            model=config.agent.jev.fallback_model,
            base_url=config.agent.jev.fallback_base_url,
            api_key_env=config.agent.jev.fallback_api_key_env,
            thinking_mode=config.agent.jev.fallback_thinking,
            client=fallback_client,
            timeout=45,
            max_history_turns=0,
            reasoning_model=config.agent.jev.reasoning_model,
        ),
        timeout=args.timeout + 30,
    )
    agent.decision_trace = DurableTrace(directory / "trace.jsonl")
    started = time.monotonic()
    result = None
    error = None
    artifact_errors = []
    try:
        result = await asyncio.wait_for(agent.run(), timeout=args.timeout)
    except Exception as exc:
        error = {"type": type(exc).__name__, "message": str(exc)}
    finally:
        elapsed = round(time.monotonic() - started, 3)
        try:
            artifact_errors.extend(
                await capture_snapshot(driver, directory, "completion")
            )
        finally:
            try:
                input_method_restored = await driver.restore_input_method()
            finally:
                await fallback_client.aclose()
    (directory / "trace.json").write_text(
        json.dumps(agent.decision_trace, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    final_nodes = []
    # Snapshot Portal before attaching UiAutomation for the independent XML
    # dump: on LDPlayer, that probe can omit/invalidate floating-window nodes.
    artifact_errors.extend(await capture_snapshot(driver, directory, "after"))
    if not skip_xml:
        try:
            final_xml, final_nodes = await asyncio.to_thread(read_xml, args)
            (directory / "after.xml").write_text(final_xml, encoding="utf-8")
        except Exception as exc:
            artifact_errors.append("independent_xml: " + type(exc).__name__)
    decisions = Counter(
        row["source"] for row in agent.decision_trace if row["event"] == "decision"
    )
    api_calls = {
        "jev": sum(
            row.get("source") == "jev" and row.get("api_called", False)
            for row in agent.decision_trace
            if row["event"] == "decision"
        ),
        "autoglm": (
            calls_by_model[config.agent.jev.fallback_model]
            if config.agent.jev.fallback_model != config.agent.jev.reasoning_model
            else 0
        ),
        "reasoning": calls_by_model[config.agent.jev.reasoning_model],
        "llm": decisions["llm"],
    }
    actions = [
        row
        for row in agent.decision_trace
        if row["event"] == "action" and row["tool"] != "complete"
    ]
    independent = verify_xml(final_nodes, criteria) if criteria else None
    report = {
        "provider": {
            "base_url": config.agent.jev.fallback_base_url,
            "api_key_env": config.agent.jev.fallback_api_key_env,
            "action_model": config.agent.jev.fallback_model,
            "reasoning_model": config.agent.jev.reasoning_model,
            "thinking": config.agent.jev.fallback_thinking,
        },
        "capabilities": {
            name: getattr(config.agent.jev, name)
            for name in (
                "query_memory",
                "input_repair",
                "semantic_recovery",
                "review_history_images",
            )
        },
        "scenario": name,
        "goal": goal,
        "criteria": criteria,
        "vision_only": vision_only,
        "model_result": (
            result.model_dump() if hasattr(result, "model_dump") else str(result)
        ),
        "error": error,
        "elapsed_seconds": elapsed,
        "independent_xml_verified": independent,
        "artifact_errors": artifact_errors,
        "xml_artifact_skipped": skip_xml,
        "input_method_restored": input_method_restored,
        "api_calls": api_calls,
        "api_calls_by_model": {
            config.agent.jev.model: api_calls["jev"],
            **calls_by_model,
        },
        "decisions": dict(decisions),
        "executed_actions": len(actions),
        "completion_basis": "agent_with_assertions" if criteria else "agent_assessment",
        "success": bool(
            result
            and result.success
            and not error
            and (independent if criteria else True)
        ),
    }
    if args.e2e and vision_only:
        report["fallback_exercised"] = decisions["autoglm"] > 0
        report["success"] = report["success"] and report["fallback_exercised"]
    elif args.e2e:
        report["jev_exercised"] = decisions["jev"] > 0 and any(
            row["source"] == "jev" for row in actions
        )
        report["success"] = report["success"] and report["jev_exercised"]
    if args.e2e:
        report["success"] = report["success"] and bool(actions)
    if args.e2e and texts:
        report["input_exercised"] = any(
            row["tool"] == "type"
            and row["success"]
            and row["arguments"].get("text") in texts
            for row in actions
        )
        report["input_verified"] = any(
            row["event"] == "input_verification" and row.get("success")
            for row in agent.decision_trace
        )
        report["success"] = (
            report["success"] and report["input_exercised"] and report["input_verified"]
        )
    (directory / "result.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print("CASE_RESULT " + json.dumps(report, ensure_ascii=False), flush=True)
    return report


async def main(args):
    destination = Path(
        args.output
        or (".artifacts/jev-e2e" if args.e2e else ".artifacts/android-agent")
    ) / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    destination.mkdir(parents=True, exist_ok=False)
    cases = [
        (
            "navigation",
            "打开系统设置，进入网络和互联网里的WLAN（互联网）页面，只查看，不切换任何开关。",
            {"package": "com.android.settings", "text_all": ["WLAN", "添加网络"]},
            [],
            False,
        ),
        (
            "chinese-input",
            "打开系统设置，在设置搜索框中输入蓝牙，停留在搜索结果页面，不点击任何结果。",
            {
                "package": "com.android.settings.intelligence",
                "input_text": "蓝牙",
                "text_all": ["已连接的设备"],
            },
            ["蓝牙"],
            False,
        ),
        (
            "visual-fallback",
            "进入系统设置的显示（亮度）页面，停留查看，不修改任何设置。",
            {"package": "com.android.settings", "text_all": ["屏幕超时", "锁定屏幕"]},
            [],
            True,
        ),
    ]
    if not args.e2e:
        if not args.goal or not args.goal.strip():
            raise ValueError(
                "Provide a nonempty --goal, or use --e2e for the test scenarios"
            )
        criteria = {}
        if args.success_package:
            criteria["package"] = args.success_package
        if args.success_text:
            criteria["text_all"] = args.success_text
        cases = [("custom", args.goal, criteria, args.text, False)]
    elif args.scenario != "all":
        cases = [case for case in cases if case[0] == args.scenario]
    reports = []
    for name, goal, criteria, texts, vision_only in cases:
        print("CASE_START " + name, flush=True)
        reports.append(
            await execute_case(
                args,
                name,
                goal,
                criteria,
                texts,
                destination / name,
                vision_only=vision_only,
            )
        )
    summary = {
        "directory": str(destination.resolve()),
        "success": all(report["success"] for report in reports),
        "cases": reports,
    }
    (destination / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    print(
        ("E2E_SUMMARY " if args.e2e else "RUN_SUMMARY ")
        + json.dumps(summary, ensure_ascii=False),
        flush=True,
    )
    return 0 if summary["success"] else 1


def build_parser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--e2e", action="store_true")
    parser.add_argument(
        "--scenario",
        choices=["all", "navigation", "chinese-input", "visual-fallback"],
        default="all",
    )
    parser.add_argument(
        "--goal", help="Natural-language task; required unless --e2e is used."
    )
    parser.add_argument(
        "--success-package",
        help="Optional additional assertion for the final foreground app.",
    )
    parser.add_argument(
        "--success-text",
        action="append",
        default=[],
        help="Optional additional visible-text assertion (repeatable).",
    )
    parser.add_argument(
        "--text",
        action="append",
        default=[],
        help="Optional exact text candidates; omitted values are resolved by the model when needed.",
    )
    parser.add_argument("--steps", type=int, default=60)
    parser.add_argument(
        "--timeout", type=float, default=600, help="Task time budget in seconds."
    )
    parser.add_argument("--confidence", type=float, default=0.5)
    parser.add_argument(
        "--reasoning-model",
        help="Optional visual reasoning model for text arguments, recovery and completion review.",
    )
    parser.add_argument("--fallback-model", help="Optional visual action model.")
    parser.add_argument(
        "--disable-capability",
        action="append",
        choices=[
            "query_memory",
            "input_repair",
            "semantic_recovery",
            "review_history_images",
        ],
        help="Disable one capability for a controlled evaluation; repeat for additive stages.",
    )
    parser.add_argument(
        "--thinking",
        choices=["enabled", "disabled", "default"],
        help="Visual model thinking mode; default uses the provider's own default.",
    )
    parser.add_argument(
        "--fallback-base-url",
        help="OpenAI-compatible base URL or chat/completions URL.",
    )
    parser.add_argument(
        "--fallback-api-key-env",
        help="Dedicated provider credential environment variable.",
    )
    parser.add_argument("--serial", default="emulator-5554")
    parser.add_argument("--adb", default=r"C:\leidian\LDPlayer14\adb.exe")
    parser.add_argument(
        "--output", help="Optional directory for results, trace and progress."
    )
    parser.add_argument(
        "--skip-xml-artifact",
        action="store_true",
        help="Skip optional UiAutomation XML capture when an external benchmark owns grading. Incompatible with E2E or success assertions.",
    )
    return parser


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")
    parser = build_parser()
    args = parser.parse_args()
    if not args.e2e and not (args.goal and args.goal.strip()):
        parser.error("Provide --goal or --e2e.")
    if args.steps <= 0 or not 0 < args.timeout <= 86400:
        parser.error("--steps must be positive and --timeout must be in (0, 86400].")
    os.environ["PATH"] = (
        str(Path(args.adb).parent) + os.pathsep + os.environ.get("PATH", "")
    )
    from mobilerun.config_manager.config_manager import JevConfig

    key_env = args.fallback_api_key_env or JevConfig().fallback_api_key_env
    for name in ("TYPESAFE_API_KEY", key_env):
        if (
            name == "BIGMODEL_API_KEY"
            and not os.environ.get(name)
            and os.environ.get("ZAI_API_KEY")
        ):
            os.environ[name] = os.environ["ZAI_API_KEY"]
        if not os.environ.get(name):
            if not sys.stdin.isatty():
                parser.error(f"Set {name} in the environment for noninteractive runs.")
            os.environ[name] = (
                getpass.getpass(name + " (hidden): ").strip().replace("\\_", "_")
            )
    raise SystemExit(asyncio.run(main(args)))
