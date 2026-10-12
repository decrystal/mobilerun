"""ADBKeyboard adaptation never retries input or routes text into Portal."""

import asyncio
import base64
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from mobilerun.tools.driver.adb_keyboard import (
    ADB_KEYBOARD_IME,
    ADB_KEYBOARD_PACKAGE,
    AdbKeyboardAndroidDriver,
)

ORIGINAL_IME = "com.android.inputmethod.pinyin/.InputService"


class FakeDevice:
    def __init__(
        self,
        *,
        ime=ORIGINAL_IME,
        available=True,
        error=None,
        block=False,
        ime_set_effective=True,
        secure_set_effective=True,
    ):
        self.ime = ime
        self.available = available
        self.error = error
        self.block = block
        self.ime_set_effective = ime_set_effective
        self.secure_set_effective = secure_set_effective
        self.commands = []

    async def get_state(self):
        return "device"

    async def shell(self, command, *, timeout):
        assert timeout > 0
        self.commands.append(list(command))
        if command == ["settings", "get", "secure", "default_input_method"]:
            return self.ime
        if command == ["ime", "list", "-s"]:
            return ORIGINAL_IME + ("\n" + ADB_KEYBOARD_IME if self.available else "")
        if command[:2] == ["ime", "set"]:
            if self.ime_set_effective:
                self.ime = command[2]
            return "Input method selected"
        if command[:4] == ["settings", "put", "secure", "default_input_method"]:
            if self.secure_set_effective:
                self.ime = command[4]
            return ""
        if command == ["dumpsys", "input_method"]:
            return f"mCurId={self.ime} mBoundToMethod=true"
        if command[:2] == ["am", "broadcast"]:
            if self.error:
                raise RuntimeError(self.error)
            if self.block:
                await asyncio.Event().wait()
            return "Broadcast completed: result=0"
        raise AssertionError("Unexpected command")


def driver(device=None, timeout=1):
    result = AdbKeyboardAndroidDriver(serial="offline", input_timeout=timeout)
    result.device = device or FakeDevice()
    result._connected = True
    result.portal = SimpleNamespace(
        input_text=AsyncMock(side_effect=AssertionError("Portal typing forbidden"))
    )
    return result


def test_connect_keeps_portal_control_without_switching_to_portal_ime():
    async def run():
        device = FakeDevice()
        portal = SimpleNamespace(
            connect=AsyncMock(), ping=AsyncMock(return_value={"status": "success"})
        )
        subject = AdbKeyboardAndroidDriver(serial="offline")
        with (
            patch(
                "mobilerun_core_local.driver.android.adb.adb.device",
                AsyncMock(return_value=device),
            ),
            patch(
                "mobilerun_core_local.driver.android.adb.PortalClient",
                return_value=portal,
            ),
        ):
            await subject.connect()
        assert subject.portal_available
        assert subject.portal is portal
        assert subject._portal_keyboard_available is False
        assert device.commands == []
        assert device.ime == ORIGINAL_IME

    asyncio.run(run())


def test_chinese_input_is_one_scoped_base64_broadcast_after_clear():
    async def run():
        subject = driver()
        text = "蓝牙 ' & $(echo not-executed) `test`\n北京"
        assert await subject.input_text(text, clear=True)
        broadcasts = [
            c for c in subject.device.commands if c[:2] == ["am", "broadcast"]
        ]
        assert broadcasts == [
            ["am", "broadcast", "-a", "ADB_CLEAR_TEXT", "-p", ADB_KEYBOARD_PACKAGE],
            [
                "am",
                "broadcast",
                "-a",
                "ADB_INPUT_B64",
                "-p",
                ADB_KEYBOARD_PACKAGE,
                "--es",
                "msg",
                base64.b64encode(text.encode("utf-8")).decode("ascii"),
            ],
        ]
        assert not subject.portal.input_text.called
        assert subject.device.ime == ADB_KEYBOARD_IME
        assert await subject.restore_input_method()
        assert subject.device.ime == ORIGINAL_IME

    asyncio.run(run())


def test_multiple_inputs_keep_the_original_ime_for_final_restoration():
    async def run():
        subject = driver()
        await subject.input_text("北")
        await subject.input_text("京")
        assert await subject.restore_input_method()
        assert subject.device.ime == ORIGINAL_IME
        assert len([c for c in subject.device.commands if c[:2] == ["ime", "set"]]) == 2
        assert not any(c[:2] == ["settings", "put"] for c in subject.device.commands)
        command_count = len(subject.device.commands)
        assert await subject.restore_input_method()
        assert len(subject.device.commands) == command_count

    asyncio.run(run())


def test_fake_successful_ime_set_uses_only_the_same_target_for_selection_and_restore():
    async def run():
        subject = driver(FakeDevice(ime_set_effective=False))
        assert await subject.input_text("蓝牙", clear=True)
        selected = [
            "settings",
            "put",
            "secure",
            "default_input_method",
            ADB_KEYBOARD_IME,
        ]
        assert subject.device.commands.count(["ime", "set", ADB_KEYBOARD_IME]) == 1
        assert subject.device.commands.count(selected) == 1
        assert subject.device.ime == ADB_KEYBOARD_IME
        binding = subject.device.commands.index(["dumpsys", "input_method"])
        assert subject.device.commands.index(selected) < binding
        broadcasts = [
            c for c in subject.device.commands if c[:2] == ["am", "broadcast"]
        ]
        assert len(broadcasts) == 2
        assert binding < subject.device.commands.index(broadcasts[0])
        assert subject.last_input_error is None

        assert await subject.restore_input_method()
        assert subject.device.commands.count(["ime", "set", ORIGINAL_IME]) == 1
        assert (
            subject.device.commands.count(
                ["settings", "put", "secure", "default_input_method", ORIGINAL_IME]
            )
            == 1
        )
        assert subject.device.ime == ORIGINAL_IME
        assert subject.last_restore_error is None

    asyncio.run(run())


def test_ineffective_secure_fallback_fails_before_input_with_a_fixed_stage_code():
    async def run():
        subject = driver(
            FakeDevice(ime_set_effective=False, secure_set_effective=False)
        )
        with pytest.raises(
            RuntimeError, match="stage=select_secure_verify code=ime_mismatch"
        ) as caught:
            await subject.input_text("敏感中文", clear=True)
        assert "敏感中文" not in str(caught.value)
        assert base64.b64encode("敏感中文".encode()).decode() not in str(caught.value)
        assert not any(c[:2] == ["am", "broadcast"] for c in subject.device.commands)
        assert (
            subject.last_input_error == "stage=select_secure_verify code=ime_mismatch"
        )
        assert subject.device.ime == ORIGINAL_IME

    asyncio.run(run())


def test_restore_failure_keeps_original_ime_and_fixed_diagnostics():
    async def run():
        subject = driver()
        await subject.input_text("蓝牙")
        subject.device.ime_set_effective = False
        subject.device.secure_set_effective = False
        assert await subject.restore_input_method() is False
        assert (
            subject.last_restore_error
            == "stage=restore_secure_verify code=ime_mismatch"
        )
        assert subject._original_input_method == ORIGINAL_IME
        assert subject.device.ime == ADB_KEYBOARD_IME

    asyncio.run(run())


def test_failed_ime_set_does_not_dispatch_secure_fallback_or_any_input():
    async def run():
        class FailingSelectionDevice(FakeDevice):
            async def shell(self, command, *, timeout):
                if command == ["ime", "set", ADB_KEYBOARD_IME]:
                    self.commands.append(list(command))
                    raise RuntimeError("private command contents")
                return await super().shell(command, timeout=timeout)

        subject = driver(FailingSelectionDevice())
        with pytest.raises(
            RuntimeError, match="stage=select_ime_set code=operation_failed"
        ) as caught:
            await subject.input_text("蓝牙")
        assert "private" not in str(caught.value)
        assert not any(
            c[:2] in (["settings", "put"], ["am", "broadcast"])
            for c in subject.device.commands
        )

    asyncio.run(run())


def test_missing_keyboard_fails_before_dispatching_any_input():
    async def run():
        subject = driver(FakeDevice(available=False))
        with pytest.raises(RuntimeError, match="verify the device state"):
            await subject.input_text("蓝牙", clear=True)
        assert not any(c[:2] == ["am", "broadcast"] for c in subject.device.commands)
        assert subject.device.ime == ORIGINAL_IME

    asyncio.run(run())


def test_input_timeout_does_not_retry_and_saved_ime_remains_restorable():
    async def run():
        subject = driver(FakeDevice(block=True), timeout=0.05)
        with pytest.raises(TimeoutError, match="No input was retried"):
            await subject.input_text("蓝牙")
        broadcasts = [
            c for c in subject.device.commands if c[:2] == ["am", "broadcast"]
        ]
        assert len(broadcasts) == 1
        assert broadcasts[0][3] == "ADB_INPUT_B64"
        assert subject.last_input_error == "stage=input_text code=timeout"
        assert await subject.restore_input_method()
        assert subject.device.ime == ORIGINAL_IME

    asyncio.run(run())


def test_unbound_keyboard_times_out_before_clear_or_text_is_sent():
    async def run():
        class UnboundDevice(FakeDevice):
            async def shell(self, command, *, timeout):
                response = await super().shell(command, timeout=timeout)
                if command == ["dumpsys", "input_method"]:
                    return response.replace(
                        "mBoundToMethod=true", "mBoundToMethod=false"
                    )
                return response

        subject = driver(UnboundDevice(), timeout=0.02)
        with pytest.raises(TimeoutError):
            await subject.input_text("蓝牙", clear=True)
        assert not any(c[:2] == ["am", "broadcast"] for c in subject.device.commands)
        assert await subject.restore_input_method()

    asyncio.run(run())


def test_restore_timeout_returns_false_without_retrying_the_switch():
    async def run():
        class BlockedRestoreDevice(FakeDevice):
            async def shell(self, command, *, timeout):
                if command == ["ime", "set", ORIGINAL_IME]:
                    self.commands.append(list(command))
                    await asyncio.Event().wait()
                return await super().shell(command, timeout=timeout)

        subject = driver(BlockedRestoreDevice(), timeout=0.02)
        await subject.input_text("蓝牙")
        assert await subject.restore_input_method() is False
        assert subject.device.commands.count(["ime", "set", ORIGINAL_IME]) == 1
        assert subject.last_restore_error == "stage=restore_ime_set code=timeout"
        assert not any(c[:2] == ["settings", "put"] for c in subject.device.commands)

    asyncio.run(run())


def test_failed_clear_never_sends_text_or_leaks_raw_adb_error():
    async def run():
        subject = driver(FakeDevice(error="sensitive encoded input in shell error"))
        with pytest.raises(RuntimeError) as caught:
            await subject.input_text("蓝牙", clear=True)
        assert "sensitive" not in str(caught.value)
        assert "stage=input_clear code=operation_failed" in str(caught.value)
        broadcasts = [
            c for c in subject.device.commands if c[:2] == ["am", "broadcast"]
        ]
        assert len(broadcasts) == 1
        assert broadcasts[0][3] == "ADB_CLEAR_TEXT"
        assert not subject.portal.input_text.called

    asyncio.run(run())


def test_empty_noop_does_not_connect_or_change_ime_and_clear_only_is_supported():
    async def run():
        subject = driver()
        assert await subject.input_text("")
        assert subject.device.commands == []
        assert await subject.input_text("", clear=True)
        broadcasts = [
            c for c in subject.device.commands if c[:2] == ["am", "broadcast"]
        ]
        assert len(broadcasts) == 1
        assert broadcasts[0][3] == "ADB_CLEAR_TEXT"

    asyncio.run(run())


def test_restore_does_not_override_an_external_input_method_change():
    async def run():
        subject = driver()
        await subject.input_text("蓝牙")
        subject.device.ime = "com.example.keyboard/.Ime"
        previous_sets = len(
            [c for c in subject.device.commands if c[:2] == ["ime", "set"]]
        )
        assert await subject.restore_input_method() is False
        assert subject.device.ime == "com.example.keyboard/.Ime"
        assert (
            len([c for c in subject.device.commands if c[:2] == ["ime", "set"]])
            == previous_sets
        )

    asyncio.run(run())


@pytest.mark.parametrize("timeout", [0, -1, float("inf"), float("nan"), True])
def test_invalid_timeouts_are_rejected(timeout):
    with pytest.raises(ValueError):
        AdbKeyboardAndroidDriver(input_timeout=timeout)
