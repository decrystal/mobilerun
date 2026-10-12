"""Portal-backed Android control with ADBKeyboard text input.

This optional driver avoids the Portal IME on emulators where its input path
can stall Android's input service. It requires an already installed, enabled
ADBKeyboard. Call ``restore_input_method`` in the caller's ``finally`` block.
"""

from __future__ import annotations

import asyncio
import base64
import math
import re

from mobilerun_core_local.driver.android import AndroidDriver

ADB_KEYBOARD_PACKAGE = "com.android.adbkeyboard"
ADB_KEYBOARD_IME = ADB_KEYBOARD_PACKAGE + "/.AdbIME"


class _KeyboardFailure(RuntimeError):
    """An internal fixed error code, never a device command or response."""

    def __init__(self, code: str) -> None:
        super().__init__(code)
        self.code = code


def _expanded_ime(value: str) -> str:
    package, separator, component = value.partition("/")
    if separator and component.startswith("."):
        component = package + component
    return package + separator + component


class AdbKeyboardAndroidDriver(AndroidDriver):
    """Use Portal for state/control and scoped UTF-8 broadcasts for typing."""

    def __init__(
        self,
        serial: str | None = None,
        use_tcp: bool = False,
        remote_tcp_port: int = 8080,
        portal_mode: str = "required",
        *,
        input_timeout: float = 10.0,
    ) -> None:
        if (
            isinstance(input_timeout, bool)
            or not math.isfinite(input_timeout)
            or input_timeout <= 0
        ):
            raise ValueError("input_timeout must be a positive finite number")
        super().__init__(
            serial=serial,
            use_tcp=use_tcp,
            remote_tcp_port=remote_tcp_port,
            portal_mode=portal_mode,
        )
        self.input_timeout = float(input_timeout)
        self._original_input_method: str | None = None
        self._input_lock = asyncio.Lock()
        self._operation_stage = "idle"
        self.last_input_error: str | None = None
        self.last_restore_error: str | None = None

    async def _setup_portal_keyboard(self) -> None:
        # AndroidDriver.connect calls this after connecting Portal. Keep the
        # current IME unchanged until this driver actually needs to type.
        self._portal_keyboard_available = False

    async def _keyboard_shell(self, arguments: list[str], *, stage: str) -> str:
        # async_adbutils quotes each list argument with shlex.quote. Input text
        # reaches this boundary only as a single ASCII Base64 argument.
        self._operation_stage = stage
        result = await self.device.shell(arguments, timeout=self.input_timeout)
        if not isinstance(result, str):
            raise _KeyboardFailure("invalid_shell_response")
        return result.strip()

    async def _current_input_method(self, *, stage: str) -> str:
        current = await self._keyboard_shell(
            ["settings", "get", "secure", "default_input_method"], stage=stage
        )
        if not current or current == "null" or "/" not in current:
            raise _KeyboardFailure("unknown_input_method")
        return current

    async def _switch_input_method(self, target: str, *, phase: str) -> None:
        await self._keyboard_shell(["ime", "set", target], stage=f"{phase}_ime_set")
        selected = await self._current_input_method(stage=f"{phase}_ime_verify")
        if _expanded_ime(selected) == _expanded_ime(target):
            return
        # Some LDPlayer versions report "selected" without changing the
        # setting. Write only the same already chosen IME, once, then read it
        # back. A timeout/exception in ime set never causes a second dispatch.
        await self._keyboard_shell(
            ["settings", "put", "secure", "default_input_method", target],
            stage=f"{phase}_secure_set",
        )
        selected = await self._current_input_method(stage=f"{phase}_secure_verify")
        if _expanded_ime(selected) != _expanded_ime(target):
            raise _KeyboardFailure("ime_mismatch")

    async def _select_adb_keyboard(self) -> None:
        current = await self._current_input_method(stage="select_read_original")
        if self._original_input_method is None:
            self._original_input_method = current
        if _expanded_ime(current) != _expanded_ime(ADB_KEYBOARD_IME):
            enabled = await self._keyboard_shell(
                ["ime", "list", "-s"], stage="select_inventory"
            )
            if not any(
                _expanded_ime(line.strip()) == _expanded_ime(ADB_KEYBOARD_IME)
                for line in enabled.splitlines()
            ):
                raise _KeyboardFailure("keyboard_not_enabled")
            # Save the original before dispatch: a timed-out switch can still
            # have taken effect and must remain restorable.
            await self._switch_input_method(ADB_KEYBOARD_IME, phase="select")

        # IME switching is asynchronous. Observe the binding before sending
        # any mutation; only these read-only checks are polled.
        while True:
            state = await self._keyboard_shell(
                ["dumpsys", "input_method"], stage="select_binding"
            )
            current_id = re.search(r"\bmCurId=([^\s]+)", state)
            if current_id is None:
                current_id = re.search(r"\bmCurMethodId=([^\s]+)", state)
            if (
                current_id
                and _expanded_ime(current_id.group(1))
                == _expanded_ime(ADB_KEYBOARD_IME)
                and re.search(r"\bmBoundToMethod=true\b", state)
            ):
                return
            await asyncio.sleep(0.05)

    async def _broadcast(self, action: str, encoded: str | None = None) -> None:
        command = ["am", "broadcast", "-a", action, "-p", ADB_KEYBOARD_PACKAGE]
        if encoded is not None:
            command.extend(["--es", "msg", encoded])
        stage = "input_clear" if action == "ADB_CLEAR_TEXT" else "input_text"
        response = await self._keyboard_shell(command, stage=stage)
        if re.search(r"\bBroadcast completed: result=-?\d+", response) is None:
            raise _KeyboardFailure("broadcast_unconfirmed")

    async def input_text(self, text: str, clear: bool = False) -> bool:
        """Send each clear/input once; UI read-back remains the caller's job."""
        if not isinstance(text, str) or not isinstance(clear, bool):
            raise TypeError("ADBKeyboard input requires text and a boolean clear flag")
        if not text and not clear:
            return True
        encoded = base64.b64encode(text.encode("utf-8")).decode("ascii")
        locked = False
        self.last_input_error = None
        try:
            async with asyncio.timeout(self.input_timeout):
                async with self._input_lock:
                    locked = True
                    self._operation_stage = "input_connect"
                    await self.ensure_connected()
                    await self._select_adb_keyboard()
                    if clear:
                        await self._broadcast("ADB_CLEAR_TEXT")
                    if text:
                        await self._broadcast("ADB_INPUT_B64", encoded)
                    return True
        except TimeoutError:
            stage = self._operation_stage if locked else "input_lock"
            self.last_input_error = f"stage={stage} code=timeout"
            raise TimeoutError(
                f"ADBKeyboard input timed out [{self.last_input_error}]; "
                "the outcome may be uncertain. No input was retried."
            ) from None
        except Exception as exc:
            # ADB errors can include the full command, including encoded text.
            # Do not leak it through ToolRegistry's exception summaries.
            stage = self._operation_stage if locked else "input_lock"
            code = exc.code if isinstance(exc, _KeyboardFailure) else "operation_failed"
            self.last_input_error = f"stage={stage} code={code}"
            raise RuntimeError(
                f"ADBKeyboard input failed [{self.last_input_error}]; "
                "verify the device state before retrying."
            ) from None

    async def restore_input_method(self) -> bool:
        """Restore the saved IME, without overriding a later external choice."""
        original = self._original_input_method
        self.last_restore_error = None
        if original is None:
            return True
        locked = False
        try:
            async with asyncio.timeout(self.input_timeout):
                async with self._input_lock:
                    locked = True
                    self._operation_stage = "restore_connect"
                    await self.ensure_connected()
                    current = await self._current_input_method(
                        stage="restore_read_current"
                    )
                    if _expanded_ime(current) == _expanded_ime(original):
                        self._original_input_method = None
                        return True
                    if _expanded_ime(current) != _expanded_ime(ADB_KEYBOARD_IME):
                        self.last_restore_error = (
                            "stage=restore_read_current code=external_ime_change"
                        )
                        return False
                    await self._switch_input_method(original, phase="restore")
                    # A restored IME need not be bound while no input is
                    # focused. Verify its selected setting without opening it.
                    self._original_input_method = None
                    return True
        except Exception as exc:
            stage = self._operation_stage if locked else "restore_lock"
            code = (
                "timeout"
                if isinstance(exc, TimeoutError)
                else (
                    exc.code
                    if isinstance(exc, _KeyboardFailure)
                    else "operation_failed"
                )
            )
            self.last_restore_error = f"stage={stage} code={code}"
            return False
