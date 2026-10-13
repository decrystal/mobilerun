"""Jev-first orchestration, agent completion and optional assertion checks.

This module selects tools; the existing FastAgent/ToolRegistry executes them.
Agent completion and independently checked completion are reported separately.
"""

from __future__ import annotations

import asyncio
import copy
import hashlib
import inspect
import json
import logging
import re
import time
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.decision.completion import validate_completion_criteria
from mobilerun.agent.decision.effects import (
    VERIFY_TOOLS,
    EffectLedger,
    EffectNotes,
    effect_key,
    evidence_valid,
    returned_to_previous_screen,
)
from mobilerun.agent.decision.input_verification import (
    capture_input_target,
    input_value_matches,
    read_input_value,
)
from mobilerun.agent.decision.jev import (
    JevDecisionPolicy,
    coordinate_fingerprint,
    focused_input_observation,
    native_action_owner,
    snapshot_fingerprint,
)
from mobilerun.agent.decision.native_proof_binding import (
    NativeProofBindingError,
    bind_native_proofs,
    corroborate_native_binding,
)
from mobilerun.agent.decision.native_query_binding import (
    NativeQueryBindingError,
    assessed_query_refs,
    bind_native_queries,
    corroborate_native_queries,
)
from mobilerun.agent.decision.query import QueryError
from mobilerun.agent.decision.task_state import (
    TaskState,
    corroborate_nodes,
    observed_nodes,
)
from mobilerun.agent.fast_agent.xml_parser import ToolCall
from mobilerun.agent.utils.actions import _macro_pre_ui, _record_macro_action
from mobilerun.tools.ui.provider import require_fresh_observation

logger = logging.getLogger("mobilerun")


def flatten_elements(ui) -> list[dict]:
    result = []

    def visit(nodes):
        for node in nodes:
            result.append(node)
            visit(node.get("children", []))

    visit(ui.elements)
    return result


def completion_matches(ui, criteria: dict) -> bool:
    """Evaluate explicit postconditions against the current visible UI."""
    validate_completion_criteria(criteria)
    if not criteria:
        return False
    phone = ui.phone_state
    nodes = [
        n
        for n in flatten_elements(ui)
        if n.get("visible", n.get("isVisibleToUser", n.get("isVisible", True)))
        is not False
    ]
    labels = [
        str(n[k])
        for n in nodes
        for k in ("text", "contentDescription", "label")
        if n.get(k)
    ]
    if "package" in criteria and phone.get("packageName") != criteria["package"]:
        return False
    if "activity_contains" in criteria and criteria["activity_contains"] not in str(
        phone.get("currentApp", "")
    ):
        return False
    if "text_all" in criteria and not all(
        any(text in label for label in labels) for text in criteria["text_all"]
    ):
        return False
    if "text_any" in criteria and not any(
        text in label for text in criteria["text_any"] for label in labels
    ):
        return False
    if "input_text" in criteria:
        if not any(
            (
                n.get("editable")
                or n.get("isEditable")
                or "EditText" in n.get("className", "")
            )
            and n.get("inputValue", n.get("text")) == criteria["input_text"]
            for n in nodes
        ):
            return False
    for label, expected in criteria.get("checked", {}).items():
        matches = [
            n
            for n in nodes
            if label in [n.get("text"), n.get("contentDescription"), n.get("label")]
        ]
        values = []
        for node in matches:
            value = node.get("checked", node.get("isChecked"))
            if value is None and node.get("checkedState") in (
                "checked",
                "unchecked",
                "isChecked=True",
                "isChecked=False",
            ):
                value = node["checkedState"] in ("checked", "isChecked=True")
            if isinstance(value, bool):
                values.append(value)
        if len(values) != 1 or values[0] is not expected:
            return False
    return True


async def launch_app(package_name: str, *, ctx) -> ActionResult:
    """Launch an observed exact app without calling the app-opener LLM."""
    apps = await ctx.driver.get_apps(include_system=True)
    packages = {
        app.get("package_name")
        or app.get("package")
        or app.get("packageName")
        or app.get("app_id")
        for app in apps
    }
    if package_name not in packages:
        return ActionResult(
            success=False, summary="App is not in the installed-app inventory"
        )
    pre_ui = await _macro_pre_ui(ctx)
    result = await ctx.driver.start_app(package_name)
    success = not (isinstance(result, str) and result.lower().startswith("failed"))
    if success:
        _record_macro_action(
            ctx,
            {"action_type": "start_app", "package": package_name, "activity": None},
            pre_ui=pre_ui,
        )
    return ActionResult(success=success, summary=str(result or "App launched"))


def register_decision_tools(registry):
    registry.register(
        "launch_app",
        fn=launch_app,
        params={"package_name": {"type": "string", "required": True}},
        description="Launch an installed application by its exact observed package name, without inference.",
        deps={"start_app", "get_apps"},
    )
    # The ordinary Android open_app helper would invoke another LLM internally.
    registry.disable(["open_app"])


@dataclass
class Selection:
    source: str
    calls: list[ToolCall]
    reason: str = ""
    guard: Any = None
    snapshot: Any = None
    stop: bool = False
    visual_fingerprint: str | None = None
    navigation_context: dict | None = None
    progress: str = ""
    coordinate_guard: str | None = None
    input_guard: dict | None = None
    input_source: dict | None = None
    completion_checks: dict | None = None
    completion_proofs: list[dict] | None = None
    completion_native_binding: dict | None = None
    completion_native_binding_required: bool = False
    completion_query_binding: dict | None = None
    completion_query_binding_required: bool = False
    completion_image_guard: str | None = None
    completion_observation_id: str | None = None
    native_target_binding: dict | None = None
    expected_effect: str = ""

    def __post_init__(self):
        if (
            self.snapshot is not None
            and len(self.calls) == 1
            and self.native_target_binding is None
        ):
            try:
                self.native_target_binding = native_action_owner(
                    self.snapshot, self.calls[0]
                )
            except (ValueError, TypeError, KeyError):
                self.native_target_binding = {"invalid": True}


class DecisionRuntime:
    def __init__(
        self,
        config,
        action_ctx,
        registry,
        state_provider,
        *,
        verifier=None,
        policy=None,
        visual_fallback=None,
        trace=None,
        app_card_config=None,
    ):
        self.config = config
        self.ctx = action_ctx
        self.registry = registry
        self.state_provider = state_provider
        self.verifier = verifier
        validate_completion_criteria(config.completion)
        self.policy = policy or JevDecisionPolicy(
            model=config.model,
            timeout=config.timeout,
            confidence_threshold=config.confidence_threshold,
        )
        if visual_fallback is None and config.fallback == "visual":
            from mobilerun.agent.decision.visual_fallback import VisualFallback

            visual_fallback = VisualFallback(
                model=config.fallback_model,
                reasoning_model=config.reasoning_model,
                base_url=config.fallback_base_url,
                api_key_env=config.fallback_api_key_env,
                thinking_mode=config.fallback_thinking,
                timeout=config.fallback_timeout,
                max_history_turns=0,
                effect_scopes=getattr(config, "effect_scopes", False),
                query_scopes=getattr(config, "query_scopes", False),
            )
        self.visual_fallback = visual_fallback
        self.trace = trace if trace is not None else []
        self.history = []
        self.apps = None
        self.pending = None
        self.executed = set()
        self.last_executed_state = None
        self.feedback = ""
        self.last_feedback = ""
        self.goal = ""
        self.stale_count = 0
        self.input_target = None
        self.instructions = (
            Path(__file__).with_name("android.md").read_text(encoding="utf-8")
        )
        self.app_card_provider = None
        if app_card_config and app_card_config.enabled:
            from mobilerun.app_cards.providers import (
                CompositeAppCardProvider,
                LocalAppCardProvider,
                ServerAppCardProvider,
            )

            mode = app_card_config.mode.lower()
            if mode == "server" and app_card_config.server_url:
                self.app_card_provider = ServerAppCardProvider(
                    server_url=app_card_config.server_url,
                    timeout=app_card_config.server_timeout,
                    max_retries=app_card_config.max_retries,
                )
            elif mode == "composite" and app_card_config.server_url:
                self.app_card_provider = CompositeAppCardProvider(
                    server_url=app_card_config.server_url,
                    app_cards_dir=app_card_config.app_cards_dir,
                    server_timeout=app_card_config.server_timeout,
                    max_retries=app_card_config.max_retries,
                )
            else:
                self.app_card_provider = LocalAppCardProvider(
                    app_card_config.app_cards_dir
                )
        self.app_card = ""
        self.progress_note = ""
        self.action_signatures = []
        self.review_rejections = 0
        self.completion_refresh_pending = False
        self.completion_refresh_native = False
        self.last_completion_failure = None
        self.task_state = TaskState(
            query_memory=getattr(config, "query_memory", True),
            visual_frame_observations=getattr(
                config, "visual_frame_observations", False
            ),
        )
        self.input_retry = None
        self.input_retry_source = None
        self.input_retry_activated = False
        self.input_retry_counts = {}
        self.last_input_readback = None
        self.review_diagnostic = ""
        self.query_failures = 0
        self.query_feedback = ""
        self.effects = EffectLedger()
        self.effect_trail = []
        self.effect_probes = 0
        self.media_seen = {}  # md5 -> file name saved earlier in this task
        self.media_attributed = set()  # gallery files already linked to an effect
        self.effect_notes = EffectNotes(None)
        if config.fallback == "visual":
            import platformdirs

            self.effect_notes = EffectNotes(
                Path(platformdirs.user_data_dir("mobilerun")) / "effect_notes.json"
            )
            if callable(getattr(self.registry, "register", None)):
                self.registry.register(
                    "recent_media",
                    fn=self.recent_media,
                    params={},
                    description="List photos and videos added to the device gallery after the action being verified, with md5. No device mutation.",
                )
        if getattr(config, "query_memory", True) and callable(
            getattr(self.registry, "register", None)
        ):
            self.registry.register(
                "query_collection",
                fn=self.query_collection,
                params={"plan": {"type": "object", "required": True}},
                description="Calculate count, sum, max or list from an observed native collection. Bind facts to native row references; complete coverage is required. No device mutation.",
            )

    def query_collection(self, plan: dict, *, ctx) -> ActionResult:
        try:
            result = self.task_state.query(plan)
        except (QueryError, KeyError, TypeError, ValueError) as exc:
            code = str(exc) if isinstance(exc, QueryError) else "query_plan_invalid"
            correction = {
                "fact_span_invalid": "For a complete source value, omit span. A substring span must be two Python character offsets [start,end], never screen coordinates or bounds. Prefer selectors for the same unique resourceId in each row.",
                "query_filter_invalid": 'A relative_week filter accepts only field, operator, next_week, week_start and format, with no value: {"field":"date","operator":"relative_week","week_start":0,"next_week":false}. eq/contains require field, operator and value. Filters are AND, not OR.',
                "collection_coverage_incomplete": "The native row coverage is incomplete. Scroll and observe missing row positions before calculating; do not change scope to pretend the visible subset is the entire collection.",
                "query_scope_stale": "Copy an actual current scope and generation from ledger.collections. Do not construct a scope from a package name, description or old generation.",
                "date_ambiguous": "Read an explicit date from the record details or an actual UI date filter. Do not guess a year or the meaning of a weekday label.",
                "filter_property_unknown": "The requested property must be bound in every non-heading row. Use an observed unique resourceId selector, or inspect missing row properties before calculating.",
            }.get(
                code,
                "Bind facts to native row references and ensure complete collection coverage.",
            )
            self.record(
                "query_diagnostic",
                status="rejected",
                error_code=code,
                device_action_executed=False,
            )
            return ActionResult(
                False,
                "Local query rejected ("
                + code
                + "). "
                + correction
                + " No device action was executed.",
            )
        self.record("query_result", **result)
        return ActionResult(True, json.dumps(result, ensure_ascii=False))

    def recover_local_failure(self, call, result) -> bool:
        if call.name != "query_collection" or self.query_failures >= 2:
            return False
        self.query_failures += 1
        self.query_feedback = (
            result.summary
            + " Correct the local query using observed scope/fact IDs, or gather missing UI evidence. Never invent a scope key."
        )
        self.record(
            "recovery",
            source="query",
            reason="local_validation_rejected",
            device_action_executed=False,
        )
        return True

    @property
    def has_completion_check(self):
        return bool(self.verifier or self.config.completion)

    async def load_guidance(self, ui):
        self.app_card = ""
        if self.app_card_provider:
            package = ui.phone_state.get("packageName", "")
            try:
                self.app_card = await self.app_card_provider.load_app_card(package)
            except Exception:
                self.record("guidance", status="unavailable", package=package)
        notes = self.effect_notes.for_app(ui.phone_state.get("packageName", ""))
        if notes:
            self.app_card += "\nEffect verification notes from earlier runs:\n" + "\n".join(
                "- " + note for note in notes
            )

    def task_context(self, *, purpose="review"):
        rich = (self.task_state.contract or {}).get("schema_version") == 2
        compact = rich and purpose in {"action", "jev"}
        share_values = bool(
            rich
            and purpose != "jev"
            and getattr(self.config, "share_native_values", False)
            and getattr(self.visual_fallback, "supports_shared_native_values", False)
        )
        limit = 4 if compact else 8 if rich else 64
        ledger = (
            self.task_state.context(
                max_observations=1,
                max_chars=2500,
                max_actions=4,
                retain_proofs=False,
                share_native_values=share_values,
            )
            if compact
            else self.task_state.context(
                max_observations=4 if getattr(self.config, "query_memory", True) else 1,
                share_native_values=share_values,
            )
        )
        if not getattr(self.config, "semantic_recovery", True):
            ledger.pop("semantic_progress", None)
        if self.effects.entries:
            ledger["side_effects"] = self.effects.context()
        return {
            "goal": self.goal,
            "progress_note": self.progress_note,
            "executed_actions": len(self.history),
            "execution_log": [
                {
                    key: value
                    for key, value in entry.items()
                    if key != "source" and (not compact or key != "after")
                }
                for entry in self.history[-limit:]
            ],
            "earlier_actions_omitted": max(0, len(self.history) - limit),
            "ledger": ledger,
        }

    def observe_result(self, ui):
        self.task_state.observe(ui)
        self.record_collection_diagnostics()
        if self.history and "after" not in self.history[-1]:
            self.history[-1]["after"] = {
                "package": ui.phone_state.get("packageName"),
                "visible_text": [
                    str(n.get("text") or n.get("contentDescription") or n.get("label"))[
                        :160
                    ]
                    for n in flatten_elements(ui)
                    if (n.get("text") or n.get("contentDescription") or n.get("label"))
                    and not (n.get("password") or n.get("isPassword"))
                ][:24],
            }
            self.record("progress", **self.task_context())

    def record_collection_diagnostics(self):
        collections = getattr(self.task_state, "collections", None)
        take = getattr(collections, "take_diagnostics", None)
        if callable(take):
            for diagnostic in take():
                self.record("collection_invalidation", **diagnostic)

    def record(self, event, **fields):
        entry = {"event": event, **fields}
        self.trace.append(entry)
        logger.log(
            (
                logging.DEBUG
                if event
                in {
                    "progress",
                    "completion_guard_binding",
                    "completion_guard",
                    "collection_invalidation",
                }
                else logging.INFO
            ),
            "Decision: %s",
            json.dumps(entry, ensure_ascii=False),
        )

    async def verified(self, ui):
        if self.verifier:
            result = self.verifier(ui)
            if inspect.isawaitable(result):
                result = await result
            return result is True
        return completion_matches(ui, self.config.completion)

    @staticmethod
    def coordinate_guard(call, ui):
        if call.name in {"click_at", "long_press_at", "double_tap_at"}:
            x, y = call.parameters.get("x"), call.parameters.get("y")
            if type(x) is int and type(y) is int:
                return coordinate_fingerprint(ui, x, y)
        elif call.name == "swipe":
            start = call.parameters.get("coordinate")
            end = call.parameters.get("coordinate2")
            if all(
                isinstance(point, (list, tuple))
                and len(point) == 2
                and all(type(value) is int for value in point)
                for point in (start, end)
            ):
                return coordinate_fingerprint(ui, *start, end=tuple(end))
        return None

    def signature(self, call, ui, visual_fingerprint=None):
        state = (
            self.coordinate_guard(call, ui)
            or visual_fingerprint
            or snapshot_fingerprint(ui)
        )
        if call.name == "click":
            node = next(
                (
                    n
                    for n in flatten_elements(ui)
                    if n.get("index") == call.parameters.get("index")
                ),
                None,
            )
            if node and isinstance(node.get("bounds"), str):
                left, top, right, bottom = map(int, node["bounds"].split(","))
                x, y = (left + right) // 2, (top + bottom) // 2
                if not ui.use_normalized:
                    x, y = round(x / ui.coordinate_scale_x), round(
                        y / ui.coordinate_scale_y
                    )
                state = coordinate_fingerprint(ui, x, y) or state
        return (
            state,
            call.name,
            json.dumps(call.parameters, sort_keys=True, ensure_ascii=False),
        )

    def completion(
        self,
        ui,
        *,
        source="verifier",
        message=None,
        visual_fingerprint=None,
        final_answer=None,
    ):
        self.record(
            "completion",
            source=source,
            basis=("assertions" if self.has_completion_check else "agent_assessment"),
            message=message
            or "Configured checks passed after the agent reported completion.",
        )
        return Selection(
            source,
            [
                ToolCall(
                    "complete",
                    {
                        "success": True,
                        **(
                            {"final_answer": final_answer}
                            if final_answer is not None
                            else {}
                        ),
                        "message": message
                        or "Model reported completion; configured completion checks passed.",
                    },
                )
            ],
            snapshot=ui,
            visual_fingerprint=visual_fingerprint,
        )

    def observe_progress(self, fingerprint):
        if self.last_executed_state and fingerprint != self.last_executed_state:
            # Returning to a page after a successful navigation is legitimate.
            self.executed.clear()
            self.last_executed_state = None

    async def select(self, goal, ui):
        if (
            self.goal
            and goal != self.goal
            and (self.config.completion or self.verifier)
        ):
            self.record(
                "verification", success=False, reason="completion_contract_changed"
            )
            self.pending = Selection(
                "verifier",
                [],
                reason="completion_contract_changed: update the verifier for the new goal",
                stop=True,
            )
            return self.pending
        if self.goal and self.goal != goal:
            self.progress_note = ""
            self.action_signatures.clear()
            self.review_rejections = 0
            self.completion_refresh_pending = False
            self.completion_refresh_native = False
            self.last_completion_failure = None
            self.history.clear()
            self.executed.clear()
            self.last_executed_state = None
            self.input_retry = None
            self.input_retry_source = None
            self.input_retry_activated = False
            self.input_retry_counts.clear()
            self.input_target = None
            self.last_input_readback = None
            self.review_diagnostic = ""
            self.query_failures = 0
            self.query_feedback = ""
            self.last_feedback = ""
            self.effects = EffectLedger()
            self.effect_trail = []
            self.media_seen = {}
            self.media_attributed = set()
            self.record("goal_changed", previous=self.goal, goal=goal)
        self.goal = goal
        if self.task_state.goal != goal:
            self.task_state.reset(goal)
        self.observe_result(ui)
        await self.load_guidance(ui)
        if ui.elements:
            self.observe_progress(snapshot_fingerprint(ui))
        # A postcondition can hold at an intermediate step (for example, the
        # target app is open before searching). Only verify after a DONE request.
        if self.apps is None:
            try:
                self.apps = await self.ctx.driver.get_apps(include_system=True)
            except Exception:
                self.apps = []
                self.record("inventory", status="unavailable")
        prepare = getattr(self.visual_fallback, "prepare_task", None)
        if callable(prepare) and self.task_state.contract is None:
            started = time.monotonic()
            error = await prepare(goal=goal)
            self.record(
                "task_preparation",
                status="error" if error else "ready",
                latency_ms=round((time.monotonic() - started) * 1000, 1),
                error_code=getattr(error, "error_code", ""),
                error_field=getattr(error, "error_field", ""),
            )
            if error:
                self.pending = Selection(
                    "reasoning", [], reason=error.message, stop=True
                )
                return self.pending
            contract = getattr(self.visual_fallback, "completion_contract", None)
            if isinstance(contract, dict):
                self.task_state.set_contract(contract)
            # Task preparation may have taken seconds; decisions use a new screen.
            ui = await self.state_provider.get_state()
            self.ctx.ui = ui
        self.task_state.observe(ui)
        self.record_collection_diagnostics()
        await self.observe_media_effect()
        self.detect_structural_effect(ui)
        if self.completion_refresh_pending:
            self.completion_refresh_pending = False
            native = self.completion_refresh_native
            self.completion_refresh_native = False
            return await self._visual(
                ui, completion_review=True, native_only_review=native
            )
        if self.task_state.contract and not self.task_state.device_context:
            device = getattr(self.ctx.driver, "device", None)
            shell = getattr(device, "shell", None)
            if callable(shell):
                try:
                    from datetime import datetime

                    device_time = (
                        await asyncio.wait_for(
                            shell(["date", "+%Y-%m-%dT%H:%M:%S%z"]), timeout=3
                        )
                    ).strip()
                    parsed = datetime.fromisoformat(device_time)
                    self.task_state.device_context = {
                        "datetime": parsed.isoformat(),
                        "source": "device_clock",
                        "captured_at": time.time(),
                    }
                except Exception:
                    self.task_state.device_context = {
                        "datetime": None,
                        "source": "unavailable",
                    }
        if self.input_retry is not None:
            call, target = self.input_retry
            input_source = self.input_retry_source
            self.input_retry = None
            self.input_retry_source = None
            if input_source:
                from mobilerun.agent.decision.native_text import (
                    validate_native_text_receipt,
                )

                if not validate_native_text_receipt(
                    self.task_state.observations,
                    self.goal,
                    input_source,
                    call.parameters.get("text"),
                ):
                    self.pending = Selection(
                        "input_recovery",
                        [],
                        reason="The recorded native text source no longer binds this replacement; no input was dispatched.",
                        stop=True,
                    )
                    return self.pending
            readback = await read_input_value(
                self.ctx.driver, ui, target, call.parameters["text"]
            )
            if readback.status == "confirmed":
                self.task_state.record_input_effect("confirmed")
            else:
                focused = await capture_input_target(self.ctx.driver, ui)
                same = (
                    focused is not None
                    and target is not None
                    and (
                        focused.package_name,
                        focused.resource_id,
                        focused.class_name,
                        focused.unique_resource_id,
                    )
                    == (
                        target.package_name,
                        target.resource_id,
                        target.class_name,
                        target.unique_resource_id,
                    )
                    and (
                        target.unique_resource_id
                        or (
                            focused.bounds == target.bounds
                            and focused.parent == target.parent
                        )
                    )
                )
                if readback.status != "mismatch" or not same:
                    self.pending = Selection(
                        "input_recovery",
                        [],
                        reason="Input effect remains unknown; replacement was not dispatched.",
                        stop=True,
                    )
                    return self.pending
                if not self.input_retry_activated and "click" in self.registry.tools:
                    nodes = [
                        node
                        for node in flatten_elements(ui)
                        if node.get("resourceId") == target.resource_id
                        and node.get("className", "").split(".")[-1]
                        == target.class_name
                    ]
                    if not target.unique_resource_id:
                        nodes = [
                            node
                            for node in nodes
                            if node.get("focused", node.get("isFocused")) is True
                        ]
                    if len(nodes) == 1 and type(nodes[0].get("index")) is int:
                        # Activation is an explicit device action and consumes a
                        # normal step; do not hide a tap inside the typing driver.
                        self.input_retry = (call, target)
                        self.input_retry_source = copy.deepcopy(input_source)
                        self.input_retry_activated = True
                        activation = ToolCall("click", {"index": nodes[0]["index"]})
                        self.pending = Selection(
                            "input_activation", [activation], snapshot=ui
                        )
                        self.record(
                            "recovery",
                            source="input",
                            reason="activate_bound_editor",
                            action_executed=False,
                        )
                        return self.pending
                self.record(
                    "recovery",
                    source="input",
                    reason="confirmed_field_mismatch",
                    action_executed=False,
                )
                self.pending = Selection(
                    "input_recovery",
                    [call],
                    snapshot=ui,
                    input_guard=focused_input_observation(ui),
                    input_source=copy.deepcopy(input_source),
                )
                return self.pending
        effect = self.effects.pending()
        if effect is not None and self.config.fallback == "visual":
            # An unverified side effect is resolved before any new decision so
            # that it can never be blindly replayed.
            selection = await self._verify_effect(ui, effect)
            if selection is not None:
                return selection
            ui = await self.state_provider.get_state()
            self.ctx.ui = ui
        if self.query_feedback:
            self.feedback, self.query_feedback = self.query_feedback, ""
            return await self._visual(ui, recovery=True)
        if (
            getattr(self.config, "semantic_recovery", True)
            and (self.task_state.contract or {}).get("schema_version") == 2
            and self.task_state.progress.request_recovery(self.task_state)
        ):
            self.feedback = (
                "Several actions produced no grounded task progress. Review the current requirement, "
                "confirmed input values, observed entities and actual execution outcomes. Use an alternate "
                "strategy such as search/filter, inspect a detail page or navigate back to a known page. "
                "Do not replay a send/delete/submit whose outcome is unknown; first inspect its result. "
                "Choose one normal action; rendering changes or a dispatched action do not prove completion."
            )
            self.record(
                "recovery",
                source="semantic_progress",
                reason="no_grounded_progress",
                action_executed=False,
            )
            return await self._visual(ui, recovery=True)
        result = await self.policy.decide(
            goal=goal,
            ui_state=ui,
            history=[
                {key: value for key, value in entry.items() if key != "after"}
                for entry in self.history[-4:]
            ],
            texts=self.config.text_values,
            apps=self.apps,
            available_tools=set(self.registry.tools),
            guidance=self.instructions + "\nApp-specific hints:\n" + self.app_card,
            task_state=self.task_context(purpose="jev"),
        )
        self.record(
            "decision",
            source="jev",
            status=result.status,
            reason=result.reason,
            model=self.config.model,
            confidence=result.confidence,
            target_confidence=result.target_confidence,
            target_index=result.target_guard.index if result.target_guard else None,
            target_bounds=result.target_guard.bounds if result.target_guard else None,
            operation=result.operation,
            latency_ms=result.latency_ms,
            usage=result.usage,
            api_called=result.api_called,
        )
        recovery = False
        review_requested = result.status == "done" or (
            self.config.fallback == "visual"
            and result.operation == "DONE"
            and result.reason == "low_operation_confidence"
        )
        if result.status == "action" and result.tool_call:
            signature = self.signature(result.tool_call, ui)
            confirmed = self.confirmed_effect_for(result.tool_call, ui)
            if (
                confirmed is None
                and signature not in self.executed
                and self.action_signatures[-12:].count(signature) < 2
            ):
                self.pending = Selection(
                    "jev", [result.tool_call], guard=result, snapshot=ui
                )
                return self.pending
            if confirmed is not None:
                # Jev cannot see pixels; the visual model decides whether this
                # is a different item or a duplicate of a confirmed effect.
                self.feedback = (
                    f"{confirmed.id} ({confirmed.description}) was already confirmed for the same "
                    "control. Repeat it only if the current screen shows a different item; "
                    "otherwise continue with the remaining work."
                )
            else:
                self.feedback = "The same action on the same screen made no progress. Choose a different way to advance the task."
            recovery = True
        elif review_requested:
            fresh = await self.state_provider.get_state()
            self.ctx.ui = fresh
            verified = self.has_completion_check and await self.verified(fresh)
            self.record(
                "verification",
                source="jev",
                success=verified,
                required=self.has_completion_check,
            )
            if verified and self.config.fallback != "visual":
                self.pending = self.completion(fresh)
                return self.pending
            self.feedback = (
                "Jev proposed completion. Review EVERY requirement in the full goal against "
                "the current screen and actual execution record. Return finish with concrete "
                "evidence only if all requirements are met; otherwise continue the task."
            )
            if self.has_completion_check and not verified:
                self.feedback += (
                    " The caller's additional completion checks have not passed."
                )
        else:
            self.feedback = "Jev could not select a reliable action: " + result.reason
            if result.reason == "text_value_required":
                self.feedback += (
                    " The observed input is already focused. Derive the exact text from "
                    "the user's goal and use Type if the fresh observation still confirms focus."
                )
            elif result.reason == "text_long_press_requires_visual_grounding":
                self.feedback += (
                    " Jev selected an editable control for a text long press. Use the screenshot "
                    "to press ON the visible text glyphs, not the blank center of the field. "
                    "Do not tap a menu item. Only ground the Long Press action."
                )
                if result.target_guard:
                    self.feedback += (
                        f" Previous control index: {result.target_guard.index}; "
                        f"raw device bounds: {result.target_guard.bounds}. Recheck the fresh screen."
                    )
        if result.reason == "query_requires_grounding":
            recovery = True
            self.feedback += " Bind native facts and use Query for deterministic aggregation. Verify collection coverage and user filters; do not invent unseen rows."
        self.record(
            "fallback", source="jev", target=self.config.fallback, reason=self.feedback
        )
        if self.last_feedback:
            self.feedback += "\nPrevious execution outcome: " + self.last_feedback
        if self.config.fallback == "llm":
            self.pending = None
            return None
        return await self._visual(
            ui, completion_review=review_requested, recovery=recovery
        )

    async def navigation_context(self):
        if not hasattr(self.ctx.driver, "get_ui_tree"):
            return None
        try:
            raw = await asyncio.wait_for(self.ctx.driver.get_ui_tree(), timeout=5)
            require_fresh_observation(raw)
            phone = raw.get("phone_state", {})
            if phone.get("packageName"):
                context = {
                    key: phone.get(key)
                    for key in (
                        "packageName",
                        "currentApp",
                        "activityName",
                        "isEditable",
                        "keyboardVisible",
                    )
                }
                # Page semantics, rather than cursor pixels, distinguish two
                # screens inside the same app. Do not include selection offsets
                # or other rendering-only properties from the accessibility API.
                nodes = []

                def visit(tree):
                    if isinstance(tree, list):
                        for child in tree:
                            visit(child)
                    elif isinstance(tree, dict):
                        nodes.append(
                            {
                                key: tree.get(key)
                                for key in (
                                    "resourceId",
                                    "className",
                                    "text",
                                    "contentDescription",
                                    "boundsInScreen",
                                    "isClickable",
                                    "isEditable",
                                    "isChecked",
                                    "isEnabled",
                                )
                            }
                        )
                        visit(tree.get("children", []))

                visit(raw.get("a11y_tree"))
                context["tree"] = hashlib.sha256(
                    json.dumps(nodes, sort_keys=True).encode()
                ).hexdigest()
                return context
        except Exception:
            return None
        return None

    def _derive_review_dependencies(self):
        return bool(
            getattr(self.config, "derive_review_dependencies", False)
            and getattr(
                self.visual_fallback, "supports_derived_review_dependencies", False
            )
        )

    async def _visual(
        self,
        ui,
        attempt=0,
        completion_review=False,
        recovery=False,
        native_only_review=False,
        corrective_only=False,
        verify_effect=None,
    ):
        started = time.monotonic()
        # Jev may have waited for a network timeout; observe again for the VLM.
        ui = await self.state_provider.get_state()
        self.ctx.ui = ui
        native_only_review = bool(
            native_only_review and completion_review and ui.elements
        )
        self.observe_result(ui)
        await self.load_guidance(ui)
        screenshot = await self.ctx.driver.screenshot()
        self.task_state.observe(ui, screenshot=screenshot)
        self.record_collection_diagnostics()
        visual_fingerprint = (
            hashlib.sha256(screenshot).hexdigest() if not ui.elements else None
        )
        if visual_fingerprint:
            self.observe_progress(visual_fingerprint)
        navigation_context = (
            await self.navigation_context() if visual_fingerprint else None
        )
        tools = set(self.registry.tools)
        query_requirements = any(
            item["kind"] in {"query", "collection"}
            for item in (self.task_state.contract or {}).get("requirements", [])
        )
        if not query_requirements or not any(
            key.startswith(str(ui.phone_state.get("packageName")) + ":")
            for key in self.task_state.collections.scopes
        ):
            tools.discard("query_collection")
        if verify_effect is not None:
            # Verification may navigate and probe, never commit.
            tools = (tools & VERIFY_TOOLS) | {"verify_effect"}
        else:
            tools.discard("recent_media")
        # RESOLVE_INPUT asks for an argument to an already-grounded typing action.
        # A missing on-screen keyboard does not negate accessibility input focus.
        typing_only = (
            not completion_review
            and "text_value_required" in self.feedback
            and "type" in tools
            and focused_input_observation(ui) is not None
        )
        if typing_only:
            tools = {"type"}
        elif (
            not completion_review
            and "text_long_press_requires_visual_grounding" in self.feedback
        ):
            tools &= {"long_press_at"}
        self.record(
            "fallback_context",
            source="autoglm",
            typing_only=typing_only,
            completion_review=completion_review,
            review_history_images=getattr(self.config, "review_history_images", True),
            native_only_review=native_only_review,
            corrective_only=corrective_only,
            recovery=recovery,
        )
        request_context = self.task_context(
            purpose="review" if completion_review else "action"
        )
        native_snapshot = None
        if (
            not completion_review
            and "type" in tools
            and (self.task_state.contract or {}).get("schema_version") == 2
            and self.task_state.goal == self.goal
            and getattr(self.config, "native_text_references", False)
            and getattr(self.visual_fallback, "supports_native_text_references", False)
        ):
            native_snapshot = self.task_state.native_text_snapshot()
            request_context["native_text_sources"] = native_snapshot["index"]
        result = await self.visual_fallback.decide(
            goal=self.goal,
            screenshot=screenshot,
            ui_state=ui,
            available_tools=tools,
            apps=self.apps or [],
            recent_actions=self.history[-8:],
            feedback=self.feedback,
            guidance=self.instructions + "\nApp-specific hints:\n" + self.app_card,
            task_state=request_context,
            completion_review=completion_review,
            review_history_images=getattr(self.config, "review_history_images", True),
            recovery=recovery,
            **({"native_only_review": True} if native_only_review else {}),
            **({"corrective_only": True} if corrective_only else {}),
            **(
                {"derive_review_dependencies": True}
                if self._derive_review_dependencies()
                else {}
            ),
            **(
                {"native_text_sources": native_snapshot["private"]}
                if native_snapshot is not None
                else {}
            ),
            **(
                {"native_text_edits": True}
                if native_snapshot is not None
                and getattr(self.config, "native_text_edits", False) is True
                and getattr(self.visual_fallback, "supports_native_text_edits", False)
                is True
                else {}
            ),
            **(
                {"native_text_patches": True}
                if native_snapshot is not None
                and getattr(self.config, "native_text_patches", False) is True
                and getattr(self.visual_fallback, "supports_native_text_patches", False)
                is True
                else {}
            ),
            review_feedback=self.review_diagnostic
            + ("\n" if self.review_diagnostic else "")
            + (
                "The previous typed review was invalid. Return one JSON object with complete, evidence, "
                "checks, final_answer, assessments. Every assessment has id, status "
                "(supported/contradicted/unknown), evidence, evidence_refs. Cite actual specific "
                "O#:n# controls or the explicitly uploaded O#:image# screenshots; do not cite a whole observation to prove an effect. "
                + (
                    "Historical screenshots prove their recorded time, not a final-page requirement. "
                    if self._derive_review_dependencies()
                    else "Historical screenshots cannot go in verification_refs or prove a final-page requirement. "
                )
                + "Only explicit final-page requirements belong in checks; otherwise {} is valid. "
                "Negative query answers do not make an answered requirement unsupported."
                if completion_review
                and attempt
                and (self.task_state.contract or {}).get("schema_version") == 2
                else (
                    "Your previous reply was invalid. Return JSON with checks, evidence, missing, "
                    "final_answer (the actual answer for a query, null for an action-only task), "
                    "complete, and assessments. Include every requirement id exactly once with "
                    "satisfied and its own evidence; a missing assessment cannot count as success. "
                    "On an accessible page, checks must contain the final-state "
                    "requirements from the goal, including every requested final label and exact input. "
                    "Allowed checks keys ONLY: package, text_all, text_any, "
                    "input_text, checked. Put required visible words in text_all; do not invent "
                    "boolean fields. Do not return empty checks when claiming completion. "
                    "The foreground package may differ from the app launched earlier. "
                    "Unless the goal requires a different final package, copy the observed package "
                    "if including that check: "
                    + str(ui.phone_state.get("packageName", "unknown"))
                    if completion_review and attempt
                    else ""
                )
            ),
        )
        if corrective_only and (
            result.status == "done"
            or any(call.name == "complete" for call in result.calls)
        ):
            result = replace(
                result,
                status="error",
                calls=(),
                assessments=(),
                message="Visual model returned an invalid or unavailable action.", format_error=True,
                error_code="completion_disallowed",
                error_field="action",
                final_answer=None,
            )
        if result.assessments:
            self.record(
                "completion_assessment",
                model=result.model,
                requirements=result.assessments,
            )
        if result.status == "done" and result.completion_checks is not None:
            checks = result.completion_checks
            try:
                validate_completion_criteria(checks)
                matched = (
                    (bool(getattr(result, "evidence_refs", ())) and not checks)
                    or (not ui.elements and not checks)
                    or completion_matches(ui, checks)
                )
            except ValueError:
                matched = False
                result = replace(
                    result,
                    status="error",
                    message="Completion reviewer returned invalid evidence.", format_error=True,
                )
            self.record(
                "evidence_check", model=result.model, checks=checks, success=matched
            )
            if not matched and result.status != "error":
                result = replace(
                    result,
                    status="incomplete",
                    message="Current controls do not satisfy the required final-state checks: "
                    + json.dumps(checks, ensure_ascii=False),
                )
        error_refs = ()
        proposed_error_refs = getattr(result, "error_refs", ())
        if isinstance(proposed_error_refs, (list, tuple)) and proposed_error_refs:
            known_refs = self.task_state.evidence()
            error_refs = tuple(
                dict.fromkeys(
                    ref
                    for ref in proposed_error_refs
                    if isinstance(ref, str)
                    and re.fullmatch(r"O\d+:(?:n|image)\d+", ref)
                    and ref in known_refs
                )
            )[:16]
        argument_diagnostic = None
        if getattr(result, "error_code", "") == "unknown_argument":
            from mobilerun.agent.decision.visual_fallback import action_argument_diagnostic

            argument_diagnostic = action_argument_diagnostic(result)
        self.record(
            "decision",
            source="autoglm",
            model=result.model,
            status=result.status,
            reason=result.message,
            error_code=getattr(result, "error_code", ""),
            error_field=getattr(result, "error_field", ""),
            **({"error_refs": list(error_refs)} if error_refs else {}),
            **(
                {"argument_diagnostic": argument_diagnostic}
                if argument_diagnostic is not None
                else {}
            ),
            latency_ms=round((time.monotonic() - started) * 1000, 1),
        )
        for assessment in result.assessments:
            if "status" in assessment:
                self.task_state.assessments[assessment["id"]] = dict(assessment)
        if result.assessments:
            self.task_state.progress.update(self.task_state)
        if (
            result.status == "error"
            and attempt == 0
            and getattr(result, "format_error", False)
        ):
            # Use the existing reasoner for the single bounded retry. Phone
            # models can repeat translated app labels instead of installed IDs.
            retry_recovery = recovery or (
                not completion_review
                and bool(getattr(self.config, "reasoning_model", None))
            )
            self.feedback += (
                "\nReturn the required JSON object with complete, evidence, missing, checks, "
                "and assessments covering every requirement id."
                if completion_review
                else (
                    "\nThe previous response was rejected before any device input. Return one "
                    "JSON action object with the documented action name and flat argument fields."
                    if (retry_recovery or typing_only) and self.config.reasoning_model
                    else "\nThe previous model response was rejected before any device input. "
                    "Return exactly one short do(...) or finish(...) call, with no explanation. "
                    "For Launch, copy the exact package_name from the installed apps."
                )
            )
            self.record("format_retry", source="autoglm", action_executed=False)
            if corrective_only and result.error_code == "completion_disallowed":
                self.feedback += (
                    "\nThe correction-stage completion claim was rejected; no device input occurred. "
                    "Finish is unavailable in this request. Choose one valid normal tool action "
                    "to inspect or correct the unmet requirement, or Take_over with the blocker. "
                    "Do not invent facts or replay a submit/send/delete with an unknown outcome."
                )
            if not completion_review and argument_diagnostic is not None:
                self.feedback += (
                    "\nThe rejected action's argument schema is "
                    + json.dumps(argument_diagnostic, ensure_ascii=False)
                    + ". These flat arguments are in addition to action and the optional progress string. "
                    "Use only progress for an unverified progress note; do not repeat conflicting progress fields. "
                    "Re-derive one action from the fresh observation. Do not delete a target or constraint "
                    "merely to make the previous response parse. Type edits the already focused field; "
                    "if another target is needed, first choose a separate Tap using a current control."
                )
            if not completion_review and result.error_code == "coordinate_invalid":
                self.feedback = (
                    "The previous action was rejected because its coordinate declaration or point was invalid. "
                    "No device input occurred. Locate the target on the new screenshot. For a JSON Tap, prefer "
                    "an actual current control index. For a visual JSON point, declare coordinate_space=screenshot_pixels "
                    "and use integer pixels inside the new screenshot, or explicitly declare normalized_1000 and 0..999 integers. "
                    "Phone do actions always use 0..999. Do not copy native bounds or infer units from their magnitude.\n"
                    + self.feedback[-2000:]
                )
            if completion_review:
                self.review_diagnostic = (
                    "Previous review rejected: "
                    + getattr(result, "error_code", "")
                    + "/"
                    + getattr(result, "error_field", "")
                )
                if result.error_code == "query_scope_unverified":
                    self.review_diagnostic += (
                        ". Recheck the original query_scope and the actual cited data source. "
                        "When using a Q#, scope_refs cites its actual scope_source.container_ref and relevant context_refs; "
                        "constraint_evidence cites only native context_refs or filter_evidence_refs for that query. "
                        "Q# belongs in assessment.evidence_refs, not scope_refs or constraint_evidence; "
                        "a computed number does not prove a filter. For a direct visual answer without a Q#, "
                        "scope_refs and constraint_evidence may cite the actually uploaded O#:image# supporting "
                        "the observed range and value. Do not invent a native control or collection to answer from a screenshot. "
                        "Cover every full original constraints quote verbatim. Keep every used native or image range ref in the "
                        "same assessment.evidence_refs. Current questions cannot use an old Q source; "
                        "if the intended range or data is unknown, mark unknown and complete=false. "
                        "Do not change the requested range or invent evidence to satisfy the schema."
                    )
                elif result.error_code == "query_property_mismatch":
                    self.review_diagnostic += (
                        ". The asked property is object category, not checkbox completion. "
                        "Inspect the identified object's contentDescription and accessibility_type_hint. "
                        "An explicit matching category prefix supports membership even when unchecked. "
                        "Re-derive the answer for the correct identified object; do not copy your previous answer."
                    )
                    self.review_diagnostic += (
                        " Include all supporting current object identity/category controls in each assessment's evidence_refs. "
                        "The client constructs the complete live recheck dependencies from these validated references."
                        if self._derive_review_dependencies()
                        else " Include the current object identity/category controls in both evidence_refs and verification_refs. "
                        "Query controls still need rechecking; do not return an empty verification_refs array."
                    )
                elif (
                    result.error_code == "effect_unverified"
                    and result.error_field
                    in {
                        "evidence_refs",
                        "content_evidence",
                        "post_action_observation",
                        "persistent_result",
                        "reopen_trace",
                        "result_refs",
                    }
                ) or (
                    result.error_code == "evidence_unknown"
                    and result.error_field
                    in {
                        "result_refs",
                        "reopen_trace",
                    }
                ):
                    self.review_diagnostic += (
                        ". Check every supported non-process requirement separately. "
                        "A# action references alone cannot prove a saved value, resulting state, artifact "
                        "or query answer; input dispatch does not prove persistence. Cite actual specific "
                        "O#:n# controls or uploaded O#:image# screenshots supporting that requirement, "
                        "or a complete deterministic query result where applicable. Evidence for another "
                        "requirement does not fill this gap. Preserve object identity, requested values "
                        "and coverage. If the needed result is not observable, mark that requirement unknown "
                        "and complete=false so the agent can inspect the result. Do not invent references, "
                        "reuse unavailable images or change a state requirement into a process requirement."
                    )
                    if result.error_field == "post_action_observation":
                        self.review_diagnostic += (
                            " A cited action's result needs a specific observation taken at or after its "
                            "recorded after boundary. A pre-dispatch page cannot prove that effect. "
                            "Preserve before/after comparison evidence, but inspect the actual result "
                            "instead of replaying a send, submit, save or delete with an unknown outcome. "
                            "A calculated collection must use post-action native sources throughout."
                        )
                    if result.error_field in {
                        "persistent_result",
                        "reopen_trace",
                        "result_refs",
                    }:
                        self.review_diagnostic += (
                            " A persistent requirement needs result_refs naming actual saved-result evidence. "
                            "A field value alone proves input. A reopened editor needs the actual before/away/returned "
                            "observations and dispatched departure/return actions, with the original field absent while away, "
                            "the exact value reread after return and no intervening typing. Verify the same business object "
                            "semantically; a reusable resource ID is not a business identity. If the result is still unknown, "
                            "return unknown and complete=false, then inspect using normal actions. Do not replay an unknown side effect."
                            " Include each trace reference (before, away, returned and both actions) in that same "
                            "assessment's evidence_refs; top-level or other-assessment references do not count. "
                            "Every result_refs entry must also belong to that assessment's evidence_refs. "
                            "For an editor reread, result_refs uses the returned field, not the before input or action. "
                            "Keep historical inputs in evidence_refs for comparison, never as saved result_refs. "
                            "Use only references actually present in this request; do not repair a nonexistent trace."
                        )
                elif result.error_field == "final_answer" and result.error_code in {
                    "protocol_error",
                    "answer_missing",
                }:
                    self.review_diagnostic += (
                        ". final_answer is the actual answer text, separate from the completion evidence. "
                        "A completed query requires a non-empty string matching the user's requested format; "
                        "encode an evidenced scalar answer as text. Negative or empty query results are "
                        "valid answers, not missing answers: express the observed result explicitly. "
                        "Use null for action-only or incomplete tasks. If the answer is not supported, "
                        "mark the unresolved requirement unknown and complete=false. Do not guess an "
                        "answer or manufacture results merely to satisfy this field."
                    )
                elif result.error_field in {
                    "verification_refs",
                    "omitted_state_property",
                    "state_verification_refs",
                }:
                    self.review_diagnostic += (
                        ". Cite all supporting current stable controls and actual uploaded screenshots in each assessment's evidence_refs; do not separately select verification_refs. "
                        "Keep range values, input values, checkbox properties and user-requested exact numbers."
                        if self._derive_review_dependencies()
                        else ". Include verification_refs with the current stable state controls and any cited current screenshots supporting each requirement. "
                        "Keep range values, input values, checkbox properties and user-requested exact numbers."
                    )
                    if error_refs:
                        self.review_diagnostic += (
                            " Actual references omitted from the previous reply: "
                            + json.dumps(error_refs)
                            + ". These IDs describe the previous reply. If the new screenshot has a newer ref, "
                            + (
                                "use the newly uploaded screenshot_evidence.ref in the assessment's evidence_refs. "
                                if self._derive_review_dependencies()
                                else "use the newly uploaded screenshot_evidence.ref in evidence_refs and verification_refs. "
                            )
                            + "Do not reuse an old image ref or claim unavailable evidence."
                        )
            return await self._visual(
                ui,
                attempt=1,
                completion_review=completion_review,
                recovery=retry_recovery,
                native_only_review=native_only_review,
                corrective_only=corrective_only,
                verify_effect=verify_effect,
            )
        if result.status == "verified":
            return await self.resolve_effect(ui, verify_effect, result)
        if verify_effect is not None and result.status in {"error", "done"}:
            # Malformed replies or a premature Finish spend verification budget
            # instead of ending the task while an effect is still unknown.
            self.last_feedback = (
                f"Answer the verification of {verify_effect.id} with a single Verify JSON object "
                "or one read-only action."
            )
            return await self._verify_effect(ui, verify_effect)
        if result.status == "incomplete":
            self.review_rejections += 1
            self.record(
                "verification",
                source="autoglm",
                model=result.model,
                success=False,
                reason=result.message,
            )
            if self.review_rejections >= 2:
                self.pending = Selection(
                    "autoglm",
                    [],
                    reason="completion_unverified: " + result.message,
                    stop=True,
                )
                return self.pending
            self.feedback = (
                "The independent completion review found unmet requirements: "
                + result.message
                + "\nContinue with a corrective action based on the current screen. "
                "If the task cannot be completed, use Take_over and describe the blocker."
            )
            # Correct a rejected completion using the existing reasoner instead
            # of priming the action model to repeat its earlier completion claim.
            self.record(
                "recovery",
                source="autoglm",
                reason="completion_review_rejected",
                action_executed=False,
            )
            corrective = bool(
                getattr(self.config, "semantic_recovery", True)
                and (self.task_state.contract or {}).get("schema_version") == 2
                and getattr(self.visual_fallback, "supports_corrective_actions", False)
            )
            return await self._visual(ui, recovery=True, corrective_only=corrective)
        if result.status != "action":
            if result.status == "done":
                if not completion_review:
                    self.feedback = (
                        "Final review: EVERY requirement must be supported by the NEW screen "
                        "and actual actions. The previous completion claim may be WRONG: "
                        + result.message[:2000]
                        + "\n请逐条核对用户目标，特别是最终停留位置、禁止操作、数量和未完成事项。"
                        "曾经到达一个页面不等于现在仍停留在那里。与当前画面不符时继续纠正操作，"
                        "不要照抄上次的完成声明。只在全部符合时finish，否则执行下一步或请求协助。"
                    )
                    self.record("completion_review", source="autoglm")
                    return await self._visual(ui, completion_review=True)
                if not self.has_completion_check:
                    self.pending = self.completion(
                        ui,
                        source="autoglm",
                        message=result.message,
                        visual_fingerprint=visual_fingerprint,
                        final_answer=result.final_answer,
                    )
                    self.pending.completion_checks = result.completion_checks
                    self.bind_completion_evidence(result)
                    return self.pending
                fresh = await self.state_provider.get_state()
                self.ctx.ui = fresh
                verified = await self.verified(fresh)
                self.record("verification", source="autoglm", success=verified)
                if verified:
                    self.pending = self.completion(
                        ui,
                        message=result.message,
                        visual_fingerprint=visual_fingerprint,
                        final_answer=result.final_answer,
                    )
                    self.pending.completion_checks = result.completion_checks
                    self.bind_completion_evidence(result)
                    return self.pending
            reason = (
                "completion_unverified" if result.status == "done" else result.message
            )
            self.pending = Selection("autoglm", [], reason=reason, stop=True)
        else:
            calls = [ToolCall(call.name, dict(call.arguments)) for call in result.calls]
            if verify_effect is not None and (
                getattr(result, "effect", "")
                or self._effect_key(calls[0], ui) == verify_effect.key
            ):
                # Each rejected replay spends verification budget; it is never executed.
                self.record(
                    "effect_verification",
                    effect=verify_effect.id,
                    status="replay_rejected",
                )
                self.last_feedback = (
                    f"Rejected: {verify_effect.action_ref} must not be repeated while verifying. "
                    "If it did not happen, output Verify with status not_happened; "
                    "one retry is then allowed."
                )
                return await self._verify_effect(ui, verify_effect)
            self.pending = self.accept_fallback(
                calls, ui, "autoglm", visual_fingerprint
            )
            self.pending.navigation_context = navigation_context
            self.pending.progress = getattr(result, "progress", "")
            self.pending.expected_effect = getattr(result, "effect", "")
            self.pending.input_source = copy.deepcopy(
                getattr(result, "input_source", None)
            )
            if (
                self.pending.stop
                and attempt == 0
                and self.pending.reason == "Repeated action without observable progress"
            ):
                self.feedback += (
                    "\nThat action was already executed without the required effect. It was NOT replayed. "
                    "Inspect the current screen and choose a different way forward; use Wait only "
                    "if loading is observable, or report the blocker."
                )
                self.record(
                    "recovery",
                    source="autoglm",
                    reason=self.pending.reason,
                    action_executed=False,
                )
                return await self._visual(
                    ui,
                    attempt=1,
                    completion_review=completion_review,
                    recovery=True,
                    verify_effect=verify_effect,
                )
        return self.pending

    def bind_completion_evidence(self, result):
        if not result.evidence_refs:
            return
        current = self.task_state.current_id
        known = self.task_state.evidence()
        refs = (
            result.verification_refs
            if result.verification_refs is not None
            else result.evidence_refs
        )
        self.pending.completion_proofs = [
            known[ref] for ref in refs if ref.startswith(current + ":n")
        ]
        self.pending.completion_observation_id = current
        query_refs = [
            ref
            for ref in result.evidence_refs
            if known.get(ref, {}).get("source") == "deterministic_query"
        ]
        if query_refs and self._native_query_bindings_enabled():
            self.pending.completion_query_binding_required = True
            try:
                self.pending.completion_query_binding = bind_native_queries(
                    query_refs, self.task_state, current, goal=self.goal
                )
            except (NativeQueryBindingError, TypeError, ValueError, KeyError) as exc:
                self.pending.completion_query_binding = {"invalid": str(exc)}
        if (
            self.pending.completion_proofs
            and (self.task_state.contract or {}).get("schema_version") == 2
            and self.task_state.goal == self.goal
            and getattr(self.config, "native_proof_bindings", False) is True
            and getattr(self.visual_fallback, "supports_native_proof_bindings", False)
            is True
        ):
            self.pending.completion_native_binding_required = True
            try:
                self.pending.completion_native_binding = bind_native_proofs(
                    self.pending.completion_proofs, known[current], goal=self.goal
                )
            except (NativeProofBindingError, ValueError, TypeError, KeyError) as exc:
                # An enabled but invalid binding cannot silently fall back to
                # weaker proof matching. The existing validation/recovery owns it.
                self.pending.completion_native_binding = {"invalid": str(exc)}
        images = [known[ref] for ref in refs if ref.startswith(current + ":image")]
        if images:
            self.pending.completion_image_guard = images[-1]["sha256"]
        self.record(
            "completion_guard_binding",
            source=self.pending.source,
            reviewed_observation_id=current,
            verification_refs=list(refs),
            native_proofs=copy.deepcopy(self.pending.completion_proofs),
            native_object_binding=copy.deepcopy(self.pending.completion_native_binding),
            **(
                {
                    "native_query_binding": copy.deepcopy(
                        self.pending.completion_query_binding
                    )
                }
                if self.pending.completion_query_binding is not None
                else {}
            ),
            image_proofs=copy.deepcopy(images),
            reviewed_package=self.pending.snapshot.phone_state.get("packageName"),
            reviewed_state_fingerprint=snapshot_fingerprint(self.pending.snapshot),
        )

    def _native_query_bindings_enabled(self):
        return (
            (self.task_state.contract or {}).get("schema_version") == 2
            and getattr(self.config, "native_query_bindings", False) is True
            and getattr(self.visual_fallback, "supports_native_query_bindings", False)
            is True
        )

    def record_completion_validation(
        self,
        selection,
        current,
        *,
        reason=None,
        failed_check=None,
        observed_image_sha256=None,
    ):
        """Retain actual guard inputs; diagnostics never decide task success."""
        self.last_completion_failure = failed_check if reason else None
        self.record(
            "completion_guard",
            source=selection.source,
            success=reason is None,
            reason=reason,
            failed_check=failed_check,
            reviewed_observation_id=selection.completion_observation_id,
            reviewed_package=selection.snapshot.phone_state.get("packageName"),
            observed_package=current.phone_state.get("packageName"),
            reviewed_state_fingerprint=snapshot_fingerprint(selection.snapshot),
            observed_state_fingerprint=snapshot_fingerprint(current),
            explicit_checks=copy.deepcopy(self.config.completion),
            reviewed_checks=copy.deepcopy(selection.completion_checks),
            reviewed_native_proofs=copy.deepcopy(selection.completion_proofs),
            reviewed_native_object_binding=copy.deepcopy(
                selection.completion_native_binding
            ),
            observed_native_nodes=observed_nodes(current),
            expected_image_sha256=selection.completion_image_guard
            or selection.visual_fingerprint,
            observed_image_sha256=observed_image_sha256,
        )
        return current, reason

    def recover_completion_validation(self, reason) -> bool:
        """Request a fresh review within the existing rejection and step budgets."""
        selection = self.pending
        if (
            reason != "completion_unverified"
            or not getattr(self.config, "semantic_recovery", True)
            or self.config.fallback != "visual"
            or self.visual_fallback is None
            or selection is None
            or selection.source != "autoglm"
            or selection.completion_proofs is None
            or len(selection.calls) != 1
            or selection.calls[0].name != "complete"
            or selection.calls[0].parameters.get("success") is not True
            or self.last_completion_failure
            not in {
                "current_image",
                "native_evidence",
                "foreground_package",
                "reviewed_postconditions",
            }
        ):
            return False
        self.review_rejections += 1
        if self.review_rejections >= 2:
            return False
        self.feedback = (
            "Completion was NOT accepted because a reviewed state dependency changed: "
            + self.last_completion_failure
            + ". No complete tool or device action was executed. Review the fresh observation "
            "and explicitly uploaded current screenshot again. Use actual native property references "
            "when they fully establish the requirement; cite the newest uploaded image if visual "
            "evidence is necessary. "
            + (
                "Cite all required state properties in each assessment's evidence_refs; the client constructs the live guard. "
                if self._derive_review_dependencies()
                else "Keep all required state properties in verification_refs. "
            )
            + "Do not reuse stale image references or replay actions whose effects are already observed. "
            "If any requirement is unmet, report it accurately and choose a corrective action."
        )
        self.review_diagnostic = ""
        self.completion_refresh_pending = True
        self.completion_refresh_native = bool(
            self.last_completion_failure == "current_image"
            and selection.completion_proofs
            and getattr(self.visual_fallback, "supports_native_review", False)
        )
        self.record(
            "recovery",
            source="completion_guard",
            reason=self.last_completion_failure,
            action_executed=False,
            review_rejections=self.review_rejections,
        )
        self.pending = None
        return True

    def accept_fallback(self, calls, ui, source="llm", visual_fingerprint=None):
        if len(calls) != 1:
            return Selection(
                source,
                [],
                reason="Decision-first mode requires exactly one tool call per step",
                stop=True,
            )
        call = calls[0]
        if call.error or call.name not in self.registry.tools:
            return Selection(
                source,
                [],
                reason="Fallback returned an invalid or unavailable tool",
                stop=True,
            )
        effect = self.effects.pending()
        if (
            effect is not None
            and call.name != "complete"
            and self._effect_key(call, ui) == effect.key
        ):
            return Selection(
                source,
                [],
                reason=f"{effect.id} is unverified; it was not replayed.",
                stop=True,
            )
        if call.name != "complete" and (
            self.signature(call, ui, visual_fingerprint) in self.executed
            or self.action_signatures[-12:].count(
                self.signature(call, ui, visual_fingerprint)
            )
            >= 2
        ):
            return Selection(
                source,
                [],
                reason="Repeated action without observable progress",
                stop=True,
            )
        return Selection(
            source,
            calls,
            snapshot=ui,
            visual_fingerprint=visual_fingerprint,
            coordinate_guard=self.coordinate_guard(call, ui),
            input_guard=focused_input_observation(ui) if call.name == "type" else None,
        )

    async def validate_before_execute(self):
        """Return fresh state or a reason; no mutation has occurred at this point."""
        selection = self.pending
        current = await self.state_provider.get_state()
        call = selection.calls[0]
        if selection.input_source:
            from mobilerun.agent.decision.native_text import (
                validate_native_text_receipt,
            )

            if (
                call.name != "type"
                or self.task_state.goal != self.goal
                or not validate_native_text_receipt(
                    self.task_state.observations,
                    self.goal,
                    selection.input_source,
                    call.parameters.get("text"),
                )
            ):
                self.stale_count += 1
                self.last_feedback = "The captured native text source changed or could not be bound; no device input was dispatched. Reobserve the intended source and target."
                self.record(
                    "stale",
                    source=selection.source,
                    reason="native_text_source_changed",
                )
                return current, "native_text_source_changed"
        try:
            expected_owner = (
                native_action_owner(selection.snapshot, call)
                if selection.snapshot is not None
                else None
            )
            if (
                selection.native_target_binding is not None
                or expected_owner is not None
            ):
                if (
                    selection.native_target_binding != expected_owner
                    or native_action_owner(current, call) != expected_owner
                ):
                    raise ValueError("native_target_owner_changed")
        except (ValueError, TypeError, KeyError):
            self.stale_count += 1
            self.last_feedback = "The native record owning this control changed or could not be bound. No action was dispatched. Reobserve its full name and qualifiers before selecting the intended record."
            self.record(
                "stale", source=selection.source, reason="native_target_owner_changed"
            )
            return current, "native_target_owner_changed"
        if call.name == "complete":
            if not call.parameters.get("success"):
                return current, None
            if self.has_completion_check:
                if not await self.verified(current):
                    return self.record_completion_validation(
                        selection,
                        current,
                        reason="completion_unverified",
                        failed_check="explicit_postconditions",
                    )
            elif not str(call.parameters.get("message", "")).strip():
                return self.record_completion_validation(
                    selection,
                    current,
                    reason="completion_unverified",
                    failed_check="completion_message",
                )
            query_binding_required = (
                selection.completion_query_binding_required
                or bool(
                    self._native_query_bindings_enabled()
                    and assessed_query_refs(self.task_state)
                )
            )
            if query_binding_required:
                nodes = observed_nodes(current)
                for node in nodes:
                    node["ref"] = "V:n" + str(node["path"])
                fresh_query_observation = {
                    "id": "V",
                    "source": "accessibility",
                    "nodes": nodes,
                    "phone": current.phone_state,
                }
                if not corroborate_native_queries(
                    selection.completion_query_binding,
                    self.task_state,
                    selection.completion_observation_id,
                    fresh_query_observation,
                    goal=self.goal,
                ):
                    self.record(
                        "native_query_binding_rejected",
                        source=selection.source,
                        action_executed=False,
                    )
                    return self.record_completion_validation(
                        selection,
                        current,
                        reason="completion_unverified",
                        failed_check="native_evidence",
                    )
            if selection.completion_proofs is not None:
                if current.phone_state.get(
                    "packageName"
                ) != selection.snapshot.phone_state.get("packageName"):
                    return self.record_completion_validation(
                        selection,
                        current,
                        reason="completion_unverified",
                        failed_check="foreground_package",
                    )
                if not corroborate_nodes(selection.completion_proofs, current):
                    return self.record_completion_validation(
                        selection,
                        current,
                        reason="completion_unverified",
                        failed_check="native_evidence",
                    )
                binding_required = selection.completion_native_binding_required or bool(
                    selection.completion_proofs
                    and (self.task_state.contract or {}).get("schema_version") == 2
                    and getattr(self.config, "native_proof_bindings", False) is True
                    and getattr(
                        self.visual_fallback, "supports_native_proof_bindings", False
                    )
                    is True
                )
                if binding_required and (
                    self.task_state.goal != self.goal
                    or selection.completion_native_binding is None
                    or not corroborate_native_binding(
                        selection.completion_native_binding,
                        selection.completion_proofs,
                        self.task_state.evidence().get(
                            selection.completion_observation_id
                        ),
                        observed_nodes(current),
                        current.phone_state.get("packageName"),
                        goal=self.goal,
                    )
                ):
                    self.record(
                        "native_object_binding_rejected",
                        source=selection.source,
                        action_executed=False,
                    )
                    return self.record_completion_validation(
                        selection,
                        current,
                        reason="completion_unverified",
                        failed_check="native_evidence",
                    )
                if selection.completion_checks and not completion_matches(
                    current, selection.completion_checks
                ):
                    return self.record_completion_validation(
                        selection,
                        current,
                        reason="completion_unverified",
                        failed_check="reviewed_postconditions",
                    )
                observed_image_sha256 = None
                if selection.completion_image_guard:
                    image = await self.ctx.driver.screenshot()
                    observed_image_sha256 = hashlib.sha256(image).hexdigest()
                    if observed_image_sha256 != selection.completion_image_guard:
                        return self.record_completion_validation(
                            selection,
                            current,
                            reason="completion_unverified",
                            failed_check="current_image",
                            observed_image_sha256=observed_image_sha256,
                        )
                return self.record_completion_validation(
                    selection, current, observed_image_sha256=observed_image_sha256
                )
            if selection.completion_checks:
                # Recheck the reviewed outcome, rather than an entire frame
                # containing unrelated clocks or continuously changing readouts.
                if current.phone_state.get(
                    "packageName"
                ) != selection.snapshot.phone_state.get("packageName"):
                    return self.record_completion_validation(
                        selection,
                        current,
                        reason="completion_unverified",
                        failed_check="foreground_package",
                    )
                if not completion_matches(current, selection.completion_checks):
                    return self.record_completion_validation(
                        selection,
                        current,
                        reason="completion_unverified",
                        failed_check="reviewed_postconditions",
                    )
                if set(selection.completion_checks) & {
                    "text_all",
                    "text_any",
                    "input_text",
                    "checked",
                }:
                    return self.record_completion_validation(selection, current)
            # Autonomous completion still belongs to the observed state.
        observed_image_sha256 = None
        if selection.guard:
            reason = selection.guard.validate(current)
        elif selection.input_guard:
            reason = (
                None
                if focused_input_observation(current) == selection.input_guard
                else "input_focus_or_value_changed"
            )
        elif selection.coordinate_guard:
            reason = (
                None
                if self.coordinate_guard(call, current) == selection.coordinate_guard
                else "stale_fallback_target"
            )
        else:
            reason = (
                None
                if snapshot_fingerprint(current)
                == snapshot_fingerprint(selection.snapshot)
                else "stale_fallback_state"
            )
            if not reason and selection.visual_fingerprint:
                if call.name == "system_button" and selection.navigation_context:
                    # Cursor animation does not change a navigation target.
                    if await self.navigation_context() != selection.navigation_context:
                        reason = "stale_navigation_context"
                else:
                    image = await self.ctx.driver.screenshot()
                    observed_image_sha256 = hashlib.sha256(image).hexdigest()
                    if observed_image_sha256 != selection.visual_fingerprint:
                        reason = "stale_visual_frame"
        if reason:
            self.stale_count += 1
            self.last_feedback = "The previous proposed action was NOT executed because the observation changed. Re-evaluate the current screen."
            self.record("stale", source=selection.source, reason=reason)
        else:
            self.stale_count = 0
            if call.name == "type" and call.parameters.get("clear"):
                guard = getattr(selection.guard, "target_guard", None)
                index = guard.index if guard else call.parameters.get("index")
                self.input_target = await capture_input_target(
                    self.ctx.driver, current, index=index
                )
                if self.input_target is None:
                    return current, "input_target_unverifiable"
        if call.name == "complete" and call.parameters.get("success"):
            return self.record_completion_validation(
                selection,
                current,
                reason=reason,
                failed_check=reason,
                observed_image_sha256=observed_image_sha256,
            )
        return current, reason

    def record_execution(self, call, result, ui):
        # Store before any read-back, including an uncertain tool result.
        self.review_rejections = 0
        self.executed.add(self.signature(call, ui, self.pending.visual_fingerprint))
        self.last_executed_state = (
            self.pending.visual_fingerprint or snapshot_fingerprint(ui)
        )
        entry = {
            "source": self.pending.source,
            "tool": call.name,
            "arguments": call.parameters,
            "success": result.success,
            "summary": result.summary,
            "package": ui.phone_state.get("packageName"),
        }
        self.history.append(entry)
        if self.pending.input_source:
            entry["input_source"] = copy.deepcopy(self.pending.input_source)
        recorded_action = self.task_state.record_action(
            call,
            result,
            **(
                {"input_source": self.pending.input_source}
                if self.pending.input_source
                else {}
            ),
            **(
                {"input_target": self.input_target}
                if call.name == "type" and call.parameters.get("clear")
                else {}
            ),
            **(
                {"native_target_binding": self.pending.native_target_binding}
                if self.pending.native_target_binding is not None
                else {}
            ),
        )
        if "native_target_binding" in recorded_action:
            entry["native_target_binding"] = copy.deepcopy(
                recorded_action["native_target_binding"]
            )
        self.action_signatures.append(
            self.signature(call, ui, self.pending.visual_fingerprint)
        )
        if result.success and self.pending.progress:
            self.progress_note = self.pending.progress
        if result.success:
            self.track_effect(call, ui, recorded_action)
        self.record("progress", **self.task_context())
        self.last_feedback = "Actually executed: " + call.name + ". " + result.summary
        self.record("action", **entry)

    def _effect_key(self, call, ui):
        return effect_key(call, ui.phone_state.get("packageName", ""), ui.elements)

    def confirmed_effect_for(self, call, ui):
        key = self._effect_key(call, ui)
        return next(
            (
                e
                for e in reversed(self.effects.entries)
                if e.key == key and e.status == "confirmed"
            ),
            None,
        )

    def track_effect(self, call, ui, action):
        """Record declared side effects and the trail for the structural net."""
        if self.effects.pending() is not None:
            return  # Verification steps are read-only by policy.
        snapshot = self.pending.snapshot or ui
        key = self._effect_key(call, snapshot)
        signature = self.signature(call, snapshot, self.pending.visual_fingerprint)
        declared = bool(self.pending.expected_effect)
        self.effect_trail = self.effect_trail[-3:] + [
            {
                "tool": call.name,
                "fingerprint": snapshot_fingerprint(snapshot)
                if snapshot.elements
                else None,
                "declared": declared,
                "key": key,
                "signature": signature,
                "action": action,
                "at": time.time(),
            }
        ]
        if declared:
            effect = self.effects.record(
                key=key,
                description=self.pending.expected_effect,
                action_ref=action["ref"],
                before=action.get("before"),
                signature=signature,
                source="declared",
            )
            self.record("side_effect", source="declared", **effect.context())

    def _confirm_effect(self, effect, evidence, note=""):
        self.effects.resolve(effect, "confirmed", evidence)
        self.task_state.progress.stalled_actions = 0
        # A confirmed effect is real progress; earlier repeats of the same
        # control (for example on the next image) no longer imply a loop.
        self.executed.clear()
        self.action_signatures.clear()
        self.effect_notes.add(effect.key[0], note)
        self.last_feedback = f"{effect.id} confirmed: {effect.description}."

    async def observe_media_effect(self):
        """After any tap, record new gallery files as confirmed effects.

        App-independent and model-free: it catches saves that nobody declared,
        for example a save chosen by Jev, and flags duplicate bytes.
        """
        if (
            not self.effect_trail
            or self.effect_trail[-1].get("media_checked")
            or "recent_media" not in self.registry.tools
            or self.effects.pending() is not None
        ):
            return
        last = self.effect_trail[-1]
        last["media_checked"] = True
        if last["tool"] not in {"click", "click_at", "double_tap_at"}:
            return
        rows = await self._media_rows(self.ctx, time.time() - last["at"] + 5)
        new = [row for row in rows or [] if row["name"] not in self.media_attributed]
        if not new:
            return
        self.media_attributed.update(row["name"] for row in new)
        last["evaluated"] = True  # A media change supersedes the structural net.
        self.effect_probes += 1
        ref = f"P{self.effect_probes}"
        summary = "; ".join(self._media_text(row) for row in new)
        duplicate = any(row["duplicate_of"] for row in new)
        effect = self.effects.record(
            key=last["key"],
            description=f"new gallery file after tapping '{last['key'][2]}'"
            + (" (DUPLICATE of an earlier save)" if duplicate else ""),
            action_ref=last["action"]["ref"],
            before=last["action"].get("before"),
            signature=last["signature"],
            source="observed",
        )
        effect.probes += ((ref, summary),)
        self.record("effect_probe", effect=effect.id, ref=ref, summary=summary)
        self._confirm_effect(effect, (ref,))
        if duplicate:
            self.last_feedback += (
                " The saved file duplicates an earlier save; move to a different item."
            )
        self.record("side_effect", source="observed", **effect.context())

    def detect_structural_effect(self, ui):
        """Safety net for undeclared effects; it only forbids a blind replay."""
        if not self.effect_trail or self.effect_trail[-1].get("evaluated"):
            return
        last = self.effect_trail[-1]
        last["evaluated"] = True
        if (
            self.effects.pending() is None
            and ui.elements
            and returned_to_previous_screen(self.effect_trail, snapshot_fingerprint(ui))
        ):
            effect = self.effects.record(
                key=last["key"],
                description=f"possible side effect of tapping '{last['key'][2]}': "
                "a menu or dialog closed and the previous page returned",
                action_ref=last["action"]["ref"],
                before=last["action"].get("before"),
                signature=last["signature"],
                source="structural",
            )
            self.record("side_effect", source="structural", **effect.context())

    async def _verify_effect(self, ui, effect):
        now = time.monotonic()
        if effect.verify_started is None:
            effect.verify_started = now
        if self.effects.budget_exhausted(effect, now):
            self.effects.resolve(effect, "unverifiable")
            self.record(
                "effect_verification",
                effect=effect.id,
                status="unverifiable",
                reason="budget_exhausted",
            )
            self.pending = Selection(
                "effects",
                [],
                reason=f"{effect.id} could not be verified within the budget; "
                "stopped without replaying it.",
                stop=True,
            )
            return self.pending
        if effect.verify_steps == 0 and "recent_media" in self.registry.tools:
            # A cheap, app-independent check runs before the model is asked, so
            # the verdict never depends on the model remembering to probe.
            probe = await self.recent_media(ctx=self.ctx)
            self.effect_probes += 1
            ref = f"P{self.effect_probes}"
            effect.probes += ((ref, probe.summary),)
            self.record("effect_probe", effect=effect.id, ref=ref, summary=probe.summary)
        effect.verify_steps += 1
        self.feedback = (
            f"Side-effect verification for {effect.id}: {effect.action_ref} "
            f"(target '{effect.key[2]}') may have had an effect that is not visible: "
            f"{effect.description}. Determine whether it actually happened. Do NOT repeat "
            f"{effect.action_ref} or any other data-changing action. Use Recent_media for photos "
            "or videos, or navigate to where the result is visible, then output Verify with "
            f"evidence recorded after {effect.action_ref}. If the current screen already shows it "
            "did not happen (for example the menu is still open), output Verify not_happened "
            "citing the current observation; one retry is then allowed."
        )
        if effect.probes:
            self.feedback += "\nAutomatic checks after the action (citable): " + "; ".join(
                f"{ref}: {summary}" for ref, summary in effect.probes
            )
        if self.last_feedback:
            self.feedback += "\nPrevious execution outcome: " + self.last_feedback
        self.record(
            "effect_verification",
            effect=effect.id,
            status="probing",
            step=effect.verify_steps,
        )
        return await self._visual(ui, recovery=True, verify_effect=effect)

    async def resolve_effect(self, ui, effect, result):
        verdict = result.verification or {}
        status = verdict.get("status")
        if effect is None:
            self.pending = Selection(
                "effects", [], reason="Verify returned outside verification.", stop=True
            )
            return self.pending
        if status == "unknown":
            self.effects.resolve(effect, "unverifiable")
            self.record(
                "effect_verification",
                effect=effect.id,
                status="unverifiable",
                reason="model_unknown",
            )
            self.pending = Selection(
                "effects",
                [],
                reason=f"{effect.id} could not be verified; stopped without replaying it.",
                stop=True,
            )
            return self.pending
        # Only an explicit reference to a different effect is rejected.
        cited = set(re.findall(r"E[1-9]\d*", str(verdict.get("effect") or "")))
        if cited - {effect.id} or not evidence_valid(
            effect,
            verdict.get("evidence"),
            self.task_state.observations,
            self.task_state.actions,
            {ref for ref, _ in effect.probes},
        ):
            self.record(
                "effect_verification", effect=effect.id, status="evidence_rejected"
            )
            self.last_feedback = (
                f"Verify rejected: cite {effect.id} and observations (O) or execution "
                f"records (A) recorded after {effect.action_ref}."
            )
            return await self._verify_effect(ui, effect)
        if status == "confirmed":
            self._confirm_effect(effect, verdict["evidence"], verdict.get("note", ""))
        else:
            self.effects.resolve(effect, "not_happened", verdict["evidence"])
            if self.effects.may_retry(effect):
                self.executed.discard(effect.signature)
                self.action_signatures = [
                    s for s in self.action_signatures if s != effect.signature
                ]
                self.last_feedback = (
                    f"{effect.id} did not happen; you may retry that action once."
                )
            else:
                self.last_feedback = (
                    f"{effect.id} did not happen again; do not retry it. "
                    "Try a different way or report the blocker."
                )
        for action in self.task_state.actions:
            if action.get("ref") == effect.action_ref:
                action["side_effect"] = {"id": effect.id, "status": effect.status}
        self.record(
            "effect_verification",
            effect=effect.id,
            status=effect.status,
            evidence=list(effect.evidence),
        )
        self.pending = None
        return None

    async def _media_rows(self, ctx, age_seconds: float) -> list[dict] | None:
        """Gallery photos and videos added in the last ``age_seconds``; None if unavailable."""
        shell = getattr(getattr(ctx.driver, "device", None), "shell", None)
        if not callable(shell):
            return None
        try:
            now = int(str(await asyncio.wait_for(shell("date +%s"), 3)).strip())
            # Host and device clocks may differ; anchor on the device clock.
            since = now - int(age_seconds)
            listing = await asyncio.wait_for(
                shell(
                    "content query --uri content://media/external/file "
                    "--projection _display_name:_size:date_added:_data "
                    f'--where "media_type IN (1,3) AND date_added>={since}" '
                    '--sort "date_added DESC"'
                ),
                5,
            )
            rows = []
            for line in str(listing).splitlines()[:6]:
                fields = dict(re.findall(r"(\w+)=(.*?)(?=, \w+=|$)", line.strip()))
                path = fields.get("_data", "")
                if not path.startswith("/storage/"):
                    continue
                digest = str(
                    await asyncio.wait_for(shell(f"md5sum '{path}'"), 5)
                ).split(" ")[0]
                name = fields.get("_display_name")
                earlier = self.media_seen.setdefault(digest, name)
                rows.append(
                    {
                        "name": name,
                        "size": fields.get("_size"),
                        "md5": digest,
                        # Identical bytes to another file from this task.
                        "duplicate_of": earlier if earlier != name else None,
                    }
                )
            return rows
        except Exception:
            return None

    @staticmethod
    def _media_text(row: dict) -> str:
        return f"{row['name']} size={row['size']} md5={row['md5']}" + (
            f" DUPLICATE of {row['duplicate_of']}" if row["duplicate_of"] else ""
        )

    async def recent_media(self, *, ctx) -> ActionResult:
        """List gallery photos and videos added after the action being verified."""
        effect = self.effects.pending()
        age = (time.time() - effect.recorded_at) + 10 if effect else 300
        rows = await self._media_rows(ctx, age)
        if rows is None:
            return ActionResult(
                success=False, summary="Gallery listing failed; no device state changed."
            )
        self.media_attributed.update(row["name"] for row in rows)
        rows = [self._media_text(row) for row in rows]
        if not rows:
            return ActionResult(
                success=True, summary="No photos or videos were added after the action."
            )
        return ActionResult(
            success=True,
            summary=f"{len(rows)} gallery item(s) added after the action: "
            + "; ".join(rows),
        )

    async def verify_input(self, call):
        if call.name != "type":
            return True
        rich = (self.task_state.contract or {}).get("schema_version") == 2
        matched = True
        if call.parameters.get("clear"):
            expected = call.parameters["text"]
            if self.input_target is None:
                return False
            deadline = time.monotonic() + 3
            while True:
                current = await self.state_provider.get_state()
                self.ctx.ui = current
                if rich:
                    self.task_state.observe(current)
                    self.record_collection_diagnostics()
                    self.last_input_readback = await read_input_value(
                        self.ctx.driver, current, self.input_target, expected
                    )
                    matched = self.last_input_readback.status == "confirmed"
                else:
                    matched = await input_value_matches(
                        self.ctx.driver, current, self.input_target, expected
                    )
                if matched:
                    self.task_state.record_input_effect("confirmed")
                    self.record("input_verification", success=True)
                    break
                if time.monotonic() >= deadline:
                    if rich:
                        self.task_state.record_input_effect(
                            self.last_input_readback.status
                        )
                        self.record(
                            "input_diagnostic",
                            status=self.last_input_readback.status,
                            reason=self.last_input_readback.reason,
                            observed_length=self.last_input_readback.observed_length,
                            expected_length=len(expected),
                        )
                    self.record("input_verification", success=False)
                    if not rich:
                        return False
                    break
                await asyncio.sleep(0.15)
        # Releasing the temporary IME can dismiss text-selection menus. Do it
        # before later actions and final verification, not after task completion.
        restore = getattr(self.ctx.driver, "restore_input_method", None)
        restored = True
        if callable(restore):
            try:
                restored = bool(await restore())
            except Exception:
                restored = False
            self.record("input_method_restoration", success=restored)
        if (
            getattr(self.config, "input_repair", True)
            and rich
            and not matched
            and restored
            and self.last_input_readback.status == "mismatch"
        ):
            key = (
                self.input_target.package_name,
                self.input_target.resource_id,
                self.input_target.class_name,
                call.parameters["text"],
            )
            action = self.task_state.actions[-1] if self.task_state.actions else {}
            same_action = (
                action.get("tool") == call.name
                and action.get("arguments") == call.parameters
            )
            selected_source = self.pending.input_source if self.pending else None
            if selected_source and (
                not same_action or action.get("input_source") != selected_source
            ):
                # A reference cannot become an unreferenced replay if its
                # recorded action or arguments were changed after dispatch.
                return False
            if not self.input_retry_counts.get(key):
                self.input_retry_counts[key] = 1
                self.input_retry = (
                    ToolCall(call.name, dict(call.parameters)),
                    self.input_target,
                )
                # Only carry a source from the same recorded Type. The explicit
                # activation step may add a click to history before replacement.
                self.input_retry_source = (
                    copy.deepcopy(action.get("input_source")) if same_action else None
                )
                self.input_retry_activated = False
        return matched and restored

    async def aclose(self):
        for client in (self.policy, self.visual_fallback):
            if client and hasattr(client, "aclose"):
                try:
                    await client.aclose()
                except Exception:
                    logger.warning(
                        "Decision client cleanup failed; preserving task outcome"
                    )
