"""Track actions whose effect may not be visible on the current screen.

The model declares expected effects and judges verification. Code only blocks
blind replays of unverified effects, budgets verification, and checks that the
cited evidence exists and was recorded after the action.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass
from pathlib import Path

VERIFY_STEP_BUDGET = 5
# A visual-model call takes about 10 s on the GLM relay; allow several probes.
VERIFY_SECONDS_BUDGET = 120.0
RETRY_AFTER_NOT_HAPPENED = 1
# Tools allowed while verifying: navigation and read-only probes, never a commit.
VERIFY_TOOLS = frozenset({"click", "swipe", "system_button", "launch_app", "wait", "recent_media"})
_CLICK_TOOLS = frozenset({"click", "click_at"})
_STATUSES = ("pending", "confirmed", "not_happened", "unverifiable")


def _walk(nodes):
    for node in nodes or []:
        yield node
        yield from _walk(node.get("children"))


def _label(node) -> str:
    texts = [str(n.get("text") or n.get("contentDescription") or "").strip() for n in _walk([node])]
    return " ".join(t for t in texts if t)[:80]


def effect_key(call, package: str, elements) -> tuple:
    """Identify an action by app, tool and target text, not by screen pixels.

    Screenshots change with the status-bar clock; target text does not.
    """
    target = ""
    if call.name == "click":
        node = next(
            (n for n in _walk(elements) if n.get("index") == call.parameters.get("index")),
            None,
        )
        if node is not None:
            target = _label(node) or str(node.get("resourceId") or "")
    if not target:
        target = json.dumps(call.parameters, sort_keys=True, ensure_ascii=False)
    return (package or "", call.name, target)


def returned_to_previous_screen(trail, current_fingerprint) -> bool:
    """Structural safety net: a tap closed a transient surface and the screen
    returned to the page before it opened (bottom sheet, dialog, popup menu).

    Used only to forbid a blind replay, never to decide success.
    """
    if len(trail) < 2 or not current_fingerprint:
        return False
    previous, last = trail[-2], trail[-1]
    return (
        last["tool"] in _CLICK_TOOLS
        and not last["declared"]
        and last["fingerprint"] != current_fingerprint
        and previous["fingerprint"] == current_fingerprint
    )


def _number(ref: str, prefix: str) -> int | None:
    match = re.fullmatch(prefix + r"([1-9]\d*)", ref) if isinstance(ref, str) else None
    return int(match[1]) if match else None


@dataclass
class SideEffect:
    id: str
    key: tuple
    description: str
    action_ref: str
    before: str | None
    signature: tuple
    recorded_at: float
    source: str
    status: str = "pending"
    evidence: tuple = ()
    retries: int = 0
    verify_steps: int = 0
    verify_started: float | None = None
    probes: tuple = ()  # refs of automatic checks run for this effect

    def context(self) -> dict:
        return {
            "id": self.id,
            "action": self.action_ref,
            "app": self.key[0],
            "target": self.key[2],
            "expected_effect": self.description,
            "status": self.status,
            "evidence": list(self.evidence),
            "checks": [f"{ref}: {summary[:300]}" for ref, summary in self.probes],
        }


class EffectLedger:
    def __init__(self):
        self.entries: list[SideEffect] = []

    def record(self, *, key, description, action_ref, before, signature, source) -> SideEffect:
        retries = sum(1 for e in self.entries if e.key == key and e.status == "not_happened")
        entry = SideEffect(
            id=f"E{len(self.entries) + 1}",
            key=key,
            description=description[:300],
            action_ref=action_ref,
            before=before,
            signature=signature,
            recorded_at=time.time(),
            source=source,
            retries=retries,
        )
        self.entries.append(entry)
        return entry

    def pending(self) -> SideEffect | None:
        return next((e for e in self.entries if e.status == "pending"), None)

    def get(self, effect_id) -> SideEffect | None:
        return next((e for e in self.entries if e.id == effect_id), None)

    def budget_exhausted(self, entry: SideEffect, now: float | None = None) -> bool:
        now = time.monotonic() if now is None else now
        return entry.verify_steps >= VERIFY_STEP_BUDGET or (
            entry.verify_started is not None and now - entry.verify_started > VERIFY_SECONDS_BUDGET
        )

    def resolve(self, entry: SideEffect, status: str, evidence=()) -> None:
        if status not in _STATUSES:
            raise ValueError("unknown effect status")
        entry.status = status
        entry.evidence = tuple(evidence)

    def may_retry(self, entry: SideEffect) -> bool:
        return entry.retries < RETRY_AFTER_NOT_HAPPENED

    def context(self) -> list[dict]:
        return [e.context() for e in self.entries[-8:]]

    def summary(self) -> str:
        if not self.entries:
            return ""
        labels = {
            "confirmed": "confirmed",
            "pending": "unverified",
            "not_happened": "did not happen",
            "unverifiable": "could not be verified",
        }
        lines = [
            f"{e.id} [{labels[e.status]}] {e.description or e.key[2]}"
            + (f" (evidence: {', '.join(e.evidence)})" if e.evidence else "")
            for e in self.entries
        ]
        return "Side effects:\n" + "\n".join(lines)


def evidence_valid(entry: SideEffect, refs, observations, actions, probes=()) -> bool:
    """Cited evidence must exist and be recorded after the action.

    ``probes`` are refs of automatic checks run for this effect after the action.
    """
    if not refs or not isinstance(refs, (list, tuple)):
        return False
    observation_ids = {o.get("id") for o in observations}
    action_refs = {a.get("ref") for a in actions}
    before = _number(entry.before or "", "O") or 0
    action = _number(entry.action_ref, "A") or 0
    for ref in refs:
        if ref in observation_ids and (_number(ref, "O") or 0) > before:
            continue
        if ref in action_refs and (_number(ref, "A") or 0) > action:
            continue
        if ref in probes:
            continue
        return False
    return True


class EffectNotes:
    """Per-app verification hints written by the model; prompts, not rules."""

    MAX_PER_APP = 5

    def __init__(self, path: Path | None):
        self.path = path

    def _load(self) -> dict:
        if self.path is None:
            return {}
        try:
            data = json.loads(self.path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError):
            return {}

    def for_app(self, package: str) -> list[str]:
        notes = self._load().get(package or "", [])
        return [n for n in notes if isinstance(n, str)][-self.MAX_PER_APP :]

    def add(self, package: str, note: str) -> None:
        note = " ".join(str(note).split())[:200]
        if self.path is None or not package or not note:
            return
        data = self._load()
        notes = [n for n in data.get(package, []) if n != note] + [note]
        data[package] = notes[-self.MAX_PER_APP :]
        try:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding="utf-8")
        except OSError:
            pass  # A lost hint only costs a slower verification next time.
