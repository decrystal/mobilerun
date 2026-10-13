"""Validate the observable form of result evidence for persistence obligations.

The reviewer owns semantic attribution. These checks reject editor-only claims
and fabricated reopen traces; they do not prove disk durability or business IDs.
"""

from __future__ import annotations

import re

_INPUT_CLASSES = {
    "EditText",
    "AutoCompleteTextView",
    "MultiAutoCompleteTextView",
    "TextField",
}
_NAVIGATION_TOOLS = {"click", "click_at", "system_button", "launch_app"}


def _number(ref, prefix):
    match = re.fullmatch(prefix + r"([1-9]\d*)", ref) if isinstance(ref, str) else None
    return int(match[1]) if match else None


def _input(node):
    return (
        node.get("editable") is True
        or node.get("isEditable") is True
        or "inputValue" in node
        or str(node.get("className", "")).split(".")[-1] in _INPUT_CLASSES
    )


def _field_ancestor(ref, evidence):
    match = re.fullmatch(r"(O[1-9]\d*):n\d+", ref)
    if not match:
        return False, None
    observation = evidence.get(match[1], {})
    nodes = observation.get("nodes", [])
    matching = [node for node in nodes if node.get("ref") == ref]
    if len(matching) != 1 or matching[0] != evidence.get(ref):
        return False, None
    node, seen = matching[0], set()
    while node is not None:
        path = node.get("path")
        if type(path) is not int or path < 0 or path in seen or len(seen) >= 128:
            return False, None
        seen.add(path)
        if _input(node):
            return True, node
        parent = node.get("parent_path")
        if parent is None:
            return True, None
        matching = [candidate for candidate in nodes if candidate.get("path") == parent]
        if type(parent) is not int or len(matching) != 1:
            return False, None
        node = matching[0]
    return False, None


def _same_field(node, field):
    return (
        node.get("resourceId") == field.get("resourceId")
        and str(node.get("className", "")).split(".")[-1]
        == str(field.get("className", "")).split(".")[-1]
    )


def _value(node):
    if node.get("isShowingHintText") is True:
        return None
    for key in ("inputValue", "rawText", "text"):
        if key in node:
            return node[key] if isinstance(node[key], str) else None
    return None


def _reopened(ref, field, trace, evidence):
    if not isinstance(trace, dict) or set(trace) != {
        "before",
        "away",
        "returned",
        "actions",
    }:
        return False
    before, away, returned = (
        _number(trace.get(key), "O") for key in ("before", "away", "returned")
    )
    if (
        any(number is None for number in (before, away, returned))
        or not before < away < returned
    ):
        return False
    if not ref.startswith(trace["returned"] + ":n") or not field.get("resourceId"):
        return False
    observations = [
        evidence.get(trace[key], {}) for key in ("before", "away", "returned")
    ]
    if any(
        observation.get("id") != trace[key]
        or not isinstance(observation.get("nodes"), list)
        for key, observation in zip(
            ("before", "away", "returned"), observations, strict=True
        )
    ):
        return False
    old, absent, current = observations
    if any(observation.get("nodes_budget_omitted", 0) for observation in observations):
        return False
    package = old.get("phone", {}).get("packageName")
    if not package or current.get("phone", {}).get("packageName") != package:
        return False
    previous = [node for node in old["nodes"] if _same_field(node, field)]
    latest = [node for node in current["nodes"] if _same_field(node, field)]
    if (
        len(previous) != 1
        or len(latest) != 1
        or _value(previous[0]) is None
        or _value(previous[0]) != _value(field)
    ):
        return False
    if absent.get("phone", {}).get("packageName") == package and any(
        _same_field(node, field) for node in absent["nodes"]
    ):
        return False
    action_refs = trace["actions"]
    if not isinstance(action_refs, list) or len(action_refs) != 2:
        return False
    first, last = (_number(action, "A") for action in action_refs)
    if first is None or last is None or first >= last:
        return False
    if last - first + 1 > sum(_number(ref, "A") is not None for ref in evidence):
        return False
    actions = [evidence.get("A" + str(number), {}) for number in range(first, last + 1)]
    if any(
        action.get("ref") != "A" + str(number)
        or action.get("dispatch_status") != "dispatched"
        for number, action in zip(range(first, last + 1), actions, strict=True)
    ):
        return False
    if (
        actions[0].get("tool") not in _NAVIGATION_TOOLS
        or actions[-1].get("tool") not in _NAVIGATION_TOOLS
    ):
        return False
    if any(action.get("tool") == "type" for action in actions):
        return False
    for action in actions:
        start, end = _number(action.get("before"), "O"), _number(
            action.get("after"), "O"
        )
        if start is None or end is None or not before <= start <= end <= returned:
            return False
    return (
        before <= _number(actions[0]["before"], "O") < away
        and _number(actions[0]["after"], "O") <= away
        and away <= _number(actions[-1]["before"], "O") < returned
        and _number(actions[-1]["after"], "O") <= returned
    )


def persistent_result_supported(
    result_refs, evidence, *, uploaded_images=None, reopen_trace=None
):
    """Require actual result sources, not just an editable value or dispatch."""
    if not isinstance(result_refs, list) or not result_refs:
        return False
    images = uploaded_images or {}

    def witness(ref, seen=()):
        if (
            not isinstance(ref, str)
            or ref in seen
            or len(seen) >= 128
            or ref not in evidence
        ):
            return False
        item = evidence[ref]
        if item.get("source") == "screenshot":
            return bool(item.get("sha256") and images.get(ref) == item["sha256"])
        if item.get("source") == "device_probe":
            # A file the device itself reported as added after the action.
            return item.get("status") == "confirmed"
        if item.get("source") == "deterministic_query":
            sources = item.get("evidence_refs")
            return (
                item.get("coverage", {}).get("complete") is True
                and isinstance(sources, list)
                and bool(sources)
                and all(witness(source, (*seen, ref)) for source in sources)
            )
        valid, field = _field_ancestor(ref, evidence)
        if not valid:
            return False
        return field is None or _reopened(ref, field, reopen_trace, evidence)

    try:
        return all(witness(ref) for ref in result_refs)
    except (ValueError, TypeError, KeyError, AttributeError, RecursionError):
        return False
