"""Preserve the captured focused input while adding an explicit prefix/suffix."""

from __future__ import annotations

from collections.abc import Mapping

from mobilerun.agent.decision.contracts import ContractError
from mobilerun.agent.decision.jev import focused_input_observation
from mobilerun.agent.decision.native_text import (
    MAX_TEXT_CHARS,
    resolve_native_text,
    text_digest,
)


def _compose(before, position, text, node):
    if (
        not isinstance(position, str)
        or position not in {"append", "prepend"}
        or not isinstance(text, str)
        or not text
        or len(text) > MAX_TEXT_CHARS
    ):
        raise ContractError("text_edit_invalid", "text_edit")
    result = before + text if position == "append" else text + before
    validate_native_edit_length(result, node)
    return result


def validate_native_edit_length(result, node, *, error_code="text_edit_invalid"):
    """Check the complete value without clipping either preserved or added text."""
    if len(result) > MAX_TEXT_CHARS:
        raise ContractError(error_code, "result_length")
    limit = node.get("maxTextLength")
    if (
        type(limit) is int
        and limit >= 0
        and len(result.encode("utf-16-le")) // 2 > limit
    ):
        raise ContractError(error_code, "maxTextLength")


def _target(node, package):
    if (
        type(node.get("index")) is not int
        or not isinstance(node.get("resourceId"), str)
        or not node["resourceId"]
    ):
        raise ContractError("text_edit_invalid", "target")
    return {
        "index": node["index"],
        "resourceId": node["resourceId"],
        "className": node.get("className"),
        "bounds": node.get("bounds"),
        "package": package,
    }


def _receipt(base, node, position, text, result):
    return {
        **base,
        "kind": "native_text_edit",
        "position": position,
        "text": text,
        "target": _target(node, base["package"]),
        "result_sha256": text_digest(result),
        "result_chars": len(result),
    }


def _same_receipt(actual, expected):
    if type(actual) is not type(expected):
        return False
    if isinstance(expected, dict):
        return set(actual) == set(expected) and all(
            _same_receipt(actual[key], value) for key, value in expected.items()
        )
    if isinstance(expected, (list, tuple)):
        return len(actual) == len(expected) and all(
            _same_receipt(left, right)
            for left, right in zip(actual, expected, strict=True)
        )
    return actual == expected


def resolve_native_text_edit(spec, sources, goal, ui, current_id):
    if (
        not isinstance(spec, dict)
        or set(spec) != {"ref", "field", "position", "text"}
        or spec["field"] != "inputValue"
    ):
        raise ContractError("text_edit_invalid", "text_edit")
    before, base, node = current_native_editor(spec, sources, goal, ui, current_id)
    result = _compose(before, spec["position"], spec["text"], node)
    return result, _receipt(base, node, spec["position"], spec["text"], result)


def current_native_editor(
    spec, sources, goal, ui, current_id, *, error_code="text_edit_invalid"
):
    """Bind a captured value to the same current, focused native input."""
    before, base = resolve_native_text(
        {"ref": spec["ref"], "field": spec["field"]}, sources, goal
    )
    node = sources[spec["ref"]]["node"]
    focus = focused_input_observation(ui)
    target = _target(node, base["package"])
    if (
        not isinstance(current_id, str)
        or base["observation"] != current_id
        or focus is None
        or focus.get("value") != before
        or target["package"] != ui.phone_state.get("packageName")
        or any(
            focus.get(name) != target[name]
            for name in ("index", "resourceId", "className", "bounds")
        )
        or node.get("focused", node.get("isFocused")) is not True
    ):
        raise ContractError(error_code, "current_focused_input")
    return before, base, node


def validate_edit_result_receipt(source, base, before, receipt, result):
    if not isinstance(receipt, Mapping) or base["field"] != "inputValue":
        return False
    node = source["node"]
    if node.get("focused", node.get("isFocused")) is not True:
        return False
    try:
        expected = _compose(before, receipt["position"], receipt["text"], node)
        return result == expected and _same_receipt(
            dict(receipt),
            _receipt(base, node, receipt["position"], receipt["text"], expected),
        )
    except (ContractError, KeyError, TypeError, ValueError):
        return False
