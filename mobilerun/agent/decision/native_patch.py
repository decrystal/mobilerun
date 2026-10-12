"""Apply one uniquely anchored edit to the exact captured native input value."""

from __future__ import annotations

from collections.abc import Mapping

from mobilerun.agent.decision.contracts import ContractError
from mobilerun.agent.decision.native_edit import (
    _same_receipt,
    _target,
    current_native_editor,
    validate_native_edit_length,
)
from mobilerun.agent.decision.native_text import MAX_TEXT_CHARS, text_digest

PATCH_FIELDS = frozenset({"ref", "field", "before", "match", "after", "replacement"})
ERROR = "text_patch_invalid"


def _compose(original, spec, node):
    parts = (spec["before"], spec["match"], spec["after"], spec["replacement"])
    if any(not isinstance(part, str) or len(part) > MAX_TEXT_CHARS for part in parts):
        raise ContractError(ERROR, "text_patch")
    left, match, right, replacement = parts
    pattern = left + match + right
    if not pattern or match == replacement:
        raise ContractError(ERROR, "empty_anchor_or_no_change")
    first = original.find(pattern)
    if first < 0:
        raise ContractError(ERROR, "anchor_not_found")
    # Include overlapping matches: "ana" occurs twice in "banana".
    if original.find(pattern, first + 1) >= 0:
        raise ContractError(ERROR, "anchor_ambiguous")
    start, end = first + len(left), first + len(left) + len(match)
    result = original[:start] + replacement + original[end:]
    validate_native_edit_length(result, node, error_code=ERROR)
    return result, start, end


def _receipt(base, node, spec, result, start, end):
    return {
        **base,
        "kind": "native_text_patch",
        **{name: spec[name] for name in ("before", "match", "after", "replacement")},
        "start": start,
        "end": end,
        "target": _target(node, base["package"]),
        "result_sha256": text_digest(result),
        "result_chars": len(result),
    }


def resolve_native_text_patch(spec, sources, goal, ui, current_id):
    """Replace/delete a unique fragment, or insert at its unique context boundary."""
    if (
        not isinstance(spec, dict)
        or set(spec) != PATCH_FIELDS
        or any(not isinstance(value, str) for value in spec.values())
        or spec["field"] != "inputValue"
    ):
        raise ContractError(ERROR, "text_patch")
    original, base, node = current_native_editor(
        spec, sources, goal, ui, current_id, error_code=ERROR
    )
    result, start, end = _compose(original, spec, node)
    return result, _receipt(base, node, spec, result, start, end)


def validate_patch_result_receipt(source, base, original, receipt, result):
    if not isinstance(receipt, Mapping) or base["field"] != "inputValue":
        return False
    node = source["node"]
    if node.get("focused", node.get("isFocused")) is not True:
        return False
    try:
        expected, start, end = _compose(original, receipt, node)
        return result == expected and _same_receipt(
            dict(receipt), _receipt(base, node, receipt, expected, start, end)
        )
    except (ContractError, KeyError, TypeError, ValueError):
        return False
