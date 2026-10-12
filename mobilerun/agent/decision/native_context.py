"""Lossless, node-local sharing of duplicate native text values in model context."""

from __future__ import annotations

import copy
import json
from collections.abc import Mapping

from mobilerun.agent.decision.contracts import ContractError

VALUE_FIELDS = (
    "rawText",
    "text",
    "inputValue",
    "contentDescription",
    "stateDescription",
)
ALIASES_FIELD = "native_value_aliases"


def compact_native_values(node: Mapping) -> dict:
    """Share only exact, nonempty text values; retain all field roles and refs."""
    original = copy.deepcopy(dict(node))
    if ALIASES_FIELD in original:
        raise ContractError("context_alias_invalid", ALIASES_FIELD)
    compact = copy.deepcopy(original)
    by_value, aliases = {}, {}
    for field in VALUE_FIELDS:
        value = original.get(field)
        if not isinstance(value, str) or len(value) < 32:
            continue
        if value in by_value:
            aliases[field] = by_value[value]
            del compact[field]
        else:
            by_value[value] = field
    if aliases:
        compact[ALIASES_FIELD] = aliases
        if len(json.dumps(compact, ensure_ascii=False, default=str)) < len(
            json.dumps(original, ensure_ascii=False, default=str)
        ):
            return compact
    return original


def expand_native_values(node: Mapping) -> dict:
    """Restore values before grounding; never resolve across nodes or chains."""
    expanded = copy.deepcopy(dict(node))
    if ALIASES_FIELD not in expanded:
        return expanded
    aliases = expanded.pop(ALIASES_FIELD)
    if not isinstance(aliases, dict) or not aliases:
        raise ContractError("context_alias_invalid", ALIASES_FIELD)
    for field, source in aliases.items():
        if (
            field not in VALUE_FIELDS
            or source not in VALUE_FIELDS
            or field in expanded
            or source in aliases
            or not isinstance(expanded.get(source), str)
            or not expanded[source]
        ):
            raise ContractError("context_alias_invalid", ALIASES_FIELD)
    for field, source in aliases.items():
        expanded[field] = expanded[source]
    return expanded


def native_context_expansion_chars(ledger: Mapping) -> int:
    """Exact JSON size difference after restoring shared node values.

    Routing uses the logical context size; a transport optimization must not
    change the provider or the action protocol. Keep serialization identical
    to the adapter's task-state JSON and leave the transmitted ledger intact.
    """
    nodes = [
        node
        for observed in ledger.get("observations", [])
        for node in observed.get("nodes", [])
    ]
    nodes.extend(
        node
        for collection in ledger.get("collections", [])
        for row in collection.get("rows", [])
        for node in row.get("nodes", [])
    )
    return sum(
        len(json.dumps(expand_native_values(node), ensure_ascii=False, default=str))
        - len(json.dumps(node, ensure_ascii=False, default=str))
        for node in nodes
        if ALIASES_FIELD in node
    )
