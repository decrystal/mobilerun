"""Reference captured native values without asking a model to rewrite them."""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections.abc import Mapping

from mobilerun.agent.decision.contracts import ContractError

TEXT_FIELDS = ("rawText", "inputValue")
MAX_TEXT_CHARS = 16_384


def text_digest(value: str) -> str:
    return hashlib.sha256(value.encode("utf8")).hexdigest()


def _fields(node: Mapping) -> list[str]:
    if (
        node.get("password")
        or node.get("isPassword")
        or node.get("isShowingHintText")
        or node.get("visible", node.get("isVisibleToUser", True)) is False
    ):
        return []
    input_type = node.get("inputType")
    if isinstance(input_type, str) and "password" in input_type.casefold():
        return []
    if type(input_type) is int and (
        input_type & 0xF == 1
        and input_type & 0xFF0 in {0x80, 0x90, 0xE0}
        or input_type & 0xF == 2
        and input_type & 0xFF0 == 0x10
    ):
        return []
    return [
        name
        for name in TEXT_FIELDS
        if isinstance(node.get(name), str) and 0 < len(node[name]) <= MAX_TEXT_CHARS
    ]


def _source(observation: Mapping, node: Mapping, goal: str) -> dict | None:
    fields = _fields(node)
    ref = node.get("ref")
    if (
        observation.get("source") != "accessibility"
        or not isinstance(ref, str)
        or not re.fullmatch(r"O\d+:n\d+", ref)
        or ref.split(":")[0] != observation.get("id")
        or not fields
    ):
        return None
    labels = []
    for sibling in observation.get("nodes", []):
        value = sibling.get("rawText")
        if (
            sibling.get("ref") != ref
            and _fields(sibling)
            and isinstance(value, str)
            and 0 < len(value) <= 80
        ):
            labels.append({"ref": sibling["ref"], "rawText": value})
            if len(labels) == 3:
                break
    metadata = {
        "ref": ref,
        "observation": observation["id"],
        "package": observation.get("phone", {}).get("packageName", ""),
        "page": observation.get("phone", {}).get("currentApp", ""),
        "captured_at": observation.get("captured_at"),
        "resourceId": node.get("resourceId", ""),
        "className": node.get("className", ""),
        "parent_path": node.get("parent_path"),
        "fields": {
            name: {"chars": len(node[name]), "sha256": text_digest(node[name])}
            for name in fields
        },
        "preview": node[fields[0]][:60],
        "preview_truncated": len(node[fields[0]]) > 60,
        "page_labels": labels,
    }
    return {
        "node": copy.deepcopy(dict(node)),
        "metadata": metadata,
        "goal_sha256": text_digest(goal),
    }


def native_text_snapshot(
    observations, goal: str, *, max_sources=12, max_chars=2500
) -> dict:
    """Bound the advertised index; full values stay in a private goal-local map."""
    candidates = []
    seen = set()
    for observation in reversed(observations):
        for node in observation.get("nodes", []):
            source = _source(observation, node, goal)
            if source is None:
                continue
            ref = source["metadata"]["ref"]
            if ref in seen:
                raise ContractError("text_source_invalid", "duplicate_ref")
            seen.add(ref)
            candidates.append(source)
    # Long values benefit most from references. Never merge different objects
    # merely because their captured strings or names are equal.
    candidates.sort(
        key=lambda item: max(
            field["chars"] for field in item["metadata"]["fields"].values()
        ),
        reverse=True,
    )
    index, private, budget = [], {}, max_chars
    for source in candidates:
        size = len(json.dumps(source["metadata"], ensure_ascii=False))
        if len(index) >= max_sources or size > budget:
            continue
        index.append(copy.deepcopy(source["metadata"]))
        private[source["metadata"]["ref"]] = source
        budget -= size
    return {
        "index": {"sources": index, "sources_omitted": len(candidates) - len(index)},
        "private": private,
    }


def advertised_native_sources(index: Mapping, private: Mapping, goal: str) -> dict:
    if (
        not isinstance(index, Mapping)
        or not isinstance(private, Mapping)
        or set(index) != {"sources", "sources_omitted"}
        or not isinstance(index["sources"], list)
        or type(index["sources_omitted"]) is not int
        or index["sources_omitted"] < 0
    ):
        raise ContractError("text_source_invalid", "index")
    selected = {}
    for metadata in index["sources"]:
        if not isinstance(metadata, Mapping) or not isinstance(
            metadata.get("ref"), str
        ):
            raise ContractError("text_source_invalid", "index")
        ref = metadata.get("ref")
        source = private.get(ref)
        if (
            not isinstance(source, Mapping)
            or source.get("metadata") != metadata
            or source.get("goal_sha256") != text_digest(goal)
            or ref in selected
        ):
            raise ContractError("text_source_invalid", "index")
        selected[ref] = source
    return selected


def resolve_native_text(spec: Mapping, sources: Mapping, goal: str) -> tuple[str, dict]:
    """Resolve one exact field; no alias, interpolation, trimming or file I/O."""
    if not isinstance(spec, dict) or set(spec) != {"ref", "field"}:
        raise ContractError("text_source_invalid", "text_source")
    ref, field = spec["ref"], spec["field"]
    if (
        not isinstance(ref, str)
        or not re.fullmatch(r"O\d+:n\d+", ref)
        or field not in TEXT_FIELDS
    ):
        raise ContractError("text_source_invalid", "text_source")
    if not isinstance(sources, Mapping):
        raise ContractError("text_source_invalid", "text_source")
    source = sources.get(ref)
    if not isinstance(source, Mapping) or source.get("goal_sha256") != text_digest(
        goal
    ):
        raise ContractError("text_source_unknown", "text_source")
    node, metadata = source.get("node"), source.get("metadata")
    if not isinstance(node, Mapping) or not isinstance(metadata, Mapping):
        raise ContractError("text_source_invalid", "text_source")
    if (
        node.get("ref") != ref
        or metadata.get("ref") != ref
        or field not in _fields(node)
    ):
        raise ContractError("text_source_invalid", "text_source")
    value = node[field]
    fields = metadata.get("fields")
    if not isinstance(fields, Mapping) or fields.get(field) != {
        "chars": len(value),
        "sha256": text_digest(value),
    }:
        raise ContractError("text_source_changed", "text_source")
    if metadata.get("observation") != ref.split(":")[0] or not isinstance(
        metadata.get("package"), str
    ):
        raise ContractError("text_source_invalid", "text_source")
    receipt = {
        "ref": ref,
        "field": field,
        "sha256": text_digest(value),
        "value_chars": len(value),
        "goal_sha256": source["goal_sha256"],
        "observation": metadata["observation"],
        "package": metadata["package"],
    }
    return value, receipt


def validate_native_text_receipt(
    observations, goal: str, receipt: Mapping, value: str
) -> bool:
    try:
        spec = {"ref": receipt["ref"], "field": receipt["field"]}
        found = []
        for observation in observations:
            for node in observation.get("nodes", []):
                if node.get("ref") == spec["ref"]:
                    source = _source(observation, node, goal)
                    if source is not None:
                        found.append(source)
        if len(found) != 1:
            return False
        resolved, actual = resolve_native_text(spec, {spec["ref"]: found[0]}, goal)
        if receipt.get("kind") == "native_text_edit":
            from mobilerun.agent.decision.native_edit import (
                validate_edit_result_receipt,
            )

            return validate_edit_result_receipt(
                found[0], actual, resolved, receipt, value
            )
        if receipt.get("kind") == "native_text_patch":
            from mobilerun.agent.decision.native_patch import (
                validate_patch_result_receipt,
            )

            return validate_patch_result_receipt(
                found[0], actual, resolved, receipt, value
            )
        return actual == receipt and resolved == value
    except (ContractError, KeyError, TypeError, ValueError):
        return False
