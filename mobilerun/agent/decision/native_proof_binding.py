"""Keep distinct reviewed controls and their native ancestry distinct at refresh."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping

PROOF_PROPERTIES = (
    "text",
    "rawText",
    "contentDescription",
    "stateDescription",
    "accessibility_type_hint",
    "inputValue",
    "textSelectionStart",
    "textSelectionEnd",
    "checkedState",
    "checked",
    "isChecked",
    "isCheckable",
    "checkable",
    "rangeInfo",
    "selected",
    "isSelected",
    "childCount",
    "actionList",
    "enabled",
    "isEnabled",
)
ANCESTOR_PROPERTIES = (
    "resourceId",
    "className",
    "rawText",
    "text",
    "contentDescription",
    "accessibility_type_hint",
)
MAX_PROOFS = 64
MAX_DEPTH = 128


class NativeProofBindingError(ValueError):
    pass


def _digest(value):
    serialized = json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False)
    return hashlib.sha256(serialized.encode("utf8")).hexdigest()


def _paths(nodes):
    if not isinstance(nodes, list) or any(
        not isinstance(node, Mapping) for node in nodes
    ):
        raise NativeProofBindingError("invalid_native_nodes")
    paths = {}
    for node in nodes:
        path = node.get("path")
        if type(path) is not int or path < 0 or path in paths:
            raise NativeProofBindingError("ambiguous_native_path")
        paths[path] = node
    return paths


def _ancestry(node, paths):
    result, visited = [], {node["path"]}
    parent = node.get("parent_path")
    while parent is not None:
        if (
            type(parent) is not int
            or parent not in paths
            or parent in visited
            or len(result) >= MAX_DEPTH
        ):
            raise NativeProofBindingError("invalid_native_ancestry")
        visited.add(parent)
        ancestor = paths[parent]
        identity = {
            name: ancestor[name] for name in ANCESTOR_PROPERTIES if name in ancestor
        }
        # Positions/counts are layout/coverage data, not persistent object IDs.
        # Retain the observed structural role while comparing only own labels.
        identity["collection"] = isinstance(ancestor.get("collectionInfo"), Mapping)
        identity["collection_item"] = isinstance(
            ancestor.get("collectionItemInfo"), Mapping
        )
        result.append({"path": parent, "identity_sha256": _digest(identity)})
        parent = ancestor.get("parent_path")
    return list(reversed(result))


def bind_native_proofs(proofs, observation, *, goal):
    """Bind exact local source controls; this is not a business-object ID claim."""
    if (
        not isinstance(proofs, list)
        or not 1 <= len(proofs) <= MAX_PROOFS
        or not isinstance(goal, str)
        or not goal
        or not isinstance(observation, Mapping)
        or observation.get("source") != "accessibility"
        or not isinstance(observation.get("id"), str)
        or not isinstance(observation.get("phone"), Mapping)
        or not isinstance(observation.get("phone", {}).get("packageName"), str)
        or not observation["phone"]["packageName"]
    ):
        raise NativeProofBindingError("invalid_native_proof_source")
    paths = _paths(observation.get("nodes"))
    references = {}
    for node in paths.values():
        ref = node.get("ref")
        if not isinstance(ref, str) or ref in references:
            raise NativeProofBindingError("ambiguous_native_ref")
        references[ref] = node
    bound, seen = [], set()
    for proof in proofs:
        if not isinstance(proof, Mapping):
            raise NativeProofBindingError("invalid_native_proof")
        ref = proof.get("ref")
        if (
            not isinstance(ref, str)
            or not ref.startswith(observation["id"] + ":n")
            or ref in seen
        ):
            raise NativeProofBindingError("invalid_native_proof_ref")
        actual = references.get(ref)
        if actual is None or _digest(actual) != _digest(proof):
            raise NativeProofBindingError("native_proof_source_changed")
        seen.add(ref)
        bound.append(
            {
                "ref": ref,
                "path": actual["path"],
                "source_sha256": _digest(actual),
                "ancestors": _ancestry(actual, paths),
            }
        )
    return {
        "schema_version": 1,
        "goal_sha256": _digest(goal),
        "observation": observation["id"],
        "package": observation["phone"]["packageName"],
        "proofs": bound,
    }


def _candidate(proof, nodes):
    # Exactly the existing corroborate_nodes rule, including bounds when the
    # resource/class is repeated. Never accept a looser per-control match.
    matches = [
        node
        for node in nodes
        if node.get("className") == proof.get("className")
        and node.get("resourceId") == proof.get("resourceId")
    ]
    if len(matches) != 1:
        matches = [
            node
            for node in matches
            if node.get("bounds") == proof.get("bounds")
            and all(node.get(name) == proof.get(name) for name in PROOF_PROPERTIES)
        ]
    if len(matches) != 1 or any(
        matches[0].get(name) != proof.get(name) for name in PROOF_PROPERTIES
    ):
        return None
    return matches[0]


def _shared_depth(left, right):
    depth = 0
    for before, after in zip(left, right, strict=False):
        if before["path"] != after["path"]:
            break
        depth += 1
    return depth


def corroborate_native_binding(
    binding, proofs, observation, current_nodes, package, *, goal
):
    """Add injectivity and parent-ownership checks to unchanged value guards."""
    try:
        expected = bind_native_proofs(proofs, observation, goal=goal)
        if _digest(binding) != _digest(expected) or package != expected["package"]:
            return False
        paths = _paths(current_nodes)
        assigned, ancestries = set(), []
        for proof, bound in zip(proofs, expected["proofs"], strict=True):
            candidate = _candidate(proof, current_nodes)
            if candidate is None or candidate["path"] in assigned:
                return False
            assigned.add(candidate["path"])
            ancestry = _ancestry(candidate, paths)
            if [item["identity_sha256"] for item in ancestry] != [
                item["identity_sha256"] for item in bound["ancestors"]
            ]:
                return False
            ancestries.append(ancestry + [{"path": candidate["path"]}])
        for left in range(len(proofs)):
            for right in range(left + 1, len(proofs)):
                original_left, original_right = (
                    expected["proofs"][left],
                    expected["proofs"][right],
                )
                if _shared_depth(
                    original_left["ancestors"] + [{"path": original_left["path"]}],
                    original_right["ancestors"] + [{"path": original_right["path"]}],
                ) != _shared_depth(ancestries[left], ancestries[right]):
                    return False
        return True
    except (
        NativeProofBindingError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        RecursionError,
    ):
        return False
