"""Recheck a reviewed query's current native scope without changing its ledger."""

from __future__ import annotations

import copy
import hashlib
import json
import re

from mobilerun.agent.decision.query import CollectionLedger, QueryError

MAX_QUERIES = 64


class NativeQueryBindingError(ValueError):
    pass


def _digest(value):
    return hashlib.sha256(
        json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False).encode(
            "utf8"
        )
    ).hexdigest()


def assessed_query_refs(state):
    """Infer accepted query dependencies again if pending guard metadata is lost."""
    return tuple(
        dict.fromkeys(
            ref
            for item in state.assessments.values()
            if item.get("status") == "supported"
            for ref in item.get("evidence_refs", ())
            # The accepted assessment already cited a real Q result. Its later
            # disappearance/staleness must not make a required guard optional.
            if isinstance(ref, str) and re.fullmatch(r"Q\d+", ref)
        )
    )


def _scope_digest(scope):
    return _digest(
        {
            "generation": scope.generation,
            "expected_count": scope.expected_count,
            "rows": [
                (index, row["signature"], row["heading"])
                for index, row in sorted(scope.rows.items())
            ],
        }
    )


def capture_query_source(result, scope, *, goal, evidence):
    """Freeze the actual computation source privately before any later review."""
    source = result.get("scope_source", {})
    refs = list(result["evidence_refs"])
    if source:
        refs.extend([source["container_ref"], *source["context_refs"]])
    return {
        "goal_sha256": _digest(goal),
        "source_sha256": _digest(result),
        "scope_sha256": _scope_digest(scope),
        "evidence_sha256": _digest({ref: evidence.get(ref) for ref in refs}),
    }


def bind_native_queries(query_refs, state, current_id, *, goal):
    """Bind actual results, complete coverage and the reviewed displayed scopes.

    Historical queries remain historical facts. Only a scope observed on this
    reviewed page is required to stay present during the final fresh check.
    """
    if (
        not isinstance(query_refs, (list, tuple))
        or not 1 <= len(query_refs) <= MAX_QUERIES
        or not isinstance(goal, str)
        or not goal
        or state.goal != goal
        or not isinstance(current_id, str)
        or re.fullmatch(r"O\d+", current_id) is None
    ):
        raise NativeQueryBindingError("invalid_native_query_source")
    known = state.evidence()
    if known.get(current_id, {}).get("source") != "accessibility":
        raise NativeQueryBindingError("missing_reviewed_observation")
    queries, seen = [], set()
    for ref in query_refs:
        if (
            not isinstance(ref, str)
            or re.fullmatch(r"Q\d+", ref) is None
            or ref in seen
        ):
            raise NativeQueryBindingError("invalid_native_query_ref")
        seen.add(ref)
        result = known.get(ref, {})
        scope = state.collections.scopes.get(result.get("scope"))
        coverage = result.get("coverage", {})
        refs = result.get("evidence_refs")
        if (
            result.get("source") != "deterministic_query"
            or scope is None
            or not scope.complete
            or type(result.get("generation")) is not int
            or result["generation"] != scope.generation
            or coverage.get("complete") is not True
            or type(coverage.get("observed")) is not int
            or type(coverage.get("expected")) is not int
            or coverage["observed"] != scope.expected_count
            or coverage["expected"] != scope.expected_count
            or not isinstance(refs, list)
            or not refs
            or any(
                not isinstance(item, str)
                or re.fullmatch(r"O\d+:n\d+", item) is None
                or item not in known
                for item in refs
            )
        ):
            raise NativeQueryBindingError("invalid_native_query_coverage")
        source = state._query_native_sources.get(ref)
        if source is None or _digest(source) != _digest(
            capture_query_source(result, scope, goal=goal, evidence=known)
        ):
            raise NativeQueryBindingError("native_query_source_changed")
        queries.append(
            {
                "ref": ref,
                "source_sha256": _digest(result),
                "scope": scope.key,
                "scope_sha256": _scope_digest(scope),
                "current": scope.container_ref.startswith(current_id + ":n"),
            }
        )
    return {
        "schema_version": 1,
        "goal_sha256": _digest(goal),
        "observation": current_id,
        "queries": queries,
    }


def corroborate_native_queries(
    binding, state, current_id, current_observation, *, goal
):
    """Use the unchanged native collection rules against a private ledger copy."""
    try:
        query_refs = [item["ref"] for item in binding["queries"]]
        expected = bind_native_queries(query_refs, state, current_id, goal=goal)
        if _digest(binding) != _digest(expected):
            return False
        required = {item["scope"] for item in expected["queries"] if item["current"]}
        if not required:
            return True
        if current_observation.get("source") != "accessibility":
            return False
        private = CollectionLedger()
        for key in required:
            source = state.collections.scopes[key]
            scope = copy.copy(source)
            # observe replaces observed rows/container metadata, never edits
            # existing facts. A private row map preserves the actual ledger.
            scope.rows = dict(source.rows)
            private.scopes[key] = scope
        observed = set(private.observe(current_observation))
        for key in required:
            if key not in observed or _scope_digest(
                private.scopes[key]
            ) != _scope_digest(state.collections.scopes[key]):
                return False
        return True
    except (
        NativeQueryBindingError,
        QueryError,
        KeyError,
        TypeError,
        ValueError,
        AttributeError,
        RecursionError,
    ):
        return False
