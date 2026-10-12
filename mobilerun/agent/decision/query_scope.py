"""Bind user-declared query ranges to cited native computation sources.

The model interprets range meaning; these checks never infer it from an App ID
or resource name. An observed filter is evidence, not a semantic match oracle.
"""

from __future__ import annotations

import re


class QueryScopeError(ValueError):
    pass


def validate_query_scope(spec, goal):
    if (
        not isinstance(spec, dict)
        or set(spec) != {"reference", "quote", "constraints"}
        or not isinstance(spec.get("reference"), str)
        or spec["reference"] not in {"current", "observed"}
        or not isinstance(spec.get("quote"), str)
        or not spec["quote"].strip()
        or spec["quote"] not in goal
        or not isinstance(spec.get("constraints"), list)
        or len(spec["constraints"]) > 16
        or any(
            not isinstance(quote, str)
            or not quote.strip()
            or len(quote) > 1024
            or quote not in goal
            for quote in spec["constraints"]
        )
        or len(set(spec["constraints"])) != len(spec["constraints"])
    ):
        raise QueryScopeError("query_scope_invalid")


def validate_scope_assessment(item, requirement, evidence, current_id):
    """Require range and constraint evidence from the calculation it qualifies."""
    spec = requirement["query_scope"]
    refs = item["evidence_refs"]
    scope_refs = item.get("scope_refs", [])
    conditions = item.get("constraint_evidence", [])
    if (
        not isinstance(scope_refs, list)
        or len(scope_refs) > 64
        or any(
            not isinstance(ref, str) or ref not in refs or ref not in evidence
            for ref in scope_refs
        )
        or len(set(scope_refs)) != len(scope_refs)
        or not isinstance(conditions, list)
        or len(conditions) > 16
    ):
        raise QueryScopeError("query_scope_unverified")
    for condition in conditions:
        if (
            not isinstance(condition, dict)
            or set(condition) != {"quote", "evidence_refs"}
            or not isinstance(condition.get("quote"), str)
            or condition["quote"] not in spec["constraints"]
            or not isinstance(condition.get("evidence_refs"), list)
            or not condition["evidence_refs"]
            or len(condition["evidence_refs"]) > 64
            or any(
                not isinstance(ref, str) or ref not in refs or ref not in evidence
                for ref in condition["evidence_refs"]
            )
        ):
            raise QueryScopeError("query_scope_unverified")
    if len({condition["quote"] for condition in conditions}) != len(conditions):
        raise QueryScopeError("query_scope_unverified")
    if item["status"] != "supported":
        return
    if not scope_refs or {condition["quote"] for condition in conditions} != set(
        spec["constraints"]
    ):
        raise QueryScopeError("query_scope_unverified")
    queries = [
        evidence[ref]
        for ref in refs
        if evidence[ref].get("source") == "deterministic_query"
    ]
    if queries:
        # A comparison may cite multiple ranges. Every calculation needs its
        # own source, and each condition must stay within those real sources.
        allowed_conditions = set()
        for result in queries:
            source = result.get("scope_source")
            if (
                not isinstance(source, dict)
                or set(source) != {"observation", "container_ref", "context_refs"}
                or not isinstance(source.get("observation"), str)
                or re.fullmatch(r"O\d+", source["observation"]) is None
                or not isinstance(source.get("container_ref"), str)
                or re.fullmatch(
                    re.escape(source["observation"]) + r":n\d+", source["container_ref"]
                )
                is None
                or not isinstance(source.get("context_refs"), list)
                or any(
                    not isinstance(ref, str)
                    or re.fullmatch(re.escape(source["observation"]) + r":n\d+", ref)
                    is None
                    for ref in source["context_refs"]
                )
            ):
                raise QueryScopeError("query_scope_unverified")
            allowed = {source["container_ref"], *source["context_refs"]}
            actual_scope = set(scope_refs) & allowed
            if source["container_ref"] not in actual_scope or any(
                ref not in evidence for ref in actual_scope
            ):
                raise QueryScopeError("query_scope_unverified")
            if spec["reference"] == "current" and source["observation"] != current_id:
                raise QueryScopeError("query_scope_unverified")
            # Row-property constraints may be supported by actual filter fact
            # refs. Arbitrary row labels cannot masquerade as global filters.
            filter_refs = set(result.get("filter_evidence_refs", [])) & set(
                result.get("evidence_refs", [])
            )
            allowed_conditions.update(actual_scope | filter_refs)
        all_scope = {
            ref
            for result in queries
            for ref in [
                result["scope_source"]["container_ref"],
                *result["scope_source"]["context_refs"],
            ]
        }
        if not set(scope_refs) <= all_scope:
            raise QueryScopeError("query_scope_unverified")
        if any(
            not set(condition["evidence_refs"]) <= allowed_conditions
            for condition in conditions
        ):
            raise QueryScopeError("query_scope_unverified")
    else:
        # Direct native/visual answers remain available. They must actually
        # cite the requested observed time, without borrowing old value refs.
        if any(re.fullmatch(r"O\d+:(?:n|image)\d+", ref) is None for ref in scope_refs):
            raise QueryScopeError("query_scope_unverified")
        if spec["reference"] == "current" and any(
            ref.startswith("O") and not ref.startswith(current_id + ":") for ref in refs
        ):
            raise QueryScopeError("query_scope_unverified")
        for condition in conditions:
            if not any(
                re.fullmatch(r"O\d+:(?:n|image)\d+", ref)
                for ref in condition["evidence_refs"]
            ):
                raise QueryScopeError("query_scope_unverified")
