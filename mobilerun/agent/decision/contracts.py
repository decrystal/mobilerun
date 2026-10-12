"""Typed goal contracts and grounded, three-valued completion assessments."""

from __future__ import annotations

import json
import re
from typing import Any

from mobilerun.agent.decision.completion import validate_completion_criteria
from mobilerun.agent.decision.effect_order import has_post_action_result
from mobilerun.agent.decision.persistent_results import persistent_result_supported
from mobilerun.agent.decision.query_scope import (
    QueryScopeError,
    validate_query_scope,
    validate_scope_assessment,
)

KINDS = {"query", "state", "process", "collection", "artifact", "final_state"}


class ContractError(ValueError):
    """Stable diagnostics without copying potentially sensitive model output."""

    def __init__(self, code: str, field: str = "", *, refs=()):
        self.code = code
        self.field = field
        # References are diagnostics, never model text or observed values.
        self.refs = tuple(
            dict.fromkeys(
                ref
                for ref in refs
                if isinstance(ref, str) and re.fullmatch(r"O\d+:(?:n|image)\d+", ref)
            )
        )[:16]
        super().__init__(code)


def json_object(content: str) -> dict:
    if not isinstance(content, str) or len(content) > 32_768:
        raise ContractError("protocol_error", "response")
    content = re.sub(r"^```(?:json)?\s*|\s*```$", "", content.strip())
    # An orphan Markdown closing delimiter is formatting, not an extra object.
    content = content.rstrip("`").rstrip()
    try:
        data = json.loads(content)
    except ValueError as exc:
        raise ContractError("protocol_error", "json") from exc
    if not isinstance(data, dict):
        raise ContractError("protocol_error", "object")
    return data


def parse_task_contract(
    content: str,
    goal: str,
    *,
    require_query_attributes=False,
    require_effect_scopes=False,
    require_query_scopes=False,
) -> dict[str, Any]:
    if type(require_effect_scopes) is not bool:
        raise ContractError("contract_invalid", "effect_scope_switch")
    if type(require_query_scopes) is not bool:
        raise ContractError("contract_invalid", "query_scope_switch")
    data = json_object(content)
    requirements = data.get("requirements")
    if not isinstance(requirements, list) or not 1 <= len(requirements) <= 32:
        raise ContractError("contract_invalid", "requirements")
    for index, item in enumerate(requirements, 1):
        if (
            not isinstance(item, dict)
            or not {"id", "description", "kind", "quote"} <= set(item)
            or set(item)
            - {
                "id",
                "description",
                "kind",
                "quote",
                "attribute",
                "effect_scope",
                "query_scope",
            }
        ):
            raise ContractError("contract_invalid", "requirement_fields")
        if (
            type(item["id"]) is not int
            or item["id"] != index
            or item["kind"] not in KINDS
        ):
            raise ContractError("contract_invalid", "requirement_identity")
        if (
            item["kind"] == "query"
            and require_query_scopes
            and "query_scope" not in item
        ):
            raise ContractError("contract_invalid", "query_scope")
        if "query_scope" in item:
            try:
                if item["kind"] != "query":
                    raise QueryScopeError("query_scope_invalid")
                validate_query_scope(item["query_scope"], goal)
            except QueryScopeError as exc:
                raise ContractError("contract_invalid", "query_scope") from exc
        effect_kind = item["kind"] in {"state", "collection", "artifact"}
        if require_effect_scopes and effect_kind and "effect_scope" not in item:
            raise ContractError("contract_invalid", "effect_scope")
        if "effect_scope" in item and (
            not effect_kind
            or not isinstance(item["effect_scope"], str)
            or item["effect_scope"] not in {"visible", "persistent"}
        ):
            raise ContractError("contract_invalid", "effect_scope")
        if not isinstance(item["description"], str) or not item["description"].strip():
            raise ContractError("contract_invalid", "description")
        if (
            not isinstance(item["quote"], str)
            or not item["quote"].strip()
            or item["quote"] not in goal
        ):
            raise ContractError("contract_invalid", "goal_quote")
        attribute = item.get("attribute")
        if item["kind"] == "query" and require_query_attributes and attribute is None:
            raise ContractError("contract_invalid", "query_attribute")
        if attribute is not None:
            if not isinstance(attribute, dict) or set(attribute) != {"kind", "label"}:
                raise ContractError("contract_invalid", "attribute")
            if attribute["kind"] not in {
                "type",
                "checked",
                "value",
                "count",
                "list",
                "other",
            } or not isinstance(attribute["label"], str):
                raise ContractError("contract_invalid", "attribute_kind")
            if attribute["kind"] == "type" and (
                not attribute["label"]
                or canonical_label(attribute["label"])
                not in canonical_label(item["quote"])
            ):
                raise ContractError("contract_invalid", "attribute_label")
            if (
                item["kind"] == "query"
                and re.search(
                    r"\b(?:marked|classified|categorized)\s+as\s+(?:a|an)\s+",
                    item["quote"],
                    re.IGNORECASE,
                )
                and attribute["kind"] != "type"
            ):
                raise ContractError("query_property_mismatch", "type_not_completion")
    answer_required = data.get("answer_required")
    if type(answer_required) is not bool:
        raise ContractError("contract_invalid", "answer_required")
    if any(item["kind"] == "query" for item in requirements) and not answer_required:
        raise ContractError("contract_invalid", "query_answer_required")
    if answer_required and not any(item["kind"] == "query" for item in requirements):
        raise ContractError("contract_invalid", "nonquery_answer_required")
    # UI bindings are established from observations, never guessed from the goal.
    # Caller-supplied assertions remain separately binding in DecisionRuntime.
    return {
        "schema_version": 2,
        "requirements": requirements,
        "checks": {},
        "answer_required": answer_required,
    }


def parse_grounded_review(
    content: str,
    contract: dict,
    evidence: dict,
    current_id: str,
    *,
    uploaded_images: dict[str, str] | None = None,
    derive_review_dependencies: bool = False,
) -> dict:
    if type(derive_review_dependencies) is not bool:
        raise ContractError("protocol_error", "derive_review_dependencies")
    data = json_object(content)
    if type(data.get("complete")) is not bool:
        raise ContractError("protocol_error", "complete")
    if not isinstance(data.get("evidence"), str) or not data["evidence"].strip():
        raise ContractError("protocol_error", "evidence")
    checks = data.get("checks", {})
    try:
        validate_completion_criteria(checks)
    except ValueError as exc:
        raise ContractError("protocol_error", "checks") from exc
    requirements = contract["requirements"]
    assessments = data.get("assessments")
    if not isinstance(assessments, list) or len(assessments) != len(requirements):
        raise ContractError("review_coverage", "assessments")
    seen, verified = set(), []
    for item in assessments:
        if not isinstance(item, dict) or not {
            "id",
            "status",
            "evidence",
            "evidence_refs",
        } <= set(item):
            raise ContractError("protocol_error", "assessment_fields")
        rid = item["id"]
        if type(rid) is not int or not 1 <= rid <= len(requirements) or rid in seen:
            raise ContractError("review_coverage", "requirement_id")
        seen.add(rid)
        if item["status"] not in {"supported", "contradicted", "unknown"}:
            raise ContractError("protocol_error", "assessment_status")
        if not isinstance(item["evidence"], str) or not item["evidence"].strip():
            raise ContractError("protocol_error", "assessment_evidence")
        refs = item["evidence_refs"]
        if not isinstance(refs, list) or any(
            not isinstance(ref, str) or ref not in evidence for ref in refs
        ):
            raise ContractError("evidence_unknown", "evidence_refs")
        requirement = requirements[rid - 1]
        persistent = requirement.get("effect_scope") == "persistent"
        allowed_fields = {"id", "status", "evidence", "evidence_refs"} | (
            {"result_refs", "reopen_trace"} if persistent else set()
        )
        if requirement.get("query_scope") is not None:
            allowed_fields |= {"scope_refs", "constraint_evidence"}
        if set(item) - allowed_fields:
            raise ContractError("protocol_error", "assessment_fields")
        if requirement.get("query_scope") is not None:
            try:
                validate_scope_assessment(item, requirement, evidence, current_id)
            except QueryScopeError as exc:
                raise ContractError("query_scope_unverified", "query_scope") from exc
        results = item.get("result_refs", [])
        if persistent and (
            not isinstance(results, list)
            or any(not isinstance(ref, str) or ref not in refs for ref in results)
        ):
            raise ContractError("evidence_unknown", "result_refs")
        trace = item.get("reopen_trace")
        if trace is not None:
            if (
                not isinstance(trace, dict)
                or set(trace) != {"before", "away", "returned", "actions"}
                or not isinstance(trace.get("actions"), list)
            ):
                raise ContractError("effect_unverified", "reopen_trace")
            trace_refs = [
                trace["before"],
                trace["away"],
                trace["returned"],
                *trace["actions"],
            ]
            if any(not isinstance(ref, str) or ref not in refs for ref in trace_refs):
                raise ContractError("evidence_unknown", "reopen_trace")
        if item["status"] == "supported":
            if not refs:
                raise ContractError("evidence_missing", "evidence_refs")
            calculated = requirement["kind"] in {"query", "collection"} and any(
                evidence[ref].get("source") == "deterministic_query"
                and evidence[ref].get("coverage", {}).get("complete") is True
                for ref in refs
            )
            if requirement["kind"] != "process" and not (
                calculated or any(ref.startswith("O") for ref in refs)
            ):
                raise ContractError("effect_unverified", "evidence_refs")
            if (
                requirement["kind"] in {"query", "state", "collection", "artifact"}
                and not calculated
                and not any(":n" in ref or ":image" in ref for ref in refs)
            ):
                raise ContractError("effect_unverified", "content_evidence")
            for ref in refs:
                if evidence[ref].get("source") == "screenshot" and (
                    not ref.startswith(current_id + ":image")
                    or any(
                        other.startswith(current_id + ":image")
                        and int(other.rsplit("image", 1)[1])
                        > int(ref.rsplit("image", 1)[1])
                        for other in evidence
                    )
                ):
                    historical_uploaded = (
                        not ref.startswith(current_id + ":image")
                        and (uploaded_images or {}).get(ref)
                        == evidence[ref].get("sha256")
                        and evidence[ref].get("sha256") is not None
                        and requirement["kind"] != "final_state"
                    )
                    if not historical_uploaded:
                        raise ContractError("evidence_stale", "historical_image")
            if requirement["kind"] == "final_state" and not any(
                ref == current_id or ref.startswith(current_id + ":") for ref in refs
            ):
                raise ContractError("evidence_stale", "final_state")
            if (
                data["complete"]
                and derive_review_dependencies
                and requirement["kind"]
                in {"state", "collection", "artifact", "final_state"}
                and not has_post_action_result(refs, evidence, current_id)
            ):
                raise ContractError("effect_unverified", "post_action_observation")
            if persistent:
                if not persistent_result_supported(
                    results,
                    evidence,
                    uploaded_images=uploaded_images,
                    reopen_trace=trace,
                ):
                    raise ContractError("effect_unverified", "persistent_result")
                action_refs = [ref for ref in refs if re.fullmatch(r"A[1-9]\d*", ref)]
                if not has_post_action_result(
                    [*action_refs, *results], evidence, current_id
                ):
                    raise ContractError("effect_unverified", "post_action_observation")
        verified.append(
            {
                **item,
                "requirement": requirement["description"],
                "satisfied": item["status"] == "supported",
            }
        )
    missing = [item["requirement"] for item in verified if not item["satisfied"]]
    if data["complete"] and missing:
        raise ContractError("review_contradiction", "complete")
    answer = data.get("final_answer")
    if answer is not None and (not isinstance(answer, str) or not answer.strip()):
        raise ContractError("protocol_error", "final_answer")
    if data["complete"] and contract["answer_required"] and answer is None:
        raise ContractError("answer_missing", "final_answer")
    if data["complete"]:
        for item in verified:
            attribute = requirements[item["id"] - 1].get("attribute", {})
            if attribute.get("kind") == "type" and str(answer).casefold() in {
                "true",
                "false",
            }:
                hints = {
                    canonical_type_label(evidence[ref]["accessibility_type_hint"])
                    for ref in item["evidence_refs"]
                    if evidence[ref].get("accessibility_type_hint")
                }
                if (
                    hints
                    and canonical_type_label(attribute["label"]) in hints
                    and str(answer).casefold() != "true"
                ):
                    raise ContractError(
                        "query_property_mismatch", "type_not_completion"
                    )
    all_refs = tuple(
        dict.fromkeys(ref for item in verified for ref in item["evidence_refs"])
    )
    verification_refs = data.get("verification_refs")
    if (
        derive_review_dependencies
        and verification_refs is not None
        and (
            not isinstance(verification_refs, list)
            or any(ref not in all_refs for ref in verification_refs)
        )
    ):
        # An unsolicited selector cannot introduce an unknown dependency.
        raise ContractError("evidence_unknown", "verification_refs")
    if data["complete"] and (derive_review_dependencies or verification_refs is None):
        # The complete, validated assessment references determine the live
        # guard. A model-supplied subset cannot omit a supporting property.
        verification_refs = [
            ref
            for ref in all_refs
            if evidence[ref].get("source") != "screenshot"
            or ref.startswith(current_id + ":image")
        ]
    if verification_refs is not None and data["complete"]:
        # A continuously changing readout is supporting temporal evidence, not
        # an exact final value. Derive this from consecutive real observations;
        # range/input/checkbox values and explicit numbers remain mandatory.
        for item in verified:
            requirement = requirements[item["id"] - 1]
            if requirement["kind"] not in {"state", "final_state"} or re.search(
                r"\d", requirement["quote"]
            ):
                continue
            refs = [
                ref
                for ref in item["evidence_refs"]
                if ref.startswith(current_id + ":n")
            ]
            volatile = [
                ref
                for ref in refs
                if observed_temporal_readout(ref, evidence, current_id)
            ]
            if set(refs) - set(volatile):
                verification_refs = [
                    ref for ref in verification_refs if ref not in volatile
                ]
    if verification_refs is not None:
        if not isinstance(verification_refs, list) or any(
            ref not in all_refs for ref in verification_refs
        ):
            raise ContractError("evidence_unknown", "verification_refs")
        # Uploaded historical images prove their recorded time, not the live
        # state. Project these refs out of the live guard; their assessments
        # and evidence remain unchanged. Never accept an unuploaded image.
        verification_refs = [
            ref
            for ref in verification_refs
            if not (
                evidence[ref].get("source") == "screenshot"
                and not ref.startswith(current_id + ":image")
                and (uploaded_images or {}).get(ref) == evidence[ref].get("sha256")
                and evidence[ref].get("sha256") is not None
            )
        ]
        if any(
            evidence[ref].get("source") == "screenshot"
            and not ref.startswith(current_id + ":image")
            for ref in verification_refs
        ):
            raise ContractError("evidence_stale", "historical_image_verification")
        if (
            data["complete"]
            and any(
                ref.startswith(current_id + ":n")
                or ref.startswith(current_id + ":image")
                for ref in all_refs
            )
            and not verification_refs
        ):
            raise ContractError(
                "evidence_missing",
                "verification_refs",
                refs=[ref for ref in all_refs if ref.startswith(current_id + ":")],
            )
        if data["complete"]:
            for item in verified:
                kind = requirements[item["id"] - 1]["kind"]
                if kind not in {"state", "final_state"}:
                    continue
                current_refs = [
                    ref
                    for ref in item["evidence_refs"]
                    if ref.startswith(current_id + ":n")
                    or ref.startswith(current_id + ":image")
                ]
                if current_refs and not set(current_refs).intersection(
                    verification_refs
                ):
                    raise ContractError(
                        "evidence_missing", "state_verification_refs", refs=current_refs
                    )
                omitted = sorted(
                    ref
                    for ref in set(current_refs) - set(verification_refs)
                    if re.search(r"\d", requirements[item["id"] - 1]["quote"])
                    or not observed_temporal_readout(ref, evidence, current_id)
                )
                if omitted:
                    raise ContractError(
                        "evidence_missing", "omitted_state_property", refs=omitted
                    )
    return {
        "status": "done" if data["complete"] else "incomplete",
        "message": data["evidence"],
        "completion_checks": checks,
        "assessments": tuple(sorted(verified, key=lambda item: item["id"])),
        "final_answer": answer if data["complete"] else None,
        "evidence_refs": all_refs,
        "verification_refs": (
            tuple(verification_refs) if verification_refs is not None else None
        ),
        "missing": missing,
    }


def canonical_label(value: str) -> str:
    return re.sub(r"[\s_-]+", "", value).casefold()


def canonical_type_label(value: str) -> str:
    # Generic object nouns do not change the category ("reminder item").
    label = re.sub(r"\s+(?:item|object|entry)$", "", value.strip(), flags=re.IGNORECASE)
    return canonical_label(label)


def observed_temporal_readout(ref: str, evidence: dict, current_id: str) -> bool:
    """An omitted state witness must be an actually changing temporal readout.

    Models cannot label arbitrary exact values volatile. A distinct stable
    witness for the same requirement is also required by the caller above.
    """
    node = evidence[ref]
    text = node.get("rawText", node.get("text"))
    if (
        not isinstance(text, str)
        or not re.fullmatch(r"\d{1,3}(?::\d{2}){0,2}(?:[.,]\d+)?", text.strip())
        or not node.get("resourceId")
        or any(
            node.get(key)
            for key in (
                "rangeInfo",
                "editable",
                "isEditable",
                "checkable",
                "isCheckable",
            )
        )
        or "inputValue" in node
    ):
        return False
    package = evidence.get(current_id, {}).get("phone", {}).get("packageName")
    for observation in evidence.values():
        if (
            observation.get("id") == current_id
            or observation.get("phone", {}).get("packageName") != package
        ):
            continue
        # A changed numeric value across a user action is not autonomous motion.
        oid = observation.get("id", "")
        if not re.fullmatch(r"O\d+", oid) or int(oid[1:]) >= int(current_id[1:]):
            continue
        if any(
            action.get("tool") != "wait"
            and re.fullmatch(r"O\d+", action.get("before") or "")
            and int(action["before"][1:]) >= int(oid[1:])
            for action in evidence.values()
            if action.get("dispatch_status")
        ):
            continue
        for previous in observation.get("nodes", []):
            previous_text = previous.get("rawText", previous.get("text"))
            if (
                previous.get("resourceId") == node["resourceId"]
                and previous.get("className") == node.get("className")
                and isinstance(previous_text, str)
                and previous_text != text
                and re.fullmatch(
                    r"\d{1,3}(?::\d{2}){0,2}(?:[.,]\d+)?", previous_text.strip()
                )
            ):
                return True
    return False
