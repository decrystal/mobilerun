"""Goal-local observations and action evidence shared by all decision paths.

Observations are immutable copies, not model-written memories. References remain
valid across navigation; only final-position requirements need the current page.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
import time
from typing import Any

from mobilerun.agent.decision.input_progress import observed_keypad_input
from mobilerun.agent.decision.input_verification import input_target_changed
from mobilerun.agent.decision.jev import control_observations
from mobilerun.agent.decision.native_context import compact_native_values
from mobilerun.agent.decision.native_proof_binding import PROOF_PROPERTIES
from mobilerun.agent.decision.native_query_binding import capture_query_source
from mobilerun.agent.decision.native_text import native_text_snapshot
from mobilerun.agent.decision.progress import (
    SemanticProgress,
    input_target_progress_key,
)
from mobilerun.agent.decision.query import CollectionLedger, QueryError

NODE_FIELDS = (
    "index",
    "parentIndex",
    "text",
    "contentDescription",
    "label",
    "inputValue",
    "hint",
    "inputType",
    "maxTextLength",
    "textSelectionStart",
    "textSelectionEnd",
    "isMultiLine",
    "resourceId",
    "className",
    "bounds",
    "clickable",
    "longClickable",
    "scrollable",
    "editable",
    "isEditable",
    "focused",
    "isFocused",
    "checked",
    "isChecked",
    "checkedState",
    "checkable",
    "isCheckable",
    "selected",
    "isSelected",
    "rangeInfo",
    "isEnabled",
    "enabled",
    "isShowingHintText",
    "collectionInfo",
    "collectionItemInfo",
    "actionList",
    "childCount",
    "rawText",
    "stateDescription",
    "isShowingHintText",
)
PHONE_FIELDS = (
    "packageName",
    "currentApp",
    "observationMode",
    "datetime",
    "dateTime",
    "timezone",
    "locale",
    "deviceTime",
    "device_time",
    "timestamp",
)


def observed_nodes(ui) -> list[dict]:
    """Keep structure and full values; never turn password/hint labels into facts."""
    result = []

    def visit(nodes, parent=None):
        for node in nodes:
            if not isinstance(node, dict):
                continue
            if node.get("visible", node.get("isVisibleToUser", True)) is False:
                continue
            if node.get("password") or node.get("isPassword"):
                continue
            item = {key: copy.deepcopy(node[key]) for key in NODE_FIELDS if key in node}
            item["path"] = len(result)
            item["parent_path"] = parent
            description = node.get("contentDescription", "")
            if isinstance(description, str) and ":" in description:
                prefix = description.split(":", 1)[0].strip()
                if re.fullmatch(r"[A-Za-z\u3400-\u9fff][\w -]{0,40}", prefix):
                    item["accessibility_type_hint"] = prefix
            result.append(item)
            visit(node.get("children", []), item["path"])

    visit(ui.elements)
    indices = {
        node.get("index"): node["path"]
        for node in result
        if type(node.get("index")) is int
    }
    for node in result:
        if node["parent_path"] is None and node.get("parentIndex") in indices:
            node["parent_path"] = indices[node["parentIndex"]]
    return result


class TaskState:
    """Evidence collected in this task, independent of a benchmark or App name."""

    def __init__(
        self,
        goal: str = "",
        *,
        query_memory: bool = True,
        visual_frame_observations: bool = False,
    ):
        if type(visual_frame_observations) is not bool:
            raise ValueError("visual_frame_observations must be a boolean")
        self.goal = goal
        self.query_memory = query_memory
        self.visual_frame_observations = visual_frame_observations
        self.contract: dict | None = None
        self.observations: list[dict] = []
        self.actions: list[dict] = []
        self._input_progress_targets: dict[str, str | None] = {}
        self._input_progress_changes: dict[str, bool | None] = {}
        self.assessments: dict[int, dict] = {}
        self._last_signature = ""
        self.current_id: str | None = None
        self.device_context: dict = {}
        self.collections = CollectionLedger()
        self.query_results: list[dict] = []
        self._query_native_sources: dict[str, dict] = {}
        self.collection_diagnostic = ""
        self.progress = SemanticProgress()

    def reset(self, goal: str):
        self.__init__(
            goal,
            query_memory=self.query_memory,
            visual_frame_observations=self.visual_frame_observations,
        )

    def set_contract(self, contract: dict):
        self.contract = copy.deepcopy(contract)

    def observe(self, ui, *, screenshot: bytes | None = None) -> dict:
        nodes = observed_nodes(ui)
        image_digest = (
            hashlib.sha256(screenshot).hexdigest() if screenshot is not None else None
        )
        phone = {
            key: copy.deepcopy(ui.phone_state[key])
            for key in PHONE_FIELDS
            if key in ui.phone_state
        }
        signature = hashlib.sha256(
            json.dumps(
                {
                    "phone": phone,
                    "nodes": nodes,
                    "screen_size": [ui.screen_width, ui.screen_height],
                    "normalized": ui.use_normalized,
                },
                sort_keys=True,
                ensure_ascii=False,
                default=str,
            ).encode()
        ).hexdigest()
        # An empty native tree cannot distinguish visual pages. Only an actual
        # changed image creates a new visual observation; native captures keep
        # their existing O#/frame semantics, even if animations change pixels.
        visual_frame_changed = bool(
            self.visual_frame_observations
            and not nodes
            and image_digest is not None
            and self.observations
            and self.observations[-1].get("images")
            and self.observations[-1]["images"][-1]["sha256"] != image_digest
        )
        if (
            self.observations
            and signature == self._last_signature
            and not visual_frame_changed
        ):
            observation = self.observations[-1]
        else:
            oid = f"O{len(self.observations) + 1}"
            for index, node in enumerate(nodes):
                node["ref"] = f"{oid}:n{index}"
            boundaries = control_observations(ui)
            refs = {node.get("index"): node["ref"] for node in nodes}
            containers = []
            for container in boundaries["containers"]:
                if container["index"] not in refs:
                    continue
                children = container.pop("child_indices")
                container["ref"] = refs[container["index"]]
                container["child_refs"] = [
                    refs[index] for index in children if index in refs
                ]
                container["child_refs_omitted"] = container.pop(
                    "child_indices_omitted"
                ) + sum(index not in refs for index in children)
                containers.append(container)
            observation = {
                "id": oid,
                "source": "accessibility",
                "captured_at": time.time(),
                "phone": phone,
                "nodes": nodes,
                "containers": containers,
                "containers_omitted": boundaries["containers_omitted"]
                + len(boundaries["containers"])
                - len(containers),
                "coordinate_space": (
                    "normalized_1000" if ui.use_normalized else "native_pixels"
                ),
                "screen_size": [ui.screen_width, ui.screen_height],
            }
            self.observations.append(observation)
            self._last_signature = signature
            try:
                if self.query_memory:
                    self.collections.observe(observation)
                self.collection_diagnostic = ""
            except QueryError as exc:
                self.collection_diagnostic = str(exc)
        if screenshot is not None:
            digest = image_digest
            # A new image must not silently overwrite an older image reference.
            images = observation.setdefault("images", [])
            # Deduplicate a consecutive identical capture, not an earlier image
            # after the screen has changed away and back (A -> B -> A).
            if not images or images[-1]["sha256"] != digest:
                images.append(
                    {
                        "ref": f"{observation['id']}:image{len(images) + 1}",
                        "sha256": digest,
                        "source": "screenshot",
                    }
                )
        self.current_id = observation["id"]
        if self.actions and "after" not in self.actions[-1]:
            self.actions[-1]["after"] = self.current_id
            intermediate = observed_keypad_input(self, self.actions[-1])
            if intermediate is not None:
                self.actions[-1]["intermediate_effect"] = intermediate
            self.progress.update(self)
        elif self.progress.signature is None:
            self.progress.update(self)
        return copy.deepcopy(observation)

    def record_action(
        self,
        call,
        result,
        *,
        input_source=None,
        input_target=None,
        native_target_binding=None,
    ) -> dict:
        action = {
            "ref": f"A{len(self.actions) + 1}",
            "before": self.current_id,
            "tool": call.name,
            "arguments": copy.deepcopy(call.parameters),
            "dispatch_status": "dispatched" if result.success else "unknown",
            "summary": result.summary,
            "effect_status": "unknown",
        }
        if input_source is not None:
            action["input_source"] = copy.deepcopy(input_source)
        if native_target_binding is not None:
            action["native_target_binding"] = copy.deepcopy(native_target_binding)
            owner_index = native_target_binding.get("owner_index")
            if type(owner_index) is int and self.observations:
                owners = [
                    node
                    for node in self.observations[-1]["nodes"]
                    if node.get("index") == owner_index
                    and isinstance(node.get("collectionItemInfo"), dict)
                ]
                if len(owners) == 1:
                    action["native_target_binding"]["owner_ref"] = owners[0]["ref"]
        if call.name == "type" and input_target is not None:
            self._input_progress_targets[action["ref"]] = input_target_progress_key(
                input_target
            )
            self._input_progress_changes[action["ref"]] = input_target_changed(
                input_target, call.parameters.get("text")
            )
        self.actions.append(action)
        return copy.deepcopy(action)

    def record_input_effect(self, status: str):
        if self.actions and self.actions[-1]["tool"] == "type":
            self.actions[-1]["effect_status"] = status
            self.progress.update(self)

    def native_text_snapshot(self):
        return native_text_snapshot(self.observations, self.goal)

    def evidence(self) -> dict[str, dict]:
        items = {item["ref"]: item for item in self.actions}
        for observation in self.observations:
            items[observation["id"]] = observation
            for item in (*observation["nodes"], *observation.get("images", [])):
                items[item["ref"]] = item
        for result in self.query_results:
            scope = self.collections.scopes.get(result["scope"])
            if scope is not None and result["generation"] == scope.generation:
                items[result["ref"]] = result
        return items

    def query(self, plan: dict) -> dict:
        result = self.collections.execute(
            plan, device_datetime=self.device_context.get("datetime")
        )
        result.update(
            ref=f"Q{len(self.query_results) + 1}", source="deterministic_query"
        )
        scope = self.collections.scopes[result["scope"]]
        result["scope_source"] = {
            "observation": scope.container_ref.split(":")[0],
            "container_ref": scope.container_ref,
            "context_refs": [item["ref"] for item in scope.native_context],
        }
        if result["filters"]:
            names = {condition["field"] for condition in result["filters"]}
            result["filter_evidence_refs"] = list(
                dict.fromkeys(
                    ref
                    for row in scope.rows.values()
                    if not row["heading"]
                    for name, fact in row["facts"].items()
                    if name in names
                    for ref in fact["evidence_refs"]
                )
            )
        self.query_results.append(copy.deepcopy(result))
        self._query_native_sources[result["ref"]] = capture_query_source(
            result,
            self.collections.scopes[result["scope"]],
            goal=self.goal,
            evidence=self.evidence(),
        )
        self.progress.update(self)
        return copy.deepcopy(result)

    def context(
        self,
        *,
        max_observations=4,
        max_chars=18_000,
        max_actions=32,
        retain_proofs=True,
        share_native_values=False,
    ) -> dict[str, Any]:
        """Bound model context while keeping the full ledger locally.

        Recent pages and pages supporting accepted requirements are retained.
        Explicit omission counts prevent treating a summary as full coverage.
        """
        pages: dict[str, list[dict]] = {}
        for observation in self.observations:
            markers = [
                {
                    key: node[key]
                    for key in (
                        "resourceId",
                        "className",
                        "rawText",
                        "text",
                        "contentDescription",
                    )
                    if key in node
                }
                for node in observation["nodes"]
                if not (
                    node.get("editable")
                    or node.get("isEditable")
                    or "EditText" in node.get("className", "")
                )
                and not re.fullmatch(
                    r"[\d\s:.,%+-]+", str(node.get("rawText", node.get("text", "")))
                )
                and (
                    node.get("rawText")
                    or node.get("contentDescription")
                    or node.get("text")
                )
            ]
            page_key = {
                "package": observation["phone"].get("packageName"),
                "markers": markers,
            }
            # Empty native trees have no page markers. Group their actual
            # captured pixels so a third visual page does not evict the first
            # while still inside the existing recent-page window. A digest
            # identifies a captured frame, not a semantic page or object.
            if (
                self.visual_frame_observations
                and not observation["nodes"]
                and observation.get("images")
            ):
                page_key["visual_frame_sha256"] = observation["images"][-1]["sha256"]
            key = json.dumps(
                page_key,
                sort_keys=True,
                ensure_ascii=False,
            )
            pages.setdefault(key, []).append(observation)
        recent_pages = sorted(pages.values(), key=lambda page: page[-1]["captured_at"])[
            -max_observations:
        ]
        retained = {
            item["id"]
            for page in recent_pages
            for item in page[-(2 if retain_proofs else 1) :]
        }
        for assessment in self.assessments.values() if retain_proofs else ():
            retained.update(
                ref.split(":")[0] for ref in assessment.get("evidence_refs", [])
            )
        if retain_proofs:
            retained.update(
                action["input_source"]["ref"].split(":")[0]
                for action in self.actions
                if action.get("input_source")
            )
            retained.update(
                action["before"]
                for action in self.actions[-max_actions:]
                if action.get("native_target_binding") is not None
                and action.get("before")
            )
        observations = [
            copy.deepcopy(item) for item in self.observations if item["id"] in retained
        ]
        # Decision candidates already contain geometry. The memory needs facts,
        # not a second full set of coordinates, class flags and duplicate labels.
        for observation in observations:
            boundary_refs = {
                ref
                for container in observation.get("containers", [])
                for ref in [container["ref"], *container["child_refs"]]
            }
            compact = []
            for node in observation["nodes"]:
                item = {
                    key: value
                    for key, value in node.items()
                    if key
                    in {
                        "ref",
                        "parent_path",
                        "path",
                        "index",
                        "text",
                        "contentDescription",
                        "rawText",
                        "stateDescription",
                        "accessibility_type_hint",
                        "inputValue",
                        "hint",
                        "inputType",
                        "maxTextLength",
                        "textSelectionStart",
                        "textSelectionEnd",
                        "isMultiLine",
                        "className",
                        "resourceId",
                        "checkedState",
                        "isChecked",
                        "checked",
                        "isCheckable",
                        "checkable",
                        "rangeInfo",
                        "collectionInfo",
                        "collectionItemInfo",
                        "actionList",
                        "childCount",
                        "enabled",
                        "isEnabled",
                        "isShowingHintText",
                    }
                }
                actual_label = (
                    node.get("rawText")
                    or node.get("contentDescription")
                    or node.get("label")
                )
                if not actual_label and "rawText" not in node:
                    text = node.get("text", "")
                    actual_label = (
                        text
                        and text != node.get("resourceId")
                        and text.split(".")[-1] != node.get("className")
                    )
                if not (
                    actual_label
                    or node["ref"] in boundary_refs
                    or "inputValue" in item
                    or item.get("rangeInfo")
                    or item.get("collectionInfo")
                    or item.get("collectionItemInfo")
                    or "actionList" in item
                    or "childCount" in item
                    or item.get("isCheckable") is True
                    or item.get("checkable") is True
                    or item.get("checkedState")
                ):
                    continue
                if node["ref"] in boundary_refs and "bounds" in node:
                    item["bounds"] = node["bounds"]
                # Empty fields provide no extra fact; retain explicit False/0.
                item = {
                    key: value
                    for key, value in item.items()
                    if value not in (None, "") or key == "inputValue" and value == ""
                }
                compact.append(
                    compact_native_values(item) if share_native_values else item
                )
            observation["nodes_omitted"] = len(observation["nodes"]) - len(compact)
            observation["nodes_without_retained_facts"] = observation["nodes_omitted"]
            observation["nodes_budget_omitted"] = 0
            observation["nodes"] = compact
        # Prefer the current and newer pages under the model budget. All original
        # observations are still available locally, and omissions are explicit.
        budget = max_chars
        for observation in reversed(observations):
            containers = []
            for container in observation.get("containers", []):
                cost = len(json.dumps(container, ensure_ascii=False, default=str))
                if cost <= budget:
                    containers.append(container)
                    budget -= cost
                else:
                    observation["containers_omitted"] += 1
            observation["containers"] = containers
            kept = []
            for node in observation["nodes"]:
                cost = len(json.dumps(node, ensure_ascii=False, default=str))
                if cost <= budget:
                    kept.append(node)
                    budget -= cost
                else:
                    observation["nodes_omitted"] += 1
                    observation["nodes_budget_omitted"] += 1
            observation["nodes"] = kept
            kept_refs = {node["ref"] for node in kept}
            grounded = []
            for container in observation["containers"]:
                if container["ref"] not in kept_refs:
                    observation["containers_omitted"] += 1
                    continue
                child_refs = container["child_refs"]
                container["child_refs"] = [
                    ref for ref in child_refs if ref in kept_refs
                ]
                container["child_refs_omitted"] += len(child_refs) - len(
                    container["child_refs"]
                )
                grounded.append(container)
            observation["containers"] = grounded
        requirements = (self.contract or {}).get("requirements", [])
        remaining = [
            index
            for index in range(1, len(requirements) + 1)
            if self.assessments.get(index, {}).get("status") != "supported"
        ]
        return {
            "contract": copy.deepcopy(self.contract),
            "current_observation": self.current_id,
            "device_context": copy.deepcopy(self.device_context),
            "observations": observations,
            "actions": copy.deepcopy(self.actions[-max_actions:]),
            "observations_omitted": len(self.observations) - len(observations),
            "actions_omitted": max(0, len(self.actions) - max_actions),
            "remaining_requirement_ids": remaining,
            "assessments": list(copy.deepcopy(self.assessments).values()),
            "collections": self.collections.context(
                max_rows=3 if not retain_proofs else 40,
                max_chars=2000 if not retain_proofs else 12000,
            ),
            "collection_diagnostic": self.collection_diagnostic,
            "query_results": [
                copy.deepcopy(result)
                for result in self.query_results[-8:]
                if result["ref"] in self.evidence()
            ],
            "semantic_progress": self.progress.context(self),
        }


def corroborate_nodes(proofs: list[dict], ui) -> bool:
    """Recheck cited controls, allowing unrelated dynamic UI to change."""
    current = observed_nodes(ui)
    properties = PROOF_PROPERTIES
    for proof in proofs:
        candidates = [
            node
            for node in current
            if node.get("className") == proof.get("className")
            and node.get("resourceId") == proof.get("resourceId")
        ]
        if len(candidates) != 1:
            candidates = [
                node
                for node in candidates
                if node.get("bounds") == proof.get("bounds")
                and all(node.get(key) == proof.get(key) for key in properties)
            ]
        if len(candidates) != 1 or any(
            candidates[0].get(key) != proof.get(key) for key in properties
        ):
            return False
    return True
