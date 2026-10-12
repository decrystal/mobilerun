"""Bounded recovery based on grounded progress, independent of rendering noise."""

from __future__ import annotations

import hashlib
import json

from mobilerun.agent.decision.input_verification import InputTarget


def input_target_progress_key(target) -> str | None:
    """Privately distinguish native fields using the existing readback binding.

    This is a goal-local progress key, not a persistent object ID or save proof.
    Position is only an identity fallback when resource/hint uniqueness is absent.
    """
    if not isinstance(target, InputTarget):
        return None

    def position(value):
        if value is None:
            return None
        if not isinstance(value, (tuple, list)) or len(value) != 4:
            raise ValueError("Invalid native position")
        if any(type(item) not in (int, float) for item in value):
            raise ValueError("Invalid native position number")
        return tuple(float(item) for item in value)

    try:
        identity = {
            "package": target.package_name,
            "resource": target.resource_id,
            "class": target.class_name,
            "window": target.window_id,
            "page_context": target.page_context,
        }
        if target.native_owner_sha256 is not None:
            identity["native_owner_sha256"] = target.native_owner_sha256
        if target.unique_resource_id:
            identity["binding"] = "resource"
        elif target.unique_hint:
            identity.update(binding="hint", hint=target.hint, parent=target.parent[:2])
        else:
            parent = target.parent
            if parent:
                if len(parent) != 3:
                    return None
                parent = (parent[0], parent[1], position(parent[2]))
            identity.update(
                binding="position", bounds=position(target.bounds), parent=parent
            )
        encoded = json.dumps(
            identity, sort_keys=True, ensure_ascii=False, allow_nan=False
        )
        return hashlib.sha256(encoded.encode()).hexdigest()
    except (TypeError, ValueError, OverflowError):
        return None


class SemanticProgress:
    def __init__(self):
        self.signature = None
        self.last_action_count = 0
        self.stalled_actions = 0
        self.recoveries: dict[int, int] = {}
        self.seen_supported = set()
        self.seen_inputs = set()
        self.seen_queries = set()
        self.seen_intermediate_inputs = set()
        self.seen_scopes = set()
        self.seen_rows: dict[tuple[str, int], set[int]] = {}

    def _confirmed_inputs(self, state):
        inputs = set()
        for action in state.actions:
            if (
                action.get("tool") != "type"
                or action.get("effect_status") != "confirmed"
            ):
                continue
            arguments = action["arguments"]
            if action["ref"] in state._input_progress_targets:
                target = state._input_progress_targets[action["ref"]]
                if (
                    target is None
                    or state._input_progress_changes.get(action["ref"]) is not True
                ):
                    continue
                # An index or other call metadata cannot turn retyping the
                # same confirmed value into new progress in the same field.
                value = {"target": target, "text": arguments.get("text")}
            else:
                # Compatibility for older callers recording only arguments.
                # Runtime clear=True inputs always supply their native binding.
                value = {"legacy_arguments": arguments}
            inputs.add(json.dumps(value, sort_keys=True, ensure_ascii=False))
        return inputs

    def _collection_advanced(self, rows):
        advanced = False
        for scope, generation, positions in rows:
            key = (scope, generation)
            current = set(positions)
            if key in self.seen_rows:
                advanced |= bool(current - self.seen_rows[key])
                self.seen_rows[key].update(current)
            else:
                # A generation change discards coverage. Establish its new
                # baseline without crediting that invalidation as advancement.
                advanced |= scope not in self.seen_scopes and bool(current)
                self.seen_rows[key] = current
            self.seen_scopes.add(scope)
        return advanced

    def update(self, state):
        supported = {
            rid
            for rid, item in state.assessments.items()
            if item.get("status") == "supported"
        }
        inputs = self._confirmed_inputs(state)
        querying = any(
            item["kind"] in {"query", "collection"}
            for item in (state.contract or {}).get("requirements", [])
        )
        rows = (
            sorted(
                (key, scope.generation, tuple(sorted(scope.rows)))
                for key, scope in state.collections.scopes.items()
            )
            if querying
            else []
        )
        evidence = state.evidence()
        queries = {
            json.dumps(
                {
                    key: result.get(key)
                    for key in (
                        "scope",
                        "generation",
                        "operation",
                        "field",
                        "filters",
                        "value",
                    )
                },
                sort_keys=True,
                ensure_ascii=False,
            )
            for result in state.query_results
            if result["ref"] in evidence
        }
        intermediate_inputs = {
            (
                effect["requirement_id"],
                effect["package"],
                effect["resource_id"],
                effect["after_digits"],
            )
            for action in state.actions
            if (effect := action.get("intermediate_effect", {})).get("kind")
            == "native_keypad_input"
            and effect.get("status") == "observed"
        }
        collection_advanced = self._collection_advanced(rows)
        advanced = self.signature is not None and bool(
            supported - self.seen_supported
            or inputs - self.seen_inputs
            or queries - self.seen_queries
            or intermediate_inputs - self.seen_intermediate_inputs
            or collection_advanced
        )
        if advanced:
            self.stalled_actions = 0
        elif len(state.actions) > self.last_action_count:
            self.stalled_actions += len(state.actions) - self.last_action_count
        self.seen_supported.update(supported)
        self.seen_inputs.update(inputs)
        self.seen_queries.update(queries)
        self.seen_intermediate_inputs.update(intermediate_inputs)
        self.signature = (
            sorted(supported),
            sorted(inputs),
            rows,
            sorted(queries),
            sorted(intermediate_inputs),
        )
        self.last_action_count = len(state.actions)

    def current_requirement(self, state) -> int:
        return next(
            (
                item["id"]
                for item in (state.contract or {}).get("requirements", [])
                if state.assessments.get(item["id"], {}).get("status") != "supported"
            ),
            0,
        )

    def request_recovery(self, state, *, threshold=4, budget=2) -> bool:
        rid = self.current_requirement(state)
        if (
            not rid
            or self.stalled_actions < threshold
            or self.recoveries.get(rid, 0) >= budget
        ):
            return False
        self.recoveries[rid] = self.recoveries.get(rid, 0) + 1
        self.stalled_actions = 0
        return True

    def context(self, state) -> dict:
        return {
            "current_requirement_id": self.current_requirement(state),
            "actions_without_grounded_progress": self.stalled_actions,
            "recovery_attempts": dict(self.recoveries),
            "progress_rule": "New requirement support, new confirmed changes from a bound native field's captured before-value, novel exact native keypad readbacks for numeric requirements, new collection rows in a stable generation or newly computed grounded query results. Losing evidence, invalidating coverage, retyping an already-present value, repeating prior support/input/results, or pixel/page changes alone are not progress. Intermediate readbacks do not prove completion or persistence.",
        }
