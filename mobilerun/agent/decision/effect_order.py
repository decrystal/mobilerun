"""Check result-read ordering using the existing task-local action ledger.

This proves that a result was read after a cited action, not that the action
caused it, selected the right business object, or persisted it to storage.
"""

from __future__ import annotations

import re


def _observation_number(ref):
    match = re.fullmatch(r"O([1-9]\d*)", ref) if isinstance(ref, str) else None
    return int(match[1]) if match else None


def has_post_action_result(refs: list[str], evidence: dict, current_id: str) -> bool:
    """Require a result witness after at least one action cited for this effect.

    Other cited actions may be later navigation. Do not force a saved historical
    result onto the final page, or treat all cited actions as mutations.
    """
    actions = [evidence[ref] for ref in refs if re.fullmatch(r"A[1-9]\d*", ref)]
    if not actions:
        return True
    # A device check only lists results added after its own action.
    if any(evidence[ref].get("source") == "device_probe" for ref in refs):
        return True
    current = _observation_number(current_id)
    if current is None:
        return False
    boundaries = []
    for action in actions:
        before = _observation_number(action.get("before"))
        after = _observation_number(action.get("after"))
        if before is not None and after is not None and before <= after <= current:
            boundaries.append((before, after))

    def follows(ref, boundary):
        match = re.fullmatch(r"O([1-9]\d*):(n|image)\d+", ref)
        if not match or ref not in evidence:
            return False
        number = int(match[1])
        before, after = boundary
        # Identical native captures reuse their O# but still set action.after.
        # Pixels can change under the same native O#: without a separate pixel
        # capture boundary, an image at the pre-action O# remains ambiguous.
        return after <= number <= current and (match[2] == "n" or number > before)

    for boundary in boundaries:
        for ref in refs:
            if follows(ref, boundary):
                return True
            result = evidence[ref]
            source_refs = result.get("evidence_refs")
            if (
                result.get("source") == "deterministic_query"
                and result.get("coverage", {}).get("complete") is True
                and isinstance(source_refs, list)
                and source_refs
                and all(
                    isinstance(source, str) and follows(source, boundary)
                    for source in source_refs
                )
            ):
                return True
    return False
