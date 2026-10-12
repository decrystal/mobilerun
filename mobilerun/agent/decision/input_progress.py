"""Read back indexed native keypad input; never certify a task requirement."""

from __future__ import annotations

import re

from mobilerun.agent.decision.contracts import observed_temporal_readout


def value(node):
    if (
        node.get("isShowingHintText")
        or node.get("rangeInfo")
        or node.get("checkable")
        or node.get("isCheckable")
    ):
        return None
    text = node.get("inputValue", node.get("rawText", node.get("text")))
    if not isinstance(text, str) or not re.search(r"[0-9]", text):
        return None
    digits = "".join(re.findall(r"[0-9]", text))
    if len(digits) > 32:
        return None
    return text, int(digits), re.sub(r"[0-9]+", "#", text)


def observed_keypad_input(state, action):
    """Require a numeric goal, a full native keypad and a unique exact readback."""
    rid = state.progress.current_requirement(state)
    requirement = next(
        (
            item
            for item in (state.contract or {}).get("requirements", [])
            if item["id"] == rid
        ),
        None,
    )
    if (
        not requirement
        or requirement["kind"] not in {"state", "final_state"}
        or not re.search(r"[0-9]", requirement["quote"])
        or requirement["quote"] not in state.goal
        or action.get("tool") != "click"
        or action.get("dispatch_status") != "dispatched"
        or type(action.get("arguments", {}).get("index")) is not int
    ):
        return None
    pages = {page["id"]: page for page in state.observations}
    before, after = pages.get(action.get("before")), pages.get(action.get("after"))
    if (
        before is None
        or after is None
        or before["id"] == after["id"]
        or not before["phone"].get("packageName")
        or before["phone"].get("packageName") != after["phone"].get("packageName")
        or before["screen_size"] != after["screen_size"]
        or before["coordinate_space"] != after["coordinate_space"]
    ):
        return None
    keys = {}
    for node in before["nodes"]:
        label = node.get("rawText", node.get("text", ""))
        if (
            isinstance(label, str)
            and re.fullmatch(r"[0-9]", label)
            and node.get("className", "").split(".")[-1] == "Button"
            and node.get("isEnabled", node.get("enabled", True)) is not False
        ):
            keys.setdefault(label, []).append(node)
    if set(keys) != set("0123456789") or any(
        len(items) != 1 for items in keys.values()
    ):
        return None
    tapped = [
        node
        for nodes in keys.values()
        for node in nodes
        if node.get("index") == action["arguments"]["index"]
    ]
    if len(tapped) != 1:
        return None
    digit = tapped[0].get("rawText", tapped[0].get("text"))
    candidates = []
    for old in before["nodes"]:
        resource = old.get("resourceId")
        previous = value(old)
        if not resource or previous is None or old in tapped:
            continue
        identity = (resource, old.get("className"))
        old_matches = [
            node
            for node in before["nodes"]
            if (node.get("resourceId"), node.get("className")) == identity
        ]
        new_matches = [
            node
            for node in after["nodes"]
            if (node.get("resourceId"), node.get("className")) == identity
        ]
        if len(old_matches) != 1 or len(new_matches) != 1:
            continue
        new = new_matches[0]
        current = value(new)
        if (
            current is None
            or current[0] == previous[0]
            or current[2] != previous[2]
            or current[1] != previous[1] * 10 + int(digit)
        ):
            continue
        evidence = state.evidence()
        evidence.pop(action["ref"], None)
        if observed_temporal_readout(old["ref"], evidence, before["id"]):
            continue
        candidates.append(
            {
                "kind": "native_keypad_input",
                "requirement_id": rid,
                "package": before["phone"]["packageName"],
                "resource_id": resource,
                "before_ref": old["ref"],
                "after_ref": new["ref"],
                "before_value": previous[0],
                "after_value": current[0],
                "after_digits": str(current[1]),
                "digit": digit,
                "status": "observed",
                "requirement_satisfied": False,
            }
        )
    return candidates[0] if len(candidates) == 1 else None
