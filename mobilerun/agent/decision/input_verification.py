"""Bind a text input before typing and read back that same field afterwards.

Formatted UI labels are not input values: IndexedFormatter can replace an empty
text with a hint, resource ID, or class name. Prefer the raw Portal value and
only use explicit value metadata when operating on a formatted UIState alone.
"""

from __future__ import annotations

import hashlib
import json
import math
import re
from dataclasses import dataclass
from typing import Any

from mobilerun.tools.ui.provider import StaleObservationError, require_fresh_observation


@dataclass(frozen=True)
class InputTarget:
    package_name: str
    resource_id: str
    class_name: str
    bounds: tuple[float, float, float, float] | None
    parent: tuple[Any, ...]
    unique_resource_id: bool
    require_focused: bool  # Capture-time requirement only; apps may blur after typing.
    page_context: tuple = ()
    hint: str = ""
    window_id: int | None = None
    unique_hint: bool = False
    native_type: int | None = None
    before_value_sha256: str | None = None
    before_phone_sha256: str | None = None
    native_owner_sha256: str | None = None


@dataclass(frozen=True)
class InputReadback:
    status: str  # confirmed, mismatch, or unknown
    reason: str
    observed_length: int | None = None


@dataclass(frozen=True)
class _InputNode:
    package: str
    resource: str
    class_name: str
    bounds: tuple[float, float, float, float] | None
    parent: tuple[Any, ...]
    index: int | None
    focused: bool
    editable: bool
    password: bool
    enabled: bool
    value: str | None
    hint: str
    window_id: int | None
    native_type: int | None = None
    native_owner_sha256: str | None = None
    native_owner_valid: bool = True
    selection: tuple[int, int] | None = None


@dataclass(frozen=True)
class _Snapshot:
    package: str
    nodes: tuple[_InputNode, ...]


def _string(value: Any) -> str:
    return value if isinstance(value, str) else ""


def _package(node: dict) -> str:
    return next(
        (
            _string(node[k])
            for k in ("packageName", "package_name", "package")
            if node.get(k)
        ),
        "",
    )


def _resource(node: dict) -> str:
    return _string(
        node.get("resourceId")
        or node.get("resource_id")
        or node.get("viewIdResourceName")
    )


def _class(node: dict) -> str:
    return _string(node.get("className") or node.get("class")).split(".")[-1]


def _flag(node: dict, *keys: str) -> bool | None:
    observed = [node[key] for key in keys if key in node]
    if not observed:
        return None
    if any(value is True for value in observed):
        return True
    if all(value is False for value in observed):
        return False
    return None


def _bounds(node: dict, ui=None) -> tuple[float, float, float, float] | None:
    value = node.get("boundsInScreen", node.get("bounds"))
    if isinstance(value, dict):
        value = [value.get(k) for k in ("left", "top", "right", "bottom")]
    elif isinstance(value, str):
        value = value.split(",")
    if not isinstance(value, (list, tuple)) or len(value) != 4:
        return None
    try:
        if any(isinstance(v, bool) for v in value):
            return None
        result = tuple(float(v) for v in value)
        if any(not math.isfinite(v) for v in result):
            return None
        if ui is not None and getattr(ui, "use_normalized", False):
            result = tuple(
                v * size / 1000
                for v, size in zip(
                    result, (ui.screen_width, ui.screen_height) * 2, strict=True
                )
            )
        left, top, right, bottom = result
        return result if right > left and bottom > top else None
    except (ValueError, TypeError, OverflowError):
        return None


def _anchor(node: dict | None, ui=None) -> tuple[Any, ...]:
    if node is None:
        return ()
    return (_resource(node), _class(node), _bounds(node, ui))


def _value(node: dict, *, raw: bool) -> str | None:
    if node.get("isShowingHintText") is True:
        return ""
    keys = (
        ("text", "inputValue", "rawText", "value")
        if raw
        else ("inputValue", "rawText", "value")
    )
    for key in keys:
        if key in node:
            value = node[key]
            if raw and key == "text" and value is None:
                return ""  # Android AccessibilityNodeInfo.getText() can be null for an empty input.
            return value if isinstance(value, str) else None
    return None


def _walk(tree: Any, parent=None, inherited_package: str = ""):
    if isinstance(tree, list):
        for entry in tree:
            yield from _walk(entry, parent, inherited_package)
        return
    if not isinstance(tree, dict):
        return
    markers = {
        "className",
        "class",
        "resourceId",
        "boundsInScreen",
        "bounds",
        "isEditable",
        "editable",
    }
    if not markers.intersection(tree):
        # Portal snapshots may contain multiple window roots in a list, keyed
        # dictionary, or roots/windows wrapper. Never treat focusedElement as a root.
        for key, value in tree.items():
            if key not in {
                "phone_state",
                "phoneState",
                "device_context",
                "focusedElement",
            }:
                yield from _walk(value, parent, inherited_package)
        return
    package = _package(tree) or inherited_package
    yield tree, parent, package
    yield from _walk(tree.get("children", []), tree, package)


def _native_input_owner(selected, entries, parents, children, invalid_parents, *, raw):
    """Bind an explicit native row without freezing the edited value or focus.

    Indices and coordinates are observation-local; native row roles, hierarchy,
    other values and ancestor labels supply this capture-time dependency. This
    does not identify a persistent business object or prove a saved effect.
    """

    def ancestry(node):
        seen = set()
        while node is not None:
            key = id(node)
            if key in seen or key in invalid_parents or len(seen) >= 128:
                raise ValueError("invalid_native_input_owner")
            seen.add(key)
            item = node.get("collectionItemInfo")
            if item is not None and not isinstance(item, dict):
                raise ValueError("invalid_native_input_owner")
            yield node
            node = parents.get(key)

    def meaning(node):
        edited = node is selected
        value = None if edited else _value(node, raw=raw)
        if not edited and value is None:
            # IndexedFormatter generates display labels for missing raw text.
            label = _string(node.get("text"))
            generated = label in {_resource(node), _class(node)} or label.split(".")[
                -1
            ] == _class(node)
            value = "" if raw or generated else label
        return {
            "resource": _resource(node),
            "class": _class(node),
            "value": "[edited field]" if edited else value,
            "description": _string(node.get("contentDescription")),
            "item": node.get("collectionItemInfo"),
            "collection": isinstance(node.get("collectionInfo"), dict),
        }

    def subtree(node, seen=()):
        key = id(node)
        if key in seen or key in invalid_parents or len(seen) >= 128:
            raise ValueError("invalid_native_input_owner")
        return {
            "meaning": meaning(node),
            "children": [
                subtree(child, (*seen, key)) for child in children.get(key, ())
            ],
        }

    try:
        chain = list(ancestry(selected))
        owner = next(
            (
                node
                for node in chain
                if isinstance(node.get("collectionItemInfo"), dict)
            ),
            None,
        )
        if owner is None:
            return True, None
        owner_position = next(
            position for position, node in enumerate(chain) if node is owner
        )
        ancestors = chain[owner_position + 1 :]
        payload = {
            "row": subtree(owner),
            "ancestors": [
                (
                    subtree(node)
                    if isinstance(node.get("collectionItemInfo"), dict)
                    else meaning(node)
                )
                for node in ancestors
            ],
        }
        # entries guards against a parent pointing outside this observed tree.
        if any(id(node) not in entries for node in chain):
            return False, None
        encoded = json.dumps(
            payload, ensure_ascii=False, sort_keys=True, allow_nan=False
        )
        return True, hashlib.sha256(encoded.encode("utf-8")).hexdigest()
    except (ValueError, TypeError, UnicodeError, RecursionError):
        return False, None


def _snapshot(tree: Any, phone_state: dict, *, raw: bool, ui=None) -> _Snapshot:
    package = _package(phone_state)
    entries = list(_walk(tree, inherited_package=package))
    if not package:
        observed = {pkg for _, _, pkg in entries if pkg}
        if len(observed) == 1:
            package = observed.pop()

    # Flat indexed UI trees preserve parentIndex; nested raw trees use actual parents.
    indexed: dict[int, list[dict]] = {}
    for node, _, _ in entries:
        index = node.get("index")
        if type(index) is int:
            indexed.setdefault(index, []).append(node)
    parents, children, invalid_parents = {}, {}, set()
    for node, parent, _ in entries:
        if not raw and node.get("parentIndex") is not None:
            index = node["parentIndex"]
            possible = indexed.get(index, []) if type(index) is int else []
            if len(possible) != 1:
                invalid_parents.add(id(node))
            parent = possible[0] if len(possible) == 1 else None
        parents[id(node)] = parent
        if parent is not None:
            children.setdefault(id(parent), []).append(node)
    by_identity = {id(node) for node, _, _ in entries}
    nodes = []
    for node, parent, node_package in entries:
        if not raw and type(node.get("parentIndex")) is int:
            possible = indexed.get(node["parentIndex"], [])
            parent = possible[0] if len(possible) == 1 else None
        cls = _class(node)
        editable_flag = _flag(node, "editable", "isEditable")
        editable = editable_flag is True or (
            editable_flag is None
            and cls
            in {
                "EditText",
                "AutoCompleteTextView",
                "MultiAutoCompleteTextView",
                "TextField",
            }
        )
        password = (
            _flag(node, "password", "isPassword") is True
            or "password" in cls.lower()
            or "secure" in cls.lower()
        )
        input_type = node.get("inputType")
        if type(input_type) is int:
            password |= (input_type & 0xFFF) in {0x81, 0x91, 0xE1, 0x12}
        owner_valid, owner_sha256 = (
            _native_input_owner(
                node,
                by_identity,
                parents,
                children,
                invalid_parents,
                raw=raw,
            )
            if editable
            else (True, None)
        )
        nodes.append(
            _InputNode(
                package=node_package or package,
                resource=_resource(node),
                class_name=cls,
                bounds=_bounds(node, ui),
                parent=_anchor(parent, ui),
                index=node.get("index") if type(node.get("index")) is int else None,
                focused=_flag(node, "focused", "isFocused") is True,
                editable=editable,
                password=password,
                enabled=_flag(node, "enabled", "isEnabled") is not False
                and _flag(node, "visible", "isVisibleToUser") is not False,
                value=_value(node, raw=raw),
                hint=_string(node.get("hint")),
                window_id=(
                    node["windowId"]
                    if type(node.get("windowId")) is int and node["windowId"] >= 0
                    else None
                ),
                native_type=(
                    input_type
                    if type(input_type) is int and 0 <= input_type <= 0x7FFFFFFF
                    else None
                ),
                native_owner_sha256=owner_sha256,
                native_owner_valid=owner_valid,
                selection=(
                    (node["textSelectionStart"], node["textSelectionEnd"])
                    if type(node.get("textSelectionStart")) is int
                    and type(node.get("textSelectionEnd")) is int
                    else None
                ),
            )
        )
    return _Snapshot(package, tuple(nodes))


def _ui_snapshot(ui) -> _Snapshot:
    return _snapshot(ui.elements, ui.phone_state, raw=False, ui=ui)


async def _raw_snapshot(driver) -> _Snapshot | None:
    method = getattr(driver, "get_ui_tree", None)
    if not callable(method):
        return None
    try:
        payload = await method()
        require_fresh_observation(payload)
        if isinstance(payload, dict):
            phone = payload.get("phone_state", payload.get("phoneState", {}))
            phone = phone if isinstance(phone, dict) else {}
            tree = payload.get("a11y_tree", payload.get("tree", payload))
        elif isinstance(payload, list):
            phone, tree = {}, payload
        else:
            return None
        result = _snapshot(tree, phone, raw=True)
        return result if result.package and result.nodes else None
    except StaleObservationError:
        # A known stale response must not fall back to a previous parsed UI.
        return _Snapshot("", ())
    except Exception:
        # Reading is optional; do not leak Portal errors, URLs or field values.
        return None


def _eligible(node: _InputNode) -> bool:
    return (
        node.editable and node.enabled and not node.password and bool(node.class_name)
    )


def _select(snapshot: _Snapshot, index: int | None) -> _InputNode | None:
    if index is not None:
        if type(index) is not int or index < 0:
            return None
        matches = [node for node in snapshot.nodes if node.index == index]
    else:
        matches = [node for node in snapshot.nodes if node.focused and node.editable]
    if len(matches) != 1 or not _eligible(matches[0]):
        return None
    return matches[0]


def _same_field(
    node: _InputNode,
    selected: _InputNode,
    candidates: tuple[_InputNode, ...],
    *,
    require_position: bool,
) -> bool:
    if node.class_name != selected.class_name:
        return False
    if selected.resource:
        if node.resource != selected.resource:
            return False
        matching = [
            n
            for n in candidates
            if n.resource == selected.resource and n.class_name == selected.class_name
        ]
        if len(matching) == 1 and not require_position:
            return True
    return (
        node.resource == selected.resource
        and selected.bounds is not None
        and node.bounds == selected.bounds
        and node.parent == selected.parent
    )


def _bind(
    snapshot: _Snapshot, selected: _InputNode, *, require_focused: bool
) -> InputTarget | None:
    if (
        not snapshot.package
        or selected.package != snapshot.package
        or not _eligible(selected)
        or selected.value is None
        or not selected.native_owner_valid
        or (require_focused and not selected.focused)
    ):
        return None
    siblings = [
        n
        for n in snapshot.nodes
        if n.package == selected.package
        and n.resource == selected.resource
        and n.class_name == selected.class_name
    ]
    unique = bool(selected.resource) and len(siblings) == 1
    # A native hint identifies a field's role even when the IME resizes its
    # window. Capture still binds the observed position first; readback may
    # follow a unique hint only inside that same native window and parent.
    unique_hint = (
        not unique
        and bool(selected.hint)
        and selected.window_id is not None
        and len(
            [
                n
                for n in siblings
                if n.hint == selected.hint and n.window_id == selected.window_id
            ]
        )
        == 1
    )
    if not unique:
        located = [
            n
            for n in siblings
            if n.bounds == selected.bounds and n.parent == selected.parent
        ]
        if selected.bounds is None or len(located) != 1:
            return None
    return InputTarget(
        snapshot.package,
        selected.resource,
        selected.class_name,
        selected.bounds,
        selected.parent,
        unique,
        require_focused,
        _page_context(snapshot),
        selected.hint,
        selected.window_id,
        unique_hint,
        selected.native_type,
        _value_digest(selected.value),
        (
            _value_digest(_phone_value(selected.value))
            if selected.native_type is not None and selected.native_type & 0xF == 3
            else None
        ),
        selected.native_owner_sha256,
    )


def _value_digest(value: str | None) -> str | None:
    if not isinstance(value, str):
        return None
    try:
        return hashlib.sha256(value.encode("utf8")).hexdigest()
    except UnicodeEncodeError:
        return None


def input_target_changed(target: InputTarget | None, text: str) -> bool | None:
    """Compare the intended value with the privately captured pre-input value.

    This classifies progress only. The ordinary independent post-input readback
    must still confirm the complete value and field; this never authorizes Type.
    """
    if not isinstance(target, InputTarget) or not isinstance(text, str):
        return None
    before, intended = target.before_value_sha256, _value_digest(text)
    if (
        not isinstance(before, str)
        or re.fullmatch(r"[a-f0-9]{64}", before) is None
        or intended is None
    ):
        return None
    if intended == before:
        return False
    if target.native_type is not None:
        if type(target.native_type) is not int:
            return None
        if target.native_type & 0xF == 3:
            phone = _phone_value(text)
            if phone is not None and _value_digest(phone) == target.before_phone_sha256:
                return False
    return True


def _page_context(snapshot: _Snapshot) -> tuple:
    """Use observed text labels to guard replacement in a reused editor ID.

    This is a guard, not a task fact. Dynamic numeric readouts and other
    packages (including the IME) do not identify the edited object.
    """
    return tuple(
        sorted(
            {
                (node.resource, node.value)
                for node in snapshot.nodes
                if node.package == snapshot.package
                and not node.editable
                and node.class_name == "TextView"
                and node.value
                and not re.fullmatch(r"[\d\s:.,%+-]+", node.value)
            }
        )
    )


async def capture_input_target(driver, ui_state, index=None) -> InputTarget | None:
    """Capture only the indexed input or the one unambiguously focused input.

    None means the caller must reject typing before it mutates the device.
    """
    before = _ui_snapshot(ui_state)
    if ui_state.elements:
        selected = _select(before, index)
        if selected is None:
            return None
        if before.package and selected.package != before.package:
            return None
        raw = await _raw_snapshot(driver)
        if raw is not None:
            if (before.package and before.package != raw.package) or (
                selected.package and selected.package != raw.package
            ):
                return None
            candidates = tuple(n for n in raw.nodes if n.package == raw.package)
            before_matches = [
                n
                for n in before.nodes
                if n.resource == selected.resource
                and n.class_name == selected.class_name
            ]
            matching = [
                n
                for n in candidates
                if _same_field(
                    n,
                    selected,
                    candidates,
                    require_position=len(before_matches) != 1,
                )
            ]
            if len(matching) != 1:
                return None
            if (
                not selected.native_owner_valid
                or not matching[0].native_owner_valid
                or selected.native_owner_sha256 != matching[0].native_owner_sha256
            ):
                return None
            return _bind(raw, matching[0], require_focused=index is None)
        return _bind(before, selected, require_focused=index is None)

    # There is no meaningful UI element index in screenshot-only mode.
    if index is not None:
        return None
    raw = await _raw_snapshot(driver)
    if raw is None or (before.package and before.package != raw.package):
        return None
    selected = _select(raw, None)
    return _bind(raw, selected, require_focused=True) if selected is not None else None


async def input_value_matches(
    driver, ui_state, target: InputTarget | None, text: str
) -> bool:
    """Read the bound field; native phone formatting has narrow equivalence."""
    return (
        await read_input_value(driver, ui_state, target, text)
    ).status == "confirmed"


async def read_input_value(
    driver, ui_state, target: InputTarget | None, text: str
) -> InputReadback:
    """Distinguish an observed wrong value from an unavailable or ambiguous field."""
    if target is None or not isinstance(text, str):
        return InputReadback("unknown", "target_missing")
    ui = _ui_snapshot(ui_state)
    if ui.package and ui.package != target.package_name:
        return InputReadback("unknown", "package_changed")
    current = await _raw_snapshot(driver)
    if current is None:
        current = ui
    if current.package != target.package_name:
        return InputReadback("unknown", "package_changed_or_stale")
    matching = [
        n
        for n in current.nodes
        if n.package == target.package_name
        and n.resource == target.resource_id
        and n.class_name == target.class_name
    ]
    if target.unique_hint:
        matching = [
            n
            for n in matching
            if n.hint == target.hint
            and n.window_id == target.window_id
            and n.parent[:2] == target.parent[:2]
        ]
    elif not target.unique_resource_id:
        matching = [
            n
            for n in matching
            if n.bounds == target.bounds and n.parent == target.parent
        ]
    if len(matching) != 1:
        return InputReadback("unknown", "field_missing_or_ambiguous")
    found = matching[0]
    if target.window_id is not None and found.window_id != target.window_id:
        return InputReadback("unknown", "input_window_changed")
    # Live search and auto-advance can move focus after successful input. The
    # original field identity and exact value establish readback, not its focus.
    if not _eligible(found) or found.value is None:
        return InputReadback("unknown", "value_unobservable")
    if (
        not found.native_owner_valid
        or found.native_owner_sha256 != target.native_owner_sha256
    ):
        return InputReadback("unknown", "input_owner_changed")
    if found.value == text:
        return InputReadback("confirmed", "exact_value", len(found.value))
    # Some apps prefix their field label to accessibility text ("搜尋, value").
    # A collapsed native caret exactly at the typed length (UTF-16 units) shows
    # the editable content is the typed value, not the composite label text.
    typed_units = len(text.encode("utf-16-le")) // 2
    if text and found.value.endswith(text) and found.selection == (typed_units,) * 2:
        return InputReadback("confirmed", "labelled_exact_value", len(text))
    if target.page_context and _page_context(current) != target.page_context:
        return InputReadback("unknown", "edited_object_context_changed")
    if target.native_type is not None and target.native_type & 0xF == 3:
        # Android TYPE_CLASS_PHONE can insert display separators. This comparison
        # happens only after field identity, window and eligibility checks; never
        # locate a field by matching its contents or infer type from its hint.
        if found.native_type is None:
            return InputReadback("unknown", "native_input_type_unobservable")
        if found.native_type & 0xF != 3:
            return InputReadback("unknown", "native_input_type_changed")
        wanted, observed = _phone_value(text), _phone_value(found.value)
        if wanted is not None and wanted == observed:
            return InputReadback("confirmed", "native_phone_format", len(found.value))
    return InputReadback("mismatch", "different_value", len(found.value))


def _phone_value(value: str) -> str | None:
    """Remove display separators, preserving +, leading zeros and all digits.

    No inferred country codes, extensions, pause/wait operators, dial commands,
    letters, Unicode digits or line breaks participate in this equivalence.
    """
    canonical = re.sub(r"[ ()\-\u00a0\u202f]", "", value)
    return canonical if re.fullmatch(r"\+?[0-9]+", canonical) else None
