"""Input readback must prove the exact value in the original editable field."""

import asyncio
from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import AsyncMock

import pytest

from mobilerun.agent.decision.input_verification import (
    capture_input_target,
    input_value_matches,
    read_input_value,
)
from mobilerun.agent.decision.jev import control_observations, focused_input_observation
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.ui.state import UIState

PACKAGE = "com.example.settings"


def _input(
    index=7, resource="search", value="", focused=True, bounds="10,20,300,80", **extra
):
    return {
        "index": index,
        "resourceId": resource,
        "className": "EditText",
        "bounds": bounds,
        "editable": True,
        "focused": focused,
        "password": False,
        "enabled": True,
        "text": value or "搜索",
        "inputValue": value,
        **extra,
    }


def _ui(elements=(), package=PACKAGE):
    return UIState(
        list(elements),
        "",
        "",
        {"packageName": package} if package else {},
        screen_width=1080,
        screen_height=2400,
    )


def _raw_input(text="", resource="search", focused=True, **extra):
    return {
        "className": "android.widget.EditText",
        "resourceId": resource,
        "text": text,
        "isEditable": True,
        "isFocused": focused,
        "isPassword": False,
        "isEnabled": True,
        "boundsInScreen": {"left": 10, "top": 20, "right": 300, "bottom": 80},
        **extra,
    }


def _raw(nodes, package=PACKAGE):
    return {"phone_state": {"packageName": package}, "a11y_tree": nodes}


def _capture(ui, driver=None, index=None):
    return asyncio.run(capture_input_target(driver or SimpleNamespace(), ui, index))


def _matches(ui, target, text, driver=None):
    return asyncio.run(
        input_value_matches(driver or SimpleNamespace(), ui, target, text)
    )


def test_unique_resource_id_survives_index_and_layout_changes():
    before = _ui([_input()])
    target = _capture(before, index=7)
    assert target is not None and target.unique_resource_id
    after = _ui([_input(index=8, value="蓝牙", bounds="20,40,320,100")])
    assert _matches(after, target, "蓝牙")


@pytest.mark.parametrize("actual", ["蓝牙设备", " 蓝牙", "蓝牙 ", "蓝\n牙", "", "搜索"])
def test_readback_is_a_full_exact_value_without_substring_or_whitespace_matching(
    actual,
):
    target = _capture(_ui([_input()]))
    assert not _matches(_ui([_input(value=actual)]), target, "蓝牙")


def test_raw_android_hint_cannot_count_as_successful_typing():
    before = _ui([_input()])
    target = _capture(before)
    driver = SimpleNamespace(
        get_state=AsyncMock(
            return_value=_raw([_raw_input("Search", isShowingHintText=True)])
        )
    )
    assert not _matches(before, target, "Search", driver)


def test_a_different_field_with_the_desired_text_cannot_satisfy_readback():
    target = _capture(_ui([_input(resource="search")]))
    after = _ui(
        [
            _input(resource="search", value="wrong", focused=True),
            _input(index=8, resource="other", value="蓝牙", focused=False),
        ]
    )
    assert not _matches(after, target, "蓝牙")


def test_duplicate_resource_ids_are_bound_by_position_not_the_first_text_match():
    before = _ui(
        [
            _input(index=7, resource="entry", bounds="10,20,300,80"),
            _input(index=8, resource="entry", bounds="10,100,300,160", focused=False),
        ]
    )
    target = _capture(before, index=7)
    assert target is not None and not target.unique_resource_id
    wrong_field = _ui(
        [
            _input(index=9, resource="entry", value="wrong", bounds="10,20,300,80"),
            _input(
                index=10,
                resource="entry",
                value="蓝牙",
                bounds="10,100,300,160",
                focused=False,
            ),
        ]
    )
    assert not _matches(wrong_field, target, "蓝牙")
    wrong_field.elements[0]["inputValue"] = "蓝牙"
    assert _matches(wrong_field, target, "蓝牙")


def test_duplicate_resource_and_position_is_not_a_verifiable_target():
    assert _capture(_ui([_input(), _input(index=8)]), index=7) is None


def test_new_duplicate_id_after_typing_is_ambiguous_even_when_one_value_matches():
    target = _capture(_ui([_input()]), index=7)
    after = _ui(
        [_input(value="蓝牙"), _input(index=8, value="wrong", bounds="10,100,300,160")]
    )
    assert not _matches(after, target, "蓝牙")


def test_focus_change_does_not_redirect_implicit_typing_verification():
    target = _capture(_ui([_input(resource="search")]))
    after = _ui(
        [
            _input(resource="search", value="wrong", focused=False),
            _input(index=8, resource="other", value="蓝牙", focused=True),
        ]
    )
    assert not _matches(after, target, "蓝牙")


def test_same_bound_field_can_lose_focus_after_live_search_refresh():
    target = _capture(_ui([_input(resource="search")]))
    assert target is not None and target.require_focused
    after = _ui(
        [
            _input(index=8, resource="search", value="蓝牙", focused=False),
            _input(index=9, resource="other", value="different", focused=True),
        ]
    )
    assert _matches(after, target, "蓝牙")


def test_vision_only_readback_accepts_auto_blur_only_on_the_original_field():
    search = _raw_input()
    driver = SimpleNamespace(get_ui_tree=AsyncMock(return_value=_raw([search])))
    vision = _ui(package="")
    target = _capture(vision, driver)
    assert target is not None and target.require_focused
    driver.get_ui_tree.return_value = _raw(
        [
            _raw_input(text="蓝牙", focused=False),
            _raw_input(text="other", resource="other", focused=True),
        ]
    )
    assert _matches(vision, target, "蓝牙", driver)
    driver.get_ui_tree.return_value = _raw(
        [
            _raw_input(text="wrong", focused=False),
            _raw_input(text="蓝牙", resource="other", focused=True),
        ]
    )
    assert not _matches(vision, target, "蓝牙", driver)


def test_explicit_index_can_bind_an_unfocused_field_but_does_not_select_another_one():
    ui = _ui([_input(focused=False), _input(index=8, resource="other", focused=True)])
    assert _capture(ui, index=7).resource_id == "search"
    assert _capture(ui, index=99) is None


@pytest.mark.parametrize(
    "nodes",
    [
        [_input(focused=False)],
        [_input(), _input(index=8, resource="other")],
        [_input(password=True)],
        [_input(className="SecureTextField")],
        [_input(inputType=0x81)],
        [_input(editable=False)],
        [_input(enabled=False)],
        [_input(visible=False)],
    ],
)
def test_capture_rejects_ambiguous_unfocused_disabled_or_password_inputs(nodes):
    assert _capture(_ui(nodes)) is None


def test_foreground_package_and_target_package_must_match():
    ui = _ui([_input(packageName="com.other.app")])
    assert _capture(ui, index=7) is None
    target = _capture(_ui([_input()]))
    assert not _matches(
        _ui([_input(value="蓝牙")], package="com.other.app"), target, "蓝牙"
    )


def test_formatted_placeholder_is_not_used_as_input_value():
    raw = _raw_input(text="", contentDescription="蓝牙")
    _, _, elements, _ = IndexedFormatter().format(raw, {"packageName": PACKAGE})
    assert elements[0]["text"] == "蓝牙"
    target = _capture(_ui(elements))
    assert target is not None
    assert not _matches(_ui(elements), target, "蓝牙")
    assert _matches(_ui(elements), target, "")


def test_legacy_ui_without_actual_value_and_without_portal_is_rejected_before_typing():
    node = _input(value="hint")
    node.pop("inputValue")
    assert _capture(_ui([node])) is None


@pytest.mark.parametrize(
    "root_shape", ["list", "keyed_dict", "roots_wrapper", "bare_list"]
)
def test_vision_only_reads_the_focused_field_from_multi_root_portal_trees(root_shape):
    node = _raw_input(packageName=PACKAGE)
    decoration = {
        "className": "android.widget.TextView",
        "text": "other",
        "packageName": PACKAGE,
    }
    if root_shape == "list":
        payload = _raw([node, decoration])
    elif root_shape == "keyed_dict":
        payload = _raw({"window1": node, "window2": decoration})
    elif root_shape == "roots_wrapper":
        payload = _raw({"roots": [node, decoration]})
    else:
        payload = [node, decoration]
    driver = SimpleNamespace(get_ui_tree=AsyncMock(return_value=payload))
    target = _capture(_ui(package=""), driver)
    assert target is not None and target.package_name == PACKAGE
    node["text"] = "中文完整值\n第二行"
    assert _matches(_ui(package=""), target, "中文完整值\n第二行", driver)
    assert driver.get_ui_tree.await_count == 2


def test_vision_only_cannot_invent_an_element_index_or_trust_focused_text_alone():
    driver = SimpleNamespace(
        get_ui_tree=AsyncMock(
            return_value={
                "a11y_tree": [],
                "phone_state": {
                    "packageName": PACKAGE,
                    "focusedElement": {"text": "蓝牙"},
                },
            }
        )
    )
    assert _capture(_ui(package=""), driver, index=7) is None
    assert _capture(_ui(package=""), driver) is None


def test_portal_actual_value_overrides_a_stale_ui_value():
    before = _ui([_input()])
    driver = SimpleNamespace(get_ui_tree=AsyncMock(return_value=_raw(_raw_input())))
    target = _capture(before, driver)
    assert target is not None
    ui_claims_success = _ui([_input(value="蓝牙")])
    driver.get_ui_tree.return_value = _raw(_raw_input(text="failed"))
    assert not _matches(ui_claims_success, target, "蓝牙", driver)


def test_degraded_portal_cache_cannot_bind_input_or_confirm_a_matching_ui_value():
    current = _ui([_input(value="蓝牙")])
    target = _capture(current)
    assert target is not None
    stale = {
        **_raw(_raw_input(text="蓝牙")),
        "degraded": True,
        "degradedReason": "a11y_timeout",
        "snapshotAgeMs": 10000,
    }
    driver = SimpleNamespace(get_ui_tree=AsyncMock(return_value=stale))
    assert _capture(current, driver) is None
    assert not _matches(current, target, "蓝牙", driver)


def test_portal_cross_app_focus_changes_are_rejected_at_capture_and_readback():
    before = _ui([_input()])
    driver = SimpleNamespace(
        get_ui_tree=AsyncMock(return_value=_raw(_raw_input(), package="com.other.app"))
    )
    assert _capture(before, driver) is None
    target = _capture(before)
    assert not _matches(_ui([_input(value="蓝牙")]), target, "蓝牙", driver)


def test_disappearing_duplicate_field_is_not_replaced_by_the_remaining_same_id():
    before = _ui(
        [
            _input(resource="entry", bounds="10,20,300,80"),
            _input(index=8, resource="entry", bounds="10,100,300,160", focused=False),
        ]
    )
    remaining_other_field = _raw_input(
        resource="entry",
        boundsInScreen={"left": 10, "top": 100, "right": 300, "bottom": 160},
    )
    driver = SimpleNamespace(
        get_ui_tree=AsyncMock(return_value=_raw(remaining_other_field))
    )
    assert _capture(before, driver, index=7) is None


def test_missing_id_uses_stable_position_and_parent_without_text_identity():
    parent = {
        "index": 1,
        "className": "ViewGroup",
        "resourceId": "form",
        "bounds": "0,0,1080,2000",
    }
    before = _ui([parent, _input(resource="", parentIndex=1)])
    target = _capture(before, index=7)
    assert target is not None and not target.unique_resource_id
    after = deepcopy(before)
    after.elements[1]["index"] = 9
    after.elements[1]["inputValue"] = "蓝牙"
    assert _matches(after, target, "蓝牙")
    after.elements[0]["resourceId"] = "different_form"
    assert not _matches(after, target, "蓝牙")


def test_unavailable_portal_and_no_explicit_value_do_not_raise_or_leak_errors():
    driver = SimpleNamespace(get_ui_tree=AsyncMock(side_effect=RuntimeError("secret")))
    assert _capture(_ui(package=""), driver) is None
    assert not _matches(_ui(package=""), None, "text", driver)


def test_native_unique_hint_reads_same_idless_field_after_ime_layout_change():
    before = _ui(
        [
            _input(resource=""),
            _input(index=8, resource="", focused=False, bounds="10,100,300,160"),
        ]
    )
    first = _raw_input(resource="", hint="Given name", windowId=57)
    last = _raw_input(
        resource="",
        hint="Family name",
        windowId=57,
        focused=False,
        boundsInScreen={"left": 10, "top": 100, "right": 300, "bottom": 160},
    )
    driver = SimpleNamespace(get_ui_tree=AsyncMock(return_value=_raw([first, last])))
    target = _capture(before, driver, index=7)
    assert target is not None and target.unique_hint
    first.update(
        text="Mei",
        isFocused=False,
        boundsInScreen={"left": 30, "top": 250, "right": 320, "bottom": 310},
    )
    last["boundsInScreen"] = {"left": 30, "top": 350, "right": 320, "bottom": 410}
    assert _matches(before, target, "Mei", driver)
    first["text"], last["text"] = "wrong", "Mei"
    assert not _matches(before, target, "Mei", driver)


def test_native_hint_readback_rejects_window_change_and_new_ambiguity():
    node = _raw_input(resource="", hint="Record title", windowId=57)
    driver = SimpleNamespace(get_ui_tree=AsyncMock(return_value=_raw(node)))
    before = _ui([_input(resource="")])
    target = _capture(before, driver, index=7)
    assert target is not None and target.unique_hint
    node.update(text="Mei", windowId=58)
    assert not _matches(before, target, "Mei", driver)
    node["windowId"] = 57
    other = {
        **node,
        "text": "wrong",
        "boundsInScreen": {"left": 10, "top": 100, "right": 300, "bottom": 160},
    }
    driver.get_ui_tree.return_value = _raw([node, other])
    assert not _matches(before, target, "Mei", driver)


def test_native_hint_does_not_follow_a_field_to_a_different_parent_form():
    node = _raw_input(resource="", hint="Record title", windowId=57)
    form = {"className": "ViewGroup", "resourceId": "edit_form", "children": [node]}
    driver = SimpleNamespace(get_ui_tree=AsyncMock(return_value=_raw(form)))
    before = _ui(
        [
            {"index": 1, "className": "ViewGroup", "resourceId": "edit_form"},
            _input(resource="", parentIndex=1),
        ]
    )
    target = _capture(before, driver, index=7)
    assert target is not None and target.unique_hint
    node["text"] = "Mei"
    form["resourceId"] = "different_form"
    assert not _matches(before, target, "Mei", driver)


def test_formatter_preserves_native_numeric_constraints_for_both_decision_paths():
    raw = _raw_input(
        resource="amount",
        hint="Amount",
        inputType=8194,
        maxTextLength=18,
        windowId=57,
        isMultiLine=False,
        isShowingHintText=True,
    )
    _, _, elements, _ = IndexedFormatter().format(raw, {"packageName": PACKAGE})
    ui = _ui(elements)
    expected = {
        "hint": "Amount",
        "native_type": 8194,
        "kind": "number",
        "decimal": True,
        "signed": False,
        "max_length": 18,
        "multiline": False,
    }
    assert focused_input_observation(ui)["constraints"] == expected
    assert control_observations(ui)["controls"][0]["input_constraints"] == expected
    assert elements[0]["windowId"] == 57
    assert elements[0]["inputValue"] == ""


def test_missing_native_constraints_do_not_become_default_numeric_limits():
    ui = _ui([_input(value="Text with $ and + signs")])
    assert focused_input_observation(ui)["constraints"] == {}
    assert ui.elements[0]["inputValue"] == "Text with $ and + signs"
    phone = _ui([_input(inputType=3)])
    assert focused_input_observation(phone)["constraints"] == {
        "native_type": 3,
        "kind": "phone",
    }


@pytest.mark.parametrize("native_type", [3, 3 | 0x80000])
@pytest.mark.parametrize(
    "expected,actual",
    [
        ("+441234567890", "+44 1234 567890"),
        ("02071234567", "(020) 7123-4567"),
        ("001234", "00 12-34"),
        ("+3906123", "+39\u00a006\u202f123"),
        ("123456", " 123 456 "),
    ],
)
def test_native_phone_display_separators_confirm_the_same_bound_number(
    native_type, expected, actual
):
    before = _ui([_input(inputType=native_type, hint="Unrelated native label")])
    target = _capture(before)
    assert target.native_type == native_type
    after = _ui([_input(value=actual, inputType=native_type, focused=False)])
    result = asyncio.run(read_input_value(SimpleNamespace(), after, target, expected))
    assert result.status == "confirmed" and result.reason == "native_phone_format"
    assert result.observed_length == len(actual)


@pytest.mark.parametrize(
    "expected,actual",
    [
        ("+44123456", "44 123456"),
        ("0044123456", "+44 123456"),
        ("0201234", "20 1234"),
        ("123456", "123 457"),
        ("1234", "1\n234"),
        ("1234", "1\t234"),
        ("1234", "１２３４"),
        ("1234", "١٢٣٤"),
        ("1234", "12 34 ext 5"),
        ("1234", "12 34#"),
        ("1234", "12 34*"),
        ("1234,5", "12 34,5"),
        ("1234;5", "12 34;5"),
    ],
)
def test_phone_equivalence_never_changes_digits_prefix_or_dial_operators(
    expected, actual
):
    target = _capture(_ui([_input(inputType=3)]))
    after = _ui([_input(value=actual, inputType=3)])
    result = asyncio.run(read_input_value(SimpleNamespace(), after, target, expected))
    assert result.status == "mismatch" and result.reason == "different_value"


def test_app_label_prefix_confirms_only_when_native_caret_matches_typed_length():
    target = _capture(_ui([_input()]))
    after = _ui(
        [_input(value="搜尋, meovv anna 壁纸", textSelectionStart=13, textSelectionEnd=13)]
    )
    result = asyncio.run(read_input_value(SimpleNamespace(), after, target, "meovv anna 壁纸"))
    assert result.status == "confirmed" and result.reason == "labelled_exact_value"


@pytest.mark.parametrize(
    "actual,selection",
    [
        ("搜尋, meovv anna 壁纸", (17, 17)),  # prefix is real content
        ("搜尋, meovv anna 壁纸", None),  # no caret evidence
        ("搜尋, meovv anna 壁纸", (0, 13)),  # selection, not a collapsed caret
        ("搜尋, meovv anna 壁", (12, 12)),  # value is not the typed text
        ("搜尋, 😀", (1, 1)),  # emoji is 2 UTF-16 units
    ],
)
def test_app_label_prefix_without_matching_caret_stays_a_mismatch(actual, selection):
    expected = "😀" if "😀" in actual else "meovv anna 壁纸"
    extra = (
        {"textSelectionStart": selection[0], "textSelectionEnd": selection[1]}
        if selection
        else {}
    )
    target = _capture(_ui([_input()]))
    after = _ui([_input(value=actual, **extra)])
    result = asyncio.run(read_input_value(SimpleNamespace(), after, target, expected))
    assert result.status == "mismatch"


@pytest.mark.parametrize("native_type", [None, 1, 2, True, -13, 0x100000003])
def test_phone_hint_or_invalid_native_type_cannot_enable_value_normalization(
    native_type,
):
    node = _input(hint="Phone", inputType=native_type)
    target = _capture(_ui([node]))
    after = _ui([{**node, "inputValue": "12 34"}])
    result = asyncio.run(read_input_value(SimpleNamespace(), after, target, "1234"))
    assert result.status == "mismatch"


@pytest.mark.parametrize(
    "native_type,reason",
    [
        (None, "native_input_type_unobservable"),
        (1, "native_input_type_changed"),
    ],
)
def test_changed_or_missing_phone_type_is_unknown_instead_of_retyping(
    native_type, reason
):
    target = _capture(_ui([_input(inputType=3)]))
    after = _ui([_input(value="12 34", inputType=native_type)])
    result = asyncio.run(read_input_value(SimpleNamespace(), after, target, "1234"))
    assert result.status == "unknown" and result.reason == reason


def test_native_formatted_phone_does_not_resolve_duplicate_field_identity():
    before = _ui([_input(resource="", inputType=3)])
    first = _raw_input(resource="", hint="Phone", windowId=57, inputType=3)
    driver = SimpleNamespace(get_ui_tree=AsyncMock(return_value=_raw(first)))
    target = _capture(before, driver)
    assert target.unique_hint and target.native_type == 3
    first.update(text="12 34", isFocused=False)
    other = {
        **first,
        "text": "",
        "boundsInScreen": {"left": 10, "top": 100, "right": 300, "bottom": 160},
    }
    driver.get_ui_tree.return_value = _raw([first, other])
    result = asyncio.run(read_input_value(driver, before, target, "1234"))
    assert result.status == "unknown" and result.reason == "field_missing_or_ambiguous"


def test_phone_equivalence_is_blocked_by_a_changed_edited_object():
    before = _ui(
        [
            {
                "className": "TextView",
                "resourceId": "object_title",
                "rawText": "Object A",
            },
            _input(inputType=3),
        ]
    )
    target = _capture(before)
    after = deepcopy(before)
    after.elements[0]["rawText"] = "Object B"
    after.elements[1]["inputValue"] = "12 34"
    result = asyncio.run(read_input_value(SimpleNamespace(), after, target, "1234"))
    assert (
        result.status == "unknown" and result.reason == "edited_object_context_changed"
    )
