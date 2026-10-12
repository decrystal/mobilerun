"""The decision policy sees affordances without changing the legacy LLM view."""

from mobilerun.agent.decision.jev import control_observations
from mobilerun.agent.decision.task_state import TaskState, corroborate_nodes
from mobilerun.tools.formatters import IndexedFormatter
from mobilerun.tools.ui.state import UIState


def _node(text="设置", **metadata):
    return {
        "className": "android.widget.TextView",
        "text": text,
        "boundsInScreen": {"left": 100, "top": 200, "right": 300, "bottom": 400},
        "children": [],
        **metadata,
    }


def test_native_scroll_capabilities_survive_without_claiming_collection_coverage():
    actions = [{"id": 4096, "name": "SCROLL_FORWARD", "label": ""}]
    raw = _node(
        text="",
        className="android.widget.ScrollView",
        isScrollable=True,
        childCount=8,
        actionList=actions,
        isEnabled=True,
    )
    _, _, elements, _ = IndexedFormatter().format(raw, {})
    ui = UIState(elements, "", "", {"packageName": "com.example"}, 400, 800)
    task = TaskState("Count entries")
    observation = task.observe(ui)
    control = control_observations(ui)["controls"][0]
    assert control["children_total"] == 8 and control["actions"] == actions
    assert task.context()["observations"][0]["nodes"][0]["childCount"] == 8
    assert task.context()["collections"] == []
    assert corroborate_nodes(observation["nodes"], ui)
    elements[0]["childCount"] = 9
    assert not corroborate_nodes(observation["nodes"], ui)
    elements[0]["childCount"] = 8
    elements[0]["actionList"] = []
    assert not corroborate_nodes(observation["nodes"], ui)


def test_container_summary_retains_duplicate_children_and_exposes_clipping_and_missing_count():
    root = {
        "index": 1,
        "className": "ScrollView",
        "bounds": "0,0,400,800",
        "scrollable": True,
        "childCount": 3,
        "children": [
            {"index": 2, "text": "Same name", "bounds": "0,0,400,300"},
            {"index": 3, "text": "Same name", "bounds": "0,600,400,900"},
        ],
    }
    ui = UIState([root], "", "", {}, 400, 800)
    container = control_observations(ui)["containers"][0]
    assert container["children_total"] == 3 and container["observed_children"] == 2
    assert container["fully_visible_children"] == 1 and container["child_indices"] == [
        2,
        3,
    ]
    assert "business coverage still requires review" in container["scope"]
    del root["childCount"]
    assert control_observations(ui)["containers"][0]["children_total"] is None


def test_boundary_evidence_reaches_compact_review_without_certifying_business_coverage():
    root = {
        "index": 1,
        "className": "ScrollView",
        "bounds": "0,0,400,900",
        "scrollable": True,
        "childCount": 4,
        "children": [
            {"index": 2, "text": "Same name", "bounds": "0,0,400,300"},
            {"index": 3, "text": "Same name", "bounds": "0,600,400,900"},
            {"index": 4, "text": "", "bounds": "0,300,400,350"},
        ],
    }
    decoration = {"index": 5, "text": "", "bounds": "0,700,100,800"}
    ui = UIState([root, decoration], "", "", {}, 400, 800)
    state = TaskState("Count entries")
    observed = state.observe(ui)
    context = state.context()
    page = context["observations"][0]
    container = page["containers"][0]
    assert container["viewport"] == [0, 0, 400, 800]
    assert (
        container["observed_children"] == 3 and container["fully_visible_children"] == 2
    )
    assert container["children_total"] == 4
    assert container["ref"] == "O1:n0"
    assert container["child_refs"] == ["O1:n1", "O1:n2", "O1:n3"]
    assert container["child_refs_omitted"] == 0
    assert (
        next(node for node in page["nodes"] if node["ref"] == "O1:n3")["bounds"]
        == "0,300,400,350"
    )
    assert (
        page["nodes_without_retained_facts"] == 1 and page["nodes_budget_omitted"] == 0
    )
    assert context["collections"] == [] and "complete" not in container
    root["children"][0]["bounds"] = "0,900,400,1200"
    assert observed["containers"][0]["fully_visible_children"] == 2
    changed = state.observe(ui)
    assert (
        changed["id"] == "O2"
        and changed["containers"][0]["fully_visible_children"] == 1
    )


def test_budget_omissions_never_advertise_unavailable_boundary_refs():
    root = {
        "index": 1,
        "className": "ScrollView",
        "scrollable": True,
        "bounds": "0,0,400,800",
        "childCount": 8,
        "children": [
            {"index": index + 2, "text": "Large value " * 80, "bounds": "0,0,400,100"}
            for index in range(8)
        ],
    }
    state = TaskState("Count entries")
    state.observe(UIState([root], "", "", {}, 400, 800))
    for budget in [0, 600, 1800]:
        page = state.context(max_chars=budget)["observations"][0]
        assert page["nodes_budget_omitted"] > 0
        refs = {node["ref"] for node in page["nodes"]}
        for container in page["containers"]:
            assert container["ref"] in refs
            assert set(container["child_refs"]) <= refs
        assert page["containers_omitted"] + len(page["containers"]) == 1
    assert len(state.observations[0]["containers"][0]["child_refs"]) == 8


def test_explicit_native_null_editor_text_is_empty_but_missing_value_stays_unknown():
    formatter = IndexedFormatter()
    raw = _node(text=None, className="android.widget.EditText", isEditable=True)
    _, _, nodes, _ = formatter.format(raw, {})
    assert nodes[0]["inputValue"] == ""
    del raw["text"]
    _, _, nodes, _ = formatter.format(raw, {})
    assert "inputValue" not in nodes[0]


def test_partial_selection_is_visible_to_policy_memory_and_live_verification():
    raw = _node(
        "Coupon code: OFF50",
        className="android.widget.EditText",
        isEditable=True,
        textSelectionStart=7,
        textSelectionEnd=11,
    )
    _, _, nodes, _ = IndexedFormatter().format(raw, {})
    ui = UIState(nodes, "", "", {}, 400, 800)
    control = control_observations(ui)["controls"][0]
    assert control["text_selection"]["selected_text"] == "code"
    state = TaskState("Copy the entire text")
    observed = state.observe(ui)
    assert state.context()["observations"][0]["nodes"][0]["textSelectionStart"] == 7
    assert corroborate_nodes(observed["nodes"], ui)
    nodes[0]["textSelectionStart"] = 0
    nodes[0]["textSelectionEnd"] = len(raw["text"])
    assert not corroborate_nodes(observed["nodes"], ui)
    assert (
        control_observations(ui)["controls"][0]["text_selection"]["selected_text"]
        == raw["text"]
    )


def test_selection_uses_native_utf16_units_and_unknown_or_password_text_is_not_inferred():
    raw = _node(
        "A🌈B",
        className="android.widget.EditText",
        isEditable=True,
        textSelectionStart=1,
        textSelectionEnd=3,
    )

    def selection():
        _, _, nodes, _ = IndexedFormatter().format(raw, {})
        return control_observations(UIState(nodes, "", "", {}, 400, 800))["controls"][
            0
        ].get("text_selection")

    assert selection() == {
        "start": 1,
        "end": 3,
        "units": "utf16",
        "selected_text": "🌈",
    }
    for start, end in [(-1, -1), (1, 2), (1, 8), (True, 3)]:
        raw.update(textSelectionStart=start, textSelectionEnd=end)
        assert selection() is None
    raw.update(textSelectionStart=1, textSelectionEnd=3, isPassword=True)
    assert selection() is None


def test_preserves_observed_affordances_and_explicit_false_values():
    raw = _node(
        isClickable=True,
        isEditable=False,
        isEnabled=False,
        isScrollable=False,
        isFocused=True,
        isPassword=False,
        isCheckable=True,
        isChecked=False,
        isSelected=False,
        isVisibleToUser=True,
    )

    _, _, elements, _ = IndexedFormatter().format(raw, {})
    element = elements[0]

    expected = {
        "clickable": True,
        "editable": False,
        "enabled": False,
        "scrollable": False,
        "focused": True,
        "password": False,
        "checkable": True,
        "checked": False,
        "selected": False,
        "visible": True,
    }
    for key, value in expected.items():
        assert element[key] is value
    for key, value in raw.items():
        if key.startswith("is"):
            assert element[key] is value
    assert element["checkedState"] == "isChecked=False"


def test_unknown_affordances_stay_absent_and_canonical_metadata_survives():
    formatter = IndexedFormatter()
    _, _, unknown, _ = formatter.format(_node(), {})
    assert "enabled" not in unknown[0]
    assert "clickable" not in unknown[0]
    assert "password" not in unknown[0]

    _, _, observed, _ = formatter.format(
        _node(clickable=False, editable=True, enabled=None, visible=False), {}
    )
    assert observed[0]["clickable"] is False
    assert observed[0]["editable"] is True
    assert observed[0]["enabled"] is None
    assert observed[0]["visible"] is False


def test_empty_input_value_does_not_become_its_accessible_label():
    formatter = IndexedFormatter()
    text, _, elements, _ = formatter.format(
        _node("", className="android.widget.EditText", contentDescription="搜索设置"),
        {},
    )

    assert elements[0]["inputValue"] == ""
    assert elements[0]["text"] == "搜索设置"
    assert '"搜索设置"' in text
    assert "inputValue" not in text

    _, _, populated, _ = formatter.format(_node("蓝牙", isEditable=True), {})
    assert populated[0]["inputValue"] == "蓝牙"
    _, _, label_only, _ = formatter.format(_node("搜索设置"), {})
    assert "inputValue" not in label_only[0]


def test_android_hint_text_is_not_a_typed_value():
    _, _, elements, _ = IndexedFormatter().format(
        _node("搜索或输入网址", isEditable=True, isShowingHintText=True), {}
    )
    assert elements[0]["inputValue"] == ""
    assert elements[0]["text"] == "搜索或输入网址"


def test_flat_parent_links_preserve_indices_and_legacy_formatted_text():
    root = _node(
        children=[
            _node("显示", isClickable=True, children=[_node("亮度")]),
            _node("搜索", isEditable=True),
        ]
    )
    phone_state = {"currentApp": "设置", "packageName": "com.android.settings"}

    text, _, elements, _ = IndexedFormatter().format(root, phone_state)

    assert [(e["index"], e["parentIndex"]) for e in elements] == [
        (1, None),
        (2, 1),
        (3, 2),
        (4, 1),
    ]
    assert all(e["children"] == [] for e in elements)
    assert text == (
        "**Current Phone State:**\n"
        "• **App:** 设置 (com.android.settings)\n"
        "• **Keyboard:** Hidden\n"
        "• **Focused Element:** ''\n\n"
        "Current Clickable UI elements:\n"
        "'index. className: resourceId; checkedState, text - bounds(x1,y1,x2,y2)':\n"
        '1. TextView: "设置" - (100,200,300,400)\n'
        '2. TextView: "显示" - (100,200,300,400)\n'
        '3. TextView: "亮度" - (100,200,300,400)\n'
        '4. TextView: "搜索" - (100,200,300,400)'
    )


def test_metadata_keeps_index_taps_in_native_space_when_text_is_scaled():
    formatter = IndexedFormatter()
    formatter.display_scale_x = 2.0
    formatter.display_scale_y = 2.0
    text, focused, elements, phone_state = formatter.format(
        _node(isClickable=True, isEnabled=True), {}
    )
    state = UIState(
        elements,
        text,
        focused,
        phone_state,
        screen_width=1000,
        screen_height=2000,
        coordinate_scale_x=2.0,
        coordinate_scale_y=2.0,
    )

    assert elements[0]["displayBounds"] == "50,100,150,200"
    assert elements[0]["bounds"] == "100,200,300,400"
    assert state.get_element_coords(1) == (200, 300)
    assert state.convert_point(100, 150) == (200, 300)
    assert "(50,100,150,200)" in text
    assert "clickable" not in text
