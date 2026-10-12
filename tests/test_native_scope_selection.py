"""Collection coverage belongs to an observed native selection context."""

import copy
from dataclasses import asdict

import pytest
from mobilerun.agent.decision.native_query_binding import (
    bind_native_queries,
    corroborate_native_queries,
)
from mobilerun.agent.decision.query import CollectionLedger, QueryError
from mobilerun.agent.decision.task_state import TaskState
from test_collection_query import page
from test_native_query_binding import fresh
from test_task_evidence import task, ui

STATES = [
    ("selected", True, False),
    ("isSelected", True, False),
    ("checked", True, False),
    ("isChecked", True, False),
    ("checkedState", "checked", "unchecked"),
    ("stateDescription", "Enabled filter", "Disabled filter"),
    (
        "rangeInfo",
        {"current": 1, "min": 0, "max": 2},
        {"current": 2, "min": 0, "max": 2},
    ),
]


def selected_page(
    oid,
    rows,
    *,
    field="selected",
    value=True,
    label="All records",
    icon_only=False,
    resource="records",
):
    observed = page(oid, rows, total=2)
    observed["nodes"][0]["resourceId"] = resource
    path = len(observed["nodes"])
    selector = {
        "path": path,
        "ref": oid + ":n" + str(path),
        "index": path + 1,
        "resourceId": "selector",
        "className": "Button",
        field: copy.deepcopy(value),
    }
    if not icon_only:
        selector["rawText"] = label
    observed["nodes"].append(selector)
    return observed


@pytest.mark.parametrize("field,before,after", STATES)
def test_selection_state_change_cannot_merge_disjoint_pages_with_identical_labels(
    field, before, after
):
    ledger = CollectionLedger()
    first = ledger.observe(
        selected_page("O1", [(0, "Same name")], field=field, value=before)
    )[0]
    second = ledger.observe(
        selected_page("O2", [(1, "Same name")], field=field, value=after)
    )[0]
    assert first != second
    for key in (first, second):
        assert not ledger.scopes[key].complete
        with pytest.raises(QueryError, match="coverage_incomplete"):
            ledger.aggregate(key, "count")


@pytest.mark.parametrize("field", ["selected", "isSelected", "checked"])
def test_icon_only_native_selector_state_also_partitions_coverage(field):
    ledger = CollectionLedger()
    first = ledger.observe(
        selected_page("O1", [(0, "A")], field=field, icon_only=True)
    )[0]
    second = ledger.observe(
        selected_page("O2", [(1, "B")], field=field, value=False, icon_only=True)
    )[0]
    assert first != second and not ledger.scopes[second].complete


@pytest.mark.parametrize("using_label", [False, True])
def test_return_to_prior_filter_requires_new_coverage_after_an_observed_switch(
    using_label,
):
    ledger = CollectionLedger()
    first = selected_page("O1", [(0, "A"), (1, "B")])
    key = ledger.observe(first)[0]
    assert ledger.aggregate(key, "count")["value"] == 2
    changed = selected_page(
        "O2",
        [(0, "Other")],
        value=False,
        label="Other records" if using_label else "All records",
    )
    ledger.observe(changed)
    returned = ledger.observe(selected_page("O3", [(0, "A")]))[0]
    assert returned == key
    assert ledger.scopes[key].generation == 2
    assert set(ledger.scopes[key].rows) == {0}
    with pytest.raises(QueryError, match="coverage_incomplete"):
        ledger.aggregate(key, "count")
    diagnostic = ledger.take_diagnostics()[-1]
    assert diagnostic["scope_context_changed"] is True
    assert diagnostic["discarded_coverage"] == [0, 1]


def test_normal_scroll_retains_coverage_and_ignores_selector_geometry_or_focus():
    ledger = CollectionLedger()
    key = ledger.observe(selected_page("O1", [(0, "Same name")]))[0]
    next_page = selected_page("O2", [(1, "Same name")])
    next_page["nodes"][-1].update(
        bounds=[12, 24, 36, 48], focused=True, textSelectionStart=1
    )
    assert ledger.observe(next_page)[0] == key
    assert ledger.scopes[key].generation == 1
    assert ledger.aggregate(key, "count")["value"] == 2


def test_detail_navigation_without_another_observed_scope_keeps_the_list_evidence():
    ledger = CollectionLedger()
    key = ledger.observe(selected_page("O1", [(0, "A"), (1, "B")]))[0]
    detail = {
        "id": "O2",
        "phone": {"packageName": "com.example.list"},
        "nodes": [{"path": 0, "ref": "O2:n0", "rawText": "Record details"}],
    }
    assert ledger.observe(detail) == []
    assert ledger.observe(selected_page("O3", [(0, "A")]))[0] == key
    assert ledger.scopes[key].generation == 1
    assert ledger.aggregate(key, "count")["value"] == 2


def test_a_different_native_container_does_not_reset_the_original_container():
    ledger = CollectionLedger()
    key = ledger.observe(selected_page("O1", [(0, "A"), (1, "B")]))[0]
    ledger.observe(
        selected_page("O2", [(0, "Other")], value=False, resource="other_list")
    )
    ledger.observe(selected_page("O3", [(0, "A")]))
    assert ledger.scopes[key].generation == 1 and ledger.scopes[key].complete


@pytest.mark.parametrize("field,before,after", STATES)
def test_fresh_query_guard_rejects_changed_selection_without_mutating_source_ledger(
    field, before, after
):
    goal = "Count the records in the selected view"
    state = TaskState(goal)
    state.set_contract(task(goal))
    screen = ui(
        selected_page("O1", [(0, "A"), (1, "B")], field=field, value=before)["nodes"]
    )
    state.observe(screen)
    key = next(iter(state.collections.scopes))
    state.query({"scope": key, "generation": 1, "operation": "count"})
    binding = bind_native_queries(["Q1"], state, state.current_id, goal=goal)
    captured = copy.deepcopy(asdict(state.collections.scopes[key]))
    changed = copy.deepcopy(screen)
    changed.elements[-1][field] = copy.deepcopy(after)
    assert not corroborate_native_queries(
        binding, state, state.current_id, fresh(changed), goal=goal
    )
    assert asdict(state.collections.scopes[key]) == captured


def test_old_query_result_is_stale_after_switching_away_and_returning_to_its_scope():
    goal = "Count the records in the selected view"
    state = TaskState(goal)
    state.set_contract(task(goal))
    state.observe(ui(selected_page("O1", [(0, "A"), (1, "B")])["nodes"]))
    key = next(iter(state.collections.scopes))
    state.query({"scope": key, "generation": 1, "operation": "count"})
    state.observe(
        ui(
            selected_page("O2", [(0, "Other")], label="Other records", value=False)[
                "nodes"
            ]
        )
    )
    state.observe(ui(selected_page("O3", [(0, "A")])["nodes"]))
    assert state.query_results[0]["value"] == 2
    assert "Q1" not in state.evidence()
    with pytest.raises(QueryError, match="query_scope_stale"):
        state.query({"scope": key, "generation": 1, "operation": "count"})


def test_historical_query_remains_available_while_inspecting_a_detail_page():
    goal = "Count the records in the selected view"
    state = TaskState(goal)
    state.set_contract(task(goal))
    state.observe(ui(selected_page("O1", [(0, "A"), (1, "B")])["nodes"]))
    key = next(iter(state.collections.scopes))
    state.query({"scope": key, "generation": 1, "operation": "count"})
    state.observe(ui([{"index": 1, "rawText": "Record details"}]))
    assert "Q1" in state.evidence()
    binding = bind_native_queries(["Q1"], state, state.current_id, goal=goal)
    assert binding["queries"][0]["current"] is False


@pytest.mark.parametrize("field,before,_after", STATES)
def test_model_context_contains_actual_native_scope_selector_values_and_refs(
    field, before, _after
):
    ledger = CollectionLedger()
    ledger.observe(selected_page("O1", [(0, "A")], field=field, value=before))
    context = ledger.context()[0]
    selector = context["scope_context"][0]
    assert selector["ref"] == "O1:n3" and selector[field] == before
    assert (
        selector["resourceId"] == "selector" and context["scope_context_omitted"] == 0
    )


def test_scope_context_uses_existing_character_budget_and_records_omissions():
    ledger = CollectionLedger()
    ledger.observe(selected_page("O1", [(0, "A")], label="Long label " * 100))
    context = ledger.context(max_chars=1)[0]
    assert context["scope_context"] == [] and context["scope_context_omitted"] == 1
    assert context["rows"] == [] and context["rows_omitted"] == 1
