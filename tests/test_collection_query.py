from decimal import Decimal

import pytest

from mobilerun.agent.decision.query import (
    CollectionLedger,
    QueryError,
    observed_date,
    quantity,
    week_interval,
)


def page(oid, rows, total=3):
    nodes = [
        {
            "path": 0,
            "ref": f"{oid}:n0",
            "resourceId": "records",
            "className": "RecyclerView",
            "collectionInfo": {"rowCount": total, "columnCount": 1},
        }
    ]
    for index, name in rows:
        path = len(nodes)
        nodes.extend(
            [
                {
                    "path": path,
                    "parent_path": 0,
                    "ref": f"{oid}:n{path}",
                    "collectionItemInfo": {"rowIndex": index, "rowSpan": 1},
                },
                {
                    "path": path + 1,
                    "parent_path": path,
                    "ref": f"{oid}:n{path + 1}",
                    "rawText": name,
                },
            ]
        )
    for node in nodes:
        node["index"] = node["path"] + 1
        node["parentIndex"] = node["parent_path"] + 1 if "parent_path" in node else None
    return {"id": oid, "phone": {"packageName": "com.example.list"}, "nodes": nodes}


def test_cross_page_coverage_and_dedup_use_positions_not_display_names():
    ledger = CollectionLedger()
    key = ledger.observe(page("O1", [(0, "Same name"), (1, "Same name")]))[0]
    with pytest.raises(QueryError, match="coverage_incomplete"):
        ledger.aggregate(key, "count")
    ledger.observe(page("O2", [(1, "Same name"), (2, "Last item")]))
    assert ledger.aggregate(key, "count")["value"] == 3
    assert ledger.context()[0]["complete"]


def test_changed_collection_invalidates_old_coverage_even_when_size_is_unchanged():
    ledger = CollectionLedger()
    key = ledger.observe(page("O1", [(0, "A"), (1, "B"), (2, "C")]))[0]
    assert ledger.aggregate(key, "count")["value"] == 3
    ledger.observe(page("O2", [(0, "Different")]))
    assert ledger.context()[0]["generation"] == 2
    with pytest.raises(QueryError, match="coverage_incomplete"):
        ledger.aggregate(key, "count")


def test_zero_requires_an_observed_empty_collection_and_unsupported_metadata_is_unknown():
    ledger = CollectionLedger()
    key = ledger.observe(page("O1", [], total=0))[0]
    assert ledger.aggregate(key, "count")["value"] == 0
    ambiguous = page("O2", [])
    del ambiguous["nodes"][0]["resourceId"]
    assert ledger.observe(ambiguous) == []


def test_fact_source_must_belong_to_the_identified_row():
    ledger = CollectionLedger()
    key = ledger.observe(page("O1", [(0, "2 km"), (1, "3 km")], total=2))[0]
    ledger.add_native_fact(key, 0, "distance", "O1:n2", "rawText")
    with pytest.raises(QueryError, match="fact_not_observed_in_row"):
        ledger.add_native_fact(key, 1, "distance", "O1:n2", "rawText")
    ledger.add_native_fact(key, 1, "distance", "O1:n4", "rawText")
    assert ledger.aggregate(key, "sum", field_name="distance", output_unit="m")[
        "value"
    ] == Decimal(5000)


def test_unknown_filter_properties_do_not_silently_become_nonmatches():
    ledger = CollectionLedger()
    key = ledger.observe(page("O1", [(0, "A")], total=1))[0]
    with pytest.raises(QueryError, match="filter_property_unknown"):
        ledger.aggregate(
            key,
            "count",
            filters=[{"field": "category", "operator": "eq", "value": "cycling"}],
        )


@pytest.mark.parametrize(
    "raw,unit,value",
    [
        ("2:20:00", "minutes", "140"),
        ("1.5 h", "min", "90"),
        ("1 mi", "m", "1609.344"),
        ("3 km", "m", "3000"),
    ],
)
def test_unit_conversion_is_deterministic(raw, unit, value):
    assert quantity(raw, unit) == Decimal(value)


@pytest.mark.parametrize(
    "raw,unit",
    [
        ("1,200 m", "m"),
        (True, None),
        ("1 kg", "m"),
        ("1:99:00", "min"),
        ("2 km maybe", "m"),
    ],
)
def test_ambiguous_numbers_and_incompatible_units_are_not_guessed(raw, unit):
    with pytest.raises(QueryError):
        quantity(raw, unit)


def test_week_boundaries_use_the_observed_device_clock_not_host_time():
    start, end = week_interval("2024-12-31T23:00:00+08:00", next_week=True)
    assert start.isoformat() == "2025-01-06T00:00:00+08:00"
    assert end.isoformat() == "2025-01-13T00:00:00+08:00"
    with pytest.raises(QueryError, match="device_timezone_unknown"):
        week_interval("2024-12-31T23:00:00")


def test_runtime_query_binds_actual_values_and_rejects_stale_generation():
    ledger = CollectionLedger()
    observed = page("O1", [(0, "2 km"), (1, "3 km")], total=2)
    for node in observed["nodes"]:
        if "rawText" in node:
            node["resourceId"] = "distance"
    key = ledger.observe(observed)[0]
    plan = {
        "scope": key,
        "generation": 1,
        "operation": "sum",
        "field": "distance",
        "output_unit": "m",
        "selectors": [
            {"name": "distance", "resource_id": "distance", "source_field": "rawText"}
        ],
    }
    assert ledger.execute(plan)["value"] == "5000"
    ledger.observe(page("O2", [(0, "Changed")], total=2))
    with pytest.raises(QueryError, match="query_scope_stale"):
        ledger.execute(plan)


def test_query_plan_cannot_insert_unobserved_value_and_rejects_ambiguous_dates():
    ledger = CollectionLedger()
    key = ledger.observe(page("O1", [(0, "2024-12-31")], total=1))[0]
    plan = {
        "scope": key,
        "generation": 1,
        "operation": "count",
        "facts": [
            {
                "row": 0,
                "name": "date",
                "ref": "O1:n2",
                "source_field": "rawText",
                "value": "2024-12-31",
            }
        ],
    }
    with pytest.raises(QueryError, match="query_fact_invalid"):
        ledger.execute(plan)
    del plan["facts"][0]["value"]
    plan["filters"] = [{"field": "date", "operator": "relative_week", "week_start": 0}]
    assert (
        ledger.execute(plan, device_datetime="2024-12-31T23:00:00+08:00")["value"] == 1
    )
    plan["filters"][0]["next_week"] = True
    assert (
        ledger.execute(plan, device_datetime="2024-12-31T23:00:00+08:00")["value"] == 0
    )
    with pytest.raises(QueryError, match="device_time_unknown"):
        ledger.execute(plan)


def test_reused_container_on_another_search_does_not_merge_coverage():
    ledger = CollectionLedger()
    first = page("O1", [(0, "Same")], total=2)
    first["nodes"].append({"path": 3, "ref": "O1:n3", "inputValue": "alpha"})
    second = page("O2", [(1, "Other")], total=2)
    second["nodes"].append({"path": 3, "ref": "O2:n3", "inputValue": "beta"})
    key = ledger.observe(first)[0]
    assert ledger.observe(second)[0] != key
    with pytest.raises(QueryError, match="coverage_incomplete"):
        ledger.aggregate(key, "count")


def test_actual_runtime_tool_and_reviewer_share_cross_page_calculated_evidence():
    import asyncio
    import json
    from types import SimpleNamespace

    from test_task_evidence import review, task, ui

    from mobilerun.agent.decision.contracts import parse_grounded_review
    from mobilerun.agent.decision.runtime import DecisionRuntime
    from mobilerun.agent.decision.task_state import TaskState
    from mobilerun.agent.tool_registry import ToolRegistry

    state = TaskState("Count all records")
    state.set_contract(task("Count all records"))
    for oid, rows in [
        ("O1", [(0, "Same"), (1, "Same")]),
        ("O2", [(1, "Same"), (2, "Last")]),
    ]:
        state.observe(ui(page(oid, rows)["nodes"]))
    key = next(iter(state.collections.scopes))
    registry = ToolRegistry()
    # Register the actual execution function, then pass through the same
    # ToolRegistry dispatch used for device actions. No private DB or task IDs.
    runner = object.__new__(DecisionRuntime)
    runner.task_state, runner.trace = state, []
    registry.register("query_collection", runner.query_collection, {}, "query")
    result = asyncio.run(
        registry.execute(
            "query_collection",
            {"plan": {"scope": key, "generation": 1, "operation": "count"}},
            SimpleNamespace(),
        )
    )
    assert result.success and json.loads(result.summary)["value"] == 3
    assessment = parse_grounded_review(
        review(["Q1"], answer="3"), state.contract, state.evidence(), state.current_id
    )
    assert assessment["status"] == "done"
    state.observe(ui(page("O3", [(0, "Changed")])["nodes"]))
    assert "Q1" not in state.evidence()


def test_relative_day_dates_use_device_timezone_across_year_boundary():
    ledger = CollectionLedger()
    key = ledger.observe(page("O1", [(0, "Today"), (1, "Yesterday"), (2, "Tomorrow")]))[
        0
    ]
    plan = {
        "scope": key,
        "generation": 1,
        "operation": "count",
        "facts": [
            {
                "row": row,
                "name": "date",
                "ref": f"O1:n{2 * row + 2}",
                "source_field": "rawText",
            }
            for row in range(3)
        ],
        "filters": [{"field": "date", "operator": "relative_week", "week_start": 0}],
    }
    result = ledger.execute(plan, device_datetime="2024-12-31T23:00:00+08:00")
    assert result["value"] == 3
    assert result["filters"][0]["start"] == "2024-12-30T00:00:00+08:00"
    assert observed_date(
        "明天", device_datetime="2024-12-31T23:00:00+08:00"
    ).isoformat() == ("2025-01-01T00:00:00+08:00")


@pytest.mark.parametrize(
    "label", ["Friday", "Oct 13", "10/13", "next week", "last Tuesday"]
)
def test_calendar_labels_without_a_unique_day_are_not_guessed(label):
    with pytest.raises(QueryError, match="date_ambiguous"):
        observed_date(label, device_datetime="2023-10-15T15:34:47+00:00")


@pytest.mark.parametrize(
    "clock,code",
    [(None, "device_time_unknown"), ("2024-01-01", "device_timezone_unknown")],
)
def test_relative_day_without_an_observed_zoned_clock_is_rejected(clock, code):
    with pytest.raises(QueryError, match=code):
        observed_date("Today", device_datetime=clock)


def test_invalid_span_reports_actionable_feedback_without_mutating_evidence():
    import json

    from test_task_evidence import ui

    from mobilerun.agent.decision.runtime import DecisionRuntime
    from mobilerun.agent.decision.task_state import TaskState

    state = TaskState("Count records")
    state.observe(ui(page("O1", [(0, "12 km")], total=1)["nodes"]))
    key = next(iter(state.collections.scopes))
    runner = object.__new__(DecisionRuntime)
    runner.task_state, runner.trace = state, []
    before = json.dumps(state.evidence(), sort_keys=True)
    result = runner.query_collection(
        {
            "scope": key,
            "generation": 1,
            "operation": "count",
            "facts": [
                {
                    "row": 0,
                    "name": "distance",
                    "ref": "O1:n2",
                    "source_field": "rawText",
                    "span": [100, 200, 300, 400],
                }
            ],
        },
        ctx=None,
    )
    assert not result.success
    assert (
        "omit span" in result.summary and "never screen coordinates" in result.summary
    )
    assert json.dumps(state.evidence(), sort_keys=True) == before
    assert runner.trace[-1]["device_action_executed"] is False
