"""Exact invalidation evidence without changing model-visible query semantics."""

import copy
from types import SimpleNamespace

import pytest
from test_collection_query import page

from mobilerun.agent.decision.query import CollectionLedger, QueryError
from mobilerun.agent.decision.runtime import DecisionRuntime


def test_invalidation_saves_full_changed_row_and_container_then_drains_once():
    ledger = CollectionLedger()
    first = page("O1", [(0, "A"), (1, "B")], total=2)
    first["nodes"][0]["bounds"] = [0, 100, 100, 500]
    first["nodes"][1]["childCount"] = 1
    first["nodes"][1]["bounds"] = [0, 100, 100, 200]
    key = ledger.observe(first)[0]
    second = page("O2", [(1, "Changed")], total=2)
    second.update(screen_size=[100, 600], coordinate_space="native_pixels")
    second["nodes"][0]["bounds"] = [0, 100, 100, 500]
    second["nodes"][1]["childCount"] = 1
    second["nodes"][1]["bounds"] = [0, 100, 100, 110]
    ledger.observe(second)
    # Diagnostic snapshots must survive later mutation of the passed UI.
    second["nodes"][2]["rawText"] = "Later mutation"
    records = ledger.take_diagnostics()
    assert len(records) == 1
    item = records[0]
    assert item["scope"] == key
    assert item["from_generation"] == 1 and item["to_generation"] == 2
    assert item["changed_overlap_rows"] == [1]
    assert item["discarded_coverage"] == [0, 1]
    assert item["old_container"]["ref"] == "O1:n0"
    assert item["new_container"]["ref"] == "O2:n0"
    assert item["screen_size"] == [100, 600]
    row = item["changed_rows"][0]
    assert row["before"]["nodes"][1]["rawText"] == "B"
    assert row["after"]["nodes"][1]["rawText"] == "Changed"
    assert row["after"]["nodes"][0]["childCount"] == 1
    assert row["after"]["nodes"][0]["bounds"] == [0, 100, 100, 110]
    assert row["before"]["signature"] != row["after"]["signature"]
    assert ledger.take_diagnostics() == []
    with pytest.raises(QueryError, match="coverage_incomplete"):
        ledger.aggregate(key, "count")


def test_disappearing_descendants_still_invalidate_and_are_not_assumed_clipped():
    ledger = CollectionLedger()
    key = ledger.observe(page("O1", [(0, "A"), (1, "B")], total=2))[0]
    second = page("O2", [(1, "B")], total=2)
    second["nodes"] = second["nodes"][:2]
    ledger.observe(second)
    item = ledger.take_diagnostics()[0]
    assert len(item["changed_rows"][0]["before"]["nodes"]) == 2
    assert len(item["changed_rows"][0]["after"]["nodes"]) == 1
    assert ledger.scopes[key].generation == 2
    assert not ledger.scopes[key].complete
    assert "cause" not in item


def test_geometry_changes_do_not_change_signature_or_invalidate_rows():
    ledger = CollectionLedger()
    first = page("O1", [(0, "A")], total=1)
    ledger.observe(first)
    second = page("O2", [(0, "A")], total=1)
    second["nodes"][1]["bounds"] = [0, 20, 100, 50]
    ledger.observe(second)
    assert ledger.take_diagnostics() == []
    assert ledger.context()[0]["generation"] == 1


def test_count_change_is_recorded_even_with_no_overlapping_rows():
    ledger = CollectionLedger()
    ledger.observe(page("O1", [(0, "A")], total=1))
    ledger.observe(page("O2", [], total=0))
    item = ledger.take_diagnostics()[0]
    assert item["expected_count_changed"]
    assert item["old_expected_rows"] == 1 and item["new_expected_rows"] == 0
    assert item["changed_rows"] == []
    assert ledger.context()[0]["complete"]


def test_evidence_and_queue_bounds_record_omissions_instead_of_complete_claim():
    ledger = CollectionLedger()
    ledger.observe(page("O1", [(i, "A") for i in range(70)], total=70))
    ledger.observe(page("O2", [(i, "B") for i in range(70)], total=70))
    item = ledger.take_diagnostics()[0]
    assert len(item["changed_overlap_rows"]) == 64
    assert item["changed_overlap_rows_omitted"] == 6
    assert len(item["changed_rows"]) == 8
    assert item["changed_row_evidence_omitted"] == 62
    for i in range(10):
        ledger.observe(page(f"O{i + 3}", [(0, str(i))], total=70))
    items = ledger.take_diagnostics()
    assert len(items) == 8
    assert items[0]["earlier_invalidations_omitted"] == 2
    assert ledger.take_diagnostics() == []


def test_long_row_diagnostic_marks_native_node_omission():
    ledger = CollectionLedger()
    first = page("O1", [(0, "A")], total=1)
    for i in range(2, 80):
        first["nodes"].append(
            {"path": i + 1, "parent_path": 1, "ref": f"O1:n{i + 1}", "rawText": str(i)}
        )
    ledger.observe(first)
    second = copy.deepcopy(first)
    second["id"] = "O2"
    second["nodes"][2]["rawText"] = "Changed"
    ledger.observe(second)
    row = ledger.take_diagnostics()[0]["changed_rows"][0]["before"]
    assert len(row["nodes"]) == 64
    assert row["nodes_omitted"] == 16


def test_diagnostic_drain_never_changes_model_context_or_computed_query():
    ledger = CollectionLedger()
    key = ledger.observe(page("O1", [(0, "A")], total=1))[0]
    ledger.observe(page("O2", [(0, "B")], total=1))
    before_context = copy.deepcopy(ledger.context())
    before_query = ledger.aggregate(key, "count")
    ledger.take_diagnostics()
    assert ledger.context() == before_context
    assert ledger.aggregate(key, "count") == before_query
    assert "diagnostics" not in before_context[0]


def test_runtime_records_diagnostic_once_without_model_or_device_calls():
    ledger = CollectionLedger()
    ledger.observe(page("O1", [(0, "A")], total=1))
    ledger.observe(page("O2", [(0, "B")], total=1))
    runtime = object.__new__(DecisionRuntime)
    runtime.task_state = SimpleNamespace(collections=ledger)
    runtime.trace = []
    runtime.record_collection_diagnostics()
    runtime.record_collection_diagnostics()
    assert len(runtime.trace) == 1
    assert runtime.trace[0]["event"] == "collection_invalidation"
    assert runtime.trace[0]["changed_rows"][0]["after"]["nodes"][1]["rawText"] == "B"
