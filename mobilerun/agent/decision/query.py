"""Cross-page collection coverage and deterministic arithmetic over observed facts.

No task IDs, application names, private databases, or expected answers are used.
Absolute accessibility row positions disambiguate identical display names.
"""

from __future__ import annotations

import copy
import hashlib
import json
import re
from collections import deque
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from decimal import Decimal, InvalidOperation
from typing import Any


class QueryError(ValueError):
    pass


_SCOPE_STATE_FIELDS = (
    "selected",
    "isSelected",
    "checkedState",
    "checked",
    "isChecked",
    "stateDescription",
    "rangeInfo",
)


@dataclass
class CollectionScope:
    key: str
    expected_count: int
    rows: dict[int, dict] = field(default_factory=dict)
    generation: int = 1
    invalidations: int = 0
    container_ref: str = ""
    container_node: dict = field(default_factory=dict)
    native_context: list[dict] = field(default_factory=list)

    @property
    def complete(self):
        return set(self.rows) == set(range(self.expected_count))


class CollectionLedger:
    def __init__(self):
        self.scopes: dict[str, CollectionScope] = {}
        # Only an actually observed different context of this same container
        # starts a new coverage visit. Opening a detail page is not such proof.
        self._active_scopes: dict[str, str] = {}
        # Diagnostic side channel only: never enters model context/evidence.
        self._diagnostics: deque[dict] = deque(maxlen=8)
        self._diagnostics_dropped = 0

    def take_diagnostics(self) -> list[dict]:
        records = copy.deepcopy(list(self._diagnostics))
        if records:
            records[0]["earlier_invalidations_omitted"] = self._diagnostics_dropped
        self._diagnostics.clear()
        self._diagnostics_dropped = 0
        return records

    @staticmethod
    def _diagnostic_row(row: dict) -> dict:
        return {
            "row_ref": row["row_ref"],
            "observation": row["observation"],
            "signature": row["signature"],
            "nodes": copy.deepcopy(row["nodes"][:64]),
            "nodes_omitted": max(0, len(row["nodes"]) - 64),
            "heading": row["heading"],
        }

    def observe(self, observation: dict) -> list[str]:
        nodes = observation["nodes"]
        paths = {node["path"]: node for node in nodes}
        indices = {
            node.get("index"): node["path"]
            for node in nodes
            if type(node.get("index")) is int
        }

        def parent(node):
            return (
                node.get("parent_path")
                if node.get("parent_path") is not None
                else indices.get(node.get("parentIndex"))
            )

        def inside(node, root):
            visited = set()
            path = node["path"]
            while path in paths and path not in visited:
                if path == root["path"]:
                    return True
                visited.add(path)
                path = parent(paths[path])
            return False

        updated = []
        for root in nodes:
            info = root.get("collectionInfo")
            if not isinstance(info, dict) or info.get("columnCount") != 1:
                continue
            count = info.get("rowCount")
            if type(count) is not int or not 0 <= count <= 100_000:
                continue
            package = observation.get("phone", {}).get("packageName")
            resource = root.get("resourceId")
            if not package or not resource:
                continue  # No stable container identity: inspect further rather than invent one.
            container_key = package + ":" + resource
            key = container_key
            # A reused list resource on another filter/search/detail page must
            # not silently inherit a previous page's coverage.
            context_nodes = [
                node
                for node in nodes
                if not inside(node, root)
                and (
                    node.get("rawText")
                    or node.get("contentDescription")
                    or node.get("inputValue")
                    or any(name in node for name in _SCOPE_STATE_FIELDS)
                )
            ]
            outside = [
                {
                    name: node[name]
                    for name in (
                        "rawText",
                        "contentDescription",
                        "inputValue",
                        "resourceId",
                        "className",
                        *_SCOPE_STATE_FIELDS,
                    )
                    if name in node
                }
                for node in context_nodes
            ]
            if outside:
                key += (
                    ":"
                    + hashlib.sha256(
                        json.dumps(outside, sort_keys=True).encode()
                    ).hexdigest()[:12]
                )
            if (
                sum(
                    node.get("resourceId") == resource
                    and isinstance(node.get("collectionInfo"), dict)
                    for node in nodes
                )
                > 1
            ):
                continue
            rows = {}
            for row in nodes:
                item = row.get("collectionItemInfo")
                if not isinstance(item, dict) or not inside(row, root):
                    continue
                index = item.get("rowIndex")
                if (
                    type(index) is not int
                    or not 0 <= index < count
                    or item.get("rowSpan", 1) != 1
                ):
                    continue
                descendants = [
                    copy.deepcopy(node) for node in nodes if inside(node, row)
                ]
                semantic = [
                    {
                        key: value
                        for key, value in node.items()
                        if key
                        in {
                            "rawText",
                            "text",
                            "contentDescription",
                            "checkedState",
                            "checked",
                            "isChecked",
                            "stateDescription",
                            "accessibility_type_hint",
                        }
                    }
                    for node in descendants
                ]
                signature = hashlib.sha256(
                    json.dumps(semantic, sort_keys=True, ensure_ascii=False).encode()
                ).hexdigest()
                if index in rows:
                    raise QueryError("ambiguous_row_position")
                rows[index] = {
                    "row_ref": row["ref"],
                    "observation": observation["id"],
                    "nodes": descendants,
                    "signature": signature,
                    "heading": item.get("isHeading") is True,
                    "facts": {},
                }
            scope = self.scopes.get(key)
            previous_active = self._active_scopes.get(container_key)
            context_changed = previous_active is not None and previous_active != key
            changed_positions = (
                [
                    index
                    for index, row in rows.items()
                    if index in scope.rows
                    and scope.rows[index]["signature"] != row["signature"]
                ]
                if scope is not None
                else []
            )
            changed = scope is not None and (
                scope.expected_count != count
                or bool(changed_positions)
                or context_changed
            )
            if changed:
                if len(self._diagnostics) == self._diagnostics.maxlen:
                    self._diagnostics_dropped += 1
                self._diagnostics.append(
                    {
                        "scope": key,
                        "observation": observation["id"],
                        "screen_size": copy.deepcopy(observation.get("screen_size")),
                        "coordinate_space": observation.get("coordinate_space"),
                        "from_generation": scope.generation,
                        "to_generation": scope.generation + 1,
                        "old_expected_rows": scope.expected_count,
                        "new_expected_rows": count,
                        "expected_count_changed": scope.expected_count != count,
                        "scope_context_changed": context_changed,
                        "previous_active_scope": previous_active,
                        "changed_overlap_rows": changed_positions[:64],
                        "changed_overlap_rows_omitted": max(
                            0, len(changed_positions) - 64
                        ),
                        "discarded_coverage": sorted(scope.rows)[:256],
                        "discarded_coverage_omitted": max(0, len(scope.rows) - 256),
                        "old_container": copy.deepcopy(scope.container_node),
                        "new_container": copy.deepcopy(root),
                        "changed_rows": [
                            {
                                "row_index": index,
                                "before": self._diagnostic_row(scope.rows[index]),
                                "after": self._diagnostic_row(rows[index]),
                            }
                            for index in changed_positions[:8]
                        ],
                        "changed_row_evidence_omitted": max(
                            0, len(changed_positions) - 8
                        ),
                    }
                )
            if scope is None or changed:
                previous_generation = scope.generation if scope else 0
                invalidations = scope.invalidations + 1 if scope else 0
                scope = CollectionScope(
                    key,
                    count,
                    generation=previous_generation + 1,
                    invalidations=invalidations,
                )
                self.scopes[key] = scope
            for index, row in rows.items():
                if index in scope.rows:
                    row["facts"] = scope.rows[index]["facts"]
                scope.rows[index] = row
            scope.container_ref = root["ref"]
            scope.container_node = copy.deepcopy(root)
            scope.native_context = copy.deepcopy(
                [
                    {**item, "ref": node.get("ref")}
                    for item, node in zip(outside, context_nodes, strict=True)
                ]
            )
            self._active_scopes[container_key] = key
            updated.append(key)
        return updated

    def add_native_fact(
        self,
        key: str,
        row_index: int,
        name: str,
        ref: str,
        source_field: str,
        *,
        span: list[int] | None = None,
    ):
        scope = self.scopes[key]
        row = scope.rows[row_index]
        node = next((node for node in row["nodes"] if node["ref"] == ref), None)
        allowed = {
            "text",
            "rawText",
            "contentDescription",
            "checkedState",
            "checked",
            "isChecked",
            "stateDescription",
            "accessibility_type_hint",
        }
        if node is None or source_field not in allowed or source_field not in node:
            raise QueryError("fact_not_observed_in_row")
        if node.get("isShowingHintText") is True and source_field in {
            "text",
            "rawText",
        }:
            raise QueryError("fact_is_hint")
        value = node[source_field]
        if not isinstance(value, (str, bool, int, float)):
            raise QueryError("fact_value_invalid")
        if span is not None:
            if (
                not isinstance(value, str)
                or not isinstance(span, list)
                or len(span) != 2
                or any(type(part) is not int for part in span)
                or not 0 <= span[0] < span[1] <= len(value)
            ):
                raise QueryError("fact_span_invalid")
            value = value[span[0] : span[1]]
        row["facts"][name] = {
            "value": value,
            "evidence_refs": [ref],
            "source": "accessibility",
            "source_field": source_field,
            "span": span,
        }

    def aggregate(
        self,
        key: str,
        operation: str,
        *,
        field_name: str | None = None,
        filters: list[dict] | None = None,
        output_unit: str | None = None,
        device_datetime: str | None = None,
    ) -> dict:
        scope = self.scopes[key]
        if not scope.complete:
            raise QueryError("collection_coverage_incomplete")
        rows, evidence = [], []
        for index in sorted(scope.rows):
            row = scope.rows[index]
            if row["heading"]:
                continue
            included = True
            for condition in filters or []:
                fact = row["facts"].get(condition["field"])
                if fact is None:
                    raise QueryError("filter_property_unknown")
                evidence.extend(fact["evidence_refs"])
                if condition["operator"] == "eq":
                    included &= fact["value"] == condition["value"]
                elif condition["operator"] == "contains" and isinstance(
                    fact["value"], str
                ):
                    included &= str(condition["value"]) in fact["value"]
                elif condition["operator"] == "date_interval":
                    actual = observed_date(
                        fact["value"],
                        condition.get("format"),
                        device_datetime=device_datetime,
                    )
                    start = datetime.fromisoformat(condition["start"])
                    end = datetime.fromisoformat(condition["end"])
                    if not start.tzinfo or not end.tzinfo or start >= end:
                        raise QueryError("date_interval_invalid")
                    if actual.tzinfo is None:
                        actual = actual.replace(tzinfo=start.tzinfo)
                    included &= start <= actual < end
                else:
                    raise QueryError("filter_operator_unsupported")
            if included:
                rows.append(row)
        if operation == "count":
            result = len(rows)
        elif operation in {"sum", "max", "list"}:
            values = []
            for row in rows:
                fact = row["facts"].get(field_name)
                if fact is None:
                    raise QueryError("aggregate_property_unknown")
                evidence.extend(fact["evidence_refs"])
                values.append(
                    fact["value"]
                    if operation == "list"
                    else quantity(fact["value"], output_unit)
                )
            if operation == "list":
                result = values
            elif operation == "sum":
                result = sum(values, Decimal(0))
            elif values:
                result = max(values)
            else:
                raise QueryError("maximum_of_empty_collection")
        else:
            raise QueryError("aggregate_operation_unsupported")
        evidence.append(scope.container_ref)
        evidence.extend(row["row_ref"] for row in scope.rows.values())
        return {
            "value": result,
            "scope": key,
            "generation": scope.generation,
            "coverage": {
                "observed": len(scope.rows),
                "expected": scope.expected_count,
                "complete": True,
            },
            "evidence_refs": list(dict.fromkeys(evidence)),
        }

    def context(self, *, max_rows=40, max_chars=12000) -> list[dict]:
        contexts = [
            {
                "scope": scope.key,
                "generation": scope.generation,
                "expected_rows": scope.expected_count,
                "observed_rows": sorted(scope.rows),
                "complete": scope.complete,
                "invalidations": scope.invalidations,
                "container_ref": scope.container_ref,
                "scope_context": copy.deepcopy(scope.native_context),
                "scope_context_omitted": 0,
                "rows": [
                    {
                        "row_index": index,
                        "row_ref": row["row_ref"],
                        "heading": row["heading"],
                        "nodes": [
                            {
                                name: node[name]
                                for name in (
                                    "ref",
                                    "resourceId",
                                    "rawText",
                                    "text",
                                    "contentDescription",
                                    "stateDescription",
                                    "checkedState",
                                    "checked",
                                    "accessibility_type_hint",
                                )
                                if name in node
                            }
                            for node in row["nodes"]
                        ],
                        "facts": copy.deepcopy(row["facts"]),
                    }
                    for index, row in list(sorted(scope.rows.items()))[-max_rows:]
                ],
                "rows_omitted": max(0, len(scope.rows) - max_rows),
            }
            for scope in list(self.scopes.values())[-8:]
        ]
        remaining = max_chars
        for context in reversed(contexts):
            kept_context = []
            for item in context["scope_context"]:
                cost = len(json.dumps(item, ensure_ascii=False))
                if cost <= remaining:
                    kept_context.append(item)
                    remaining -= cost
                else:
                    context["scope_context_omitted"] += 1
            context["scope_context"] = kept_context
            kept = []
            for row in context["rows"]:
                cost = len(json.dumps(row, ensure_ascii=False))
                if cost <= remaining:
                    kept.append(row)
                    remaining -= cost
                else:
                    context["rows_omitted"] += 1
            context["rows"] = kept
        return contexts

    def execute(
        self, plan: dict[str, Any], *, device_datetime: str | None = None
    ) -> dict:
        """Bind only actual native values, then calculate in one atomic local step."""
        if (
            not isinstance(plan, dict)
            or set(plan)
            - {
                "scope",
                "generation",
                "operation",
                "facts",
                "field",
                "filters",
                "output_unit",
                "selectors",
            }
            or not {"scope", "generation", "operation"} <= set(plan)
        ):
            raise QueryError("query_plan_invalid")
        key = plan["scope"]
        scope = self.scopes.get(key) if isinstance(key, str) else None
        if (
            scope is None
            or type(plan["generation"]) is not int
            or plan["generation"] != scope.generation
        ):
            raise QueryError("query_scope_stale")
        working = copy.deepcopy(self)
        selectors = plan.get("selectors", [])
        if not isinstance(selectors, list) or len(selectors) > 16:
            raise QueryError("query_selectors_invalid")
        for selector in selectors:
            if (
                not isinstance(selector, dict)
                or set(selector) != {"name", "resource_id", "source_field"}
                or not all(
                    isinstance(selector[key], str) and selector[key] for key in selector
                )
            ):
                raise QueryError("query_selector_invalid")
            for index, row in scope.rows.items():
                if row["heading"]:
                    continue
                matches = [
                    node
                    for node in row["nodes"]
                    if node.get("resourceId") == selector["resource_id"]
                ]
                if len(matches) != 1:
                    raise QueryError("query_selector_ambiguous")
                working.add_native_fact(
                    key,
                    index,
                    selector["name"],
                    matches[0]["ref"],
                    selector["source_field"],
                )
        facts = plan.get("facts", [])
        if not isinstance(facts, list) or len(facts) > 200:
            raise QueryError("query_facts_invalid")
        for fact in facts:
            if (
                not isinstance(fact, dict)
                or set(fact) - {"row", "name", "ref", "source_field", "span"}
                or not {"row", "name", "ref", "source_field"} <= set(fact)
                or type(fact["row"]) is not int
                or fact["row"] not in scope.rows
                or not isinstance(fact["name"], str)
                or not re.fullmatch(r"[\w -]{1,64}", fact["name"])
            ):
                raise QueryError("query_fact_invalid")
            working.add_native_fact(
                key,
                fact["row"],
                fact["name"],
                fact["ref"],
                fact["source_field"],
                span=fact.get("span"),
            )
        filters = copy.deepcopy(plan.get("filters", []))
        if not isinstance(filters, list) or len(filters) > 16:
            raise QueryError("query_filters_invalid")
        for condition in filters:
            if not isinstance(condition, dict) or not {"field", "operator"} <= set(
                condition
            ):
                raise QueryError("query_filter_invalid")
            if condition["operator"] == "relative_week":
                if (
                    set(condition)
                    - {"field", "operator", "next_week", "week_start", "format"}
                    or type(condition.get("next_week", False)) is not bool
                ):
                    raise QueryError("query_filter_invalid")
                start, end = week_interval(
                    device_datetime,
                    next_week=condition.get("next_week", False),
                    week_start=condition.get("week_start", 0),
                )
                condition.update(
                    operator="date_interval",
                    start=start.isoformat(),
                    end=end.isoformat(),
                )
            elif condition["operator"] in {"eq", "contains"}:
                if set(condition) != {"field", "operator", "value"}:
                    raise QueryError("query_filter_invalid")
            elif condition["operator"] != "date_interval":
                raise QueryError("filter_operator_unsupported")
        result = working.aggregate(
            key,
            plan["operation"],
            field_name=plan.get("field"),
            filters=filters,
            output_unit=plan.get("output_unit"),
            device_datetime=device_datetime,
        )
        result["value"] = (
            str(result["value"])
            if isinstance(result["value"], Decimal)
            else result["value"]
        )
        result["operation"] = plan["operation"]
        result["field"] = plan.get("field")
        result["filters"] = filters
        self.scopes[key] = working.scopes[key]
        return result


def observed_date(
    value, date_format: str | None = None, *, device_datetime: str | None = None
) -> datetime:
    if not isinstance(value, str):
        raise QueryError("date_value_invalid")
    # These labels have an unambiguous offset from the observed device day.
    # Weekday names and yearless month/day labels deliberately remain unknown.
    relative_days = {
        "today": 0,
        "yesterday": -1,
        "tomorrow": 1,
        "今天": 0,
        "昨天": -1,
        "明天": 1,
    }
    if date_format is None and value.strip().casefold() in relative_days:
        try:
            reference = datetime.fromisoformat(device_datetime)
        except (ValueError, TypeError) as exc:
            raise QueryError("device_time_unknown") from exc
        if reference.tzinfo is None:
            raise QueryError("device_timezone_unknown")
        return reference.replace(hour=0, minute=0, second=0, microsecond=0) + timedelta(
            days=relative_days[value.strip().casefold()]
        )
    if date_format is not None and date_format not in {
        "%Y-%m-%d",
        "%Y/%m/%d",
        "%d/%m/%Y",
        "%m/%d/%Y",
        "%b %d, %Y",
        "%B %d, %Y",
    }:
        raise QueryError("date_format_unsupported")
    try:
        return (
            datetime.strptime(value.strip(), date_format)
            if date_format
            else datetime.fromisoformat(value.strip())
        )
    except ValueError as exc:
        raise QueryError("date_ambiguous") from exc


DISTANCE_UNITS = {
    "m": Decimal(1),
    "meters": Decimal(1),
    "km": Decimal(1000),
    "mi": Decimal("1609.344"),
    "ft": Decimal("0.3048"),
}
TIME_UNITS = {
    "s": Decimal(1),
    "seconds": Decimal(1),
    "min": Decimal(60),
    "minutes": Decimal(60),
    "h": Decimal(3600),
    "hours": Decimal(3600),
}


def quantity(raw, output_unit: str | None = None) -> Decimal:
    if isinstance(raw, bool) or not isinstance(raw, (str, int, float, Decimal)):
        raise QueryError("quantity_invalid")
    text = str(raw).strip()
    clock = re.fullmatch(r"(\d+):(\d{2}):(\d{2})", text)
    if clock:
        hours, minutes, seconds = map(int, clock.groups())
        if minutes >= 60 or seconds >= 60 or output_unit not in TIME_UNITS:
            raise QueryError("duration_invalid")
        return Decimal(hours * 3600 + minutes * 60 + seconds) / TIME_UNITS[output_unit]
    match = re.fullmatch(r"([+-]?\d+(?:\.\d+)?)\s*([A-Za-z]*)", text)
    if not match:
        raise QueryError("quantity_or_locale_ambiguous")
    try:
        number = Decimal(match[1])
    except InvalidOperation as exc:
        raise QueryError("quantity_invalid") from exc
    source = match[2].lower()
    if not source and output_unit is None:
        return number
    for units in (DISTANCE_UNITS, TIME_UNITS):
        if source in units and output_unit in units:
            return number * units[source] / units[output_unit]
    raise QueryError("unit_unknown_or_incompatible")


def week_interval(
    device_datetime: str, *, next_week=False, week_start=0
) -> tuple[datetime, datetime]:
    if type(week_start) is not int or not 0 <= week_start <= 6:
        raise QueryError("week_start_invalid")
    try:
        reference = datetime.fromisoformat(device_datetime)
    except (ValueError, TypeError) as exc:
        raise QueryError("device_time_unknown") from exc
    if reference.tzinfo is None:
        raise QueryError("device_timezone_unknown")
    start = (
        reference - timedelta(days=(reference.weekday() - week_start) % 7)
    ).replace(hour=0, minute=0, second=0, microsecond=0)
    if next_week:
        start += timedelta(days=7)
    return start, start + timedelta(days=7)
