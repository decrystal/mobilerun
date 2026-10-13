"""Goal scope claims must use the actual query's native selection evidence."""

import asyncio
import copy
import json

import httpx
import pytest
from test_decision_visual_guards import _runtime
from test_native_query_binding import collection
from test_task_evidence import task, ui

from mobilerun.agent.decision.contracts import (
    ContractError,
    parse_grounded_review,
    parse_task_contract,
)
from mobilerun.agent.decision.native_query_binding import (
    NativeQueryBindingError,
    bind_native_queries,
)
from mobilerun.agent.decision.runtime import DecisionRuntime
from mobilerun.agent.decision.task_state import TaskState
from mobilerun.agent.decision.visual_fallback import VisualFallback, _ledger_evidence
from mobilerun.config_manager.config_manager import JevConfig

GOAL = "Count the records in the currently displayed Active list"


def scoped_contract(reference="current", constraints=()):
    contract = task(GOAL)
    contract["requirements"][0]["attribute"] = {"kind": "count", "label": "records"}
    contract["requirements"][0]["query_scope"] = {
        "reference": reference,
        "quote": GOAL,
        "constraints": list(constraints),
    }
    return contract


def fixture(reference="current", constraints=()):
    state = TaskState(GOAL)
    state.set_contract(scoped_contract(reference, constraints))
    state.observe(collection(filter_text="Earlier records"))
    prior = next(iter(state.collections.scopes))
    state.query({"scope": prior, "generation": 1, "operation": "count"})
    state.observe(collection(filter_text="Active"))
    current = next(key for key in state.collections.scopes if key != prior)
    state.query({"scope": current, "generation": 1, "operation": "count"})
    return state


def response(state, ref="Q2", *, constraints=None, supported=True, scoped=True):
    result = state.evidence()[ref]
    scope = state.collections.scopes[result["scope"]]
    refs = [ref, scope.container_ref, *[item["ref"] for item in scope.native_context]]
    item = {
        "id": 1,
        "status": "supported" if supported else "unknown",
        "evidence": "Used the observed list",
        "evidence_refs": refs,
    }
    if scoped:
        item["scope_refs"] = refs[1:]
        item["constraint_evidence"] = constraints if constraints is not None else []
    return {
        "complete": supported,
        "evidence": "Observed the query source",
        "checks": {},
        "final_answer": "3" if supported else None,
        "assessments": [item],
    }


def parsed(state, data):
    return parse_grounded_review(
        json.dumps(data),
        state.contract,
        state.evidence(),
        state.current_id,
        derive_review_dependencies=True,
    )


def test_current_query_without_scope_evidence_cannot_use_an_earlier_complete_list():
    state = fixture()
    with pytest.raises(ContractError):
        parsed(state, response(state, "Q1", scoped=False))


def test_declared_filter_without_a_constraint_proof_cannot_count_as_success():
    state = fixture("observed", ["Active"])
    with pytest.raises(ContractError):
        parsed(state, response(state, "Q1", scoped=False))


def test_actual_query_records_its_native_scope_source_instead_of_a_model_claim():
    state = fixture()
    result = state.evidence()["Q2"]
    scope = state.collections.scopes[result["scope"]]
    assert result["scope_source"] == {
        "observation": "O2",
        "container_ref": scope.container_ref,
        "context_refs": [item["ref"] for item in scope.native_context],
    }


def test_current_native_answer_cannot_cite_only_an_earlier_observation():
    state = fixture()
    data = response(state, scoped=False)
    data["assessments"][0]["evidence_refs"] = ["O1:n2"]
    with pytest.raises(ContractError):
        parsed(state, data)


def test_current_query_with_its_actual_current_scope_is_supported():
    state = fixture()
    result = parsed(state, response(state))
    assert result["status"] == "done" and result["final_answer"] == "3"


def test_current_claim_rejects_an_earlier_query_even_with_valid_scope_refs():
    state = fixture()
    with pytest.raises(ContractError, match="query_scope_unverified"):
        parsed(state, response(state, "Q1"))


def test_explicit_observed_range_can_use_earlier_query_after_detail_navigation():
    state = fixture("observed")
    state.observe(ui([{"index": 1, "rawText": "Unrelated details"}]))
    assert parsed(state, response(state, "Q1"))["status"] == "done"


def test_filter_evidence_binds_to_the_same_scope_source():
    state = fixture("observed", ["Active"])
    selector = state.evidence()["Q2"]["scope_source"]["context_refs"][0]
    data = response(
        state, constraints=[{"quote": "Active", "evidence_refs": [selector]}]
    )
    assert parsed(state, data)["status"] == "done"


@pytest.mark.parametrize(
    "change",
    [
        "missing",
        "unknown",
        "different_scope",
        "row_without_filter",
        "extra",
        "duplicate",
        "absent_membership",
    ],
)
def test_constraints_cannot_be_missing_or_bound_to_unrelated_proof(change):
    state = fixture("observed", ["Active"])
    selector = state.evidence()["Q2"]["scope_source"]["context_refs"][0]
    proof = {"quote": "Active", "evidence_refs": [selector]}
    data = response(state, constraints=[proof])
    item = data["assessments"][0]
    if change == "missing":
        item["constraint_evidence"] = []
    elif change == "unknown":
        proof["evidence_refs"] = ["O99:n0"]
    elif change == "different_scope":
        proof["evidence_refs"] = [
            state.evidence()["Q1"]["scope_source"]["context_refs"][0]
        ]
        item["evidence_refs"] += proof["evidence_refs"]
    elif change == "row_without_filter":
        proof["evidence_refs"] = ["O2:n2"]
        item["evidence_refs"] += proof["evidence_refs"]
    elif change == "extra":
        item["constraint_evidence"].append(
            {"quote": "Earlier", "evidence_refs": [selector]}
        )
    elif change == "duplicate":
        item["constraint_evidence"].append(copy.deepcopy(proof))
    else:
        item["evidence_refs"].remove(selector)
    with pytest.raises(ContractError):
        parsed(state, data)


@pytest.mark.parametrize(
    "change", ["missing", "false_observation", "false_container", "different_context"]
)
def test_query_source_metadata_cannot_weaken_the_private_source_binding(change):
    state = fixture()
    result = state.query_results[1]
    if change == "missing":
        del result["scope_source"]
    elif change == "false_observation":
        result["scope_source"]["observation"] = "O1"
    elif change == "false_container":
        result["scope_source"]["container_ref"] = "O1:n0"
    else:
        result["scope_source"]["context_refs"] = ["O1:n7"]
    with pytest.raises(NativeQueryBindingError):
        bind_native_queries(["Q2"], state, state.current_id, goal=GOAL)


def test_unknown_scope_can_remain_unknown_without_fabricating_range_proofs():
    state = fixture()
    data = response(state, supported=False, scoped=False)
    assert parsed(state, data)["status"] == "incomplete"


def test_current_direct_native_reading_with_current_scope_is_supported():
    state = fixture()
    data = response(state)
    item = data["assessments"][0]
    item["evidence_refs"].remove("Q2")
    item["evidence_refs"].append("O2:n2")
    assert parsed(state, data)["status"] == "done"


def test_current_scope_does_not_allow_old_direct_values_with_a_new_selector():
    state = fixture()
    data = response(state)
    item = data["assessments"][0]
    item["evidence_refs"].remove("Q2")
    item["evidence_refs"].append("O1:n2")
    with pytest.raises(ContractError, match="query_scope_unverified"):
        parsed(state, data)


@pytest.mark.parametrize(
    "spec",
    [
        None,
        {},
        {"reference": "invented", "quote": GOAL, "constraints": []},
        {"reference": "current", "quote": "Not in goal", "constraints": []},
        {"reference": "observed", "quote": GOAL, "constraints": ["Not in goal"]},
        {"reference": "observed", "quote": GOAL, "constraints": ["Active", "Active"]},
        {"reference": "current", "quote": GOAL, "constraints": "Active"},
    ],
)
def test_required_query_scope_is_strict_and_grounded_in_original_goal(spec):
    contract = scoped_contract()
    if spec is None:
        contract["requirements"][0].pop("query_scope")
    else:
        contract["requirements"][0]["query_scope"] = spec
    with pytest.raises(ContractError):
        parse_task_contract(json.dumps(contract), GOAL, require_query_scopes=True)


def test_valid_query_scope_roundtrips_and_does_not_create_ui_bindings_from_goal():
    contract = scoped_contract("observed", ["Active"])
    assert (
        parse_task_contract(json.dumps(contract), GOAL, require_query_scopes=True)
        == contract
    )
    assert contract["checks"] == {}


@pytest.mark.parametrize("switch", [1, "yes", None])
def test_scope_switch_requires_a_real_boolean(switch):
    with pytest.raises(ValueError):
        JevConfig(query_scopes=switch)
    with pytest.raises(ValueError):
        VisualFallback(query_scopes=switch)
    with pytest.raises(ContractError):
        parse_task_contract(
            json.dumps(scoped_contract()), GOAL, require_query_scopes=switch
        )


def test_default_runtime_enables_scope_contracts_and_keeps_provider_and_route():
    config = JevConfig(fallback="autoglm")
    runner = DecisionRuntime(config, None, None, None)
    assert config.query_scopes is True and runner.visual_fallback.query_scopes is True
    assert config.fallback_model == config.reasoning_model == "glm-5.3-flash"
    assert config.fallback_thinking == "disabled"
    assert config.fallback_base_url == "http://66.154.100.178:8080/v1"


def test_existing_prepare_stage_requires_scope_and_uses_its_existing_single_retry(
    monkeypatch,
):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-secret")
    requests = []

    def respond(request):
        requests.append(json.loads(request.content))
        contract = scoped_contract()
        if len(requests) == 1:
            contract["requirements"][0].pop("query_scope")
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(contract)}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = VisualFallback(
                model="glm-5.3-flash",
                reasoning_model="glm-5.3-flash",
                api_key_env="GLM_API_KEY",
                query_scopes=True,
                client=client,
            )
            assert await adapter.prepare_task(goal=GOAL) is None
            assert adapter.completion_contract == scoped_contract()
            assert await adapter.prepare_task(goal=GOAL) is None

    asyncio.run(run())
    assert len(requests) == 2
    assert all(request["max_tokens"] == 2400 for request in requests)
    assert all(
        "query_scope" in request["messages"][0]["content"] for request in requests
    )


def filtered_fixture(*, dates=False):
    goal = "Count records from 2026-10-01 to 2026-10-07" if dates else GOAL
    quotes = ["2026-10-01", "2026-10-07"] if dates else ["Active"]
    contract = scoped_contract("observed", quotes)
    contract["requirements"][0].update(description=goal, quote=goal)
    contract["requirements"][0]["query_scope"]["quote"] = goal
    contract = parse_task_contract(
        json.dumps(contract), goal, require_query_scopes=True
    )
    names = (
        ["2026-10-01", "2026-10-03", "2026-11-01"]
        if dates
        else ["Active", "Inactive", "Active"]
    )
    state = TaskState(goal)
    state.set_contract(contract)
    state.observe(collection(rows=list(enumerate(names)), filter_text="All records"))
    key = next(iter(state.collections.scopes))
    condition = (
        {
            "field": "observed",
            "operator": "date_interval",
            "start": "2026-10-01T00:00:00+00:00",
            "end": "2026-10-08T00:00:00+00:00",
        }
        if dates
        else {
            "field": "observed",
            "operator": "eq",
            "value": "Active",
        }
    )
    facts = [
        {
            "row": index,
            "name": "observed",
            "ref": f"O1:n{index * 2 + 2}",
            "source_field": "rawText",
        }
        for index in range(3)
    ]
    result = state.query(
        {
            "scope": key,
            "generation": 1,
            "operation": "count",
            "facts": facts,
            "filters": [condition],
        }
    )
    return state, result


@pytest.mark.parametrize("dates", [False, True])
def test_explicit_date_or_row_property_can_use_actual_query_filter_facts(dates):
    state, result = filtered_fixture(dates=dates)
    assert result["value"] == 2 and result["filter_evidence_refs"] == [
        "O1:n2",
        "O1:n4",
        "O1:n6",
    ]
    conditions = [
        {"quote": quote, "evidence_refs": result["filter_evidence_refs"]}
        for quote in state.contract["requirements"][0]["query_scope"]["constraints"]
    ]
    data = response(state, "Q1", constraints=conditions)
    data["assessments"][0]["evidence_refs"] += result["filter_evidence_refs"]
    data["final_answer"] = "2"
    assert parsed(state, data)["status"] == "done"


def test_scope_condition_cannot_mix_valid_selector_with_a_foreign_list():
    state = fixture("observed", ["Active"])
    selector = state.evidence()["Q2"]["scope_source"]["context_refs"][0]
    foreign = state.evidence()["Q1"]["scope_source"]["context_refs"][0]
    data = response(
        state, constraints=[{"quote": "Active", "evidence_refs": [selector, foreign]}]
    )
    data["assessments"][0]["evidence_refs"].append(foreign)
    with pytest.raises(ContractError, match="query_scope_unverified"):
        parsed(state, data)


def test_scope_selector_value_tampering_invalidates_private_query_receipt():
    state = fixture()
    context_ref = state.evidence()["Q2"]["scope_source"]["context_refs"][0]
    state.evidence()[context_ref]["rawText"] = "Another selection"
    with pytest.raises(NativeQueryBindingError, match="native_query_source_changed"):
        bind_native_queries(["Q2"], state, state.current_id, goal=GOAL)


def test_scope_context_is_available_to_reviewer_when_observation_window_has_evicted_it():
    state = fixture("observed", ["Active"])
    state.observe(ui([{"index": 1, "rawText": "Details"}]))
    context = state.context(max_observations=1)
    proof = state.evidence()["Q2"]["scope_source"]["context_refs"][0]
    assert proof in _ledger_evidence(context)
    assert any(
        proof == item["ref"]
        for scope in context["collections"]
        for item in scope["scope_context"]
    )


def test_reviewer_does_not_recover_budget_omitted_scope_values_from_private_state():
    state = fixture()
    context = state.context(max_observations=0, max_chars=0)
    context["observations"] = []
    context["collections"] = state.collections.context(max_chars=0)
    proof = state.evidence()["Q2"]["scope_source"]["context_refs"][0]
    assert proof not in _ledger_evidence(context)


def test_comparison_may_bind_two_actual_observed_ranges_without_reopening_them():
    state = fixture("observed", ["Active"])
    first = response(state, "Q1")["assessments"][0]
    second = response(state, "Q2")["assessments"][0]
    data = response(
        state, constraints=[{"quote": "Active", "evidence_refs": second["scope_refs"]}]
    )
    data["assessments"][0]["scope_refs"] += first["scope_refs"]
    data["assessments"][0]["evidence_refs"] += first["evidence_refs"]
    assert parsed(state, data)["status"] == "done"


@pytest.mark.parametrize(
    "change",
    [
        "empty_scope",
        "bool_scope",
        "unknown_scope",
        "bare_observation",
        "query_as_scope",
        "duplicate_scope",
        "missing_container",
    ],
)
def test_supported_scope_requires_specific_real_source_refs(change):
    state = fixture()
    data = response(state)
    item = data["assessments"][0]
    if change == "empty_scope":
        item["scope_refs"] = []
    elif change == "bool_scope":
        item["scope_refs"] = True
    elif change == "unknown_scope":
        item["scope_refs"] = ["O99:n0"]
    elif change == "bare_observation":
        item["scope_refs"] = ["O2"]
        item["evidence_refs"].append("O2")
    elif change == "query_as_scope":
        item["scope_refs"] = ["Q2"]
    elif change == "duplicate_scope":
        item["scope_refs"].append(item["scope_refs"][0])
    else:
        item["scope_refs"] = item["scope_refs"][1:]
    with pytest.raises(ContractError):
        parsed(state, data)


@pytest.mark.parametrize(
    "bad_scope", ["query_in_condition", "query_in_scope", "wrong_quote"]
)
def test_range_protocol_feedback_uses_the_existing_retry_without_relaxing_the_guard(
    monkeypatch, bad_scope
):
    monkeypatch.setenv("GLM_API_KEY", "offline-test-secret")
    state = fixture("current", ["Active"])
    requests = []
    screen = collection(filter_text="Active")

    def respond(request):
        requests.append(json.loads(request.content))
        selector = state.evidence()["Q2"]["scope_source"]["context_refs"][0]
        data = response(
            state, constraints=[{"quote": "Active", "evidence_refs": [selector]}]
        )
        if len(requests) == 1:
            item = data["assessments"][0]
            if bad_scope == "query_in_condition":
                item["constraint_evidence"][0]["evidence_refs"].append("Q2")
            elif bad_scope == "query_in_scope":
                item["scope_refs"].append("Q2")
            else:
                item["constraint_evidence"][0]["quote"] = "Active list"
        return httpx.Response(
            200, json={"choices": [{"message": {"content": json.dumps(data)}}]}
        )

    async def run():
        async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
            adapter = VisualFallback(
                model="glm-5.3-flash",
                reasoning_model="glm-5.3-flash",
                client=client,
                api_key_env="GLM_API_KEY",
                query_scopes=True,
                base_url="https://example.invalid/v1",
            )
            adapter._goal, adapter._completion_contract = GOAL, state.contract
            runner, _, _ = _runtime(screen, visual=adapter)
            runner.config = JevConfig(fallback="autoglm")
            runner.goal, runner.task_state = GOAL, state
            selected = await runner._visual(
                screen, completion_review=True, native_only_review=True
            )
            assert (
                not selected.stop
                and selected.calls[0].parameters["final_answer"] == "3"
            )
            assert selected.completion_query_binding_required
            _, reason = await runner.validate_before_execute()
            assert reason is None and runner.history == []
            assert sum(item["event"] == "format_retry" for item in runner.trace) == 1
            assert "not scope_refs or constraint_evidence" in runner.review_diagnostic

    asyncio.run(run())
    assert len(requests) == 2
    assert "verbatim" in json.dumps(requests[1])
    assert all(request["max_tokens"] == 3000 for request in requests)


def test_query_result_is_not_itself_evidence_of_the_requested_filter():
    state = fixture("current", ["Active"])
    selector = state.evidence()["Q2"]["scope_source"]["context_refs"][0]
    data = response(
        state, constraints=[{"quote": "Active", "evidence_refs": [selector, "Q2"]}]
    )
    with pytest.raises(ContractError, match="query_scope_unverified"):
        parsed(state, data)
