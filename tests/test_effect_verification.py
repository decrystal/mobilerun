"""Side effects are verified before any new decision and never blindly replayed."""

import asyncio
import json
from types import SimpleNamespace

import pytest
from test_jev_first_runtime import config, runtime, ui_state

from mobilerun.agent.action_result import ActionResult
from mobilerun.agent.decision.effects import (
    VERIFY_STEP_BUDGET,
    EffectLedger,
    EffectNotes,
    effect_key,
    evidence_valid,
    returned_to_previous_screen,
)
from mobilerun.agent.decision.runtime import Selection
from mobilerun.agent.decision.visual_fallback import (
    VisualDecision,
    VisualToolCall,
    parse_visual_decision,
)
from mobilerun.agent.fast_agent.xml_parser import ToolCall

PAGE = [
    {"index": 1, "text": "ANNA壁纸", "bounds": "0,0,400,80"},
    {"index": 2, "text": "立即追蹤", "bounds": "0,80,400,160", "clickable": True},
]
MENU = PAGE + [
    {
        "index": 9,
        "className": "FrameLayout",
        "bounds": "0,600,200,700",
        "clickable": True,
        "children": [{"index": 10, "text": "儲存", "bounds": "0,600,200,700"}],
    }
]
GOAL = "保存 2 张壁纸到相册"


class Verdicts:
    """Fake visual model that answers each request with the next factory result."""

    def __init__(self, *factories):
        self.factories = list(factories)
        self.requests = []

    async def decide(self, **kwargs):
        self.requests.append(kwargs)
        factory = self.factories.pop(0) if len(self.factories) > 1 else self.factories[0]
        return factory(kwargs)


def pending_effect(request):
    effects = request["task_state"]["ledger"]["side_effects"]
    return next(e["id"] for e in effects if e["status"] == "pending")


def verified(status, evidence):
    return lambda request: VisualDecision(
        "verified",
        verification={
            "effect": pending_effect(request),
            "status": status,
            "evidence": tuple(evidence(request)) if callable(evidence) else evidence,
            "note": "长按图片选「儲存」后用Recent_media确认",
        },
    )


def current_observation(request):
    return [request["task_state"]["ledger"]["current_observation"]]


def _runner(tmp_path, visual=None, **kwargs):
    runner, events, policy, fake, provider = runtime(ui=ui_state(PAGE), **kwargs)
    if visual is not None:
        runner.visual_fallback = visual
    runner.effect_notes = EffectNotes(tmp_path / "notes.json")
    runner.goal = GOAL
    runner.task_state.reset(GOAL)
    return runner, events, policy, provider


def _execute(runner, call, snapshot, effect=""):
    runner.task_state.observe(snapshot)
    runner.pending = Selection("autoglm", [call], snapshot=snapshot, expected_effect=effect)
    runner.record_execution(call, ActionResult(True, "ok"), snapshot)


def test_effect_key_uses_target_text_not_pixels():
    call = ToolCall("click", {"index": 9})
    assert effect_key(call, "com.xingin.xhs", MENU) == ("com.xingin.xhs", "click", "儲存")
    fallback = effect_key(ToolCall("long_press_at", {"x": 5, "y": 6}), "app", MENU)
    assert fallback == ("app", "long_press_at", '{"x": 5, "y": 6}')


@pytest.mark.parametrize(
    "trail,current,expected",
    [
        ([("long_press_at", "page", False), ("click", "menu", False)], "page", True),
        ([("long_press_at", "page", False), ("click", "menu", True)], "page", False),
        ([("long_press_at", "page", False), ("system_button", "menu", False)], "page", False),
        ([("click", "page", False), ("click", "menu", False)], "detail", False),
        ([("click", "menu", False)], "page", False),
    ],
)
def test_structural_net_flags_a_closed_menu_returning_to_the_page(trail, current, expected):
    steps = [{"tool": t, "fingerprint": f, "declared": d} for t, f, d in trail]
    assert returned_to_previous_screen(steps, current) is expected


def test_evidence_must_exist_and_follow_the_action():
    ledger = EffectLedger()
    entry = ledger.record(
        key=("app", "click", "儲存"),
        description="保存",
        action_ref="A3",
        before="O4",
        signature=("sig",),
        source="declared",
    )
    observations = [{"id": f"O{i}"} for i in range(1, 7)]
    actions = [{"ref": f"A{i}"} for i in range(1, 5)]
    assert evidence_valid(entry, ["O5"], observations, actions)
    assert evidence_valid(entry, ["A4", "O6"], observations, actions)
    assert not evidence_valid(entry, ["O4"], observations, actions)  # before the action
    assert not evidence_valid(entry, ["A3"], observations, actions)  # the action itself
    assert not evidence_valid(entry, ["O9"], observations, actions)  # unknown
    assert not evidence_valid(entry, [], observations, actions)


def test_ledger_summary_and_single_retry_after_not_happened():
    ledger = EffectLedger()
    key = ("app", "click", "儲存")
    first = ledger.record(
        key=key,
        description="保存第1张",
        action_ref="A1",
        before="O1",
        signature=(1,),
        source="declared",
    )
    ledger.resolve(first, "confirmed", ["O2"])
    second = ledger.record(
        key=key,
        description="保存第2张",
        action_ref="A2",
        before="O2",
        signature=(2,),
        source="declared",
    )
    assert ledger.may_retry(second)
    ledger.resolve(second, "not_happened", ["A3"])
    retry = ledger.record(
        key=key,
        description="保存第2张",
        action_ref="A4",
        before="O3",
        signature=(2,),
        source="declared",
    )
    assert not ledger.may_retry(retry)
    summary = ledger.summary()
    assert "E1 [confirmed] 保存第1张 (evidence: O2)" in summary
    assert "E2 [did not happen]" in summary and "E3 [unverified]" in summary


def test_budget_counts_steps_and_seconds():
    ledger = EffectLedger()
    entry = ledger.record(
        key=("a", "click", "x"),
        description="",
        action_ref="A1",
        before=None,
        signature=(),
        source="declared",
    )
    entry.verify_steps = VERIFY_STEP_BUDGET
    assert ledger.budget_exhausted(entry, now=0)
    entry.verify_steps, entry.verify_started = 0, 0
    assert ledger.budget_exhausted(entry, now=121)
    assert not ledger.budget_exhausted(entry, now=60)


def test_notes_are_capped_deduplicated_and_tolerate_a_missing_file(tmp_path):
    notes = EffectNotes(tmp_path / "missing" / "notes.json")
    assert notes.for_app("app") == []
    for i in range(7):
        notes.add("app", f"note {i}")
    notes.add("app", "note 6")
    assert notes.for_app("app") == [f"note {i}" for i in range(2, 7)]
    assert EffectNotes(None).for_app("app") == []


def test_parser_reads_declared_effect_verify_and_recent_media():
    ui = ui_state(MENU)
    tap = parse_visual_decision(
        json.dumps({"action": "Tap", "index": 9, "effect": "当前图片保存到相册"}),
        ui_state=ui,
        screenshot_size=(400, 800),
        available_tools={"click"},
        structured=True,
    )
    assert tap.status == "action" and tap.effect == "当前图片保存到相册"
    probe = parse_visual_decision(
        json.dumps({"action": "Recent_media"}),
        ui_state=ui,
        screenshot_size=(400, 800),
        available_tools={"recent_media"},
        structured=True,
    )
    assert probe.calls == (VisualToolCall("recent_media", {}),)
    verdict = json.dumps(
        {
            "action": "Verify",
            "id": "E1",
            "status": "confirmed",
            "evidence": ["A4"],
            "note": "查相册",
        }
    )
    ok = parse_visual_decision(
        verdict,
        ui_state=ui,
        screenshot_size=(400, 800),
        available_tools={"verify_effect"},
        structured=True,
    )
    assert ok.status == "verified" and ok.verification["evidence"] == ("A4",)
    outside = parse_visual_decision(
        verdict, ui_state=ui, screenshot_size=(400, 800), available_tools={"click"}, structured=True
    )
    assert outside.status == "error"
    bad = parse_visual_decision(
        verdict.replace("confirmed", "probably"),
        ui_state=ui,
        screenshot_size=(400, 800),
        available_tools={"verify_effect"},
        structured=True,
    )
    assert bad.status == "error"


def test_declared_effect_is_verified_before_the_next_decision(tmp_path):
    visual = Verdicts(verified("confirmed", current_observation))
    runner, events, policy, provider = _runner(tmp_path, visual=visual)
    # Tap 「儲存」 in the menu; the next observation is the post page again.
    _execute(runner, ToolCall("click", {"index": 9}), ui_state(MENU), effect="当前图片保存到相册")
    assert runner.effects.pending().id == "E1"

    selection = asyncio.run(runner.select(GOAL, ui_state(PAGE)))

    tools = visual.requests[0]["available_tools"]
    assert "verify_effect" in tools and "type" not in tools
    assert runner.effects.entries[0].status == "confirmed"
    assert runner.executed == set()  # confirmed progress resets loop history
    assert selection.source == "jev"  # the main loop continues with Jev
    assert "verify_effect" not in (runner.task_context()["ledger"]["side_effects"][0])
    assert EffectNotes(tmp_path / "notes.json").for_app("com.android.settings")


def test_structural_net_records_an_undeclared_menu_commit(tmp_path):
    runner, *_ = _runner(tmp_path)
    _execute(runner, ToolCall("long_press_at", {"x": 100, "y": 300}), ui_state(PAGE))
    _execute(runner, ToolCall("click", {"index": 9}), ui_state(MENU))
    runner.detect_structural_effect(ui_state(PAGE))
    entry = runner.effects.pending()
    assert entry.source == "structural" and entry.key[2] == "儲存"


def test_unknown_verdict_stops_without_replay(tmp_path):
    runner, *_ = _runner(tmp_path, visual=Verdicts(verified("unknown", [])))
    _execute(runner, ToolCall("click", {"index": 2}), ui_state(PAGE), effect="关注")
    selection = asyncio.run(runner.select(GOAL, ui_state(PAGE)))
    assert selection.stop and "E1 could not be verified" in selection.reason
    assert runner.effects.entries[0].status == "unverifiable"


@pytest.mark.parametrize(
    "cited,accepted",
    [
        ('点击"儲存"将当前图片保存到相册', True),  # real reply: description echoed
        ('E1: 点击"儲存"将当前图片保存到相册', True),  # real reply: ID plus text
        ("", True),
        ("E2", False),  # an explicit reference to another effect
    ],
)
def test_verdict_applies_to_the_effect_under_verification(tmp_path, cited, accepted):
    probe = lambda request: VisualDecision(  # noqa: E731
        "verified",
        verification={
            "effect": cited,
            "status": "not_happened",
            "evidence": tuple(current_observation(request)),
            "note": "",
        },
    )
    runner, *_ = _runner(tmp_path, visual=Verdicts(probe))
    _execute(runner, ToolCall("click", {"index": 9}), ui_state(MENU), effect="保存")
    asyncio.run(runner.select(GOAL, ui_state(PAGE)))
    status = runner.effects.entries[0].status
    assert status == ("not_happened" if accepted else "unverifiable")


def test_invalid_evidence_is_rejected_until_the_budget_is_spent(tmp_path):
    visual = Verdicts(verified("confirmed", ["O999"]))
    runner, *_ = _runner(tmp_path, visual=visual)
    _execute(runner, ToolCall("click", {"index": 2}), ui_state(PAGE), effect="关注")
    selection = asyncio.run(runner.select(GOAL, ui_state(PAGE)))
    assert len(visual.requests) == VERIFY_STEP_BUDGET
    assert selection.stop and "within the budget" in selection.reason


def test_verification_cannot_replay_the_action_being_verified(tmp_path):
    repeat = lambda request: VisualDecision(  # noqa: E731
        "action", calls=(VisualToolCall("click", {"index": 2}),)
    )
    visual = Verdicts(repeat)
    runner, events, policy, provider = _runner(tmp_path, visual=visual)
    _execute(runner, ToolCall("click", {"index": 2}), ui_state(PAGE), effect="关注")
    selection = asyncio.run(runner.select(GOAL, ui_state(PAGE)))
    # Every replay proposal is rejected and spends budget; none is executed.
    assert len(visual.requests) == VERIFY_STEP_BUDGET
    assert selection.stop and "within the budget" in selection.reason
    assert "one retry is then allowed" in runner.last_feedback


@pytest.mark.parametrize(
    "replies,steps",
    [
        # The existing in-step format retry absorbs one malformed reply.
        (
            [VisualDecision("error", message="Visual model returned an invalid or unavailable action.", format_error=True)],
            1,
        ),
        # A second malformed reply spends verification budget instead of stopping.
        (
            [
                VisualDecision(
                    "error", message="Visual model returned an invalid or unavailable action.", format_error=True
                ),
                VisualDecision(
                    "error", message="Visual model returned an invalid or unavailable action.", format_error=True
                ),
            ],
            2,
        ),
        ([VisualDecision("done", message="已保存")], 2),
    ],
)
def test_malformed_or_premature_finish_spends_budget_then_recovers(tmp_path, replies, steps):
    factories = [lambda request, reply=reply: reply for reply in replies]
    visual = Verdicts(*factories, verified("confirmed", current_observation))
    runner, *_ = _runner(tmp_path, visual=visual)
    _execute(runner, ToolCall("click", {"index": 9}), ui_state(MENU), effect="保存")
    asyncio.run(runner.select(GOAL, ui_state(PAGE)))
    assert runner.effects.entries[0].status == "confirmed"
    assert runner.effects.entries[0].verify_steps == steps


def test_not_happened_allows_exactly_one_retry(tmp_path):
    visual = Verdicts(verified("not_happened", current_observation))
    runner, *_ = _runner(tmp_path, visual=visual)
    call = ToolCall("click", {"index": 9})
    _execute(runner, call, ui_state(MENU), effect="保存")
    signature = runner.effects.entries[0].signature
    asyncio.run(runner.select(GOAL, ui_state(PAGE)))
    assert signature not in runner.executed
    assert "retry that action once" in runner.last_feedback

    _execute(runner, call, ui_state(MENU), effect="保存")
    asyncio.run(runner.select(GOAL, ui_state(PAGE)))
    assert runner.effects.entries[1].retries == 1
    assert "do not retry it" in runner.last_feedback


def test_jev_routes_a_repeat_of_a_confirmed_effect_to_the_visual_model(tmp_path):
    visual = Verdicts(
        lambda request: VisualDecision("action", calls=(VisualToolCall("wait", {"duration": 1.0}),))
    )
    runner, events, policy, provider = _runner(tmp_path, visual=visual)
    entry = runner.effects.record(
        key=runner._effect_key(ToolCall("click", {"index": 2}), ui_state(PAGE)),
        description="关注该作者",
        action_ref="A1",
        before=None,
        signature=(),
        source="declared",
    )
    runner.effects.resolve(entry, "confirmed", ["O2"])
    asyncio.run(runner.select(GOAL, ui_state(PAGE)))
    assert (
        visual.requests
        and "E1 (关注该作者) was already confirmed" in visual.requests[0]["feedback"]
    )


def test_llm_fallback_never_replays_an_unverified_effect(tmp_path):
    runner, *_ = _runner(tmp_path, settings=config(fallback="llm"))
    _execute(runner, ToolCall("click", {"index": 2}), ui_state(PAGE), effect="关注")
    selection = runner.accept_fallback([ToolCall("click", {"index": 2})], ui_state(PAGE))
    assert selection.stop and "E1 is unverified" in selection.reason


def test_verification_steps_do_not_create_new_side_effects(tmp_path):
    runner, *_ = _runner(tmp_path)
    _execute(runner, ToolCall("click", {"index": 2}), ui_state(PAGE), effect="关注")
    _execute(runner, ToolCall("click", {"index": 1}), ui_state(PAGE), effect="打开")
    assert [e.id for e in runner.effects.entries] == ["E1"]


def test_recent_media_lists_items_after_the_action_with_md5(tmp_path):
    runner, *_ = _runner(tmp_path)
    _execute(runner, ToolCall("click", {"index": 2}), ui_state(PAGE), effect="保存")
    calls = []

    async def shell(command):
        calls.append(command)
        if command == "date +%s":
            return "1000\n"
        if command.startswith("content query"):
            return (
                "Row: 0 _display_name=a.jpg, _size=10, date_added=999, "
                "_data=/storage/emulated/0/DCIM/a.jpg\n"
            )
        return "abc123  /storage/emulated/0/DCIM/a.jpg\n"

    ctx = SimpleNamespace(driver=SimpleNamespace(device=SimpleNamespace(shell=shell)))
    result = asyncio.run(runner.recent_media(ctx=ctx))
    assert result.success and "a.jpg size=10 md5=abc123" in result.summary
    assert "DUPLICATE" not in result.summary
    assert "date_added>=" in calls[1] and "media_type IN (1,3)" in calls[1]
    # Listing the same file again is not a duplicate of itself.
    again = asyncio.run(runner.recent_media(ctx=ctx))
    assert "DUPLICATE" not in again.summary


def gallery_shell(files):
    """Fake device shell whose gallery holds ``files`` as (name, md5) pairs."""

    async def shell(command):
        if command == "date +%s":
            return "1000\n"
        if command.startswith("content query"):
            return "".join(
                f"Row: {i} _display_name={name}, _size=10, date_added=999, "
                f"_data=/storage/emulated/0/DCIM/{name}\n"
                for i, (name, _) in enumerate(files)
            )
        name = command.split("/")[-1].rstrip("'")
        return dict(files)[name] + "  path\n"

    return shell


def test_recent_media_flags_identical_bytes_saved_earlier_in_the_task(tmp_path):
    runner, *_ = _runner(tmp_path)
    runner.media_seen = {"same": "first.jpg"}
    ctx = SimpleNamespace(
        driver=SimpleNamespace(
            device=SimpleNamespace(shell=gallery_shell([("second.jpg", "same")]))
        )
    )
    result = asyncio.run(runner.recent_media(ctx=ctx))
    assert "second.jpg size=10 md5=same DUPLICATE of first.jpg" in result.summary


def test_an_undeclared_tap_that_adds_a_gallery_file_is_recorded(tmp_path):
    runner, events, policy, provider = _runner(tmp_path)
    runner.registry.tools["recent_media"] = object()
    runner.ctx.driver.device = SimpleNamespace(shell=gallery_shell([("first.jpg", "same")]))
    # Jev taps 「儲存」 without declaring an effect.
    runner.pending = Selection("jev", [ToolCall("click", {"index": 9})], snapshot=ui_state(MENU))
    runner.task_state.observe(ui_state(MENU))
    runner.record_execution(
        ToolCall("click", {"index": 9}), ActionResult(True, "ok"), ui_state(MENU)
    )
    asyncio.run(runner.observe_media_effect())
    entry = runner.effects.entries[0]
    assert entry.source == "observed" and entry.status == "confirmed"
    assert entry.evidence == ("P1",) and "DUPLICATE" not in entry.description

    # The same bytes saved again under a new name are flagged as a duplicate.
    runner.ctx.driver.device.shell = gallery_shell([("second.jpg", "same")])
    runner.pending = Selection("jev", [ToolCall("click", {"index": 9})], snapshot=ui_state(MENU))
    runner.record_execution(
        ToolCall("click", {"index": 9}), ActionResult(True, "ok"), ui_state(MENU)
    )
    asyncio.run(runner.observe_media_effect())
    assert "DUPLICATE" in runner.effects.entries[1].description
    assert "different item" in runner.last_feedback


def test_verification_starts_with_an_automatic_citable_gallery_check(tmp_path):
    visual = Verdicts(verified("confirmed", ["P1"]))
    runner, *_ = _runner(tmp_path, visual=visual)
    runner.registry.tools["recent_media"] = object()
    runner.ctx.driver.device = SimpleNamespace(shell=gallery_shell([("saved.jpg", "abc")]))
    _execute(runner, ToolCall("click", {"index": 9}), ui_state(MENU), effect="保存")
    asyncio.run(runner.select(GOAL, ui_state(PAGE)))
    feedback = visual.requests[0]["feedback"]
    assert "P1: 1 gallery item(s) added after the action: saved.jpg" in feedback
    entry = runner.effects.entries[0]
    assert entry.status == "confirmed" and entry.evidence == ("P1",)
    assert entry.context()["checks"][0].startswith("P1: 1 gallery item")


def test_only_device_checked_effects_become_completion_evidence():
    from mobilerun.agent.decision.visual_fallback import _ledger_evidence

    evidence = _ledger_evidence(
        {
            "side_effects": [
                {"id": "E1", "status": "confirmed", "evidence": ["P1"]},
                {"id": "E2", "status": "confirmed", "evidence": ["O5"]},
                {"id": "E3", "status": "pending", "evidence": []},
            ]
        }
    )
    assert evidence["E1"]["source"] == "device_probe"
    assert evidence["P1"] is evidence["E1"]
    # Model-only verdicts and unverified effects are not completion evidence.
    assert "E2" not in evidence and "E3" not in evidence


@pytest.mark.parametrize("refs,accepted", [(["E1"], True), (["E2"], False)])
def test_completion_review_accepts_a_device_checked_save(refs, accepted):
    from test_persistent_result_evidence import contract
    from test_task_evidence import review

    from mobilerun.agent.decision.contracts import ContractError, parse_grounded_review
    from mobilerun.agent.decision.visual_fallback import _ledger_evidence

    evidence = _ledger_evidence(
        {"side_effects": [{"id": "E1", "status": "confirmed", "evidence": ["P1"]}]}
    )
    evidence["E2"] = {"ref": "E2", "status": "confirmed"}  # a model-only claim
    data = json.loads(review(refs, answer=None))
    data["assessments"][0]["result_refs"] = refs

    def parse():
        return parse_grounded_review(
            json.dumps(data),
            contract("state", "persistent"),
            evidence,
            "O1",
            uploaded_images={},
            derive_review_dependencies=True,
        )

    if accepted:
        assert parse()["status"] == "done"
    else:
        with pytest.raises(ContractError):
            parse()


def test_legacy_autoglm_fallback_name_still_loads_as_visual():
    from mobilerun.config_manager.config_manager import JevConfig

    assert JevConfig(fallback="autoglm").fallback == "visual"
    assert JevConfig(fallback="visual").fallback == "visual"
    with pytest.raises(ValueError):
        JevConfig(fallback="glm")
