"""Longform harness policy and verdict regression tests."""

from types import SimpleNamespace

import pytest

from test_harness import TestSession as HarnessSession, _looks_like_attack_action
from test_long_horizon import (
    SCENARIOS,
    GeminiFlashPlayer,
    Seed,
    _canonical_seed_candidates,
    _determine_verdict,
    _player_action_problem,
    _redact_seed_text,
    _seed_matches,
    _seed_is_fallback,
    _supports_structured_effect_accounting,
    evaluate_canonical_npc_identity,
    evaluate_player_action_quality,
    evaluate_narrator_prose_quality,
    evaluate_tool_coverage,
    evaluate_tool_omission_signals,
)


@pytest.mark.parametrize(
    ("overrides", "expected"),
    [
        ({}, "PASS"),
        ({"run_error": "boom"}, "FAIL"),
        ({"run_complete": False}, "FAIL"),
        ({"orchestrator_failures": [4]}, "FAIL"),
        ({"combat_policy_failed": True}, "FAIL"),
        ({"fallback_turns": [7]}, "INVALID(player-error)"),
        ({"verdict_trusted": False}, "UNTRUSTED"),
        ({"passed": 5}, "FAIL"),
        ({"total": 0, "passed": 0}, "FAIL"),
    ],
)
def test_only_complete_trusted_assertion_run_passes(overrides, expected):
    values = {
        "run_error": None,
        "run_complete": True,
        "orchestrator_failures": [],
        "combat_policy_failed": False,
        "fallback_turns": [],
        "verdict_trusted": True,
        "passed": 6,
        "total": 6,
    }
    values.update(overrides)

    assert _determine_verdict(**values) == expected


@pytest.mark.asyncio
async def test_simulated_victory_uses_canonical_teardown_and_records_outcome():
    calls = []

    async def end_combat(channel_id):
        calls.append(channel_id)
        return True

    session = HarnessSession(combat_policy="simulate_victory")
    session.manager = SimpleNamespace(end_combat=end_combat)
    session.turn_number = 3

    intervention = await session._resolve_test_combat()

    assert calls == [session.channel_id]
    assert intervention == {
        "turn": 3,
        "policy": "simulate_victory",
        "outcome": "player_victory",
        "teardown_complete": True,
    }
    assert session.all_issues == []


@pytest.mark.asyncio
async def test_simulated_victory_restores_zero_hp_player_agency(monkeypatch):
    hp_updates = []
    death_updates = []
    death_saves = SimpleNamespace(successes=1, failures=2)
    death_saves.reset = lambda: (
        setattr(death_saves, "successes", 0),
        setattr(death_saves, "failures", 0),
    )
    character = SimpleNamespace(
        id="kael-id",
        hp=SimpleNamespace(current=0, maximum=11, temporary=0),
        death_saves=death_saves,
    )

    class Repo:
        async def update_hp(self, character_id, current, temporary):
            hp_updates.append((character_id, current, temporary))

        async def update_death_saves(self, character_id, successes, failures):
            death_updates.append((character_id, successes, failures))

    game_session = SimpleNamespace(
        get_player_character=lambda user_id: character,
    )
    session = HarnessSession(combat_policy="simulate_victory")
    session.manager = SimpleNamespace(get_session=lambda channel_id: game_session)
    session.turn_number = 79

    async def get_repo():
        return Repo()

    monkeypatch.setattr("test_harness.get_character_repo", get_repo)

    intervention = await session._restore_test_player_agency()

    assert intervention["outcome"] == "player_agency_restored"
    assert intervention["hp_after"] == 11
    assert character.hp.current == 11
    assert hp_updates == [("kael-id", 11, 0)]
    assert death_updates == [("kael-id", 0, 0)]


@pytest.mark.asyncio
async def test_fail_combat_policy_records_hard_issue_after_teardown():
    async def end_combat(channel_id):
        return True

    session = HarnessSession(combat_policy="fail")
    session.manager = SimpleNamespace(end_combat=end_combat)

    intervention = await session._resolve_test_combat()

    assert intervention["outcome"] == "failed"
    assert [issue.category for issue in session.all_issues] == ["combat_policy"]


def test_unknown_combat_policy_is_rejected():
    with pytest.raises(ValueError, match="Unknown combat policy"):
        HarnessSession(combat_policy="silently_ignore")


@pytest.mark.parametrize("verb", ["attack", "strike", "hit", "stab", "slash", "shoot", "fight", "kill"])
def test_attack_detection_matches_standalone_attack_verbs(verb):
    assert _looks_like_attack_action(f"I {verb} the goblin")


def test_attack_detection_does_not_match_stabilize():
    assert not _looks_like_attack_action(
        "I examine the courier's badge, then attempt to stabilize him."
    )


def test_deep_soak_has_long_gap_and_multi_turn_callback():
    scenario = SCENARIOS["deep_emergent_callback"]

    assert scenario.total_turns == 80
    assert scenario.seed_pick_after_turn == 8
    assert [(p.name, p.turn_range) for p in scenario.phases] == [
        ("explore", (1, 8)),
        ("filler", (9, 70)),
        ("callback", (71, 80)),
    ]


def test_seeded_soak_has_complete_multi_act_trajectory_and_blank_slate_seed():
    scenario = SCENARIOS["deep_seeded_callback"]

    assert scenario.total_turns == 80
    assert scenario.memory_silence_range == (15, 70)
    assert scenario.creativity_gate is True
    assert scenario.tool_coverage_gate is True
    assert scenario.required_seed_type == "npc"
    assert scenario.world_setting and "Veyr" in scenario.world_setting

    covered = []
    for phase in scenario.phases:
        covered.extend(range(phase.turn_range[0], phase.turn_range[1] + 1))
    assert covered == list(range(1, 81))

    fixed_seed = Seed(
        type="item",
        name="living brass compass",
        reason="from fixed premise",
        chosen_after_turn=8,
    )
    emergent_seed = Seed(
        type="npc",
        name="Sable Quill",
        reason="an archivist invented during play",
        chosen_after_turn=8,
    )
    assert _seed_is_fallback(fixed_seed, scenario) is True
    assert _seed_is_fallback(Seed(
        type="item",
        name="the compass",
        reason="short alias of fixed premise item",
        chosen_after_turn=8,
    ), scenario) is True
    assert _seed_is_fallback(emergent_seed, scenario) is False


def test_generic_npc_label_is_untrusted_but_proper_name_is_valid():
    scenario = SCENARIOS["targeted_relevance_callback"]

    assert _seed_is_fallback(Seed("npc", "Beggar", "mysterious", 8), scenario) is True
    assert _seed_is_fallback(Seed("npc", "Unseen Woman", "mysterious", 8), scenario) is True
    assert _seed_is_fallback(Seed("npc", "Sera", "mysterious", 8), scenario) is False


def test_generic_place_label_is_untrusted_but_distinct_place_is_valid():
    scenario = SCENARIOS["deep_emergent_callback"]

    assert _seed_is_fallback(Seed("place", "depot", "generic", 8), scenario) is True
    assert _seed_is_fallback(Seed("place", "the market square", "generic", 8), scenario) is True
    assert _seed_is_fallback(Seed("place", "Grey Lantern Depot", "named", 8), scenario) is False


def test_canonical_candidates_prefer_named_npcs_and_drop_generic_labels():
    def entity(name, entity_type, **properties):
        return SimpleNamespace(
            name=name,
            entity_type=SimpleNamespace(value=entity_type),
            properties=properties,
        )

    graph = SimpleNamespace(_entities={
        "beggar": entity("Beggar", "npc"),
        "lira": entity("Lira", "npc", alive="false", description="A dead courier."),
        "sera": entity("Sera", "npc", description="A wary Ragpicker."),
        "map": entity("Ashglass Map", "item", description="A brittle route map."),
        "compass": entity("living brass compass", "item"),
    })
    session = SimpleNamespace(knowledge_graph=graph)

    candidates = _canonical_seed_candidates(
        session,
        ["Sera takes the Ashglass Map from the Beggar beside the living brass compass."],
        SCENARIOS["targeted_relevance_callback"],
    )

    assert candidates == [
        {"type": "npc", "name": "Sera", "description": "A wary Ragpicker."}
    ]


def test_targeted_gate_rejects_ambient_item_as_callback_seed():
    graph = SimpleNamespace(_entities={
        "map": SimpleNamespace(
            name="map fragment",
            entity_type=SimpleNamespace(value="item"),
            properties={"description": "A route carried by the player."},
        )
    })

    assert _canonical_seed_candidates(
        SimpleNamespace(knowledge_graph=graph),
        ["I put the map fragment in my pack."],
        SCENARIOS["targeted_relevance_callback"],
    ) == []


def test_targeted_gate_rejects_new_name_for_fixed_opening_courier():
    graph = SimpleNamespace(_entities={
        "cinder": SimpleNamespace(
            name="Cinder Vex",
            entity_type=SimpleNamespace(value="npc"),
            properties={
                "description": (
                    "A masked courier collapsed at Saint Orra's Wake with a "
                    "living brass compass."
                )
            },
        )
    })

    assert _canonical_seed_candidates(
        SimpleNamespace(knowledge_graph=graph),
        ["The masked courier gives her name as Cinder Vex."],
        SCENARIOS["targeted_relevance_callback"],
    ) == []


def test_targeted_gate_covers_buffer_cooloff_strict_silence_and_callback():
    scenario = SCENARIOS["targeted_relevance_callback"]

    assert scenario.total_turns == 30
    assert [(p.name, p.turn_range) for p in scenario.phases] == [
        ("explore", (1, 8)),
        ("washout", (9, 24)),
        ("callback", (25, 30)),
    ]
    assert scenario.seed_pick_after_turn == 8
    assert scenario.memory_silence_range == (15, 24)
    assert scenario.required_seed_type == "npc"


@pytest.mark.parametrize(
    "action",
    [
        'Indeed," I reply,',
        'I ask Bren, "I',
        "I press on, looking",
        "I wait.",
    ],
)
def test_player_action_validator_rejects_truncated_or_passive_fragments(action):
    assert _player_action_problem(action)


def test_player_quality_floor_rewards_varied_consequential_actions():
    actions = [
        "I question the courier about the ash beneath her fingernails, offering water before I ask whom she fears.",
        "I trade my silver clasp for the vendor's cracked memory vial, but insist that he name its former owner.",
        "I confess my future letter to Mara and ask her to burn it if I begin treating people like evidence.",
        "I climb the inverted rain chain, testing whether the compass points toward guilt instead of north.",
        "I promise the frightened clerk safe passage if she shows me who altered the archive ledger.",
        "I place the stolen vial between both factions and demand each explain what returning it would cost.",
        "I sketch the impossible route on my sleeve, deliberately leaving one false turn for anyone following us.",
        "I surrender my childhood song as collateral, making the broker state the bargain where everyone can hear.",
        "I follow Mara into the rib tunnels while marking our route with knots only refugee guides would recognize.",
        "I challenge the Anchor priest to read the erased names aloud before I decide whether his ritual deserves protection.",
        "I give the compass to the child it keeps pointing toward and ask what memory the city took from her.",
        "I refuse the clean escape, choosing instead to carry the witness across the collapsing gravity road myself.",
    ]

    assert all(result.passed for result in evaluate_player_action_quality(actions))


def test_tool_coverage_requires_durable_mutations_not_reference_spam():
    broad = [
        [{"type": "ref_entity"}, {"type": "add_npc"}],
        [{"type": "spawn_object"}, {"type": "change_location"}],
        [{"type": "update_player"}],
        [{"type": "update_entity"}],
        [{"type": "change_location"}],
        [{"type": "update_player"}],
        [{"type": "update_entity"}],
        [{"type": "ref_entity"}],
        [],
        [],
    ]
    narrow = [[{"type": "ref_entity"}]] * 10

    assert all(result.passed for result in evaluate_tool_coverage(broad))
    assert not all(result.passed for result in evaluate_tool_coverage(narrow))

    unreliable = evaluate_tool_coverage(
        broad,
        proposed_by_turn=broad,
        rejected_by_turn=[[{"type": "ref_entity"}]] * len(broad),
    )
    reliability = next(
        result for result in unreliable
        if result.name == "tool_effect_execution_reliability"
    )
    assert reliability.passed is False
    accounting = next(
        result for result in unreliable
        if result.name == "tool_effect_accounting_balanced"
    )
    assert accounting.passed is False

    balanced = evaluate_tool_coverage(
        broad,
        proposed_by_turn=broad,
        rejected_by_turn=[[] for _ in broad],
    )
    accounting = next(
        result for result in balanced
        if result.name == "tool_effect_accounting_balanced"
    )
    assert accounting.passed is True


def test_tool_coverage_budgets_malformed_fail_closed_calls():
    broad = [
        [{"type": "ref_entity"}, {"type": "add_npc"}],
        [{"type": "spawn_object"}, {"type": "change_location"}],
        [{"type": "update_player"}],
        [{"type": "update_entity"}],
        [{"type": "change_location"}],
        [{"type": "update_player"}],
        [{"type": "update_entity"}],
        [{"type": "ref_entity"}],
        [],
        [],
    ]
    healthy_diagnostics = [{} for _ in broad]
    healthy_diagnostics[3] = {
        "tool_followup_structural_errors": 1,
        "tool_repair_structural_errors": 1,
        "tool_invalid_effects_dropped": 0,
        "tool_repair_failed_closed": False,
    }
    results = evaluate_tool_coverage(
        broad,
        proposed_by_turn=broad,
        rejected_by_turn=[[] for _ in broad],
        diagnostics_by_turn=healthy_diagnostics,
    )
    budget = next(
        result for result in results
        if result.name == "tool_structural_failure_budget"
    )
    assert budget.passed is True

    excessive_diagnostics = [{} for _ in broad]
    excessive_diagnostics[1] = {
        "tool_followup_structural_errors": 2,
        "tool_repair_structural_errors": 2,
        "tool_invalid_effects_dropped": 2,
        "tool_repair_failed_closed": True,
    }
    excessive_diagnostics[5] = {
        "tool_followup_structural_errors": 1,
        "tool_repair_structural_errors": 1,
        "tool_invalid_effects_dropped": 1,
        "tool_repair_failed_closed": True,
    }
    results = evaluate_tool_coverage(
        broad,
        proposed_by_turn=broad,
        rejected_by_turn=[[] for _ in broad],
        diagnostics_by_turn=excessive_diagnostics,
    )
    budget = next(
        result for result in results
        if result.name == "tool_structural_failure_budget"
    )
    assert budget.passed is False


def test_tool_coverage_budgets_policy_suppressed_attempts_separately():
    broad = [
        [{"type": "ref_entity"}, {"type": "add_npc"}],
        [{"type": "spawn_object"}, {"type": "change_location"}],
        [{"type": "update_player"}],
        [{"type": "update_entity"}],
        [{"type": "change_location"}],
        [{"type": "update_player"}],
        [{"type": "update_entity"}],
        [{"type": "ref_entity"}],
        [],
        [],
    ]
    diagnostics = [{} for _ in broad]
    diagnostics[2] = {"tool_policy_suppressed_effects": 1}

    results = evaluate_tool_coverage(
        broad,
        proposed_by_turn=broad,
        rejected_by_turn=[[] for _ in broad],
        diagnostics_by_turn=diagnostics,
    )
    budget = next(
        result for result in results
        if result.name == "tool_policy_suppression_budget"
    )
    assert budget.passed is True

    diagnostics[5] = {"tool_policy_suppressed_effects": 2}
    results = evaluate_tool_coverage(
        broad,
        proposed_by_turn=broad,
        rejected_by_turn=[[] for _ in broad],
        diagnostics_by_turn=diagnostics,
    )
    budget = next(
        result for result in results
        if result.name == "tool_policy_suppression_budget"
    )
    assert budget.passed is False


def test_tool_coverage_fails_unmet_runtime_effect_obligations():
    broad = [
        [{"type": "ref_entity"}, {"type": "add_npc"}],
        [{"type": "spawn_object"}, {"type": "change_location"}],
        [{"type": "update_player"}],
        [{"type": "update_entity"}],
        [{"type": "change_location"}],
        [{"type": "update_player"}],
        [{"type": "update_entity"}],
        [{"type": "ref_entity"}],
        [],
        [],
    ]
    diagnostics = [{} for _ in broad]
    diagnostics[3] = {
        "effect_obligation_missing_final": ["update_entity"],
    }
    results = evaluate_tool_coverage(
        broad,
        proposed_by_turn=broad,
        rejected_by_turn=[[] for _ in broad],
        diagnostics_by_turn=diagnostics,
    )

    obligation_gate = next(
        result for result in results
        if result.name == "runtime_effect_obligations_met"
    )
    assert obligation_gate.passed is False
    assert "update_entity" in obligation_gate.detail


def test_pgi_blocked_turn_does_not_disable_structured_effect_gates():
    records = {
        1: {"effects": {"proposed": [], "executed": [], "rejected": []}},
        2: {"pgi_blocked": True},
        3: {"effects": {"proposed": [], "executed": [], "rejected": []}},
    }
    log = SimpleNamespace(get=lambda turn: records.get(turn))

    assert _supports_structured_effect_accounting(log, [1, 2, 3]) is True

    records[2]["effects"] = []  # legacy flat-list format cannot prove balance
    assert _supports_structured_effect_accounting(log, [1, 2, 3]) is False


def test_tool_omission_audit_catches_absent_tools_without_flagging_anonymous_npcs():
    records = [
        (1, {
            "narrator_response": {"raw": "Mara Venn enters beside the cloaked woman."},
            "state_delta": {"delta": {
                "location_change": "Mirror Market",
                "new_npcs": [
                    {"name": "Mara Venn"},
                    {"name": "the cloaked woman"},
                ],
                "npc_updates": [
                    {"id": "mara", "disposition": "friendly"},
                    {"id": "mara", "add_aliases": ["the broker"]},
                    {"id": "mara", "description": "Restated full description"},
                ],
            }},
            "effects": {"proposed": [
                {"type": "change_location"},
                {"type": "update_entity"},
            ]},
        }),
    ]

    results = evaluate_tool_omission_signals(records)
    coverage = next(r for r in results if r.name == "tool_omission_signal_coverage")

    assert coverage.passed is False
    assert "add_npc(Mara Venn)" in coverage.detail
    assert "cloaked woman" not in coverage.detail


def test_tool_omission_audit_skips_sub_scene_location_refinement():
    """A base place vs its qualified sub-scene owes no change_location."""
    records = [
        (15, {
            "narrator_response": {"raw": (
                "You guide him out of the alley, into the grey morning "
                "light of the Tallow Rows."
            )},
            "world_state": {"before": "location: Tallow Rows alley"},
            "state_delta": {"delta": {"location_change": "Tallow Rows"}},
            "effects": {"proposed": [{"type": "ref_entity", "ref_id": "x",
                                      "ref_alias": "alley"}]},
        }),
        # A genuine move between unrelated named places is still owed.
        (25, {
            "narrator_response": {"raw": (
                "You leave the shop and step into the Tallow Rows."
            )},
            "world_state": {"before": "location: Harrow's Drippings - Upstairs"},
            "state_delta": {"delta": {"location_change": "Tallow Rows"}},
            "effects": {"proposed": []},
        }),
    ]

    results = evaluate_tool_omission_signals(records)
    coverage = next(
        r for r in results if r.name == "tool_omission_signal_coverage"
    )

    assert coverage.passed is False
    assert "T15" not in coverage.detail
    assert "T25 change_location(Tallow Rows)" in coverage.detail


def test_reference_grounding_skips_numbered_generic_and_title_aliases():
    """Live gate false positives from soak 20260723_122931.

    T4: 'acolyte 1' was ref'd with the newly revealed name Pell — the
    catalog snapshot is recorded before Step 4's promotion, so the target
    still wore the numbered generic label. T12: alias 'Brother' is a
    monastic address of Pell, not a competing identity claim.
    """
    records = [
        (4, {
            "narrator_response": {"raw": "Pell bows. Brother Pell, they call him."},
            "state_delta": {"delta": {}},
            "knowledge_graph": {"catalog_entities": [
                {"id": "acolyte-1", "name": "acolyte 1", "type": "npc",
                 "aliases": []},
                {"id": "pell-id", "name": "Pell", "type": "npc",
                 "aliases": []},
            ]},
            "effects": {"proposed": [
                {"type": "ref_entity", "ref_id": "acolyte-1",
                 "ref_alias": "Pell"},
                {"type": "ref_entity", "ref_id": "pell-id",
                 "ref_alias": "Brother"},
            ]},
        }),
    ]

    results = evaluate_tool_omission_signals(records)
    grounding = next(
        r for r in results if r.name == "tool_reference_identity_grounding"
    )
    assert grounding.passed, grounding.detail


def test_tool_omission_audit_skips_store_rejected_updates():
    """A delta update the store rejected mutated nothing and owes no tool.

    Live case (run 20260723_002014, T26): the extractor emitted inventory
    updates for "the baker", a background vendor never present in the
    roster; both updates were rejected with "NPC not found for update".
    """
    records = [
        (26, {
            "narrator_response": {"raw": (
                "The baker gives you a short nod, pockets the wax sliver, "
                "and returns to her work."
            )},
            "state_delta": {
                "delta": {"npc_updates": [
                    {"name": "the baker", "remove_inventory": ["wax sliver"]},
                    {"new_name": "the baker", "remove_inventory": ["crumb"]},
                ]},
                "rejections": [
                    "NPC not found for update: id=None name='the baker'",
                    "NPC not found for update: id=None name=''",
                ],
            },
            "effects": {"proposed": [{"type": "ref_entity", "ref_id": "x",
                                      "ref_alias": "baker"}]},
        }),
    ]

    results = evaluate_tool_omission_signals(records)
    coverage = next(
        r for r in results if r.name == "tool_omission_signal_coverage"
    )
    assert "update_entity" not in coverage.detail


def test_tool_omission_audit_defers_to_ref_identity_and_ignores_inferred_importance():
    records = [
        (3, {
            "state_delta": {"delta": {
                "new_npcs": [{"id": "mira-id", "name": "Mira"}],
                "npc_updates": [{"id": "mira-id", "important": True}],
            }},
            "effects": {"proposed": [
                {
                    "type": "ref_entity",
                    "ref_id": "old-placeholder-id",
                    "ref_alias": "Mira",
                },
            ]},
        }),
    ]

    results = evaluate_tool_omission_signals(records)
    coverage = next(r for r in results if r.name == "tool_omission_signal_coverage")

    assert coverage.passed is False  # no independent mutations remain to exercise
    assert "missing=[]" in coverage.detail


def test_tool_omission_audit_recognizes_slug_reference_as_named_identity():
    records = [
        (11, {
            "narrator_response": {"raw": "Cinder Vex raises her lantern."},
            "state_delta": {"delta": {"new_npcs": [{
                "id": "fresh-id",
                "name": "Cinder Vex",
            }]}},
            "effects": {"proposed": [{
                "type": "ref_entity",
                "ref_id": "cinder-vex",
            }]},
        }),
    ]

    coverage = next(
        result
        for result in evaluate_tool_omission_signals(records)
        if result.name == "tool_omission_signal_coverage"
    )

    assert coverage.passed is False  # identity ref removes the only mutation signal
    assert "missing=[]" in coverage.detail


def test_tool_omission_audit_does_not_recreate_prior_catalog_npc_projection():
    lira = {
        "id": "lira-id",
        "name": "Lira Venn",
        "type": "npc",
        "aliases": [],
    }
    records = [
        (26, {
            "narrator_response": {"raw": "The apothecary mentions Lira Venn."},
            "knowledge_graph": {"catalog_entities": [lira]},
            "state_delta": {"delta": {}},
            "effects": {"proposed": []},
        }),
        (27, {
            "narrator_response": {"raw": "Lira Venn is still missing with the old vial."},
            "knowledge_graph": {"catalog_entities": [lira]},
            "state_delta": {"delta": {
                "new_npcs": [{"id": "lira-id", "name": "Lira Venn"}],
                "npc_updates": [{"id": "lira-id", "add_inventory": ["old vial"]}],
            }},
            "effects": {"proposed": [{"type": "update_entity"}]},
        }),
    ]

    coverage = next(
        result
        for result in evaluate_tool_omission_signals(records)
        if result.name == "tool_omission_signal_coverage"
    )

    assert coverage.passed is True
    assert "add_npc(Lira Venn)" not in coverage.detail


def test_canonical_npc_identity_gate_rejects_duplicate_proper_names_only():
    result = evaluate_canonical_npc_identity([
        {"id": "mira-1", "type": "npc", "name": "Mira"},
        {"id": "mira-2", "type": "npc", "name": "Mira"},
        {"id": "guard-1", "type": "npc", "name": "Guard"},
        {"id": "guard-2", "type": "npc", "name": "Guard"},
    ])

    assert result.passed is False
    assert "Mira" in result.detail
    assert "Guard" not in result.detail


def test_tool_omission_audit_passes_when_each_independent_signal_was_proposed():
    records = [
        (7, {
            "narrator_response": {
                "raw": "You enter Chainwright Alley, where Orla Hask lies dead."
            },
            "state_delta": {"delta": {
                "location_change": "Chainwright Alley",
                "new_npcs": [{"name": "Orla Hask"}],
                "npc_updates": [{"id": "orla", "alive": False}],
            }},
            "effects": {"proposed": [
                {"type": "change_location"},
                {"type": "add_npc", "npc_name": "Orla Hask"},
                {"type": "update_entity"},
            ]},
        }),
    ]

    assert all(result.passed for result in evaluate_tool_omission_signals(records))


def test_tool_omission_audit_ignores_ungrounded_extractor_location():
    records = [
        (6, {
            "narrator_response": {
                "raw": "Outside, Veyr unfolds ahead as Sorin steps through the archway."
            },
            "state_delta": {"delta": {"location_change": "Saint Orra's Wake"}},
            "world_state": {"before": "turn: 6\n"},
            "effects": {"proposed": [{
                "type": "ref_entity",
                "ref_id": "sorin",
                "ref_alias": "Sorin",
            }]},
            "knowledge_graph": {"catalog_entities": [{
                "id": "sorin",
                "name": "Sorin",
                "type": "npc",
                "aliases": [],
            }]},
        }),
    ]

    results = evaluate_tool_omission_signals(records)
    coverage = next(r for r in results if r.name == "tool_omission_signal_coverage")

    assert coverage.passed is False  # no independent mutation remains to exercise
    assert "missing=[]" in coverage.detail


def test_tool_omission_audit_catches_strong_named_npc_prose_cue():
    records = [
        (41, {
            "player": "Kael Windrunner",
            "narrator_response": {
                "raw": "Elara's eyes fix on the vial as she reaches for it."
            },
            "knowledge_graph": {"catalog_entities": []},
            "state_delta": {"delta": {}},
            "effects": {"proposed": [{"type": "update_player"}]},
        }),
    ]

    results = evaluate_tool_omission_signals(records)
    coverage = next(r for r in results if r.name == "tool_omission_signal_coverage")

    assert coverage.passed is False
    assert "add_npc(Elara; prose cue)" in coverage.detail


def test_tool_reference_alias_must_be_grounded_in_narrator_prose():
    records = [
        (4, {
            "narrator_response": {"raw": "Elara closes the ledger."},
            "state_delta": {"delta": {"location_change": "The Depot"}},
            "effects": {"proposed": [
                {"type": "change_location"},
                {"type": "ref_entity", "ref_alias": "Lys"},
            ]},
        }),
    ]

    results = evaluate_tool_omission_signals(records)
    grounding = next(r for r in results if r.name == "tool_reference_alias_grounding")

    assert grounding.passed is False
    assert "ref_entity(Lys)" in grounding.detail


def test_tool_reference_alias_cannot_be_only_an_article():
    records = [
        (4, {
            "narrator_response": {"raw": "You enter the Rusty Hinge."},
            "state_delta": {"delta": {}},
            "effects": {"proposed": [{
                "type": "ref_entity",
                "ref_id": "old-alley",
                "ref_alias": "the",
            }]},
        }),
    ]

    grounding = next(
        result
        for result in evaluate_tool_omission_signals(records)
        if result.name == "tool_reference_alias_grounding"
    )

    assert grounding.passed is False
    assert "ref_entity(the)" in grounding.detail


def test_tool_reference_alias_must_belong_to_claimed_catalog_identity():
    records = [
        (29, {
            "narrator_response": {
                "raw": "Elena Voss watches while the Tollman closes the gate."
            },
            "state_delta": {"delta": {}},
            "effects": {"proposed": [{
                "type": "ref_entity",
                "ref_id": "elena-id",
                "ref_alias": "the Tollman",
            }]},
            "knowledge_graph": {"catalog_entities": [
                {
                    "id": "elena-id",
                    "name": "Elena Voss",
                    "type": "npc",
                    "aliases": [],
                },
                {
                    "id": "tollman-id",
                    "name": "the Tollman",
                    "type": "npc",
                    "aliases": [],
                },
            ]},
        }),
    ]

    identity = next(
        result
        for result in evaluate_tool_omission_signals(records)
        if result.name == "tool_reference_identity_grounding"
    )

    assert identity.passed is False
    assert "Elena Voss <- the Tollman" in identity.detail


def test_tool_reference_identity_grounding_allows_canonical_partial_alias():
    records = [
        (5, {
            "narrator_response": {"raw": "You cross the Market at dusk."},
            "state_delta": {"delta": {}},
            "effects": {"proposed": [{
                "type": "ref_entity",
                "ref_id": "market-id",
                "ref_alias": "Market",
            }]},
            "knowledge_graph": {"catalog_entities": [{
                "id": "market-id",
                "name": "Market Ring south tier",
                "type": "location",
                "aliases": [],
            }]},
        }),
    ]

    identity = next(
        result
        for result in evaluate_tool_omission_signals(records)
        if result.name == "tool_reference_identity_grounding"
    )

    assert identity.passed is True


@pytest.mark.asyncio
async def test_player_prompt_carries_private_continuity_and_structured_contract():
    scenario = SCENARIOS["deep_seeded_callback"]
    captured = {}

    class FakeClient:
        model = "gemini-test"

        async def chat(self, **kwargs):
            captured.update(kwargs)
            return SimpleNamespace(
                content=(
                    '{"action":"I shelter the courier beneath my cloak and ask who taught her '
                    'to recognize my handwriting.","continuity":"Kael protects the courier and '
                    'suspects the future letter was meant to manipulate his loyalty."}'
                ),
                prompt_tokens=10,
                completion_tokens=20,
                cache_read_tokens=0,
                cache_write_tokens=0,
            )

    player = GeminiFlashPlayer.__new__(GeminiFlashPlayer)
    player.client = FakeClient()
    player.scenario = scenario
    player.history_window = 10
    player.history = []
    player.character_state = "Kael begins wary of institutions and protective of the courier."
    player._record_usage = lambda response, elapsed: None

    action = await player.next_action("", scenario.phases[0], None)

    assert action.startswith("I shelter the courier")
    assert "future letter" in player.character_state
    assert captured["json_mode"] is True
    assert captured["json_schema"]["required"] == ["action", "continuity"]
    assert "PRIVATE CONTINUITY NOTE" in captured["messages"][0]["content"]
    assert "Player-visible opening situation" in captured["messages"][1]["content"]


@pytest.mark.asyncio
async def test_player_uses_local_fallback_when_gemini_returns_no_content():
    scenario = SCENARIOS["targeted_relevance_callback"]

    class SafetyBlockedGemini:
        model = "gemini-test"

        def __init__(self):
            self.calls = 0

        async def chat(self, **kwargs):
            self.calls += 1
            return SimpleNamespace(
                content="",
                finish_reason="SAFETY",
                model=self.model,
                prompt_tokens=10,
                completion_tokens=0,
                cache_read_tokens=0,
                cache_write_tokens=0,
            )

    class LocalPlayer:
        model = "qwen3.5:9b"

        def __init__(self):
            self.calls = 0

        async def chat(self, **kwargs):
            self.calls += 1
            assert kwargs["think"] is False
            return SimpleNamespace(
                content=(
                    '{"action":"I wedge the brass compass beneath the gate and ask the guard '
                    'what price would make them let me pass.","continuity":"Kael risks the '
                    'compass to force a bargain and suspects the guard values leverage."}'
                ),
                finish_reason="stop",
                model=self.model,
                prompt_tokens=20,
                completion_tokens=30,
                cache_read_tokens=0,
                cache_write_tokens=0,
            )

    primary = SafetyBlockedGemini()
    fallback = LocalPlayer()
    player = GeminiFlashPlayer.__new__(GeminiFlashPlayer)
    player.client = primary
    player.fallback_client = fallback
    player.scenario = scenario
    player.history_window = 10
    player.history = []
    player.character_state = "Kael distrusts institutions and seeks leverage."
    player._record_usage = lambda response, elapsed: None

    action = await player.next_action("", scenario.phases[0], None)

    assert action.startswith("I wedge the brass compass")
    assert primary.calls == 1
    assert fallback.calls == 1
    assert player.last_provider_fallbacks == 1
    assert player.last_regenerations == 0


@pytest.mark.asyncio
async def test_seed_picker_retries_forbidden_premise_candidate():
    scenario = SCENARIOS["deep_seeded_callback"]
    responses = iter([
        SimpleNamespace(
            content=(
                '{"type":"item","name":"the compass",'
                '"reason":"Its purposeful motion makes it memorable."}'
            ),
            prompt_tokens=10,
            completion_tokens=10,
            cache_read_tokens=0,
            cache_write_tokens=0,
        ),
        SimpleNamespace(
            content=(
                '{"type":"npc","name":"Sera Vex",'
                '"reason":"Her missing mother and broken-chain insignia create an unresolved promise."}'
            ),
            prompt_tokens=10,
            completion_tokens=10,
            cache_read_tokens=0,
            cache_write_tokens=0,
        ),
    ])
    calls = []

    class FakeClient:
        model = "gemini-test"

        async def chat(self, **kwargs):
            calls.append(kwargs)
            return next(responses)

    player = GeminiFlashPlayer.__new__(GeminiFlashPlayer)
    player.client = FakeClient()
    player._record_usage = lambda response, elapsed: None
    narration = [
        "At the Glass Archive, the compass turns in Kael's hand while Sera Vex "
        "lowers her curved blade. The broken-chain insignia belonged to her "
        "mother, who vanished forty years ago."
    ]

    candidates = [
        {"type": "npc", "name": "Sera Vex", "description": "A blade-bearing stranger."},
        {"type": "item", "name": "broken-chain insignia", "description": "A family token."},
    ]
    seed = await player.pick_seed(narration, scenario, candidates)

    assert seed.name == "Sera Vex"
    assert len(calls) == 2
    assert "PREVIOUS CANDIDATE WAS REJECTED" in calls[1]["messages"][1]["content"]
    assert calls[1]["json_schema"]["required"] == ["type", "name", "reason"]


@pytest.mark.asyncio
async def test_seed_picker_skips_model_when_graph_has_one_candidate():
    scenario = SCENARIOS["targeted_relevance_callback"]

    class FailIfCalled:
        async def chat(self, **kwargs):
            raise AssertionError("single canonical seed should not require an LLM call")

    player = GeminiFlashPlayer.__new__(GeminiFlashPlayer)
    player.client = FailIfCalled()
    player._record_usage = lambda response, elapsed: None
    seed = await player.pick_seed(
        ["Elara Venn watches the upward rain."],
        scenario,
        [{"type": "npc", "name": "Elara Venn", "description": "Keeper of the shrine."}],
    )

    assert seed.name == "Elara Venn"
    assert seed.reason == "Keeper of the shrine."


@pytest.mark.asyncio
async def test_forced_seed_setup_action_skips_player_model_and_updates_history():
    class FailIfCalled:
        async def chat(self, **kwargs):
            raise AssertionError("forced setup action should not call the player model")

    player = GeminiFlashPlayer.__new__(GeminiFlashPlayer)
    player.client = FailIfCalled()
    player.history = [("I inspect the compass.", "")]
    player.last_regenerations = 2
    forced = "I ask the guard for their exact name."

    action = await player.next_action(
        "A guard approaches.",
        SCENARIOS["targeted_relevance_callback"].phases[0],
        None,
        forced_action=forced,
    )

    assert action == forced
    assert player.history == [
        ("I inspect the compass.", "A guard approaches."),
        (forced, ""),
    ]
    assert player.last_regenerations == 0


def test_seed_matching_and_redaction_normalize_hyphenation():
    seed = Seed(
        type="item",
        name="the cracked-bell emblem",
        reason="emergent mystery",
        chosen_after_turn=8,
    )

    assert _seed_matches("I sketch the cracked bell on paper.", "cracked-bell emblem", "cracked-bell")
    redacted = _redact_seed_text(
        "I carry the cracked-bell emblem and keep investigating the cracked bell.",
        seed,
    )
    assert "cracked" not in redacted.lower()
    assert redacted.count("[sealed callback detail]") == 2

    ring = Seed(
        type="item",
        name="silver ring",
        reason="emergent clue",
        chosen_after_turn=8,
    )
    assert _seed_matches("I ask Corvin to decipher the ring.", "silver ring", "silver")
    assert "ring" not in _redact_seed_text("Corvin examines the ring.", ring).lower()


def test_npc_seed_matching_does_not_treat_shared_surname_as_same_person():
    seed = Seed("npc", "Lira Venn", "emergent witness", 8)

    assert _seed_matches("Lira steps from the archway.", "lira venn", "lira", "npc")
    assert not _seed_matches(
        "Elara Venn steps from the archway.", "lira venn", "lira", "npc"
    )
    redacted = _redact_seed_text(
        "Lira Venn vanished, but Elara Venn remained.", seed
    )
    assert "Lira" not in redacted
    assert "Elara Venn" in redacted


def test_narrator_quality_gate_catches_private_reasoning_leak():
    results = evaluate_narrator_prose_quality([
        (1, "Rain needles the empty street."),
        (2, "Let me check the world state before I narrate this."),
    ])

    assert results[0].passed is False
    assert "T2" in results[0].detail


@pytest.mark.asyncio
async def test_player_regenerates_action_that_leaks_seed_during_washout():
    scenario = SCENARIOS["deep_seeded_callback"]
    seed = Seed(
        type="item",
        name="the cracked-bell emblem",
        reason="emergent mystery",
        chosen_after_turn=8,
    )
    responses = iter([
        SimpleNamespace(
            content=(
                '{"action":"I show Mara the cracked bell emblem and ask why the Archive fears it so much.",'
                '"continuity":"Kael remains focused on the cracked-bell emblem and Mara."}'
            ),
            prompt_tokens=10,
            completion_tokens=10,
            cache_read_tokens=0,
            cache_write_tokens=0,
        ),
        SimpleNamespace(
            content=(
                '{"action":"I offer Mara my climbing rope and ask her to lead us through the flooded cistern quietly.",'
                '"continuity":"Kael trusts Mara provisionally and wants a quiet route through the cistern."}'
            ),
            prompt_tokens=10,
            completion_tokens=10,
            cache_read_tokens=0,
            cache_write_tokens=0,
        ),
    ])
    calls = []

    class FakeClient:
        model = "gemini-test"

        async def chat(self, **kwargs):
            calls.append(kwargs)
            return next(responses)

    player = GeminiFlashPlayer.__new__(GeminiFlashPlayer)
    player.client = FakeClient()
    player.scenario = scenario
    player.history_window = 10
    player.history = [
        (
            "I study the cracked-bell emblem and promise to learn who made it.",
            "The cracked bell glints beneath the Archive lamps.",
        )
    ] * 14
    player.character_state = "Kael is obsessed with the cracked-bell emblem."
    player._record_usage = lambda response, elapsed: None

    action = await player.next_action(
        "Mara asks whether the cracked-bell emblem matters more than the living.",
        next(p for p in scenario.phases if p.name == "mirror_market"),
        seed,
    )

    assert "climbing rope" in action
    assert player.last_regenerations == 1
    assert len(calls) == 2
    first_prompt = "\n".join(m["content"] for m in calls[0]["messages"])
    assert "cracked-bell" not in first_prompt.lower()
    assert "cracked bell" not in first_prompt.lower()
    assert "leaked the sealed callback detail" in calls[1]["messages"][1]["content"]


@pytest.mark.asyncio
async def test_player_seed_is_redacted_during_cooloff_before_strict_silence():
    scenario = SCENARIOS["targeted_relevance_callback"]
    seed = Seed("npc", "Father Orlan", "Knows who opened the window.", 8)
    calls = []

    class FakeClient:
        model = "gemini-test"

        async def chat(self, **kwargs):
            calls.append(kwargs)
            return SimpleNamespace(
                content=(
                    '{"action":"I leave the shrine and follow the acid trail toward the market.",'
                    '"continuity":"Kael follows the visible acid trail into a different district."}'
                ),
                prompt_tokens=10,
                completion_tokens=10,
                cache_read_tokens=0,
                cache_write_tokens=0,
            )

    player = GeminiFlashPlayer.__new__(GeminiFlashPlayer)
    player.client = FakeClient()
    player.scenario = scenario
    player.history_window = 10
    player.history = [("I question Father Orlan.", "Father Orlan closes the window.")] * 8
    player.character_state = "Kael suspects Father Orlan is hiding something."
    player._record_usage = lambda response, elapsed: None

    await player.next_action(
        "Father Orlan turns away.",
        next(phase for phase in scenario.phases if phase.name == "washout"),
        seed,
    )

    prompt = "\n".join(message["content"] for message in calls[0]["messages"])
    assert "father orlan" not in prompt.lower()
    assert "[sealed callback detail]" in prompt
