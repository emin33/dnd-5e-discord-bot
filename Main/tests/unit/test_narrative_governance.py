"""Regression coverage for code-owned narrative continuity invariants."""

from types import SimpleNamespace

from dnd_bot.game.world_state import NPCState, WorldState
from dnd_bot.game.identity import (
    identity_keys,
    locations_equivalent,
    resolve_unique_identity,
)
from dnd_bot.llm.continuity import NarrativeGovernance
from dnd_bot.llm.effects import EffectType, EffectValidator, ProposedEffect


def _governance() -> NarrativeGovernance:
    world = WorldState(current_location="Ash Gate")
    world.npcs["bram-id"] = NPCState(
        id="bram-id",
        name="Old Bram",
        aliases=["the ferryman"],
        location="Ash Gate",
        alive=False,
    )
    return NarrativeGovernance.from_world_state(world)


def test_dead_npc_living_actions_and_aliases_are_blocked():
    governance = _governance()

    blocked = [
        "Old Bram smiles and waves you inside.",
        "You see Old Bram waiting beside the gate.",
        'Old Bram: "You took your time."',
        "Old Bram's eyes open.",
        "The ferryman walks out of the fog.",
        "Old Bram is alive and standing at the crossing.",
        '"You took your time," says Old Bram.',
        "At the gate stands Old Bram.",
        "Old Bram is dead, but Old Bram rises and opens the door.",
    ]

    for prose in blocked:
        violations = governance.validate(prose)
        assert len(violations) == 1, prose
        assert violations[0].rule == "dead_npc_cannot_act"
        assert violations[0].entity_id == "bram-id"


def test_explicit_nonliving_frames_and_passive_legacy_are_allowed():
    governance = _governance()

    allowed = [
        "Old Bram's corpse lies beneath the ash.",
        "You remember Old Bram warning you about the gate.",
        "The ghost of Old Bram whispers your name.",
        "In a dream, the ferryman points toward black water.",
        "Old Bram's dagger rests on the table.",
        "The late Old Bram had told you never to ring the bell.",
    ]

    for prose in allowed:
        assert governance.validate(prose) == [], prose


def test_nonliving_exemption_does_not_leak_into_next_sentence():
    governance = _governance()

    violations = governance.validate(
        "You remember Old Bram's last warning. Old Bram enters the room and laughs."
    )

    assert len(violations) == 1
    assert violations[0].excerpt == "Old Bram enters the room"


def test_governance_only_requires_stream_buffering_when_rules_exist():
    assert _governance().requires_buffering is True
    assert NarrativeGovernance.from_world_state(WorldState()).requires_buffering is False


def test_private_model_reasoning_is_blocked_without_campaign_facts():
    governance = NarrativeGovernance()
    leaked = (
        "I need to process this carefully. The player's action says she opens "
        "the gate. Let me check the world state. Let me write the narration."
    )

    violations = governance.validate(leaked)

    assert [violation.rule for violation in violations] == ["meta_reasoning_leak"]
    assert "player-visible story prose" in governance.repair_instruction(violations)
    assert "world state" not in governance.safe_fallback(violations).lower()


def test_character_dialogue_that_merely_reflects_is_not_meta_reasoning():
    governance = NarrativeGovernance()

    assert governance.validate(
        "Elara frowns. \"I need to think carefully about the road ahead.\""
    ) == []


def test_narrator_tool_cannot_authorize_dead_to_alive_transition():
    world = WorldState()
    dead = NPCState(id="bram-id", name="Old Bram", alive=False)
    world.npcs[dead.id] = dead
    validator = EffectValidator(session=SimpleNamespace(world_state=world))

    result = validator.validate(ProposedEffect(
        effect_type=EffectType.UPDATE_ENTITY,
        update_entity_id="bram-id",
        update_status="alive",
    ))

    assert result.valid is False
    assert "authoritative resurrection mechanic" in result.rejection_reason


def test_latest_system_reminder_surfaces_dead_names_and_aliases():
    from dnd_bot.llm.orchestrator import DMOrchestrator

    world = WorldState()
    world.npcs["bram-id"] = NPCState(
        id="bram-id",
        name="Old Bram",
        aliases=["the ferryman"],
        alive=False,
    )
    orchestrator = object.__new__(DMOrchestrator)
    orchestrator._current_session = SimpleNamespace(world_state=world)
    messages = [{"role": "user", "content": "I remember the ferryman."}]

    orchestrator._append_tool_reminder(messages)

    reminder = messages[-1]["content"]
    assert "IMMUTABLE DEAD-NPC FACTS" in reminder
    assert "Old Bram (aliases: the ferryman): DEAD" in reminder
    assert "Never invent a resurrection" in reminder


def test_latest_system_reminder_excludes_living_offscene_npcs():
    from dnd_bot.llm.orchestrator import DMOrchestrator

    world = WorldState(current_location="Archive gate")
    world.npcs["guard-id"] = NPCState(
        id="guard-id", name="Archive Guard", location="Archive gate", alive=True,
    )
    world.npcs["sera-id"] = NPCState(
        id="sera-id", name="Sera Vell", location="Tallow Market",
        important=True, alive=True,
    )
    orchestrator = object.__new__(DMOrchestrator)
    orchestrator._current_session = SimpleNamespace(world_state=world)
    messages = [{"role": "user", "content": "I pause and look around."}]

    orchestrator._append_tool_reminder(messages)

    reminder = messages[-1]["content"]
    assert "Archive Guard" in reminder
    assert "Sera Vell" not in reminder


def test_cross_session_death_catalog_feeds_governance_without_scene_state():
    from dnd_bot.llm.orchestrator import DMOrchestrator

    dead = NPCState(id="bram-id", name="Old Bram", alive=False)
    orchestrator = object.__new__(DMOrchestrator)
    orchestrator._current_session = SimpleNamespace(
        world_state=WorldState(),
        campaign_dead_npcs={dead.id: dead},
    )

    violations = orchestrator._get_narrative_governance().validate(
        "Old Bram walks into the tavern and orders ale."
    )

    assert len(violations) == 1
    assert violations[0].entity_id == "bram-id"


def test_identity_resolution_handles_leading_descriptor_but_abstains_on_collision():
    old_bram = NPCState(id="old", name="Old Bram")
    assert "bram" in identity_keys("Old Bram")
    assert resolve_unique_identity("Bram", [old_bram]) is old_bram

    other_bram = NPCState(id="other", name="Captain Bram")
    assert resolve_unique_identity("Bram", [old_bram, other_bram]) is None


def test_location_equivalence_is_conservative():
    assert locations_equivalent("Ash Gate", "the Ash Gate clearing") is True
    assert locations_equivalent("Copper Finch", "The Copper Finch") is True
    assert locations_equivalent("tavern", "back room of the tavern") is False
    assert locations_equivalent("Ash Gate", "Copper Finch") is False


def test_state_delta_cannot_reintroduce_dead_npc_as_living():
    from dnd_bot.game.world_state import StateDelta
    from dnd_bot.llm.orchestrator import _drop_dead_npc_reintroductions

    dead = NPCState(id="bram-id", name="Old Bram", alive=False)
    delta = StateDelta(new_npcs=[
        NPCState(name="Bram"),
        NPCState(name="Sable Quill"),
    ])

    rejections = _drop_dead_npc_reintroductions(delta, [dead])

    assert rejections == ["Dead NPC cannot be reintroduced as living: Old Bram"]
    assert [npc.name for npc in delta.new_npcs] == ["Sable Quill"]


def test_add_npc_cannot_reintroduce_cross_session_dead_identity():
    dead = NPCState(id="bram-id", name="Old Bram", alive=False)
    validator = EffectValidator(session=SimpleNamespace(
        world_state=WorldState(),
        campaign_dead_npcs={dead.id: dead},
    ))

    result = validator.validate(ProposedEffect(
        effect_type=EffectType.ADD_NPC,
        npc_name="Bram",
    ))

    assert result.valid is False
    assert "authoritative resurrection mechanic" in result.rejection_reason
