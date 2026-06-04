"""Integration tests for the orchestrator's single public entrypoint.

These are the test net (REFACTOR_PLAN.md Step 0): they drive
``DMOrchestrator.process_action`` end-to-end with deterministic fake brains
(tests/fakes.py) over a real WorldState / EffectExecutor / tmp DB, and assert on
the **tool-call / effect sequence and the resulting state diff** — never on
narration prose (testing-llm-pipelines.md §2/§4).

``ALLOW_MODEL_REQUESTS`` is flipped off for the whole fixture, so any LLM seam a
test forgets to inject raises loudly instead of doing real provider I/O.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnd_bot.data.database import Database
from dnd_bot.data.repositories.character_repo import CharacterRepository
from dnd_bot.data.repositories.inventory_repo import InventoryRepository
from dnd_bot.data.repositories import character_repo as character_repo_module
from dnd_bot.llm import client as llm_client
from dnd_bot.llm import orchestrator as orchestrator_module
from dnd_bot.llm.orchestrator import DMOrchestrator
from dnd_bot.llm.brains.base import BrainContext
from dnd_bot.llm.effects import EffectType
from dnd_bot.llm.extractors.dedup_judge import get_dedup_judge
from dnd_bot.game.session import GameSession, PlayerInfo
from dnd_bot.game.scene.registry import SceneEntityRegistry
from dnd_bot.game.world_state import WorldState
from dnd_bot.models import (
    AbilityScore, AbilityScores, Character, HitPoints, HitDice, SpellSlots,
)
from dnd_bot.models.npc import SceneEntity, EntityType, Disposition

from tests.fakes import (
    FunctionBrain, ScriptedBrain, brain_router, narration_response, triage_response,
)


def _async_return(value):
    async def _coro():
        return value
    return _coro()


class _Net:
    """Wires fake brains into every LLM seam and drives one turn.

    ``get_llm_client()`` feeds triage + the state/entity extractors + the dedup
    judge — one ``FunctionBrain`` answers all of them (triage by its system
    prompt, ``{}`` no-op for the rest). The narrator is the separate tier seam,
    pinned via the injected ``_narrator_client_factory``.
    """

    def __init__(self, orch, session, character, char_repo, inv_repo):
        self.orch = orch
        self.session = session
        self.character = character
        self.char_repo = char_repo
        self.inv_repo = inv_repo
        self.registry = orch._scene_registry
        self.brain: FunctionBrain | None = None        # triage + extractors
        self.narrator: ScriptedBrain | None = None     # narration

    async def run(self, action, triage, narration=None, player_name=None):
        self.brain = FunctionBrain(brain_router(triage))
        # Every get_llm_client()-backed seam → the one branching fake.
        self.orch.client = self.brain
        self.orch._state_extractor.client = self.brain
        self.orch._entity_extractor.client = self.brain
        get_dedup_judge().client = self.brain
        # Narrator tier seam → a scripted fake (one prose+tools response/turn).
        self.narrator = ScriptedBrain([narration or narration_response("")])
        self.orch._narrator_client_factory = lambda tier: self.narrator

        context = BrainContext(
            campaign_id=self.session.campaign_id,
            session_id=self.session.id,
        )
        return await self.orch.process_action(
            action, player_name or self.character.name, context
        )


@pytest.fixture
async def net(tmp_path: Path, monkeypatch):
    """Tmp DB + seeded wizard + real session/world-state/registry + orchestrator.

    Real provider calls are blocked for the duration; fakes are injected per
    ``_Net.run``.
    """
    db = Database(db_path=tmp_path / "test.db")
    await db.connect()
    await db.execute(
        "INSERT INTO campaign (id, guild_id, name, dm_user_id) VALUES (?, ?, ?, ?)",
        ("camp", 1, "Camp", 1),
    )
    await db.commit()

    char_repo = CharacterRepository(db=db)
    inv_repo = InventoryRepository(db=db)

    wizard = Character(
        discord_user_id=42, campaign_id="camp", name="Elara",
        race_index="elf", class_index="wizard", level=5,
        abilities=AbilityScores(intelligence=18),
        hp=HitPoints(maximum=30, current=30),
        hit_dice=HitDice(die_type=6, total=5, remaining=5),
        spellcasting_ability=AbilityScore.INTELLIGENCE,
        spell_slots=SpellSlots(level_1=(4, 4), level_2=(3, 3)),
        known_spells=["fire-bolt", "magic-missile"],
        prepared_spells=["magic-missile"],
    )
    await char_repo.create(wizard)

    # Point every repo factory the turn touches at the tmp DB. effects.py and
    # the orchestrator import these from different namespaces — patch both.
    monkeypatch.setattr(character_repo_module, "get_character_repo",
                        lambda: _async_return(char_repo))
    monkeypatch.setattr(orchestrator_module, "get_character_repo",
                        lambda: _async_return(char_repo))
    monkeypatch.setattr(orchestrator_module, "get_inventory_repo",
                        lambda: _async_return(inv_repo))

    # Real session graph (no KG → the KG/Chroma bridge steps no-op).
    session = GameSession(id="sess", channel_id=99, guild_id=1, campaign_id="camp")
    session.world_state = WorldState(current_location="Tavern")
    session.add_player(42, "Elara", wizard)

    registry = SceneEntityRegistry(campaign_id="camp", channel_id=99)

    # Block real provider I/O for the whole test; fakes bypass the guard.
    llm_client.set_model_requests_allowed(False)

    orch = DMOrchestrator()
    orch.set_session(session)
    orch.set_scene_registry(registry)

    try:
        yield _Net(orch, session, wizard, char_repo, inv_repo)
    finally:
        llm_client.set_model_requests_allowed(True)
        # Close aiosqlite or its background thread keeps pytest from exiting.
        await db.disconnect()


# ── The three representative trajectories ────────────────────────────────────

@pytest.mark.asyncio
async def test_attack_creature_starts_combat_without_narrating(net):
    """Attack→creature routes to combat init and returns BEFORE the narrator.

    Trajectory: [triage] → combat_triggered. The narrator brain is never
    invoked; that early-return is the contract this turn type protects.
    """
    net.registry.register_entity(SceneEntity(
        name="Goblin", entity_type=EntityType.CREATURE,
        description="A snarling goblin", disposition=Disposition.HOSTILE,
        hostility_score=90,
    ))

    result = await net.run(
        action="I attack the goblin",
        triage=triage_response(
            "attack", target_name="Goblin", is_creature_target=True,
        ),
    )

    assert result.combat_triggered is True
    # Narrator NOT called — combat start skips narration entirely.
    assert net.narrator.calls == []
    # Exactly one brain call: triage (no extractors, since we returned early).
    assert len(net.brain.calls) == 1


@pytest.mark.asyncio
async def test_social_action_narrates_and_tags_entity(net):
    """A social turn narrates and emits a ref_entity tag, mutating no PC state.

    Trajectory: [triage, narrate] → effects == [REF_ENTITY]; HP/slots unchanged.
    """
    result = await net.run(
        action="I greet the barkeep warmly",
        triage=triage_response("social", needs_roll=False),
        narration=narration_response(
            "The barkeep nods, wiping a mug.",
            tool_calls=[{"name": "ref_entity", "arguments": {"entity_id": "barkeep"}}],
        ),
    )

    # Narrator was invoked exactly once; effect sequence is a single ref tag.
    assert len(net.narrator.calls) == 1
    assert [e.effect_type for e in result.proposed_effects] == [EffectType.REF_ENTITY]

    # State diff: nothing changed on the character sheet.
    reloaded = await net.char_repo.get_by_id(net.character.id)
    assert reloaded.hp.current == 30
    assert reloaded.spell_slots.get_slots(1) == (4, 4)


@pytest.mark.asyncio
async def test_cast_spell_consumes_and_persists_a_slot(net):
    """A cast-spell turn narrates and an update_player effect spends a slot.

    Trajectory: [triage, narrate] → effects == [UPDATE_PLAYER]; L1 slot 4→3,
    persisted to the DB (the strongest state-diff assertion).
    """
    result = await net.run(
        action="I cast magic missile at the darkness",
        triage=triage_response("cast_spell", needs_roll=False),
        narration=narration_response(
            "Three darts of force streak into the gloom.",
            tool_calls=[{"name": "update_player", "arguments": {"spell_slot_used": 1}}],
        ),
    )

    assert [e.effect_type for e in result.proposed_effects] == [EffectType.UPDATE_PLAYER]

    # State diff: one level-1 slot consumed and persisted.
    reloaded = await net.char_repo.get_by_id(net.character.id)
    assert reloaded.spell_slots.get_slots(1) == (3, 4)


@pytest.mark.asyncio
async def test_guard_blocks_unmocked_real_calls(net):
    """Safety net itself: a real provider call raises while the guard is armed."""
    from dnd_bot.llm.client import OllamaClient

    with pytest.raises(RuntimeError, match="ALLOW_MODEL_REQUESTS is False"):
        await OllamaClient().chat(messages=[{"role": "user", "content": "hi"}])
