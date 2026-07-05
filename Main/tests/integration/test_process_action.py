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
from dnd_bot.llm.effects import EffectExecutor, EffectType, ProposedEffect
from dnd_bot.llm.extractors.dedup_judge import get_dedup_judge
from dnd_bot.game.session import GameSession, PlayerInfo
from dnd_bot.game.scene.registry import SceneEntityRegistry
from dnd_bot.game.world_state import WorldState
from dnd_bot.models import (
    AbilityScore, AbilityScores, Character, HitPoints, HitDice, InventoryItem,
    SpellSlots,
)
from dnd_bot.models.npc import SceneEntity, EntityType, Disposition

from tests.fakes import (
    FunctionBrain, ScriptedBrain, brain_router, narration_response, triage_response,
)


def _async_return(value):
    async def _coro():
        return value
    return _coro()


async def _async_none(*args, **kwargs):
    return None


class _Net:
    """Wires fake brains into every LLM seam and drives one turn.

    ``get_llm_client()`` feeds triage + the state/entity extractors + the dedup
    judge — one ``FunctionBrain`` answers all of them (triage by its system
    prompt, ``{}`` no-op for the rest). The narrator is the separate tier seam,
    pinned via the injected ``_narrator_client_factory``.
    """

    def __init__(self, orch, session, character, char_repo, inv_repo, monkeypatch):
        self.orch = orch
        self.session = session
        self.character = character
        self.char_repo = char_repo
        self.inv_repo = inv_repo
        self.monkeypatch = monkeypatch
        self.registry = orch._scene_registry
        self.brain: FunctionBrain | None = None        # triage + extractors
        self.narrator: ScriptedBrain | None = None     # narration

    async def run(self, action, triage, narration=None, player_name=None):
        self.brain = FunctionBrain(brain_router(triage))
        # Every get_llm_client()-backed seam → the one branching fake. The
        # extractors and the dedup judge are process-wide singletons, so they
        # go through monkeypatch — teardown restores the real clients instead
        # of leaking the fake into later tests.
        self.orch.client = self.brain
        self.monkeypatch.setattr(self.orch._state_extractor, "client", self.brain)
        self.monkeypatch.setattr(self.orch._entity_extractor, "client", self.brain)
        self.monkeypatch.setattr(get_dedup_judge(), "client", self.brain)
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
        yield _Net(orch, session, wizard, char_repo, inv_repo, monkeypatch)
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
async def test_skill_check_rolls_and_narrates_outcome(net):
    """A skill check rolls a d20 and narrates the outcome (a 2nd narration path).

    Trajectory: [triage(roll)] → one dice roll → [narrate_outcome] → effects.
    The roll value is non-deterministic by design, so we scrub it — assert a
    roll happened (count + result shape), not the number (testing §2).
    """
    result = await net.run(
        action="I search the room for traps",
        triage=triage_response(
            "skill_check", needs_roll=True, skill="perception", ability="wisdom", dc=10,
        ),
        narration=narration_response(
            "Your eyes catch a glint of wire near the floor.",
            tool_calls=[{"name": "ref_entity", "arguments": {"entity_id": "tripwire"}}],
        ),
    )

    # A roll occurred and was surfaced; value scrubbed.
    assert len(result.dice_rolls) == 1
    assert result.mechanical_result["action_type"] == "skill_check"
    assert "success" in result.mechanical_result
    # Narrated via the roll-outcome path and tagged an entity.
    assert net.narrator.calls  # invoked through _narrate_outcome
    assert [e.effect_type for e in result.proposed_effects] == [EffectType.REF_ENTITY]


@pytest.mark.asyncio
async def test_guard_blocks_unmocked_real_calls(net):
    """Safety net itself: a real provider call raises while the guard is armed."""
    from dnd_bot.llm.client import OllamaClient

    with pytest.raises(RuntimeError, match="ALLOW_MODEL_REQUESTS is False"):
        await OllamaClient().chat(messages=[{"role": "user", "content": "hi"}])


# ── Step-1 net expansion: per-tool producer paths ────────────────────────────
# REFACTOR_PLAN.md "real Step 1" wants these turns BEFORE the tool-registry
# cut: add_npc / spawn_object / purchase / inventory, plus a pinned
# remove_entity no-op. Tests marked PINNED-BROKEN assert today's defective
# behavior on purpose — the registry step flips them (see inline arrows).


@pytest.mark.asyncio
async def test_add_npc_tool_registers_npc_in_registry_and_world_state(net):
    """An add_npc tool call lands in BOTH stores the live path writes.

    Producer path: narrator tool → _convert_tool_call('add_npc') →
    EffectExecutor._execute_add_npc (SceneEntityRegistry) →
    _sync_effect_to_world_state (mints an NPCState in WorldState.npcs).
    The dedup judge is skipped here because the roster starts empty.
    """
    # _execute_add_npc's best-effort voice auto-assign reaches for the GLOBAL
    # immersion repo/DB — stub it out to keep the test hermetic.
    net.monkeypatch.setattr(
        "dnd_bot.immersion.voice_assigner.assign_voice", _async_none
    )

    result = await net.run(
        action="I wave the old miner over to our table",
        triage=triage_response("social", needs_roll=False),
        narration=narration_response(
            "A grizzled miner shuffles over, lantern swinging.",
            tool_calls=[{
                "name": "add_npc",
                "arguments": {
                    "npc_id": "old-bram",
                    "name": "Old Bram",
                    "description": "A grizzled old miner",
                    "disposition": "friendly",
                },
            }],
        ),
    )

    assert [e.effect_type for e in result.proposed_effects] == [EffectType.ADD_NPC]

    # Scene-registry diff: the executor registered the NPC.
    entity = net.registry.get_by_name("Old Bram")
    assert entity is not None
    assert entity.entity_type == EntityType.NPC
    assert entity.disposition == Disposition.FRIENDLY
    assert entity.description == "A grizzled old miner"

    # World-state diff: the orchestrator sync minted an NPCState at the
    # current location.
    npcs = [n for n in net.session.world_state.npcs.values() if n.name == "Old Bram"]
    assert len(npcs) == 1
    assert npcs[0].location == "Tavern"
    assert npcs[0].disposition == "friendly"


@pytest.mark.asyncio
async def test_spawn_object_tool_registers_object_and_scene_item(net):
    """A spawn_object tool call registers an OBJECT entity + a scene item.

    Producer path: narrator tool → _convert_tool_call('spawn_object') →
    EffectExecutor._execute_spawn_object (registry) →
    _sync_effect_to_world_state (WorldState.spawn_item + transfer log).
    The live converter carries only object_id/name/description — nothing on
    the tool path populates ProposedEffect.object_properties, so the
    SceneEntity's `properties` field stays empty (registry-step design input).
    """
    result = await net.run(
        action="I search behind the bar",
        triage=triage_response("social", needs_roll=False),
        narration=narration_response(
            "Behind the bar, a rusty key hangs on a hook.",
            tool_calls=[{
                "name": "spawn_object",
                "arguments": {
                    "object_id": "rusty_key_1",
                    "name": "Rusty Key",
                    "description": "A rusty iron key on a hook",
                },
            }],
        ),
    )

    assert [e.effect_type for e in result.proposed_effects] == [EffectType.SPAWN_OBJECT]

    # Scene-registry diff: object entity present with its description.
    entity = net.registry.get_by_name("Rusty Key")
    assert entity is not None
    assert entity.entity_type == EntityType.OBJECT
    assert entity.description == "A rusty iron key on a hook"
    assert entity.properties == {}  # no tool-path producer for properties

    # World-state diff: scene item + transfer log entry.
    ws = net.session.world_state
    assert ws.scene_items == {"Rusty Key": "A rusty iron key on a hook"}
    assert ws.recent_transfers == ["A rusty iron key on a hook appeared in the scene"]


@pytest.mark.asyncio
async def test_purchase_turn_currently_fails_character_resolution(net):
    """PINNED-BROKEN: the live purchase route can never complete a purchase.

    Producer path: triage 'purchase' → _handle_purchase →
    _execute_purchase_item → inventory repo. _handle_purchase resolves the
    player and passes character.id (a UUID) as buyer_id, but
    _execute_purchase_item re-resolves that UUID through
    _resolve_character_by_name, which matches character NAMES only — so the
    lookup always fails and every purchase is refused with 'not found'
    before touching gold or inventory. This is the path REFACTOR_PLAN.md
    (Step 1a, "KEPT ... LIVE but UNTESTED") preserved; first covered here.
    The registry step fixes the id-vs-name contract and flips the arrows.
    """
    await net.inv_repo.add_gold(net.character.id, 100)

    result = await net.run(
        action="I buy a healing potion from the merchant",
        triage=triage_response(
            "purchase", needs_roll=False,
            item_name="Healing Potion", item_cost=30, quantity=1,
        ),
        narration=narration_response("The merchant frowns and shakes his head."),
    )

    mech = result.mechanical_result
    assert mech["action_type"] == "purchase"
    assert mech["success"] is False                      # ← True when fixed
    assert "not found" in (mech["error"] or "")          # ← None when fixed

    # The commerce tool call is surfaced even on failure.
    assert [t["name"] for t in result.tool_calls_made] == ["purchase_item"]
    assert result.tool_calls_made[0]["result"]["purchased"] is False

    # State diff: nothing changed — gold intact, no item row.
    currency = await net.inv_repo.get_currency(net.character.id)
    assert currency.gold == 100                          # ← 70 when fixed
    assert await net.inv_repo.get_all_items(net.character.id) == []

    # The refusal is still narrated via the mechanical-result path.
    assert net.narrator.calls
    assert result.proposed_effects == []


@pytest.mark.asyncio
async def test_inventory_pickup_turn_currently_pgi_blocked(net):
    """PINNED-BROKEN: the live pickup route never adds the item AND gets blocked.

    Producer path: triage 'inventory' → _handle_inventory (keyword 'pick up')
    → _execute_add_item. Same defect as purchase: _handle_inventory passes
    character.id as character_id and _execute_add_item re-resolves it by
    NAME, so the add fails silently. PGI then runs validate_item_exists
    against the (still-empty) inventory — an acquisition validated as if it
    were consumption — and HARD-fails the turn: pgi_blocked response, no
    narration. Both layers must change for pickups to work; the registry
    step flips this to: item row present + narrated turn.
    """
    result = await net.run(
        action="I pick up the brass lantern",
        triage=triage_response(
            "inventory", needs_roll=False, item_name="brass lantern",
        ),
    )

    assert result.mechanical_result.get("pgi_blocked") is True  # ← absent when fixed
    assert net.narrator.calls == []       # blocked before narration ← flips
    assert result.proposed_effects == []
    # No row was ever written.
    assert await net.inv_repo.get_all_items(net.character.id) == []


@pytest.mark.asyncio
async def test_update_player_item_grant_and_currency_persist(net):
    """The WORKING narrator-tool inventory path: update_player persists.

    Producer path: narrator tool → _convert_tool_call('update_player') →
    EffectExecutor._execute_update_player → inventory repo (item + currency
    tables). Unlike the triage routes above, this path resolves the acting
    character via acting_character_id (set from context.player_name each
    turn) — so it actually lands. The registry step must keep this green.
    """
    result = await net.run(
        action="I take the torches and the coin pouch from the chest",
        triage=triage_response("social", needs_roll=False),
        narration=narration_response(
            "You pocket two torches and a small pouch of gold.",
            tool_calls=[{
                "name": "update_player",
                "arguments": {
                    "item_grant": [{"name": "Torch", "quantity": 2}],
                    "currency_delta": {"gp": 25},
                },
            }],
        ),
    )

    assert [e.effect_type for e in result.proposed_effects] == [EffectType.UPDATE_PLAYER]

    # DB round-trip: item row + currency row both persisted.
    items = await net.inv_repo.get_all_items(net.character.id)
    assert [(i.item_name, i.quantity) for i in items] == [("Torch", 2)]
    currency = await net.inv_repo.get_currency(net.character.id)
    assert currency.gold == 25


@pytest.mark.asyncio
async def test_update_player_item_remove_persists(net):
    """update_player item_remove reduces the persisted inventory row."""
    await net.inv_repo.add_item(InventoryItem(
        character_id=net.character.id, item_index="torch",
        item_name="Torch", quantity=2,
    ))

    result = await net.run(
        action="I hand one of my torches to the guide",
        triage=triage_response("social", needs_roll=False),
        narration=narration_response(
            "The guide takes the torch with a nod.",
            tool_calls=[{
                "name": "update_player",
                "arguments": {"item_remove": [{"name": "Torch", "quantity": 1}]},
            }],
        ),
    )

    assert [e.effect_type for e in result.proposed_effects] == [EffectType.UPDATE_PLAYER]

    items = await net.inv_repo.get_all_items(net.character.id)
    assert [(i.item_name, i.quantity) for i in items] == [("Torch", 1)]


@pytest.mark.asyncio
async def test_remove_entity_is_a_silent_noop_end_to_end(net):
    """PINNED-BROKEN: remove_entity does nothing on ANY live layer.

    AUDIT_QUALITY_2026_06_09 (Duplication P0, effects.py:686): the effect-type
    dispatch sites have drifted — remove_entity is not a narrator tool
    (_convert_tool_call logs unknown_narrator_tool and drops it), and even a
    constructed REMOVE_ENTITY effect has no row in EffectExecutor's dict, so
    execute() fails and the world-state sync (which only runs on success)
    never fires: a silent end-to-end no-op. The registry step gives every
    EffectType exactly one registration — this test then flips from pinning
    the no-op to asserting the removal.
    """
    net.registry.register_entity(SceneEntity(
        name="Cellar Rat", entity_type=EntityType.CREATURE,
        description="A fat cellar rat",
    ))

    result = await net.run(
        action="I stomp on the rat, killing it",
        triage=triage_response("social", needs_roll=False),
        narration=narration_response(
            "The rat is flattened under your boot.",
            tool_calls=[{
                "name": "remove_entity",
                "arguments": {"entity_id": "Cellar Rat", "reason": "killed"},
            }],
        ),
    )

    # Layer 1 (converter): the tool call is dropped → zero effects proposed.
    assert result.proposed_effects == []        # ← [REMOVE_ENTITY] when fixed
    # The scene still contains the 'removed' entity.
    assert net.registry.get_by_name("Cellar Rat") is not None  # ← None when fixed

    # Layer 2 (executor): even a hand-built REMOVE_ENTITY effect has no
    # executor row ("No executor for effect type").
    executor = EffectExecutor(
        scene_registry=net.registry, session=net.session,
        inventory_repo=net.inv_repo,
    )
    exec_result = await executor.execute(ProposedEffect(
        effect_type=EffectType.REMOVE_ENTITY, target="Cellar Rat",
    ))
    assert exec_result.success is False         # ← True when fixed
    assert "No executor" in (exec_result.error or "")
