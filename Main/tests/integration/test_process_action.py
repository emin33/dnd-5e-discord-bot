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

import json
from pathlib import Path

import pytest

from dnd_bot.data.database import Database
from dnd_bot.data.repositories.character_repo import CharacterRepository
from dnd_bot.data.repositories.inventory_repo import InventoryRepository
from dnd_bot.data.repositories import character_repo as character_repo_module
from dnd_bot.llm import client as llm_client
from dnd_bot.llm import orchestrator as orchestrator_module
from dnd_bot.llm.client import LLMResponse
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

    async def run(self, action, triage, narration=None, player_name=None,
                  context=None, narrations=None, on_narrative_token=None,
                  brain_fn=None):
        self.brain = FunctionBrain(brain_fn or brain_router(triage))
        # Every get_llm_client()-backed seam → the one branching fake. The
        # extractors and the dedup judge are process-wide singletons, so they
        # go through monkeypatch — teardown restores the real clients instead
        # of leaking the fake into later tests.
        self.orch.client = self.brain
        self.monkeypatch.setattr(self.orch._state_extractor, "client", self.brain)
        self.monkeypatch.setattr(self.orch._entity_extractor, "client", self.brain)
        self.monkeypatch.setattr(get_dedup_judge(), "client", self.brain)
        # Narrator tier seam → a scripted fake. One prose+tools response per
        # turn normally; the narration pins pass ``narrations`` to script the
        # tool-followup / streaming second leg too.
        self.narrator = ScriptedBrain(
            list(narrations) if narrations else [narration or narration_response("")]
        )
        self.orch._narrator_client_factory = lambda tier: self.narrator

        context = context or BrainContext(
            campaign_id=self.session.campaign_id,
            session_id=self.session.id,
        )
        return await self.orch.process_action(
            action, player_name or self.character.name, context,
            on_narrative_token=on_narrative_token,
        )


@pytest.fixture(autouse=True)
def _clean_channel_99_registries():
    """The net session uses the fixed channel 99, and combat entry registers
    a CombatManager into module-global registries that outlive each test —
    the Step-0 attack pin had been leaking its manager at ``discord:99``
    into every later test in the run (masked only because nothing else read
    that channel). Clean on both sides.
    """
    from dnd_bot.game.combat.coordinator import clear_coordinator_by_key
    from dnd_bot.game.combat.manager import clear_combat_by_key

    clear_combat_by_key("discord:99")
    clear_coordinator_by_key("discord:99")
    yield
    clear_combat_by_key("discord:99")
    clear_coordinator_by_key("discord:99")


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
# remove_entity no-op. The tests originally pinned BROKEN have all been
# flipped to the working behavior: remove_entity by the registry cut,
# purchase / inventory-pickup by the commerce id-vs-name fix.


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
async def test_purchase_turn_deducts_gold_and_adds_item(net):
    """FLIPPED from pinned-broken by the commerce id-vs-name fix.

    Producer path: triage 'purchase' → _handle_purchase →
    _execute_purchase_item → inventory repo. _handle_purchase resolved the
    player and passed character.id (a UUID) as buyer_id, but the executor
    re-resolved it through _resolve_character_by_name, which matches
    character NAMES only — every purchase was refused with 'not found'
    before touching gold or inventory. The executor now takes the
    already-resolved Character from its single caller, so the purchase
    lands: gold deducted, item row written, success narrated.
    """
    await net.inv_repo.add_gold(net.character.id, 100)

    result = await net.run(
        action="I buy a healing potion from the merchant",
        triage=triage_response(
            "purchase", needs_roll=False,
            item_name="Healing Potion", item_cost=30, quantity=1,
        ),
        narration=narration_response("The merchant counts your coin and nods."),
    )

    mech = result.mechanical_result
    assert mech["action_type"] == "purchase"
    assert mech["success"] is True
    assert mech["error"] is None
    assert mech["gold_after"] == 70

    # The commerce tool call is surfaced with the executed result.
    assert [t["name"] for t in result.tool_calls_made] == ["purchase_item"]
    assert result.tool_calls_made[0]["result"]["purchased"] is True

    # State diff: gold moved and the item row exists.
    currency = await net.inv_repo.get_currency(net.character.id)
    assert currency.gold == 70
    items = await net.inv_repo.get_all_items(net.character.id)
    assert [(i.item_name, i.quantity) for i in items] == [("Healing Potion", 1)]

    # The purchase is narrated via the mechanical-result path.
    assert net.narrator.calls
    assert result.proposed_effects == []


@pytest.mark.asyncio
async def test_inventory_pickup_turn_adds_item_and_narrates(net):
    """FLIPPED from pinned-broken by the commerce id-vs-name fix.

    Producer path: triage 'inventory' → _handle_inventory (keyword 'pick up')
    → _execute_add_item. Same defect as purchase: _handle_inventory passed
    character.id as character_id and _execute_add_item re-resolved it by
    NAME, so the add failed silently — and PGI then ran validate_item_exists
    against the still-empty inventory and HARD-failed the turn (pgi_blocked,
    no narration). With the executor taking the resolved Character, the add
    lands BEFORE PGI's prefetch, validate_item_exists sees the row, and the
    turn narrates.
    """
    result = await net.run(
        action="I pick up the brass lantern",
        triage=triage_response(
            "inventory", needs_roll=False, item_name="brass lantern",
        ),
        narration=narration_response("You lift the brass lantern from the table."),
    )

    mech = result.mechanical_result
    assert mech.get("pgi_blocked") is None
    assert mech["action_type"] == "inventory"
    assert mech["operation"] == "pickup"
    assert mech["success"] is True
    assert [t["name"] for t in result.tool_calls_made] == ["add_item"]

    # The row was written (and survived PGI).
    items = await net.inv_repo.get_all_items(net.character.id)
    assert [(i.item_name, i.quantity) for i in items] == [("Brass Lantern", 1)]

    # Narrated turn, not a block.
    assert net.narrator.calls
    assert result.proposed_effects == []


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
async def test_remove_entity_tool_removes_entity_end_to_end(net):
    """FLIPPED from pinned-broken by the Step-1 registry cut.

    AUDIT_QUALITY_2026_06_09 (Duplication P0, effects.py:686): remove_entity
    used to be a silent end-to-end no-op — dropped at the converter (no tool)
    AND at the executor (no dict row), so the world-state sync (which only
    runs on success) never fired. The registry wires all layers: full-tier
    tool → registry converter → EffectExecutor._execute_remove_entity
    (scene-registry removal) → REMOVE_ENTITY sync branch.

    The tool call uses the DOCUMENTED id dialect (final review): the roster
    the narrator sees lists entities as [id: slug] — slugify(name),
    hyphenated — so 'cellar-rat' for a multi-word name is the contract the
    executor's lookup must honor, not just the display name.
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
                "arguments": {"entity_id": "cellar-rat", "reason": "killed"},
            }],
        ),
    )

    # Layer 1 (converter): the tool call becomes a REMOVE_ENTITY effect.
    assert [e.effect_type for e in result.proposed_effects] == [EffectType.REMOVE_ENTITY]
    # Layer 2+ (executor): the entity is gone from the scene registry.
    assert net.registry.get_by_name("Cellar Rat") is None

    # The INTENTS-fallback producer shape (target=<id>) executes through the
    # same registered handler — both producers, one registration.
    net.registry.register_entity(SceneEntity(
        name="Cellar Rat", entity_type=EntityType.CREATURE,
        description="A fat cellar rat, back for more",
    ))
    executor = EffectExecutor(
        scene_registry=net.registry, session=net.session,
        inventory_repo=net.inv_repo,
    )
    exec_result = await executor.execute(ProposedEffect(
        effect_type=EffectType.REMOVE_ENTITY, target="Cellar Rat",
    ))
    assert exec_result.success is True
    assert net.registry.get_by_name("Cellar Rat") is None
    # And an honest failure when the target does not exist — no silent success.
    missing = await executor.execute(ProposedEffect(
        effect_type=EffectType.REMOVE_ENTITY, target="Cellar Rat",
    ))
    assert missing.success is False
    assert "not in scene registry" in (missing.error or "")


# ── Step-2 narration pins: prompt-assembly inputs per narration path ──────────
# REFACTOR_PLAN.md Step 2 collapsed the three near-duplicate narration paths
# (AUDIT_QUALITY_2026_06_09, Duplication P1) into one NarrationStrategy +
# NarrationSpec (llm/narration.py). These tests golden-pin what each path
# sends through the narrator-client seam — originally the drifted per-path
# rebuilds, now the CONVERGED strategy:
#
#   A  _narrate_mechanical_result  (purchase/sell/inventory mech dict)
#   B  _narrate_action             (no-mechanics: social/exploration/…)
#   C  _narrate_outcome            (dice-roll resolution)
#   +  the tool followup           (shared 2nd leg inside the strategy,
#                                   NOT a 4th path)
#
# Pinned per path, exact-match:
#   1. field→appears-in-prompt map: every BrainContext string field of the
#      ORIGINAL context is seeded with a unique sentinel; presence in the
#      call-1 messages == that field reaches the narrator's prompt. The
#      strategy carries EVERY field via dataclasses.replace, so all three
#      paths share ONE table now (_FIELDS_IN_PROMPT below records which
#      entries the Step-2 migration flipped, and why).
#   2. message-list role shape + the structured player_action decoration
#      (raw / [NARRATIVE DIRECTION: …] / [RESOLUTION: …] with an injected
#      fixed roller — no random rolls pinned).
#   3. chat kwargs (exact key set + values; tools = the profile-tier set).
#
# Cross-path facts the pins encode:
#   - The pre-Step-2 drift (A's 8-field rebuild losing memory/history/
#     summary/quests; kg_context_yaml/narrative_memory/character_stats/
#     last_turn_trace dropped by ALL THREE) is resolved: all were verdicted
#     BUG and unioned — see the Step-2 migration commit for the verdicts.
#   - Only B streams (intent), and the streaming call carries NO tools
#     kwargs — tool recovery on streamed turns rides the followup leg.
# Prose is never asserted; SRD rule injection is neutralized for determinism.


_BOOKEND_REMINDER = (
    "<reminder>\n"
    "Last turn: S_LAST_TURN_TRACE\n"  # FLIPPED by Step 2: was never rendered
    "Use ONLY the world state provided above as ground truth.\n"
    "Every NPC you mention must appear in the world state at their listed location.\n"
    "</reminder>"
)

# Sentinel presence for ALL THREE paths: the strategy's dataclasses.replace
# carries every upstream field, so the per-path tables converged into this
# one. Entries marked FLIPPED were pinned False pre-Step-2 (the drift the
# audit called out); each flip is a deliberately-resolved BUG verdict argued
# in the Step-2 migration commit body.
_FIELDS_IN_PROMPT = {
    "campaign_id": False,       # carried but builders never render it
    "session_id": False,        # carried but builders never render it
    "party_members": True,
    "party_status": False,      # carried but shadowed by party_members
    "current_scene": True,
    "active_quests": True,      # FLIPPED for A (8-field rebuild dropped it)
    "combat_state": True,
    "current_combatant": False,  # carried; latent (renders only w/o combat_state)
    "initiative_order": False,   # carried; latent (renders only w/o combat_state)
    "memory_context": True,     # FLIPPED for A
    "message_history": True,    # FLIPPED for A
    "recent_messages": False,   # carried but shadowed by message_history
    "session_summary": True,    # FLIPPED for A
    "character_stats": True,    # FLIPPED for all three (<acting_character>)
    "world_state_yaml": True,
    "kg_context_yaml": True,    # FLIPPED for all three (<entity_relationships>)
    "narrative_memory": True,   # FLIPPED for all three (<past_narration>)
    "last_turn_trace": True,    # FLIPPED for all three ("Last turn:" reminder)
}

# Bookend shape (world_state_yaml present) for a context whose middle blocks
# (memory/summary) are populated, plus one history message, plus the
# path-appended instruction system message + tool-reminder system message.
_FULL_ROLES = [
    "system",     # persona + ## Combat (combat_state)
    "user",       # bookend top: world_state/party/scene/quests
    "assistant",  # ack anchor
    "user",       # bookend middle: memory + session summary
    "assistant",  # ack anchor
    "user",       # message_history splice (1 sentinel message)
    "user",       # final: <player_action> + <reminder>
    "system",     # path-specific ###INSTRUCTION###
    "system",     # _append_tool_reminder
]

# Path A shares the full bookend shape now (Step 2 restored memory/summary/
# history to mechanical-result narration); its per-path prompt is still a
# USER message carrying the mech outcome (not ###INSTRUCTION### system).
_MECH_ROLES = [
    "system",     # persona + ## Combat (combat_state)
    "user",       # bookend top: world_state/party/scene/quests
    "assistant",  # ack anchor
    "user",       # bookend middle: memory + session summary  (FLIPPED: was absent)
    "assistant",  # ack anchor
    "user",       # message_history splice (1 sentinel message) (FLIPPED: was absent)
    "user",       # final: <player_action> + <reminder>
    "user",       # mech prompt ("The player … attempted …")
    "system",     # _append_tool_reminder
]


def _sentinel_context() -> tuple[BrainContext, dict[str, str]]:
    """An original context with one unique sentinel per narrator-relevant field.

    Mirrors everything session._build_context + process_action populate for
    the narrator (session.py:741, orchestrator.py:776/1070/1086).
    """
    sentinels = {
        "campaign_id": "S_CAMPAIGN_ID",
        "session_id": "S_SESSION_ID",
        "party_members": "S_PARTY_MEMBERS",
        "party_status": "S_PARTY_STATUS",
        "current_scene": "S_CURRENT_SCENE",
        "active_quests": "S_ACTIVE_QUESTS",
        "combat_state": "S_COMBAT_STATE",
        "current_combatant": "S_CURRENT_COMBATANT",
        "initiative_order": "S_INITIATIVE_ORDER",
        "memory_context": "S_MEMORY_CONTEXT",
        "message_history": "S_MESSAGE_HISTORY",
        "recent_messages": "S_RECENT_MESSAGES",
        "session_summary": "S_SESSION_SUMMARY",
        "character_stats": "S_CHARACTER_STATS",
        "world_state_yaml": "S_WORLD_STATE_YAML",
        "kg_context_yaml": "S_KG_CONTEXT_YAML",
        "narrative_memory": "S_NARRATIVE_MEMORY",
        "last_turn_trace": "S_LAST_TURN_TRACE",
    }
    context = BrainContext(
        campaign_id=sentinels["campaign_id"],
        session_id=sentinels["session_id"],
        party_members=sentinels["party_members"],
        party_status=sentinels["party_status"],
        current_scene=sentinels["current_scene"],
        active_quests=sentinels["active_quests"],
        in_combat=False,
        combat_state=sentinels["combat_state"],
        combat_round=7,
        current_combatant=sentinels["current_combatant"],
        initiative_order=sentinels["initiative_order"],
        memory_context=sentinels["memory_context"],
        recent_messages=[{"role": "user", "content": sentinels["recent_messages"]}],
        message_history=[{"role": "user", "content": sentinels["message_history"]}],
        session_summary=sentinels["session_summary"],
        character_stats=sentinels["character_stats"],
        world_state_yaml=sentinels["world_state_yaml"],
        kg_context_yaml=sentinels["kg_context_yaml"],
        narrative_memory=sentinels["narrative_memory"],
        last_turn_trace=sentinels["last_turn_trace"],
    )
    return context, sentinels


def _fields_in_prompt(call: dict, sentinels: dict[str, str]) -> dict[str, bool]:
    """field → does its sentinel appear anywhere in this call's messages."""
    blob = json.dumps(call["messages"])
    return {field: (marker in blob) for field, marker in sentinels.items()}


def _neutralize_rule_injection(monkeypatch) -> None:
    """SRD rule injection depends on data availability — pin without it."""
    class _NoRules:
        def get_relevant_rules(self, *args, **kwargs) -> str:
            return ""
    monkeypatch.setattr(
        "dnd_bot.llm.brains.base.get_rule_injector", lambda: _NoRules()
    )


class _FixedRoller:
    """Deterministic d20 so the [RESOLUTION: …] decoration is exact-pinnable."""

    def __init__(self, total: int):
        self._total = total

    def roll(self, notation, advantage=False, disadvantage=False, reason=""):
        from dnd_bot.game.mechanics.dice import DiceRoll
        return DiceRoll(
            notation=notation, dice_results=[self._total],
            kept_dice=[self._total], total=self._total, reason=reason,
        )


def _assert_primary_narration_kwargs(net, call) -> None:
    """The shared chat contract of all three paths' primary (non-stream) call."""
    kw = call["kwargs"]
    assert set(kw) == {
        "temperature", "max_tokens", "frequency_penalty", "presence_penalty",
        "tools", "tool_choice",
    }
    assert kw["temperature"] == net.orch.narrator.temperature
    assert kw["max_tokens"] == 1500
    assert kw["frequency_penalty"] == 0.4   # NARRATOR_FREQUENCY_PENALTY
    assert kw["presence_penalty"] == 0.3    # NARRATOR_PRESENCE_PENALTY
    assert kw["tool_choice"] == "auto"
    # The profile-tier tool set, not a hardcoded list (audit #2/N2 regression).
    assert [t["function"]["name"] for t in kw["tools"]] == [
        t["function"]["name"] for t in net.orch._get_narrator_tools()
    ]


@pytest.mark.asyncio
async def test_pin_mechanical_result_narration_prompt_inputs(net):
    """PIN path A (_narrate_mechanical_result) via a purchase turn.

    Step 2 flipped this table: purchase/sale narration used to reach the
    model through an 8-field rebuild with NO memory, NO message history, NO
    session summary, NO quests (the audit's live drift cost). The strategy's
    union restored them — A now shares the converged field table and the
    full bookend shape; its per-path delta is only the USER mech prompt.
    """
    _neutralize_rule_injection(net.monkeypatch)
    await net.inv_repo.add_gold(net.character.id, 100)
    context, sentinels = _sentinel_context()
    tokens: list[str] = []

    async def on_token(t):
        tokens.append(t)

    await net.run(
        action="I buy a healing potion from the merchant",
        triage=triage_response(
            "purchase", needs_roll=False,
            item_name="Healing Potion", item_cost=30, quantity=1,
        ),
        narration=narration_response(
            "The merchant slides the potion across the counter.",
            tool_calls=[{"name": "ref_entity", "arguments": {"entity_id": "merchant"}}],
        ),
        context=context,
        on_narrative_token=on_token,
    )

    # One primary call; tool call present → no followup. Streaming callback
    # provided but IGNORED — only path B streams (intent, kept per-path via
    # NarrationSpec.allow_streaming).
    assert len(net.narrator.calls) == 1
    call = net.narrator.calls[0]
    assert call["method"] == "chat"
    assert tokens == []

    assert _fields_in_prompt(call, sentinels) == _FIELDS_IN_PROMPT
    assert [m["role"] for m in call["messages"]] == _MECH_ROLES

    # player_action reaches the bookend RAW (no per-path decoration on A).
    assert call["messages"][-3]["content"] == (
        "<player_action>[Elara]: I buy a healing potion from the merchant"
        "</player_action>\n\n" + _BOOKEND_REMINDER
    )
    # A's per-path prompt is a USER message carrying the mech outcome.
    assert call["messages"][-2]["content"] == (
        'The player Elara attempted: "I buy a healing potion from the merchant"\n'
        "\n"
        "[RESULT: SUCCESS] Elara successfully purchases 1x Healing Potion for "
        "30gp. They have 70gp remaining.\n"
        "\n"
        "Narrate this action dramatically. Remember:\n"
        "- Show the world's REACTION to the action (environment, NPCs, atmosphere)\n"
        "- Connect to ongoing tension or stakes from the current scene\n"
        "- End with something that maintains momentum\n"
        "\n"
        "Write your narration directly."
    )
    assert call["messages"][-1]["content"].startswith("AFTER writing your prose")

    _assert_primary_narration_kwargs(net, call)


@pytest.mark.asyncio
async def test_pin_action_narration_prompt_inputs(net):
    """PIN path B (_narrate_action) via a social turn (non-streaming).

    B always carried memory/history/summary/quests; Step 2 additionally
    flipped kg_context_yaml, narrative_memory, character_stats and
    last_turn_trace into the prompt (computed for the narrator upstream,
    dropped by every pre-Step-2 rebuild).
    """
    _neutralize_rule_injection(net.monkeypatch)
    context, sentinels = _sentinel_context()

    await net.run(
        action="I greet the barkeep warmly",
        triage=triage_response(
            "social", needs_roll=False,
            narrative_direction="Respond with a calm beat",
        ),
        narration=narration_response(
            "The barkeep nods, wiping a mug.",
            tool_calls=[{"name": "ref_entity", "arguments": {"entity_id": "barkeep"}}],
        ),
        context=context,
    )

    assert len(net.narrator.calls) == 1
    call = net.narrator.calls[0]
    assert call["method"] == "chat"

    assert _fields_in_prompt(call, sentinels) == _FIELDS_IN_PROMPT
    assert [m["role"] for m in call["messages"]] == _FULL_ROLES

    # message_history is spliced verbatim between the bookend middle and the
    # final user message.
    assert call["messages"][5] == {"role": "user", "content": "S_MESSAGE_HISTORY"}

    # player_action carries the triage direction as B's decoration.
    assert call["messages"][-3]["content"] == (
        "<player_action>[Elara]: I greet the barkeep warmly\n"
        "\n"
        "[NARRATIVE DIRECTION: Respond with a calm beat]</player_action>\n"
        "\n" + _BOOKEND_REMINDER
    )
    # B's per-path prompt is an ###INSTRUCTION### SYSTEM message with the
    # intent guidance for the triaged action type (+ style/phase hints,
    # not pinned — rotation text, not structure).
    instruction = call["messages"][-2]
    assert instruction["role"] == "system"
    assert instruction["content"].startswith(
        "###INSTRUCTION###\n"
        "Narrate the player's action according to the NARRATIVE DIRECTION above.\n"
        "\n"
        "This is a SOCIAL interaction."
    )
    assert call["messages"][-1]["content"].startswith("AFTER writing your prose")

    _assert_primary_narration_kwargs(net, call)


@pytest.mark.asyncio
async def test_pin_outcome_narration_prompt_inputs(net):
    """PIN path C (_narrate_outcome) via a skill check with a FIXED roller.

    Same converged field table as A/B; the per-path delta is the
    [RESOLUTION: …] decoration + the success/failure instruction. Roll
    injected (14 vs DC 10 → NARROW SUCCESS) so the pin is exact without
    pinning randomness.
    """
    _neutralize_rule_injection(net.monkeypatch)
    net.monkeypatch.setattr(net.orch, "roller", _FixedRoller(14))
    context, sentinels = _sentinel_context()
    tokens: list[str] = []

    async def on_token(t):
        tokens.append(t)

    result = await net.run(
        action="I study the mosaic on the floor",
        triage=triage_response(
            "skill_check", needs_roll=True, skill="perception",
            ability="wisdom", dc=10,
            on_success=["a loose tile hiding a recess"],
        ),
        narration=narration_response(
            "One tile sits a hair higher than its neighbors.",
            tool_calls=[{"name": "ref_entity", "arguments": {"entity_id": "mosaic"}}],
        ),
        context=context,
        on_narrative_token=on_token,
    )

    assert result.dice_rolls[0].total == 14

    # One primary call; C never streams even with a callback (intent, kept
    # per-path via NarrationSpec.allow_streaming).
    assert len(net.narrator.calls) == 1
    call = net.narrator.calls[0]
    assert call["method"] == "chat"
    assert tokens == []

    assert _fields_in_prompt(call, sentinels) == _FIELDS_IN_PROMPT
    assert [m["role"] for m in call["messages"]] == _FULL_ROLES

    expected_resolution = (
        "Perception Check: SUCCESS | "
        "Rolled 14 vs DC 10 (margin: +4) | "
        "NARROW SUCCESS — reveal the basics, but incompletely | "
        "AUTHORIZED REVEALS (narrate ONLY these): a loose tile hiding a recess | "
        "Do NOT invent additional discoveries or outcomes beyond what is listed above."
    )
    # player_action carries the resolution as C's decoration.
    assert call["messages"][-3]["content"] == (
        "<player_action>[Elara]: I study the mosaic on the floor\n"
        "\n"
        f"[RESOLUTION: {expected_resolution}]</player_action>\n"
        "\n" + _BOOKEND_REMINDER
    )
    # C's per-path prompt: SYSTEM instruction restating the resolution and
    # the explicit success directive.
    instruction = call["messages"][-2]
    assert instruction["role"] == "system"
    assert instruction["content"].startswith(
        f"###INSTRUCTION###\nRESOLUTION: {expected_resolution}\n\n"
    )
    assert "This perception roll SUCCEEDED (rolled 14 vs DC 10)" in instruction["content"]
    assert call["messages"][-1]["content"].startswith("AFTER writing your prose")

    _assert_primary_narration_kwargs(net, call)


@pytest.mark.asyncio
async def test_pin_tool_followup_reuses_path_messages(net):
    """PIN the shared followup leg: not a 4th path, a 2nd call inside each.

    When the primary response has prose but no tool calls, the strategy's
    followup leg re-sends the SAME message stack (audit #20 contract)
    + assistant prose + a tool-only user turn, under followup-specific kwargs.
    Driven through path B; the leg itself is path-agnostic.
    """
    _neutralize_rule_injection(net.monkeypatch)
    context, _ = _sentinel_context()

    result = await net.run(
        action="I greet the barkeep warmly",
        triage=triage_response(
            "social", needs_roll=False,
            narrative_direction="Respond with a calm beat",
        ),
        narrations=[
            narration_response("The tavern hums quietly."),  # no tool calls
            narration_response("", tool_calls=[
                {"name": "ref_entity", "arguments": {"entity_id": "barkeep"}},
            ]),
        ],
        context=context,
    )

    assert len(net.narrator.calls) == 2
    primary, followup = net.narrator.calls
    assert followup["method"] == "chat"

    # The followup REUSES the full primary stack (roster/world-state intact)…
    n = len(primary["messages"])
    assert followup["messages"][:n] == primary["messages"]
    # …then appends the assistant prose + the tool-only instruction.
    assert followup["messages"][n] == {
        "role": "assistant", "content": "The tavern hums quietly.",
    }
    assert followup["messages"][n + 1]["role"] == "user"
    assert followup["messages"][n + 1]["content"].startswith(
        "Now call a tool for everything you narrated above"
    )
    assert len(followup["messages"]) == n + 2

    # Followup-specific kwargs: deterministic, capped, tools REQUIRED, and the
    # same profile-tier tool set as the primary call (audit #2/N2).
    kw = followup["kwargs"]
    assert set(kw) == {"temperature", "max_tokens", "think", "tools", "tool_choice"}
    assert kw["temperature"] == 0
    assert kw["max_tokens"] == 500
    assert kw["think"] is False
    assert kw["tool_choice"] == "required"
    assert [t["function"]["name"] for t in kw["tools"]] == [
        t["function"]["name"] for t in primary["kwargs"]["tools"]
    ]

    # The followup's tool calls are adopted as the turn's effects.
    assert [e.effect_type for e in result.proposed_effects] == [EffectType.REF_ENTITY]


@pytest.mark.asyncio
async def test_pin_streaming_only_on_action_path_and_drops_tools(net):
    """PIN the streaming drift: path B streams when a token callback exists,
    and the streaming call carries NO tools kwargs at all — tool recovery on
    a streamed turn rides ENTIRELY on the followup leg.
    """
    _neutralize_rule_injection(net.monkeypatch)
    context, _ = _sentinel_context()
    tokens: list[str] = []

    async def on_token(t):
        tokens.append(t)

    result = await net.run(
        action="I greet the barkeep warmly",
        triage=triage_response(
            "social", needs_roll=False,
            narrative_direction="Respond with a calm beat",
        ),
        narrations=[
            narration_response("The hearth crackles softly."),  # streamed leg
            narration_response("", tool_calls=[
                {"name": "ref_entity", "arguments": {"entity_id": "barkeep"}},
            ]),
        ],
        context=context,
        on_narrative_token=on_token,
    )

    assert len(net.narrator.calls) == 2
    stream, followup = net.narrator.calls

    assert stream["method"] == "chat_stream"
    assert tokens == ["The hearth crackles softly."]
    # Exact kwargs: penalties/limits identical to chat, but NO tools and NO
    # tool_choice — the pinned hole the followup exists to cover.
    kw = stream["kwargs"]
    assert set(kw) == {
        "temperature", "max_tokens", "frequency_penalty", "presence_penalty",
    }
    assert kw["temperature"] == net.orch.narrator.temperature
    assert kw["max_tokens"] == 1500
    assert kw["frequency_penalty"] == 0.4
    assert kw["presence_penalty"] == 0.3

    assert followup["method"] == "chat"
    assert followup["kwargs"]["tool_choice"] == "required"
    assert [e.effect_type for e in result.proposed_effects] == [EffectType.REF_ENTITY]


# ── Step-3 combat-entry signal pins (REFACTOR_PLAN Step 3) ────────────────────
# Combat entry is decided by THREE live signals with different participant
# logic (AUDIT_QUALITY_2026_06_09, Architecture P2 "three live combat-entry
# deciders"):
#   1. the triage attack branch          → _initiate_combat_from_attack
#   2. the entity-extractor's combat_initiated flag → all scene hostiles
#   3. the narrator's start_combat tool  → sets combat_triggered ONLY —
#      builds no encounter, so the session can flip to COMBAT with no
#      CombatManager ever created (the wedge end_combat has to heal).
# These pins capture each signal's full observable outcome BEFORE the
# ModeMachine / EncounterBuilder rewire. Pins marked BROKEN carry flip
# arrows for the rewire commit.


def _hostile_goblin() -> SceneEntity:
    return SceneEntity(
        name="Goblin", entity_type=EntityType.CREATURE,
        description="A snarling goblin", disposition=Disposition.HOSTILE,
        hostility_score=90,
    )


@pytest.mark.asyncio
async def test_attack_entry_builds_surprised_encounter(net):
    """Signal 1: the triage attack branch builds and registers a full
    encounter (players + target + other hostiles), enemies surprised."""
    from dnd_bot.game.combat.manager import get_combat_for_channel
    from dnd_bot.game.session import SessionState
    from dnd_bot.models import CombatState

    net.registry.register_entity(_hostile_goblin())

    result = await net.run(
        action="I attack the goblin",
        triage=triage_response(
            "attack", target_name="Goblin", is_creature_target=True,
        ),
    )

    assert result.combat_triggered is True
    manager = get_combat_for_channel(99)
    assert manager is not None
    assert net.session.combat_manager is manager
    assert sorted(c.name for c in manager.combat.combatants) == ["Elara", "Goblin"]
    # Player-initiated: enemies surprised, player not.
    goblin = next(c for c in manager.combat.combatants if not c.is_player)
    player = next(c for c in manager.combat.combatants if c.is_player)
    assert goblin.is_surprised is True
    assert player.is_surprised is False
    # Initiative rolled and combat started, ready for the first turn.
    assert manager.combat.state == CombatState.AWAITING_ACTION
    # FLIPPED by the Step-3 rewire: the encounter builder owns the mode
    # push now, so the flip happens IN the entry point instead of lagging
    # to process_message's (deleted) inline branch.
    assert net.session.state == SessionState.COMBAT
    assert net.session.world_state.phase == "combat"
    assert net.session.modes.in_combat is True


@pytest.mark.asyncio
async def test_narrative_entry_drafts_scene_hostiles_unsurprised(net):
    """Signal 2: the entity-extractor's combat_initiated flag builds an
    encounter from ALL hostile/unfriendly scene NPCs+creatures, no surprise."""
    from dnd_bot.game.combat.manager import get_combat_for_channel
    from dnd_bot.game.session import SessionState
    from dnd_bot.models import CombatState

    net.registry.register_entity(_hostile_goblin())
    # Entity extraction only runs when the session has NO world state
    # (world-state extraction supersedes it) — drop it for this signal.
    net.session.world_state = None
    # ...and don't let the registry sync test entities into the real NPC DB.
    net.monkeypatch.setattr(net.registry, "sync_to_npc_repo", _async_none)

    social = triage_response("social", needs_roll=False)
    extraction = LLMResponse(content=json.dumps({
        "entities": [], "scene_update": None, "hostility_changes": [],
        "combat_initiated": True,
    }))

    def _brain(messages, **kwargs):
        system = messages[0].get("content", "") if messages else ""
        if "action classifier" in system:
            return social
        if system.startswith("You extract entities"):
            return extraction
        return LLMResponse(content="{}")

    narration = narration_response(
        "The goblin's blade flashes as it lunges straight at Elara."
    )
    result = await net.run(
        action="I stand my ground",
        triage=social,
        # No tool calls on either leg: the followup returns prose again and
        # the effect list stays empty — this signal is extraction-driven.
        narrations=[narration, narration],
        brain_fn=_brain,
    )

    assert result.combat_triggered is True
    manager = get_combat_for_channel(99)
    assert manager is not None
    assert sorted(c.name for c in manager.combat.combatants) == ["Elara", "Goblin"]
    # NPC-initiated: nobody is surprised.
    assert all(not c.is_surprised for c in manager.combat.combatants)
    assert manager.combat.state == CombatState.AWAITING_ACTION
    # FLIPPED by the Step-3 rewire: the entry point pushed combat mode.
    assert net.session.state == SessionState.COMBAT
    assert net.session.modes.in_combat is True


@pytest.mark.asyncio
async def test_start_combat_effect_builds_encounter_from_scene(net):
    """Signal 3, FLIPPED by the Step-3 rewire (was PINNED BROKEN: the
    narrator's start_combat tool only set combat_triggered — no encounter,
    no participants — flipping the session to COMBAT with no CombatManager,
    the audit's no-manager wedge). The signal now routes through the single
    entry point and drafts the scene hostiles like the extractor path."""
    from dnd_bot.game.combat.manager import get_combat_for_channel
    from dnd_bot.game.session import SessionState

    net.registry.register_entity(_hostile_goblin())

    result = await net.run(
        action="I stand my ground",
        triage=triage_response("social", needs_roll=False),
        narration=narration_response(
            "The goblin snarls and springs at Elara, blade first!",
            tool_calls=[{
                "name": "start_combat",
                "arguments": {"reason": "The goblin attacks"},
            }],
        ),
    )

    assert result.combat_triggered is True
    manager = get_combat_for_channel(99)
    assert manager is not None
    assert net.session.combat_manager is manager
    assert sorted(c.name for c in manager.combat.combatants) == ["Elara", "Goblin"]
    # Narrator-initiated: nobody is surprised.
    assert all(not c.is_surprised for c in manager.combat.combatants)
    assert net.session.state == SessionState.COMBAT


@pytest.mark.asyncio
async def test_start_combat_effect_with_no_hostiles_refuses(net):
    """Signal 3 with an EMPTY scene, FLIPPED by the Step-3 rewire (was
    PINNED BROKEN: combat_triggered=True with nothing to fight). The entry
    point now refuses to trigger an encounter with no participants — no
    manager, no mode flip, combat_triggered False."""
    from dnd_bot.game.combat.manager import get_combat_for_channel
    from dnd_bot.game.session import SessionState

    result = await net.run(
        action="I ready my staff",
        triage=triage_response("social", needs_roll=False),
        narration=narration_response(
            "Shadows coil at the room's edge, and something unseen strikes!",
            tool_calls=[{
                "name": "start_combat",
                "arguments": {"reason": "Ambush from the shadows"},
            }],
        ),
    )

    assert result.combat_triggered is False
    assert get_combat_for_channel(99) is None
    assert net.session.state == SessionState.STARTING
