"""Cheap, deterministic narrator-tool reliability preflight.

This is deliberately shorter and less creative than ``test_long_horizon``.
It seeds a tiny valid scene, issues state-transition-heavy player actions,
and checks the production turn log for executed/rejected effects and repair
telemetry. Run this before paying for a long narrative soak.

Two turn scripts share the harness:

- ``baseline`` — the original 12-turn gauntlet (items, currency, NPC state,
  scene objects, location, dead-NPC continuity).
- ``player_state_sweep`` — the matrix tool-reliability track's explicit
  mutation coverage: currency in both directions, item grant/remove with
  npc source/destination mirrors, condition add/remove, and a spell-slot
  expenditure, plus the scene families. Every executed ``update_player``
  receipt is then reconciled against the character/inventory DB
  (receipt-vs-state agreement): a receipt with no matching write, or a
  write with no receipt, fails the run.

Usage::

    python test_tool_reliability.py --profile deepseek_v4_flash_qwen9b
    python test_tool_reliability.py --scenario player_state_sweep
"""

from __future__ import annotations

import argparse
import asyncio
from collections import Counter
import json
import os
from pathlib import Path
import sys
import time
import uuid

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.chdir(Path(__file__).parent)

from dotenv import load_dotenv

load_dotenv()


TOOL_ACTIONS = [
    "I greet Mara Venn by name and ask what she has learned about the Ash Gate.",
    (
        "This is an uncontested item transfer with no roll: I hand my brass "
        "compass to Mara Venn, she accepts it, and it is now in her coat "
        "rather than my pack."
    ),
    (
        "Mara reveals the fresh crescent scar beneath her glove and swears to "
        "become my ally in exposing the gate conspiracy."
    ),
    (
        "This is an uncontested return with no roll: Mara Venn takes the brass "
        "compass from her coat, hands it back to me, and it is now in my pack "
        "rather than hers."
    ),
    "I pay Mara Venn exactly two gold pieces for her help, and she accepts them.",
    (
        "Mara Venn sets a newly revealed obsidian key on the table between us; "
        "it is a distinct object I can pick up."
    ),
    "I pick up the obsidian key from the table and put it in my pack.",
    (
        "This is an established automatic trigger with no roll: I do not "
        "touch or use the sealed reliquary. From across the room I say 'now,' "
        "and its already-armed charge finishes the countdown and destroys "
        "the reliquary completely."
    ),
    (
        "I warn Mara Venn that the watch is at the door. She chooses to flee "
        "the Copper Finch immediately, taking nothing else."
    ),
    (
        "I call out for Old Bram, though I know he died. I listen for no living "
        "answer and instead recall his last warning about the Ash Gate."
    ),
    (
        "I leave the Copper Finch and travel to the Ash Gate, arriving beneath "
        "its cracked black arch."
    ),
    (
        "At the Ash Gate I meet a new, physically present courier named Sable "
        "Quill, introduce myself, and ask why she was waiting for me."
    ),
]


# Each set is a conjunction: every listed effect family should execute on the
# turn. Turn 10 is continuity-only and intentionally has no tool requirement.
EXPECTED_EFFECTS = {
    1: {"ref_entity"},
    2: {"update_player", "update_entity"},
    3: {"update_entity"},
    4: {"update_player", "update_entity"},
    5: {"update_player"},
    6: {"spawn_object"},
    7: {"update_player"},
    8: {"remove_entity"},
    9: {"update_entity"},
    11: {"change_location"},
    12: {"add_npc"},
}


# The player-state sweep forces every update_player mutation family the
# matrix's tool-reliability track names. Uncontested/no-roll phrasing keeps
# each turn on the narrator-tool path rather than dice mechanics; the
# poison/antidote/charm premises are grounded as established facts by
# _seed_player_casting_state so the declarations are canon, not player fiat.
STATE_SWEEP_ACTIONS = [
    "I greet Mara Venn by name and ask what she has learned about the Ash Gate.",
    (
        # Deliberately purchase-classifier bait (the live T2 defect phrasing,
        # run 20260724_231350): a social payment for help. The single-writer
        # seam must keep the ledger receipt-exact however triage routes it.
        "I pay Mara Venn exactly two gold pieces for her help, and she "
        "accepts them."
    ),
    (
        "This is an uncontested item transfer with no roll: Mara Venn takes "
        "the test draught from her coat and hands it to me; it is now in my "
        "pack rather than her coat."
    ),
    (
        "This is an established automatic effect with no roll: I drink the "
        "test draught, and its mild venom takes hold — I now have the "
        "poisoned condition."
    ),
    (
        "This is an uncontested item transfer with no roll: Mara Venn takes "
        "the silver antidote from her coat and hands it to me; it is now in "
        "my pack rather than her coat."
    ),
    (
        # Deliberately uses a drink verb (the live consumption defect class,
        # run 20260724_232516 T6): consumption defers to the narrator, and
        # Step 5's receipted net covers a forgotten removal either way.
        "This is an established automatic effect with no roll: I drink the "
        "silver antidote, and it purges the venom — my poisoned condition "
        "ends now, and the emptied vial crumbles to inert dust, gone from "
        "my pack."
    ),
    (
        # Passive ward drain, not "I channel/cast": casting language triages
        # as `cast_spell` and the deterministic spell mechanic swallows the
        # turn without any update_player (observed live, run 20260724_231350
        # T7: zero effects executed).
        "I sit deliberately at the warded corner table and let its sigil "
        "take its established due: the ward drains one of my first-level "
        "spell slots — an automatic effect, nothing cast, no roll."
    ),
    (
        # Deliberately the live T8 defect phrasing (run 20260724_231350):
        # the player RECEIVES money. Mistriaged as `purchase`, the old
        # commerce handler charged 2gp and minted an 'unknown item' row.
        "Mara Venn insists on repaying part of my coin: she counts exactly "
        "five silver pieces into my palm, and I accept them."
    ),
    (
        "This is an uncontested item transfer with no roll: I hand my brass "
        "compass to Mara Venn, she accepts it, and it is now in her coat "
        "rather than my pack."
    ),
    (
        "Mara Venn sets a newly revealed obsidian key on the table between "
        "us; it is a distinct object I can pick up."
    ),
    (
        "This is an established automatic trigger with no roll: I do not "
        "touch or use the sealed reliquary. From across the room I say 'now,' "
        "and its already-armed charge finishes the countdown and destroys "
        "the reliquary completely."
    ),
    (
        "I leave the Copper Finch and travel to the Ash Gate, arriving beneath "
        "its cracked black arch."
    ),
    (
        "At the Ash Gate I meet a new, physically present courier named Sable "
        "Quill, introduce myself, and ask why she was waiting for me."
    ),
]

STATE_SWEEP_EXPECTED = {
    1: {"ref_entity"},
    2: {"update_player"},              # currency out
    3: {"update_player", "update_entity"},  # item grant, npc source mirror
    4: {"update_player"},              # condition add
    5: {"update_player", "update_entity"},  # item grant, npc source mirror
    6: {"update_player"},              # condition remove
    7: {"update_player"},              # spell slot expenditure
    8: {"update_player"},              # currency in
    9: {"update_player", "update_entity"},  # item remove, npc destination mirror
    10: {"spawn_object"},
    11: {"remove_entity"},
    12: {"change_location"},
    13: {"add_npc"},
}

SCENARIOS: dict[str, tuple[list[str], dict[int, set[str]]]] = {
    "baseline": (TOOL_ACTIONS, EXPECTED_EFFECTS),
    "player_state_sweep": (STATE_SWEEP_ACTIONS, STATE_SWEEP_EXPECTED),
}


async def _seed_scene(session) -> tuple[str, str]:
    """Create an unambiguous roster and valid player resources."""
    from dnd_bot.game.scene.registry import get_scene_registry
    from dnd_bot.game.knowledge.models import (
        AddEdge,
        AddNode,
        Entity,
        EntityType as GraphEntityType,
        RelationType,
        Relationship,
        slugify,
    )
    from dnd_bot.game.world_state import NPCState
    from dnd_bot.models.inventory import InventoryItem
    from dnd_bot.models.npc import Disposition, EntityType, SceneEntity
    from dnd_bot.data.repositories.inventory_repo import get_inventory_repo

    live = session.manager.get_session(session.channel_id)
    if live is None or live.world_state is None:
        raise RuntimeError("test session did not expose world state")

    world = live.world_state
    world.current_location = "Copper Finch"
    world.location_description = (
        "A rain-dark tavern of copper lamps, scarred tables, and shuttered windows."
    )
    world.established_facts.extend([
        (
            "Mara Venn has already asked Kael to hand her his brass compass "
            "for safekeeping and has explicitly promised to accept it."
        ),
        (
            "The sealed reliquary is an on-table scene object, not in Kael's "
            "inventory. Its demolition charge is armed and the spoken word "
            "'now' completes its countdown without a roll."
        ),
    ])

    mara_id = str(uuid.uuid4())
    mara = NPCState(
        id=mara_id,
        name="Mara Venn",
        location="Copper Finch",
        disposition="friendly",
        description="A sharp-eyed investigator in a charcoal coat.",
        alive=True,
        important=True,
    )
    world.npcs[mara.id] = mara

    bram_id = str(uuid.uuid4())
    old_bram = NPCState(
        id=bram_id,
        name="Old Bram",
        aliases=["the ash ferryman"],
        location="Ash Gate",
        description="The former ferryman who warned travelers away from the gate.",
        alive=False,
        important=True,
    )
    live.campaign_dead_npcs[old_bram.id] = old_bram

    # The fixture represents already-established campaign canon, so seed its
    # graph projection as well as WorldState/registry. Otherwise extractor
    # updates correctly aimed at Mara are rejected only because the synthetic
    # fixture omitted the KG node—not because the product pipeline failed.
    location_id = slugify(world.current_location)
    graph_ops = [
        AddNode(entity=Entity(
            node_id=location_id,
            entity_type=GraphEntityType.LOCATION,
            name=world.current_location,
            campaign_id=live.campaign_id,
        )),
    ]
    for npc in (mara, old_bram):
        graph_ops.append(AddNode(entity=Entity(
            node_id=npc.id,
            entity_type=GraphEntityType.NPC,
            name=npc.name,
            aliases=list(npc.aliases),
            campaign_id=live.campaign_id,
            properties={
                "description": npc.description,
                "alive": str(npc.alive).lower(),
            },
        )))
    graph_ops.append(AddEdge(relationship=Relationship(
        source_id=mara.id,
        target_id=location_id,
        relation_type=RelationType.LOCATED_AT,
        campaign_id=live.campaign_id,
    )))
    graph_rejections = await live.knowledge_graph.apply_operations(graph_ops)
    if graph_rejections:
        raise RuntimeError(f"tool preflight graph seed rejected: {graph_rejections}")

    registry = get_scene_registry(live.campaign_id, live.session_key)
    registry.register_entity(SceneEntity(
        name=mara.name,
        npc_id=mara.id,
        entity_type=EntityType.NPC,
        description=mara.description,
        disposition=Disposition.FRIENDLY,
        important=True,
    ))

    for entity_id, name, description in (
        (
            "brass-compass",
            "brass compass",
            "Kael's palm-sized brass compass, currently in his pack.",
        ),
        (
            "sealed-reliquary",
            "sealed reliquary",
            "A fist-sized iron reliquary fitted with a bronze demolition pin.",
        ),
    ):
        registry.register_entity(SceneEntity(
            id=entity_id,
            name=name,
            entity_type=EntityType.OBJECT,
            description=description,
            disposition=Disposition.NEUTRAL,
        ))
    world.spawn_item("sealed-reliquary", "Iron box with a bronze demolition pin")

    inventory = await get_inventory_repo()
    await inventory.add_item(InventoryItem(
        character_id=session.character.id,
        item_index="brass-compass",
        item_name="brass compass",
        quantity=1,
    ))
    currency = await inventory.get_currency(session.character.id)
    currency.gold = max(currency.gold, 10)
    await inventory.update_currency(currency)
    return mara_id, bram_id


async def _seed_player_casting_state(session, mara_id: str) -> None:
    """Extra fixture for the player-state sweep.

    The level-1 ranger template has no spell slots, so the slot-drain turn
    seeds a two-slot first-level row directly: ``CharacterRepository.create``
    skips zero-max levels and ``update`` only UPDATEs existing rows (found
    live — run 20260724_231350 captured ``[0, 0]`` after a model-only seed),
    so the row must be inserted before ``update`` can sync it. The
    consumables Mara hands over exist in her narrative inventory, and the
    poison/antidote/ward premises are established facts so the sweep's
    uncontested-no-roll declarations are canon rather than player fiat.
    """
    from dnd_bot.data.database import get_database
    from dnd_bot.data.repositories.character_repo import get_character_repo

    character = session.character
    character.spell_slots.level_1 = (2, 2)
    db = await get_database()
    await db.execute(
        "DELETE FROM character_spell_slots "
        "WHERE character_id = ? AND slot_level = 1",
        (character.id,),
    )
    await db.execute(
        "INSERT INTO character_spell_slots "
        "(character_id, slot_level, slots_max, slots_current) "
        "VALUES (?, 1, 2, 2)",
        (character.id,),
    )
    await db.commit()
    char_repo = await get_character_repo()
    # Syncs the remaining model state and invalidates the repo cache so the
    # raw row insert above is visible to every later get_by_id.
    await char_repo.update(character)

    live = session.manager.get_session(session.channel_id)
    world = live.world_state
    mara = world.npcs[mara_id]
    mara.inventory.extend(["test draught", "silver antidote"])
    world.established_facts.extend([
        (
            "Kael owes Mara Venn an old debt of exactly two gold pieces; "
            "repaying it is uncontested, needs no roll, and buys nothing."
        ),
        (
            "Mara Venn carries a stoppered test draught; drinking it is "
            "uncontested, needs no roll, and reliably inflicts the poisoned "
            "condition."
        ),
        (
            "Mara Venn carries a silver antidote; once uncorked, its vapor "
            "automatically and completely ends the poisoned condition — no "
            "roll — and the spent vial then crumbles to inert dust."
        ),
        (
            "The Copper Finch's warded corner table bears an old sigil that, "
            "once per visit, saps exactly one first-level spell slot from a "
            "seated spellcaster; the drain is automatic, harmless otherwise, "
            "and needs no roll."
        ),
    ])


_CURRENCY_FIELDS = ("copper", "silver", "electrum", "gold", "platinum")
# Mirrors _execute_update_player's denomination mapping (effects.py).
_DENOM_FIELDS = {
    "cp": "copper", "sp": "silver", "ep": "electrum", "gp": "gold",
    "pp": "platinum",
}


def _item_index(name: str) -> str:
    """The inventory index _execute_update_player derives from an item name."""
    return name.strip().lower().replace(" ", "-")


async def _capture_player_state(character_id: str) -> dict:
    """Authoritative player-state snapshot straight from the DB repos."""
    from dnd_bot.data.repositories.character_repo import get_character_repo
    from dnd_bot.data.repositories.inventory_repo import get_inventory_repo

    char_repo = await get_character_repo()
    inventory_repo = await get_inventory_repo()
    character = await char_repo.get_by_id(character_id)
    if character is None:
        raise RuntimeError(f"player-state capture: character {character_id} missing")
    currency = await inventory_repo.get_currency(character_id)
    items = await inventory_repo.get_all_items(character_id)
    inventory: dict[str, int] = {}
    for item in items:
        inventory[item.item_index] = inventory.get(item.item_index, 0) + item.quantity
    return {
        "currency": {
            field: getattr(currency, field) for field in _CURRENCY_FIELDS
        },
        "inventory": inventory,
        "conditions": sorted(c.condition.value for c in character.conditions),
        "spell_slots": {
            str(level): list(character.spell_slots.get_slots(level))
            for level in range(1, 10)
        },
    }


def _update_player_receipts(turn_rows: list[dict]) -> list[dict]:
    """Executed update_player receipts ("applied" payloads) in turn order.

    Every sanctioned player-state writer lands here: narrator tools,
    authoritative purchase/inventory effects, and Step 5's deterministic
    consumption all execute through the effect pipeline, so their receipts
    share this one shape. Idempotent duplicates are skipped: their write
    already happened under the first receipt, so counting them again would
    double the ledger.
    """
    receipts: list[dict] = []
    for row in turn_rows:
        for effect in row.get("executed") or []:
            effect_type = str(
                effect.get("type") or effect.get("effect_type") or ""
            )
            if effect_type != "update_player" or effect.get("was_duplicate"):
                continue
            applied = (effect.get("details") or {}).get("applied") or {}
            if applied:
                receipts.append(dict(applied))
    return receipts


def evaluate_player_state_agreement(
    initial: dict,
    final: dict,
    receipts: list[dict],
) -> dict[str, dict]:
    """Receipt-vs-state agreement: replay every update_player receipt over
    the initial DB snapshot and require the result to equal the final DB
    snapshot, per family. A receipt whose write never landed, or a write
    that produced no receipt, both surface as a mismatch (matrix gate:
    "receipt matches DB/WorldState").

    Player numerics have no extractor path by construction, and every
    remaining writer — narrator update_player tools, the authoritative
    purchase/inventory effects, Step 5's deterministic consumption — commits
    through the effect pipeline with an update_player receipt. Any write
    outside that seam shows up here as an unreceipted mismatch —
    deliberately: on its first live run (20260724_231350) that is how this
    gate exposed the commerce handler double-charging a narrated payment and
    charging a player who was being PAID. Attribute mismatches via each
    turn_row's ``triage_action_type`` and triage consumption fields.
    """
    currency_delta: Counter[str] = Counter()
    item_delta: Counter[str] = Counter()
    conditions = set(initial.get("conditions") or [])
    slots_spent: Counter[int] = Counter()
    families = Counter()

    for applied in receipts:
        delta = applied.get("currency_delta") or {}
        if delta:
            families["currency"] += 1
            for key, value in delta.items():
                field = _DENOM_FIELDS.get(str(key).strip().lower()[:2])
                if field and isinstance(value, int):
                    currency_delta[field] += value
        for entry in applied.get("items_granted") or []:
            families["items_granted"] += 1
            item_delta[_item_index(str(entry.get("name") or ""))] += int(
                entry.get("quantity") or 1
            )
        for entry in applied.get("items_removed") or []:
            families["items_removed"] += 1
            item_delta[_item_index(str(entry.get("name") or ""))] -= int(
                entry.get("quantity") or 1
            )
        for value in applied.get("conditions_added") or []:
            families["conditions_added"] += 1
            conditions.add(str(value))
        for value in applied.get("conditions_removed") or []:
            families["conditions_removed"] += 1
            conditions.discard(str(value))
        slot = applied.get("spell_slot_used")
        if isinstance(slot, int):
            families["spell_slot_used"] += 1
            slots_spent[slot] += 1

    checks: dict[str, dict] = {}

    currency_problems = []
    for field in _CURRENCY_FIELDS:
        expected = int((initial.get("currency") or {}).get(field, 0) or 0)
        expected += currency_delta[field]
        actual = int((final.get("currency") or {}).get(field, 0) or 0)
        if expected != actual:
            currency_problems.append(
                f"{field}: initial+receipts={expected} != final={actual}"
            )
    checks["currency_receipts_match_state"] = {
        "passed": not currency_problems,
        "detail": "; ".join(currency_problems)
        or f"delta={dict(currency_delta)} over {families['currency']} receipts",
    }

    item_problems = []
    initial_items = dict(initial.get("inventory") or {})
    final_items = dict(final.get("inventory") or {})
    for index in sorted(
        set(initial_items) | set(final_items) | set(item_delta)
    ):
        expected = initial_items.get(index, 0) + item_delta[index]
        actual = final_items.get(index, 0)
        if expected != actual:
            item_problems.append(
                f"{index}: initial+receipts={expected} != final={actual}"
            )
    checks["inventory_receipts_match_state"] = {
        "passed": not item_problems,
        "detail": "; ".join(item_problems)
        or (
            f"granted={families['items_granted']} "
            f"removed={families['items_removed']} receipts reconciled"
        ),
    }

    final_conditions = set(final.get("conditions") or [])
    checks["condition_receipts_match_state"] = {
        "passed": conditions == final_conditions,
        "detail": (
            f"replayed={sorted(conditions)} final={sorted(final_conditions)}"
        ),
    }

    slot_problems = []
    for level in range(1, 10):
        initial_pair = list(
            (initial.get("spell_slots") or {}).get(str(level)) or (0, 0)
        )
        final_pair = list(
            (final.get("spell_slots") or {}).get(str(level)) or (0, 0)
        )
        expected_current = initial_pair[0] - slots_spent[level]
        if [expected_current, initial_pair[1]] != final_pair:
            slot_problems.append(
                f"L{level}: initial={initial_pair} spent={slots_spent[level]} "
                f"final={final_pair}"
            )
    checks["spell_slot_receipts_match_state"] = {
        "passed": not slot_problems,
        "detail": "; ".join(slot_problems)
        or f"spent={dict(slots_spent)} reconciled",
    }

    required_families = (
        "currency", "items_granted", "items_removed",
        "conditions_added", "conditions_removed", "spell_slot_used",
    )
    missing_families = [
        name for name in required_families if not families[name]
    ]
    checks["receipts_cover_all_player_state_families"] = {
        "passed": not missing_families,
        "detail": (
            f"missing={missing_families}; counts={dict(families)}"
            if missing_families else f"counts={dict(families)}"
        ),
    }
    return checks


def _cost_summary(events: list) -> dict:
    from test_long_horizon import _event_cost

    known_costs = [_event_cost(event) for event in events]
    return {
        "calls": len(events),
        "prompt_tokens": sum(event.prompt_tokens for event in events),
        "completion_tokens": sum(event.completion_tokens for event in events),
        "cache_read_tokens": sum(event.cache_read_tokens for event in events),
        "cost_usd": round(sum(c for c in known_costs if c is not None), 6),
        "cost_complete": all(c is not None for c in known_costs),
    }


async def run(profile: str, scenario: str = "baseline") -> tuple[dict, bool]:
    os.environ["ACTIVE_PROFILE"] = profile
    actions, expected_effects = SCENARIOS[scenario]

    from dnd_bot.llm import usage_recorder
    from dnd_bot.llm.continuity import NarrativeGovernance
    from dnd_bot.llm.turn_log_reader import TurnLogReader
    from dnd_bot.game.identity import resolve_unique_identity
    from test_harness import TestSession

    usage_recorder.enable()
    usage_recorder.reset()

    harness = TestSession(
        combat_policy="fail",
        world_setting=(
            "The Copper Finch stands beside the forbidden Ash Gate. Mara Venn is "
            "a living investigator and ally. Old Bram, the former ferryman, is "
            "unambiguously dead and may appear only as memory or remains."
        ),
    )
    if not await harness.setup():
        raise RuntimeError("tool reliability harness setup failed")

    live = harness.manager.get_session(harness.channel_id)
    session_id = live.id
    mara_id, bram_id = await _seed_scene(harness)
    initial_player_state: dict = {}
    final_player_state: dict = {}
    if scenario == "player_state_sweep":
        await _seed_player_casting_state(harness, mara_id)
        initial_player_state = await _capture_player_state(harness.character.id)
    responses: dict[int, str] = {}
    errors: list[dict] = []
    started = time.time()

    try:
        for turn, action in enumerate(actions, 1):
            response = await harness.send_action(action)
            if response is None:
                errors.append({"turn": turn, "error": "no response"})
                continue
            responses[turn] = response.narrative or ""
    finally:
        # Capture live state before isolated storage is removed.
        live = harness.manager.get_session(harness.channel_id)
        world_snapshot = (
            live.world_state.model_dump(mode="json")
            if live is not None and live.world_state is not None
            else {}
        )
        if scenario == "player_state_sweep":
            final_player_state = await _capture_player_state(
                harness.character.id
            )
        await harness.cleanup()

    elapsed = time.time() - started
    log = TurnLogReader.load(session_id)
    turn_elapsed = {
        entry.get("turn"): entry.get("elapsed")
        for entry in harness.action_log
        if isinstance(entry.get("elapsed"), (int, float))
    }
    turn_rows = []
    type_counts: Counter[str] = Counter()
    proposed_total = executed_total = rejected_total = 0
    expected_passed = 0
    continuity_failures = []
    dead_state_reintroductions = []

    from dnd_bot.game.world_state import NPCState
    dead_fact = NPCState(
        id=bram_id,
        name="Old Bram",
        aliases=["the ash ferryman"],
        alive=False,
    )
    governance = NarrativeGovernance([dead_fact])

    for turn in range(1, len(actions) + 1):
        effects = log.effects_at(turn)
        turn_triage = dict((log.get(turn) or {}).get("triage") or {})
        proposed = list(effects.proposed or [])
        executed = list(effects.executed or effects.effects or [])
        rejected = list(effects.rejected or [])
        executed_types = {
            str(effect.get("type") or effect.get("effect_type") or "")
            for effect in executed
        }
        type_counts.update(executed_types)
        proposed_total += len(proposed)
        executed_total += len(executed)
        rejected_total += len(rejected)
        expected = expected_effects.get(turn, set())
        expected_ok = not expected or expected.issubset(executed_types)
        if expected == {"add_npc"}:
            # The Sable Quill finale in both scripts. The state extractor can
            # establish Sable before the effect leg; in that valid ordering,
            # add_npc is deterministically rewritten to ref_entity so the two
            # channels converge on one canonical NPC.
            sable_count = sum(
                1
                for npc in (world_snapshot.get("npcs") or {}).values()
                if str(npc.get("name", "")).casefold() == "sable quill"
            )
            expected_ok = sable_count == 1 and bool(
                {"add_npc", "ref_entity"}.intersection(executed_types)
            )
        if expected:
            expected_passed += int(expected_ok)

        final_violations = governance.validate(responses.get(turn, ""))
        if final_violations:
            continuity_failures.append({
                "turn": turn,
                "violations": [v.to_prompt_line() for v in final_violations],
            })

        for npc in log.world_state_after(turn).all_npcs():
            if resolve_unique_identity(npc.name, [dead_fact]) is not None:
                dead_state_reintroductions.append({
                    "turn": turn,
                    "name": npc.name,
                    "location": npc.location,
                })

        turn_rows.append({
            "turn": turn,
            "action": actions[turn - 1],
            # Whole-turn wall clock from the harness action log — the
            # multi-run threshold gate's per-turn p95 input.
            "elapsed_seconds": (
                round(turn_elapsed[turn], 3) if turn in turn_elapsed else None
            ),
            # Attribution for ledger mismatches: which handler the turn
            # routed to, since `purchase`/`inventory` turns may claim their
            # player-state write deterministically (as authoritative
            # effects) rather than leaving it to the narrator's tools.
            "triage_action_type": str(turn_triage.get("action_type") or ""),
            # What triage proposed for Step 5's deterministic consumption —
            # attribution for its (receipted, deduped) update_player writes.
            "triage_currency_spent": dict(
                turn_triage.get("currency_spent") or {}
            ),
            "triage_resources_consumed": list(
                turn_triage.get("resources_consumed") or []
            ),
            "executed_types": sorted(executed_types),
            "proposed": proposed,
            "executed": executed,
            "rejected": rejected,
            "expected": sorted(expected),
            "expected_ok": expected_ok,
            "narration_diagnostics": log.narration_diagnostics(turn),
            "narrative": responses.get(turn, ""),
        })

    accounting_balanced = proposed_total == executed_total + rejected_total
    reliability = executed_total / proposed_total if proposed_total else 0.0
    expected_total = len(expected_effects)
    diagnostics = [row["narration_diagnostics"] for row in turn_rows]
    unmet_obligation_turns = [
        {
            "turn": row["turn"],
            "missing": list(
                row["narration_diagnostics"].get(
                    "effect_obligation_missing_final"
                )
                or []
            ),
        }
        for row in turn_rows
        if row["narration_diagnostics"].get("effect_obligation_missing_final")
    ]
    resolved_outcome_failed_closed_turns = [
        row["turn"]
        for row in turn_rows
        if row["narration_diagnostics"].get("resolved_outcome_failed_closed")
    ]
    player_state_agreement: dict[str, dict] = {}
    if scenario == "player_state_sweep":
        player_state_agreement = evaluate_player_state_agreement(
            initial_player_state,
            final_player_state,
            _update_player_receipts(turn_rows),
        )

    gates = {
        "all_turns_returned": len(responses) == len(actions),
        "effect_reliability_at_least_95pct": reliability >= 0.95,
        "effect_accounting_balanced": accounting_balanced,
        "at_least_six_effect_families": len(type_counts) >= 6,
        "expected_turn_coverage_at_least_80pct": (
            expected_passed / expected_total >= 0.80
        ),
        "no_unmet_runtime_effect_obligations": not unmet_obligation_turns,
        "no_resolved_outcome_failed_closed": (
            not resolved_outcome_failed_closed_turns
        ),
        "no_final_dead_npc_contradiction": not continuity_failures,
        "no_dead_npc_state_reintroduction": not dead_state_reintroductions,
        "no_harness_errors": not errors,
        "ended_at_ash_gate": "ash gate" in str(
            world_snapshot.get("current_location", "")
        ).casefold(),
    }
    gates.update({
        name: bool(check.get("passed"))
        for name, check in player_state_agreement.items()
    })

    report = {
        "profile": profile,
        "scenario": scenario,
        "session_id": session_id,
        "turns": len(actions),
        "elapsed_seconds": round(elapsed, 2),
        "usage": _cost_summary(usage_recorder.events()),
        "proposed_total": proposed_total,
        "executed_total": executed_total,
        "rejected_total": rejected_total,
        "effect_reliability": round(reliability, 4),
        "effect_type_counts": dict(type_counts),
        "expected_turns_passed": expected_passed,
        "expected_turns_total": expected_total,
        "tool_followup_turns": [
            row["turn"] for row in turn_rows
            if row["narration_diagnostics"].get("tool_followup_attempted")
        ],
        "tool_repair_turns": [
            row["turn"] for row in turn_rows
            if row["narration_diagnostics"].get("tool_repair_attempted")
        ],
        "structural_error_turns": [
            row["turn"] for row in turn_rows
            if any(
                row["narration_diagnostics"].get(key, 0)
                for key in (
                    "primary_structural_errors",
                    "tool_followup_structural_errors",
                    "tool_repair_structural_errors",
                    "effect_obligation_terminal_structural_errors",
                )
            )
        ],
        "unmet_obligation_turns": unmet_obligation_turns,
        "resolved_outcome_failed_closed_turns": (
            resolved_outcome_failed_closed_turns
        ),
        "continuity_repair_turns": [
            row["turn"] for row in turn_rows
            if row["narration_diagnostics"].get("continuity_repair_attempted")
        ],
        "continuity_failures": continuity_failures,
        "dead_state_reintroductions": dead_state_reintroductions,
        "errors": errors,
        "player_state_initial": initial_player_state,
        "player_state_final": final_player_state,
        "player_state_agreement": player_state_agreement,
        "gates": gates,
        "turn_rows": turn_rows,
    }
    return report, all(gates.values())


def _print_report(report: dict, passed: bool, artifact: Path) -> None:
    print("\n" + "=" * 72)
    print(f"TOOL RELIABILITY: {'PASS' if passed else 'FAIL'}")
    print("=" * 72)
    print(
        f"effects: proposed={report['proposed_total']} "
        f"executed={report['executed_total']} rejected={report['rejected_total']} "
        f"reliability={report['effect_reliability']:.1%}"
    )
    print(f"families: {report['effect_type_counts']}")
    print(
        f"expected turns: {report['expected_turns_passed']}/"
        f"{report['expected_turns_total']}"
    )
    print(
        f"followup turns={report['tool_followup_turns']} "
        f"repair turns={report['tool_repair_turns']} "
        f"structural errors={report['structural_error_turns']}"
    )
    print(
        f"unmet obligations={report['unmet_obligation_turns']} "
        "resolved-outcome failed-closed="
        f"{report['resolved_outcome_failed_closed_turns']}"
    )
    print(
        f"cost=${report['usage']['cost_usd']:.5f} "
        f"elapsed={report['elapsed_seconds']:.1f}s"
    )
    for name, check in (report.get("player_state_agreement") or {}).items():
        print(f"  agreement {name}: {check.get('detail')}")
    for name, ok in report["gates"].items():
        print(f"  {'PASS' if ok else 'FAIL'}  {name}")
    print(f"artifact: {artifact}")


async def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--profile",
        default="deepseek_v4_flash_qwen9b",
        help="Profile from config/profiles.yaml",
    )
    parser.add_argument(
        "--scenario",
        default="baseline",
        choices=sorted(SCENARIOS),
        help="Turn script to run (see module docstring)",
    )
    args = parser.parse_args()

    report, passed = await run(args.profile, scenario=args.scenario)
    out_dir = Path("data/tool_reliability")
    out_dir.mkdir(parents=True, exist_ok=True)
    scenario_tag = "" if args.scenario == "baseline" else f"{args.scenario}_"
    artifact = out_dir / (
        f"{time.strftime('%Y%m%d_%H%M%S')}_{scenario_tag}{args.profile}.json"
    )
    artifact.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _print_report(report, passed, artifact)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
