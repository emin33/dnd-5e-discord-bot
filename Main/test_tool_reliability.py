"""Cheap, deterministic narrator-tool reliability preflight.

This is deliberately shorter and less creative than ``test_long_horizon``.
It seeds a tiny valid scene, issues state-transition-heavy player actions,
and checks the production turn log for executed/rejected effects and repair
telemetry. Run this before paying for a long narrative soak.

Usage::

    python test_tool_reliability.py --profile deepseek_v4_flash_qwen9b
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


async def run(profile: str) -> tuple[dict, bool]:
    os.environ["ACTIVE_PROFILE"] = profile

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
    _, bram_id = await _seed_scene(harness)
    responses: dict[int, str] = {}
    errors: list[dict] = []
    started = time.time()

    try:
        for turn, action in enumerate(TOOL_ACTIONS, 1):
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
        await harness.cleanup()

    elapsed = time.time() - started
    log = TurnLogReader.load(session_id)
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

    for turn in range(1, len(TOOL_ACTIONS) + 1):
        effects = log.effects_at(turn)
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
        expected = EXPECTED_EFFECTS.get(turn, set())
        expected_ok = not expected or expected.issubset(executed_types)
        if turn == 12:
            # The state extractor can establish Sable before the effect leg.
            # In that valid ordering, add_npc is deterministically rewritten
            # to ref_entity so the two channels converge on one canonical NPC.
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
            "action": TOOL_ACTIONS[turn - 1],
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
    expected_total = len(EXPECTED_EFFECTS)
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
    gates = {
        "all_turns_returned": len(responses) == len(TOOL_ACTIONS),
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

    report = {
        "profile": profile,
        "session_id": session_id,
        "turns": len(TOOL_ACTIONS),
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
    args = parser.parse_args()

    report, passed = await run(args.profile)
    out_dir = Path("data/tool_reliability")
    out_dir.mkdir(parents=True, exist_ok=True)
    artifact = out_dir / (
        f"{time.strftime('%Y%m%d_%H%M%S')}_{args.profile}.json"
    )
    artifact.write_text(json.dumps(report, indent=2, default=str), encoding="utf-8")
    _print_report(report, passed, artifact)
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
