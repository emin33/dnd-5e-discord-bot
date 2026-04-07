"""Generic combat turn loop - frontend-agnostic.

Orchestrates combat turns through a GameFrontend. Both Discord text
and voice frontends use this same loop. The frontend handles presentation
(buttons vs voice prompts) while this module handles game logic flow.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import structlog

from .actions import CombatActionType
from ..frontend import GameFrontend, GameEvent

if TYPE_CHECKING:
    from .coordinator import CombatTurnCoordinator
    from .manager import CombatManager

logger = structlog.get_logger()

MAX_NPC_TURNS = 10  # Safety limit to prevent infinite NPC loops


async def run_combat_loop(
    frontend: GameFrontend,
    coordinator: CombatTurnCoordinator,
    manager: CombatManager,
) -> None:
    """Run the full combat loop from current combatant forward.

    Handles:
    - Skipping surprised combatants
    - Player turns via frontend.get_combat_action()
    - NPC auto-turns via coordinator.run_npc_turn()
    - Combat end detection
    - Round advancement announcements
    """
    current = manager.combat.get_current_combatant()
    if not current:
        return

    # Skip surprised combatants at the start
    while current and not current.is_player and current.is_surprised:
        await frontend.on_event(GameEvent.error(
            f"**{current.name}** is surprised and cannot act!"
        ))
        await coordinator.start_turn(current)
        await coordinator.end_turn(current)
        current = manager.combat.get_current_combatant()

    if not current:
        return

    if current.is_player:
        await _run_player_turn(frontend, coordinator, manager, current)
    else:
        await _run_npc_turns(frontend, coordinator, manager)


async def _run_player_turn(
    frontend: GameFrontend,
    coordinator: CombatTurnCoordinator,
    manager: CombatManager,
    combatant,
) -> None:
    """Run a single player's turn via the frontend."""
    turn_ctx = await coordinator.start_turn(combatant)

    # Emit turn prompt to frontend (shows buttons or speaks options)
    await frontend.on_event(GameEvent.turn_prompt(turn_ctx))

    # Await player's combat action choice (button click or voice command)
    action = await frontend.get_combat_action(turn_ctx)

    # Execute the action
    result = await coordinator.execute_action(action)

    # Narrate the result
    narrative = None
    if result.action.action_type != CombatActionType.END_TURN:
        try:
            narrative = await coordinator.narrate_result(result)
        except Exception as e:
            logger.warning("combat_narration_failed", error=str(e))

    # Send action result + narrative to frontend
    await frontend.on_event(GameEvent.action_result(result, narrative))

    # Check combat end
    if manager.combat.is_combat_over():
        await _handle_combat_end(frontend, coordinator, manager)
        return

    # End turn and advance
    end_result = await coordinator.end_turn(combatant)
    await frontend.on_event(GameEvent.turn_end(
        next_combatant_name=end_result.next_combatant_name,
        next_is_player=end_result.next_is_player,
        round_advanced=end_result.round_advanced,
        new_round=end_result.new_round,
    ))

    # Continue to NPC turns if next combatant is NPC
    if not end_result.next_is_player:
        next_combatant = manager.combat.get_current_combatant()
        if next_combatant:
            await _run_npc_turns(frontend, coordinator, manager)


async def _run_npc_turns(
    frontend: GameFrontend,
    coordinator: CombatTurnCoordinator,
    manager: CombatManager,
) -> None:
    """Auto-run all consecutive NPC turns, then hand off to next player."""
    turns_run = 0

    while turns_run < MAX_NPC_TURNS:
        current = manager.combat.get_current_combatant()
        if not current or current.is_player:
            break

        turns_run += 1

        try:
            results = await coordinator.run_npc_turn(current)
        except Exception as e:
            logger.error("npc_turn_failed", combatant=current.name, error=str(e))
            await frontend.on_event(GameEvent.error(
                f"*{current.name} hesitates...* (Error: {str(e)[:80]})"
            ))
            try:
                await coordinator.end_turn(current)
            except Exception:
                pass
            continue

        # Send mechanical results
        narratable_results = []
        for result in results:
            if result.action.action_type != CombatActionType.END_TURN:
                narratable_results.append(result)
            # Send each result without narrative first
            await frontend.on_event(GameEvent.action_result(result, narrative=None))

        # Batch-narrate all NPC actions in one LLM call
        if narratable_results:
            try:
                narrative = await coordinator.narrate_turn_results(narratable_results)
                if narrative:
                    await frontend.on_event(GameEvent.narrative_complete(narrative))
            except Exception as e:
                logger.warning("npc_narration_failed", error=str(e))

        # Check combat end
        if manager.combat.is_combat_over():
            await _handle_combat_end(frontend, coordinator, manager)
            return

    # After NPC turns, run next player's turn
    current = manager.combat.get_current_combatant()
    if current and current.is_player:
        await _run_player_turn(frontend, coordinator, manager, current)


async def _handle_combat_end(
    frontend: GameFrontend,
    coordinator: CombatTurnCoordinator,
    manager: CombatManager,
) -> None:
    """Handle combat ending - cleanup and notify frontend."""
    from .manager import clear_combat_by_key
    from .coordinator import clear_coordinator_by_key

    players_alive = any(
        c.is_player and c.hp_current > 0
        for c in manager.combat.combatants
    )

    await frontend.on_event(GameEvent.combat_end(victory=players_alive))

    # Get session key for cleanup
    session_key = None
    if coordinator.session and hasattr(coordinator.session, "session_key"):
        session_key = coordinator.session.session_key

    manager.end_combat()

    if session_key:
        clear_combat_by_key(session_key)
        clear_coordinator_by_key(session_key)
