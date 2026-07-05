"""Audit P1-15: force_advance_turn unwedges combat after a failed turn.

The bot-layer NPC auto-run loop (bot/cogs/game.py:_auto_run_npc_turns)
recovers from a failed run_npc_turn with a ladder: coordinator.end_turn ->
coordinator.force_advance_turn() -> GameSessionManager.end_combat. The cog
branch itself is not importable under the test python (py-cord conflict),
so these tests pin the game-layer primitive the ladder relies on: a
raising end_turn leaves initiative recoverable, force_advance_turn moves
it under the turn lock, and it returns None (the cog's abort signal) when
combat is over.
"""

import pytest

from dnd_bot.game.combat.coordinator import CombatTurnCoordinator
from dnd_bot.game.combat.manager import CombatManager


def _make_combat(channel_id: int, character) -> CombatManager:
    """Mid-combat encounter with a pinned order: player first, goblin second.

    ``channel_id`` comes from the run-unique ``unique_channel_id`` fixture —
    the combat/coordinator/turn-lock registries are module-level globals.
    """
    manager = CombatManager.create_encounter(
        session_id="force-advance-test-session",
        channel_id=channel_id,
        name="Force Advance Test",
    )
    manager.add_player(character)
    manager.add_custom_combatant(name="Goblin", hp=7, ac=13)
    manager.start_combat()
    # Initiative rolls are random — pin the order for determinism.
    player = next(c for c in manager.combat.combatants if c.is_player)
    goblin = next(c for c in manager.combat.combatants if not c.is_player)
    player.turn_order = 0
    goblin.turn_order = 1
    manager.combat.current_turn_index = 0
    return manager


class TestForceAdvanceTurn:
    """The last-resort advance primitive behind the cog's recovery ladder."""

    async def test_advances_initiative_and_returns_next_combatant(
        self, mock_character, unique_channel_id
    ):
        manager = _make_combat(unique_channel_id, mock_character)
        coordinator = CombatTurnCoordinator(manager)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)

        advanced = await coordinator.force_advance_turn()

        assert advanced is goblin
        assert manager.combat.get_current_combatant() is goblin

    async def test_recovers_after_end_turn_failure(
        self, mock_character, unique_channel_id, monkeypatch
    ):
        """The audit P1-15 wedge: end_turn raises and initiative stays parked
        on the broken combatant — force_advance_turn must still move it."""
        manager = _make_combat(unique_channel_id, mock_character)
        coordinator = CombatTurnCoordinator(manager)
        player = manager.combat.get_current_combatant()
        goblin = next(c for c in manager.combat.combatants if not c.is_player)

        def boom(combatant_id):
            raise RuntimeError("zone tracker corrupted")

        monkeypatch.setattr(coordinator.zone_tracker, "on_turn_end", boom)

        with pytest.raises(RuntimeError):
            await coordinator.end_turn(player)

        # Wedged: same combatant's turn, but the lock was released.
        assert manager.combat.get_current_combatant() is player
        assert coordinator.turn_lock.locked() is False

        advanced = await coordinator.force_advance_turn()

        assert advanced is goblin
        assert manager.combat.get_current_combatant() is goblin

    async def test_returns_none_when_combat_is_over(
        self, mock_character, unique_channel_id
    ):
        """None is the caller's abort signal — the cog ends combat via the
        single teardown owner (GameSessionManager.end_combat)."""
        manager = _make_combat(unique_channel_id, mock_character)
        coordinator = CombatTurnCoordinator(manager)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)
        goblin.hp_current = 0  # last enemy down -> combat over during advance

        advanced = await coordinator.force_advance_turn()

        assert advanced is None

    async def test_runs_under_the_turn_lock(
        self, mock_character, unique_channel_id, monkeypatch
    ):
        """Matches the P0-6 locking discipline: the advance mutates turn
        state, so it must hold the per-channel turn lock."""
        manager = _make_combat(unique_channel_id, mock_character)
        coordinator = CombatTurnCoordinator(manager)
        locked_during_advance: list[bool] = []
        real_next_turn = manager.next_turn

        def probe():
            locked_during_advance.append(coordinator.turn_lock.locked())
            return real_next_turn()

        monkeypatch.setattr(manager, "next_turn", probe)

        await coordinator.force_advance_turn()

        assert locked_during_advance == [True]
        assert coordinator.turn_lock.locked() is False
