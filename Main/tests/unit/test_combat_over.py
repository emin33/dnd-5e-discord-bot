"""Adversarial review, blocker 1: a combat-ending turn advance must be a
first-class outcome, not a crash.

``Combat.VALID_TRANSITIONS`` rejected AWAITING_ACTION -> COMBAT_END, so when
a turn advance ended combat with living winners (NPC-side victory / TPK:
manager.next_turn() lands on the surviving goblin, THEN sees
is_combat_over()), ``end_combat()``'s transition silently returned False and
``next_turn`` returned ``(None, ...)``. ``coordinator._end_turn_locked``
dereferenced the ``None`` combatant -> AttributeError, bricking the NPC views
and wedging the session in COMBAT.

These tests pin the fixed contract: the transition is legal, ``end_turn``
returns a typed ``TurnEndResult(combat_over=True)``, ``run_npc_turn``
completes a TPK turn without raising, teardown through the single owner
leaves the session ACTIVE with clean registries, and a dead encounter never
re-processes end-of-turn effects (the repeated-/combat-next DoT double-tick,
blocker 2).
"""

import pytest

from dnd_bot.game.combat.actions import CombatAction, CombatActionType
from dnd_bot.game.combat.coordinator import (
    CombatTurnCoordinator,
    clear_coordinator_by_key,
    get_coordinator,
    get_coordinator_by_key,
)
from dnd_bot.game.combat.manager import (
    CombatManager,
    clear_combat_by_key,
    get_combat_by_key,
    set_combat_for_channel,
)
from dnd_bot.game.session import GameSession, GameSessionManager, SessionState
from dnd_bot.models import CombatState

# Distinct from channel ids used elsewhere in the suite — the combat,
# coordinator, and turn-lock registries are module-level globals.
CHANNEL = 557_001
KEY = f"discord:{CHANNEL}"


def _make_combat(character) -> CombatManager:
    """Mid-combat encounter with a pinned order: player first, goblin second."""
    manager = CombatManager.create_encounter(
        session_id="combat-over-test-session",
        channel_id=CHANNEL,
        name="Combat Over Test",
    )
    manager.add_player(character)
    manager.add_custom_combatant(name="Goblin", hp=7, ac=13)
    manager.start_combat()
    # Initiative rolls are random — pin the order for determinism.
    player = next(c for c in manager.combat.combatants if c.is_player)
    goblin = next(c for c in manager.combat.combatants if not c.is_player)
    player.turn_order = 0
    goblin.turn_order = 1
    manager.combat.current_turn_index = 1  # goblin is acting
    return manager


@pytest.fixture(autouse=True)
def _isolated_registries():
    """Leave the module-global registries (and lock entries) clean."""
    clear_combat_by_key(KEY)
    clear_coordinator_by_key(KEY)
    yield
    clear_combat_by_key(KEY)
    clear_coordinator_by_key(KEY)


class TestCombatEndingAdvance:
    """end_turn / next_turn when the outgoing turn ended the encounter."""

    async def test_end_turn_after_last_player_downed_returns_combat_over(
        self, mock_character
    ):
        """The empirical repro: goblin downs the last conscious player, then
        its turn ends. Previously AttributeError; now a typed result."""
        manager = _make_combat(mock_character)
        player = next(c for c in manager.combat.combatants if c.is_player)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)
        player.hp_current = 0  # last conscious player just went down

        coordinator = CombatTurnCoordinator(manager)
        result = await coordinator.end_turn(goblin)

        assert result.combat_over is True
        assert result.next_combatant_id == ""
        # The transition fix: the encounter actually reaches COMBAT_END
        # instead of silently staying AWAITING_ACTION.
        assert manager.combat.state == CombatState.COMBAT_END
        assert manager.combat.ended_at is not None

    async def test_next_turn_ending_combat_reaches_combat_end(self, mock_character):
        """Manager-level pin of the AWAITING_ACTION -> COMBAT_END edge."""
        manager = _make_combat(mock_character)
        player = next(c for c in manager.combat.combatants if c.is_player)
        player.hp_current = 0
        assert manager.combat.state == CombatState.AWAITING_ACTION

        next_combatant, _end, _start, _recharge = manager.next_turn()

        assert next_combatant is None
        assert manager.combat.state == CombatState.COMBAT_END

    async def test_dead_encounter_does_not_reprocess_end_of_turn_effects(
        self, mock_character, monkeypatch
    ):
        """Blocker 2 follow-through: repeated advances on an ended encounter
        must be inert — no second end-of-turn effect tick (DoT double-tick)."""
        manager = _make_combat(mock_character)
        player = next(c for c in manager.combat.combatants if c.is_player)
        player.hp_current = 0

        tick_counts: list[str] = []
        real_process = manager.process_end_of_turn_effects

        def counted(combatant):
            tick_counts.append(combatant.name)
            return real_process(combatant)

        monkeypatch.setattr(manager, "process_end_of_turn_effects", counted)

        manager.next_turn()  # ends combat, ticks the goblin's effects once
        assert manager.combat.state == CombatState.COMBAT_END
        assert tick_counts == ["Goblin"]

        result = manager.next_turn()  # a stray /combat next on a dead encounter

        assert result == (None, [], [], [])
        assert tick_counts == ["Goblin"]  # no second tick


class TestNpcTurnTpk:
    """run_npc_turn completing a TPK, then teardown via the single owner."""

    async def test_npc_turn_downing_last_player_ends_cleanly(
        self, mock_character, monkeypatch
    ):
        manager = _make_combat(mock_character)
        player = next(c for c in manager.combat.combatants if c.is_player)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)

        class _LethalBrain:
            """Downs the last conscious player during its decision."""

            def roll_recharge(self, combatant):
                return []

            async def decide_action(self, combatant, combat_state, zones):
                manager.apply_damage(player.id, 999)
                return CombatAction(
                    action_type=CombatActionType.END_TURN,
                    combatant_id=combatant.id,
                )

        monkeypatch.setattr(
            "dnd_bot.game.combat.npc_brain.get_npc_brain", lambda: _LethalBrain()
        )

        async def _fake_persist(self):
            pass

        monkeypatch.setattr(
            CombatTurnCoordinator, "persist_player_characters", _fake_persist
        )

        session = GameSession(
            id="combat-over-session",
            channel_id=CHANNEL,
            guild_id=1,
            campaign_id="combat-over-campaign",
            state=SessionState.COMBAT,
        )
        session.combat_manager = manager
        set_combat_for_channel(CHANNEL, manager)
        coordinator = get_coordinator(manager)
        sessions = GameSessionManager()
        sessions._sessions[session.session_key] = session

        # Previously raised AttributeError from _end_turn_locked.
        results = await coordinator.run_npc_turn(goblin)

        assert results == []  # END_TURN decision, no executed actions
        assert manager.combat.is_combat_over()
        assert player.hp_current == 0

        # Teardown through the single owner unwinds everything.
        assert await sessions.end_combat(CHANNEL) is True
        assert manager.combat.state == CombatState.COMBAT_END
        assert session.state == SessionState.ACTIVE
        assert session.combat_manager is None
        assert get_combat_by_key(KEY) is None
        assert get_coordinator_by_key(KEY) is None


class TestTornDownCoordinatorGuards:
    """Adversarial review, should-fix 2: a coroutine parked on the turn lock
    can resume AFTER GameSessionManager.end_combat finalized the manager and
    cleared the registries. The ``*_locked`` bodies must return typed no-op
    results against the torn-down combat instead of mutating or crashing."""

    async def _torn_down(self, mock_character, monkeypatch):
        """A registered combat torn down through the single owner."""
        manager = _make_combat(mock_character)
        set_combat_for_channel(CHANNEL, manager)
        coordinator = get_coordinator(manager)

        async def _fake_persist(self):
            pass

        monkeypatch.setattr(
            CombatTurnCoordinator, "persist_player_characters", _fake_persist
        )
        sessions = GameSessionManager()
        assert await sessions.end_combat(CHANNEL) is True
        assert manager.combat.state == CombatState.COMBAT_END
        return manager, coordinator

    async def test_locked_calls_after_teardown_are_noops(
        self, mock_character, monkeypatch
    ):
        manager, coordinator = await self._torn_down(mock_character, monkeypatch)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)
        goblin.turn_resources.action = True
        turn_index_before = manager.combat.current_turn_index

        # execute_action: typed failure, no resource consumed
        result = await coordinator.execute_action(
            CombatAction(
                action_type=CombatActionType.DASH,
                combatant_id=goblin.id,
            )
        )
        assert result.success is False
        assert "ended" in (result.error or "").lower()
        assert goblin.turn_resources.action is True  # untouched

        # end_turn: combat-over result, initiative not advanced
        end_result = await coordinator.end_turn(goblin)
        assert end_result.combat_over is True
        assert manager.combat.current_turn_index == turn_index_before

        # start_turn: flagged no-op context, no resources granted
        turn_ctx = await coordinator.start_turn(goblin)
        assert turn_ctx.combat_over is True
        assert turn_ctx.has_action is False

        # run_npc_turn: no actions executed
        assert await coordinator.run_npc_turn(goblin) == []

    async def test_replaced_coordinator_is_stale(self, mock_character):
        """A NEW combat re-registering the channel makes calls from the OLD
        coordinator no-ops even though its own manager was never finalized."""
        manager_old = _make_combat(mock_character)
        set_combat_for_channel(CHANNEL, manager_old)
        coordinator_old = get_coordinator(manager_old)

        clear_coordinator_by_key(KEY)
        manager_new = _make_combat(mock_character)
        set_combat_for_channel(CHANNEL, manager_new)
        coordinator_new = get_coordinator(manager_new)
        assert coordinator_new is not coordinator_old
        assert manager_old.combat.state == CombatState.AWAITING_ACTION

        goblin_old = next(
            c for c in manager_old.combat.combatants if not c.is_player
        )
        result = await coordinator_old.execute_action(
            CombatAction(
                action_type=CombatActionType.DASH,
                combatant_id=goblin_old.id,
            )
        )
        assert result.success is False

        end_result = await coordinator_old.end_turn(goblin_old)
        assert end_result.combat_over is True
        # The new combat is unaffected and fully operational.
        assert manager_new.combat.state == CombatState.AWAITING_ACTION
