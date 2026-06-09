"""Audit P0-6: combat-turn mutations serialize on a per-channel lock.

CombatTurnCoordinator owns one asyncio.Lock per registry key (module-level
``_turn_locks`` next to the coordinator registry, reachable via
``get_turn_lock``/``get_turn_lock_for_channel``). Every public mutating
method — start_turn / end_turn / execute_action / run_npc_turn — acquires
it and delegates to an unlocked ``*_locked`` impl, so the three concurrent
entry-point paths (button views, /combat slash commands, the on_message
combat-UI flow) all serialize at this single game-layer chokepoint.
Teardown (GameSessionManager.end_combat) takes the same lock so it cannot
interleave with a half-finished turn, and clearing the coordinator
registry drops the lock entry.

These tests pin: atomic interleaving of concurrent mutations, the NPC
brain decision staying inside the lock (turn atomicity), per-channel lock
isolation, registry-mirrored keying, and teardown-takes-the-lock.
"""

import asyncio

import pytest

from dnd_bot.game.combat.actions import ActionResult, CombatAction, CombatActionType
from dnd_bot.game.combat.coordinator import (
    CombatTurnCoordinator,
    clear_coordinator_by_key,
    get_coordinator,
    get_turn_lock,
    get_turn_lock_for_channel,
)
from dnd_bot.game.combat.manager import (
    CombatManager,
    clear_combat_by_key,
    set_combat_for_channel,
)
from dnd_bot.game.session import GameSession, GameSessionManager

# Distinct from channel ids used elsewhere in the suite — the combat,
# coordinator, and turn-lock registries are module-level globals.
CHANNEL = 555_001
OTHER_CHANNEL = 555_002
KEY = f"discord:{CHANNEL}"
OTHER_KEY = f"discord:{OTHER_CHANNEL}"
VOICE_KEY = "voice:turn-lock-room"


def _make_combat(character=None, channel_id: int = CHANNEL) -> CombatManager:
    """A mid-combat encounter (initiative rolled, AWAITING_ACTION)."""
    manager = CombatManager.create_encounter(
        session_id="turn-lock-test-session",
        channel_id=channel_id,
        name="Turn Lock Test",
    )
    if character is not None:
        manager.add_player(character)
    manager.add_custom_combatant(name="Goblin", hp=7, ac=13)
    manager.start_combat()
    return manager


def _dash(combatant_id: str) -> CombatAction:
    return CombatAction(
        action_type=CombatActionType.DASH,
        combatant_id=combatant_id,
    )


@pytest.fixture(autouse=True)
def _isolated_registries():
    """Leave the module-global registries (and lock entries) clean."""
    for key in (KEY, OTHER_KEY, VOICE_KEY):
        clear_combat_by_key(key)
        clear_coordinator_by_key(key)
    yield
    for key in (KEY, OTHER_KEY, VOICE_KEY):
        clear_combat_by_key(key)
        clear_coordinator_by_key(key)


class TestTurnMutationSerialization:
    """Concurrent mutating calls must run one at a time, not interleave."""

    async def test_concurrent_execute_action_runs_atomically(self, monkeypatch):
        combat = _make_combat()
        coordinator = CombatTurnCoordinator(combat)
        events: list[str] = []

        async def probe(self, action):
            events.append(f"enter-{action.combatant_id}")
            # Unserialized callers would interleave at these yield points
            await asyncio.sleep(0)
            await asyncio.sleep(0)
            events.append(f"exit-{action.combatant_id}")
            return ActionResult(action=action, success=True)

        monkeypatch.setattr(
            CombatTurnCoordinator, "_execute_action_locked", probe
        )

        await asyncio.wait_for(
            asyncio.gather(
                coordinator.execute_action(_dash("a")),
                coordinator.execute_action(_dash("b")),
            ),
            timeout=5,
        )

        # Strictly enter/exit pairs — never enter-a, enter-b, ...
        assert events == ["enter-a", "exit-a", "enter-b", "exit-b"]

    async def test_npc_turn_holds_lock_across_brain_decision(
        self, mock_character, monkeypatch
    ):
        """The decide->execute window stays under the lock (turn atomicity),
        and run_npc_turn's internal start/execute/end calls don't deadlock."""
        combat = _make_combat(mock_character)
        coordinator = CombatTurnCoordinator(combat)
        goblin = next(c for c in combat.combat.combatants if not c.is_player)
        goblin.is_surprised = False
        locked_during_decision: list[bool] = []

        class _FakeBrain:
            def roll_recharge(self, combatant):
                return []

            async def decide_action(self, combatant, combat_state, zones):
                locked_during_decision.append(coordinator.turn_lock.locked())
                await asyncio.sleep(0)
                return CombatAction(
                    action_type=CombatActionType.END_TURN,
                    combatant_id=combatant.id,
                )

        monkeypatch.setattr(
            "dnd_bot.game.combat.npc_brain.get_npc_brain", lambda: _FakeBrain()
        )

        await asyncio.wait_for(coordinator.run_npc_turn(goblin), timeout=5)

        assert locked_during_decision == [True]
        assert coordinator.turn_lock.locked() is False  # released afterwards


class TestLockKeying:
    """One lock per channel/session key, mirroring the coordinator registry."""

    def test_lock_key_mirrors_registry_keying(self):
        combat = _make_combat()

        # Channel-keyed (no session) — the /combat cog path
        coordinator = CombatTurnCoordinator(combat)
        assert coordinator.turn_lock is get_turn_lock(KEY)
        assert get_turn_lock_for_channel(CHANNEL) is get_turn_lock(KEY)

        # Session-keyed — voice/web frontends
        session = GameSession(
            id="turn-lock-session",
            channel_id=CHANNEL,
            guild_id=1,
            campaign_id="turn-lock-campaign",
            session_key=VOICE_KEY,
        )
        keyed = CombatTurnCoordinator(combat, session)
        assert keyed.turn_lock is get_turn_lock(VOICE_KEY)
        assert keyed.turn_lock is not coordinator.turn_lock

    async def test_channel_a_lock_does_not_block_channel_b(self, monkeypatch):
        combat_b = _make_combat(channel_id=OTHER_CHANNEL)
        coordinator_b = CombatTurnCoordinator(combat_b)

        async def quick(self, action):
            return ActionResult(action=action, success=True)

        monkeypatch.setattr(
            CombatTurnCoordinator, "_execute_action_locked", quick
        )

        assert get_turn_lock_for_channel(CHANNEL) is not get_turn_lock_for_channel(
            OTHER_CHANNEL
        )

        # Hold channel A's lock; channel B's turn must complete immediately.
        async with get_turn_lock_for_channel(CHANNEL):
            result = await asyncio.wait_for(
                coordinator_b.execute_action(_dash("b")), timeout=1
            )
        assert result.success


class TestTeardownTakesTheLock:
    """end_combat serializes behind in-flight turns and drops the lock entry."""

    async def test_teardown_waits_for_inflight_turn_and_drops_lock(
        self, mock_character, monkeypatch
    ):
        combat = _make_combat(mock_character)
        set_combat_for_channel(CHANNEL, combat)
        coordinator = get_coordinator(combat)

        async def _fake_persist(self):
            pass

        monkeypatch.setattr(
            CombatTurnCoordinator, "persist_player_characters", _fake_persist
        )

        events: list[str] = []

        async def slow_action(self, action):
            events.append("turn-enter")
            await asyncio.sleep(0.05)
            events.append("turn-exit")
            return ActionResult(action=action, success=True)

        monkeypatch.setattr(
            CombatTurnCoordinator, "_execute_action_locked", slow_action
        )

        lock_before = get_turn_lock(KEY)
        sessions = GameSessionManager()

        turn = asyncio.create_task(coordinator.execute_action(_dash("x")))
        # Let the turn enter its critical section before teardown fires
        while not events:
            await asyncio.sleep(0)

        assert await asyncio.wait_for(sessions.end_combat(CHANNEL), timeout=5) is True
        events.append("teardown-done")
        await turn

        # Teardown could not interleave with the half-finished turn
        assert events == ["turn-enter", "turn-exit", "teardown-done"]
        # Clearing the coordinator registry dropped the lock entry:
        # a later combat on this channel gets a fresh lock.
        assert get_turn_lock(KEY) is not lock_before
