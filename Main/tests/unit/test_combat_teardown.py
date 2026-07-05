"""Audit P0-3: single game-layer owner for combat teardown.

``GameSessionManager.end_combat`` is the ONLY path that ends combat: it
persists player combatants (via the coordinator's one implementation),
finalizes the CombatManager, clears BOTH module registries (combat manager
+ turn coordinator), and returns the session to ACTIVE with its
``combat_manager`` reference dropped.

Previously this was copy-pasted at 9 bot-layer sites (7 in cogs/combat.py,
2 in cogs/game.py) with drifted contents — only some cleared the
coordinator registry, only some persisted characters, and NONE reset
``session.state``, so a session reported COMBAT forever after its first
fight. These tests pin the unified owner's invariants.
"""

import pytest
from structlog.testing import capture_logs

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
    set_combat_by_key,
    set_combat_for_channel,
)
from dnd_bot.game.session import GameSession, GameSessionManager, SessionState
from dnd_bot.models import CombatState

# Channel ids come from the run-unique ``unique_channel_id`` fixture — the
# combat and coordinator registries are module-level globals. Voice keys
# aren't channel-derived, so this one stays a constant.
VOICE_KEY = "voice:teardown-room"


def _make_session(channel_id: int, session_key: str = "") -> GameSession:
    """A session already in COMBAT, the state the owner must unwind."""
    return GameSession(
        id="teardown-test-session",
        channel_id=channel_id,
        guild_id=1,
        campaign_id="teardown-campaign",
        session_key=session_key,
        state=SessionState.COMBAT,
    )


def _make_combat(channel_id: int, character=None) -> CombatManager:
    """A mid-combat encounter (initiative rolled, AWAITING_ACTION)."""
    manager = CombatManager.create_encounter(
        session_id="teardown-test-session",
        channel_id=channel_id,
        name="Teardown Test",
    )
    if character is not None:
        manager.add_player(character)
    manager.add_custom_combatant(name="Goblin", hp=7, ac=13)
    manager.start_combat()
    return manager


def _make_session_manager(session: GameSession = None) -> GameSessionManager:
    """A fresh (non-singleton) manager, optionally seeded with a session."""
    sessions = GameSessionManager()
    if session is not None:
        sessions._sessions[session.session_key] = session
    return sessions


@pytest.fixture(autouse=True)
def _isolated_voice_registry():
    """Leave the VOICE_KEY registry entries clean even if a test fails."""
    clear_combat_by_key(VOICE_KEY)
    clear_coordinator_by_key(VOICE_KEY)
    yield
    clear_combat_by_key(VOICE_KEY)
    clear_coordinator_by_key(VOICE_KEY)


@pytest.fixture
def persist_calls(monkeypatch):
    """Record persist_player_characters calls instead of touching the DB."""
    calls: list[CombatTurnCoordinator] = []

    async def _fake_persist(self):
        calls.append(self)

    monkeypatch.setattr(
        CombatTurnCoordinator, "persist_player_characters", _fake_persist
    )
    return calls


class TestEndCombatOwner:
    """The owner's core invariants after a normal teardown."""

    async def test_full_teardown_from_mid_combat(
        self, mock_character, unique_channel_id, persist_calls
    ):
        key = f"discord:{unique_channel_id}"
        combat = _make_combat(unique_channel_id, mock_character)
        session = _make_session(unique_channel_id)
        session.combat_manager = combat
        set_combat_for_channel(unique_channel_id, combat)
        # cogs/combat.py registers coordinators WITHOUT a session
        coordinator = get_coordinator(combat)
        assert coordinator.session is None

        sessions = _make_session_manager(session)
        result = await sessions.end_combat(unique_channel_id)

        assert result is True
        # session returned to non-combat play
        assert session.state == SessionState.ACTIVE
        assert session.combat_manager is None
        # both registries released the channel entry
        assert get_combat_by_key(key) is None
        assert get_coordinator_by_key(key) is None
        # manager finalized exactly once
        assert combat.combat.ended_at is not None
        # final persistence ran once, on the registered coordinator, with the
        # session bound so it resolves session-owned Characters (Stage A.2)
        assert len(persist_calls) == 1
        assert persist_calls[0] is coordinator
        assert persist_calls[0].session is session

    async def test_second_call_is_safe_noop(
        self, mock_character, unique_channel_id, persist_calls
    ):
        key = f"discord:{unique_channel_id}"
        combat = _make_combat(unique_channel_id, mock_character)
        session = _make_session(unique_channel_id)
        session.combat_manager = combat
        set_combat_for_channel(unique_channel_id, combat)
        get_coordinator(combat)

        sessions = _make_session_manager(session)
        assert await sessions.end_combat(unique_channel_id) is True
        ended_at = combat.combat.ended_at

        # bot-layer paths can double-fire (button callback + slash command)
        assert await sessions.end_combat(unique_channel_id) is False

        assert session.state == SessionState.ACTIVE
        assert session.combat_manager is None
        assert get_combat_by_key(key) is None
        assert get_coordinator_by_key(key) is None
        assert combat.combat.ended_at == ended_at  # not re-stamped
        assert len(persist_calls) == 1  # no second persistence pass

    async def test_standalone_combat_without_session(
        self, unique_channel_id, persist_calls
    ):
        # /combat slash commands run with no GameSession at all
        key = f"discord:{unique_channel_id}"
        combat = _make_combat(unique_channel_id)
        set_combat_for_channel(unique_channel_id, combat)

        sessions = _make_session_manager()
        result = await sessions.end_combat(unique_channel_id)

        assert result is True
        assert get_combat_by_key(key) is None
        assert get_coordinator_by_key(key) is None
        assert combat.combat.ended_at is not None
        # no coordinator was registered: a transient one still persists players
        assert len(persist_calls) == 1
        assert persist_calls[0].manager is combat
        assert persist_calls[0].session is None

    async def test_already_finalized_manager_still_cleans_up(
        self, mock_character, unique_channel_id, persist_calls
    ):
        # /combat next: manager.next_turn() ends combat internally BEFORE the
        # cog reaches teardown. The owner must not re-finalize, but must still
        # persist, clear both registries, and reset the session.
        key = f"discord:{unique_channel_id}"
        combat = _make_combat(unique_channel_id, mock_character)
        combat.combat.transition(CombatState.END_TURN)
        combat.end_combat()
        assert combat.combat.state == CombatState.COMBAT_END
        ended_at = combat.combat.ended_at

        session = _make_session(unique_channel_id)
        session.combat_manager = combat
        set_combat_for_channel(unique_channel_id, combat)

        sessions = _make_session_manager(session)
        result = await sessions.end_combat(unique_channel_id)

        assert result is True
        assert combat.combat.ended_at == ended_at  # finalized exactly once
        assert session.state == SessionState.ACTIVE
        assert session.combat_manager is None
        assert get_combat_by_key(key) is None
        assert len(persist_calls) == 1


class TestEndCombatResilience:
    """Teardown must complete even when parts of the world are broken."""

    async def test_persist_failure_does_not_block_teardown(
        self, mock_character, unique_channel_id, monkeypatch
    ):
        async def _boom(self):
            raise RuntimeError("db down")

        monkeypatch.setattr(
            CombatTurnCoordinator, "persist_player_characters", _boom
        )

        key = f"discord:{unique_channel_id}"
        combat = _make_combat(unique_channel_id, mock_character)
        session = _make_session(unique_channel_id)
        session.combat_manager = combat
        set_combat_for_channel(unique_channel_id, combat)
        get_coordinator(combat)

        sessions = _make_session_manager(session)
        with capture_logs() as logs:
            result = await sessions.end_combat(unique_channel_id)

        # a wedged COMBAT session is worse than one missed sync
        assert result is True
        assert session.state == SessionState.ACTIVE
        assert session.combat_manager is None
        assert get_combat_by_key(key) is None
        assert get_coordinator_by_key(key) is None
        assert combat.combat.ended_at is not None

        # ...but the miss must be LOUD (persist-failure policy): one uniform
        # error-level persist_failed event, not a demoted warning.
        assert any(
            e["event"] == "persist_failed" and e["log_level"] == "error"
            for e in logs
        ), f"expected error-level persist_failed event, got: {logs}"

    async def test_heals_combat_session_with_no_manager(
        self, unique_channel_id, persist_calls
    ):
        # A narrator START_COMBAT effect can flip the session to COMBAT
        # without any CombatManager ever being created (audit). The owner
        # must still return the session to ACTIVE.
        session = _make_session(unique_channel_id)
        assert session.combat_manager is None

        sessions = _make_session_manager(session)
        result = await sessions.end_combat(unique_channel_id)

        assert result is True
        assert session.state == SessionState.ACTIVE
        assert persist_calls == []  # nothing to persist

    async def test_session_key_addressing_for_non_discord_frontends(
        self, persist_calls
    ):
        # Voice/web sessions are keyed by session_key, not channel id.
        combat = _make_combat(0)
        session = _make_session(0, session_key=VOICE_KEY)
        session.combat_manager = combat
        set_combat_by_key(VOICE_KEY, combat)

        sessions = _make_session_manager(session)
        result = await sessions.end_combat(0, session_key=VOICE_KEY)

        assert result is True
        assert session.state == SessionState.ACTIVE
        assert session.combat_manager is None
        assert get_combat_by_key(VOICE_KEY) is None
        assert len(persist_calls) == 1
