"""REFACTOR_PLAN Step 3: the game-mode pushdown machine and its session surface.

``GameSession.enter_combat_mode``/``exit_combat_mode`` are the ONLY writers
of the combat-mode flip and its derived surfaces (session.state,
combat_manager, world_state.phase). These tests pin the machine's pushdown
semantics and the derived-surface contract — including the two deliberate
unifications the machine introduced:

- ``world_state.phase`` now follows the mode IMMEDIATELY on push/pop
  (previously it lagged until the next turn's phase sync / never reset at
  teardown), and
- a narrative phase the delta extractor set (dialogue, rest, …) survives
  the pop — only a literal "combat" phase returns to exploration.
"""

from dnd_bot.game.combat.manager import CombatManager
from dnd_bot.game.modes import GameMode, ModeMachine
from dnd_bot.game.session import GameSession, GameSessionManager, SessionState
from dnd_bot.game.world_state import WorldState


def _session(channel_id: int = 1, state: SessionState = SessionState.ACTIVE) -> GameSession:
    session = GameSession(
        id="modes-test-session",
        channel_id=channel_id,
        guild_id=1,
        campaign_id="modes-campaign",
        state=state,
    )
    session.world_state = WorldState()
    return session


def _manager(channel_id: int = 1) -> CombatManager:
    return CombatManager.create_encounter(
        session_id="modes-test-session", channel_id=channel_id, name="Modes Test"
    )


class TestModeMachine:
    def test_base_is_exploration(self):
        machine = ModeMachine()
        assert machine.current is GameMode.EXPLORATION
        assert machine.in_combat is False

    def test_push_and_pop_combat(self):
        machine = ModeMachine()
        machine.push(GameMode.COMBAT)
        assert machine.current is GameMode.COMBAT
        assert machine.in_combat is True
        assert machine.pop() is GameMode.EXPLORATION
        assert machine.in_combat is False

    def test_push_is_idempotent(self):
        # Combat-entry signals can arrive from more than one path per turn;
        # a duplicate push must not need a matching duplicate pop.
        machine = ModeMachine()
        machine.push(GameMode.COMBAT)
        machine.push(GameMode.COMBAT)
        assert machine.pop() is GameMode.EXPLORATION

    def test_base_never_pops(self):
        machine = ModeMachine()
        assert machine.pop() is GameMode.EXPLORATION
        assert machine.pop() is GameMode.EXPLORATION


class TestSessionModeSurface:
    def test_enter_combat_mode_writes_every_derived_surface(self, unique_channel_id):
        session = _session(unique_channel_id)
        manager = _manager(unique_channel_id)

        session.enter_combat_mode(manager)

        assert session.modes.in_combat is True
        assert session.state == SessionState.COMBAT
        assert session.combat_manager is manager
        assert session.world_state.phase == "combat"

    def test_enter_without_manager_keeps_existing_reference(self, unique_channel_id):
        # process_message's push: the entry signal already stored the
        # manager (or built none — signal 3); None must not clobber it.
        session = _session(unique_channel_id)
        manager = _manager(unique_channel_id)
        session.combat_manager = manager

        session.enter_combat_mode()

        assert session.combat_manager is manager
        assert session.state == SessionState.COMBAT

    def test_enter_survives_missing_world_state(self, unique_channel_id):
        session = _session(unique_channel_id)
        session.world_state = None
        session.enter_combat_mode(_manager(unique_channel_id))
        assert session.state == SessionState.COMBAT

    def test_exit_pops_back_and_reports_did_work(self, unique_channel_id):
        session = _session(unique_channel_id)
        session.enter_combat_mode(_manager(unique_channel_id))

        assert session.exit_combat_mode() is True

        assert session.modes.in_combat is False
        assert session.state == SessionState.ACTIVE
        assert session.combat_manager is None
        assert session.world_state.phase == "exploration"

    def test_exit_when_not_in_combat_is_a_noop_that_still_drops_manager(
        self, unique_channel_id
    ):
        # end_combat's healing path: no COMBAT state, but a stale manager
        # reference must still be dropped (previous teardown behavior).
        session = _session(unique_channel_id)
        session.combat_manager = _manager(unique_channel_id)

        assert session.exit_combat_mode() is False
        assert session.state == SessionState.ACTIVE
        assert session.combat_manager is None

    def test_exit_preserves_narrative_phase(self, unique_channel_id):
        # A phase the delta extractor set (dialogue/rest/…) is NOT stomped
        # by the pop — only a literal "combat" phase resets.
        session = _session(unique_channel_id, state=SessionState.COMBAT)
        session.world_state.phase = "dialogue"

        session.exit_combat_mode()

        assert session.world_state.phase == "dialogue"


class TestEndCombatPopsPhase:
    async def test_end_combat_returns_phase_to_exploration(self, unique_channel_id):
        # The teardown owner's pop now resets world_state.phase (previously
        # it lagged until the next process_message phase sync).
        session = _session(unique_channel_id, state=SessionState.COMBAT)
        session.world_state.phase = "combat"
        sessions = GameSessionManager()
        sessions._sessions[session.session_key] = session

        assert await sessions.end_combat(unique_channel_id) is True

        assert session.state == SessionState.ACTIVE
        assert session.world_state.phase == "exploration"
