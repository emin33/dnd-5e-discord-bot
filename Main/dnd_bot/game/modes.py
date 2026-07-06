"""Game-mode pushdown machine (REFACTOR_PLAN Step 3).

Nystrom's State pattern + pushdown automaton: a stack of modes with
EXPLORATION as the permanent base; combat pushes onto exploration and pops
back to whatever was underneath. ``GameSession`` owns one, and its
``enter_combat_mode``/``exit_combat_mode`` methods are the ONLY writers of
the mode flip and its derived surfaces (``session.state`` COMBAT/ACTIVE,
``world_state.phase``, ``session.combat_manager``) — replacing the inline
branching that used to live in ``process_message`` and the dead
``enter_combat`` twin.

Deliberately mode VALUES on a stack, not per-mode classes: nothing in the
codebase consumes per-mode policy yet (per-mode legal tools and
NarrationSpecs are the research's later-step aspiration); classes today
would be speculative structure. When a consumer appears, each enum row can
grow into a state object without changing the machine's surface.
"""

from enum import Enum


class GameMode(str, Enum):
    """A play mode. Distinct from SessionState, which also carries session
    lifecycle (STARTING/PAUSED/ENDED); modes only describe how play resolves."""

    EXPLORATION = "exploration"
    COMBAT = "combat"


class ModeMachine:
    """Pushdown stack of :class:`GameMode`; EXPLORATION is the un-poppable base."""

    def __init__(self) -> None:
        self._stack: list[GameMode] = [GameMode.EXPLORATION]

    @property
    def current(self) -> GameMode:
        return self._stack[-1]

    @property
    def in_combat(self) -> bool:
        return self.current is GameMode.COMBAT

    def push(self, mode: GameMode) -> None:
        """Push a mode. Idempotent when already in that mode — combat-entry
        signals can arrive from more than one path in the same turn."""
        if self.current is not mode:
            self._stack.append(mode)

    def pop(self) -> GameMode:
        """Pop back to the underlying mode; the base never pops. Returns the
        mode now current."""
        if len(self._stack) > 1:
            self._stack.pop()
        return self.current
