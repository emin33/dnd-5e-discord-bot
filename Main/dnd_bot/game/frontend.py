"""GameFrontend protocol - abstraction layer between game engine and UI.

Any frontend (Discord text, voice, web UI) implements this protocol.
The game engine emits GameEvents and awaits input via get_combat_action().
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Optional, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from .combat.actions import (
        CombatAction,
        ActionResult,
        TurnContext,
    )
    from .mechanics.dice import DiceRoll
    from ..models import Combat, Combatant


class GameEventType(str, Enum):
    """Types of events the game engine emits to frontends."""

    # Exploration / general play
    MECHANICS_READY = "mechanics_ready"
    NARRATIVE_TOKEN = "narrative_token"
    NARRATIVE_COMPLETE = "narrative_complete"

    # Combat lifecycle
    COMBAT_START = "combat_start"
    TURN_PROMPT = "turn_prompt"
    ACTION_RESULT = "action_result"
    TURN_END = "turn_end"
    COMBAT_END = "combat_end"

    # Errors
    ERROR = "error"


@dataclass
class GameEvent:
    """An event emitted by the game engine to attached frontends."""

    type: GameEventType
    data: dict[str, Any] = field(default_factory=dict)

    # --- Factory methods for type-safe event construction ---

    @staticmethod
    def mechanics_ready(
        mechanical_result: dict,
        dice_rolls: list[DiceRoll],
    ) -> GameEvent:
        return GameEvent(
            type=GameEventType.MECHANICS_READY,
            data={"mechanical_result": mechanical_result, "dice_rolls": dice_rolls},
        )

    @staticmethod
    def narrative_token(token: str) -> GameEvent:
        return GameEvent(
            type=GameEventType.NARRATIVE_TOKEN,
            data={"token": token},
        )

    @staticmethod
    def narrative_complete(narrative: str) -> GameEvent:
        return GameEvent(
            type=GameEventType.NARRATIVE_COMPLETE,
            data={"narrative": narrative},
        )

    @staticmethod
    def combat_start(combat: Combat) -> GameEvent:
        return GameEvent(
            type=GameEventType.COMBAT_START,
            data={"combat": combat},
        )

    @staticmethod
    def turn_prompt(turn_context: TurnContext) -> GameEvent:
        return GameEvent(
            type=GameEventType.TURN_PROMPT,
            data={"turn_context": turn_context},
        )

    @staticmethod
    def action_result(
        result: ActionResult,
        narrative: Optional[str] = None,
    ) -> GameEvent:
        return GameEvent(
            type=GameEventType.ACTION_RESULT,
            data={"result": result, "narrative": narrative},
        )

    @staticmethod
    def turn_end(
        next_combatant_name: Optional[str] = None,
        next_is_player: bool = False,
        round_advanced: bool = False,
        new_round: int = 0,
    ) -> GameEvent:
        return GameEvent(
            type=GameEventType.TURN_END,
            data={
                "next_combatant_name": next_combatant_name,
                "next_is_player": next_is_player,
                "round_advanced": round_advanced,
                "new_round": new_round,
            },
        )

    @staticmethod
    def combat_end(victory: bool = True, summary: str = "") -> GameEvent:
        return GameEvent(
            type=GameEventType.COMBAT_END,
            data={"victory": victory, "summary": summary},
        )

    @staticmethod
    def error(message: str) -> GameEvent:
        return GameEvent(
            type=GameEventType.ERROR,
            data={"message": message},
        )


@runtime_checkable
class GameFrontend(Protocol):
    """Protocol that any game frontend must implement.

    Frontends receive game events (narrative, mechanics, combat)
    and provide player input (text, combat actions).
    """

    @property
    def frontend_type(self) -> str:
        """Identifier for this frontend type (e.g., 'discord_text', 'voice')."""
        ...

    async def on_event(self, event: GameEvent) -> None:
        """Handle a game event (narrative, mechanics, combat updates).

        Frontends dispatch on event.type to render appropriately:
        - Discord: embeds, progressive edits, button views
        - Voice: TTS sentence streaming, spoken prompts
        - Web: DOM updates, audio playback
        """
        ...

    async def get_combat_action(self, turn_context: TurnContext) -> CombatAction:
        """Await and return a player's combat action choice.

        Called when it's a player's turn in combat. The frontend should
        present available options and wait for the player's selection.

        For Discord: show button menu, await interaction callback.
        For Voice: speak options, await ASR -> triage -> CombatAction.
        For Web: show button panel AND accept voice, race both.
        """
        ...
