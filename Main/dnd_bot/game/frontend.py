"""GameFrontend protocol - abstraction layer between game engine and UI.

Any frontend (Discord text, voice, web UI) implements this protocol.
The game engine emits three narration events per processed message
(session.py): MECHANICS_READY, then NARRATIVE_TOKEN as prose streams,
then NARRATIVE_COMPLETE with the final text and proposed effects.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Protocol, runtime_checkable, TYPE_CHECKING

if TYPE_CHECKING:
    from .mechanics.dice import DiceRoll


class GameEventType(str, Enum):
    """Types of events the game engine emits to frontends."""

    # Exploration / general play
    MECHANICS_READY = "mechanics_ready"
    NARRATIVE_TOKEN = "narrative_token"
    NARRATIVE_COMPLETE = "narrative_complete"

    # Dead combat-protocol members: never emitted anywhere; kept only
    # because dnd_bot/voice/frontend.py still registers handlers for them
    # at import time. Delete alongside that handler suite.
    COMBAT_START = "combat_start"
    ACTION_RESULT = "action_result"
    TURN_END = "turn_end"
    COMBAT_END = "combat_end"
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
    def narrative_complete(
        narrative: str,
        proposed_effects: list = None,
        scene_entities: list = None,
        player_characters: list = None,
    ) -> GameEvent:
        return GameEvent(
            type=GameEventType.NARRATIVE_COMPLETE,
            data={
                "narrative": narrative,
                "proposed_effects": proposed_effects or [],
                "scene_entities": scene_entities or [],
                "player_characters": player_characters or [],
            },
        )


@runtime_checkable
class GameFrontend(Protocol):
    """Protocol that any game frontend must implement.

    Frontends receive narration events (mechanics, streamed tokens,
    final narrative) and render them for their medium.
    """

    @property
    def frontend_type(self) -> str:
        """Identifier for this frontend type (e.g., 'discord_text', 'voice')."""
        ...

    async def on_event(self, event: GameEvent) -> None:
        """Handle a game event (mechanics, narrative token, final narrative).

        Frontends dispatch on event.type to render appropriately:
        - Discord: embeds, progressive edits
        - Voice: TTS on final narrative
        - Web: DOM updates, audio playback
        """
        ...
