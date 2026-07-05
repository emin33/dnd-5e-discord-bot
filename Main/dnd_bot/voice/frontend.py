"""Voice frontend implementing GameFrontend protocol.

Converts game events to speech (TTS).
"""

from __future__ import annotations

from typing import Callable, Awaitable, Optional, TYPE_CHECKING

import structlog

from ..game.frontend import GameEvent, GameEventType

if TYPE_CHECKING:
    from ..game.session import GameSession

logger = structlog.get_logger()


def _mechanics_to_speech(mechanical_result: dict, dice_rolls: list) -> str:
    """Convert mechanical results to concise spoken text.

    Instead of Discord embeds, we produce terse speech-optimized text.
    """
    parts = []

    if mechanical_result:
        action_type = mechanical_result.get("action_type", "")

        if action_type == "attack":
            roll = mechanical_result.get("attack_roll", "?")
            ac = mechanical_result.get("target_ac", "?")
            parts.append(f"Attack roll {roll} versus AC {ac}.")
            if mechanical_result.get("hit"):
                dmg = mechanical_result.get("damage", 0)
                dmg_type = mechanical_result.get("damage_type", "")
                parts.append(f"Hit! {dmg} {dmg_type} damage.")
            else:
                parts.append("Miss!")

        elif action_type in ("skill_check", "ability_check", "check"):
            skill = (
                mechanical_result.get("skill")
                or mechanical_result.get("ability")
                or "Check"
            )
            roll = mechanical_result.get("roll", "?")
            dc = mechanical_result.get("dc", "?")
            success = "Success" if mechanical_result.get("success") else "Failure"
            parts.append(f"{skill.title()} check. {roll} versus DC {dc}. {success}.")

        elif action_type == "saving_throw":
            ability = mechanical_result.get("ability", "")
            roll = mechanical_result.get("roll", "?")
            dc = mechanical_result.get("dc", "?")
            success = "Success" if mechanical_result.get("success") else "Failure"
            parts.append(f"{ability} saving throw. {roll} versus DC {dc}. {success}.")

        elif action_type == "spell":
            spell = mechanical_result.get("spell_name", "Unknown")
            parts.append(f"Casting {spell}.")
            if mechanical_result.get("damage"):
                parts.append(f"{mechanical_result['damage']} damage.")
            if mechanical_result.get("healing"):
                parts.append(f"{mechanical_result['healing']} healing.")

    return " ".join(parts)


class VoiceFrontend:
    """GameFrontend implementation for voice interaction.

    Converts game events to speech via TTS.

    The `speak` callback is provided by the transport layer (LiveKit
    or Discord voice) and handles actually playing audio.
    """

    def __init__(
        self,
        speak_fn: Callable[[str], Awaitable[None]],
        session: Optional[GameSession] = None,
    ):
        """
        Args:
            speak_fn: Async callable that takes text and speaks it via TTS.
                Provided by the transport layer (LiveKit, Discord voice, etc.)
            session: Current game session (for combat context building).
        """
        self._speak = speak_fn
        self._session = session

    @property
    def frontend_type(self) -> str:
        return "voice"

    async def on_event(self, event: GameEvent) -> None:
        """Dispatch game events to voice output."""
        handler = _VOICE_EVENT_HANDLERS.get(event.type)
        if handler:
            await handler(self, event)
        else:
            logger.debug("unhandled_voice_event", event_type=event.type)

    # --- Event handlers ---

    async def _handle_mechanics_ready(self, event: GameEvent) -> None:
        text = _mechanics_to_speech(
            event.data["mechanical_result"],
            event.data["dice_rolls"],
        )
        if text:
            await self._speak(text)

    async def _handle_narrative_complete(self, event: GameEvent) -> None:
        narrative = event.data["narrative"]
        if narrative:
            # Speak the full narrative (TTS handles sentence splitting)
            await self._speak(narrative)


# Dispatch table for event handling
_VOICE_EVENT_HANDLERS = {
    GameEventType.MECHANICS_READY: VoiceFrontend._handle_mechanics_ready,
    GameEventType.NARRATIVE_COMPLETE: VoiceFrontend._handle_narrative_complete,
}
