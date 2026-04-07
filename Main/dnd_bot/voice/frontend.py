"""Voice frontend implementing GameFrontend protocol.

Converts game events to speech (TTS) and accepts voice input (ASR).
For combat, supports both voice commands (via triage) and web UI buttons.
"""

from __future__ import annotations

import asyncio
import re
from typing import Any, Callable, Awaitable, Optional, TYPE_CHECKING

import structlog

from ..game.frontend import GameFrontend, GameEvent, GameEventType
from ..game.combat.actions import CombatAction, CombatActionType, TurnContext
from .tts import RivaTTS, TTSSentenceQueue

if TYPE_CHECKING:
    from ..game.session import GameSession
    from ..llm.orchestrator import DMOrchestrator

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


def _turn_context_to_speech(ctx: TurnContext) -> str:
    """Convert combat turn context to a spoken prompt."""
    parts = [f"{ctx.combatant_name}, your turn."]
    parts.append(f"HP {ctx.hp_current} of {ctx.hp_max}.")

    if ctx.equipped_weapons:
        weapon_names = [w.name for w in ctx.equipped_weapons[:2]]
        parts.append(f"Weapons: {', '.join(weapon_names)}.")

    if ctx.in_melee_with:
        parts.append(f"In melee with {', '.join(ctx.in_melee_with[:3])}.")

    if ctx.available_spells:
        count = len(ctx.available_spells)
        parts.append(f"{count} spell{'s' if count != 1 else ''} available.")

    parts.append("What do you do?")
    return " ".join(parts)


def _action_result_to_speech(result) -> str:
    """Convert an ActionResult to concise spoken summary."""
    action_name = result.action.action_type.value.replace("_", " ").title()

    if not result.success:
        return f"{action_name} failed. {result.error or ''}"

    parts = [f"{action_name}."]

    if result.attack_roll:
        crit = " Critical hit!" if result.critical_hit else ""
        miss = " Critical miss!" if result.critical_miss else ""
        parts.append(f"Roll {result.attack_roll.total}.{crit}{miss}")

    if result.damage_dealt:
        total = sum(result.damage_dealt.values())
        dtype = result.damage_type or ""
        parts.append(f"{total} {dtype} damage.")

    if result.healing_done:
        total = sum(result.healing_done.values())
        parts.append(f"{total} healing.")

    if result.killed_targets:
        names = ", ".join(result.killed_targets)
        parts.append(f"Killed {names}!")

    if result.unconscious_targets:
        names = ", ".join(result.unconscious_targets)
        parts.append(f"{names} goes down!")

    return " ".join(parts)


class VoiceFrontend:
    """GameFrontend implementation for voice interaction.

    Converts game events to speech via Riva TTS and accepts player
    input via ASR transcriptions. For combat, supports both voice
    commands (parsed through triage) and web UI button clicks.

    The `speak` callback is provided by the transport layer (LiveKit
    or Discord voice) and handles actually playing audio.
    """

    def __init__(
        self,
        tts: RivaTTS,
        speak_fn: Callable[[str], Awaitable[None]],
        orchestrator: Optional[DMOrchestrator] = None,
        session: Optional[GameSession] = None,
    ):
        """
        Args:
            tts: Riva TTS instance for sentence-level synthesis.
            speak_fn: Async callable that takes text and speaks it via TTS.
                Provided by the transport layer (LiveKit, Discord voice, etc.)
            orchestrator: For triage_action() during voice combat.
            session: Current game session (for combat context building).
        """
        self._tts = tts
        self._speak = speak_fn
        self._orchestrator = orchestrator
        self._session = session

        # Sentence queue for streaming narration via TTS
        self._sentence_queue = TTSSentenceQueue(tts)

        # Combat action future - resolved by either voice or web UI
        self._combat_action_future: Optional[asyncio.Future[CombatAction]] = None

        # Callback for web UI combat actions (set by transport layer)
        self.on_web_combat_action: Optional[
            Callable[[CombatAction], None]
        ] = None

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

    async def get_combat_action(self, turn_context: TurnContext) -> CombatAction:
        """Await player's combat action via voice command or web UI button.

        Speaks the turn prompt, then waits for either:
        1. Voice: ASR transcription → triage → CombatAction
        2. Web UI: button click → CombatAction directly

        First one to resolve wins (asyncio race).
        """
        # Speak the turn prompt
        prompt = _turn_context_to_speech(turn_context)
        await self._speak(prompt)

        # Create a future that either voice or web UI can resolve
        loop = asyncio.get_running_loop()
        self._combat_action_future = loop.create_future()

        try:
            action = await self._combat_action_future
            return action
        finally:
            self._combat_action_future = None

    async def resolve_voice_combat(
        self,
        transcription: str,
        turn_context: TurnContext,
        player_name: str,
    ) -> None:
        """Resolve combat action from a voice transcription.

        Called by the transport layer when ASR produces text during
        a player's combat turn. Uses triage to classify the spoken
        command into a CombatAction.
        """
        if not self._combat_action_future or self._combat_action_future.done():
            return

        if not self._orchestrator:
            logger.warning("no_orchestrator_for_voice_combat")
            return

        # Use triage to classify the voice command
        from ..llm.brains.base import BrainContext

        # Build minimal context for triage
        context = BrainContext(
            player_input=transcription,
            memory_context="",
            character_context="",
            world_state_yaml="",
            scene_context="",
            knowledge_context="",
        )

        try:
            triage_result = await self._orchestrator.triage_action(
                transcription, player_name, context
            )
        except Exception as e:
            logger.warning("voice_triage_failed", error=str(e))
            await self._speak(
                "I didn't catch that. You can attack, cast a spell, "
                "dash, dodge, disengage, or end your turn."
            )
            return

        # Map triage result to CombatAction
        action = self._triage_to_combat_action(triage_result, turn_context)
        if action and not self._combat_action_future.done():
            self._combat_action_future.set_result(action)
        else:
            await self._speak(
                "I didn't understand that as a combat action. "
                "Try saying something like 'I attack the goblin' "
                "or 'I cast fireball'."
            )

    def resolve_web_combat(self, action: CombatAction) -> None:
        """Resolve combat action from a web UI button click.

        Called by the transport layer when the web UI sends a
        structured CombatAction (e.g., player clicked Attack + target).
        """
        if self._combat_action_future and not self._combat_action_future.done():
            self._combat_action_future.set_result(action)

    def _triage_to_combat_action(
        self,
        triage_result,
        turn_context: TurnContext,
    ) -> Optional[CombatAction]:
        """Map a TriageResult to a CombatAction.

        Handles fuzzy target matching and action type mapping.
        """
        action_type_map = {
            "attack": CombatActionType.ATTACK,
            "cast_spell": CombatActionType.CAST_SPELL,
            "dash": CombatActionType.DASH,
            "dodge": CombatActionType.DODGE,
            "disengage": CombatActionType.DISENGAGE,
            "help": CombatActionType.HELP,
            "hide": CombatActionType.HIDE,
            "end_turn": CombatActionType.END_TURN,
        }

        action_type = action_type_map.get(triage_result.action_type)
        if not action_type:
            # Triage classified it as something non-combat (social, roleplay, etc.)
            return None

        # Resolve target
        target_ids = []
        if triage_result.target_name and turn_context.in_melee_with:
            # Fuzzy match target name against available targets
            target_name_lower = triage_result.target_name.lower()
            for target in turn_context.in_melee_with:
                if target_name_lower in target.lower():
                    target_ids.append(target)
                    break
            if not target_ids:
                # Default to first melee target
                target_ids.append(turn_context.in_melee_with[0])

        return CombatAction(
            action_type=action_type,
            combatant_id=turn_context.combatant_id,
            target_ids=target_ids,
            weapon_index=(
                turn_context.equipped_weapons[0].name
                if turn_context.equipped_weapons
                else None
            ),
            spell_index=triage_result.item_name if action_type == CombatActionType.CAST_SPELL else None,
        )

    # --- Event handlers ---

    async def _handle_mechanics_ready(self, event: GameEvent) -> None:
        text = _mechanics_to_speech(
            event.data["mechanical_result"],
            event.data["dice_rolls"],
        )
        if text:
            await self._speak(text)

    async def _handle_narrative_token(self, event: GameEvent) -> None:
        # Buffer tokens into sentences for TTS streaming
        self._sentence_queue.add_token(event.data["token"])

    async def _handle_narrative_complete(self, event: GameEvent) -> None:
        narrative = event.data["narrative"]
        if narrative:
            # Speak the full narrative (TTS handles sentence splitting)
            await self._speak(narrative)

    async def _handle_combat_start(self, event: GameEvent) -> None:
        await self._speak("Combat begins! Roll for initiative!")

    async def _handle_action_result(self, event: GameEvent) -> None:
        result = event.data["result"]
        narrative = event.data.get("narrative")

        # Speak mechanical result summary
        summary = _action_result_to_speech(result)
        if summary:
            await self._speak(summary)

        # Speak narrative if provided
        if narrative:
            await self._speak(narrative)

    async def _handle_turn_end(self, event: GameEvent) -> None:
        next_name = event.data.get("next_combatant_name")
        round_advanced = event.data.get("round_advanced", False)
        new_round = event.data.get("new_round", 0)

        if round_advanced:
            await self._speak(f"Round {new_round}.")
        if next_name:
            await self._speak(f"{next_name}'s turn.")

    async def _handle_combat_end(self, event: GameEvent) -> None:
        victory = event.data.get("victory", True)
        if victory:
            await self._speak("Victory! All enemies have been defeated.")
        else:
            await self._speak("The party has fallen in battle.")

    async def _handle_error(self, event: GameEvent) -> None:
        message = event.data.get("message", "Something went wrong.")
        # Strip markdown formatting for voice
        clean = re.sub(r"[*_`]", "", message)
        await self._speak(clean)


# Dispatch table for event handling
_VOICE_EVENT_HANDLERS = {
    GameEventType.MECHANICS_READY: VoiceFrontend._handle_mechanics_ready,
    GameEventType.NARRATIVE_TOKEN: VoiceFrontend._handle_narrative_token,
    GameEventType.NARRATIVE_COMPLETE: VoiceFrontend._handle_narrative_complete,
    GameEventType.COMBAT_START: VoiceFrontend._handle_combat_start,
    GameEventType.ACTION_RESULT: VoiceFrontend._handle_action_result,
    GameEventType.TURN_END: VoiceFrontend._handle_turn_end,
    GameEventType.COMBAT_END: VoiceFrontend._handle_combat_end,
    GameEventType.ERROR: VoiceFrontend._handle_error,
}
