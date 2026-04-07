"""Discord text channel implementation of GameFrontend.

Wraps all existing Discord presentation logic (embeds, progressive edits,
combat button views) behind the generic GameFrontend protocol.
"""

from __future__ import annotations

import asyncio
import time
from typing import Optional

import discord
import structlog

from ...game.frontend import GameFrontend, GameEvent, GameEventType
from ...game.combat.actions import CombatAction, ActionResult, TurnContext
from ..views.combat_actions import CombatActionView, ActionResultEmbed
from ..embeds.combat_embed import (
    build_combat_tracker_embed,
    build_combat_start_embed,
    build_combat_end_embed,
)

logger = structlog.get_logger()

# Discord limits
EMBED_DESCRIPTION_LIMIT = 4096
_EDIT_INTERVAL = 0.8  # Seconds between Discord edits (rate limit safe)


def _build_mechanics_embed(
    mechanical_result: Optional[dict],
    dice_rolls: list,
) -> Optional[discord.Embed]:
    """Build a standalone blue embed for mechanics (rolls, checks, saves).

    Returns None if there's nothing to show.
    """
    fields = []

    if mechanical_result:
        action_type = mechanical_result.get("action_type", "")

        if action_type == "attack":
            attack_info = []
            if mechanical_result.get("attack_roll"):
                attack_info.append(
                    f"**Attack:** {mechanical_result['attack_roll']} vs AC "
                    f"{mechanical_result.get('target_ac', '?')}"
                )
            if mechanical_result.get("hit"):
                attack_info.append(
                    f"**Damage:** {mechanical_result.get('damage', 0)} "
                    f"{mechanical_result.get('damage_type', '')}"
                )
            if attack_info:
                fields.append((":crossed_swords: Combat", "\n".join(attack_info)))

        elif action_type in ("skill_check", "ability_check", "check"):
            skill_name = (
                mechanical_result.get("skill")
                or mechanical_result.get("ability")
                or "Check"
            )
            success = "Success" if mechanical_result.get("success") else "Failure"
            fields.append((
                f":mag: {skill_name.title()}",
                f"Roll: {mechanical_result.get('roll', '?')} vs DC "
                f"{mechanical_result.get('dc', '?')} - **{success}**",
            ))

        elif action_type == "saving_throw":
            success = "Success" if mechanical_result.get("success") else "Failure"
            fields.append((
                f":shield: {mechanical_result.get('ability', '')} Save",
                f"Roll: {mechanical_result.get('roll', '?')} vs DC "
                f"{mechanical_result.get('dc', '?')} - **{success}**",
            ))

        elif action_type == "spell":
            spell_info = [
                f"**Spell:** {mechanical_result.get('spell_name', 'Unknown')}"
            ]
            if mechanical_result.get("damage"):
                spell_info.append(f"**Damage:** {mechanical_result.get('damage')}")
            if mechanical_result.get("healing"):
                spell_info.append(f"**Healing:** {mechanical_result.get('healing')}")
            fields.append((":sparkles: Spellcasting", "\n".join(spell_info)))

    if dice_rolls:
        roll_text = []
        for roll in dice_rolls[:5]:
            dice_str = f"[{', '.join(str(d) for d in roll.kept_dice)}]"
            if roll.modifier != 0:
                mod_str = (
                    f" + {roll.modifier}"
                    if roll.modifier > 0
                    else f" - {abs(roll.modifier)}"
                )
                dice_str += mod_str
            dice_str += f" = **{roll.total}**"
            if roll.reason:
                dice_str = f"{roll.reason}: {dice_str}"
            roll_text.append(dice_str)
        if roll_text:
            fields.append((":game_die: Dice", "\n".join(roll_text)))

    if not fields:
        return None

    embed = discord.Embed(color=discord.Color.blue())
    for name, value in fields:
        embed.add_field(name=name, value=value, inline=False)
    return embed


def _split_text(text: str, limit: int = 2000) -> list[str]:
    """Split text into chunks that fit Discord's limits."""
    if len(text) <= limit:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        chunk = remaining[:limit]
        break_point = chunk.rfind("\n\n")
        if break_point < limit // 2:
            for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                pos = chunk.rfind(sep)
                if pos > break_point:
                    break_point = pos + len(sep) - 1

        if break_point < limit // 2:
            break_point = chunk.rfind(" ")

        if break_point < limit // 2:
            break_point = limit - 1

        chunks.append(remaining[: break_point + 1].rstrip())
        remaining = remaining[break_point + 1 :].lstrip()

    return chunks


class DiscordTextFrontend:
    """GameFrontend implementation for Discord text channels.

    Handles all Discord-specific presentation:
    - Mechanics as blue embeds
    - Narrative streaming via progressive message edits
    - Final narrative as gold embeds
    - Combat turns via button views (CombatActionView)
    """

    def __init__(self, channel: discord.TextChannel):
        self._channel = channel

        # Streaming state (reset per turn)
        self._stream_msg: Optional[discord.Message] = None
        self._stream_buf: list[str] = []
        self._last_edit: float = 0.0
        self._mechanics_sent: bool = False

    @property
    def frontend_type(self) -> str:
        return "discord_text"

    @property
    def channel(self) -> discord.TextChannel:
        return self._channel

    def reset_turn_state(self) -> None:
        """Reset per-turn streaming state before a new player message."""
        self._stream_msg = None
        self._stream_buf = []
        self._last_edit = 0.0
        self._mechanics_sent = False

    async def on_event(self, event: GameEvent) -> None:
        """Dispatch game events to Discord presentation."""
        handler = _EVENT_HANDLERS.get(event.type)
        if handler:
            await handler(self, event)
        else:
            logger.debug("unhandled_frontend_event", event_type=event.type)

    async def get_combat_action(self, turn_context: TurnContext) -> CombatAction:
        """Show combat button UI and await player's action choice.

        Creates a CombatActionView with buttons. When the player clicks
        a button, the view resolves the future with their CombatAction.
        """
        future: asyncio.Future[CombatAction] = asyncio.get_event_loop().create_future()

        # Import coordinator here to avoid circular imports at module level
        from ...game.combat.coordinator import get_coordinator_for_channel

        coordinator = get_coordinator_for_channel(self._channel.id)
        if not coordinator:
            raise RuntimeError("No combat coordinator for this channel")

        async def on_action_complete(result: ActionResult) -> None:
            # Send the mechanical result embed
            result_embed = ActionResultEmbed.build(result)
            await self._channel.send(embed=result_embed)

            # Narrate the result
            try:
                narrative = await coordinator.narrate_result(result)
                if narrative:
                    narr_embed = discord.Embed(
                        description=narrative,
                        color=discord.Color.dark_gold(),
                    )
                    await self._channel.send(embed=narr_embed)
            except Exception as e:
                logger.warning("narration_failed", error=str(e))

        async def on_turn_end() -> None:
            # The turn loop handles advancement - just resolve the future
            pass

        # Wrap the view so clicking a button resolves our future
        class _FutureResolvingView(CombatActionView):
            async def _execute_and_resolve(self_view, action: CombatAction):
                if not future.done():
                    future.set_result(action)

        view = CombatActionView(
            coordinator=coordinator,
            turn_context=turn_context,
            on_action_complete=on_action_complete,
            on_turn_end=on_turn_end,
        )

        await self._channel.send(
            f":crossed_swords: **{turn_context.combatant_name}**, it's your turn!",
            embed=view.get_embed(),
            view=view,
        )

        return await future

    # --- Event handlers ---

    async def _handle_mechanics_ready(self, event: GameEvent) -> None:
        mech_embed = _build_mechanics_embed(
            event.data["mechanical_result"],
            event.data["dice_rolls"],
        )
        if mech_embed:
            await self._channel.send(embed=mech_embed)
            self._mechanics_sent = True

    async def _handle_narrative_token(self, event: GameEvent) -> None:
        token = event.data["token"]
        self._stream_buf.append(token)

        now = time.monotonic()
        if now - self._last_edit < _EDIT_INTERVAL:
            return

        self._last_edit = now
        text = "".join(self._stream_buf).strip()
        if not text:
            return

        display = text[:4000] + "..." if len(text) > 4000 else text
        try:
            if self._stream_msg is None:
                self._stream_msg = await self._channel.send(f"*{display}*")
            else:
                await self._stream_msg.edit(content=f"*{display}*")
        except Exception:
            pass  # Best effort streaming

    async def _handle_narrative_complete(self, event: GameEvent) -> None:
        narrative = event.data["narrative"]

        # Delete streaming message if we had one
        if self._stream_msg:
            try:
                await self._stream_msg.delete()
            except Exception:
                pass

        # Build final gold embeds
        if not narrative:
            narrative = "*The scene unfolds...*"

        if len(narrative) > EMBED_DESCRIPTION_LIMIT:
            chunks = _split_text(narrative, EMBED_DESCRIPTION_LIMIT - 100)
            for i, chunk in enumerate(chunks):
                embed = discord.Embed(
                    description=chunk,
                    color=discord.Color.dark_gold(),
                )
                if len(chunks) > 1:
                    embed.set_footer(text=f"({i + 1}/{len(chunks)})")
                await self._channel.send(embed=embed)
        else:
            await self._channel.send(
                embed=discord.Embed(
                    description=narrative,
                    color=discord.Color.dark_gold(),
                )
            )

    async def _handle_combat_start(self, event: GameEvent) -> None:
        combat = event.data["combat"]
        start_embed = build_combat_start_embed(combat)
        tracker_embed = build_combat_tracker_embed(combat)
        await self._channel.send(embeds=[start_embed, tracker_embed])

    async def _handle_action_result(self, event: GameEvent) -> None:
        result = event.data["result"]
        narrative = event.data.get("narrative")

        result_embed = ActionResultEmbed.build(result)
        await self._channel.send(embed=result_embed)

        if narrative:
            narr_embed = discord.Embed(
                description=narrative,
                color=discord.Color.dark_gold(),
            )
            await self._channel.send(embed=narr_embed)

    async def _handle_turn_end(self, event: GameEvent) -> None:
        next_name = event.data.get("next_combatant_name")
        round_advanced = event.data.get("round_advanced", False)
        new_round = event.data.get("new_round", 0)

        if next_name:
            msg = f":arrow_right: **{next_name}**'s turn!"
            if round_advanced:
                msg = f"**Round {new_round}**\n{msg}"
            await self._channel.send(msg)

    async def _handle_combat_end(self, event: GameEvent) -> None:
        victory = event.data.get("victory", True)
        from ..embeds.combat_embed import build_combat_end_embed
        # We'd need the combat object here - for now just send a message
        msg = ":tada: **Victory!**" if victory else ":skull: **Defeat...**"
        await self._channel.send(msg)

    async def _handle_error(self, event: GameEvent) -> None:
        message = event.data.get("message", "Something went wrong.")
        await self._channel.send(f":warning: {message}")


# Dispatch table for event handling
_EVENT_HANDLERS = {
    GameEventType.MECHANICS_READY: DiscordTextFrontend._handle_mechanics_ready,
    GameEventType.NARRATIVE_TOKEN: DiscordTextFrontend._handle_narrative_token,
    GameEventType.NARRATIVE_COMPLETE: DiscordTextFrontend._handle_narrative_complete,
    GameEventType.COMBAT_START: DiscordTextFrontend._handle_combat_start,
    GameEventType.ACTION_RESULT: DiscordTextFrontend._handle_action_result,
    GameEventType.TURN_END: DiscordTextFrontend._handle_turn_end,
    GameEventType.COMBAT_END: DiscordTextFrontend._handle_combat_end,
    GameEventType.ERROR: DiscordTextFrontend._handle_error,
}
