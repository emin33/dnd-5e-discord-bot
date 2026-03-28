"""Game session commands and message handler."""

from typing import Optional
import discord
from discord.ext import commands
import structlog

from ...game.session import get_session_manager, SessionState
from ...data.repositories import get_character_repo, get_campaign_repo
from ...game.combat.manager import get_combat_for_channel
from ...game.combat.coordinator import get_coordinator
from ...game.combat.actions import CombatActionType
from ..embeds.combat_embed import build_combat_tracker_embed, build_combat_start_embed
from ..views.campaign_lobby import get_active_campaign_id
from ..views.combat_actions import CombatActionView, ActionResultEmbed, NPCTurnView

logger = structlog.get_logger()

# Discord limits
EMBED_DESCRIPTION_LIMIT = 4096
MESSAGE_LIMIT = 2000


def split_text(text: str, limit: int = MESSAGE_LIMIT) -> list[str]:
    """Split text into chunks that fit Discord's limits, breaking at paragraph/sentence boundaries."""
    if len(text) <= limit:
        return [text]

    chunks = []
    remaining = text

    while remaining:
        if len(remaining) <= limit:
            chunks.append(remaining)
            break

        # Find a good break point (paragraph, then sentence, then word)
        chunk = remaining[:limit]

        # Try to break at paragraph
        break_point = chunk.rfind("\n\n")
        if break_point < limit // 2:
            # Try to break at sentence
            for sep in [". ", "! ", "? ", ".\n", "!\n", "?\n"]:
                pos = chunk.rfind(sep)
                if pos > break_point:
                    break_point = pos + len(sep) - 1

        if break_point < limit // 2:
            # Try to break at word
            break_point = chunk.rfind(" ")

        if break_point < limit // 2:
            # Force break at limit
            break_point = limit - 1

        chunks.append(remaining[:break_point + 1].rstrip())
        remaining = remaining[break_point + 1:].lstrip()

    return chunks


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
                attack_info.append(f"**Attack:** {mechanical_result['attack_roll']} vs AC {mechanical_result.get('target_ac', '?')}")
            if mechanical_result.get("hit"):
                attack_info.append(f"**Damage:** {mechanical_result.get('damage', 0)} {mechanical_result.get('damage_type', '')}")
            if attack_info:
                fields.append((":crossed_swords: Combat", "\n".join(attack_info)))

        elif action_type in ("skill_check", "ability_check", "check"):
            skill_name = mechanical_result.get('skill') or mechanical_result.get('ability') or 'Check'
            fields.append((
                f":mag: {skill_name.title()}",
                f"Roll: {mechanical_result.get('roll', '?')} vs DC {mechanical_result.get('dc', '?')} - **{'Success' if mechanical_result.get('success') else 'Failure'}**",
            ))

        elif action_type == "saving_throw":
            fields.append((
                f":shield: {mechanical_result.get('ability', '')} Save",
                f"Roll: {mechanical_result.get('roll', '?')} vs DC {mechanical_result.get('dc', '?')} - **{'Success' if mechanical_result.get('success') else 'Failure'}**",
            ))

        elif action_type == "spell":
            spell_info = [f"**Spell:** {mechanical_result.get('spell_name', 'Unknown')}"]
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
                mod_str = f" + {roll.modifier}" if roll.modifier > 0 else f" - {abs(roll.modifier)}"
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


def format_dm_response(response, skip_mechanics: bool = False) -> list[discord.Embed]:
    """Format a DM response as Discord embed(s).

    If skip_mechanics is True, the mechanics embed was already sent via callback.
    """
    embeds = []

    # Include mechanics embed only if it wasn't already sent
    if not skip_mechanics:
        mech_embed = _build_mechanics_embed(response.mechanical_result, response.dice_rolls)
        if mech_embed:
            embeds.append(mech_embed)

    # --- Build narrative embed(s) ---
    narrative = response.narrative or "*The scene unfolds...*"

    if len(narrative) > EMBED_DESCRIPTION_LIMIT:
        chunks = split_text(narrative, EMBED_DESCRIPTION_LIMIT - 100)
        for i, chunk in enumerate(chunks):
            embed = discord.Embed(
                description=chunk,
                color=discord.Color.dark_gold(),
            )
            if len(chunks) > 1:
                embed.set_footer(text=f"({i+1}/{len(chunks)})")
            embeds.append(embed)
    else:
        embeds.append(discord.Embed(
            description=narrative,
            color=discord.Color.dark_gold(),
        ))

    return embeds


class GameCog(commands.Cog):
    """Game session management and message handling."""

    game = discord.SlashCommandGroup(
        "game",
        "Game session management",
    )

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.session_manager = get_session_manager()

    async def _get_campaign(self, guild_id: int, campaign_name: Optional[str] = None):
        """Get the campaign object, using active campaign if not specified."""
        campaign_repo = await get_campaign_repo()

        if campaign_name:
            return await campaign_repo.get_by_name_and_guild(campaign_name, guild_id)

        # Try active campaign first
        active_id = get_active_campaign_id(guild_id)
        if active_id:
            campaign = await campaign_repo.get_by_id(active_id)
            if campaign:
                return campaign

        # Fall back to first campaign
        campaigns = await campaign_repo.get_all_by_guild(guild_id)
        return campaigns[0] if campaigns else None

    @game.command(name="start", description="Start a new game session in this channel")
    async def game_start(
        self,
        ctx: discord.ApplicationContext,
        campaign_name: discord.Option(
            str,
            "Campaign name (uses active campaign if not specified)",
            required=False,
            default=None,
        ),
    ):
        """Start a new game session."""
        # Get the campaign
        campaign = await self._get_campaign(ctx.guild_id, campaign_name)
        if not campaign:
            await ctx.respond(
                "No campaign found. Create one with `/campaign create` first.",
                ephemeral=True,
            )
            return

        # Check if session already active
        if self.session_manager.has_active_session(ctx.channel_id):
            await ctx.respond(
                "A game session is already active in this channel. Use `/game end` first.",
                ephemeral=True,
            )
            return

        # Get characters in this campaign
        char_repo = await get_character_repo()
        characters = await char_repo.get_all_by_campaign(campaign.id)

        # Start the session with actual campaign ID
        session = await self.session_manager.start_session(
            channel_id=ctx.channel_id,
            guild_id=ctx.guild_id,
            campaign_id=campaign.id,
            dm_user_id=None,  # AI is the DM, no human DM
        )

        # Auto-join all players who have characters
        for character in characters:
            member = ctx.guild.get_member(character.discord_user_id)
            if member:
                await self.session_manager.join_session(
                    channel_id=ctx.channel_id,
                    user_id=character.discord_user_id,
                    user_name=member.display_name,
                    character=character,
                )

        # Build player list
        player_lines = []
        for char in characters:
            member = ctx.guild.get_member(char.discord_user_id)
            name = member.display_name if member else "Unknown"
            player_lines.append(f"**{char.name}** ({name}) - L{char.level} {char.class_index.title()}")

        embed = discord.Embed(
            title=f":crossed_swords: {campaign.name} Begins!",
            description=campaign.world_setting or "A new adventure awaits...",
            color=discord.Color.green(),
        )

        if characters:
            embed.add_field(
                name=f"Players ({len(characters)})",
                value="\n".join(player_lines),
                inline=False,
            )
        else:
            embed.add_field(
                name="Players",
                value="No players yet. Use `/game join` to join!",
                inline=False,
            )

        embed.add_field(
            name="How to Play",
            value=(
                "**Players:** Describe your actions in chat. The AI DM will respond.\n"
                "**DM:** Your messages won't be processed by AI - narrate directly.\n"
                "Use `/game status` to see session info."
            ),
            inline=False,
        )

        await ctx.respond(embed=embed)

        logger.info(
            "game_session_started",
            channel=ctx.channel_id,
            campaign_id=campaign.id,
            campaign_name=campaign.name,
            dm=ctx.author.id,
            players=len(characters),
        )

    @game.command(name="join", description="Join the active game session with your character")
    async def game_join(
        self,
        ctx: discord.ApplicationContext,
    ):
        """Join the game session with a character."""
        session = self.session_manager.get_session(ctx.channel_id)
        if not session:
            await ctx.respond(
                "No active game session. Ask the DM to click 'Start Game' or use `/game start`.",
                ephemeral=True,
            )
            return

        if session.get_player(ctx.author.id):
            await ctx.respond(
                "You're already in this session!",
                ephemeral=True,
            )
            return

        # Get character from the session's campaign
        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(ctx.author.id, session.campaign_id)

        if not character:
            await ctx.respond(
                "You don't have a character in this campaign. Click 'Join Campaign' in the lobby to create one!",
                ephemeral=True,
            )
            return

        # Join session
        player = await self.session_manager.join_session(
            channel_id=ctx.channel_id,
            user_id=ctx.author.id,
            user_name=ctx.author.display_name,
            character=character,
        )

        if player:
            await ctx.respond(
                f":crossed_swords: **{character.name}** joins the adventure!\n"
                f"*Level {character.level} {character.race_index.title()} {character.class_index.title()}*"
            )
        else:
            await ctx.respond(
                "Failed to join the session.",
                ephemeral=True,
            )

    @game.command(name="leave", description="Leave the active game session")
    async def game_leave(
        self,
        ctx: discord.ApplicationContext,
    ):
        """Leave the game session."""
        player = await self.session_manager.leave_session(
            channel_id=ctx.channel_id,
            user_id=ctx.author.id,
        )

        if player:
            char_name = player.character.name if player.character else ctx.author.display_name
            await ctx.respond(f":wave: **{char_name}** has left the adventure.")
        else:
            await ctx.respond(
                "You're not in a game session.",
                ephemeral=True,
            )

    @game.command(name="status", description="Show the current game session status")
    async def game_status(
        self,
        ctx: discord.ApplicationContext,
    ):
        """Show session status."""
        session = self.session_manager.get_session(ctx.channel_id)
        if not session:
            await ctx.respond(
                "No active game session in this channel.",
                ephemeral=True,
            )
            return

        embed = discord.Embed(
            title=f":game_die: Game Session - {session.campaign_id}",
            color=discord.Color.blue(),
        )

        # State
        state_icons = {
            SessionState.ACTIVE: ":green_circle: Active",
            SessionState.COMBAT: ":crossed_swords: In Combat",
            SessionState.PAUSED: ":pause_button: Paused",
            SessionState.ENDED: ":stop_button: Ended",
        }
        embed.add_field(
            name="Status",
            value=state_icons.get(session.state, str(session.state)),
            inline=True,
        )

        # DM
        if session.dm_user_id:
            dm = ctx.guild.get_member(session.dm_user_id)
            embed.add_field(
                name="DM",
                value=dm.display_name if dm else "Unknown",
                inline=True,
            )

        # Duration
        duration = (discord.utils.utcnow() - session.started_at.replace(tzinfo=None)).seconds // 60
        embed.add_field(
            name="Duration",
            value=f"{duration} minutes",
            inline=True,
        )

        # Players
        if session.players:
            player_lines = []
            for user_id, player in session.players.items():
                member = ctx.guild.get_member(user_id)
                name = member.display_name if member else "Unknown"
                if player.character:
                    char = player.character
                    hp_pct = int((char.hp.current / char.hp.maximum) * 100)
                    player_lines.append(
                        f"**{char.name}** ({name}) - "
                        f"L{char.level} {char.class_index.title()} - "
                        f"HP: {char.hp.current}/{char.hp.maximum} ({hp_pct}%)"
                    )
                else:
                    player_lines.append(f"{name} (no character)")

            embed.add_field(
                name=f"Players ({len(session.players)})",
                value="\n".join(player_lines) or "No players",
                inline=False,
            )
        else:
            embed.add_field(
                name="Players",
                value="No players have joined yet. Use `/game join` to join!",
                inline=False,
            )

        await ctx.respond(embed=embed)

    @game.command(name="end", description="End the current game session")
    async def game_end(
        self,
        ctx: discord.ApplicationContext,
    ):
        """End the game session."""
        session = self.session_manager.get_session(ctx.channel_id)
        if not session:
            await ctx.respond(
                "No active game session to end.",
                ephemeral=True,
            )
            return

        # Only DM or admin can end
        if session.dm_user_id != ctx.author.id and not ctx.author.guild_permissions.administrator:
            await ctx.respond(
                "Only the DM can end the session.",
                ephemeral=True,
            )
            return

        await self.session_manager.end_session(ctx.channel_id)

        embed = discord.Embed(
            title=":stop_button: Game Session Ended",
            description="The adventure pauses here... for now.",
            color=discord.Color.greyple(),
        )

        duration = (discord.utils.utcnow() - session.started_at.replace(tzinfo=None)).seconds // 60
        embed.add_field(
            name="Session Stats",
            value=f"Duration: {duration} minutes\nPlayers: {len(session.players)}",
            inline=False,
        )

        await ctx.respond(embed=embed)

    @game.command(name="scene", description="Set the current scene (DM only)")
    async def game_scene(
        self,
        ctx: discord.ApplicationContext,
        description: discord.Option(
            str,
            "Description of the current scene",
            required=True,
        ),
    ):
        """Set the current scene."""
        session = self.session_manager.get_session(ctx.channel_id)
        if not session:
            await ctx.respond(
                "No active game session.",
                ephemeral=True,
            )
            return

        if not session.is_dm(ctx.author.id):
            await ctx.respond(
                "Only the DM can set the scene.",
                ephemeral=True,
            )
            return

        # Update memory
        from ...memory import get_memory_manager
        memory = await get_memory_manager(session.campaign_id)
        memory.update_scene(description)

        embed = discord.Embed(
            title=":scroll: Scene Set",
            description=description,
            color=discord.Color.purple(),
        )

        await ctx.respond(embed=embed)

    async def _show_player_turn_ui(
        self,
        channel: discord.TextChannel,
        coordinator,
        manager,
        combatant,
    ) -> None:
        """Show the player action UI for a combatant's turn."""
        turn_ctx = await coordinator.start_turn(combatant)

        async def on_action_complete(result):
            import time as _time
            result_embed = ActionResultEmbed.build(result)
            await channel.send(embed=result_embed)

            # Narrate both hits and misses — use embed to match regular narration style
            _t0 = _time.monotonic()
            try:
                narrative = await coordinator.narrate_result(result)
                _t1 = _time.monotonic()
                logger.info("timing_player_narration", elapsed=f"{_t1 - _t0:.1f}s")
                if narrative:
                    narr_embed = discord.Embed(
                        description=narrative,
                        color=discord.Color.dark_gold(),
                    )
                    await channel.send(embed=narr_embed)
            except Exception as e:
                logger.warning("narration_failed", error=str(e))

            # Check combat end
            if manager.combat.is_combat_over():
                from ..embeds.combat_embed import build_combat_end_embed
                from ...game.combat.manager import clear_combat_for_channel
                from ...game.combat.coordinator import clear_coordinator

                players_alive = any(
                    c.is_player and c.hp_current > 0
                    for c in manager.combat.combatants
                )
                end_embed = build_combat_end_embed(manager.combat, victory=players_alive)
                await channel.send(embed=end_embed)
                manager.end_combat()
                clear_combat_for_channel(channel.id)
                clear_coordinator(channel.id)
                return

            # End turn and advance to next combatant
            await on_turn_end()

        async def on_turn_end():
            end_result = await coordinator.end_turn(combatant)

            next_msg = f":arrow_right: **{end_result.next_combatant_name}**'s turn!"
            if end_result.round_advanced:
                next_msg = f"**Round {end_result.new_round}**\n{next_msg}"
            await channel.send(next_msg)

            # Auto-run NPC turns instead of requiring slash commands
            if not end_result.next_is_player:
                next_combatant = manager.combat.get_current_combatant()
                if next_combatant:
                    await self._auto_run_npc_turns(channel, coordinator, manager)

        view = CombatActionView(
            coordinator=coordinator,
            turn_context=turn_ctx,
            on_action_complete=on_action_complete,
            on_turn_end=on_turn_end,
        )

        await channel.send(
            f":crossed_swords: **{combatant.name}**, it's your turn!",
            embed=view.get_embed(),
            view=view,
        )

    async def _auto_run_npc_turns(
        self,
        channel: discord.TextChannel,
        coordinator,
        manager,
    ) -> None:
        """Auto-run all consecutive NPC turns, then show player UI."""
        from ..embeds.combat_embed import build_combat_end_embed
        from ...game.combat.manager import clear_combat_for_channel
        from ...game.combat.coordinator import clear_coordinator

        max_turns = 10  # Safety limit
        turns_run = 0

        while turns_run < max_turns:
            current = manager.combat.get_current_combatant()
            if not current or current.is_player:
                break

            turns_run += 1
            await channel.send(f":skull: **{current.name}**'s turn...")

            import time as _time

            _t0 = _time.monotonic()
            try:
                results = await coordinator.run_npc_turn(current)
            except Exception as e:
                logger.error("npc_turn_failed", combatant=current.name, error=str(e))
                await channel.send(f"*{current.name} hesitates...* (Error: {str(e)[:80]})")
                try:
                    await coordinator.end_turn(current)
                except Exception:
                    pass
                continue
            _t1 = _time.monotonic()
            logger.info("timing_npc_turn", combatant=current.name, elapsed=f"{_t1 - _t0:.1f}s")

            for result in results:
                result_embed = ActionResultEmbed.build(result)
                await channel.send(embed=result_embed)

                # Skip narration for END_TURN actions (surprised NPCs, etc.)
                if result.action.action_type == CombatActionType.END_TURN:
                    continue

                _t2 = _time.monotonic()
                try:
                    narrative = await coordinator.narrate_result(result)
                    _t3 = _time.monotonic()
                    logger.info("timing_npc_narration", combatant=current.name, elapsed=f"{_t3 - _t2:.1f}s")
                    if narrative:
                        narr_embed = discord.Embed(
                            description=narrative,
                            color=discord.Color.dark_gold(),
                        )
                        await channel.send(embed=narr_embed)
                except Exception as e:
                    logger.warning("narration_failed", error=str(e))

            # Check combat end
            if manager.combat.is_combat_over():
                players_alive = any(
                    c.is_player and c.hp_current > 0
                    for c in manager.combat.combatants
                )
                end_embed = build_combat_end_embed(manager.combat, victory=players_alive)
                await channel.send(embed=end_embed)
                manager.end_combat()
                clear_combat_for_channel(channel.id)
                clear_coordinator(channel.id)
                return

        # After NPC turns, show player turn UI if it's a player's turn
        current = manager.combat.get_current_combatant()
        if current and current.is_player:
            await self._show_player_turn_ui(channel, coordinator, manager, current)

    async def _show_combat_ui(self, channel: discord.TextChannel) -> None:
        """Show the combat UI after combat is auto-triggered from narrative."""
        manager = get_combat_for_channel(channel.id)
        if not manager:
            logger.warning("combat_ui_no_manager", channel=channel.id)
            return

        # Show combat start embed with initiative order
        start_embed = build_combat_start_embed(manager.combat)
        tracker_embed = build_combat_tracker_embed(manager.combat)
        await channel.send(embeds=[start_embed, tracker_embed])

        # Get current combatant
        current = manager.combat.get_current_combatant()
        if not current:
            return

        coordinator = get_coordinator(manager)

        # Auto-skip surprised NPCs — they can't act but their turn must be processed
        while current and not current.is_player and current.is_surprised:
            await channel.send(
                f":dizzy_face: **{current.name}** is surprised and cannot act!"
            )
            await coordinator.start_turn(current)
            await coordinator.end_turn(current)
            current = manager.combat.get_current_combatant()

        if not current:
            return

        if current.is_player:
            await self._show_player_turn_ui(channel, coordinator, manager, current)
        else:
            # Non-surprised NPC goes first — show button to run NPC turns
            async def on_npc_turns_complete():
                next_combatant = manager.combat.get_current_combatant()
                if next_combatant and next_combatant.is_player:
                    await self._show_player_turn_ui(
                        channel, coordinator, manager, next_combatant
                    )

            npc_view = NPCTurnView(
                coordinator=coordinator,
                channel=channel,
                on_turns_complete=on_npc_turns_complete,
            )

            await channel.send(
                f":skull: **{current.name}** acts first!",
                view=npc_view,
            )

    @commands.Cog.listener()
    async def on_message(self, message: discord.Message):
        """
        Listen for player messages and route through the LLM DM.

        This is the core game loop entry point.
        """
        # Ignore bot messages
        if message.author.bot:
            return

        # Ignore DMs
        if not message.guild:
            return

        # Ignore commands
        if message.content.startswith("/"):
            return

        # Check if there's an active session in this channel
        if not self.session_manager.has_active_session(message.channel.id):
            return

        # Check if player is in the session
        session = self.session_manager.get_session(message.channel.id)
        if not session:
            return

        player = session.get_player(message.author.id)
        if not player:
            # Not a joined player, ignore
            return

        # Show typing indicator while processing
        async with message.channel.typing():
            try:
                # Callback to send mechanics (rolls/checks) immediately
                mechanics_sent = False

                async def send_mechanics(mechanical_result, dice_rolls):
                    nonlocal mechanics_sent
                    mech_embed = _build_mechanics_embed(mechanical_result, dice_rolls)
                    if mech_embed:
                        await message.channel.send(embed=mech_embed)
                        mechanics_sent = True

                response = await self.session_manager.process_message(
                    channel_id=message.channel.id,
                    user_id=message.author.id,
                    user_name=message.author.display_name,
                    content=message.content,
                    on_mechanics_ready=send_mechanics,
                )

                if response:
                    embeds = format_dm_response(response, skip_mechanics=mechanics_sent)
                    for embed in embeds:
                        await message.channel.send(embed=embed)

                    # Check if combat was triggered - show combat UI
                    if response.combat_triggered:
                        await self._show_combat_ui(message.channel)

            except Exception as e:
                logger.error(
                    "message_handling_error",
                    channel=message.channel.id,
                    user=message.author.id,
                    error=str(e),
                )
                await message.channel.send(
                    f":warning: *The DM pauses, consulting ancient scrolls...* (Error: {str(e)[:100]})"
                )


def setup(bot: commands.Bot):
    """Load the cog."""
    bot.add_cog(GameCog(bot))
