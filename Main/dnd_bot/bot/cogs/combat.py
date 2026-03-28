"""Combat commands cog."""

import discord
from discord.ext import commands
import structlog

from ...models import CombatState
from ...data.repositories import get_character_repo
from ...game.combat.manager import (
    CombatManager,
    get_combat_for_channel,
    set_combat_for_channel,
    clear_combat_for_channel,
)
from ...game.combat.coordinator import (
    CombatTurnCoordinator,
    get_coordinator,
    clear_coordinator,
)
from ..views.combat_actions import (
    CombatActionView,
    ActionResultEmbed,
)
from ..embeds.combat_embed import (
    build_combat_tracker_embed,
    build_initiative_results_embed,
    build_combat_start_embed,
    build_combat_end_embed,
    build_attack_result_embed,
)
from ...game.mechanics.dice import get_roller

logger = structlog.get_logger()


class CombatCog(commands.Cog):
    """Combat encounter management commands."""

    combat = discord.SlashCommandGroup(
        "combat",
        "Combat encounter management",
    )

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.roller = get_roller()

    async def _sync_player_characters(
        self,
        manager: CombatManager,
        campaign_id: str = "default",
    ) -> None:
        """
        Sync all player combatant HP/state back to the database.

        Call this after damage, healing, or combat end.
        """
        repo = await get_character_repo()
        for combatant in manager.get_player_combatants():
            if combatant.character_id:
                character = await repo.get_by_id(combatant.character_id)
                if character:
                    manager.sync_to_character(combatant, character)
                    await repo.update(character)
                    logger.debug(
                        "character_synced_after_combat",
                        character=character.name,
                        hp=f"{character.hp.current}/{character.hp.maximum}",
                    )

    @combat.command(name="start", description="Start a new combat encounter")
    async def combat_start(
        self,
        ctx: discord.ApplicationContext,
        name: discord.Option(
            str,
            "Encounter name",
            required=False,
        ),
        description: discord.Option(
            str,
            "Encounter description",
            required=False,
        ),
    ):
        """Start a new combat encounter in this channel."""
        # Check if combat already active
        existing = get_combat_for_channel(ctx.channel_id)
        if existing and existing.combat.state != CombatState.COMBAT_END:
            await ctx.respond(
                "A combat is already active in this channel. Use `/combat end` to end it first.",
                ephemeral=True,
            )
            return

        # Create new combat
        manager = CombatManager.create_encounter(
            session_id="default",  # TODO: Get from active session
            channel_id=ctx.channel_id,
            name=name,
            description=description,
        )
        set_combat_for_channel(ctx.channel_id, manager)

        embed = discord.Embed(
            title=":crossed_swords: Combat Started!",
            description=description or "A new combat encounter has begun.",
            color=discord.Color.red(),
        )

        if name:
            embed.title = f":crossed_swords: {name}"

        embed.add_field(
            name="Next Steps",
            value=(
                "1. Add combatants with `/combat add`\n"
                "2. Roll initiative with `/combat initiative roll`\n"
                "3. Begin combat with `/combat begin`"
            ),
            inline=False,
        )

        await ctx.respond(embed=embed)

        logger.info(
            "combat_started",
            channel=ctx.channel_id,
            name=name,
        )

    @combat.command(name="add_player", description="Add a player character to combat")
    async def combat_add_player(
        self,
        ctx: discord.ApplicationContext,
        member: discord.Option(
            discord.Member,
            "Player to add (uses their character)",
            required=False,
        ),
        campaign_id: discord.Option(
            str,
            "Campaign ID",
            required=False,
            default="default",
        ),
    ):
        """Add a player character to the combat."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat. Use `/combat start` first.",
                ephemeral=True,
            )
            return

        if manager.combat.state not in (CombatState.SETUP, CombatState.ROLLING_INITIATIVE):
            await ctx.respond(
                "Cannot add combatants after combat has started.",
                ephemeral=True,
            )
            return

        # Default to command user
        target = member or ctx.author

        # Get character
        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(target.id, campaign_id)

        if not character:
            await ctx.respond(
                f"{target.display_name} doesn't have a character in this campaign.",
                ephemeral=True,
            )
            return

        # Check if already added
        existing = manager.get_combatant_by_name(character.name)
        if existing:
            await ctx.respond(
                f"{character.name} is already in combat.",
                ephemeral=True,
            )
            return

        combatant = manager.add_player(character)

        await ctx.respond(
            f":crossed_swords: **{character.name}** joins the fray! "
            f"(HP: {combatant.hp_current}/{combatant.hp_max}, AC: {combatant.armor_class})"
        )

    @combat.command(name="add_monster", description="Add a monster to combat")
    async def combat_add_monster(
        self,
        ctx: discord.ApplicationContext,
        monster: discord.Option(
            str,
            "Monster type (e.g., 'goblin', 'adult-red-dragon')",
            required=True,
        ),
        name: discord.Option(
            str,
            "Custom name for this monster",
            required=False,
        ),
        hp: discord.Option(
            int,
            "Custom HP (uses default if not specified)",
            required=False,
            min_value=1,
        ),
        count: discord.Option(
            int,
            "Number to add",
            required=False,
            default=1,
            min_value=1,
            max_value=20,
        ),
    ):
        """Add a monster from the SRD to combat."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat. Use `/combat start` first.",
                ephemeral=True,
            )
            return

        if manager.combat.state not in (CombatState.SETUP, CombatState.ROLLING_INITIATIVE):
            await ctx.respond(
                "Cannot add combatants after combat has started.",
                ephemeral=True,
            )
            return

        added = []
        for i in range(count):
            # Generate name if adding multiple
            if count > 1:
                display_name = f"{name or monster.replace('-', ' ').title()} {i + 1}"
            else:
                display_name = name

            combatant = manager.add_monster(monster, name=display_name, hp=hp)
            if combatant:
                added.append(combatant)

        if not added:
            await ctx.respond(
                f"Monster '{monster}' not found in SRD. Try using the index format (e.g., 'goblin', 'adult-red-dragon').",
                ephemeral=True,
            )
            return

        if len(added) == 1:
            c = added[0]
            await ctx.respond(
                f":skull: **{c.name}** enters combat! "
                f"(HP: {c.hp_current}/{c.hp_max}, AC: {c.armor_class})"
            )
        else:
            names = ", ".join(c.name for c in added)
            await ctx.respond(
                f":skull: **{len(added)} monsters** enter combat: {names}"
            )

    @combat.command(name="add_custom", description="Add a custom combatant")
    async def combat_add_custom(
        self,
        ctx: discord.ApplicationContext,
        name: discord.Option(
            str,
            "Combatant name",
            required=True,
        ),
        hp: discord.Option(
            int,
            "Hit points",
            required=True,
            min_value=1,
        ),
        ac: discord.Option(
            int,
            "Armor class",
            required=True,
            min_value=1,
            max_value=30,
        ),
        initiative_bonus: discord.Option(
            int,
            "Initiative modifier",
            required=False,
            default=0,
        ),
        is_player: discord.Option(
            bool,
            "Is this a player character?",
            required=False,
            default=False,
        ),
    ):
        """Add a custom combatant to combat."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat. Use `/combat start` first.",
                ephemeral=True,
            )
            return

        if manager.combat.state not in (CombatState.SETUP, CombatState.ROLLING_INITIATIVE):
            await ctx.respond(
                "Cannot add combatants after combat has started.",
                ephemeral=True,
            )
            return

        combatant = manager.add_custom_combatant(
            name=name,
            hp=hp,
            ac=ac,
            initiative_bonus=initiative_bonus,
            is_player=is_player,
        )

        icon = ":crossed_swords:" if is_player else ":skull:"
        await ctx.respond(
            f"{icon} **{combatant.name}** joins combat! "
            f"(HP: {hp}, AC: {ac}, Init: {'+' if initiative_bonus >= 0 else ''}{initiative_bonus})"
        )

    @combat.command(name="remove", description="Remove a combatant from combat")
    async def combat_remove(
        self,
        ctx: discord.ApplicationContext,
        name: discord.Option(
            str,
            "Combatant name",
            required=True,
        ),
    ):
        """Remove a combatant from combat."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat.",
                ephemeral=True,
            )
            return

        combatant = manager.get_combatant_by_name(name)
        if not combatant:
            await ctx.respond(
                f"No combatant found matching '{name}'.",
                ephemeral=True,
            )
            return

        manager.remove_combatant(combatant.id)
        await ctx.respond(f":x: **{combatant.name}** has been removed from combat.")

    @combat.command(name="initiative", description="Roll initiative for all combatants")
    async def combat_roll_initiative(
        self,
        ctx: discord.ApplicationContext,
    ):
        """Roll initiative for all combatants."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat. Use `/combat start` first.",
                ephemeral=True,
            )
            return

        if not manager.combat.combatants:
            await ctx.respond(
                "No combatants in combat. Add some with `/combat add_player` or `/combat add_monster`.",
                ephemeral=True,
            )
            return

        manager.combat.transition(CombatState.ROLLING_INITIATIVE)
        results = manager.roll_all_initiative()

        embed = build_initiative_results_embed(results)
        await ctx.respond(embed=embed)

    @combat.command(name="begin", description="Begin combat after initiative is rolled")
    async def combat_begin(
        self,
        ctx: discord.ApplicationContext,
    ):
        """Begin combat and start the first turn."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat.",
                ephemeral=True,
            )
            return

        if manager.combat.state == CombatState.AWAITING_ACTION:
            await ctx.respond(
                "Combat is already in progress!",
                ephemeral=True,
            )
            return

        if not manager.combat.combatants:
            await ctx.respond(
                "No combatants! Add some before starting.",
                ephemeral=True,
            )
            return

        current = manager.start_combat()

        start_embed = build_combat_start_embed(manager.combat)
        tracker_embed = build_combat_tracker_embed(manager.combat)

        await ctx.respond(embeds=[start_embed, tracker_embed])

        if current:
            await ctx.send(
                f":arrow_right: **{current.name}**, it's your turn! "
                f"(HP: {current.hp_current}/{current.hp_max})"
            )

    @combat.command(name="tracker", description="Show the combat tracker")
    async def combat_tracker(
        self,
        ctx: discord.ApplicationContext,
        show_hp: discord.Option(
            bool,
            "Show HP values",
            required=False,
            default=True,
        ),
        show_ac: discord.Option(
            bool,
            "Show AC values",
            required=False,
            default=False,
        ),
    ):
        """Display the current combat tracker."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat.",
                ephemeral=True,
            )
            return

        embed = build_combat_tracker_embed(
            manager.combat,
            show_hp=show_hp,
            show_ac=show_ac,
        )
        await ctx.respond(embed=embed)

    @combat.command(name="next", description="End current turn and move to next combatant")
    async def combat_next_turn(
        self,
        ctx: discord.ApplicationContext,
    ):
        """Advance to the next combatant's turn."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat.",
                ephemeral=True,
            )
            return

        if manager.combat.state not in (CombatState.AWAITING_ACTION, CombatState.ACTIVE):
            await ctx.respond(
                "Combat is not in progress.",
                ephemeral=True,
            )
            return

        current_before = manager.combat.get_current_combatant()
        next_combatant = manager.next_turn()

        if manager.combat.state == CombatState.COMBAT_END:
            # Combat ended
            players_alive = any(
                c.is_player and c.hp_current > 0
                for c in manager.combat.combatants
            )
            embed = build_combat_end_embed(manager.combat, victory=players_alive)
            await ctx.respond(embed=embed)
            clear_combat_for_channel(ctx.channel_id)
            return

        if next_combatant:
            if current_before:
                end_msg = f":stop_button: **{current_before.name}**'s turn ends."
            else:
                end_msg = ""

            tracker_embed = build_combat_tracker_embed(manager.combat)
            await ctx.respond(
                f"{end_msg}\n\n:arrow_right: **{next_combatant.name}**, it's your turn!",
                embed=tracker_embed,
            )
        else:
            await ctx.respond("Combat has ended!")

    @combat.command(name="attack", description="Make an attack in combat")
    async def combat_attack(
        self,
        ctx: discord.ApplicationContext,
        target: discord.Option(
            str,
            "Target name",
            required=True,
        ),
        damage_dice: discord.Option(
            str,
            "Damage dice (e.g., 1d8+3)",
            required=True,
        ),
        attack_bonus: discord.Option(
            int,
            "Attack bonus modifier",
            required=False,
            default=0,
        ),
        advantage: discord.Option(
            bool,
            "Roll with advantage",
            required=False,
            default=False,
        ),
        disadvantage: discord.Option(
            bool,
            "Roll with disadvantage",
            required=False,
            default=False,
        ),
    ):
        """Make an attack against a target in combat."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat.",
                ephemeral=True,
            )
            return

        if manager.combat.state != CombatState.AWAITING_ACTION:
            await ctx.respond(
                "Combat is not awaiting actions.",
                ephemeral=True,
            )
            return

        # Get current combatant (attacker)
        attacker = manager.combat.get_current_combatant()
        if not attacker:
            await ctx.respond(
                "No current combatant.",
                ephemeral=True,
            )
            return

        # Find target
        target_combatant = manager.get_combatant_by_name(target)
        if not target_combatant:
            await ctx.respond(
                f"No combatant found matching '{target}'.",
                ephemeral=True,
            )
            return

        if not target_combatant.is_active or target_combatant.hp_current <= 0:
            await ctx.respond(
                f"{target_combatant.name} is already down!",
                ephemeral=True,
            )
            return

        # Cancel out advantage/disadvantage
        if advantage and disadvantage:
            advantage = False
            disadvantage = False

        # Roll attack
        attack_roll = self.roller.roll_attack(
            modifier=attack_bonus,
            advantage=advantage,
            disadvantage=disadvantage,
        )

        # Determine hit
        is_critical = attack_roll.natural_20
        is_fumble = attack_roll.natural_1
        is_hit = is_critical or (not is_fumble and attack_roll.total >= target_combatant.armor_class)

        damage_dealt = 0
        damage_roll = None

        is_unconscious = False
        is_instant_death = False

        if is_hit:
            # Roll damage
            damage_roll = self.roller.roll_damage(damage_dice, critical=is_critical)
            damage_dealt = damage_roll.total

            # Apply damage
            actual_damage, is_unconscious, is_instant_death = manager.apply_damage(
                target_combatant.id,
                damage_dealt,
                is_critical=is_critical,
            )

            # Use the attacker's action
            manager.use_action(attacker.id)

        embed = build_attack_result_embed(
            attacker_name=attacker.name,
            target_name=target_combatant.name,
            attack_roll=attack_roll,
            target_ac=target_combatant.armor_class,
            hit=is_hit,
            critical=is_critical,
            damage_roll=damage_roll,
            damage_dealt=damage_dealt,
        )

        response_text = ""
        if is_instant_death:
            response_text = f"\n\n:skull: **INSTANT DEATH!** {target_combatant.name} takes massive damage and dies instantly!"
        elif is_unconscious:
            if target_combatant.is_player:
                response_text = f"\n\n:warning: **{target_combatant.name}** falls unconscious and begins making death saves!"
            else:
                response_text = f"\n\n:skull: **{target_combatant.name}** is down!"

        await ctx.respond(content=response_text if response_text else None, embed=embed)

        # Check if combat should end
        if manager.combat.is_combat_over():
            players_alive = any(
                c.is_player and c.hp_current > 0
                for c in manager.combat.combatants
            )
            end_embed = build_combat_end_embed(manager.combat, victory=players_alive)
            await ctx.send(embed=end_embed)
            manager.end_combat()
            clear_combat_for_channel(ctx.channel_id)

    @combat.command(name="damage", description="Apply damage to a combatant")
    async def combat_damage(
        self,
        ctx: discord.ApplicationContext,
        target: discord.Option(
            str,
            "Target name",
            required=True,
        ),
        amount: discord.Option(
            int,
            "Damage amount",
            required=True,
            min_value=1,
        ),
        damage_type: discord.Option(
            str,
            "Damage type",
            required=False,
            choices=[
                "bludgeoning", "piercing", "slashing",
                "fire", "cold", "lightning", "thunder",
                "acid", "poison", "necrotic", "radiant",
                "force", "psychic",
            ],
        ),
    ):
        """Apply damage to a combatant."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat.",
                ephemeral=True,
            )
            return

        combatant = manager.get_combatant_by_name(target)
        if not combatant:
            await ctx.respond(
                f"No combatant found matching '{target}'.",
                ephemeral=True,
            )
            return

        actual_damage, is_unconscious, is_instant_death = manager.apply_damage(
            combatant.id,
            amount,
            damage_type=damage_type,
        )

        # Sync player state to database
        if combatant.is_player:
            await self._sync_player_characters(manager)

        type_text = f" {damage_type}" if damage_type else ""
        await ctx.respond(
            f":boom: **{combatant.name}** takes **{actual_damage}{type_text}** damage! "
            f"(HP: {combatant.hp_current}/{combatant.hp_max})"
        )

        if is_instant_death:
            await ctx.send(f":skull: **INSTANT DEATH!** {combatant.name} dies instantly!")
        elif is_unconscious:
            if combatant.is_player:
                await ctx.send(f":warning: **{combatant.name}** falls unconscious and must make death saves!")
            else:
                await ctx.send(f":skull: **{combatant.name}** falls unconscious!")

            # Check combat end
            if manager.combat.is_combat_over():
                players_alive = any(
                    c.is_player and c.hp_current > 0
                    for c in manager.combat.combatants
                )
                await self._sync_player_characters(manager)
                embed = build_combat_end_embed(manager.combat, victory=players_alive)
                await ctx.send(embed=embed)
                manager.end_combat()
                clear_combat_for_channel(ctx.channel_id)

    @combat.command(name="heal", description="Heal a combatant")
    async def combat_heal(
        self,
        ctx: discord.ApplicationContext,
        target: discord.Option(
            str,
            "Target name",
            required=True,
        ),
        amount: discord.Option(
            int,
            "Healing amount",
            required=True,
            min_value=1,
        ),
    ):
        """Apply healing to a combatant."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat.",
                ephemeral=True,
            )
            return

        combatant = manager.get_combatant_by_name(target)
        if not combatant:
            await ctx.respond(
                f"No combatant found matching '{target}'.",
                ephemeral=True,
            )
            return

        actual_healing, was_revived = manager.apply_healing(combatant.id, amount)

        # Sync player state to database
        if combatant.is_player:
            await self._sync_player_characters(manager)

        response = (
            f":green_heart: **{combatant.name}** heals for **{actual_healing}** HP! "
            f"(HP: {combatant.hp_current}/{combatant.hp_max})"
        )

        if was_revived:
            response += f"\n\n:sparkles: **{combatant.name}** regains consciousness!"

        await ctx.respond(response)

    @combat.command(name="death_save", description="Roll a death saving throw")
    async def combat_death_save(
        self,
        ctx: discord.ApplicationContext,
        target: discord.Option(
            str,
            "Character name (defaults to your character)",
            required=False,
        ),
        campaign_id: discord.Option(
            str,
            "Campaign ID",
            required=False,
            default="default",
        ),
    ):
        """Roll a death saving throw for an unconscious character."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat.",
                ephemeral=True,
            )
            return

        # Find the combatant
        if target:
            combatant = manager.get_combatant_by_name(target)
        else:
            # Try to find user's character
            repo = await get_character_repo()
            character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)
            if character:
                combatant = manager.get_combatant_by_name(character.name)
            else:
                combatant = None

        if not combatant:
            await ctx.respond(
                "Could not find your character in combat. Specify a target name.",
                ephemeral=True,
            )
            return

        if not combatant.is_dying:
            if combatant.hp_current > 0:
                await ctx.respond(
                    f"{combatant.name} is conscious and doesn't need death saves!",
                    ephemeral=True,
                )
            elif combatant.is_stable:
                await ctx.respond(
                    f"{combatant.name} is stable and doesn't need death saves.",
                    ephemeral=True,
                )
            elif combatant.death_saves.is_dead:
                await ctx.respond(
                    f"{combatant.name} is dead...",
                    ephemeral=True,
                )
            else:
                await ctx.respond(
                    f"{combatant.name} cannot make death saves.",
                    ephemeral=True,
                )
            return

        roll, result = manager.roll_death_save(combatant.id)

        # Build response based on result
        ds = combatant.death_saves
        saves_display = f"Successes: {':white_check_mark:' * ds.successes}{':black_large_square:' * (3 - ds.successes)} | Failures: {':x:' * ds.failures}{':black_large_square:' * (3 - ds.failures)}"

        if result == "critical_success":
            embed = discord.Embed(
                title=f":star2: NATURAL 20! {combatant.name} regains consciousness!",
                description=f"**Roll:** [{roll.kept_dice[0]}] = **{roll.total}**\n\n{combatant.name} wakes up with 1 HP!",
                color=discord.Color.gold(),
            )
        elif result == "critical_failure":
            embed = discord.Embed(
                title=f":skull: NATURAL 1! Critical Failure!",
                description=f"**Roll:** [{roll.kept_dice[0]}] = **{roll.total}**\n\nThis counts as **2 failures**!\n\n{saves_display}",
                color=discord.Color.dark_red(),
            )
        elif result == "stabilized":
            embed = discord.Embed(
                title=f":relief: {combatant.name} stabilizes!",
                description=f"**Roll:** [{roll.kept_dice[0]}] = **{roll.total}** (Success!)\n\n{combatant.name} is stable but unconscious.\n\n{saves_display}",
                color=discord.Color.green(),
            )
        elif result == "dead":
            embed = discord.Embed(
                title=f":skull: {combatant.name} has died!",
                description=f"**Roll:** [{roll.kept_dice[0]}] = **{roll.total}** (Failure)\n\nThree failures... {combatant.name} is gone.\n\n{saves_display}",
                color=discord.Color.dark_grey(),
            )
        elif result == "success":
            embed = discord.Embed(
                title=f":white_check_mark: Death Save Success!",
                description=f"**Roll:** [{roll.kept_dice[0]}] = **{roll.total}** vs DC 10\n\n{saves_display}",
                color=discord.Color.green(),
            )
        else:  # failure
            embed = discord.Embed(
                title=f":x: Death Save Failure!",
                description=f"**Roll:** [{roll.kept_dice[0]}] = **{roll.total}** vs DC 10\n\n{saves_display}",
                color=discord.Color.red(),
            )

        await ctx.respond(embed=embed)

        # Check if combat should end after death
        if result == "dead" and manager.combat.is_combat_over():
            players_alive = any(
                c.is_player and c.hp_current > 0
                for c in manager.combat.combatants
            )
            end_embed = build_combat_end_embed(manager.combat, victory=players_alive)
            await ctx.send(embed=end_embed)
            manager.end_combat()
            clear_combat_for_channel(ctx.channel_id)

    @combat.command(name="stabilize", description="Stabilize a dying creature")
    async def combat_stabilize(
        self,
        ctx: discord.ApplicationContext,
        target: discord.Option(
            str,
            "Target name",
            required=True,
        ),
    ):
        """Stabilize a dying creature (e.g., via Medicine check or spare the dying)."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat.",
                ephemeral=True,
            )
            return

        combatant = manager.get_combatant_by_name(target)
        if not combatant:
            await ctx.respond(
                f"No combatant found matching '{target}'.",
                ephemeral=True,
            )
            return

        if not combatant.is_dying:
            if combatant.hp_current > 0:
                await ctx.respond(
                    f"{combatant.name} is conscious!",
                    ephemeral=True,
                )
            elif combatant.is_stable:
                await ctx.respond(
                    f"{combatant.name} is already stable.",
                    ephemeral=True,
                )
            else:
                await ctx.respond(
                    f"{combatant.name} cannot be stabilized.",
                    ephemeral=True,
                )
            return

        manager.stabilize_combatant(combatant.id)
        await ctx.respond(
            f":medical_symbol: **{combatant.name}** has been stabilized! "
            f"They are unconscious but no longer dying."
        )

    @combat.command(name="temp_hp", description="Add temporary HP to a combatant")
    async def combat_temp_hp(
        self,
        ctx: discord.ApplicationContext,
        target: discord.Option(
            str,
            "Target name",
            required=True,
        ),
        amount: discord.Option(
            int,
            "Temp HP amount",
            required=True,
            min_value=1,
        ),
    ):
        """Add temporary HP to a combatant."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat.",
                ephemeral=True,
            )
            return

        combatant = manager.get_combatant_by_name(target)
        if not combatant:
            await ctx.respond(
                f"No combatant found matching '{target}'.",
                ephemeral=True,
            )
            return

        new_temp = manager.add_temp_hp(combatant.id, amount)

        await ctx.respond(
            f":shield: **{combatant.name}** gains temporary HP! (Temp HP: {new_temp})"
        )

    @combat.command(name="set_hp", description="Set a combatant's HP directly")
    async def combat_set_hp(
        self,
        ctx: discord.ApplicationContext,
        target: discord.Option(
            str,
            "Target name",
            required=True,
        ),
        hp: discord.Option(
            int,
            "New HP value",
            required=True,
            min_value=0,
        ),
    ):
        """Set a combatant's HP to a specific value."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat.",
                ephemeral=True,
            )
            return

        combatant = manager.get_combatant_by_name(target)
        if not combatant:
            await ctx.respond(
                f"No combatant found matching '{target}'.",
                ephemeral=True,
            )
            return

        old_hp = combatant.hp_current
        combatant.hp_current = min(hp, combatant.hp_max)

        await ctx.respond(
            f":wrench: **{combatant.name}**'s HP set to **{combatant.hp_current}** "
            f"(was {old_hp})"
        )

    @combat.command(name="set_init", description="Set a combatant's initiative")
    async def combat_set_initiative(
        self,
        ctx: discord.ApplicationContext,
        target: discord.Option(
            str,
            "Target name",
            required=True,
        ),
        initiative: discord.Option(
            int,
            "Initiative value",
            required=True,
        ),
    ):
        """Set a combatant's initiative value."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat.",
                ephemeral=True,
            )
            return

        combatant = manager.get_combatant_by_name(target)
        if not combatant:
            await ctx.respond(
                f"No combatant found matching '{target}'.",
                ephemeral=True,
            )
            return

        manager.set_initiative(combatant.id, initiative)

        await ctx.respond(
            f":zap: **{combatant.name}**'s initiative set to **{initiative}**"
        )

    @combat.command(name="cover", description="Set or show cover for a combatant")
    async def combat_cover(
        self,
        ctx: discord.ApplicationContext,
        target: discord.Option(
            str,
            "Combatant name",
            required=True,
        ),
        cover_type: discord.Option(
            str,
            "Type of cover",
            required=False,
            choices=["none", "half", "three-quarters", "full"],
            default=None,
        ),
    ):
        """Set or show the cover level for a combatant."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat.",
                ephemeral=True,
            )
            return

        combatant = manager.get_combatant_by_name(target)
        if not combatant:
            await ctx.respond(
                f"No combatant found matching '{target}'.",
                ephemeral=True,
            )
            return

        # If no cover type specified, show current cover
        if cover_type is None:
            current_cover = manager.get_cover(combatant.id)
            ac_bonus = manager.get_cover_ac_bonus(combatant.id)
            effective_ac = manager.get_effective_ac(combatant.id)

            cover_descriptions = {
                "none": "No cover",
                "half": "Half cover (+2 AC, +2 DEX saves)",
                "three-quarters": "Three-quarters cover (+5 AC, +5 DEX saves)",
                "full": "Full cover (can't be directly targeted)",
            }

            await ctx.respond(
                f":shield: **{combatant.name}** - {cover_descriptions.get(current_cover, 'No cover')}\n"
                f"Effective AC: **{effective_ac}** (base {combatant.armor_class} + {ac_bonus} cover)"
            )
            return

        # Set new cover
        success = manager.set_cover(combatant.id, cover_type)
        if not success:
            await ctx.respond(
                "Failed to set cover.",
                ephemeral=True,
            )
            return

        cover_emojis = {
            "none": ":person_standing:",
            "half": ":wood:",
            "three-quarters": ":brick:",
            "full": ":bricks:",
        }

        cover_descriptions = {
            "none": "out in the open",
            "half": "behind half cover (+2 AC)",
            "three-quarters": "behind three-quarters cover (+5 AC)",
            "full": "behind full cover (can't be directly targeted)",
        }

        await ctx.respond(
            f"{cover_emojis.get(cover_type, ':shield:')} **{combatant.name}** is now {cover_descriptions.get(cover_type, 'in cover')}"
        )

    @combat.command(name="opportunity", description="Make an opportunity attack")
    async def combat_opportunity(
        self,
        ctx: discord.ApplicationContext,
        target: discord.Option(
            str,
            "Name of the target leaving your reach",
            required=True,
        ),
        attack_bonus: discord.Option(
            int,
            "Your attack bonus (e.g., +5)",
            required=False,
            default=0,
        ),
    ):
        """Make an opportunity attack against a creature leaving your reach."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat.",
                ephemeral=True,
            )
            return

        # Find the combatant for this user
        user_combatant = None
        for combatant in manager.combat.combatants:
            if combatant.is_player and combatant.character_id:
                from ...game.session import get_session_manager
                session = get_session_manager().get_session(ctx.channel_id)
                if session:
                    player = session.get_player(ctx.author.id)
                    if player and player.character and player.character.name == combatant.name:
                        user_combatant = combatant
                        break

        if not user_combatant:
            await ctx.respond(
                "You don't have a combatant in this combat.",
                ephemeral=True,
            )
            return

        # Find target
        target_combatant = manager.get_combatant_by_name(target)
        if not target_combatant:
            await ctx.respond(
                f"No combatant found matching '{target}'.",
                ephemeral=True,
            )
            return

        # Check if can make opportunity attack
        can_attack, reason = manager.can_make_opportunity_attack(user_combatant.id)
        if not can_attack:
            await ctx.respond(
                f"Cannot make opportunity attack: {reason}",
                ephemeral=True,
            )
            return

        # Make the attack
        result = manager.make_opportunity_attack(
            user_combatant.id,
            target_combatant.id,
            attack_bonus=attack_bonus,
        )

        if not result.get("success"):
            await ctx.respond(
                f"Opportunity attack failed: {result.get('error', 'Unknown error')}",
                ephemeral=True,
            )
            return

        # Build response embed
        embed = discord.Embed(
            title=":zap: Opportunity Attack!",
            description=f"**{user_combatant.name}** strikes at **{target_combatant.name}** as they try to leave!",
            color=discord.Color.red() if result["hit"] else discord.Color.dark_grey(),
        )

        # Attack roll info
        roll_text = f"{result['attack_roll']}"
        if result["natural"] == 20:
            roll_text = f"**NAT 20!** {roll_text}"
        elif result["natural"] == 1:
            roll_text = f"**NAT 1!** {roll_text}"

        embed.add_field(
            name="Attack Roll",
            value=f"{roll_text} vs AC {result['target_ac']}",
            inline=True,
        )

        embed.add_field(
            name="Result",
            value=":boom: **HIT!**" if result["hit"] else ":shield: **MISS!**",
            inline=True,
        )

        if result["hit"]:
            embed.set_footer(text="Roll damage and apply with /combat damage")

        await ctx.respond(embed=embed)

    @combat.command(name="ready", description="Ready an action for a trigger")
    async def combat_ready(
        self,
        ctx: discord.ApplicationContext,
        action: discord.Option(
            str,
            "The action you want to ready (e.g., 'Attack the first enemy that moves')",
            required=True,
        ),
        trigger: discord.Option(
            str,
            "What triggers the action (e.g., 'An enemy comes within reach')",
            required=True,
        ),
    ):
        """Ready an action to be triggered later."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat.",
                ephemeral=True,
            )
            return

        # Find the combatant for this user
        user_combatant = None
        for combatant in manager.combat.combatants:
            if combatant.is_player and combatant.character_id:
                # Look up character to check user
                from ...game.session import get_session_manager
                session = get_session_manager().get_session(ctx.channel_id)
                if session:
                    player = session.get_player(ctx.author.id)
                    if player and player.character and player.character.name == combatant.name:
                        user_combatant = combatant
                        break

        if not user_combatant:
            await ctx.respond(
                "You don't have a combatant in this combat.",
                ephemeral=True,
            )
            return

        # Check if they have their action available
        if user_combatant.turns.action_used:
            await ctx.respond(
                "You've already used your action this turn.",
                ephemeral=True,
            )
            return

        # Mark action as used (readied)
        manager.use_action(user_combatant.id)

        # Store the readied action (simplified - in a full implementation this would go to database)
        if not hasattr(manager, "_readied_actions"):
            manager._readied_actions = {}
        manager._readied_actions[user_combatant.id] = {
            "action": action,
            "trigger": trigger,
            "combatant_name": user_combatant.name,
        }

        embed = discord.Embed(
            title=":hourglass: Action Readied",
            color=discord.Color.gold(),
        )
        embed.add_field(
            name="Combatant",
            value=user_combatant.name,
            inline=True,
        )
        embed.add_field(
            name="Readied Action",
            value=action,
            inline=False,
        )
        embed.add_field(
            name="Trigger",
            value=f"*{trigger}*",
            inline=False,
        )
        embed.set_footer(text="Use /combat trigger to execute the readied action when the trigger occurs.")

        await ctx.respond(embed=embed)

    @combat.command(name="trigger", description="Trigger a readied action")
    async def combat_trigger(
        self,
        ctx: discord.ApplicationContext,
        target: discord.Option(
            str,
            "Target of the readied action (if applicable)",
            required=False,
        ),
    ):
        """Execute a previously readied action."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat.",
                ephemeral=True,
            )
            return

        # Find the combatant for this user
        user_combatant = None
        for combatant in manager.combat.combatants:
            if combatant.is_player and combatant.character_id:
                from ...game.session import get_session_manager
                session = get_session_manager().get_session(ctx.channel_id)
                if session:
                    player = session.get_player(ctx.author.id)
                    if player and player.character and player.character.name == combatant.name:
                        user_combatant = combatant
                        break

        if not user_combatant:
            await ctx.respond(
                "You don't have a combatant in this combat.",
                ephemeral=True,
            )
            return

        # Check for readied action
        if not hasattr(manager, "_readied_actions"):
            manager._readied_actions = {}

        if user_combatant.id not in manager._readied_actions:
            await ctx.respond(
                "You don't have a readied action.",
                ephemeral=True,
            )
            return

        readied = manager._readied_actions.pop(user_combatant.id)

        embed = discord.Embed(
            title=":zap: Readied Action Triggered!",
            description=f"**{readied['combatant_name']}** executes their readied action!",
            color=discord.Color.orange(),
        )
        embed.add_field(
            name="Action",
            value=readied["action"],
            inline=False,
        )
        if target:
            embed.add_field(
                name="Target",
                value=target,
                inline=True,
            )
        embed.set_footer(text="The DM will resolve the readied action based on the current situation.")

        await ctx.respond(embed=embed)

    @combat.command(name="turn", description="Show structured turn UI for current combatant")
    async def combat_turn(
        self,
        ctx: discord.ApplicationContext,
    ):
        """Show the structured action menu for the current combatant's turn."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat.",
                ephemeral=True,
            )
            return

        if manager.combat.state not in (CombatState.AWAITING_ACTION, CombatState.ACTIVE):
            await ctx.respond(
                "Combat is not in progress.",
                ephemeral=True,
            )
            return

        current = manager.combat.get_current_combatant()
        if not current:
            await ctx.respond(
                "No current combatant.",
                ephemeral=True,
            )
            return

        # Get or create coordinator
        coordinator = get_coordinator(manager)

        if not current.is_player:
            # Run NPC turn automatically
            await ctx.defer()
            results = await coordinator.run_npc_turn(current)

            # Narrate the results
            for result in results:
                result_embed = ActionResultEmbed.build(result)
                await ctx.send(embed=result_embed)

                # Get narrator description
                try:
                    narrative = await coordinator.narrate_result(result)
                    if narrative:
                        await ctx.send(f"*{narrative}*")
                except Exception as e:
                    logger.warning("narration_failed", error=str(e))

            # Announce next turn
            next_combatant = manager.combat.get_current_combatant()
            if next_combatant:
                await ctx.send(f":arrow_right: **{next_combatant.name}**, it's your turn!")

                # If next is also NPC, prompt to run /combat turn again
                if not next_combatant.is_player:
                    await ctx.send("*(Run `/combat turn` to process this NPC's turn)*")
        else:
            # Player turn - show action UI
            turn_ctx = await coordinator.start_turn(current)

            async def on_action_complete(result):
                result_embed = ActionResultEmbed.build(result)
                await ctx.send(embed=result_embed)

                # Narrate if successful
                if result.success:
                    try:
                        narrative = await coordinator.narrate_result(result)
                        if narrative:
                            await ctx.send(f"*{narrative}*")
                    except Exception as e:
                        logger.warning("narration_failed", error=str(e))

                # Check combat end
                if manager.combat.is_combat_over():
                    players_alive = any(
                        c.is_player and c.hp_current > 0
                        for c in manager.combat.combatants
                    )
                    await self._sync_player_characters(manager)
                    end_embed = build_combat_end_embed(manager.combat, victory=players_alive)
                    await ctx.send(embed=end_embed)
                    manager.end_combat()
                    clear_combat_for_channel(ctx.channel_id)
                    clear_coordinator(ctx.channel_id)

            async def on_turn_end():
                await self._sync_player_characters(manager)
                end_result = await coordinator.end_turn(current)

                # Announce next combatant
                next_msg = f":arrow_right: **{end_result.next_combatant_name}**'s turn!"
                if end_result.round_advanced:
                    next_msg = f"**Round {end_result.new_round}**\n{next_msg}"
                await ctx.send(next_msg)

                if not end_result.next_is_player:
                    await ctx.send("*(Run `/combat turn` to process this NPC's turn)*")

            view = CombatActionView(
                coordinator=coordinator,
                turn_context=turn_ctx,
                on_action_complete=on_action_complete,
                on_turn_end=on_turn_end,
            )

            await ctx.respond(embed=view.get_embed(), view=view)

    @combat.command(name="npc_turn", description="Run all pending NPC turns automatically")
    async def combat_npc_turns(
        self,
        ctx: discord.ApplicationContext,
    ):
        """Run all consecutive NPC turns automatically."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat.",
                ephemeral=True,
            )
            return

        if manager.combat.state not in (CombatState.AWAITING_ACTION, CombatState.ACTIVE):
            await ctx.respond(
                "Combat is not in progress.",
                ephemeral=True,
            )
            return

        await ctx.defer()

        coordinator = get_coordinator(manager)
        turns_run = 0
        max_turns = 10  # Safety limit

        while turns_run < max_turns:
            current = manager.combat.get_current_combatant()
            if not current:
                break

            if current.is_player:
                # Stop at player turn
                break

            turns_run += 1
            await ctx.send(f":skull: **{current.name}**'s turn...")

            # Run NPC turn
            results = await coordinator.run_npc_turn(current)

            for result in results:
                result_embed = ActionResultEmbed.build(result)
                await ctx.send(embed=result_embed)

                try:
                    narrative = await coordinator.narrate_result(result)
                    if narrative:
                        await ctx.send(f"*{narrative}*")
                except Exception as e:
                    logger.warning("narration_failed", error=str(e))

            # Check combat end
            if manager.combat.is_combat_over():
                players_alive = any(
                    c.is_player and c.hp_current > 0
                    for c in manager.combat.combatants
                )
                await self._sync_player_characters(manager)
                end_embed = build_combat_end_embed(manager.combat, victory=players_alive)
                await ctx.send(embed=end_embed)
                manager.end_combat()
                clear_combat_for_channel(ctx.channel_id)
                clear_coordinator(ctx.channel_id)
                return

        # Announce current combatant
        current = manager.combat.get_current_combatant()
        if current:
            tracker_embed = build_combat_tracker_embed(manager.combat)
            if current.is_player:
                await ctx.send(
                    f":arrow_right: **{current.name}**, it's your turn! Use `/combat turn` for the action menu.",
                    embed=tracker_embed,
                )
            else:
                await ctx.send(
                    f"Ran {turns_run} NPC turns. More NPCs pending.",
                    embed=tracker_embed,
                )

    @combat.command(name="end", description="End the current combat")
    async def combat_end(
        self,
        ctx: discord.ApplicationContext,
    ):
        """End the current combat encounter."""
        manager = get_combat_for_channel(ctx.channel_id)
        if not manager:
            await ctx.respond(
                "No active combat to end.",
                ephemeral=True,
            )
            return

        # Sync all player characters before ending
        await self._sync_player_characters(manager)

        manager.end_combat()

        embed = discord.Embed(
            title=":stop_button: Combat Ended",
            description=f"The encounter has ended after {manager.combat.current_round} round(s).",
            color=discord.Color.greyple(),
        )

        # Show final status
        survivors = [
            c for c in manager.combat.combatants
            if c.hp_current > 0
        ]
        if survivors:
            survivor_text = "\n".join(
                f"- {c.name}: {c.hp_current}/{c.hp_max} HP"
                for c in survivors
            )
            embed.add_field(
                name="Final Status",
                value=survivor_text,
                inline=False,
            )

        clear_combat_for_channel(ctx.channel_id)
        clear_coordinator(ctx.channel_id)
        await ctx.respond(embed=embed)


def setup(bot: commands.Bot):
    """Load the cog."""
    bot.add_cog(CombatCog(bot))
