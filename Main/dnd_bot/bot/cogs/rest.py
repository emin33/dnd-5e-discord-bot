"""Rest commands cog."""

import discord
from discord.ext import commands
import structlog

from ...data.repositories import get_character_repo
from ...game.mechanics.rest import get_rest_manager

logger = structlog.get_logger()


class RestCog(commands.Cog):
    """Rest and resource recovery commands."""

    rest = discord.SlashCommandGroup(
        "rest",
        "Rest and recover resources",
    )

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.rest_manager = get_rest_manager()

    @rest.command(name="short", description="Take a short rest (1 hour)")
    async def short_rest(
        self,
        ctx: discord.ApplicationContext,
        hit_dice: discord.Option(
            int,
            "Number of hit dice to spend for healing",
            required=False,
            default=0,
            min_value=0,
            max_value=20,
        ),
        campaign_id: discord.Option(
            str,
            "Campaign ID",
            required=False,
            default="default",
        ),
    ):
        """Take a short rest to recover resources."""
        await ctx.defer()
        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)

        if not character:
            await ctx.respond(
                "You don't have a character. Use `/character create` first.",
                ephemeral=True,
            )
            return

        # Check if can rest
        can_rest, reason = self.rest_manager.can_short_rest(character)
        if not can_rest and hit_dice == 0:
            await ctx.respond(
                f"Cannot take a short rest: {reason}",
                ephemeral=True,
            )
            return

        # Take rest
        result = self.rest_manager.short_rest(character, hit_dice_to_spend=hit_dice)
        await repo.update(character)

        # Build response
        embed = discord.Embed(
            title=f":zzz: {character.name} takes a Short Rest",
            description="You spend an hour resting and recovering.",
            color=discord.Color.green(),
        )

        # Hit dice spending
        if result.hit_dice_spent:
            dice_text = []
            for hd in result.hit_dice_spent:
                roll = hd.roll
                dice_str = f"[{roll.kept_dice[0]}]"
                if roll.modifier != 0:
                    mod_str = f" + {roll.modifier}" if roll.modifier > 0 else f" - {abs(roll.modifier)}"
                    dice_str += mod_str
                dice_str += f" = {hd.healing} HP"
                dice_text.append(dice_str)

            embed.add_field(
                name=f"Hit Dice ({len(result.hit_dice_spent)} spent)",
                value="\n".join(dice_text),
                inline=False,
            )

            embed.add_field(
                name="Healing",
                value=f"**{result.total_healing}** HP restored ({result.hp_before} → {result.hp_after})",
                inline=True,
            )
        else:
            embed.add_field(
                name="Hit Dice",
                value="No hit dice spent",
                inline=False,
            )

        # Remaining hit dice
        embed.add_field(
            name="Hit Dice Remaining",
            value=f"{character.hit_dice.remaining}/{character.hit_dice.total}",
            inline=True,
        )

        # Features recovered
        if result.features_recovered:
            embed.add_field(
                name="Recovered",
                value="\n".join(f"- {f}" for f in result.features_recovered),
                inline=False,
            )

        embed.set_footer(text=f"HP: {character.hp.current}/{character.hp.maximum}")

        await ctx.respond(embed=embed)

    @rest.command(name="long", description="Take a long rest (8 hours)")
    async def long_rest(
        self,
        ctx: discord.ApplicationContext,
        campaign_id: discord.Option(
            str,
            "Campaign ID",
            required=False,
            default="default",
        ),
    ):
        """Take a long rest to fully recover."""
        await ctx.defer()
        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)

        if not character:
            await ctx.respond(
                "You don't have a character. Use `/character create` first.",
                ephemeral=True,
            )
            return

        # Take rest
        result = self.rest_manager.long_rest(character)
        await repo.update(character)

        # Build response
        embed = discord.Embed(
            title=f":crescent_moon: {character.name} takes a Long Rest",
            description="After 8 hours of rest, you feel fully refreshed!",
            color=discord.Color.blue(),
        )

        # HP restoration
        if result.hp_restored > 0:
            embed.add_field(
                name="HP Restored",
                value=f"**{result.hp_restored}** HP restored (Full: {result.hp_max})",
                inline=True,
            )
        else:
            embed.add_field(
                name="HP",
                value=f"Already at full ({result.hp_max})",
                inline=True,
            )

        # Spell slots
        if result.spell_slots_restored:
            slots_text = []
            for level, count in sorted(result.spell_slots_restored.items()):
                slots_text.append(f"Level {level}: +{count}")
            embed.add_field(
                name="Spell Slots Restored",
                value="\n".join(slots_text),
                inline=True,
            )

        # Hit dice
        if result.hit_dice_restored > 0:
            embed.add_field(
                name="Hit Dice Recovered",
                value=f"+{result.hit_dice_restored} ({character.hit_dice.remaining}/{character.hit_dice.total})",
                inline=True,
            )

        # Exhaustion
        if result.exhaustion_removed > 0:
            new_level = character.get_exhaustion_level()
            if new_level > 0:
                embed.add_field(
                    name="Exhaustion",
                    value=f"1 level removed (now at level {new_level})",
                    inline=True,
                )
            else:
                embed.add_field(
                    name="Exhaustion",
                    value="Exhaustion removed!",
                    inline=True,
                )

        # Features
        embed.add_field(
            name="Recovered",
            value="All class features and abilities restored",
            inline=False,
        )

        await ctx.respond(embed=embed)

    @rest.command(name="status", description="View your rest-related resources")
    async def rest_status(
        self,
        ctx: discord.ApplicationContext,
        campaign_id: discord.Option(
            str,
            "Campaign ID",
            required=False,
            default="default",
        ),
    ):
        """View your current resources and rest status."""
        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)

        if not character:
            await ctx.respond(
                "You don't have a character. Use `/character create` first.",
                ephemeral=True,
            )
            return

        embed = discord.Embed(
            title=f":bar_chart: {character.name}'s Resources",
            color=discord.Color.blue(),
        )

        # HP
        hp_pct = (character.hp.current / character.hp.maximum) * 100
        hp_bar = self._build_bar(hp_pct)
        embed.add_field(
            name="Hit Points",
            value=f"{hp_bar} {character.hp.current}/{character.hp.maximum}",
            inline=False,
        )

        # Temp HP
        if character.hp.temporary > 0:
            embed.add_field(
                name="Temporary HP",
                value=f":shield: {character.hp.temporary}",
                inline=True,
            )

        # Hit Dice
        hd_pct = (character.hit_dice.remaining / character.hit_dice.total) * 100
        hd_bar = self._build_bar(hd_pct)
        embed.add_field(
            name="Hit Dice",
            value=f"{hd_bar} {character.hit_dice.remaining}/{character.hit_dice.total} d{character.hit_dice.die_type}",
            inline=False,
        )

        # Spell Slots (if spellcaster)
        if character.spellcasting_ability:
            slots_text = []
            for level in range(1, 10):
                current, max_slots = character.spell_slots.get_slots(level)
                if max_slots > 0:
                    filled = ":blue_circle:" * current
                    empty = ":black_circle:" * (max_slots - current)
                    slots_text.append(f"**{level}:** {filled}{empty}")

            if slots_text:
                embed.add_field(
                    name="Spell Slots",
                    value="\n".join(slots_text),
                    inline=False,
                )

        # Exhaustion
        exhaustion = character.get_exhaustion_level()
        if exhaustion > 0:
            exhaustion_effects = {
                1: "Disadvantage on ability checks",
                2: "Speed halved",
                3: "Disadvantage on attacks and saves",
                4: "HP maximum halved",
                5: "Speed reduced to 0",
                6: "Death",
            }
            effect = exhaustion_effects.get(exhaustion, "Unknown")
            embed.add_field(
                name=f":tired_face: Exhaustion Level {exhaustion}",
                value=effect,
                inline=False,
            )

        # Concentration
        if character.concentration_spell_id:
            embed.add_field(
                name="Concentrating On",
                value=f":brain: {character.concentration_spell_id.replace('-', ' ').title()}",
                inline=False,
            )

        # Death saves (if relevant)
        if character.hp.current == 0 and not character.death_saves.is_stable:
            ds = character.death_saves
            saves = f"Successes: {':white_check_mark:' * ds.successes}{':black_large_square:' * (3 - ds.successes)}"
            fails = f"Failures: {':x:' * ds.failures}{':black_large_square:' * (3 - ds.failures)}"
            embed.add_field(
                name=":skull: Death Saves",
                value=f"{saves}\n{fails}",
                inline=False,
            )

        await ctx.respond(embed=embed)

    def _build_bar(self, percentage: float, length: int = 10) -> str:
        """Build a visual bar for percentage display."""
        filled = int(percentage / 100 * length)
        empty = length - filled

        if percentage > 75:
            fill_char = ":green_square:"
        elif percentage > 50:
            fill_char = ":yellow_square:"
        elif percentage > 25:
            fill_char = ":orange_square:"
        else:
            fill_char = ":red_square:"

        return fill_char * filled + ":black_large_square:" * empty

    @rest.command(name="exhaustion", description="Add or remove exhaustion")
    async def exhaustion(
        self,
        ctx: discord.ApplicationContext,
        action: discord.Option(
            str,
            "Add or remove exhaustion",
            required=True,
            choices=["add", "remove"],
        ),
        levels: discord.Option(
            int,
            "Number of levels",
            required=False,
            default=1,
            min_value=1,
            max_value=6,
        ),
        campaign_id: discord.Option(
            str,
            "Campaign ID",
            required=False,
            default="default",
        ),
    ):
        """Add or remove exhaustion levels."""
        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)

        if not character:
            await ctx.respond(
                "You don't have a character.",
                ephemeral=True,
            )
            return

        if action == "add":
            new_level = self.rest_manager.add_exhaustion(character, levels)
            await repo.update(character)

            if new_level >= 6:
                await ctx.respond(
                    f":skull: **{character.name}** has gained 6 levels of exhaustion and **dies**!"
                )
            else:
                await ctx.respond(
                    f":tired_face: **{character.name}** gains {levels} level(s) of exhaustion. "
                    f"(Now at level {new_level})"
                )
        else:
            new_level = self.rest_manager.remove_exhaustion(character, levels)
            await repo.update(character)

            if new_level == 0:
                await ctx.respond(
                    f":sparkles: **{character.name}**'s exhaustion is fully removed!"
                )
            else:
                await ctx.respond(
                    f":green_heart: **{character.name}** removes {levels} level(s) of exhaustion. "
                    f"(Now at level {new_level})"
                )


def setup(bot: commands.Bot):
    """Load the cog."""
    bot.add_cog(RestCog(bot))
