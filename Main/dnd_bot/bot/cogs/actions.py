"""Dice rolling and action commands."""

import discord
from discord.ext import commands
import structlog

from ...models import AbilityScore, Skill, SKILL_ABILITIES
from ...data.repositories.character_repo import get_character_repo
from ...game.mechanics.dice import get_roller, DiceRoll

logger = structlog.get_logger()


def format_roll_result(roll: DiceRoll) -> str:
    """Format a dice roll result for display."""
    parts = []

    # Show the roll type if advantage/disadvantage
    if roll.roll_type == "advantage":
        parts.append(f"**Advantage**: [{roll.advantage_rolls[0]}, {roll.advantage_rolls[1]}] → {roll.kept_dice[0]}")
    elif roll.roll_type == "disadvantage":
        parts.append(f"**Disadvantage**: [{roll.disadvantage_rolls[0]}, {roll.disadvantage_rolls[1]}] → {roll.kept_dice[0]}")
    elif roll.dropped_dice:
        kept = ", ".join(str(d) for d in roll.kept_dice)
        dropped = ", ".join(f"~~{d}~~" for d in roll.dropped_dice)
        parts.append(f"[{kept}, {dropped}]")
    else:
        parts.append(f"[{', '.join(str(d) for d in roll.kept_dice)}]")

    # Add modifier
    if roll.modifier > 0:
        parts.append(f"+ {roll.modifier}")
    elif roll.modifier < 0:
        parts.append(f"- {abs(roll.modifier)}")

    parts.append(f"= **{roll.total}**")

    result = " ".join(parts)

    # Add critical indicators
    if roll.natural_20:
        result += " :star2: **NATURAL 20!**"
    elif roll.natural_1:
        result += " :skull: **NATURAL 1!**"

    return result


def build_roll_embed(roll: DiceRoll, title: str, color: discord.Color) -> discord.Embed:
    """Build an embed for a dice roll."""
    embed = discord.Embed(title=title, color=color)

    embed.add_field(
        name="Roll",
        value=format_roll_result(roll),
        inline=False,
    )

    if roll.reason:
        embed.set_footer(text=roll.reason)

    return embed


class ActionsCog(commands.Cog):
    """Dice rolling and action commands."""

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.roller = get_roller()

    @discord.slash_command(name="roll", description="Roll dice using D&D notation")
    async def roll_dice(
        self,
        ctx: discord.ApplicationContext,
        notation: discord.Option(
            str,
            "Dice notation (e.g., 1d20+5, 2d6, 4d6kh3)",
            required=True,
        ),
        advantage: discord.Option(
            bool,
            "Roll with advantage (2d20, take higher)",
            required=False,
            default=False,
        ),
        disadvantage: discord.Option(
            bool,
            "Roll with disadvantage (2d20, take lower)",
            required=False,
            default=False,
        ),
        reason: discord.Option(
            str,
            "Reason for the roll",
            required=False,
            default="",
        ),
    ):
        """Roll dice using standard D&D notation."""
        try:
            # Cancel out advantage/disadvantage if both are set
            if advantage and disadvantage:
                advantage = False
                disadvantage = False

            roll = self.roller.roll(
                notation=notation,
                advantage=advantage,
                disadvantage=disadvantage,
                reason=reason,
            )

            # Determine color
            if roll.natural_20:
                color = discord.Color.gold()
            elif roll.natural_1:
                color = discord.Color.dark_red()
            else:
                color = discord.Color.blue()

            embed = build_roll_embed(roll, f":game_die: {notation}", color)

            await ctx.respond(embed=embed)

            logger.info(
                "dice_rolled",
                user=ctx.author.id,
                notation=notation,
                total=roll.total,
            )

        except ValueError as e:
            await ctx.respond(f"Invalid dice notation: {e}", ephemeral=True)

    @discord.slash_command(name="check", description="Make an ability or skill check")
    async def ability_check(
        self,
        ctx: discord.ApplicationContext,
        skill: discord.Option(
            str,
            "Skill to check",
            required=True,
            choices=[
                discord.OptionChoice(s.value.replace("-", " ").title(), s.value)
                for s in Skill
            ],
        ),
        dc: discord.Option(
            int,
            "Difficulty Class (optional)",
            required=False,
            min_value=1,
            max_value=30,
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
        campaign_id: discord.Option(
            str,
            "Campaign ID",
            required=False,
            default="default",
        ),
    ):
        """Make a skill check using your character's modifiers."""
        # Get character
        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)

        if not character:
            await ctx.respond(
                "You don't have a character. Use `/character create` first.",
                ephemeral=True,
            )
            return

        # Cancel out advantage/disadvantage
        if advantage and disadvantage:
            advantage = False
            disadvantage = False

        # Get modifier
        skill_enum = Skill(skill)
        modifier = character.get_skill_modifier(skill_enum)

        # Roll
        roll = self.roller.roll_check(
            modifier=modifier,
            advantage=advantage,
            disadvantage=disadvantage,
        )
        roll.reason = f"{skill.replace('-', ' ').title()} check"

        # Determine result
        skill_name = skill.replace("-", " ").title()
        title = f":mag: {character.name}'s {skill_name} Check"

        if roll.natural_20:
            color = discord.Color.gold()
        elif roll.natural_1:
            color = discord.Color.dark_red()
        else:
            color = discord.Color.blue()

        embed = build_roll_embed(roll, title, color)

        # Add DC result if provided
        if dc is not None:
            success = roll.total >= dc
            result_text = ":white_check_mark: **Success!**" if success else ":x: **Failure**"
            embed.add_field(
                name=f"vs DC {dc}",
                value=result_text,
                inline=True,
            )

            if success:
                color = discord.Color.green()
            else:
                color = discord.Color.red()
            embed.color = color

        await ctx.respond(embed=embed)

    @discord.slash_command(name="save", description="Make a saving throw")
    async def saving_throw(
        self,
        ctx: discord.ApplicationContext,
        ability: discord.Option(
            str,
            "Ability for the saving throw",
            required=True,
            choices=[
                discord.OptionChoice("Strength", "str"),
                discord.OptionChoice("Dexterity", "dex"),
                discord.OptionChoice("Constitution", "con"),
                discord.OptionChoice("Intelligence", "int"),
                discord.OptionChoice("Wisdom", "wis"),
                discord.OptionChoice("Charisma", "cha"),
            ],
        ),
        dc: discord.Option(
            int,
            "Difficulty Class (optional)",
            required=False,
            min_value=1,
            max_value=30,
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
        campaign_id: discord.Option(
            str,
            "Campaign ID",
            required=False,
            default="default",
        ),
    ):
        """Make a saving throw using your character's modifiers."""
        # Get character
        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)

        if not character:
            await ctx.respond(
                "You don't have a character. Use `/character create` first.",
                ephemeral=True,
            )
            return

        # Cancel out advantage/disadvantage
        if advantage and disadvantage:
            advantage = False
            disadvantage = False

        # Get modifier
        ability_map = {
            "str": AbilityScore.STRENGTH,
            "dex": AbilityScore.DEXTERITY,
            "con": AbilityScore.CONSTITUTION,
            "int": AbilityScore.INTELLIGENCE,
            "wis": AbilityScore.WISDOM,
            "cha": AbilityScore.CHARISMA,
        }
        ability_enum = ability_map[ability]
        modifier = character.get_save_modifier(ability_enum)

        # Roll
        roll = self.roller.roll_save(
            modifier=modifier,
            advantage=advantage,
            disadvantage=disadvantage,
        )
        roll.reason = f"{ability_enum.name.title()} saving throw"

        # Determine result
        ability_name = ability_enum.name.title()
        title = f":shield: {character.name}'s {ability_name} Save"

        if roll.natural_20:
            color = discord.Color.gold()
        elif roll.natural_1:
            color = discord.Color.dark_red()
        else:
            color = discord.Color.blue()

        embed = build_roll_embed(roll, title, color)

        # Add DC result if provided
        if dc is not None:
            success = roll.total >= dc
            result_text = ":white_check_mark: **Success!**" if success else ":x: **Failure**"
            embed.add_field(
                name=f"vs DC {dc}",
                value=result_text,
                inline=True,
            )

            if success:
                color = discord.Color.green()
            else:
                color = discord.Color.red()
            embed.color = color

        await ctx.respond(embed=embed)

    @discord.slash_command(name="attack", description="Make an attack roll")
    async def attack_roll(
        self,
        ctx: discord.ApplicationContext,
        target_ac: discord.Option(
            int,
            "Target's Armor Class",
            required=True,
            min_value=1,
            max_value=30,
        ),
        modifier: discord.Option(
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
        damage_dice: discord.Option(
            str,
            "Damage dice if hit (e.g., 1d8+3)",
            required=False,
        ),
    ):
        """Make an attack roll against an AC."""
        # Cancel out advantage/disadvantage
        if advantage and disadvantage:
            advantage = False
            disadvantage = False

        # Roll attack
        attack_roll = self.roller.roll_attack(
            modifier=modifier,
            advantage=advantage,
            disadvantage=disadvantage,
        )

        # Determine hit
        is_critical = attack_roll.natural_20
        is_fumble = attack_roll.natural_1
        is_hit = is_critical or (not is_fumble and attack_roll.total >= target_ac)

        # Build embed
        if is_critical:
            title = ":boom: Critical Hit!"
            color = discord.Color.gold()
        elif is_fumble:
            title = ":skull: Critical Miss!"
            color = discord.Color.dark_red()
        elif is_hit:
            title = ":crossed_swords: Hit!"
            color = discord.Color.green()
        else:
            title = ":shield: Miss"
            color = discord.Color.red()

        embed = discord.Embed(title=title, color=color)
        embed.add_field(
            name="Attack Roll",
            value=f"{format_roll_result(attack_roll)} vs AC {target_ac}",
            inline=False,
        )

        # Roll damage if hit and damage dice provided
        if is_hit and damage_dice:
            damage_roll = self.roller.roll_damage(damage_dice, critical=is_critical)
            embed.add_field(
                name="Damage" + (" (Critical!)" if is_critical else ""),
                value=format_roll_result(damage_roll),
                inline=False,
            )

        await ctx.respond(embed=embed)

    @discord.slash_command(name="initiative", description="Roll initiative")
    async def roll_initiative(
        self,
        ctx: discord.ApplicationContext,
        modifier: discord.Option(
            int,
            "Initiative modifier (uses DEX mod if not specified)",
            required=False,
        ),
        campaign_id: discord.Option(
            str,
            "Campaign ID",
            required=False,
            default="default",
        ),
    ):
        """Roll initiative."""
        # Try to get character for modifier
        if modifier is None:
            repo = await get_character_repo()
            character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)
            if character:
                modifier = character.initiative_bonus
            else:
                modifier = 0

        roll = self.roller.roll_initiative(modifier=modifier)

        embed = build_roll_embed(
            roll,
            f":zap: {ctx.author.display_name}'s Initiative",
            discord.Color.purple(),
        )

        await ctx.respond(embed=embed)


def setup(bot: commands.Bot):
    """Load the cog."""
    bot.add_cog(ActionsCog(bot))
