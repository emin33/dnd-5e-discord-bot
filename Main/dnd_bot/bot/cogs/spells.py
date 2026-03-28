"""Spellcasting commands cog."""

import discord
from discord.ext import commands
import structlog

from ...models import AbilityScore
from ...data.repositories import get_character_repo
from ...data.srd import get_srd
from ...game.magic import get_spellcasting_manager, SpellType

logger = structlog.get_logger()


def build_spell_info_embed(spell_info) -> discord.Embed:
    """Build an embed showing spell information."""
    # Color based on school
    school_colors = {
        "abjuration": discord.Color.blue(),
        "conjuration": discord.Color.gold(),
        "divination": discord.Color.light_grey(),
        "enchantment": discord.Color.purple(),
        "evocation": discord.Color.red(),
        "illusion": discord.Color.dark_purple(),
        "necromancy": discord.Color.dark_grey(),
        "transmutation": discord.Color.green(),
    }
    color = school_colors.get(spell_info.school.value, discord.Color.blue())

    # Title with level
    if spell_info.level == 0:
        level_text = "Cantrip"
    else:
        level_text = f"Level {spell_info.level}"

    embed = discord.Embed(
        title=f":sparkles: {spell_info.name}",
        description=f"*{level_text} {spell_info.school.value.title()}*",
        color=color,
    )

    # Casting info
    casting_info = (
        f"**Casting Time:** {spell_info.casting_time}\n"
        f"**Range:** {spell_info.range}\n"
        f"**Components:** {', '.join(spell_info.components)}"
    )
    if spell_info.material:
        casting_info += f"\n*({spell_info.material})*"
    casting_info += f"\n**Duration:** {spell_info.duration}"
    if spell_info.concentration:
        casting_info += " (Concentration)"
    if spell_info.ritual:
        casting_info += " (Ritual)"

    embed.add_field(
        name="Casting",
        value=casting_info,
        inline=False,
    )

    # Description (truncate if too long)
    description = spell_info.description
    if len(description) > 1000:
        description = description[:997] + "..."

    embed.add_field(
        name="Description",
        value=description,
        inline=False,
    )

    # Higher level effects
    if spell_info.higher_level:
        higher = spell_info.higher_level
        if len(higher) > 500:
            higher = higher[:497] + "..."
        embed.add_field(
            name="At Higher Levels",
            value=higher,
            inline=False,
        )

    # Combat info
    combat_info = []
    if spell_info.attack_type:
        combat_info.append(f"**Attack:** {spell_info.attack_type.title()} spell attack")
    if spell_info.damage_dice:
        dmg_text = f"**Damage:** {spell_info.damage_dice}"
        if spell_info.damage_type:
            dmg_text += f" {spell_info.damage_type}"
        combat_info.append(dmg_text)
    if spell_info.save_dc_ability:
        combat_info.append(f"**Save:** {spell_info.save_dc_ability.value.upper()}")

    if combat_info:
        embed.add_field(
            name="Combat",
            value="\n".join(combat_info),
            inline=False,
        )

    return embed


def build_spell_cast_embed(result, caster_name: str) -> discord.Embed:
    """Build an embed showing spell cast results."""
    spell = result.spell

    if result.critical:
        title = f":boom: Critical! {caster_name} casts {spell.name}!"
        color = discord.Color.gold()
    elif result.hit:
        title = f":sparkles: {caster_name} casts {spell.name}!"
        color = discord.Color.green()
    elif result.attack_roll and not result.hit:
        title = f":shield: {caster_name}'s {spell.name} misses!"
        color = discord.Color.red()
    else:
        title = f":sparkles: {caster_name} casts {spell.name}!"
        color = discord.Color.purple()

    embed = discord.Embed(title=title, color=color)

    # Slot used
    if result.slot_used > 0:
        embed.description = f"*Using a level {result.slot_used} spell slot*"

    # Attack roll
    if result.attack_roll:
        roll = result.attack_roll
        roll_text = f"[{roll.kept_dice[0]}]"
        if roll.modifier != 0:
            mod_str = f" + {roll.modifier}" if roll.modifier > 0 else f" - {abs(roll.modifier)}"
            roll_text += mod_str
        roll_text += f" = **{roll.total}**"

        if roll.natural_20:
            roll_text += " :star2:"
        elif roll.natural_1:
            roll_text += " :skull:"

        embed.add_field(
            name="Spell Attack",
            value=roll_text,
            inline=True,
        )

    # Save DC info
    if result.save_dc and result.save_ability:
        embed.add_field(
            name="Save",
            value=f"DC {result.save_dc} {result.save_ability.value.upper()}",
            inline=True,
        )

    # Damage
    if result.damage_roll and result.damage_dealt > 0:
        dmg_text = f"[{', '.join(str(d) for d in result.damage_roll.kept_dice)}]"
        if result.damage_roll.modifier != 0:
            mod_str = f" + {result.damage_roll.modifier}" if result.damage_roll.modifier > 0 else f" - {abs(result.damage_roll.modifier)}"
            dmg_text += mod_str
        dmg_text += f" = **{result.damage_dealt}**"
        if result.damage_type:
            dmg_text += f" {result.damage_type}"
        if result.critical:
            dmg_text += " (Critical!)"

        embed.add_field(
            name="Damage",
            value=dmg_text,
            inline=False,
        )

    # Healing
    if result.healing_roll and result.healing_amount > 0:
        heal_text = f"[{', '.join(str(d) for d in result.healing_roll.kept_dice)}]"
        if result.healing_roll.modifier != 0:
            mod_str = f" + {result.healing_roll.modifier}" if result.healing_roll.modifier > 0 else f" - {abs(result.healing_roll.modifier)}"
            heal_text += mod_str
        heal_text += f" = **{result.healing_amount}** HP"

        embed.add_field(
            name="Healing",
            value=heal_text,
            inline=False,
        )

    # Concentration
    if spell.concentration:
        if result.concentration_broken:
            embed.add_field(
                name="Concentration",
                value=":warning: Previous concentration spell ended!",
                inline=False,
            )
        embed.set_footer(text="Requires Concentration")

    return embed


class SpellsCog(commands.Cog):
    """Spellcasting commands."""

    spell = discord.SlashCommandGroup(
        "spell",
        "Spell lookup and casting",
    )

    def __init__(self, bot: commands.Bot):
        self.bot = bot
        self.srd = get_srd()
        self.spell_manager = get_spellcasting_manager()

    @spell.command(name="info", description="Look up a spell's details")
    async def spell_info(
        self,
        ctx: discord.ApplicationContext,
        name: discord.Option(
            str,
            "Spell name or index (e.g., 'fireball', 'cure-wounds')",
            required=True,
        ),
    ):
        """Look up spell information."""
        # Try to find the spell
        spell_info = self.spell_manager.get_spell_info(name.lower().replace(" ", "-"))

        if not spell_info:
            # Try a search
            all_spells = self.srd.list_spells()
            matches = [
                s for s in all_spells
                if name.lower() in s.get("name", "").lower()
            ]

            if not matches:
                await ctx.respond(
                    f"Spell '{name}' not found. Try using the SRD index format (e.g., 'cure-wounds').",
                    ephemeral=True,
                )
                return
            elif len(matches) == 1:
                spell_info = self.spell_manager.get_spell_info(matches[0]["index"])
            else:
                # Show options
                options = "\n".join(f"- `{s['index']}`: {s['name']}" for s in matches[:10])
                await ctx.respond(
                    f"Multiple spells found:\n{options}\n\nUse the exact index to look up a spell.",
                    ephemeral=True,
                )
                return

        embed = build_spell_info_embed(spell_info)
        await ctx.respond(embed=embed)

    @spell.command(name="list", description="List available spells by level or class")
    async def spell_list(
        self,
        ctx: discord.ApplicationContext,
        level: discord.Option(
            int,
            "Spell level (0 for cantrips)",
            required=False,
            min_value=0,
            max_value=9,
        ),
        class_name: discord.Option(
            str,
            "Class to filter by",
            required=False,
            choices=[
                "bard", "cleric", "druid", "paladin",
                "ranger", "sorcerer", "warlock", "wizard",
            ],
        ),
    ):
        """List available spells."""
        all_spells = self.srd.list_spells()

        # Filter by level
        if level is not None:
            all_spells = [s for s in all_spells if s.get("level") == level]

        # Filter by class
        if class_name:
            filtered = []
            for spell in all_spells:
                classes = spell.get("classes", [])
                if any(c.get("index") == class_name for c in classes):
                    filtered.append(spell)
            all_spells = filtered

        if not all_spells:
            await ctx.respond(
                "No spells found matching your criteria.",
                ephemeral=True,
            )
            return

        # Sort by level then name
        all_spells.sort(key=lambda s: (s.get("level", 0), s.get("name", "")))

        # Build response (paginate if needed)
        lines = []
        current_level = -1
        for spell in all_spells[:50]:  # Limit to first 50
            if spell.get("level", 0) != current_level:
                current_level = spell.get("level", 0)
                level_name = "Cantrips" if current_level == 0 else f"Level {current_level}"
                lines.append(f"\n**{level_name}**")
            lines.append(f"- {spell.get('name')}")

        if len(all_spells) > 50:
            lines.append(f"\n*...and {len(all_spells) - 50} more*")

        embed = discord.Embed(
            title=":book: Spell List",
            description="\n".join(lines),
            color=discord.Color.purple(),
        )

        if class_name:
            embed.set_footer(text=f"Showing {class_name.title()} spells")

        await ctx.respond(embed=embed)

    @discord.slash_command(name="cast", description="Cast a spell")
    async def cast_spell(
        self,
        ctx: discord.ApplicationContext,
        spell_name: discord.Option(
            str,
            "Spell to cast",
            required=True,
        ),
        slot_level: discord.Option(
            int,
            "Spell slot level to use (higher for upcasting)",
            required=False,
            min_value=1,
            max_value=9,
        ),
        target_ac: discord.Option(
            int,
            "Target's AC (for attack spells)",
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
        """Cast a spell from your spell list."""
        await ctx.defer()
        # Get character
        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)

        if not character:
            await ctx.respond(
                "You don't have a character. Use `/character create` first.",
                ephemeral=True,
            )
            return

        if not character.spellcasting_ability:
            await ctx.respond(
                f"{character.name} is not a spellcaster!",
                ephemeral=True,
            )
            return

        # Find the spell
        spell_index = spell_name.lower().replace(" ", "-")
        spell_info = self.spell_manager.get_spell_info(spell_index)

        if not spell_info:
            await ctx.respond(
                f"Spell '{spell_name}' not found.",
                ephemeral=True,
            )
            return

        # Determine slot level
        if spell_info.level == 0:
            use_slot = 0
        elif slot_level is None:
            use_slot = spell_info.level
        else:
            use_slot = slot_level

        # Check if can cast
        can_cast, reason = self.spell_manager.can_cast(character, spell_index, use_slot if use_slot > 0 else None)
        if not can_cast:
            await ctx.respond(
                f"Cannot cast {spell_info.name}: {reason}",
                ephemeral=True,
            )
            return

        # Cancel out advantage/disadvantage
        if advantage and disadvantage:
            advantage = False
            disadvantage = False

        # Determine spell type and cast
        spell_type = self.spell_manager.get_spell_type(spell_info)

        if spell_type == SpellType.ATTACK:
            if target_ac is None:
                await ctx.respond(
                    f"{spell_info.name} requires a target AC. Use the `target_ac` option.",
                    ephemeral=True,
                )
                return

            result = self.spell_manager.cast_attack_spell(
                character,
                spell_info,
                use_slot,
                target_ac,
                advantage=advantage,
                disadvantage=disadvantage,
            )

        elif spell_type == SpellType.SAVE:
            result = self.spell_manager.cast_save_spell(
                character,
                spell_info,
                use_slot,
            )

        elif spell_type == SpellType.HEALING:
            result = self.spell_manager.cast_healing_spell(
                character,
                spell_info,
                use_slot,
            )

        else:
            result = self.spell_manager.cast_utility_spell(
                character,
                spell_info,
                use_slot,
            )

        # Expend spell slot
        if use_slot > 0:
            character.spell_slots.expend_slot(use_slot)
            await repo.update(character)

        # Handle concentration
        if spell_info.concentration:
            result.concentration_broken = self.spell_manager.start_concentration(character, spell_info)
            result.concentration_started = True
            await repo.update(character)

        # Build response
        embed = build_spell_cast_embed(result, character.name)

        # Add remaining slots info
        if use_slot > 0:
            current, max_slots = character.spell_slots.get_slots(use_slot)
            embed.add_field(
                name="Spell Slots",
                value=f"Level {use_slot}: {current}/{max_slots} remaining",
                inline=True,
            )

        await ctx.respond(embed=embed)

        logger.info(
            "spell_cast",
            user=ctx.author.id,
            character=character.name,
            spell=spell_info.name,
            slot=use_slot,
        )

    @spell.command(name="slots", description="View your spell slots")
    async def spell_slots(
        self,
        ctx: discord.ApplicationContext,
        campaign_id: discord.Option(
            str,
            "Campaign ID",
            required=False,
            default="default",
        ),
    ):
        """View your current spell slots."""
        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)

        if not character:
            await ctx.respond(
                "You don't have a character.",
                ephemeral=True,
            )
            return

        if not character.spellcasting_ability:
            await ctx.respond(
                f"{character.name} is not a spellcaster!",
                ephemeral=True,
            )
            return

        embed = discord.Embed(
            title=f":sparkles: {character.name}'s Spell Slots",
            color=discord.Color.purple(),
        )

        # Spellcasting info
        embed.add_field(
            name="Spellcasting",
            value=(
                f"**Ability:** {character.spellcasting_ability.value.title()}\n"
                f"**Spell Save DC:** {character.spell_save_dc}\n"
                f"**Spell Attack:** +{character.spell_attack_bonus}"
            ),
            inline=False,
        )

        # Concentration
        if character.concentration_spell_id:
            spell_info = self.spell_manager.get_spell_info(character.concentration_spell_id)
            spell_name = spell_info.name if spell_info else character.concentration_spell_id
            embed.add_field(
                name="Concentrating On",
                value=f":brain: {spell_name}",
                inline=False,
            )

        # Slots
        slots_text = []
        for level in range(1, 10):
            current, max_slots = character.spell_slots.get_slots(level)
            if max_slots > 0:
                filled = ":blue_circle:" * current
                empty = ":black_circle:" * (max_slots - current)
                slots_text.append(f"**Level {level}:** {filled}{empty} ({current}/{max_slots})")

        if slots_text:
            embed.add_field(
                name="Spell Slots",
                value="\n".join(slots_text),
                inline=False,
            )
        else:
            embed.add_field(
                name="Spell Slots",
                value="No spell slots (cantrips only)",
                inline=False,
            )

        await ctx.respond(embed=embed)

    @spell.command(name="concentration", description="Check or break concentration")
    async def spell_concentration(
        self,
        ctx: discord.ApplicationContext,
        action: discord.Option(
            str,
            "What to do",
            required=True,
            choices=["check", "break"],
        ),
        campaign_id: discord.Option(
            str,
            "Campaign ID",
            required=False,
            default="default",
        ),
    ):
        """Check or break concentration on a spell."""
        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)

        if not character:
            await ctx.respond(
                "You don't have a character.",
                ephemeral=True,
            )
            return

        if action == "check":
            if character.concentration_spell_id:
                spell_info = self.spell_manager.get_spell_info(character.concentration_spell_id)
                spell_name = spell_info.name if spell_info else character.concentration_spell_id
                await ctx.respond(
                    f":brain: {character.name} is concentrating on **{spell_name}**."
                )
            else:
                await ctx.respond(
                    f"{character.name} is not concentrating on any spell.",
                    ephemeral=True,
                )

        elif action == "break":
            if not character.concentration_spell_id:
                await ctx.respond(
                    f"{character.name} is not concentrating on any spell.",
                    ephemeral=True,
                )
                return

            spell_info = self.spell_manager.get_spell_info(character.concentration_spell_id)
            spell_name = spell_info.name if spell_info else character.concentration_spell_id

            self.spell_manager.break_concentration(character)
            await repo.update(character)

            await ctx.respond(
                f":x: {character.name}'s concentration on **{spell_name}** ends!"
            )

    @spell.command(name="prepare", description="Prepare a spell for casting")
    async def spell_prepare(
        self,
        ctx: discord.ApplicationContext,
        spell_name: discord.Option(
            str,
            "Spell to prepare",
            required=True,
        ),
        campaign_id: discord.Option(
            str,
            "Campaign ID",
            required=False,
            default="default",
        ),
    ):
        """Add a spell to your prepared spells."""
        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)

        if not character:
            await ctx.respond(
                "You don't have a character.",
                ephemeral=True,
            )
            return

        spell_index = spell_name.lower().replace(" ", "-")
        spell_info = self.spell_manager.get_spell_info(spell_index)

        if not spell_info:
            await ctx.respond(
                f"Spell '{spell_name}' not found.",
                ephemeral=True,
            )
            return

        if spell_index in character.prepared_spells:
            await ctx.respond(
                f"{spell_info.name} is already prepared!",
                ephemeral=True,
            )
            return

        # Add to known spells if not already known
        if spell_index not in character.known_spells:
            character.known_spells.append(spell_index)

        character.prepared_spells.append(spell_index)
        await repo.update(character)

        await ctx.respond(
            f":sparkles: {character.name} prepares **{spell_info.name}**!"
        )

    @spell.command(name="unprepare", description="Remove a spell from prepared list")
    async def spell_unprepare(
        self,
        ctx: discord.ApplicationContext,
        spell_name: discord.Option(
            str,
            "Spell to unprepare",
            required=True,
        ),
        campaign_id: discord.Option(
            str,
            "Campaign ID",
            required=False,
            default="default",
        ),
    ):
        """Remove a spell from your prepared spells."""
        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)

        if not character:
            await ctx.respond(
                "You don't have a character.",
                ephemeral=True,
            )
            return

        spell_index = spell_name.lower().replace(" ", "-")
        spell_info = self.spell_manager.get_spell_info(spell_index)

        if spell_index not in character.prepared_spells:
            await ctx.respond(
                f"{spell_info.name if spell_info else spell_name} is not prepared.",
                ephemeral=True,
            )
            return

        character.prepared_spells.remove(spell_index)
        await repo.update(character)

        spell_display = spell_info.name if spell_info else spell_name
        await ctx.respond(
            f":x: {character.name} unprepares **{spell_display}**."
        )

    @spell.command(name="ritual", description="Cast a spell as a ritual (no slot, +10 min)")
    async def spell_ritual(
        self,
        ctx: discord.ApplicationContext,
        spell_name: discord.Option(
            str,
            "Name of the spell to cast as a ritual",
            required=True,
        ),
        campaign_id: discord.Option(
            str,
            "Campaign ID",
            required=False,
            default="default",
        ),
    ):
        """Cast a spell as a ritual without expending a spell slot."""
        await ctx.defer()
        # Get character
        repo = await get_character_repo()
        character = await repo.get_by_user_and_campaign(ctx.author.id, campaign_id)

        if not character:
            await ctx.respond(
                "You don't have a character. Use `/character create` first.",
                ephemeral=True,
            )
            return

        if not character.spellcasting_ability:
            await ctx.respond(
                f"{character.name} is not a spellcaster!",
                ephemeral=True,
            )
            return

        # Find the spell
        spell_index = spell_name.lower().replace(" ", "-")
        spell_info = self.spell_manager.get_spell_info(spell_index)

        if not spell_info:
            await ctx.respond(
                f"Spell '{spell_name}' not found.",
                ephemeral=True,
            )
            return

        # Check if can cast as ritual
        can_ritual, reason = self.spell_manager.can_cast_as_ritual(character, spell_index)
        if not can_ritual:
            await ctx.respond(
                f":x: Cannot cast as ritual: {reason}",
                ephemeral=True,
            )
            return

        # Cast the ritual
        result = self.spell_manager.cast_ritual(character, spell_index)

        if not result.success:
            await ctx.respond(
                f":x: Ritual failed: {result.error}",
                ephemeral=True,
            )
            return

        # Build response embed
        embed = discord.Embed(
            title=f":candle: Ritual: {spell_info.name}",
            description=(
                f"**{character.name}** begins the ritual casting of {spell_info.name}...\n\n"
                f"*The spell takes {self.spell_manager.get_ritual_casting_time(spell_info)} to cast.*"
            ),
            color=discord.Color.purple(),
        )

        embed.add_field(
            name="Level",
            value=f"Level {spell_info.level}" if spell_info.level > 0 else "Cantrip",
            inline=True,
        )
        embed.add_field(
            name="Spell Slot",
            value=":sparkles: None consumed (ritual)",
            inline=True,
        )
        embed.add_field(
            name="Duration",
            value=spell_info.duration,
            inline=True,
        )

        if spell_info.concentration:
            embed.add_field(
                name=":warning: Concentration",
                value="This spell requires concentration",
                inline=False,
            )
            if result.concentration_broken:
                embed.add_field(
                    name=":boom: Concentration Broken",
                    value="Your previous concentration spell has ended",
                    inline=False,
                )

        # Add description
        desc_preview = spell_info.description[:300]
        if len(spell_info.description) > 300:
            desc_preview += "..."
        embed.add_field(
            name="Effect",
            value=desc_preview,
            inline=False,
        )

        await ctx.respond(embed=embed)


def setup(bot: commands.Bot):
    """Load the cog."""
    bot.add_cog(SpellsCog(bot))
