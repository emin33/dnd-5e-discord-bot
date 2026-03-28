"""Character sheet embed builder."""

import discord

from ...models import AbilityScore, Character, Skill, SKILL_ABILITIES
from ...data.srd import get_srd


def build_character_sheet_embed(character: Character) -> discord.Embed:
    """Build a Discord embed displaying a character sheet."""
    srd = get_srd()

    # Get race and class names
    race_data = srd.get_race(character.race_index)
    class_data = srd.get_class(character.class_index)
    race_name = race_data["name"] if race_data else character.race_index.title()
    class_name = class_data["name"] if class_data else character.class_index.title()

    # HP color coding
    hp_pct = character.hp.current / character.hp.maximum
    if hp_pct > 0.75:
        color = discord.Color.green()
    elif hp_pct > 0.25:
        color = discord.Color.gold()
    elif hp_pct > 0:
        color = discord.Color.red()
    else:
        color = discord.Color.dark_grey()

    embed = discord.Embed(
        title=character.name,
        description=f"Level {character.level} {race_name} {class_name}",
        color=color,
    )

    # HP and AC
    hp_text = f"**{character.hp.current}/{character.hp.maximum}**"
    if character.hp.temporary > 0:
        hp_text += f" (+{character.hp.temporary} temp)"

    embed.add_field(
        name="Hit Points",
        value=hp_text,
        inline=True,
    )
    embed.add_field(
        name="Armor Class",
        value=f"**{character.armor_class}**",
        inline=True,
    )
    embed.add_field(
        name="Speed",
        value=f"{character.speed} ft",
        inline=True,
    )

    # Ability Scores
    abilities_text = ""
    for ability in AbilityScore:
        score = character.abilities.get_score(ability)
        mod = character.abilities.get_modifier(ability)
        mod_str = f"+{mod}" if mod >= 0 else str(mod)
        save_mod = character.get_save_modifier(ability)
        save_str = f"+{save_mod}" if save_mod >= 0 else str(save_mod)
        prof_marker = " *" if ability in character.saving_throw_proficiencies else ""

        abilities_text += f"**{ability.name[:3].upper()}**: {score} ({mod_str}) [Save: {save_str}{prof_marker}]\n"

    embed.add_field(
        name="Ability Scores",
        value=abilities_text,
        inline=False,
    )

    # Skills (only proficient ones)
    if character.skill_proficiencies:
        skills_text = ""
        for skill in sorted(character.skill_proficiencies, key=lambda s: s.value):
            mod = character.get_skill_modifier(skill)
            mod_str = f"+{mod}" if mod >= 0 else str(mod)
            expertise_marker = " (E)" if skill in character.skill_expertise else ""
            skill_name = skill.value.replace("-", " ").title()
            skills_text += f"{skill_name}: {mod_str}{expertise_marker}\n"

        embed.add_field(
            name="Skill Proficiencies",
            value=skills_text or "None",
            inline=True,
        )

    # Passive scores
    embed.add_field(
        name="Passive Scores",
        value=(
            f"Perception: {character.passive_perception}\n"
            f"Initiative: {'+' if character.initiative_bonus >= 0 else ''}{character.initiative_bonus}"
        ),
        inline=True,
    )

    # Spellcasting (if applicable)
    if character.spellcasting_ability:
        spell_dc = character.spell_save_dc
        spell_attack = character.spell_attack_bonus
        spell_atk_str = f"+{spell_attack}" if spell_attack >= 0 else str(spell_attack)

        embed.add_field(
            name="Spellcasting",
            value=(
                f"Ability: {character.spellcasting_ability.name.title()}\n"
                f"Save DC: {spell_dc}\n"
                f"Attack: {spell_atk_str}"
            ),
            inline=True,
        )

        # Spell slots
        slots_text = ""
        for level in range(1, 10):
            current, maximum = character.spell_slots.get_slots(level)
            if maximum > 0:
                slots_text += f"L{level}: {current}/{maximum}  "

        if slots_text:
            embed.add_field(
                name="Spell Slots",
                value=slots_text,
                inline=False,
            )

    # Hit Dice
    embed.add_field(
        name="Hit Dice",
        value=f"{character.hit_dice.remaining}/{character.hit_dice.total} d{character.hit_dice.die_type}",
        inline=True,
    )

    # Conditions
    if character.conditions:
        cond_text = ", ".join(c.condition.value.title() for c in character.conditions)
        embed.add_field(
            name="Conditions",
            value=cond_text,
            inline=True,
        )

    # Death saves (if at 0 HP)
    if character.hp.current == 0:
        embed.add_field(
            name="Death Saves",
            value=f"Successes: {character.death_saves.successes}/3\nFailures: {character.death_saves.failures}/3",
            inline=True,
        )

    embed.set_footer(text=f"Proficiency Bonus: +{character.proficiency_bonus}")

    return embed


def build_character_summary_embed(character: Character) -> discord.Embed:
    """Build a compact character summary embed."""
    srd = get_srd()

    race_data = srd.get_race(character.race_index)
    class_data = srd.get_class(character.class_index)
    race_name = race_data["name"] if race_data else character.race_index.title()
    class_name = class_data["name"] if class_data else character.class_index.title()

    hp_pct = character.hp.current / character.hp.maximum
    if hp_pct > 0.75:
        hp_icon = ":green_circle:"
    elif hp_pct > 0.25:
        hp_icon = ":yellow_circle:"
    elif hp_pct > 0:
        hp_icon = ":red_circle:"
    else:
        hp_icon = ":skull:"

    embed = discord.Embed(
        title=character.name,
        description=f"Level {character.level} {race_name} {class_name}",
        color=discord.Color.blue(),
    )

    embed.add_field(
        name="Status",
        value=f"{hp_icon} HP: {character.hp.current}/{character.hp.maximum} | AC: {character.armor_class}",
        inline=False,
    )

    return embed


def build_ability_roll_embed(
    character_name: str,
    rolls: list,  # List of DiceRoll
    totals: list[int],
) -> discord.Embed:
    """Build an embed showing ability score rolls."""
    embed = discord.Embed(
        title=f"{character_name} - Ability Score Rolls",
        description="Roll 4d6, drop lowest",
        color=discord.Color.gold(),
    )

    for i, (roll, total) in enumerate(zip(rolls, totals), 1):
        kept_str = ", ".join(str(d) for d in roll.kept_dice)
        dropped_str = ", ".join(f"~~{d}~~" for d in roll.dropped_dice)
        dice_str = f"[{kept_str}, {dropped_str}]" if roll.dropped_dice else f"[{kept_str}]"

        embed.add_field(
            name=f"Roll {i}",
            value=f"{dice_str} = **{total}**",
            inline=True,
        )

    sorted_totals = sorted(totals, reverse=True)
    embed.add_field(
        name="Sorted (High to Low)",
        value=", ".join(str(t) for t in sorted_totals),
        inline=False,
    )

    return embed
