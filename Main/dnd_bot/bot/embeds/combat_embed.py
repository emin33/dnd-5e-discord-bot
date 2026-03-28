"""Combat tracker embed builder."""

from typing import Optional
import discord

from ...models import Combat, Combatant, CombatState

# Discord embed limits
MAX_FIELD_VALUE = 1024
MAX_EMBED_TOTAL = 6000


def _safe_field_value(value: str, max_len: int = MAX_FIELD_VALUE) -> str:
    """Truncate a field value to fit Discord's limits."""
    if len(value) <= max_len:
        return value
    return value[: max_len - 20] + "\n*...truncated*"


# Condition icons for display
CONDITION_ICONS = {
    "blinded": ":eye:",
    "charmed": ":heart:",
    "deafened": ":ear:",
    "frightened": ":scream:",
    "grappled": ":chains:",
    "incapacitated": ":dizzy:",
    "invisible": ":ghost:",
    "paralyzed": ":cold_face:",
    "petrified": ":rock:",
    "poisoned": ":nauseated_face:",
    "prone": ":arrow_down:",
    "restrained": ":knot:",
    "stunned": ":star:",
    "unconscious": ":zzz:",
    "exhaustion": ":tired_face:",
}


def get_hp_indicator(current: int, maximum: int) -> str:
    """Get an HP indicator emoji based on percentage."""
    if maximum == 0:
        return ":skull:"

    pct = current / maximum
    if pct > 0.75:
        return ":green_circle:"
    elif pct > 0.5:
        return ":yellow_circle:"
    elif pct > 0.25:
        return ":orange_circle:"
    elif pct > 0:
        return ":red_circle:"
    else:
        return ":skull:"


def build_combat_tracker_embed(
    combat: Combat,
    show_hp: bool = True,
    show_ac: bool = False,
) -> discord.Embed:
    """Build the initiative order display embed."""
    current = combat.get_current_combatant()
    sorted_combatants = combat.get_sorted_combatants()

    # Title with round info
    title = f"Combat - Round {combat.current_round}"
    if combat.encounter_name:
        title = f"{combat.encounter_name} - Round {combat.current_round}"

    embed = discord.Embed(
        title=title,
        color=discord.Color.red(),
    )

    if combat.encounter_description:
        embed.description = combat.encounter_description

    # Build initiative list
    initiative_lines = []
    for combatant in sorted_combatants:
        line = _format_combatant_line(combatant, current, show_hp, show_ac)
        initiative_lines.append(line)

    initiative_text = "\n".join(initiative_lines) if initiative_lines else "No combatants"
    embed.add_field(
        name="Initiative Order",
        value=_safe_field_value(initiative_text),
        inline=False,
    )

    # Current turn info
    if current and combat.state == CombatState.AWAITING_ACTION:
        resources = current.turn_resources
        actions = []
        if resources.action:
            actions.append(":crossed_swords: Action")
        if resources.bonus_action:
            actions.append(":zap: Bonus")
        if resources.reaction:
            actions.append(":shield: Reaction")

        turn_info = (
            f"**HP:** {current.hp_current}/{current.hp_max}\n"
            f"**AC:** {current.armor_class}\n"
            f"**Movement:** {resources.movement} ft remaining"
        )

        embed.add_field(
            name=f":arrow_right: {current.name}'s Turn",
            value=turn_info,
            inline=True,
        )

        embed.add_field(
            name="Available",
            value=" | ".join(actions) if actions else "None remaining",
            inline=True,
        )

    # Combat state indicator
    state_text = {
        CombatState.SETUP: "Setting up...",
        CombatState.ROLLING_INITIATIVE: "Rolling initiative...",
        CombatState.ACTIVE: "Combat active",
        CombatState.AWAITING_ACTION: "Awaiting action",
        CombatState.RESOLVING_ACTION: "Resolving action...",
        CombatState.END_TURN: "Ending turn...",
        CombatState.COMBAT_END: "Combat ended",
    }.get(combat.state, "Unknown")

    embed.set_footer(text=f"State: {state_text} | Use /combat for actions")

    return embed


def _format_combatant_line(
    combatant: Combatant,
    current: Optional[Combatant],
    show_hp: bool,
    show_ac: bool,
) -> str:
    """Format a single combatant line for the tracker."""
    parts = []

    # Current turn indicator
    if current and combatant.id == current.id:
        parts.append("**>>**")
    else:
        parts.append("   ")

    # Initiative
    init_str = f"`{combatant.initiative_roll or 0:2d}`"
    parts.append(init_str)

    # HP indicator
    if show_hp:
        hp_icon = get_hp_indicator(combatant.hp_current, combatant.hp_max)
        parts.append(hp_icon)

    # Name (bold if current, strikethrough if dead)
    if not combatant.is_active or combatant.hp_current <= 0:
        name = f"~~{combatant.name}~~"
    elif current and combatant.id == current.id:
        name = f"**{combatant.name}**"
    else:
        name = combatant.name

    parts.append(name)

    # HP numbers (optional)
    if show_hp and combatant.is_active:
        parts.append(f"({combatant.hp_current}/{combatant.hp_max})")

    # AC (optional)
    if show_ac and combatant.is_active:
        parts.append(f"AC:{combatant.armor_class}")

    # NPC indicator
    if not combatant.is_player:
        parts.append("(NPC)")

    # Surprised indicator
    if combatant.is_surprised:
        parts.append(":exclamation:")

    return " ".join(parts)


def build_initiative_results_embed(
    results: list[tuple[Combatant, "DiceRoll"]],
) -> discord.Embed:
    """Build an embed showing initiative roll results."""
    embed = discord.Embed(
        title=":zap: Initiative Rolled!",
        color=discord.Color.purple(),
    )

    # Sort by initiative
    sorted_results = sorted(results, key=lambda x: x[1].total, reverse=True)

    lines = []
    for i, (combatant, roll) in enumerate(sorted_results, 1):
        mod_str = f"+{combatant.initiative_bonus}" if combatant.initiative_bonus >= 0 else str(combatant.initiative_bonus)
        lines.append(
            f"**{i}.** {combatant.name}: [{roll.kept_dice[0]}] {mod_str} = **{roll.total}**"
        )

    embed.description = "\n".join(lines)

    return embed


def build_combat_start_embed(combat: Combat) -> discord.Embed:
    """Build an embed for combat start."""
    embed = discord.Embed(
        title=":crossed_swords: Combat Begins!",
        description=combat.encounter_description or "Roll for initiative!",
        color=discord.Color.red(),
    )

    if combat.encounter_name:
        embed.title = f":crossed_swords: {combat.encounter_name}"

    # List combatants
    players = [c for c in combat.combatants if c.is_player]
    enemies = [c for c in combat.combatants if not c.is_player]

    if players:
        embed.add_field(
            name="Party",
            value="\n".join(f"- {c.name}" for c in players),
            inline=True,
        )

    if enemies:
        embed.add_field(
            name="Enemies",
            value="\n".join(f"- {c.name}" for c in enemies),
            inline=True,
        )

    return embed


def build_combat_end_embed(combat: Combat, victory: bool) -> discord.Embed:
    """Build an embed for combat end."""
    if victory:
        embed = discord.Embed(
            title=":trophy: Victory!",
            description="The enemies have been defeated!",
            color=discord.Color.gold(),
        )
    else:
        embed = discord.Embed(
            title=":skull: Defeat",
            description="The party has fallen...",
            color=discord.Color.dark_grey(),
        )

    embed.add_field(
        name="Combat Summary",
        value=f"Rounds: {combat.current_round}",
        inline=True,
    )

    # Show survivor status
    survivors = [c for c in combat.combatants if c.is_active and c.hp_current > 0]
    if survivors:
        survivor_text = "\n".join(
            f"- {c.name}: {c.hp_current}/{c.hp_max} HP"
            for c in survivors
        )
        embed.add_field(
            name="Survivors",
            value=survivor_text,
            inline=True,
        )

    return embed


def build_attack_result_embed(
    attacker_name: str,
    target_name: str,
    attack_roll: "DiceRoll",
    target_ac: int,
    hit: bool,
    critical: bool,
    damage_roll: Optional["DiceRoll"] = None,
    damage_dealt: int = 0,
) -> discord.Embed:
    """Build an embed for an attack result."""
    if critical:
        embed = discord.Embed(
            title=f":boom: Critical Hit!",
            color=discord.Color.gold(),
        )
    elif hit:
        embed = discord.Embed(
            title=f":crossed_swords: {attacker_name} hits {target_name}!",
            color=discord.Color.green(),
        )
    else:
        embed = discord.Embed(
            title=f":shield: {attacker_name} misses {target_name}",
            color=discord.Color.red(),
        )

    # Attack roll details
    roll_text = f"[{attack_roll.kept_dice[0]}]"
    if attack_roll.modifier != 0:
        mod_str = f" + {attack_roll.modifier}" if attack_roll.modifier > 0 else f" - {abs(attack_roll.modifier)}"
        roll_text += mod_str
    roll_text += f" = **{attack_roll.total}** vs AC {target_ac}"

    embed.add_field(
        name="Attack Roll",
        value=roll_text,
        inline=False,
    )

    # Damage if hit
    if hit and damage_roll:
        damage_text = f"[{', '.join(str(d) for d in damage_roll.kept_dice)}]"
        if damage_roll.modifier != 0:
            mod_str = f" + {damage_roll.modifier}" if damage_roll.modifier > 0 else f" - {abs(damage_roll.modifier)}"
            damage_text += mod_str
        damage_text += f" = **{damage_dealt}** damage"

        if critical:
            damage_text += " (Critical!)"

        embed.add_field(
            name="Damage",
            value=damage_text,
            inline=False,
        )

    return embed
