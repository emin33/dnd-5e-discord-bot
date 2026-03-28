"""Condition effects system for D&D 5e conditions."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ...models import AbilityScore, Condition


class AdvantageState(str, Enum):
    """Advantage/disadvantage state."""
    NONE = "none"
    ADVANTAGE = "advantage"
    DISADVANTAGE = "disadvantage"


@dataclass
class ConditionEffects:
    """
    The mechanical effects of a D&D 5e condition.

    This defines what modifiers a condition applies to various checks.
    """

    condition: Condition

    # Attack modifiers
    attack_advantage: AdvantageState = AdvantageState.NONE
    attacks_against_advantage: AdvantageState = AdvantageState.NONE

    # Ability check modifiers
    ability_check_advantage: AdvantageState = AdvantageState.NONE
    affected_ability_checks: list[AbilityScore] = field(default_factory=list)  # Empty = all

    # Saving throw modifiers
    saving_throw_advantage: AdvantageState = AdvantageState.NONE
    affected_saves: list[AbilityScore] = field(default_factory=list)  # Empty = all
    auto_fail_saves: list[AbilityScore] = field(default_factory=list)

    # Combat modifiers
    speed_multiplier: float = 1.0  # 0 = can't move, 0.5 = half speed
    can_take_actions: bool = True
    can_take_reactions: bool = True
    critical_hit_range: int = 20  # Hits within 5 feet are crits if <= this

    # Special flags
    auto_fail_str_dex_saves: bool = False
    attacks_hit_within_5ft_are_crits: bool = False
    grants_advantage_to_attackers: bool = False
    is_incapacitated: bool = False
    cant_speak: bool = False

    # Description for display
    description: str = ""


# Define all condition effects according to D&D 5e SRD rules
CONDITION_EFFECTS: dict[Condition, ConditionEffects] = {
    Condition.BLINDED: ConditionEffects(
        condition=Condition.BLINDED,
        attack_advantage=AdvantageState.DISADVANTAGE,
        attacks_against_advantage=AdvantageState.ADVANTAGE,
        description=(
            "A blinded creature can't see and automatically fails any ability check "
            "that requires sight. Attack rolls against the creature have advantage, "
            "and the creature's attack rolls have disadvantage."
        ),
    ),

    Condition.CHARMED: ConditionEffects(
        condition=Condition.CHARMED,
        description=(
            "A charmed creature can't attack the charmer or target the charmer with "
            "harmful abilities or magical effects. The charmer has advantage on any "
            "ability check to interact socially with the creature."
        ),
        # Note: Charmer-specific effects are handled separately
    ),

    Condition.DEAFENED: ConditionEffects(
        condition=Condition.DEAFENED,
        description=(
            "A deafened creature can't hear and automatically fails any ability check "
            "that requires hearing."
        ),
    ),

    Condition.EXHAUSTION: ConditionEffects(
        condition=Condition.EXHAUSTION,
        description=(
            "Exhaustion has 6 levels with cumulative effects:\n"
            "1. Disadvantage on ability checks\n"
            "2. Speed halved\n"
            "3. Disadvantage on attacks and saves\n"
            "4. HP maximum halved\n"
            "5. Speed reduced to 0\n"
            "6. Death"
        ),
        # Note: Level-specific effects are handled by get_exhaustion_effects()
    ),

    Condition.FRIGHTENED: ConditionEffects(
        condition=Condition.FRIGHTENED,
        attack_advantage=AdvantageState.DISADVANTAGE,
        ability_check_advantage=AdvantageState.DISADVANTAGE,
        description=(
            "A frightened creature has disadvantage on ability checks and attack rolls "
            "while the source of its fear is within line of sight. The creature can't "
            "willingly move closer to the source of its fear."
        ),
    ),

    Condition.GRAPPLED: ConditionEffects(
        condition=Condition.GRAPPLED,
        speed_multiplier=0.0,
        description=(
            "A grappled creature's speed becomes 0, and it can't benefit from any "
            "bonus to its speed. The condition ends if the grappler is incapacitated "
            "or if an effect removes the grappled creature from the grappler's reach."
        ),
    ),

    Condition.INCAPACITATED: ConditionEffects(
        condition=Condition.INCAPACITATED,
        can_take_actions=False,
        can_take_reactions=False,
        is_incapacitated=True,
        description="An incapacitated creature can't take actions or reactions.",
    ),

    Condition.INVISIBLE: ConditionEffects(
        condition=Condition.INVISIBLE,
        attack_advantage=AdvantageState.ADVANTAGE,
        attacks_against_advantage=AdvantageState.DISADVANTAGE,
        description=(
            "An invisible creature is impossible to see without magic or special sense. "
            "The creature's location can be detected by noise or tracks. Attack rolls "
            "against the creature have disadvantage, and the creature's attack rolls have advantage."
        ),
    ),

    Condition.PARALYZED: ConditionEffects(
        condition=Condition.PARALYZED,
        can_take_actions=False,
        can_take_reactions=False,
        is_incapacitated=True,
        cant_speak=True,
        speed_multiplier=0.0,
        auto_fail_str_dex_saves=True,
        attacks_against_advantage=AdvantageState.ADVANTAGE,
        attacks_hit_within_5ft_are_crits=True,
        description=(
            "A paralyzed creature is incapacitated and can't move or speak. "
            "The creature automatically fails Strength and Dexterity saving throws. "
            "Attack rolls against the creature have advantage. Any attack that hits "
            "the creature is a critical hit if the attacker is within 5 feet."
        ),
    ),

    Condition.PETRIFIED: ConditionEffects(
        condition=Condition.PETRIFIED,
        can_take_actions=False,
        can_take_reactions=False,
        is_incapacitated=True,
        speed_multiplier=0.0,
        auto_fail_str_dex_saves=True,
        attacks_against_advantage=AdvantageState.ADVANTAGE,
        description=(
            "A petrified creature is transformed into a solid inanimate substance. "
            "It is incapacitated, can't move or speak, and is unaware of its surroundings. "
            "Attack rolls against it have advantage. It automatically fails Strength and "
            "Dexterity saves. It has resistance to all damage and is immune to poison and disease."
        ),
    ),

    Condition.POISONED: ConditionEffects(
        condition=Condition.POISONED,
        attack_advantage=AdvantageState.DISADVANTAGE,
        ability_check_advantage=AdvantageState.DISADVANTAGE,
        description=(
            "A poisoned creature has disadvantage on attack rolls and ability checks."
        ),
    ),

    Condition.PRONE: ConditionEffects(
        condition=Condition.PRONE,
        attack_advantage=AdvantageState.DISADVANTAGE,
        description=(
            "A prone creature's only movement option is to crawl, unless it stands up. "
            "The creature has disadvantage on attack rolls. An attack roll against the "
            "creature has advantage if the attacker is within 5 feet, otherwise disadvantage."
        ),
        # Note: Melee vs ranged advantage is handled separately
    ),

    Condition.RESTRAINED: ConditionEffects(
        condition=Condition.RESTRAINED,
        speed_multiplier=0.0,
        attack_advantage=AdvantageState.DISADVANTAGE,
        attacks_against_advantage=AdvantageState.ADVANTAGE,
        saving_throw_advantage=AdvantageState.DISADVANTAGE,
        affected_saves=[AbilityScore.DEXTERITY],
        description=(
            "A restrained creature's speed becomes 0. Attack rolls against the creature "
            "have advantage, and the creature's attack rolls have disadvantage. "
            "The creature has disadvantage on Dexterity saving throws."
        ),
    ),

    Condition.STUNNED: ConditionEffects(
        condition=Condition.STUNNED,
        can_take_actions=False,
        can_take_reactions=False,
        is_incapacitated=True,
        cant_speak=True,
        auto_fail_str_dex_saves=True,
        attacks_against_advantage=AdvantageState.ADVANTAGE,
        description=(
            "A stunned creature is incapacitated, can't move, and can speak only falteringly. "
            "The creature automatically fails Strength and Dexterity saving throws. "
            "Attack rolls against the creature have advantage."
        ),
    ),

    Condition.UNCONSCIOUS: ConditionEffects(
        condition=Condition.UNCONSCIOUS,
        can_take_actions=False,
        can_take_reactions=False,
        is_incapacitated=True,
        speed_multiplier=0.0,
        auto_fail_str_dex_saves=True,
        attacks_against_advantage=AdvantageState.ADVANTAGE,
        attacks_hit_within_5ft_are_crits=True,
        description=(
            "An unconscious creature is incapacitated, can't move or speak, and is unaware "
            "of its surroundings. The creature drops whatever it's holding and falls prone. "
            "The creature automatically fails Strength and Dexterity saving throws. "
            "Attack rolls against the creature have advantage. Any attack that hits is a "
            "critical hit if the attacker is within 5 feet."
        ),
    ),
}


def get_condition_effects(condition: Condition) -> ConditionEffects:
    """Get the mechanical effects for a condition."""
    return CONDITION_EFFECTS.get(condition, ConditionEffects(condition=condition))


@dataclass
class ExhaustionEffects:
    """The cumulative effects of exhaustion levels."""
    level: int
    ability_check_disadvantage: bool = False
    speed_halved: bool = False
    attack_disadvantage: bool = False
    save_disadvantage: bool = False
    hp_max_halved: bool = False
    speed_zero: bool = False
    is_dead: bool = False


def get_exhaustion_effects(level: int) -> ExhaustionEffects:
    """Get the cumulative effects for an exhaustion level (1-6)."""
    level = max(0, min(6, level))

    return ExhaustionEffects(
        level=level,
        ability_check_disadvantage=level >= 1,
        speed_halved=level >= 2,
        attack_disadvantage=level >= 3,
        save_disadvantage=level >= 3,
        hp_max_halved=level >= 4,
        speed_zero=level >= 5,
        is_dead=level >= 6,
    )


class ConditionResolver:
    """
    Resolves condition effects for combat and ability checks.

    Use this to determine advantage/disadvantage and other modifiers
    based on active conditions.
    """

    @staticmethod
    def get_attack_modifiers(
        conditions: list[Condition],
        exhaustion_level: int = 0,
    ) -> tuple[bool, bool]:
        """
        Get attack roll modifiers based on conditions.

        Returns (has_advantage, has_disadvantage).
        Advantage and disadvantage cancel out.
        """
        has_advantage = False
        has_disadvantage = False

        for condition in conditions:
            effects = get_condition_effects(condition)
            if effects.attack_advantage == AdvantageState.ADVANTAGE:
                has_advantage = True
            elif effects.attack_advantage == AdvantageState.DISADVANTAGE:
                has_disadvantage = True

        # Check exhaustion
        if exhaustion_level >= 3:
            has_disadvantage = True

        return (has_advantage, has_disadvantage)

    @staticmethod
    def get_attacks_against_modifiers(
        conditions: list[Condition],
        attacker_within_5ft: bool = True,
    ) -> tuple[bool, bool]:
        """
        Get modifiers for attacks against a creature with these conditions.

        Returns (attacker_has_advantage, attacker_has_disadvantage).
        """
        has_advantage = False
        has_disadvantage = False

        for condition in conditions:
            effects = get_condition_effects(condition)
            if effects.attacks_against_advantage == AdvantageState.ADVANTAGE:
                has_advantage = True
            elif effects.attacks_against_advantage == AdvantageState.DISADVANTAGE:
                has_disadvantage = True

            # Prone special case
            if condition == Condition.PRONE:
                if attacker_within_5ft:
                    has_advantage = True
                else:
                    has_disadvantage = True

        return (has_advantage, has_disadvantage)

    @staticmethod
    def is_auto_crit(conditions: list[Condition], attacker_within_5ft: bool) -> bool:
        """Check if hits against this creature are automatic crits."""
        if not attacker_within_5ft:
            return False

        for condition in conditions:
            effects = get_condition_effects(condition)
            if effects.attacks_hit_within_5ft_are_crits:
                return True

        return False

    @staticmethod
    def get_ability_check_modifiers(
        conditions: list[Condition],
        ability: AbilityScore,
        exhaustion_level: int = 0,
    ) -> tuple[bool, bool]:
        """
        Get ability check modifiers based on conditions.

        Returns (has_advantage, has_disadvantage).
        """
        has_advantage = False
        has_disadvantage = False

        for condition in conditions:
            effects = get_condition_effects(condition)
            if effects.ability_check_advantage == AdvantageState.NONE:
                continue

            # Check if this ability is affected
            if effects.affected_ability_checks and ability not in effects.affected_ability_checks:
                continue

            if effects.ability_check_advantage == AdvantageState.ADVANTAGE:
                has_advantage = True
            elif effects.ability_check_advantage == AdvantageState.DISADVANTAGE:
                has_disadvantage = True

        # Check exhaustion
        if exhaustion_level >= 1:
            has_disadvantage = True

        return (has_advantage, has_disadvantage)

    @staticmethod
    def get_saving_throw_modifiers(
        conditions: list[Condition],
        ability: AbilityScore,
        exhaustion_level: int = 0,
    ) -> tuple[bool, bool, bool]:
        """
        Get saving throw modifiers based on conditions.

        Returns (has_advantage, has_disadvantage, auto_fails).
        """
        has_advantage = False
        has_disadvantage = False
        auto_fails = False

        for condition in conditions:
            effects = get_condition_effects(condition)

            # Check auto-fail
            if effects.auto_fail_str_dex_saves and ability in [AbilityScore.STRENGTH, AbilityScore.DEXTERITY]:
                auto_fails = True

            if ability in effects.auto_fail_saves:
                auto_fails = True

            # Check advantage/disadvantage
            if effects.saving_throw_advantage == AdvantageState.NONE:
                continue

            if effects.affected_saves and ability not in effects.affected_saves:
                continue

            if effects.saving_throw_advantage == AdvantageState.ADVANTAGE:
                has_advantage = True
            elif effects.saving_throw_advantage == AdvantageState.DISADVANTAGE:
                has_disadvantage = True

        # Check exhaustion
        if exhaustion_level >= 3:
            has_disadvantage = True

        return (has_advantage, has_disadvantage, auto_fails)

    @staticmethod
    def can_take_action(conditions: list[Condition]) -> bool:
        """Check if a creature with these conditions can take actions."""
        for condition in conditions:
            effects = get_condition_effects(condition)
            if not effects.can_take_actions:
                return False
        return True

    @staticmethod
    def can_take_reaction(conditions: list[Condition]) -> bool:
        """Check if a creature with these conditions can take reactions."""
        for condition in conditions:
            effects = get_condition_effects(condition)
            if not effects.can_take_reactions:
                return False
        return True

    @staticmethod
    def get_speed_multiplier(conditions: list[Condition], exhaustion_level: int = 0) -> float:
        """
        Get the speed multiplier based on conditions.

        Returns 0-1 multiplier (0 = can't move, 0.5 = half, 1 = normal).
        """
        multiplier = 1.0

        for condition in conditions:
            effects = get_condition_effects(condition)
            multiplier = min(multiplier, effects.speed_multiplier)

        # Exhaustion
        if exhaustion_level >= 5:
            multiplier = 0.0
        elif exhaustion_level >= 2:
            multiplier = min(multiplier, 0.5)

        return multiplier

    @staticmethod
    def is_incapacitated(conditions: list[Condition]) -> bool:
        """Check if creature is incapacitated."""
        for condition in conditions:
            effects = get_condition_effects(condition)
            if effects.is_incapacitated:
                return True
        return False


# Singleton resolver instance
_resolver = ConditionResolver()


def get_condition_resolver() -> ConditionResolver:
    """Get the condition resolver instance."""
    return _resolver
