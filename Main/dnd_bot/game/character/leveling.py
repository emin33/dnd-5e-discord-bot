"""Character leveling and XP system."""

from dataclasses import dataclass
from typing import Optional
import random

from ...models import Character, AbilityScore


# XP thresholds for each level (1-20)
# Level 1 = 0 XP, Level 2 = 300 XP, etc.
XP_THRESHOLDS = {
    1: 0,
    2: 300,
    3: 900,
    4: 2700,
    5: 6500,
    6: 14000,
    7: 23000,
    8: 34000,
    9: 48000,
    10: 64000,
    11: 85000,
    12: 100000,
    13: 120000,
    14: 140000,
    15: 165000,
    16: 195000,
    17: 225000,
    18: 265000,
    19: 305000,
    20: 355000,
}

# Hit dice by class (d6, d8, d10, or d12)
CLASS_HIT_DICE = {
    "barbarian": 12,
    "bard": 8,
    "cleric": 8,
    "druid": 8,
    "fighter": 10,
    "monk": 8,
    "paladin": 10,
    "ranger": 10,
    "rogue": 8,
    "sorcerer": 6,
    "warlock": 8,
    "wizard": 6,
}

# Proficiency bonus by level
PROFICIENCY_BY_LEVEL = {
    1: 2, 2: 2, 3: 2, 4: 2,
    5: 3, 6: 3, 7: 3, 8: 3,
    9: 4, 10: 4, 11: 4, 12: 4,
    13: 5, 14: 5, 15: 5, 16: 5,
    17: 6, 18: 6, 19: 6, 20: 6,
}

# Levels that grant Ability Score Improvement (ASI)
ASI_LEVELS = [4, 8, 12, 16, 19]

# Fighter gets extra ASIs
FIGHTER_ASI_LEVELS = [4, 6, 8, 12, 14, 16, 19]

# Rogue gets extra ASI at 10
ROGUE_ASI_LEVELS = [4, 8, 10, 12, 16, 19]

# Spell slot progression for full casters (Wizard, Cleric, Druid, Bard, Sorcerer)
FULL_CASTER_SLOTS = {
    1:  [2, 0, 0, 0, 0, 0, 0, 0, 0],
    2:  [3, 0, 0, 0, 0, 0, 0, 0, 0],
    3:  [4, 2, 0, 0, 0, 0, 0, 0, 0],
    4:  [4, 3, 0, 0, 0, 0, 0, 0, 0],
    5:  [4, 3, 2, 0, 0, 0, 0, 0, 0],
    6:  [4, 3, 3, 0, 0, 0, 0, 0, 0],
    7:  [4, 3, 3, 1, 0, 0, 0, 0, 0],
    8:  [4, 3, 3, 2, 0, 0, 0, 0, 0],
    9:  [4, 3, 3, 3, 1, 0, 0, 0, 0],
    10: [4, 3, 3, 3, 2, 0, 0, 0, 0],
    11: [4, 3, 3, 3, 2, 1, 0, 0, 0],
    12: [4, 3, 3, 3, 2, 1, 0, 0, 0],
    13: [4, 3, 3, 3, 2, 1, 1, 0, 0],
    14: [4, 3, 3, 3, 2, 1, 1, 0, 0],
    15: [4, 3, 3, 3, 2, 1, 1, 1, 0],
    16: [4, 3, 3, 3, 2, 1, 1, 1, 0],
    17: [4, 3, 3, 3, 2, 1, 1, 1, 1],
    18: [4, 3, 3, 3, 3, 1, 1, 1, 1],
    19: [4, 3, 3, 3, 3, 2, 1, 1, 1],
    20: [4, 3, 3, 3, 3, 2, 2, 1, 1],
}

# Spell slot progression for half casters (Paladin, Ranger)
HALF_CASTER_SLOTS = {
    1:  [0, 0, 0, 0, 0, 0, 0, 0, 0],
    2:  [2, 0, 0, 0, 0, 0, 0, 0, 0],
    3:  [3, 0, 0, 0, 0, 0, 0, 0, 0],
    4:  [3, 0, 0, 0, 0, 0, 0, 0, 0],
    5:  [4, 2, 0, 0, 0, 0, 0, 0, 0],
    6:  [4, 2, 0, 0, 0, 0, 0, 0, 0],
    7:  [4, 3, 0, 0, 0, 0, 0, 0, 0],
    8:  [4, 3, 0, 0, 0, 0, 0, 0, 0],
    9:  [4, 3, 2, 0, 0, 0, 0, 0, 0],
    10: [4, 3, 2, 0, 0, 0, 0, 0, 0],
    11: [4, 3, 3, 0, 0, 0, 0, 0, 0],
    12: [4, 3, 3, 0, 0, 0, 0, 0, 0],
    13: [4, 3, 3, 1, 0, 0, 0, 0, 0],
    14: [4, 3, 3, 1, 0, 0, 0, 0, 0],
    15: [4, 3, 3, 2, 0, 0, 0, 0, 0],
    16: [4, 3, 3, 2, 0, 0, 0, 0, 0],
    17: [4, 3, 3, 3, 1, 0, 0, 0, 0],
    18: [4, 3, 3, 3, 1, 0, 0, 0, 0],
    19: [4, 3, 3, 3, 2, 0, 0, 0, 0],
    20: [4, 3, 3, 3, 2, 0, 0, 0, 0],
}

# Warlock pact slots (special - few slots, short rest recovery)
WARLOCK_PACT_SLOTS = {
    1:  (1, 1),   # (num_slots, slot_level)
    2:  (2, 1),
    3:  (2, 2),
    4:  (2, 2),
    5:  (2, 3),
    6:  (2, 3),
    7:  (2, 4),
    8:  (2, 4),
    9:  (2, 5),
    10: (2, 5),
    11: (3, 5),
    12: (3, 5),
    13: (3, 5),
    14: (3, 5),
    15: (3, 5),
    16: (3, 5),
    17: (4, 5),
    18: (4, 5),
    19: (4, 5),
    20: (4, 5),
}

# Full caster classes
FULL_CASTERS = {"wizard", "cleric", "druid", "bard", "sorcerer"}
HALF_CASTERS = {"paladin", "ranger"}
THIRD_CASTERS = {"fighter", "rogue"}  # Only with specific subclasses


@dataclass
class LevelUpResult:
    """Result of leveling up a character."""

    old_level: int
    new_level: int
    hp_gained: int
    hp_rolled: int  # The die roll (before CON mod)
    new_proficiency_bonus: int
    old_proficiency_bonus: int
    has_asi: bool
    new_spell_slots: dict[int, int]  # level -> max slots
    features_gained: list[str]  # Class feature names


def get_level_for_xp(xp: int) -> int:
    """Determine character level based on XP."""
    for level in range(20, 0, -1):
        if xp >= XP_THRESHOLDS[level]:
            return level
    return 1


def get_xp_for_next_level(current_level: int) -> Optional[int]:
    """Get XP needed for next level. Returns None if at max level."""
    if current_level >= 20:
        return None
    return XP_THRESHOLDS[current_level + 1]


def get_xp_progress(current_xp: int, current_level: int) -> tuple[int, int]:
    """Get (current_progress, needed_for_level) XP values."""
    if current_level >= 20:
        return (0, 0)

    current_threshold = XP_THRESHOLDS[current_level]
    next_threshold = XP_THRESHOLDS[current_level + 1]

    progress = current_xp - current_threshold
    needed = next_threshold - current_threshold

    return (progress, needed)


def get_hit_die(class_index: str) -> int:
    """Get hit die size for a class."""
    return CLASS_HIT_DICE.get(class_index.lower(), 8)


def roll_hp_increase(hit_die: int, con_mod: int, take_average: bool = False) -> tuple[int, int]:
    """
    Roll HP increase for leveling up.
    Returns (total_hp_gained, die_roll_value).

    If take_average, uses (die/2 + 1) instead of rolling.
    """
    if take_average:
        die_roll = (hit_die // 2) + 1
    else:
        die_roll = random.randint(1, hit_die)

    total = max(1, die_roll + con_mod)  # Minimum 1 HP gained
    return (total, die_roll)


def get_spell_slots_for_level(class_index: str, level: int) -> list[int]:
    """Get spell slot maximums for a class at a given level."""
    class_lower = class_index.lower()

    if class_lower == "warlock":
        # Warlock uses pact magic - special handling
        num_slots, slot_level = WARLOCK_PACT_SLOTS.get(level, (0, 0))
        slots = [0] * 9
        if slot_level > 0 and num_slots > 0:
            slots[slot_level - 1] = num_slots
        return slots
    elif class_lower in FULL_CASTERS:
        return FULL_CASTER_SLOTS.get(level, [0] * 9)
    elif class_lower in HALF_CASTERS:
        return HALF_CASTER_SLOTS.get(level, [0] * 9)
    else:
        # Non-casters or third casters without subclass
        return [0] * 9


def get_asi_levels(class_index: str) -> list[int]:
    """Get levels that grant ASI for a class."""
    class_lower = class_index.lower()
    if class_lower == "fighter":
        return FIGHTER_ASI_LEVELS.copy()
    elif class_lower == "rogue":
        return ROGUE_ASI_LEVELS.copy()
    return ASI_LEVELS.copy()


def can_level_up(character: Character) -> bool:
    """Check if character has enough XP to level up."""
    if character.level >= 20:
        return False
    next_threshold = XP_THRESHOLDS.get(character.level + 1)
    return next_threshold is not None and character.experience >= next_threshold


def level_up(
    character: Character,
    take_average_hp: bool = True,
) -> LevelUpResult:
    """
    Level up a character.

    Modifies the character in place and returns a LevelUpResult.
    Raises ValueError if character can't level up.
    """
    if not can_level_up(character):
        raise ValueError(f"{character.name} cannot level up (insufficient XP or max level)")

    old_level = character.level
    new_level = old_level + 1

    # Increase level
    character.level = new_level

    # Calculate HP increase
    hit_die = get_hit_die(character.class_index)
    con_mod = character.abilities.con_mod
    hp_gained, hp_rolled = roll_hp_increase(hit_die, con_mod, take_average_hp)

    # Update HP
    character.hp.maximum += hp_gained
    character.hp.current += hp_gained  # Also heal by the gained amount

    # Update hit dice
    character.hit_dice.total = new_level
    character.hit_dice.remaining = min(character.hit_dice.remaining + 1, new_level)

    # Get proficiency changes
    old_prof = PROFICIENCY_BY_LEVEL[old_level]
    new_prof = PROFICIENCY_BY_LEVEL[new_level]

    # Check for ASI
    asi_levels = get_asi_levels(character.class_index)
    has_asi = new_level in asi_levels

    # Update spell slots
    new_slots = get_spell_slots_for_level(character.class_index, new_level)
    new_spell_slots = {}

    for i, max_slots in enumerate(new_slots):
        level = i + 1
        if max_slots > 0:
            current, _ = character.spell_slots.get_slots(level)
            # Set new max, keeping current slots (but cap at new max)
            setattr(character.spell_slots, f"level_{level}", (min(current, max_slots), max_slots))
            new_spell_slots[level] = max_slots

    # Get features gained (placeholder - would need SRD integration)
    features_gained = _get_class_features_for_level(character.class_index, new_level)

    return LevelUpResult(
        old_level=old_level,
        new_level=new_level,
        hp_gained=hp_gained,
        hp_rolled=hp_rolled,
        new_proficiency_bonus=new_prof,
        old_proficiency_bonus=old_prof,
        has_asi=has_asi,
        new_spell_slots=new_spell_slots,
        features_gained=features_gained,
    )


def apply_asi(
    character: Character,
    ability1: AbilityScore,
    ability2: Optional[AbilityScore] = None,
    increase1: int = 1,
    increase2: int = 1,
) -> dict[str, int]:
    """
    Apply Ability Score Improvement.

    Can either:
    - Increase one ability by 2 (ability1 only, increase1=2)
    - Increase two abilities by 1 each (both abilities, increase1=1, increase2=1)

    Returns dict of {ability_name: new_score}.
    """
    changes = {}

    # Apply first increase
    current1 = character.abilities.get_score(ability1)
    new1 = min(20, current1 + increase1)  # Cap at 20
    setattr(character.abilities, ability1.name.lower(), new1)
    changes[ability1.name] = new1

    # Apply second increase if provided
    if ability2 is not None and increase2 > 0:
        current2 = character.abilities.get_score(ability2)
        new2 = min(20, current2 + increase2)
        setattr(character.abilities, ability2.name.lower(), new2)
        changes[ability2.name] = new2

    return changes


def _get_class_features_for_level(class_index: str, level: int) -> list[str]:
    """
    Get class features gained at a specific level.

    This is a simplified version - full implementation would use SRD data.
    """
    features = []
    class_lower = class_index.lower()

    # Common features at certain levels
    if level == 5:
        if class_lower in {"fighter", "paladin", "ranger", "barbarian", "monk"}:
            features.append("Extra Attack")

    if level == 11 and class_lower == "fighter":
        features.append("Extra Attack (2)")

    if level == 20 and class_lower == "fighter":
        features.append("Extra Attack (3)")

    # Spellcasting for half-casters at level 2
    if level == 2 and class_lower in {"paladin", "ranger"}:
        features.append("Spellcasting")

    # Class-specific iconic features
    class_features = {
        "barbarian": {
            1: ["Rage", "Unarmored Defense"],
            2: ["Reckless Attack", "Danger Sense"],
            3: ["Primal Path"],
            5: ["Fast Movement"],
            7: ["Feral Instinct"],
            11: ["Relentless Rage"],
            15: ["Persistent Rage"],
            18: ["Indomitable Might"],
            20: ["Primal Champion"],
        },
        "fighter": {
            1: ["Fighting Style", "Second Wind"],
            2: ["Action Surge"],
            3: ["Martial Archetype"],
            9: ["Indomitable"],
        },
        "rogue": {
            1: ["Sneak Attack", "Thieves' Cant"],
            2: ["Cunning Action"],
            3: ["Roguish Archetype"],
            5: ["Uncanny Dodge"],
            7: ["Evasion"],
            11: ["Reliable Talent"],
            14: ["Blindsense"],
            18: ["Elusive"],
            20: ["Stroke of Luck"],
        },
        "wizard": {
            1: ["Spellcasting", "Arcane Recovery"],
            2: ["Arcane Tradition"],
            18: ["Spell Mastery"],
            20: ["Signature Spells"],
        },
        "cleric": {
            1: ["Spellcasting", "Divine Domain"],
            2: ["Channel Divinity"],
            5: ["Destroy Undead"],
            10: ["Divine Intervention"],
        },
    }

    if class_lower in class_features:
        level_features = class_features[class_lower].get(level, [])
        features.extend(level_features)

    return features


class LevelingManager:
    """Manages character leveling operations."""

    def __init__(self):
        pass

    def add_xp(self, character: Character, amount: int) -> tuple[int, bool]:
        """
        Add XP to a character.
        Returns (new_total_xp, can_level_up).
        """
        character.experience += amount
        return (character.experience, can_level_up(character))

    def set_xp(self, character: Character, amount: int) -> tuple[int, bool]:
        """
        Set character's XP to a specific value.
        Returns (new_total_xp, can_level_up).
        """
        character.experience = max(0, amount)
        return (character.experience, can_level_up(character))

    def get_xp_status(self, character: Character) -> dict:
        """Get detailed XP status for a character."""
        progress, needed = get_xp_progress(character.experience, character.level)
        next_level_xp = get_xp_for_next_level(character.level)

        return {
            "current_xp": character.experience,
            "current_level": character.level,
            "progress_to_next": progress,
            "xp_needed_for_next": needed,
            "next_level_threshold": next_level_xp,
            "can_level_up": can_level_up(character),
            "is_max_level": character.level >= 20,
        }

    def level_up_character(
        self,
        character: Character,
        take_average_hp: bool = True,
    ) -> LevelUpResult:
        """Level up a character if possible."""
        return level_up(character, take_average_hp)

    def apply_asi_to_character(
        self,
        character: Character,
        ability1: AbilityScore,
        ability2: Optional[AbilityScore] = None,
        increase1: int = 1,
        increase2: int = 1,
    ) -> dict[str, int]:
        """Apply ASI to a character."""
        return apply_asi(character, ability1, ability2, increase1, increase2)


# Global manager instance
_leveling_manager: Optional[LevelingManager] = None


def get_leveling_manager() -> LevelingManager:
    """Get the global leveling manager."""
    global _leveling_manager
    if _leveling_manager is None:
        _leveling_manager = LevelingManager()
    return _leveling_manager
