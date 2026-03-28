"""Common types and enums used across models."""

from enum import Enum
from typing import Annotated

from pydantic import Field


# Type aliases for clarity
CharacterId = Annotated[str, Field(description="Unique character identifier")]
CampaignId = Annotated[str, Field(description="Unique campaign identifier")]
CombatId = Annotated[str, Field(description="Unique combat identifier")]
UserId = Annotated[int, Field(description="Discord user ID")]
GuildId = Annotated[int, Field(description="Discord guild/server ID")]
ChannelId = Annotated[int, Field(description="Discord channel ID")]


class AbilityScore(str, Enum):
    """The six ability scores in D&D 5e."""

    STRENGTH = "str"
    DEXTERITY = "dex"
    CONSTITUTION = "con"
    INTELLIGENCE = "int"
    WISDOM = "wis"
    CHARISMA = "cha"


class Skill(str, Enum):
    """The 18 skills in D&D 5e."""

    # Strength
    ATHLETICS = "athletics"
    # Dexterity
    ACROBATICS = "acrobatics"
    SLEIGHT_OF_HAND = "sleight-of-hand"
    STEALTH = "stealth"
    # Intelligence
    ARCANA = "arcana"
    HISTORY = "history"
    INVESTIGATION = "investigation"
    NATURE = "nature"
    RELIGION = "religion"
    # Wisdom
    ANIMAL_HANDLING = "animal-handling"
    INSIGHT = "insight"
    MEDICINE = "medicine"
    PERCEPTION = "perception"
    SURVIVAL = "survival"
    # Charisma
    DECEPTION = "deception"
    INTIMIDATION = "intimidation"
    PERFORMANCE = "performance"
    PERSUASION = "persuasion"


# Mapping of skills to their governing ability
SKILL_ABILITIES: dict[Skill, AbilityScore] = {
    Skill.ATHLETICS: AbilityScore.STRENGTH,
    Skill.ACROBATICS: AbilityScore.DEXTERITY,
    Skill.SLEIGHT_OF_HAND: AbilityScore.DEXTERITY,
    Skill.STEALTH: AbilityScore.DEXTERITY,
    Skill.ARCANA: AbilityScore.INTELLIGENCE,
    Skill.HISTORY: AbilityScore.INTELLIGENCE,
    Skill.INVESTIGATION: AbilityScore.INTELLIGENCE,
    Skill.NATURE: AbilityScore.INTELLIGENCE,
    Skill.RELIGION: AbilityScore.INTELLIGENCE,
    Skill.ANIMAL_HANDLING: AbilityScore.WISDOM,
    Skill.INSIGHT: AbilityScore.WISDOM,
    Skill.MEDICINE: AbilityScore.WISDOM,
    Skill.PERCEPTION: AbilityScore.WISDOM,
    Skill.SURVIVAL: AbilityScore.WISDOM,
    Skill.DECEPTION: AbilityScore.CHARISMA,
    Skill.INTIMIDATION: AbilityScore.CHARISMA,
    Skill.PERFORMANCE: AbilityScore.CHARISMA,
    Skill.PERSUASION: AbilityScore.CHARISMA,
}


class Condition(str, Enum):
    """The 14 conditions in D&D 5e (including exhaustion)."""

    BLINDED = "blinded"
    CHARMED = "charmed"
    DEAFENED = "deafened"
    EXHAUSTION = "exhaustion"
    FRIGHTENED = "frightened"
    GRAPPLED = "grappled"
    INCAPACITATED = "incapacitated"
    INVISIBLE = "invisible"
    PARALYZED = "paralyzed"
    PETRIFIED = "petrified"
    POISONED = "poisoned"
    PRONE = "prone"
    RESTRAINED = "restrained"
    STUNNED = "stunned"
    UNCONSCIOUS = "unconscious"


class DamageType(str, Enum):
    """Damage types in D&D 5e."""

    # Physical
    BLUDGEONING = "bludgeoning"
    PIERCING = "piercing"
    SLASHING = "slashing"
    # Elemental
    ACID = "acid"
    COLD = "cold"
    FIRE = "fire"
    LIGHTNING = "lightning"
    POISON = "poison"
    THUNDER = "thunder"
    # Magical
    FORCE = "force"
    NECROTIC = "necrotic"
    PSYCHIC = "psychic"
    RADIANT = "radiant"


class GameState(str, Enum):
    """High-level game session states."""

    LOBBY = "lobby"  # Waiting for players to join
    EXPLORATION = "exploration"  # Normal play, no combat
    COMBAT = "combat"  # Active combat encounter
    SOCIAL = "social"  # Social encounter (roleplay focused)
    RESTING = "resting"  # Short or long rest
    PAUSED = "paused"  # Game paused


class CombatState(str, Enum):
    """Combat encounter states (state machine)."""

    IDLE = "idle"
    SETUP = "setup"
    ROLLING_INITIATIVE = "rolling_initiative"
    ACTIVE = "active"
    AWAITING_ACTION = "awaiting_action"
    RESOLVING_ACTION = "resolving_action"
    END_TURN = "end_turn"
    COMBAT_END = "combat_end"


class RestType(str, Enum):
    """Types of rest in D&D 5e."""

    SHORT = "short"
    LONG = "long"


class ActionType(str, Enum):
    """Types of actions in combat."""

    ACTION = "action"
    BONUS_ACTION = "bonus_action"
    REACTION = "reaction"
    MOVEMENT = "movement"
    FREE_INTERACTION = "free_interaction"


def calculate_modifier(score: int) -> int:
    """Calculate ability modifier from score: (score - 10) // 2."""
    return (score - 10) // 2


def calculate_proficiency_bonus(level: int) -> int:
    """Calculate proficiency bonus from level: floor((level - 1) / 4) + 2."""
    return ((level - 1) // 4) + 2
