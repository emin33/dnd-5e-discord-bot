"""Character-related game logic."""

from .creation import (
    AbilityScoreMethod,
    AbilityScoreRolls,
    CharacterCreationState,
    CharacterCreator,
    PointBuyState,
    get_creator,
)
from .leveling import (
    LevelingManager,
    LevelUpResult,
    XP_THRESHOLDS,
    can_level_up,
    get_level_for_xp,
    get_leveling_manager,
    get_xp_for_next_level,
    get_xp_progress,
    level_up,
)

__all__ = [
    # Creation
    "AbilityScoreMethod",
    "AbilityScoreRolls",
    "CharacterCreationState",
    "CharacterCreator",
    "PointBuyState",
    "get_creator",
    # Leveling
    "LevelingManager",
    "LevelUpResult",
    "XP_THRESHOLDS",
    "can_level_up",
    "get_level_for_xp",
    "get_leveling_manager",
    "get_xp_for_next_level",
    "get_xp_progress",
    "level_up",
]
