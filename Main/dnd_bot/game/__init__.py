"""Game logic and mechanics for D&D 5e."""

from .mechanics import DiceRoll, DiceRoller, get_roller, roll
from .combat.manager import CombatManager, get_combat_for_channel, get_combat_by_key, set_combat_for_channel, set_combat_by_key
from .frontend import GameFrontend, GameEvent, GameEventType
from .magic import SpellcastingManager, get_spellcasting_manager

__all__ = [
    "DiceRoll",
    "DiceRoller",
    "get_roller",
    "roll",
    "CombatManager",
    "get_combat_for_channel",
    "get_combat_by_key",
    "set_combat_for_channel",
    "set_combat_by_key",
    "GameFrontend",
    "GameEvent",
    "GameEventType",
    "SpellcastingManager",
    "get_spellcasting_manager",
]
