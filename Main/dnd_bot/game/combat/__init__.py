"""Combat system for D&D 5e."""

from .manager import (
    CombatManager,
    get_combat_for_channel,
    get_combat_by_key,
    set_combat_for_channel,
    set_combat_by_key,
    clear_combat_for_channel,
    clear_combat_by_key,
)

from .actions import (
    CombatAction,
    CombatActionType,
    ActionResult,
    TurnContext,
    WeaponStats,
)

from .zones import (
    ZoneTracker,
    CombatZone,
)

from .coordinator import (
    CombatTurnCoordinator,
    get_coordinator,
    get_coordinator_for_channel,
    get_coordinator_by_key,
    clear_coordinator,
    clear_coordinator_by_key,
)

from .npc_brain import (
    NPCCombatBrain,
    CreatureBehavior,
    get_npc_brain,
)

__all__ = [
    # Manager
    "CombatManager",
    "get_combat_for_channel",
    "get_combat_by_key",
    "set_combat_for_channel",
    "set_combat_by_key",
    "clear_combat_for_channel",
    "clear_combat_by_key",
    # Actions
    "CombatAction",
    "CombatActionType",
    "ActionResult",
    "TurnContext",
    "WeaponStats",
    # Zones
    "ZoneTracker",
    "CombatZone",
    # Coordinator
    "CombatTurnCoordinator",
    "get_coordinator",
    "get_coordinator_for_channel",
    "get_coordinator_by_key",
    "clear_coordinator",
    "clear_coordinator_by_key",
    # NPC AI
    "NPCCombatBrain",
    "CreatureBehavior",
    "get_npc_brain",
]
