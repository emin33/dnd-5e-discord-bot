"""Combat system for D&D 5e."""

from .manager import (
    CombatManager,
    get_combat_for_channel,
    set_combat_for_channel,
    clear_combat_for_channel,
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
    clear_coordinator,
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
    "set_combat_for_channel",
    "clear_combat_for_channel",
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
    "clear_coordinator",
    # NPC AI
    "NPCCombatBrain",
    "CreatureBehavior",
    "get_npc_brain",
]
