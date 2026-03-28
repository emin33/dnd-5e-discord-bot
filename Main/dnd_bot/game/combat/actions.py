"""Combat action and result data models."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

from ..mechanics.dice import DiceRoll
from ...models import Condition


class CombatActionType(str, Enum):
    """Types of actions that can be taken in combat."""
    ATTACK = "attack"
    CAST_SPELL = "cast_spell"
    DASH = "dash"
    DISENGAGE = "disengage"
    DODGE = "dodge"
    HELP = "help"
    HIDE = "hide"
    READY = "ready"
    USE_ITEM = "use_item"
    SHOVE = "shove"
    GRAPPLE = "grapple"
    # Bonus actions
    OFFHAND_ATTACK = "offhand_attack"
    # Special
    END_TURN = "end_turn"


@dataclass
class WeaponStats:
    """Stats for a weapon being used in an attack."""
    name: str
    damage_dice: str  # e.g., "1d8", "2d6"
    damage_type: str  # e.g., "slashing", "piercing", "bludgeoning"
    attack_bonus: int = 0  # Magic weapon bonus
    is_melee: bool = True
    is_finesse: bool = False
    is_ranged: bool = False
    is_thrown: bool = False
    range_normal: int = 5  # In feet
    range_long: int = 5
    properties: list[str] = field(default_factory=list)


@dataclass
class CombatAction:
    """
    Structured representation of a combat action.

    This is the input to the action execution system.
    """
    action_type: CombatActionType
    combatant_id: str
    target_ids: list[str] = field(default_factory=list)

    # Action-specific data
    weapon_index: Optional[str] = None
    spell_index: Optional[str] = None
    slot_level: Optional[int] = None
    ability_name: Optional[str] = None  # For monster abilities

    # Resource usage
    uses_action: bool = True
    uses_bonus_action: bool = False
    uses_reaction: bool = False
    uses_movement: int = 0

    # Modifiers (can be set by conditions, Help action, etc.)
    advantage: bool = False
    disadvantage: bool = False
    advantage_reason: Optional[str] = None
    disadvantage_reason: Optional[str] = None

    # Ready action specifics
    ready_trigger: Optional[str] = None
    ready_action: Optional["CombatAction"] = None


@dataclass
class ActionResult:
    """
    Result of executing a combat action.

    This captures all mechanical outcomes and is passed to the Narrator
    for dramatization. The Narrator is constrained to describe only
    what actually happened according to these results.
    """
    action: CombatAction
    success: bool
    error: Optional[str] = None

    # Roll details
    attack_roll: Optional[DiceRoll] = None
    damage_roll: Optional[DiceRoll] = None
    save_rolls: dict[str, DiceRoll] = field(default_factory=dict)  # target_id -> roll
    skill_roll: Optional[DiceRoll] = None
    concentration_roll: Optional[DiceRoll] = None

    # Outcomes per target
    damage_dealt: dict[str, int] = field(default_factory=dict)  # target_id -> damage
    healing_done: dict[str, int] = field(default_factory=dict)  # target_id -> healing
    damage_type: Optional[str] = None
    damage_resisted: dict[str, str] = field(default_factory=dict)  # target_id -> "resistance"/"immunity"/"vulnerability"

    # Conditions
    conditions_applied: dict[str, list[Condition]] = field(default_factory=dict)  # target_id -> conditions
    conditions_removed: dict[str, list[Condition]] = field(default_factory=dict)

    # Narrative hooks
    critical_hit: bool = False
    critical_miss: bool = False
    target_ac: Optional[int] = None  # For narration context

    # Target state changes
    unconscious_targets: list[str] = field(default_factory=list)
    killed_targets: list[str] = field(default_factory=list)
    stabilized_targets: list[str] = field(default_factory=list)

    # Special outcomes
    concentration_broken: bool = False
    concentration_maintained: bool = False
    spell_effect: Optional[str] = None
    zone_changes: list[str] = field(default_factory=list)  # Narrative descriptions

    # Resource consumption
    spell_slot_used: Optional[int] = None
    item_consumed: Optional[str] = None

    def get_summary(self) -> str:
        """Get a brief summary of the action result for logging."""
        action_name = self.action.action_type.value
        if not self.success:
            return f"{action_name}: FAILED - {self.error or 'unknown'}"

        parts = [f"{action_name}: SUCCESS"]

        if self.attack_roll:
            hit_miss = "HIT" if self.damage_dealt else "MISS"
            crit = " (CRIT!)" if self.critical_hit else ""
            parts.append(f"Attack {self.attack_roll.total}{crit} = {hit_miss}")

        if self.damage_dealt:
            total_dmg = sum(self.damage_dealt.values())
            parts.append(f"Damage: {total_dmg}")

        if self.healing_done:
            total_heal = sum(self.healing_done.values())
            parts.append(f"Healing: {total_heal}")

        if self.killed_targets:
            parts.append(f"Killed: {', '.join(self.killed_targets)}")
        elif self.unconscious_targets:
            parts.append(f"KO'd: {', '.join(self.unconscious_targets)}")

        return " | ".join(parts)


@dataclass
class TurnContext:
    """
    Context for a combatant's turn.

    Passed to the UI to display available options.
    """
    combatant_id: str
    combatant_name: str
    is_player: bool

    # Resources available
    has_action: bool
    has_bonus_action: bool
    has_reaction: bool
    movement_remaining: int

    # Character info (for players)
    character_id: Optional[str] = None
    equipped_weapons: list[WeaponStats] = field(default_factory=list)
    available_spells: list[str] = field(default_factory=list)
    spell_slots: dict[int, int] = field(default_factory=dict)  # level -> slots remaining

    # Monster info (for NPCs)
    monster_index: Optional[str] = None
    monster_actions: list[str] = field(default_factory=list)
    recharge_abilities: list[str] = field(default_factory=list)

    # Combat state
    hp_current: int = 0
    hp_max: int = 0
    armor_class: int = 10
    conditions: list[str] = field(default_factory=list)

    # Position
    in_melee_with: list[str] = field(default_factory=list)
    is_concentrating: bool = False
    concentration_spell: Optional[str] = None


@dataclass
class MultiAttackAction:
    """Represents a multiattack sequence for monsters."""
    name: str
    attacks: list[CombatAction]
    description: str = ""


@dataclass
class ReadyActionTrigger:
    """A readied action waiting for its trigger."""
    combatant_id: str
    action: CombatAction
    trigger_description: str
    expires_round: int
