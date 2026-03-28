"""NPC Combat AI - decides actions for monster combatants.

This module provides AI decision-making for NPCs in combat,
using monster stat blocks to determine available actions and
evaluating tactical situations to choose optimal targets.
"""

import random
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import structlog

from .actions import CombatAction, CombatActionType, WeaponStats
from .zones import ZoneTracker
from ...data.srd import get_srd
from ...models import Combatant, Combat

logger = structlog.get_logger()


class CreatureBehavior(str, Enum):
    """AI behavior patterns for creatures."""

    AGGRESSIVE = "aggressive"  # Maximize damage, focus on threats
    TACTICAL = "tactical"  # Focus casters/wounded, use positioning
    DEFENSIVE = "defensive"  # Protect allies, stay near leader
    COWARDLY = "cowardly"  # Flee when hurt, target weak
    MINDLESS = "mindless"  # Attack nearest, no strategy


@dataclass
class MonsterAction:
    """Parsed action from a monster stat block."""

    name: str
    is_attack: bool = False
    is_multiattack: bool = False
    attack_bonus: int = 0
    damage_dice: str = ""
    damage_type: str = ""
    damage_bonus: int = 0
    is_melee: bool = True
    is_ranged: bool = False
    reach: int = 5
    range_normal: int = 0
    range_long: int = 0
    targets_count: int = 1
    description: str = ""
    is_recharge: bool = False
    recharge_on: int = 6  # Recharges on this or higher

    # For multiattack
    multiattack_actions: list[str] = field(default_factory=list)

    # For special abilities
    save_dc: int = 0
    save_ability: str = ""
    aoe: bool = False  # Area of effect

    def get_weapon_stats(self) -> WeaponStats:
        """Convert to WeaponStats for attack execution."""
        return WeaponStats(
            name=self.name,
            damage_dice=self.damage_dice,
            damage_type=self.damage_type,
            attack_bonus=self.attack_bonus,
            is_melee=self.is_melee,
            is_ranged=self.is_ranged,
            range_normal=self.range_normal if self.is_ranged else self.reach,
            range_long=self.range_long,
        )


@dataclass
class TargetEvaluation:
    """Evaluation of a potential target."""

    combatant_id: str
    combatant_name: str
    score: float = 0.0

    # Factors
    is_wounded: bool = False
    hp_percentage: float = 1.0
    is_spellcaster: bool = False
    is_in_melee: bool = False
    is_concentrating: bool = False
    threat_level: float = 0.5


class NPCCombatBrain:
    """
    AI decision-making for NPC combatants.

    Uses monster stat blocks to determine available actions
    and evaluates the tactical situation to choose targets.
    """

    def __init__(self):
        self.srd = get_srd()

        # Behavior defaults by creature type
        self._default_behaviors = {
            "beast": CreatureBehavior.AGGRESSIVE,
            "humanoid": CreatureBehavior.TACTICAL,
            "undead": CreatureBehavior.MINDLESS,
            "construct": CreatureBehavior.MINDLESS,
            "dragon": CreatureBehavior.TACTICAL,
            "fiend": CreatureBehavior.AGGRESSIVE,
            "celestial": CreatureBehavior.DEFENSIVE,
            "aberration": CreatureBehavior.AGGRESSIVE,
            "ooze": CreatureBehavior.MINDLESS,
            "plant": CreatureBehavior.MINDLESS,
        }

    async def decide_action(
        self,
        combatant: Combatant,
        combat: Combat,
        zones: ZoneTracker,
    ) -> CombatAction:
        """
        Decide the best action for an NPC combatant.

        Returns a CombatAction ready for execution.
        """
        monster = self._get_monster_data(combatant)
        if not monster:
            return self._build_basic_attack(combatant, combat)

        behavior = self._get_behavior(monster)
        actions = self._parse_monster_actions(monster)

        # Get potential targets
        targets = self._evaluate_targets(combatant, combat, zones, behavior)
        if not targets:
            # No valid targets - end turn
            return CombatAction(
                action_type=CombatActionType.END_TURN,
                combatant_id=combatant.id,
                uses_action=False,
            )

        # Priority 1: Available recharge abilities
        recharge_action = self._check_recharge_abilities(
            combatant, actions, targets, behavior
        )
        if recharge_action:
            return recharge_action

        # Priority 2: Multiattack if available
        multiattack = self._find_multiattack(actions)
        if multiattack:
            return self._build_multiattack_action(
                combatant, multiattack, actions, targets[0]
            )

        # Priority 3: Best single attack
        best_action = self._select_best_action(
            combatant, actions, targets, zones, behavior
        )

        if best_action and best_action.is_attack:
            return self._build_attack_action(
                combatant, best_action, targets[0], zones
            )

        # Fallback: basic attack
        return self._build_basic_attack(combatant, combat)

    def _get_monster_data(self, combatant: Combatant) -> Optional[dict]:
        """Get monster stat block from SRD."""
        if not combatant.monster_index:
            return None
        return self.srd.get_monster(combatant.monster_index)

    def _get_behavior(self, monster: dict) -> CreatureBehavior:
        """Determine creature behavior based on type and intelligence."""
        creature_type = monster.get("type", "").lower()

        # Check intelligence for tactical capability
        int_score = monster.get("intelligence", 10)
        wis_score = monster.get("wisdom", 10)

        if int_score <= 4:
            return CreatureBehavior.MINDLESS

        if int_score >= 14 and wis_score >= 12:
            return CreatureBehavior.TACTICAL

        # Default by type
        return self._default_behaviors.get(
            creature_type, CreatureBehavior.AGGRESSIVE
        )

    def _parse_monster_actions(self, monster: dict) -> list[MonsterAction]:
        """Parse all actions from a monster stat block."""
        actions = []

        for action_data in monster.get("actions", []):
            action = self._parse_single_action(action_data)
            if action:
                actions.append(action)

        return actions

    def _parse_single_action(self, action_data: dict) -> Optional[MonsterAction]:
        """Parse a single action from stat block data."""
        name = action_data.get("name", "")
        desc = action_data.get("desc", "").lower()

        action = MonsterAction(name=name, description=action_data.get("desc", ""))

        # Check for multiattack
        if "multiattack" in name.lower():
            action.is_multiattack = True
            # Parse multiattack description for component attacks
            action.multiattack_actions = self._parse_multiattack_desc(desc)
            return action

        # Check for attack
        if "attack_bonus" in action_data:
            action.is_attack = True
            action.attack_bonus = action_data.get("attack_bonus", 0)

            # Determine melee vs ranged
            if "melee" in desc:
                action.is_melee = True
                # Parse reach
                if "reach" in desc:
                    try:
                        reach_idx = desc.index("reach")
                        reach_str = desc[reach_idx:reach_idx + 20]
                        for word in reach_str.split():
                            if word.rstrip("ft.,").isdigit():
                                action.reach = int(word.rstrip("ft.,"))
                                break
                    except (ValueError, IndexError):
                        pass

            if "ranged" in desc or "range" in desc:
                action.is_ranged = True
                action.is_melee = "melee" in desc  # Could be both (thrown)

            # Parse damage
            if "damage" in action_data:
                damages = action_data["damage"]
                if damages and isinstance(damages, list):
                    first = damages[0]
                    action.damage_dice = first.get("damage_dice", "1d6")
                    action.damage_bonus = first.get("damage_bonus", 0)
                    dtype = first.get("damage_type", {})
                    if isinstance(dtype, dict):
                        action.damage_type = dtype.get("name", "").lower()
                    else:
                        action.damage_type = str(dtype).lower()

        # Check for recharge
        if "usage" in action_data:
            usage = action_data["usage"]
            if usage.get("type") == "recharge on roll":
                action.is_recharge = True
                action.recharge_on = usage.get("min_value", 6)

        # Check for save DC
        if "dc" in action_data:
            dc_data = action_data["dc"]
            action.save_dc = dc_data.get("dc_value", 0)
            dc_type = dc_data.get("dc_type", {})
            if isinstance(dc_type, dict):
                action.save_ability = dc_type.get("index", "")

        return action

    def _parse_multiattack_desc(self, desc: str) -> list[str]:
        """Parse multiattack description to find component attacks."""
        attacks = []

        # Common patterns: "two X attacks" or "one X and one Y"
        # This is a simplified parser - could be enhanced

        words = desc.split()
        for i, word in enumerate(words):
            if word in ("two", "three", "four"):
                count = {"two": 2, "three": 3, "four": 4}[word]
                if i + 1 < len(words):
                    attack_name = words[i + 1].rstrip("s.,")
                    for _ in range(count):
                        attacks.append(attack_name)

        return attacks if attacks else ["attack", "attack"]

    def _find_multiattack(self, actions: list[MonsterAction]) -> Optional[MonsterAction]:
        """Find multiattack action if available."""
        for action in actions:
            if action.is_multiattack:
                return action
        return None

    def _evaluate_targets(
        self,
        combatant: Combatant,
        combat: Combat,
        zones: ZoneTracker,
        behavior: CreatureBehavior,
    ) -> list[TargetEvaluation]:
        """Evaluate all potential targets and return sorted by priority."""
        evaluations = []

        for target in combat.combatants:
            # Skip invalid targets
            if not target.is_active or not target.is_conscious:
                continue
            if target.id == combatant.id:
                continue
            if target.is_player == combatant.is_player:
                continue  # Skip allies

            eval_target = TargetEvaluation(
                combatant_id=target.id,
                combatant_name=target.name,
            )

            # HP percentage (lower = more wounded)
            eval_target.hp_percentage = target.hp_current / max(1, target.hp_max)
            eval_target.is_wounded = eval_target.hp_percentage < 0.5

            # Check if in melee
            eval_target.is_in_melee = zones.is_in_melee_with(combatant.id, target.id)

            # Score based on behavior
            score = 0.0

            if behavior == CreatureBehavior.MINDLESS:
                # Nearest target (prefer already in melee)
                score = 10.0 if eval_target.is_in_melee else 5.0

            elif behavior == CreatureBehavior.AGGRESSIVE:
                # Maximum damage potential (wounded targets)
                score = 5.0
                if eval_target.is_wounded:
                    score += 5.0 * (1.0 - eval_target.hp_percentage)
                if eval_target.is_in_melee:
                    score += 2.0

            elif behavior == CreatureBehavior.TACTICAL:
                # Focus casters, concentration, wounded
                score = 5.0
                if eval_target.is_wounded:
                    score += 3.0 * (1.0 - eval_target.hp_percentage)
                # Would check for spellcaster if we had class info
                # For now, boost priority for low HP targets
                if eval_target.hp_percentage < 0.25:
                    score += 5.0  # Finish them off

            elif behavior == CreatureBehavior.COWARDLY:
                # Target weakest
                score = 10.0 * (1.0 - eval_target.hp_percentage)

            elif behavior == CreatureBehavior.DEFENSIVE:
                # Target whoever is closest to allies
                score = 5.0
                if eval_target.is_in_melee:
                    score += 3.0

            eval_target.score = score
            evaluations.append(eval_target)

        # Sort by score descending
        evaluations.sort(key=lambda e: e.score, reverse=True)

        return evaluations

    def _check_recharge_abilities(
        self,
        combatant: Combatant,
        actions: list[MonsterAction],
        targets: list[TargetEvaluation],
        behavior: CreatureBehavior,
    ) -> Optional[CombatAction]:
        """Check if any recharge abilities should be used."""
        available = combatant.get_available_recharge_abilities()

        if not available:
            return None

        # Find matching action
        for ability in available:
            for action in actions:
                if action.name.lower() == ability.name.lower() and action.is_recharge:
                    # Evaluate if worth using
                    if self._should_use_recharge(action, targets, behavior):
                        # Build action for this ability
                        target_ids = [targets[0].combatant_id] if targets else []

                        # NOTE: Do NOT mark ability as used here — the coordinator
                        # marks it after successful execution to avoid losing the
                        # ability if the action fails or is cancelled.
                        return CombatAction(
                            action_type=CombatActionType.ATTACK,
                            combatant_id=combatant.id,
                            target_ids=target_ids,
                            ability_name=action.name,
                        )

        return None

    def _should_use_recharge(
        self,
        action: MonsterAction,
        targets: list[TargetEvaluation],
        behavior: CreatureBehavior,
    ) -> bool:
        """Determine if a recharge ability should be used now."""
        if not targets:
            return False

        # Aggressive: always use if available
        if behavior == CreatureBehavior.AGGRESSIVE:
            return True

        # Tactical: use when multiple targets are wounded
        if behavior == CreatureBehavior.TACTICAL:
            wounded_count = sum(1 for t in targets if t.is_wounded)
            return wounded_count >= 2 or len(targets) >= 3

        # Others: 50% chance
        return random.random() > 0.5

    def _select_best_action(
        self,
        combatant: Combatant,
        actions: list[MonsterAction],
        targets: list[TargetEvaluation],
        zones: ZoneTracker,
        behavior: CreatureBehavior,
    ) -> Optional[MonsterAction]:
        """Select the best single action to use."""
        if not targets:
            return None

        primary_target = targets[0]
        in_melee = zones.is_in_melee_with(combatant.id, primary_target.combatant_id)

        scored_actions = []

        for action in actions:
            if action.is_multiattack:
                continue  # Handled separately

            score = 0.0

            if action.is_attack:
                # Prefer melee if in melee range
                if action.is_melee and in_melee:
                    score += 5.0
                elif action.is_ranged and not in_melee:
                    score += 4.0
                elif action.is_melee:
                    score += 2.0  # Can close distance
                elif action.is_ranged:
                    score += 1.0  # Disadvantage in melee

                # Bonus for higher damage
                if action.damage_dice:
                    # Rough damage estimate
                    dice_parts = action.damage_dice.split("d")
                    if len(dice_parts) == 2:
                        try:
                            num = int(dice_parts[0]) if dice_parts[0] else 1
                            die = int(dice_parts[1].split("+")[0])
                            avg_damage = num * (die + 1) / 2 + action.damage_bonus
                            score += avg_damage / 10.0
                        except ValueError:
                            pass

            scored_actions.append((action, score))

        if not scored_actions:
            return None

        # Sort by score and return best
        scored_actions.sort(key=lambda x: x[1], reverse=True)
        return scored_actions[0][0]

    def _build_attack_action(
        self,
        combatant: Combatant,
        action: MonsterAction,
        target: TargetEvaluation,
        zones: ZoneTracker,
    ) -> CombatAction:
        """Build a CombatAction for an attack."""
        # Check if we need to engage melee
        needs_engage = (
            action.is_melee
            and not zones.is_in_melee_with(combatant.id, target.combatant_id)
        )

        # Build action
        return CombatAction(
            action_type=CombatActionType.ATTACK,
            combatant_id=combatant.id,
            target_ids=[target.combatant_id],
            ability_name=action.name,
            uses_movement=30 if needs_engage else 0,  # Rough movement cost
        )

    def _build_multiattack_action(
        self,
        combatant: Combatant,
        multiattack: MonsterAction,
        all_actions: list[MonsterAction],
        primary_target: TargetEvaluation,
    ) -> CombatAction:
        """Build a CombatAction for multiattack."""
        # For now, just use primary target for all attacks
        # Could enhance to split among targets

        # Find the attack action to use
        attack_action = None
        for action in all_actions:
            if action.is_attack and not action.is_multiattack:
                attack_action = action
                break

        return CombatAction(
            action_type=CombatActionType.ATTACK,
            combatant_id=combatant.id,
            target_ids=[primary_target.combatant_id],
            ability_name=attack_action.name if attack_action else "Attack",
        )

    def _build_basic_attack(
        self,
        combatant: Combatant,
        combat: Combat,
    ) -> CombatAction:
        """Build a basic unarmed attack as fallback."""
        # Find first valid enemy target
        for target in combat.combatants:
            if (
                target.is_active
                and target.is_conscious
                and target.is_player != combatant.is_player
            ):
                return CombatAction(
                    action_type=CombatActionType.ATTACK,
                    combatant_id=combatant.id,
                    target_ids=[target.id],
                )

        # No targets - end turn
        return CombatAction(
            action_type=CombatActionType.END_TURN,
            combatant_id=combatant.id,
            uses_action=False,
        )

    def roll_recharge(self, combatant: Combatant) -> list[str]:
        """
        Roll recharge for all unavailable abilities at start of turn.

        Returns list of ability names that recharged.
        """
        recharged = []

        for ability in combatant.get_unavailable_recharge_abilities():
            roll = random.randint(1, 6)
            if roll >= ability.recharge_on:
                ability.is_available = True
                recharged.append(ability.name)
                logger.info(
                    "ability_recharged",
                    combatant=combatant.name,
                    ability=ability.name,
                    roll=roll,
                )

        return recharged


# Global NPC brain instance
_npc_brain: Optional[NPCCombatBrain] = None


def get_npc_brain() -> NPCCombatBrain:
    """Get the global NPC brain instance."""
    global _npc_brain
    if _npc_brain is None:
        _npc_brain = NPCCombatBrain()
    return _npc_brain
