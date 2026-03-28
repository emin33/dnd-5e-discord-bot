"""Combat Turn Coordinator - orchestrates structured combat turns.

This is the core of the combat system, sitting between:
- CombatManager (low-level state and mechanics)
- Discord UI (buttons and menus)
- Narrator (dramatic descriptions)

Flow:
1. start_turn() - Prepare turn context, show action menu
2. execute_action() - Run mechanics, roll dice, apply effects
3. narrate_result() - Pass result to narrator for description
4. end_turn() - Advance to next combatant
"""

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING
import structlog

from .manager import CombatManager
from .zones import ZoneTracker
from .actions import (
    CombatAction,
    CombatActionType,
    ActionResult,
    TurnContext,
    WeaponStats,
)
from ..mechanics.dice import get_roller, DiceRoll
from ..mechanics.conditions import ConditionResolver
from ...models import Character, Combatant, Combat, CombatState, Condition, AbilityScore
from ...data.srd import get_srd
from ...data.repositories import get_character_repo, get_inventory_repo

if TYPE_CHECKING:
    from ..session import GameSession
    from ...llm.brains.narrator import NarratorBrain

from ...llm.brains.base import BrainContext
from ...models.combat import CombatEffect

logger = structlog.get_logger()

# Spell → condition mapping for spells that apply conditions on failed saves.
# Format: spell_index → (Condition, duration_rounds, save_ends_on_success)
# Duration of 10 rounds ≈ 1 minute in D&D 5e.
SPELL_CONDITION_MAP: dict[str, tuple[Condition, int, bool]] = {
    "hold-person": (Condition.PARALYZED, 10, True),
    "hold-monster": (Condition.PARALYZED, 10, True),
    "blindness-deafness": (Condition.BLINDED, 10, True),
    "web": (Condition.RESTRAINED, 10, True),
    "entangle": (Condition.RESTRAINED, 10, True),
    "tashas-hideous-laughter": (Condition.PRONE, 10, True),
    "command": (Condition.PRONE, 1, False),  # "grovel" variant
    "hideous-laughter": (Condition.PRONE, 10, True),  # Alternate index
    "ray-of-enfeeblement": (Condition.EXHAUSTION, 10, True),
    "contagion": (Condition.POISONED, 70, False),  # 7 days
    "flesh-to-stone": (Condition.RESTRAINED, 10, True),  # First stage
    "eyebite": (Condition.UNCONSCIOUS, 10, True),
    "banishment": (Condition.INCAPACITATED, 10, False),
    "phantasmal-killer": (Condition.FRIGHTENED, 10, True),
    "fear": (Condition.FRIGHTENED, 10, True),
    "cause-fear": (Condition.FRIGHTENED, 10, True),
}


@dataclass
class TurnEndResult:
    """Result of ending a turn."""
    next_combatant_id: str
    next_combatant_name: str
    next_is_player: bool
    round_advanced: bool
    new_round: int
    effect_messages: list[str] = field(default_factory=list)


class CombatTurnCoordinator:
    """
    Orchestrates combat turns with structured actions.

    Responsibilities:
    - Build turn context with available actions
    - Execute chosen actions through mechanics layer
    - Track zone positioning
    - Build structured results for narration
    - Handle NPC turn automation
    """

    def __init__(
        self,
        manager: CombatManager,
        session: Optional["GameSession"] = None,
    ):
        self.manager = manager
        self.session = session
        self.zone_tracker = ZoneTracker()
        self.roller = get_roller()
        self.srd = get_srd()

        # Character cache for player combatants
        self._character_cache: dict[str, Character] = {}

        # Narrator brain (set externally)
        self._narrator: Optional["NarratorBrain"] = None

    def set_narrator(self, narrator: "NarratorBrain") -> None:
        """Set the narrator for result descriptions."""
        self._narrator = narrator

    # ==================== Turn Management ====================

    async def start_turn(self, combatant: Combatant) -> TurnContext:
        """
        Prepare context for a combatant's turn.

        Resets turn resources and builds available options.
        Handles surprise: surprised creatures can't act on their first turn.
        """
        # Reset turn-based states
        self.zone_tracker.on_turn_start(combatant.id)
        combatant.turn_resources.reset_for_new_turn(combatant.speed)

        # SURPRISE: Surprised creatures can't take actions/bonus actions/reactions
        # Their surprise ends at the end of their first turn
        if combatant.is_surprised:
            combatant.turn_resources.action = False
            combatant.turn_resources.bonus_action = False
            combatant.turn_resources.reaction = False
            combatant.turn_resources.movement = 0
            logger.info(
                "combatant_surprised",
                combatant=combatant.name,
                message="Cannot act this turn due to surprise",
            )

        # Process start-of-turn effects
        effect_results = self.manager.process_start_of_turn_effects(combatant)
        for result in effect_results:
            if result.damage_dealt > 0:
                logger.info(
                    "start_of_turn_damage",
                    combatant=combatant.name,
                    damage=result.damage_dealt,
                    source=result.effect.name,
                )

        # Build turn context
        context = await self._build_turn_context(combatant)

        logger.info(
            "turn_started",
            combatant=combatant.name,
            is_player=combatant.is_player,
            hp=f"{combatant.hp_current}/{combatant.hp_max}",
            surprised=combatant.is_surprised,
        )

        return context

    async def _build_turn_context(self, combatant: Combatant) -> TurnContext:
        """Build the context object for a combatant's turn."""
        resources = combatant.turn_resources

        context = TurnContext(
            combatant_id=combatant.id,
            combatant_name=combatant.name,
            is_player=combatant.is_player,
            has_action=resources.action,
            has_bonus_action=resources.bonus_action,
            has_reaction=resources.reaction,
            movement_remaining=resources.movement,
            hp_current=combatant.hp_current,
            hp_max=combatant.hp_max,
            armor_class=combatant.armor_class,
            conditions=[c.value for c in self._get_combatant_conditions(combatant)],
            in_melee_with=self.zone_tracker.get_melee_targets(combatant.id),
        )

        if combatant.is_player and combatant.character_id:
            # Load player-specific data
            character = await self._get_character(combatant.character_id)
            if character:
                context.character_id = character.id
                context.equipped_weapons = await self._get_equipped_weapons(character)
                context.available_spells = self._get_available_spells(character)
                context.spell_slots = self._get_spell_slots(character)
                context.is_concentrating = character.is_concentrating
                context.concentration_spell = character.concentration_spell
        else:
            # Monster-specific data
            context.monster_index = combatant.monster_index
            context.monster_actions = self._get_monster_action_names(combatant)
            context.recharge_abilities = [
                a.name for a in combatant.recharge_abilities
                if a.is_available
            ]

        return context

    async def end_turn(self, combatant: Combatant) -> TurnEndResult:
        """End the current turn and advance to next combatant."""
        # Process end-of-turn effects
        effect_results = self.manager.process_end_of_turn_effects(combatant)
        effect_messages = []
        for result in effect_results:
            effect_messages.extend(result.messages)

        # SURPRISE: Surprise ends at the end of the creature's first turn
        if combatant.is_surprised:
            combatant.is_surprised = False
            logger.info("surprise_ended", combatant=combatant.name)

        self.zone_tracker.on_turn_end(combatant.id)

        # Advance turn
        next_combatant, _, _, _ = self.manager.next_turn()

        round_advanced = next_combatant.turn_order == 0

        return TurnEndResult(
            next_combatant_id=next_combatant.id,
            next_combatant_name=next_combatant.name,
            next_is_player=next_combatant.is_player,
            round_advanced=round_advanced,
            new_round=self.manager.combat.current_round,
            effect_messages=effect_messages,
        )

    # ==================== Action Execution ====================

    async def execute_action(self, action: CombatAction) -> ActionResult:
        """
        Execute a combat action through the mechanics layer.

        This is the main entry point for all combat actions.
        """
        combatant = self.manager.get_combatant(action.combatant_id)
        if not combatant:
            return ActionResult(
                action=action,
                success=False,
                error=f"Combatant not found",
            )

        # Check if conditions prevent this action
        blocking_condition = self._get_action_blocking_condition(combatant)
        if blocking_condition:
            return ActionResult(
                action=action,
                success=False,
                error=f"Cannot act while {blocking_condition.value}",
            )

        # Validate and consume resources
        if action.uses_action:
            if not self.manager.use_action(action.combatant_id):
                return ActionResult(
                    action=action,
                    success=False,
                    error="No action available",
                )

        if action.uses_bonus_action:
            if not self.manager.use_bonus_action(action.combatant_id):
                return ActionResult(
                    action=action,
                    success=False,
                    error="No bonus action available",
                )

        if action.uses_reaction:
            if not self.manager.use_reaction(action.combatant_id):
                return ActionResult(
                    action=action,
                    success=False,
                    error="No reaction available",
                )

        # Route to specific handler
        try:
            if action.action_type == CombatActionType.ATTACK:
                result = await self._execute_attack(action, combatant)
                # Mark recharge ability as used AFTER successful execution
                if action.ability_name and not combatant.is_player:
                    combatant.use_recharge_ability(action.ability_name)
                return result
            elif action.action_type == CombatActionType.CAST_SPELL:
                return await self._execute_spell(action, combatant)
            elif action.action_type == CombatActionType.DASH:
                return self._execute_dash(action, combatant)
            elif action.action_type == CombatActionType.DISENGAGE:
                return self._execute_disengage(action, combatant)
            elif action.action_type == CombatActionType.DODGE:
                return self._execute_dodge(action, combatant)
            elif action.action_type == CombatActionType.HELP:
                return self._execute_help(action, combatant)
            elif action.action_type == CombatActionType.HIDE:
                return await self._execute_hide(action, combatant)
            elif action.action_type == CombatActionType.END_TURN:
                return ActionResult(action=action, success=True)
            else:
                return ActionResult(
                    action=action,
                    success=False,
                    error=f"Unknown action type: {action.action_type}",
                )
        except Exception as e:
            logger.error("action_execution_failed", error=str(e), action=action.action_type)
            return ActionResult(
                action=action,
                success=False,
                error=str(e),
            )

    # ==================== Attack Execution ====================

    async def _execute_attack(
        self,
        action: CombatAction,
        attacker: Combatant,
    ) -> ActionResult:
        """Execute a weapon attack with real dice rolls."""
        if not action.target_ids:
            return ActionResult(action=action, success=False, error="No target specified")

        target_id = action.target_ids[0]
        target = self.manager.get_combatant(target_id)
        if not target:
            return ActionResult(action=action, success=False, error="Target not found")

        if not target.is_active or not target.is_conscious:
            return ActionResult(action=action, success=False, error="Target is not valid")

        # Get weapon stats
        weapon = await self._get_weapon_for_attack(attacker, action.weapon_index)

        # Check melee validity and engage if needed
        if weapon.is_melee and not self.zone_tracker.is_in_melee_with(attacker.id, target_id):
            # Auto-engage if movement available
            self.zone_tracker.engage_melee(attacker.id, target_id)

        # Calculate attack modifier
        attack_mod = await self._calculate_attack_modifier(attacker, weapon)

        # Determine advantage/disadvantage
        adv, dis = self._get_attack_advantage(attacker, target, weapon)

        # Apply action-level modifiers
        if action.advantage:
            adv = True
        if action.disadvantage:
            dis = True

        # Roll attack
        attack_roll = self.roller.roll_attack(
            modifier=attack_mod,
            advantage=adv and not dis,
            disadvantage=dis and not adv,
        )

        result = ActionResult(
            action=action,
            success=False,  # Will be set based on hit
            attack_roll=attack_roll,
            damage_type=weapon.damage_type,
        )

        # Determine hit
        target_ac = self.manager.get_effective_ac(target_id)
        result.target_ac = target_ac

        is_crit = attack_roll.natural_20
        is_fumble = attack_roll.natural_1

        # Auto-crit if target is paralyzed/unconscious and attacker within 5ft
        if not is_crit and self.zone_tracker.is_in_melee_with(attacker.id, target_id):
            target_conditions = self._get_combatant_conditions(target)
            if ConditionResolver.is_auto_crit(target_conditions, attacker_within_5ft=True):
                is_crit = True

        hit = is_crit or (not is_fumble and attack_roll.total >= target_ac)

        result.critical_hit = is_crit
        result.critical_miss = is_fumble
        result.success = hit

        if hit:
            # Roll damage
            damage_roll = self.roller.roll_damage(weapon.damage_dice, critical=is_crit)

            # Add ability modifier to damage
            ability_mod = await self._get_ability_modifier_for_weapon(attacker, weapon)
            total_damage = damage_roll.total + ability_mod

            result.damage_roll = damage_roll

            # Apply damage through combat manager
            actual, unconscious, instant_death, modifier = self.manager.apply_damage(
                target_id,
                total_damage,
                damage_type=weapon.damage_type,
                is_critical=is_crit,
            )

            result.damage_dealt[target_id] = actual
            if modifier != "none":
                result.damage_resisted[target_id] = modifier

            if unconscious:
                result.unconscious_targets.append(target.name)
            if instant_death:
                result.killed_targets.append(target.name)
                # Remove from zone tracking
                self.zone_tracker.remove_combatant(target_id)

            # Check concentration for target
            if actual > 0 and self._is_concentrating(target):
                conc_result = await self._check_concentration(target, actual)
                if not conc_result:
                    result.concentration_broken = True

        logger.info(
            "attack_executed",
            attacker=attacker.name,
            target=target.name,
            roll=attack_roll.total,
            hit=hit,
            damage=result.damage_dealt.get(target_id, 0),
            critical=is_crit,
        )

        return result

    async def _get_weapon_for_attack(
        self,
        combatant: Combatant,
        weapon_index: Optional[str],
    ) -> WeaponStats:
        """Get weapon stats for an attack."""
        if combatant.is_player and combatant.character_id:
            character = await self._get_character(combatant.character_id)
            if character:
                weapons = await self._get_equipped_weapons(character)
                if weapons:
                    if weapon_index:
                        matching = [w for w in weapons if w.name.lower() == weapon_index.lower()]
                        if matching:
                            return matching[0]
                    return weapons[0]

        # Monster attacks
        if combatant.monster_index:
            monster = self.srd.get_monster(combatant.monster_index)
            if monster and monster.get("actions"):
                for action_data in monster["actions"]:
                    if "attack_bonus" in action_data:
                        return self._parse_monster_attack(action_data)

        # Default unarmed strike
        return WeaponStats(
            name="Unarmed Strike",
            damage_dice="1",
            damage_type="bludgeoning",
            is_melee=True,
        )

    def _parse_monster_attack(self, action_data: dict) -> WeaponStats:
        """Parse a monster action into WeaponStats."""
        name = action_data.get("name", "Attack")
        attack_bonus = action_data.get("attack_bonus", 0)

        # Parse damage from description or damage field
        damage_dice = "1d6"
        damage_type = "bludgeoning"

        if "damage" in action_data:
            damages = action_data["damage"]
            if damages and isinstance(damages, list) and len(damages) > 0:
                first_damage = damages[0]
                damage_dice = first_damage.get("damage_dice", "1d6")
                damage_type = first_damage.get("damage_type", {})
                if isinstance(damage_type, dict):
                    damage_type = damage_type.get("name", "bludgeoning").lower()

        # Determine if melee or ranged from description
        desc = action_data.get("desc", "").lower()
        is_melee = "melee" in desc or "reach" in desc
        is_ranged = "ranged" in desc or "range" in desc

        return WeaponStats(
            name=name,
            damage_dice=damage_dice,
            damage_type=damage_type,
            attack_bonus=attack_bonus,
            is_melee=is_melee or not is_ranged,
            is_ranged=is_ranged,
        )

    async def _calculate_attack_modifier(
        self,
        attacker: Combatant,
        weapon: WeaponStats,
    ) -> int:
        """Calculate total attack modifier."""
        # For monsters, use the weapon's attack bonus directly
        if not attacker.is_player:
            return weapon.attack_bonus

        # For players, calculate from ability + proficiency
        ability_mod = await self._get_ability_modifier_for_weapon(attacker, weapon)
        proficiency = attacker.proficiency_bonus if hasattr(attacker, 'proficiency_bonus') else 2

        return ability_mod + proficiency + weapon.attack_bonus

    async def _get_ability_modifier_for_weapon(
        self,
        attacker: Combatant,
        weapon: WeaponStats,
    ) -> int:
        """Get the ability modifier to use for a weapon.

        D&D 5e rules (PHB p.194-195):
        - Melee weapons use STR modifier
        - Ranged weapons use DEX modifier
        - Finesse weapons use the higher of STR or DEX
        - Thrown weapons use STR (or DEX if also finesse)
        """
        # For monsters, this is already baked into attack_bonus
        if not attacker.is_player:
            return 0

        # Fetch character data for ability scores
        if not attacker.character_id:
            return 0
        character = await self._get_character(attacker.character_id)
        if not character:
            logger.warning("ability_mod_no_character", combatant=attacker.name)
            return 0

        str_mod = character.abilities.get_modifier(AbilityScore.STRENGTH)
        dex_mod = character.abilities.get_modifier(AbilityScore.DEXTERITY)

        if weapon.is_finesse:
            # Finesse: use the higher of STR or DEX
            return max(str_mod, dex_mod)
        elif weapon.is_ranged and not weapon.is_thrown:
            # Ranged (non-thrown): use DEX
            return dex_mod
        else:
            # Melee and thrown: use STR
            return str_mod

    def _get_attack_advantage(
        self,
        attacker: Combatant,
        target: Combatant,
        weapon: WeaponStats,
    ) -> tuple[bool, bool]:
        """Determine advantage/disadvantage for an attack."""
        adv = False
        dis = False

        # SURPRISE: Attacks against surprised creatures have advantage
        if target.is_surprised:
            adv = True
            logger.debug("advantage_from_surprise", target=target.name)

        # Check attacker conditions
        attacker_conditions = self._get_combatant_conditions(attacker)
        atk_adv, atk_dis = ConditionResolver.get_attack_modifiers(
            attacker_conditions,
            exhaustion_level=0,
        )
        if atk_adv:
            adv = True
        if atk_dis:
            dis = True

        # Check target conditions (attacks against)
        target_conditions = self._get_combatant_conditions(target)
        in_melee = self.zone_tracker.is_in_melee_with(attacker.id, target.id)
        tgt_adv, tgt_dis = ConditionResolver.get_attacks_against_modifiers(
            target_conditions,
            attacker_within_5ft=in_melee,
        )
        if tgt_adv:
            adv = True
        if tgt_dis:
            dis = True

        # Check if target is dodging
        if self.zone_tracker.is_dodging(target.id):
            dis = True

        # Ranged attack in melee = disadvantage
        if weapon.is_ranged and self.zone_tracker.is_in_melee(attacker.id):
            dis = True

        # Check help advantage
        if self.manager.consume_help_advantage(target.id):
            adv = True

        return adv, dis

    # ==================== Condition Enforcement ====================

    def _get_action_blocking_condition(self, combatant: Combatant) -> Optional[Condition]:
        """Check if any active condition prevents the combatant from acting.

        D&D 5e rules:
        - Paralyzed: can't take actions or move
        - Stunned: can't take actions or move
        - Unconscious: can't take actions or move
        - Petrified: can't take actions or move
        - Incapacitated: can't take actions or reactions (but CAN move)
        """
        conditions = combatant.get_active_conditions()
        blocking = {
            Condition.PARALYZED, Condition.STUNNED,
            Condition.UNCONSCIOUS, Condition.PETRIFIED,
            Condition.INCAPACITATED,
        }
        for c in conditions:
            if c in blocking:
                return c
        return None

    def _is_auto_crit(self, target: Combatant, in_melee: bool) -> bool:
        """Check if hits against this target are automatic crits.

        D&D 5e: Attacks within 5ft against paralyzed or unconscious targets
        are automatic critical hits.
        """
        if not in_melee:
            return False
        conditions = target.get_active_conditions()
        return Condition.PARALYZED in conditions or Condition.UNCONSCIOUS in conditions

    # ==================== Other Actions ====================

    def _execute_dash(self, action: CombatAction, combatant: Combatant) -> ActionResult:
        """Double movement for this turn."""
        extra_movement = combatant.speed
        combatant.turn_resources.movement += extra_movement

        logger.info("dash_executed", combatant=combatant.name, extra_movement=extra_movement)

        return ActionResult(
            action=action,
            success=True,
            zone_changes=[f"Movement increased by {extra_movement}ft"],
        )

    def _execute_disengage(self, action: CombatAction, combatant: Combatant) -> ActionResult:
        """Prevent opportunity attacks this turn."""
        self.zone_tracker.mark_disengaged(combatant.id)

        logger.info("disengage_executed", combatant=combatant.name)

        return ActionResult(
            action=action,
            success=True,
            zone_changes=["Can move without provoking opportunity attacks"],
        )

    def _execute_dodge(self, action: CombatAction, combatant: Combatant) -> ActionResult:
        """Attacks against have disadvantage until next turn."""
        self.zone_tracker.mark_dodging(combatant.id)

        logger.info("dodge_executed", combatant=combatant.name)

        return ActionResult(
            action=action,
            success=True,
            zone_changes=["Attacks against you have disadvantage"],
        )

    def _execute_help(self, action: CombatAction, combatant: Combatant) -> ActionResult:
        """Give an ally advantage on their next attack."""
        if not action.target_ids:
            return ActionResult(action=action, success=False, error="No target specified")

        target_id = action.target_ids[0]
        target = self.manager.get_combatant(target_id)
        if not target:
            return ActionResult(action=action, success=False, error="Target not found")

        success = self.manager.grant_help_advantage(combatant.id, target_id)
        if not success:
            return ActionResult(action=action, success=False, error="Could not grant help")

        logger.info("help_executed", helper=combatant.name, target=target.name)

        return ActionResult(
            action=action,
            success=True,
            zone_changes=[f"{target.name} has advantage on next attack"],
        )

    async def _execute_hide(self, action: CombatAction, combatant: Combatant) -> ActionResult:
        """Attempt to hide (Stealth check)."""
        # Get stealth modifier from character data or monster SRD stats
        stealth_mod = 0
        if combatant.is_player and combatant.character_id:
            char = await self._get_character(combatant.character_id)
            if char:
                from ...models.common import Skill
                stealth_mod = char.get_skill_modifier(Skill.STEALTH)
        else:
            # Monster: check SRD for stealth proficiency, else use DEX mod
            stealth_mod = combatant.get_ability_modifier("dex")
            if combatant.monster_index:
                monster = self.srd.get_monster(combatant.monster_index)
                if monster:
                    for prof in monster.get("proficiencies", []):
                        if prof.get("proficiency", {}).get("index") == "skill-stealth":
                            stealth_mod = prof.get("value", stealth_mod)
                            break

        stealth_roll = self.roller.roll_check(modifier=stealth_mod)

        # For now, compare against passive perception of enemies
        # This would need more sophisticated handling
        dc = 12  # Placeholder passive perception

        success = stealth_roll.total >= dc

        logger.info(
            "hide_executed",
            combatant=combatant.name,
            roll=stealth_roll.total,
            dc=dc,
            success=success,
        )

        result = ActionResult(
            action=action,
            success=success,
            skill_roll=stealth_roll,
        )

        if success:
            result.zone_changes = ["Successfully hidden from enemies"]
        else:
            result.zone_changes = ["Failed to hide"]

        return result

    async def _execute_spell(
        self,
        action: CombatAction,
        caster: Combatant,
    ) -> ActionResult:
        """Execute a spell cast using SpellcastingManager."""
        from ..magic.spellcasting import SpellcastingManager, SpellType

        spell_mgr = SpellcastingManager()

        # Get character for spell slots
        character = self._character_cache.get(caster.character_id) if caster.character_id else None
        if not character:
            return ActionResult(
                action=action,
                success=False,
                error="No character data for spellcasting",
            )

        spell_index = action.spell_index
        slot_level = action.slot_level

        # Get spell info
        spell = spell_mgr.get_spell_info(spell_index)
        if not spell:
            return ActionResult(
                action=action,
                success=False,
                error=f"Unknown spell: {spell_index}",
            )

        # Determine slot level (default to spell level for cantrips use 0)
        if slot_level is None:
            slot_level = spell.level

        # Check if can cast
        can_cast, reason = spell_mgr.can_cast(character, spell_index, slot_level)
        if not can_cast:
            return ActionResult(
                action=action,
                success=False,
                error=reason,
            )

        # Expend spell slot (if not cantrip)
        if spell.level > 0:
            character.spell_slots.expend_slot(slot_level)

        result = ActionResult(
            action=action,
            success=True,
            spell_slot_used=slot_level if spell.level > 0 else None,
            spell_effect=spell.name,
        )

        # Determine spell type and execute
        if spell.attack_type:
            # Attack spell
            target = self.manager.get_combatant(action.target_ids[0]) if action.target_ids else None
            if not target:
                return ActionResult(
                    action=action,
                    success=False,
                    error="No target for attack spell",
                )

            # Get advantage/disadvantage
            adv, dis = self._get_spell_attack_advantage(caster, target)

            cast_result = spell_mgr.cast_attack_spell(
                caster=character,
                spell=spell,
                slot_level=slot_level,
                target_ac=target.armor_class,
                advantage=adv or action.advantage,
                disadvantage=dis or action.disadvantage,
            )

            result.attack_roll = cast_result.attack_roll
            result.critical_hit = cast_result.critical
            result.target_ac = target.armor_class

            if cast_result.hit:
                # Apply damage
                actual, unconscious, dead, resistances = self.manager.apply_damage(
                    target.id,
                    cast_result.damage_dealt,
                    damage_type=cast_result.damage_type,
                    is_critical=cast_result.critical,
                )
                result.damage_dealt = {target.id: actual}
                result.damage_type = cast_result.damage_type
                result.damage_roll = cast_result.damage_roll

                if dead:
                    result.killed_targets.append(target.id)
                elif unconscious:
                    result.unconscious_targets.append(target.id)

                # Check concentration
                if self._is_concentrating(target):
                    maintained = await self._check_concentration(target, actual)
                    if not maintained:
                        result.concentration_broken = True

        elif spell.save_dc_ability:
            # Save-based spell
            cast_result = spell_mgr.cast_save_spell(
                caster=character,
                spell=spell,
                slot_level=slot_level,
            )

            result.damage_roll = cast_result.damage_roll
            result.damage_type = cast_result.damage_type

            # Roll saves for each target
            for target_id in action.target_ids:
                target = self.manager.get_combatant(target_id)
                if not target:
                    continue

                # Roll save
                save_mod = self._get_save_modifier(target, spell.save_dc_ability)
                save_roll = self.roller.roll_check(modifier=save_mod)
                result.save_rolls[target_id] = save_roll

                # Determine damage (full on fail, half on success for most spells)
                if save_roll.total >= cast_result.save_dc:
                    # Success - half damage
                    damage = cast_result.damage_dealt // 2
                else:
                    # Failure - full damage
                    damage = cast_result.damage_dealt

                if damage > 0:
                    actual, unconscious, dead, resistances = self.manager.apply_damage(
                        target_id,
                        damage,
                        damage_type=cast_result.damage_type,
                    )
                    result.damage_dealt[target_id] = actual

                    if dead:
                        result.killed_targets.append(target_id)
                    elif unconscious:
                        result.unconscious_targets.append(target_id)

                # Apply condition on failed save (Hold Person → Paralyzed, etc.)
                if save_roll.total < cast_result.save_dc and spell_index in SPELL_CONDITION_MAP:
                    condition, duration, save_ends = SPELL_CONDITION_MAP[spell_index]
                    effect = CombatEffect(
                        name=f"{spell.name}: {condition.value}",
                        effect_type="condition",
                        source_combatant_id=caster.id,
                        source_spell_index=spell_index,
                        is_concentration=spell.concentration,
                        condition=condition,
                        duration_rounds=duration,
                        save_ability=spell.save_dc_ability.value if spell.save_dc_ability else None,
                        save_dc=cast_result.save_dc,
                        save_ends_on_success=save_ends,
                        save_at_end_of_turn=True,
                    )
                    self.manager.add_effect_to_combatant(target_id, effect)
                    if target_id not in result.conditions_applied:
                        result.conditions_applied[target_id] = []
                    result.conditions_applied[target_id].append(condition)
                    logger.info(
                        "spell_condition_applied",
                        spell=spell.name,
                        condition=condition.value,
                        target=target.name if target else target_id,
                        duration=duration,
                    )

        elif spell.heal_at_slot_level:
            # Healing spell
            cast_result = spell_mgr.cast_healing_spell(
                caster=character,
                spell=spell,
                slot_level=slot_level,
            )

            # Apply healing to targets
            for target_id in action.target_ids or [caster.id]:
                actual, revived = self.manager.apply_healing(target_id, cast_result.healing_amount)
                result.healing_done[target_id] = actual

                if revived:
                    result.stabilized_targets.append(target_id)

        else:
            # Utility spell
            result.spell_effect = f"{spell.name} takes effect"

        # Handle concentration
        if spell.concentration:
            # Break existing concentration if any
            if character.is_concentrating:
                character.concentration_spell_id = None
                result.concentration_broken = True

            # Start new concentration
            character.concentration_spell_id = spell_index
            result.zone_changes.append(f"Concentrating on {spell.name}")

        # Use action resource
        if not action.uses_bonus_action:
            caster.turn_resources.action = False
        else:
            caster.turn_resources.bonus_action = False

        logger.info(
            "spell_cast",
            caster=caster.name,
            spell=spell.name,
            slot=slot_level,
            targets=action.target_ids,
        )

        return result

    def _get_spell_attack_advantage(
        self,
        caster: Combatant,
        target: Combatant,
    ) -> tuple[bool, bool]:
        """Get advantage/disadvantage for spell attacks."""
        adv = False
        dis = False

        # Surprised targets give advantage
        if target.is_surprised:
            adv = True

        # Ranged spell in melee = disadvantage
        if self.zone_tracker.is_in_melee(caster.id):
            dis = True

        # Target dodging = disadvantage
        if self.zone_tracker.is_dodging(target.id):
            dis = True

        return adv, dis

    def _get_save_modifier(self, combatant: Combatant, ability: "AbilityScore") -> int:
        """Get saving throw modifier for a combatant.

        Players: uses Character.get_save_modifier (includes proficiency).
        Monsters: uses Combatant.get_save_modifier (SRD save bonuses or ability mod).
        """
        if not ability:
            return 0

        # For player characters, get from character data
        if combatant.is_player and combatant.character_id:
            char = self._character_cache.get(combatant.character_id)
            if char:
                return char.get_save_modifier(ability)

        # For monsters, use SRD-populated save bonuses on the Combatant
        return combatant.get_save_modifier(ability.value)

    # ==================== Concentration ====================

    def _is_concentrating(self, combatant: Combatant) -> bool:
        """Check if combatant is concentrating on a spell."""
        if combatant.is_player and combatant.character_id:
            char = self._character_cache.get(combatant.character_id)
            if char:
                return char.is_concentrating
        return False

    async def _check_concentration(self, combatant: Combatant, damage: int) -> bool:
        """
        Roll concentration save after taking damage.

        Returns True if concentration is maintained.
        """
        dc = max(10, damage // 2)

        # Get CON save modifier from real character/monster data
        con_mod = 0
        if combatant.is_player and combatant.character_id:
            char = self._character_cache.get(combatant.character_id)
            if char:
                con_mod = char.get_save_modifier(AbilityScore.CONSTITUTION)
        else:
            con_mod = combatant.get_save_modifier("con")

        save_roll = self.roller.roll_save(modifier=con_mod)
        success = save_roll.total >= dc

        logger.info(
            "concentration_check",
            combatant=combatant.name,
            roll=save_roll.total,
            dc=dc,
            maintained=success,
        )

        if not success:
            # Break concentration
            self.manager.break_concentration(combatant.id)

        return success

    # ==================== Helper Methods ====================

    async def _get_character(self, character_id: str) -> Optional[Character]:
        """Get character from cache or repository."""
        if character_id in self._character_cache:
            return self._character_cache[character_id]

        repo = await get_character_repo()
        character = await repo.get_by_id(character_id)
        if character:
            self._character_cache[character_id] = character
        return character

    async def _get_equipped_weapons(self, character: Character) -> list[WeaponStats]:
        """Get equipped weapons for a character."""
        weapons = []

        repo = await get_inventory_repo()
        equipped_items = await repo.get_equipped_items(character.id)

        for item in equipped_items:
            weapon_data = self.srd.get_equipment(item.item_index)
            if weapon_data and weapon_data.get("equipment_category", {}).get("index") == "weapon":
                weapons.append(self._parse_srd_weapon(weapon_data))

        # Always have unarmed as fallback
        if not weapons:
            weapons.append(WeaponStats(
                name="Unarmed Strike",
                damage_dice="1",
                damage_type="bludgeoning",
                is_melee=True,
            ))

        return weapons

    def _parse_srd_weapon(self, weapon_data: dict) -> WeaponStats:
        """Parse SRD weapon data into WeaponStats."""
        damage = weapon_data.get("damage", {})
        properties = [p.get("index", "") for p in weapon_data.get("properties", [])]

        return WeaponStats(
            name=weapon_data.get("name", "Weapon"),
            damage_dice=damage.get("damage_dice", "1d4"),
            damage_type=damage.get("damage_type", {}).get("name", "bludgeoning").lower(),
            is_melee=weapon_data.get("weapon_range") == "Melee",
            is_ranged=weapon_data.get("weapon_range") == "Ranged",
            is_finesse="finesse" in properties,
            is_thrown="thrown" in properties,
            properties=properties,
        )

    def _get_available_spells(self, character: Character) -> list[str]:
        """Get list of available spell names."""
        # Would need to check prepared spells and cantrips
        return character.prepared_spells if hasattr(character, 'prepared_spells') else []

    def _get_spell_slots(self, character: Character) -> dict[int, int]:
        """Get remaining spell slots by level (current count only)."""
        if hasattr(character, 'spell_slots') and character.spell_slots:
            slots = {}
            for level in range(1, 10):
                current, maximum = character.spell_slots.get_slots(level)
                if maximum > 0:
                    slots[level] = current
            return slots
        return {}

    def _get_monster_action_names(self, combatant: Combatant) -> list[str]:
        """Get list of action names for a monster."""
        if not combatant.monster_index:
            return []

        monster = self.srd.get_monster(combatant.monster_index)
        if not monster:
            return []

        actions = []
        for action_data in monster.get("actions", []):
            actions.append(action_data.get("name", "Attack"))

        return actions

    def _get_combatant_conditions(self, combatant: Combatant) -> list[Condition]:
        """Get active conditions for a combatant."""
        conditions = []
        for effect in combatant.effects:
            if effect.condition:
                conditions.append(effect.condition)
        return conditions

    # ==================== NPC Turn Handling ====================

    async def run_npc_turn(self, combatant: Combatant) -> list[ActionResult]:
        """
        Run an NPC's turn automatically using AI decision-making.

        Returns list of action results for narration.
        """
        from .npc_brain import get_npc_brain

        brain = get_npc_brain()
        results = []

        # Start the turn
        await self.start_turn(combatant)

        # Roll recharge for any used abilities
        recharged = brain.roll_recharge(combatant)
        if recharged:
            logger.info("abilities_recharged", combatant=combatant.name, abilities=recharged)

        # Execute actions while resources available
        max_actions = 3  # Safety limit
        action_count = 0

        while combatant.turn_resources.action and action_count < max_actions:
            action_count += 1

            # Get AI decision
            action = await brain.decide_action(
                combatant,
                self.manager.combat,
                self.zone_tracker,
            )

            if action.action_type == CombatActionType.END_TURN:
                break

            # Execute the action
            result = await self.execute_action(action)
            results.append(result)

            # If action failed, stop trying
            if not result.success:
                break

            # Check if combat ended
            if self.manager.combat.is_combat_over():
                break

        # End turn
        await self.end_turn(combatant)

        logger.info(
            "npc_turn_completed",
            combatant=combatant.name,
            actions_taken=len(results),
        )

        return results

    # ==================== Narrator Integration ====================

    async def narrate_result(self, result: ActionResult) -> str:
        """
        Pass an action result to the narrator for dramatic description.

        Returns the narrative text.
        """
        from ...llm.brains.narrator import MechanicalOutcome, get_narrator

        if self._narrator is None:
            self._narrator = get_narrator()

        # Convert ActionResult to MechanicalOutcome
        outcome = self._result_to_outcome(result)

        # Build combat context
        context = self._build_narrator_context(result)

        # Get narration
        narrator_result = await self._narrator.narrate_outcome(context, outcome)

        return narrator_result.content

    def _result_to_outcome(self, result: ActionResult) -> "MechanicalOutcome":
        """Convert ActionResult to MechanicalOutcome for narrator."""
        from ...llm.brains.narrator import MechanicalOutcome

        action_type = result.action.action_type.value

        # Build description based on action type
        description = self._build_outcome_description(result)

        # Build details dict
        details = {}

        if result.attack_roll:
            details["roll"] = result.attack_roll.total
            details["critical"] = result.critical_hit
            details["fumble"] = result.critical_miss

        if result.damage_dealt:
            total_damage = sum(result.damage_dealt.values())
            details["damage"] = total_damage
            details["damage_type"] = result.damage_type or ""

        if result.healing_done:
            details["healing"] = sum(result.healing_done.values())

        if result.killed_targets:
            details["killed"] = result.killed_targets

        if result.unconscious_targets:
            details["knocked_out"] = result.unconscious_targets

        if result.concentration_broken:
            details["concentration_broken"] = True

        if result.zone_changes:
            details["effects"] = result.zone_changes

        return MechanicalOutcome(
            action_type=action_type,
            success=result.success,
            description=description,
            details=details,
        )

    def _build_outcome_description(self, result: ActionResult) -> str:
        """Build a description string for the outcome."""
        action = result.action
        action_type = action.action_type

        # Get combatant and target names
        attacker = self.manager.get_combatant(action.combatant_id)
        attacker_name = attacker.name if attacker else "Someone"

        target_name = ""
        if action.target_ids:
            target = self.manager.get_combatant(action.target_ids[0])
            target_name = target.name if target else "something"

        if action_type == CombatActionType.ATTACK:
            weapon_name = action.weapon_index or action.ability_name or "weapon"
            if result.success:
                if result.critical_hit:
                    return f"{attacker_name}'s {weapon_name} strikes {target_name} with devastating precision!"
                return f"{attacker_name}'s {weapon_name} strikes {target_name}."
            else:
                if result.critical_miss:
                    return f"{attacker_name} swings wildly and misses {target_name} completely!"
                return f"{attacker_name}'s {weapon_name} misses {target_name}."

        elif action_type == CombatActionType.CAST_SPELL:
            spell_name = action.spell_index or "spell"
            if result.success:
                return f"{attacker_name} casts {spell_name} on {target_name}."
            return f"{attacker_name}'s {spell_name} fails to take effect."

        elif action_type == CombatActionType.DASH:
            return f"{attacker_name} dashes, doubling their movement speed."

        elif action_type == CombatActionType.DISENGAGE:
            return f"{attacker_name} carefully disengages from combat."

        elif action_type == CombatActionType.DODGE:
            return f"{attacker_name} takes a defensive stance, ready to dodge attacks."

        elif action_type == CombatActionType.HELP:
            return f"{attacker_name} assists {target_name}, giving them advantage on their next attack."

        elif action_type == CombatActionType.HIDE:
            if result.success:
                return f"{attacker_name} slips into the shadows undetected."
            return f"{attacker_name} attempts to hide but is spotted."

        return f"{attacker_name} performs an action."

    def _build_narrator_context(self, result: ActionResult) -> BrainContext:
        """Build context for the narrator from current combat state."""
        combat = self.manager.combat

        # Build initiative order string
        initiative_lines = []
        for combatant in combat.get_sorted_combatants():
            status = ""
            if not combatant.is_conscious:
                status = " [UNCONSCIOUS]"
            elif not combatant.is_active:
                status = " [OUT]"
            hp_display = f"HP: {combatant.hp_current}/{combatant.hp_max}"
            initiative_lines.append(f"- {combatant.name} ({hp_display}){status}")

        initiative_order = "\n".join(initiative_lines)

        # Get current combatant
        current = combat.get_current_combatant()
        current_name = current.name if current else "Unknown"

        # Build party status
        party_lines = []
        for combatant in combat.combatants:
            if combatant.is_player:
                hp_pct = int(100 * combatant.hp_current / max(1, combatant.hp_max))
                party_lines.append(f"{combatant.name}: {hp_pct}% HP")
        party_status = ", ".join(party_lines) if party_lines else None

        return BrainContext(
            in_combat=True,
            combat_round=combat.current_round,
            current_combatant=current_name,
            initiative_order=initiative_order,
            party_status=party_status,
            player_action=result.get_summary(),
        )

    async def narrate_turn_results(self, results: list[ActionResult]) -> str:
        """
        Narrate multiple action results from a turn.

        Combines results into a single narrative.
        """
        if not results:
            return ""

        narratives = []
        for result in results:
            narrative = await self.narrate_result(result)
            narratives.append(narrative)

        return "\n\n".join(narratives)


# ==================== Factory ====================

_coordinators: dict[int, CombatTurnCoordinator] = {}


def get_coordinator(manager: CombatManager, session: Optional["GameSession"] = None) -> CombatTurnCoordinator:
    """Get or create a coordinator for a combat encounter."""
    channel_id = manager.combat.channel_id

    if channel_id not in _coordinators:
        _coordinators[channel_id] = CombatTurnCoordinator(manager, session)

    return _coordinators[channel_id]


def get_coordinator_for_channel(channel_id: int) -> Optional[CombatTurnCoordinator]:
    """Get existing coordinator for a channel."""
    return _coordinators.get(channel_id)


def clear_coordinator(channel_id: int) -> None:
    """Clear coordinator for a channel (combat ended)."""
    if channel_id in _coordinators:
        _coordinators[channel_id].zone_tracker.on_combat_end()
        del _coordinators[channel_id]
