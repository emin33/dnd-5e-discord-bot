"""Combat manager - handles combat encounters."""

from typing import Optional
import uuid

import structlog

from dataclasses import dataclass, field

from ...models import (
    Character,
    Combat,
    Combatant,
    CombatEffect,
    CombatState,
    RechargeAbility,
    TurnResources,
)
from ...models.common import AbilityScore, Condition
from ...data.srd import get_srd
from ..mechanics.dice import get_roller, DiceRoll


@dataclass
class EffectProcessResult:
    """Result of processing an effect."""
    effect: CombatEffect
    expired: bool = False
    save_required: bool = False
    save_roll: Optional[int] = None
    save_dc: Optional[int] = None
    save_succeeded: bool = False
    ended_by_save: bool = False
    damage_dealt: int = 0
    damage_type: Optional[str] = None
    damage_halved: bool = False
    messages: list[str] = field(default_factory=list)


@dataclass
class RechargeRollResult:
    """Result of rolling to recharge an ability."""
    ability_name: str
    roll: int
    recharge_on: int
    recharged: bool
    message: str

logger = structlog.get_logger()


class CombatManager:
    """
    Manages a combat encounter.

    Handles:
    - Adding/removing combatants
    - Initiative rolling and ordering
    - Turn progression
    - Combat state machine transitions
    """

    def __init__(self, combat: Combat):
        self.combat = combat
        self.roller = get_roller()
        self.srd = get_srd()

    @classmethod
    def create_encounter(
        cls,
        session_id: str,
        channel_id: int,
        name: Optional[str] = None,
        description: Optional[str] = None,
    ) -> "CombatManager":
        """Create a new combat encounter."""
        combat = Combat(
            id=str(uuid.uuid4()),
            session_id=session_id,
            channel_id=channel_id,
            state=CombatState.SETUP,
            encounter_name=name,
            encounter_description=description,
        )
        return cls(combat)

    # ==================== Combatant Management ====================

    def add_player(self, character: Character) -> Combatant:
        """Add a player character to combat."""
        # Build ability scores dict from Character.abilities
        ability_scores = {
            "str": character.abilities.strength,
            "dex": character.abilities.dexterity,
            "con": character.abilities.constitution,
            "int": character.abilities.intelligence,
            "wis": character.abilities.wisdom,
            "cha": character.abilities.charisma,
        }

        # Build save bonuses from character's save modifiers
        save_bonuses = {}
        for ability in AbilityScore:
            save_bonuses[ability.value] = character.get_save_modifier(ability)

        combatant = Combatant(
            id=str(uuid.uuid4()),
            combat_id=self.combat.id,
            name=character.name,
            is_player=True,
            character_id=character.id,
            initiative_bonus=character.initiative_bonus,
            hp_max=character.hp.maximum,
            hp_current=character.hp.current,
            armor_class=character.armor_class,
            speed=character.speed,
            ability_scores=ability_scores,
            save_bonuses=save_bonuses,
            proficiency_bonus=character.proficiency_bonus,
        )
        self.combat.add_combatant(combatant)

        logger.info(
            "player_added_to_combat",
            combat_id=self.combat.id,
            character=character.name,
        )

        return combatant

    def add_monster(
        self,
        monster_index: str,
        name: Optional[str] = None,
        hp: Optional[int] = None,
    ) -> Optional[Combatant]:
        """Add a monster from the SRD to combat."""
        monster_data = self.srd.get_monster(monster_index)
        if not monster_data:
            logger.warning("monster_not_found", index=monster_index)
            return None

        # Use provided name or monster's name
        display_name = name or monster_data.get("name", monster_index)

        # Calculate initiative bonus from DEX
        dex_score = monster_data.get("dexterity", 10)
        init_bonus = (dex_score - 10) // 2

        # Get HP (use provided or roll)
        if hp is None:
            hp_value = monster_data.get("hit_points", 10)
        else:
            hp_value = hp

        # Get AC (handle the array format in SRD)
        ac_data = monster_data.get("armor_class", [{"value": 10}])
        if isinstance(ac_data, list) and ac_data:
            ac = ac_data[0].get("value", 10)
        else:
            ac = 10

        # Get speed
        speed_data = monster_data.get("speed", {})
        speed = speed_data.get("walk", "30 ft")
        if isinstance(speed, str):
            speed = int(speed.split()[0])

        # Get damage resistances, immunities, and vulnerabilities
        resistances = self._extract_damage_types(monster_data.get("damage_resistances", []))
        immunities = self._extract_damage_types(monster_data.get("damage_immunities", []))
        vulnerabilities = self._extract_damage_types(monster_data.get("damage_vulnerabilities", []))

        # Get recharge abilities from special_abilities and actions
        recharge_abilities = self._extract_recharge_abilities(monster_data)

        # Extract ability scores
        ability_scores = {
            "str": monster_data.get("strength", 10),
            "dex": monster_data.get("dexterity", 10),
            "con": monster_data.get("constitution", 10),
            "int": monster_data.get("intelligence", 10),
            "wis": monster_data.get("wisdom", 10),
            "cha": monster_data.get("charisma", 10),
        }

        # Extract proficiency bonus
        proficiency_bonus = monster_data.get("proficiency_bonus", 2)

        # Extract explicit save bonuses from proficiencies
        save_bonuses = {}
        for prof in monster_data.get("proficiencies", []):
            prof_index = prof.get("proficiency", {}).get("index", "")
            if prof_index.startswith("saving-throw-"):
                ability_key = prof_index.replace("saving-throw-", "")
                save_bonuses[ability_key] = prof.get("value", 0)

        combatant = Combatant(
            id=str(uuid.uuid4()),
            combat_id=self.combat.id,
            name=display_name,
            is_player=False,
            monster_index=monster_index,
            initiative_bonus=init_bonus,
            hp_max=hp_value,
            hp_current=hp_value,
            armor_class=ac,
            speed=speed,
            ability_scores=ability_scores,
            save_bonuses=save_bonuses,
            proficiency_bonus=proficiency_bonus,
            resistances=resistances,
            immunities=immunities,
            vulnerabilities=vulnerabilities,
            recharge_abilities=recharge_abilities,
        )
        self.combat.add_combatant(combatant)

        logger.info(
            "monster_added_to_combat",
            combat_id=self.combat.id,
            monster=display_name,
            hp=hp_value,
            ac=ac,
            resistances=resistances,
            immunities=immunities,
            vulnerabilities=vulnerabilities,
            recharge_abilities=[a.name for a in recharge_abilities],
        )

        return combatant

    def _extract_damage_types(self, damage_list: list) -> list[str]:
        """
        Extract damage type names from SRD format.

        SRD format can be:
        - List of strings: ["fire", "cold"]
        - List of dicts with url: [{"index": "fire", "name": "Fire", "url": "..."}]
        """
        result = []
        for item in damage_list:
            if isinstance(item, str):
                result.append(item.lower())
            elif isinstance(item, dict):
                # Handle various formats
                if "index" in item:
                    result.append(item["index"].lower())
                elif "name" in item:
                    result.append(item["name"].lower())
        return result

    def _extract_recharge_abilities(self, monster_data: dict) -> list[RechargeAbility]:
        """
        Extract recharge abilities from monster actions.

        In SRD, recharge abilities are indicated in the action name or usage field,
        e.g., "Fire Breath (Recharge 5-6)" or usage: {"type": "recharge on roll", "dice": "1d6", "min_value": 5}
        """
        import re
        abilities = []

        # Check actions for recharge abilities
        for action in monster_data.get("actions", []):
            action_name = action.get("name", "")

            # Check for "Recharge X-6" or "Recharge X" pattern in name
            match = re.search(r"\(Recharge (\d)(?:-6)?\)", action_name)
            if match:
                recharge_on = int(match.group(1))
                # Clean name by removing recharge info
                clean_name = re.sub(r"\s*\(Recharge \d(?:-6)?\)", "", action_name).strip()
                abilities.append(RechargeAbility(
                    name=clean_name,
                    recharge_on=recharge_on,
                    description=action.get("desc", ""),
                ))
                continue

            # Check usage field for recharge info
            usage = action.get("usage", {})
            if usage.get("type") == "recharge on roll":
                recharge_on = usage.get("min_value", 5)
                abilities.append(RechargeAbility(
                    name=action_name,
                    recharge_on=recharge_on,
                    description=action.get("desc", ""),
                ))

        return abilities

    def add_custom_combatant(
        self,
        name: str,
        hp: int,
        ac: int,
        initiative_bonus: int = 0,
        speed: int = 30,
        is_player: bool = False,
    ) -> Combatant:
        """Add a custom combatant to combat."""
        combatant = Combatant(
            id=str(uuid.uuid4()),
            combat_id=self.combat.id,
            name=name,
            is_player=is_player,
            initiative_bonus=initiative_bonus,
            hp_max=hp,
            hp_current=hp,
            armor_class=ac,
            speed=speed,
        )
        self.combat.add_combatant(combatant)

        logger.info(
            "custom_combatant_added",
            combat_id=self.combat.id,
            name=name,
        )

        return combatant

    def remove_combatant(self, combatant_id: str) -> bool:
        """Remove a combatant from combat."""
        return self.combat.remove_combatant(combatant_id)

    def get_combatant(self, combatant_id: str) -> Optional[Combatant]:
        """Get a combatant by ID."""
        for c in self.combat.combatants:
            if c.id == combatant_id:
                return c
        return None

    def get_combatant_by_name(self, name: str) -> Optional[Combatant]:
        """Get a combatant by name (case-insensitive partial match)."""
        name_lower = name.lower()
        for c in self.combat.combatants:
            if name_lower in c.name.lower():
                return c
        return None

    # ==================== Initiative ====================

    def roll_all_initiative(self) -> list[tuple[Combatant, DiceRoll]]:
        """Roll initiative for all combatants and sort."""
        results = []

        for combatant in self.combat.combatants:
            roll = self.roller.roll_initiative(modifier=combatant.initiative_bonus)
            combatant.initiative_roll = roll.total
            results.append((combatant, roll))

        # Sort and assign turn order
        self.combat.roll_all_initiative()

        logger.info(
            "initiative_rolled",
            combat_id=self.combat.id,
            order=[c.name for c in self.combat.get_sorted_combatants()],
        )

        return results

    def set_initiative(self, combatant_id: str, value: int) -> bool:
        """Manually set a combatant's initiative."""
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return False

        combatant.initiative_roll = value
        self.combat.roll_all_initiative()  # Re-sort
        return True

    # ==================== Turn Management ====================

    def start_combat(self) -> Optional[Combatant]:
        """Start the combat after initiative is rolled."""
        if self.combat.state != CombatState.ROLLING_INITIATIVE:
            if self.combat.state == CombatState.SETUP:
                # Roll initiative first
                self.combat.transition(CombatState.ROLLING_INITIATIVE)
                self.roll_all_initiative()

        self.combat.transition(CombatState.ACTIVE)
        self.combat.transition(CombatState.AWAITING_ACTION)

        current = self.combat.get_current_combatant()
        if current:
            current.turn_resources.reset_for_new_turn(current.speed)

        logger.info(
            "combat_started",
            combat_id=self.combat.id,
            first_combatant=current.name if current else "none",
        )

        return current

    def next_turn(self) -> tuple[Optional[Combatant], list[EffectProcessResult], list[EffectProcessResult], list[RechargeRollResult]]:
        """
        Advance to the next combatant's turn.

        Returns:
            (next_combatant, end_of_turn_results, start_of_turn_results, recharge_results)

            end_of_turn_results: Effects processed at end of previous turn
            start_of_turn_results: Effects processed at start of new turn
            recharge_results: Recharge ability rolls for monsters at start of turn
        """
        end_of_turn_results = []
        start_of_turn_results = []
        recharge_results = []

        if self.combat.state not in (CombatState.AWAITING_ACTION, CombatState.END_TURN):
            return None, [], [], []

        # Process end-of-turn effects for current combatant
        current = self.combat.get_current_combatant()
        if current:
            end_of_turn_results = self.process_end_of_turn_effects(current)

        self.combat.transition(CombatState.END_TURN)

        # Skip dead/inactive combatants
        next_combatant = self.combat.next_turn()
        while next_combatant and (not next_combatant.is_active or not next_combatant.is_conscious):
            next_combatant = self.combat.next_turn()

        if next_combatant:
            self.combat.transition(CombatState.ACTIVE)
            self.combat.transition(CombatState.AWAITING_ACTION)

            # Roll for recharge abilities at start of monster's turn
            if not next_combatant.is_player and next_combatant.recharge_abilities:
                recharge_results = self.roll_for_recharge(next_combatant)

            # Process start-of-turn effects for new combatant
            start_of_turn_results = self.process_start_of_turn_effects(next_combatant)

        # Check if combat should end
        if self.combat.is_combat_over():
            self.end_combat()
            return None, end_of_turn_results, [], []

        logger.info(
            "turn_advanced",
            combat_id=self.combat.id,
            round=self.combat.current_round,
            combatant=next_combatant.name if next_combatant else "none",
            end_effects=len(end_of_turn_results),
            start_effects=len(start_of_turn_results),
            recharge_rolls=len(recharge_results),
        )

        return next_combatant, end_of_turn_results, start_of_turn_results, recharge_results

    def end_combat(self) -> None:
        """End the combat encounter."""
        self.combat.transition(CombatState.COMBAT_END)

        from datetime import datetime
        self.combat.ended_at = datetime.utcnow()

        logger.info(
            "combat_ended",
            combat_id=self.combat.id,
            rounds=self.combat.current_round,
        )

    # ==================== Actions ====================

    def use_action(self, combatant_id: str) -> bool:
        """Use a combatant's action."""
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return False
        return combatant.turn_resources.use_action()

    def use_bonus_action(self, combatant_id: str) -> bool:
        """Use a combatant's bonus action."""
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return False
        return combatant.turn_resources.use_bonus_action()

    def use_reaction(self, combatant_id: str) -> bool:
        """Use a combatant's reaction."""
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return False
        return combatant.turn_resources.use_reaction()

    def use_movement(self, combatant_id: str, feet: int) -> bool:
        """Use a combatant's movement."""
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return False
        return combatant.turn_resources.use_movement(feet)

    # ==================== Damage and Healing ====================

    def apply_damage(
        self,
        combatant_id: str,
        damage: int,
        damage_type: Optional[str] = None,
        is_critical: bool = False,
    ) -> tuple[int, bool, bool, str]:
        """
        Apply damage to a combatant.

        Returns (actual_damage, is_now_unconscious, is_instant_death, damage_modifier)
        damage_modifier: "none", "resistance", "immunity", or "vulnerability"
        """
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return (0, False, False, "none")

        was_conscious = combatant.hp_current > 0

        # take_damage now handles resistances/vulnerabilities
        actual, instant_death, modifier = combatant.take_damage(
            damage,
            damage_type=damage_type,
            is_critical=is_critical,
        )
        is_now_unconscious = was_conscious and combatant.hp_current == 0

        logger.info(
            "damage_applied",
            combat_id=self.combat.id,
            target=combatant.name,
            damage=actual,
            damage_type=damage_type,
            modifier=modifier,
            remaining_hp=combatant.hp_current,
            instant_death=instant_death,
        )

        return (actual, is_now_unconscious, instant_death, modifier)

    def apply_healing(self, combatant_id: str, healing: int) -> tuple[int, bool]:
        """
        Apply healing to a combatant.

        Returns (actual_hp_restored, was_revived)
        """
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return (0, False)

        was_unconscious = combatant.hp_current == 0
        actual = combatant.heal(healing)
        was_revived = was_unconscious and combatant.hp_current > 0

        logger.info(
            "healing_applied",
            combat_id=self.combat.id,
            target=combatant.name,
            healing=actual,
            current_hp=combatant.hp_current,
            revived=was_revived,
        )

        return (actual, was_revived)

    # ==================== Death Saves ====================

    def roll_death_save(self, combatant_id: str) -> tuple[Optional[DiceRoll], str]:
        """
        Roll a death saving throw for a combatant.

        Returns (roll, result_type) where result_type is one of:
        - "success" - regular success
        - "failure" - regular failure
        - "stabilized" - 3 successes reached
        - "dead" - 3 failures reached
        - "critical_success" - nat 20, regains 1 HP
        - "critical_failure" - nat 1, counts as 2 failures
        - "not_dying" - combatant is not dying
        """
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return (None, "not_dying")

        if not combatant.is_dying:
            return (None, "not_dying")

        roll = self.roller.roll("1d20")
        natural = roll.kept_dice[0]

        if natural == 20:
            # Crit success: regain 1 HP, clear death saves
            combatant.heal(1)
            result = "critical_success"
        elif natural == 1:
            # Crit fail: 2 failures
            combatant.death_saves.add_failure(2)
            if combatant.death_saves.is_dead:
                result = "dead"
            else:
                result = "critical_failure"
        elif roll.total >= 10:
            combatant.death_saves.add_success()
            if combatant.death_saves.is_stable:
                combatant.stabilize()
                result = "stabilized"
            else:
                result = "success"
        else:
            combatant.death_saves.add_failure()
            if combatant.death_saves.is_dead:
                result = "dead"
            else:
                result = "failure"

        logger.info(
            "death_save_rolled",
            combat_id=self.combat.id,
            combatant=combatant.name,
            roll=roll.total,
            natural=natural,
            result=result,
            successes=combatant.death_saves.successes,
            failures=combatant.death_saves.failures,
        )

        return (roll, result)

    def stabilize_combatant(self, combatant_id: str) -> bool:
        """Stabilize a dying combatant (e.g., via Medicine check or spare the dying)."""
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return False

        if not combatant.is_dying:
            return False

        combatant.stabilize()

        logger.info(
            "combatant_stabilized",
            combat_id=self.combat.id,
            combatant=combatant.name,
        )

        return True

    def add_temp_hp(self, combatant_id: str, amount: int) -> int:
        """Add temporary HP to a combatant. Returns the new temp HP total."""
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return 0

        new_temp = combatant.add_temp_hp(amount)

        logger.info(
            "temp_hp_added",
            combat_id=self.combat.id,
            combatant=combatant.name,
            amount=amount,
            new_total=new_temp,
        )

        return new_temp

    # ==================== Status ====================

    def get_status(self) -> dict:
        """Get current combat status."""
        current = self.combat.get_current_combatant()
        sorted_combatants = self.combat.get_sorted_combatants()

        return {
            "state": self.combat.state.value,
            "round": self.combat.current_round,
            "current_combatant": current.name if current else None,
            "combatants": [
                {
                    "id": c.id,
                    "name": c.name,
                    "initiative": c.initiative_roll,
                    "hp": f"{c.hp_current}/{c.hp_max}",
                    "ac": c.armor_class,
                    "is_player": c.is_player,
                    "is_active": c.is_active,
                    "is_current": c.id == current.id if current else False,
                }
                for c in sorted_combatants
            ],
        }

    def sync_to_character(self, combatant: Combatant, character: Character) -> None:
        """
        Sync combatant state back to a character model.

        Call this to persist combat changes (HP, conditions, death saves)
        back to the character for database storage.
        """
        character.hp.current = combatant.hp_current
        character.hp.temporary = combatant.hp_temp
        character.death_saves.successes = combatant.death_saves.successes
        character.death_saves.failures = combatant.death_saves.failures

        logger.debug(
            "combatant_synced_to_character",
            combatant=combatant.name,
            character=character.name,
            hp=f"{character.hp.current}/{character.hp.maximum}",
        )

    def sync_from_character(self, character: Character) -> bool:
        """
        Sync character changes INTO the active combatant.

        Call this when equipment or other character state changes mid-combat
        (e.g., AC recalculation after equipping armor).

        Returns True if a matching combatant was found and updated.
        """
        combatant = next(
            (c for c in self.combat.combatants
             if c.is_player and c.character_id == character.id),
            None,
        )
        if not combatant:
            return False

        old_ac = combatant.armor_class
        combatant.armor_class = character.armor_class
        combatant.hp_max = character.hp.maximum

        logger.info(
            "combatant_synced_from_character",
            combatant=combatant.name,
            ac=f"{old_ac}->{character.armor_class}",
        )
        return True

    def get_player_combatants(self) -> list[Combatant]:
        """Get all player combatants (for syncing)."""
        return [c for c in self.combat.combatants if c.is_player and c.character_id]

    # ==================== Cover Mechanics ====================

    # Cover bonuses per D&D 5e rules
    COVER_BONUSES = {
        "none": 0,
        "half": 2,       # +2 AC and DEX saves
        "three-quarters": 5,  # +5 AC and DEX saves
        "full": None,    # Can't be directly targeted
    }

    def set_cover(self, combatant_id: str, cover_type: str) -> bool:
        """
        Set the cover level for a combatant.

        Cover types: "none", "half", "three-quarters", "full"
        """
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return False

        if cover_type not in self.COVER_BONUSES:
            return False

        # Store cover on combatant (using a dynamic attribute)
        combatant.cover = cover_type

        logger.info(
            "cover_set",
            combat_id=self.combat.id,
            combatant=combatant.name,
            cover=cover_type,
        )

        return True

    def get_cover(self, combatant_id: str) -> str:
        """Get the cover level for a combatant."""
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return "none"
        return getattr(combatant, "cover", "none")

    def get_cover_ac_bonus(self, combatant_id: str) -> int:
        """Get the AC bonus from cover."""
        cover = self.get_cover(combatant_id)
        bonus = self.COVER_BONUSES.get(cover, 0)
        return bonus if bonus is not None else 0

    def can_target(self, attacker_id: str, target_id: str) -> tuple[bool, str]:
        """
        Check if an attacker can target a defender.

        Full cover blocks direct targeting.
        """
        target_cover = self.get_cover(target_id)
        if target_cover == "full":
            target = self.get_combatant(target_id)
            target_name = target.name if target else "Target"
            return False, f"{target_name} has full cover and can't be directly targeted"
        return True, ""

    def get_effective_ac(self, combatant_id: str) -> int:
        """Get the effective AC including cover bonus."""
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return 10
        return combatant.armor_class + self.get_cover_ac_bonus(combatant_id)

    # ==================== Opportunity Attacks ====================

    def can_make_opportunity_attack(self, combatant_id: str) -> tuple[bool, str]:
        """
        Check if a combatant can make an opportunity attack.

        Requires:
        - Having a reaction available
        - Not being incapacitated
        - Being in active combat
        """
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return False, "Combatant not found"

        if not combatant.is_active:
            return False, "Combatant is not active"

        if not combatant.turn_resources.reaction:
            return False, "Reaction already used this round"

        # Check for incapacitated condition (simplified check)
        if combatant.hp_current <= 0:
            return False, "Combatant is incapacitated"

        return True, ""

    def make_opportunity_attack(
        self,
        attacker_id: str,
        target_id: str,
        attack_bonus: int = 0,
    ) -> dict:
        """
        Make an opportunity attack.

        Returns attack result dict.
        """
        attacker = self.get_combatant(attacker_id)
        target = self.get_combatant(target_id)

        if not attacker or not target:
            return {
                "success": False,
                "error": "Invalid attacker or target",
            }

        can_attack, reason = self.can_make_opportunity_attack(attacker_id)
        if not can_attack:
            return {
                "success": False,
                "error": reason,
            }

        # Use reaction
        self.use_reaction(attacker_id)

        # Roll attack
        attack_roll = self.roller.roll_attack(modifier=attack_bonus)
        effective_ac = self.get_effective_ac(target_id)
        hit = attack_roll.natural_20 or (not attack_roll.natural_1 and attack_roll.total >= effective_ac)
        critical = attack_roll.natural_20

        result = {
            "success": True,
            "attacker": attacker.name,
            "target": target.name,
            "attack_roll": attack_roll.total,
            "natural": attack_roll.kept_dice[0] if attack_roll.kept_dice else 0,
            "target_ac": effective_ac,
            "hit": hit,
            "critical": critical,
        }

        logger.info(
            "opportunity_attack",
            attacker=attacker.name,
            target=target.name,
            roll=attack_roll.total,
            hit=hit,
            critical=critical,
        )

        return result

    # ==================== Effect Processing ====================

    def add_effect_to_combatant(
        self,
        combatant_id: str,
        effect: CombatEffect,
    ) -> bool:
        """Add an effect to a combatant."""
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return False

        effect.created_round = self.combat.current_round
        combatant.add_effect(effect)

        logger.info(
            "effect_added",
            combat_id=self.combat.id,
            combatant=combatant.name,
            effect=effect.name,
            duration=effect.duration_rounds,
        )

        return True

    def remove_effect_from_combatant(
        self,
        combatant_id: str,
        effect_id: str,
    ) -> bool:
        """Remove an effect from a combatant."""
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return False

        removed = combatant.remove_effect(effect_id)

        if removed:
            logger.info(
                "effect_removed",
                combat_id=self.combat.id,
                combatant=combatant.name,
                effect_id=effect_id,
            )

        return removed

    def break_concentration(self, caster_combatant_id: str) -> list[tuple[str, list[CombatEffect]]]:
        """
        Break concentration for a caster, removing all their concentration effects.

        Returns list of (target_name, removed_effects) tuples.
        """
        removed_effects = []

        for combatant in self.combat.combatants:
            removed = combatant.remove_effects_by_source(caster_combatant_id, concentration_only=True)
            if removed:
                removed_effects.append((combatant.name, removed))

        if removed_effects:
            logger.info(
                "concentration_broken",
                combat_id=self.combat.id,
                caster_id=caster_combatant_id,
                effects_removed=sum(len(r[1]) for r in removed_effects),
            )

        return removed_effects

    def process_start_of_turn_effects(self, combatant: Combatant) -> list[EffectProcessResult]:
        """
        Process all effects at the start of a combatant's turn.

        This handles:
        - Ongoing damage (like Moonbeam, Spirit Guardians)
        - Duration countdown for start-of-turn effects
        """
        results = []
        effects_to_remove = []

        for effect in combatant.effects:
            if effect.expires_at_end_of_turn:
                continue  # Only process end-of-turn effects here if they have start-of-turn damage

            result = EffectProcessResult(effect=effect)

            # Process ongoing damage
            if effect.damage_dice and effect.damage_type:
                damage_roll = self.roller.roll(effect.damage_dice)
                damage = damage_roll.total

                # Check for save to reduce
                if effect.save_ability and effect.save_dc and effect.damage_save_for_half:
                    save_ability = AbilityScore[effect.save_ability.upper()]
                    save_mod = combatant.get_save_modifier(save_ability.value)
                    save_roll = self.roller.roll_save(modifier=save_mod)
                    result.save_required = True
                    result.save_roll = save_roll.total
                    result.save_dc = effect.save_dc
                    result.save_succeeded = save_roll.total >= effect.save_dc

                    if result.save_succeeded:
                        damage = damage // 2
                        result.damage_halved = True

                result.damage_dealt = damage
                result.damage_type = effect.damage_type
                result.messages.append(
                    f"Takes {damage} {effect.damage_type} damage from {effect.name}"
                )

                # Apply the damage
                combatant.take_damage(damage)

            # Decrement duration for start-of-turn expiry
            if not effect.expires_at_end_of_turn:
                if effect.rounds_remaining is not None:
                    effect.rounds_remaining -= 1
                    if effect.rounds_remaining <= 0:
                        result.expired = True
                        effects_to_remove.append(effect.id)
                        result.messages.append(f"{effect.name} has expired")

            results.append(result)

        # Remove expired effects
        for effect_id in effects_to_remove:
            combatant.remove_effect(effect_id)

        return results

    def process_end_of_turn_effects(self, combatant: Combatant) -> list[EffectProcessResult]:
        """
        Process all effects at the end of a combatant's turn.

        This handles:
        - Repeating saves (like Hold Person, Frightened)
        - Duration countdown for end-of-turn effects
        """
        results = []
        effects_to_remove = []

        for effect in combatant.effects:
            result = EffectProcessResult(effect=effect)

            # Process repeating saves
            if effect.save_ability and effect.save_dc and effect.save_at_end_of_turn:
                save_ability = AbilityScore[effect.save_ability.upper()]
                save_mod = combatant.get_save_modifier(save_ability.value)
                save_roll = self.roller.roll_save(modifier=save_mod)
                result.save_required = True
                result.save_roll = save_roll.total
                result.save_dc = effect.save_dc
                result.save_succeeded = save_roll.total >= effect.save_dc

                if result.save_succeeded and effect.save_ends_on_success:
                    result.ended_by_save = True
                    effects_to_remove.append(effect.id)
                    result.messages.append(
                        f"Succeeded on save against {effect.name} (DC {effect.save_dc}) - effect ends!"
                    )
                elif not result.save_succeeded:
                    result.messages.append(
                        f"Failed save against {effect.name} (rolled {save_roll.total} vs DC {effect.save_dc})"
                    )

            # Decrement duration for end-of-turn expiry
            if effect.expires_at_end_of_turn and not result.ended_by_save:
                if effect.rounds_remaining is not None:
                    effect.rounds_remaining -= 1
                    if effect.rounds_remaining <= 0:
                        result.expired = True
                        effects_to_remove.append(effect.id)
                        result.messages.append(f"{effect.name} has expired")

            if result.save_required or result.expired:
                results.append(result)

        # Remove expired effects
        for effect_id in effects_to_remove:
            combatant.remove_effect(effect_id)

        return results

    def get_combatant_effects(self, combatant_id: str) -> list[CombatEffect]:
        """Get all active effects on a combatant."""
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return []
        return combatant.effects.copy()

    # ==================== Help Action ====================

    def grant_help_advantage(self, helper_id: str, target_id: str) -> bool:
        """
        Grant advantage to a target via the Help action.

        The target's next attack roll or ability check gains advantage.
        """
        helper = self.get_combatant(helper_id)
        target = self.get_combatant(target_id)

        if not helper or not target:
            return False

        # Use helper's action
        if not self.use_action(helper_id):
            return False

        target.has_help_advantage = True

        logger.info(
            "help_action_granted",
            combat_id=self.combat.id,
            helper=helper.name,
            target=target.name,
        )

        return True

    def consume_help_advantage(self, combatant_id: str) -> bool:
        """Consume help advantage when making a roll. Returns True if it was available."""
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return False
        return combatant.consume_help_advantage()

    # ==================== Recharge Abilities ====================

    def roll_for_recharge(self, combatant: Combatant) -> list[RechargeRollResult]:
        """
        Roll to recharge unavailable abilities at the start of a monster's turn.

        Per D&D 5e rules, at the start of a monster's turn, roll 1d6 for each
        unavailable recharge ability. If the roll meets or exceeds the recharge
        threshold (e.g., 5 for "Recharge 5-6"), the ability becomes available again.

        Returns:
            List of RechargeRollResult for each ability that was rolled for.
        """
        results = []

        for ability in combatant.recharge_abilities:
            if ability.is_available:
                # Already available, no need to roll
                continue

            # Roll d6
            roll = self.roller.roll("1d6")
            roll_value = roll.total
            recharged = roll_value >= ability.recharge_on

            if recharged:
                ability.is_available = True
                message = f"{ability.name} recharged! (rolled {roll_value}, needed {ability.recharge_on}+)"
            else:
                message = f"{ability.name} failed to recharge (rolled {roll_value}, needed {ability.recharge_on}+)"

            result = RechargeRollResult(
                ability_name=ability.name,
                roll=roll_value,
                recharge_on=ability.recharge_on,
                recharged=recharged,
                message=message,
            )
            results.append(result)

            logger.info(
                "recharge_roll",
                combat_id=self.combat.id,
                combatant=combatant.name,
                ability=ability.name,
                roll=roll_value,
                threshold=ability.recharge_on,
                recharged=recharged,
            )

        return results

    def use_recharge_ability(self, combatant_id: str, ability_name: str) -> tuple[bool, str]:
        """
        Mark a recharge ability as used.

        Returns:
            (success, message)
        """
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return False, "Combatant not found"

        used = combatant.use_recharge_ability(ability_name)
        if used:
            logger.info(
                "recharge_ability_used",
                combat_id=self.combat.id,
                combatant=combatant.name,
                ability=ability_name,
            )
            return True, f"{ability_name} used"
        else:
            # Check if ability exists but is unavailable
            for ability in combatant.recharge_abilities:
                if ability.name.lower() == ability_name.lower():
                    return False, f"{ability_name} is not available (needs to recharge)"
            return False, f"{ability_name} not found"

    def get_available_recharge_abilities(self, combatant_id: str) -> list[RechargeAbility]:
        """Get all available recharge abilities for a combatant."""
        combatant = self.get_combatant(combatant_id)
        if not combatant:
            return []
        return combatant.get_available_recharge_abilities()


# Active combat managers by session key (str).
# Discord callers pass channel_id (int) which is auto-converted.
_active_combats: dict[str, CombatManager] = {}


def _combat_key(channel_id: int) -> str:
    """Convert a Discord channel_id to a session key."""
    return f"discord:{channel_id}"


def get_combat_for_channel(channel_id: int) -> Optional[CombatManager]:
    """Get the active combat for a channel."""
    return _active_combats.get(_combat_key(channel_id))


def get_combat_by_key(session_key: str) -> Optional[CombatManager]:
    """Get the active combat by generic session key."""
    return _active_combats.get(session_key)


def set_combat_for_channel(channel_id: int, combat: CombatManager) -> None:
    """Set the active combat for a channel."""
    _active_combats[_combat_key(channel_id)] = combat


def set_combat_by_key(session_key: str, combat: CombatManager) -> None:
    """Set the active combat by generic session key."""
    _active_combats[session_key] = combat


def try_set_combat_for_channel(channel_id: int, combat: CombatManager) -> bool:
    """Atomically set combat only if none is active. Returns True if set."""
    from ...models import CombatState
    key = _combat_key(channel_id)
    existing = _active_combats.get(key)
    if existing and existing.combat.state != CombatState.COMBAT_END:
        return False
    _active_combats[key] = combat
    return True


def clear_combat_for_channel(channel_id: int) -> None:
    """Clear the active combat for a channel."""
    _active_combats.pop(_combat_key(channel_id), None)


def clear_combat_by_key(session_key: str) -> None:
    """Clear the active combat by generic session key."""
    _active_combats.pop(session_key, None)
