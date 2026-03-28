"""Spellcasting manager - handles spell resolution."""

from dataclasses import dataclass
from enum import Enum
from typing import Optional
import structlog

from ...models import Character, AbilityScore
from ...data.srd import get_srd
from ..mechanics.dice import get_roller, DiceRoll

logger = structlog.get_logger()


class SpellType(str, Enum):
    """Types of spells based on how they affect targets."""
    ATTACK = "attack"  # Spell attack roll
    SAVE = "save"  # Target makes saving throw
    UTILITY = "utility"  # No attack or save (buffs, healing, etc.)
    HEALING = "healing"  # Heals HP


class SpellSchool(str, Enum):
    """Schools of magic."""
    ABJURATION = "abjuration"
    CONJURATION = "conjuration"
    DIVINATION = "divination"
    ENCHANTMENT = "enchantment"
    EVOCATION = "evocation"
    ILLUSION = "illusion"
    NECROMANCY = "necromancy"
    TRANSMUTATION = "transmutation"


@dataclass
class SpellInfo:
    """Parsed spell information from SRD."""
    index: str
    name: str
    level: int  # 0 for cantrips
    school: SpellSchool
    casting_time: str
    range: str
    components: list[str]  # V, S, M
    material: Optional[str]
    duration: str
    concentration: bool
    ritual: bool
    description: str
    higher_level: Optional[str]

    # Combat info
    attack_type: Optional[str]  # "melee" or "ranged"
    damage_dice: Optional[str]
    damage_type: Optional[str]
    save_dc_ability: Optional[AbilityScore]
    heal_at_slot_level: Optional[dict[int, str]]  # {1: "1d8+mod", 2: "2d8+mod", ...}


@dataclass
class SpellCastResult:
    """Result of casting a spell."""
    success: bool
    spell: SpellInfo
    slot_used: int  # 0 for cantrips

    # Attack results
    attack_roll: Optional[DiceRoll] = None
    hit: bool = False
    critical: bool = False

    # Damage results
    damage_roll: Optional[DiceRoll] = None
    damage_dealt: int = 0
    damage_type: Optional[str] = None

    # Healing results
    healing_roll: Optional[DiceRoll] = None
    healing_amount: int = 0

    # Save results (for target)
    save_dc: Optional[int] = None
    save_ability: Optional[AbilityScore] = None

    # Status
    concentration_started: bool = False
    concentration_broken: bool = False
    error: Optional[str] = None


class SpellcastingManager:
    """Manages spellcasting resolution."""

    def __init__(self):
        self.roller = get_roller()
        self.srd = get_srd()

    def get_spell_info(self, spell_index: str) -> Optional[SpellInfo]:
        """Get parsed spell information from SRD."""
        spell_data = self.srd.get_spell(spell_index)
        if not spell_data:
            return None

        # Parse school
        school_data = spell_data.get("school", {})
        school_name = school_data.get("index", "evocation").lower()
        try:
            school = SpellSchool(school_name)
        except ValueError:
            school = SpellSchool.EVOCATION

        # Parse components
        components = spell_data.get("components", [])
        material = None
        if "M" in components:
            material = spell_data.get("material")

        # Parse damage
        damage_dice = None
        damage_type = None
        damage_data = spell_data.get("damage", {})
        if damage_data:
            damage_at_level = damage_data.get("damage_at_slot_level", {})
            damage_at_char_level = damage_data.get("damage_at_character_level", {})

            # Get base damage dice
            if damage_at_level:
                # Use first level's damage as base
                first_level = min(int(k) for k in damage_at_level.keys())
                damage_dice = damage_at_level.get(str(first_level))
            elif damage_at_char_level:
                damage_dice = damage_at_char_level.get("1", damage_at_char_level.get("5", ""))

            damage_type_data = damage_data.get("damage_type", {})
            damage_type = damage_type_data.get("name", "").lower()

        # Parse save DC ability
        save_dc_ability = None
        dc_data = spell_data.get("dc", {})
        if dc_data:
            dc_type = dc_data.get("dc_type", {}).get("index", "")
            ability_map = {
                "str": AbilityScore.STRENGTH,
                "dex": AbilityScore.DEXTERITY,
                "con": AbilityScore.CONSTITUTION,
                "int": AbilityScore.INTELLIGENCE,
                "wis": AbilityScore.WISDOM,
                "cha": AbilityScore.CHARISMA,
            }
            save_dc_ability = ability_map.get(dc_type)

        # Parse healing
        heal_at_slot_level = None
        heal_data = spell_data.get("heal_at_slot_level", {})
        if heal_data:
            heal_at_slot_level = {int(k): v for k, v in heal_data.items()}

        # Parse attack type
        attack_type = spell_data.get("attack_type")

        # Build description
        desc_list = spell_data.get("desc", [])
        description = "\n\n".join(desc_list) if desc_list else ""

        higher_level_list = spell_data.get("higher_level", [])
        higher_level = "\n\n".join(higher_level_list) if higher_level_list else None

        return SpellInfo(
            index=spell_data.get("index", spell_index),
            name=spell_data.get("name", spell_index.replace("-", " ").title()),
            level=spell_data.get("level", 0),
            school=school,
            casting_time=spell_data.get("casting_time", "1 action"),
            range=spell_data.get("range", "Self"),
            components=components,
            material=material,
            duration=spell_data.get("duration", "Instantaneous"),
            concentration=spell_data.get("concentration", False),
            ritual=spell_data.get("ritual", False),
            description=description,
            higher_level=higher_level,
            attack_type=attack_type,
            damage_dice=damage_dice,
            damage_type=damage_type,
            save_dc_ability=save_dc_ability,
            heal_at_slot_level=heal_at_slot_level,
        )

    def can_cast(
        self,
        character: Character,
        spell_index: str,
        slot_level: Optional[int] = None,
    ) -> tuple[bool, str]:
        """
        Check if character can cast a spell.

        Returns (can_cast, reason).
        """
        spell = self.get_spell_info(spell_index)
        if not spell:
            return False, f"Unknown spell: {spell_index}"

        # Check if known/prepared
        is_known = spell_index in character.known_spells
        is_prepared = spell_index in character.prepared_spells

        # Cantrips just need to be known
        if spell.level == 0:
            if not is_known:
                return False, f"You don't know {spell.name}"
            return True, ""

        # Non-cantrips need to be prepared (or known for classes like sorcerer)
        if not is_prepared and not is_known:
            return False, f"You haven't prepared {spell.name}"

        # Determine slot level to use
        if slot_level is None:
            slot_level = spell.level
        elif slot_level < spell.level:
            return False, f"{spell.name} requires at least a level {spell.level} slot"

        # Check spell slot availability
        if not character.spell_slots.has_slot(slot_level):
            return False, f"No level {slot_level} spell slots remaining"

        return True, ""

    def cast_attack_spell(
        self,
        caster: Character,
        spell: SpellInfo,
        slot_level: int,
        target_ac: int,
        advantage: bool = False,
        disadvantage: bool = False,
    ) -> SpellCastResult:
        """Cast a spell that requires an attack roll."""
        result = SpellCastResult(
            success=True,
            spell=spell,
            slot_used=slot_level,
        )

        # Roll spell attack
        attack_bonus = caster.spell_attack_bonus or 0
        result.attack_roll = self.roller.roll_attack(
            modifier=attack_bonus,
            advantage=advantage,
            disadvantage=disadvantage,
        )

        # Determine hit
        is_critical = result.attack_roll.natural_20
        is_fumble = result.attack_roll.natural_1
        result.hit = is_critical or (not is_fumble and result.attack_roll.total >= target_ac)
        result.critical = is_critical

        # Roll damage if hit
        if result.hit and spell.damage_dice:
            damage_dice = self._scale_damage(spell, slot_level, is_cantrip=(spell.level == 0), caster_level=caster.level)
            result.damage_roll = self.roller.roll_damage(damage_dice, critical=is_critical)
            result.damage_dealt = result.damage_roll.total
            result.damage_type = spell.damage_type

        return result

    def cast_save_spell(
        self,
        caster: Character,
        spell: SpellInfo,
        slot_level: int,
    ) -> SpellCastResult:
        """Cast a spell that requires a saving throw (target rolls save separately)."""
        result = SpellCastResult(
            success=True,
            spell=spell,
            slot_used=slot_level,
        )

        # Set save DC info
        result.save_dc = caster.spell_save_dc
        result.save_ability = spell.save_dc_ability

        # Pre-roll damage (target save determines full/half/none)
        if spell.damage_dice:
            damage_dice = self._scale_damage(spell, slot_level, is_cantrip=(spell.level == 0), caster_level=caster.level)
            result.damage_roll = self.roller.roll(damage_dice)
            result.damage_dealt = result.damage_roll.total
            result.damage_type = spell.damage_type

        return result

    def cast_healing_spell(
        self,
        caster: Character,
        spell: SpellInfo,
        slot_level: int,
    ) -> SpellCastResult:
        """Cast a healing spell."""
        result = SpellCastResult(
            success=True,
            spell=spell,
            slot_used=slot_level,
        )

        # Get healing dice for this slot level
        if spell.heal_at_slot_level:
            heal_dice = spell.heal_at_slot_level.get(slot_level)
            if heal_dice:
                # Replace "mod" with actual spellcasting modifier
                if caster.spellcasting_ability:
                    mod = caster.abilities.get_modifier(caster.spellcasting_ability)
                    heal_dice = heal_dice.replace("+ your spellcasting ability modifier", f"+{mod}")
                    heal_dice = heal_dice.replace("+ mod", f"+{mod}")

                result.healing_roll = self.roller.roll(heal_dice)
                result.healing_amount = max(0, result.healing_roll.total)

        return result

    def cast_utility_spell(
        self,
        caster: Character,
        spell: SpellInfo,
        slot_level: int,
    ) -> SpellCastResult:
        """Cast a utility spell (no attack, save, or healing)."""
        return SpellCastResult(
            success=True,
            spell=spell,
            slot_used=slot_level,
        )

    def _scale_damage(
        self,
        spell: SpellInfo,
        slot_level: int,
        is_cantrip: bool,
        caster_level: int,
    ) -> str:
        """Scale damage dice based on slot level or caster level."""
        if not spell.damage_dice:
            return "0"

        # For cantrips, scale by character level
        if is_cantrip:
            return self._scale_cantrip_damage(spell.damage_dice, caster_level)

        # For leveled spells, check SRD for scaling
        spell_data = self.srd.get_spell(spell.index)
        if spell_data:
            damage_data = spell_data.get("damage", {})
            damage_at_slot = damage_data.get("damage_at_slot_level", {})
            if damage_at_slot:
                return damage_at_slot.get(str(slot_level), spell.damage_dice)

        return spell.damage_dice

    def _scale_cantrip_damage(self, base_dice: str, caster_level: int) -> str:
        """Scale cantrip damage based on caster level (5th, 11th, 17th)."""
        # Parse base dice (e.g., "1d10")
        import re
        match = re.match(r"(\d+)d(\d+)(.*)$", base_dice)
        if not match:
            return base_dice

        num_dice = int(match.group(1))
        die_size = match.group(2)
        modifier = match.group(3)  # e.g., "+3" or ""

        # Determine multiplier based on level
        if caster_level >= 17:
            num_dice *= 4
        elif caster_level >= 11:
            num_dice *= 3
        elif caster_level >= 5:
            num_dice *= 2

        return f"{num_dice}d{die_size}{modifier}"

    def get_spell_type(self, spell: SpellInfo) -> SpellType:
        """Determine the type of spell for resolution."""
        if spell.attack_type:
            return SpellType.ATTACK
        if spell.save_dc_ability:
            return SpellType.SAVE
        if spell.heal_at_slot_level:
            return SpellType.HEALING
        return SpellType.UTILITY

    def start_concentration(self, character: Character, spell: SpellInfo) -> bool:
        """
        Start concentrating on a spell.

        Returns True if concentration started, False if another spell was broken.
        """
        was_concentrating = character.concentration_spell_id is not None
        character.concentration_spell_id = spell.index

        if was_concentrating:
            logger.info(
                "concentration_broken",
                character=character.name,
                new_spell=spell.name,
            )

        return was_concentrating

    def break_concentration(self, character: Character) -> Optional[str]:
        """
        Break concentration.

        Returns the index of the spell that was being concentrated on, if any.
        """
        spell_index = character.concentration_spell_id
        character.concentration_spell_id = None
        return spell_index

    def roll_concentration_save(
        self,
        character: Character,
        damage_taken: int,
    ) -> tuple[DiceRoll, bool]:
        """
        Roll a concentration save after taking damage.

        DC = higher of 10 or half damage taken.
        Returns (roll, maintained_concentration).
        """
        dc = max(10, damage_taken // 2)

        # Constitution save
        modifier = character.get_save_modifier(AbilityScore.CONSTITUTION)
        roll = self.roller.roll_save(modifier=modifier)

        maintained = roll.total >= dc

        if not maintained:
            self.break_concentration(character)

        logger.info(
            "concentration_save",
            character=character.name,
            damage=damage_taken,
            dc=dc,
            roll=roll.total,
            maintained=maintained,
        )

        return roll, maintained

    def get_prepared_spell_limit(self, character: Character) -> int:
        """
        Calculate the maximum number of spells a character can prepare.

        Rules by class:
        - Cleric, Druid: level + WIS mod (min 1)
        - Paladin: level/2 + CHA mod (min 1)
        - Wizard: level + INT mod (min 1)
        - Artificer: level/2 + INT mod (min 1)
        - Bard, Ranger, Sorcerer, Warlock: N/A (known spells, not prepared)
        """
        class_index = character.class_index.lower() if character.class_index else ""
        level = character.level

        if class_index in ("cleric", "druid"):
            return max(1, level + character.abilities.wis_mod)

        elif class_index == "paladin":
            return max(1, (level // 2) + character.abilities.cha_mod)

        elif class_index == "wizard":
            return max(1, level + character.abilities.int_mod)

        elif class_index == "artificer":
            return max(1, (level // 2) + character.abilities.int_mod)

        # Classes with known spells don't prepare
        elif class_index in ("bard", "ranger", "sorcerer", "warlock"):
            # Return the number of known spells as the "limit"
            return len(character.known_spells)

        # Default fallback
        return max(1, level)

    def can_prepare_spell(
        self,
        character: Character,
        spell_index: str,
    ) -> tuple[bool, str]:
        """
        Check if a character can prepare a specific spell.

        Returns (can_prepare, reason).
        """
        spell = self.get_spell_info(spell_index)
        if not spell:
            return False, f"Unknown spell: {spell_index}"

        # Cantrips are always "prepared" (known)
        if spell.level == 0:
            return True, "Cantrips are always available"

        # Check if already prepared
        if spell_index in character.prepared_spells:
            return False, f"{spell.name} is already prepared"

        # Check preparation limit
        current_prepared = len([s for s in character.prepared_spells if s not in character.known_spells])
        limit = self.get_prepared_spell_limit(character)

        # For known-spells classes, they don't "prepare" in the traditional sense
        class_index = character.class_index.lower() if character.class_index else ""
        if class_index in ("bard", "ranger", "sorcerer", "warlock"):
            if spell_index not in character.known_spells:
                return False, f"You don't know {spell.name}"
            return True, ""

        # Check if at limit
        if current_prepared >= limit:
            return False, f"You can only prepare {limit} spells (currently {current_prepared} prepared)"

        # Check if spell is on class list (simplified - would need full class spell lists)
        # For now, just check if they know it
        if spell_index not in character.known_spells:
            return False, f"You don't have access to {spell.name}"

        return True, ""

    def prepare_spell(
        self,
        character: Character,
        spell_index: str,
    ) -> tuple[bool, str]:
        """
        Prepare a spell for the character.

        Returns (success, message).
        """
        can_prepare, reason = self.can_prepare_spell(character, spell_index)
        if not can_prepare:
            return False, reason

        spell = self.get_spell_info(spell_index)
        character.prepared_spells.append(spell_index)

        logger.info(
            "spell_prepared",
            character=character.name,
            spell=spell.name if spell else spell_index,
        )

        return True, f"Prepared {spell.name if spell else spell_index}"

    def unprepare_spell(
        self,
        character: Character,
        spell_index: str,
    ) -> tuple[bool, str]:
        """
        Remove a spell from the prepared list.

        Returns (success, message).
        """
        if spell_index not in character.prepared_spells:
            return False, "That spell is not prepared"

        spell = self.get_spell_info(spell_index)
        character.prepared_spells.remove(spell_index)

        logger.info(
            "spell_unprepared",
            character=character.name,
            spell=spell.name if spell else spell_index,
        )

        return True, f"Unprepared {spell.name if spell else spell_index}"

    # ==================== Ritual Casting ====================

    def can_cast_as_ritual(
        self,
        character: Character,
        spell_index: str,
    ) -> tuple[bool, str]:
        """
        Check if a character can cast a spell as a ritual.

        Ritual casting rules:
        - Spell must have the ritual tag
        - Caster must have ritual casting feature (most spellcasters do)
        - Takes 10 extra minutes to cast
        - Does NOT consume a spell slot

        Class-specific rules:
        - Wizard: Can ritual cast any ritual spell in spellbook (known_spells)
        - Cleric/Druid: Must have the spell prepared
        - Bard: Must know the spell AND have Ritual Casting feature (level 1+)
        """
        spell = self.get_spell_info(spell_index)
        if not spell:
            return False, f"Unknown spell: {spell_index}"

        if not spell.ritual:
            return False, f"{spell.name} cannot be cast as a ritual"

        class_index = character.class_index.lower() if character.class_index else ""

        # Check class-specific ritual casting rules
        if class_index == "wizard":
            # Wizards can ritual cast any ritual spell in their spellbook
            if spell_index not in character.known_spells:
                return False, f"You don't have {spell.name} in your spellbook"
            return True, ""

        elif class_index in ("cleric", "druid", "artificer"):
            # Must have the spell prepared
            if spell_index not in character.prepared_spells:
                return False, f"You must have {spell.name} prepared to cast it as a ritual"
            return True, ""

        elif class_index == "bard":
            # Must know the spell
            if spell_index not in character.known_spells:
                return False, f"You don't know {spell.name}"
            return True, ""

        elif class_index in ("ranger", "paladin"):
            # Must have the spell prepared
            if spell_index not in character.prepared_spells:
                return False, f"You must have {spell.name} prepared to cast it as a ritual"
            return True, ""

        elif class_index in ("sorcerer", "warlock"):
            # These classes don't normally have ritual casting
            # Some subclasses grant it, but we'll keep it simple
            return False, f"{class_index.title()}s cannot cast rituals"

        # Default: check if known/prepared
        if spell_index in character.prepared_spells or spell_index in character.known_spells:
            return True, ""

        return False, f"You don't have access to {spell.name}"

    def cast_ritual(
        self,
        caster: Character,
        spell_index: str,
    ) -> SpellCastResult:
        """
        Cast a spell as a ritual.

        - No spell slot consumed
        - Takes 10 extra minutes
        - Spell must be a ritual spell

        Returns SpellCastResult (slot_used will be 0).
        """
        spell = self.get_spell_info(spell_index)
        if not spell:
            return SpellCastResult(
                success=False,
                spell=SpellInfo(
                    index=spell_index,
                    name=spell_index,
                    level=0,
                    school=SpellSchool.ABJURATION,
                    casting_time="",
                    range="",
                    components=[],
                    material=None,
                    duration="",
                    concentration=False,
                    ritual=False,
                    description="",
                    higher_level=None,
                    attack_type=None,
                    damage_dice=None,
                    damage_type=None,
                    save_dc_ability=None,
                    heal_at_slot_level=None,
                ),
                slot_used=0,
                error=f"Unknown spell: {spell_index}",
            )

        can_cast, reason = self.can_cast_as_ritual(caster, spell_index)
        if not can_cast:
            return SpellCastResult(
                success=False,
                spell=spell,
                slot_used=0,
                error=reason,
            )

        # Ritual casting successful - no slot consumed
        logger.info(
            "ritual_cast",
            character=caster.name,
            spell=spell.name,
            extra_time="10 minutes",
        )

        # Handle concentration if needed
        concentration_broken = False
        if spell.concentration:
            concentration_broken = self.start_concentration(caster, spell)

        return SpellCastResult(
            success=True,
            spell=spell,
            slot_used=0,  # No slot consumed for rituals
            concentration_started=spell.concentration,
            concentration_broken=concentration_broken,
        )

    def get_ritual_casting_time(self, spell: SpellInfo) -> str:
        """Get the casting time for ritual casting (base + 10 minutes)."""
        base_time = spell.casting_time
        return f"{base_time} + 10 minutes (ritual)"


# Singleton instance
_manager: Optional[SpellcastingManager] = None


def get_spellcasting_manager() -> SpellcastingManager:
    """Get the singleton spellcasting manager."""
    global _manager
    if _manager is None:
        _manager = SpellcastingManager()
    return _manager
