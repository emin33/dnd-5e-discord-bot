"""Character data model."""

from datetime import datetime
from typing import Optional
import uuid

from pydantic import BaseModel, Field, computed_field

from .common import (
    AbilityScore,
    CharacterId,
    CampaignId,
    Condition,
    Skill,
    UserId,
    calculate_modifier,
    calculate_proficiency_bonus,
)


class AbilityScores(BaseModel):
    """Character's six ability scores."""

    strength: int = Field(ge=1, le=30, default=10)
    dexterity: int = Field(ge=1, le=30, default=10)
    constitution: int = Field(ge=1, le=30, default=10)
    intelligence: int = Field(ge=1, le=30, default=10)
    wisdom: int = Field(ge=1, le=30, default=10)
    charisma: int = Field(ge=1, le=30, default=10)

    def get_score(self, ability: AbilityScore) -> int:
        """Get score for a specific ability."""
        return getattr(self, ability.name.lower())

    def get_modifier(self, ability: AbilityScore) -> int:
        """Get modifier for a specific ability."""
        return calculate_modifier(self.get_score(ability))

    @computed_field
    @property
    def str_mod(self) -> int:
        return calculate_modifier(self.strength)

    @computed_field
    @property
    def dex_mod(self) -> int:
        return calculate_modifier(self.dexterity)

    @computed_field
    @property
    def con_mod(self) -> int:
        return calculate_modifier(self.constitution)

    @computed_field
    @property
    def int_mod(self) -> int:
        return calculate_modifier(self.intelligence)

    @computed_field
    @property
    def wis_mod(self) -> int:
        return calculate_modifier(self.wisdom)

    @computed_field
    @property
    def cha_mod(self) -> int:
        return calculate_modifier(self.charisma)


class CharacterCondition(BaseModel):
    """An active condition on a character."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    condition: Condition
    source: str
    applied_at: datetime = Field(default_factory=datetime.utcnow)
    expires_round: Optional[int] = None  # For combat-based durations
    expires_time: Optional[datetime] = None  # For time-based durations
    combat_id: Optional[str] = None
    stacks: int = Field(default=1, ge=1, le=6)  # For exhaustion (max 6)


class SpellSlots(BaseModel):
    """Spell slot tracking for a character."""

    level_1: tuple[int, int] = (0, 0)  # (current, max)
    level_2: tuple[int, int] = (0, 0)
    level_3: tuple[int, int] = (0, 0)
    level_4: tuple[int, int] = (0, 0)
    level_5: tuple[int, int] = (0, 0)
    level_6: tuple[int, int] = (0, 0)
    level_7: tuple[int, int] = (0, 0)
    level_8: tuple[int, int] = (0, 0)
    level_9: tuple[int, int] = (0, 0)

    def get_slots(self, level: int) -> tuple[int, int]:
        """Get (current, max) slots for a spell level."""
        if level < 1 or level > 9:
            raise ValueError(f"Invalid spell level: {level}")
        return getattr(self, f"level_{level}")

    def has_slot(self, level: int) -> bool:
        """Check if a slot of the given level is available."""
        current, _ = self.get_slots(level)
        return current > 0

    def expend_slot(self, level: int) -> bool:
        """Expend a slot. Returns True if successful."""
        if not self.has_slot(level):
            return False
        current, max_slots = self.get_slots(level)
        setattr(self, f"level_{level}", (current - 1, max_slots))
        return True

    def restore_slot(self, level: int) -> bool:
        """Restore a slot. Returns True if successful. Cannot exceed maximum."""
        current, max_slots = self.get_slots(level)
        if current >= max_slots:
            return False
        setattr(self, f"level_{level}", (min(current + 1, max_slots), max_slots))
        return True

    def set_slots(self, level: int, current: int) -> None:
        """Set current slots for a level. Clamped to [0, max]."""
        _, max_slots = self.get_slots(level)
        setattr(self, f"level_{level}", (max(0, min(current, max_slots)), max_slots))

    def restore_all(self) -> None:
        """Restore all slots to maximum (for long rest)."""
        for level in range(1, 10):
            _, max_slots = self.get_slots(level)
            setattr(self, f"level_{level}", (max_slots, max_slots))


class HitPoints(BaseModel):
    """Character hit points tracking."""

    maximum: int = Field(ge=1)
    current: int = Field(ge=0)
    temporary: int = Field(default=0, ge=0)

    def take_damage(self, amount: int) -> int:
        """Apply damage. Returns actual damage taken after temp HP."""
        if amount <= 0:
            return 0

        # Temp HP absorbs first
        if self.temporary > 0:
            if amount <= self.temporary:
                self.temporary -= amount
                return 0
            else:
                remaining = amount - self.temporary
                self.temporary = 0
                self.current = max(0, self.current - remaining)
                return remaining
        else:
            actual = min(amount, self.current)
            self.current = max(0, self.current - amount)
            return actual

    def heal(self, amount: int) -> int:
        """Heal damage. Returns actual HP restored."""
        if amount <= 0:
            return 0
        old = self.current
        self.current = min(self.maximum, self.current + amount)
        return self.current - old

    def add_temp_hp(self, amount: int) -> None:
        """Add temporary HP. Temp HP doesn't stack - take higher."""
        self.temporary = max(self.temporary, amount)

    @property
    def is_unconscious(self) -> bool:
        """Character is unconscious at 0 HP."""
        return self.current == 0

    @property
    def percentage(self) -> float:
        """Current HP as percentage of max."""
        return (self.current / self.maximum) * 100


class DeathSaves(BaseModel):
    """Death saving throw tracking."""

    successes: int = Field(default=0, ge=0, le=3)
    failures: int = Field(default=0, ge=0, le=3)

    def add_success(self, count: int = 1) -> None:
        """Add success(es). Nat 20 = 2 successes + revive."""
        self.successes = min(3, self.successes + count)

    def add_failure(self, count: int = 1) -> None:
        """Add failure(s). Nat 1 = 2 failures."""
        self.failures = min(3, self.failures + count)

    def reset(self) -> None:
        """Reset death saves (on stabilize or regain HP)."""
        self.successes = 0
        self.failures = 0

    @property
    def is_stable(self) -> bool:
        """Stabilized with 3 successes."""
        return self.successes >= 3

    @property
    def is_dead(self) -> bool:
        """Dead with 3 failures."""
        return self.failures >= 3


class HitDice(BaseModel):
    """Hit dice tracking for rests."""

    die_type: int = Field(ge=6, le=12)  # d6, d8, d10, or d12
    total: int = Field(ge=1)
    remaining: int = Field(ge=0)

    def spend(self, count: int = 1) -> int:
        """Spend hit dice. Returns number actually spent."""
        spent = min(count, self.remaining)
        self.remaining -= spent
        return spent

    def recover(self, count: int) -> None:
        """Recover hit dice (half on long rest, minimum 1)."""
        self.remaining = min(self.total, self.remaining + count)

    def recover_long_rest(self) -> None:
        """Recover hit dice for long rest (half rounded up, minimum 1). PHB p.186."""
        to_recover = max(1, -(-self.total // 2))  # Ceiling division
        self.recover(to_recover)


class CharacterProficiency(BaseModel):
    """A proficiency the character has."""

    proficiency_index: str  # SRD index like "skill-athletics"
    proficiency_type: str  # 'skill', 'save', 'tool', 'weapon', 'armor'
    expertise: bool = False  # Double proficiency bonus (rogues, bards)


class Character(BaseModel):
    """Full character data model."""

    # Identity
    id: CharacterId = Field(default_factory=lambda: str(uuid.uuid4()))
    discord_user_id: UserId
    campaign_id: CampaignId
    name: str = Field(min_length=1, max_length=64)

    # Core info
    race_index: str  # SRD index: "dwarf", "elf", etc.
    class_index: str  # SRD index: "fighter", "wizard", etc.
    subclass_index: Optional[str] = None
    level: int = Field(default=1, ge=1, le=20)
    experience: int = Field(default=0, ge=0)
    background_index: Optional[str] = None

    # Ability scores
    abilities: AbilityScores = Field(default_factory=AbilityScores)

    # Combat stats
    armor_class: int = Field(default=10, ge=0)
    speed: int = Field(default=30, ge=0)
    initiative_bonus: int = Field(default=0)

    # Hit points
    hp: HitPoints
    hit_dice: HitDice
    death_saves: DeathSaves = Field(default_factory=DeathSaves)

    # Spellcasting
    spellcasting_ability: Optional[AbilityScore] = None
    spell_slots: SpellSlots = Field(default_factory=SpellSlots)
    concentration_spell_id: Optional[str] = None
    known_spells: list[str] = Field(default_factory=list)  # SRD spell indices
    prepared_spells: list[str] = Field(default_factory=list)

    # Proficiencies
    proficiencies: list[CharacterProficiency] = Field(default_factory=list)
    saving_throw_proficiencies: list[AbilityScore] = Field(default_factory=list)
    skill_proficiencies: list[Skill] = Field(default_factory=list)
    skill_expertise: list[Skill] = Field(default_factory=list)

    # Conditions
    conditions: list[CharacterCondition] = Field(default_factory=list)

    # Metadata
    is_active: bool = True
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)

    @computed_field
    @property
    def proficiency_bonus(self) -> int:
        """Proficiency bonus based on level."""
        return calculate_proficiency_bonus(self.level)

    @computed_field
    @property
    def spell_save_dc(self) -> Optional[int]:
        """Spell save DC: 8 + proficiency + spellcasting mod."""
        if self.spellcasting_ability is None:
            return None
        mod = self.abilities.get_modifier(self.spellcasting_ability)
        return 8 + self.proficiency_bonus + mod

    @computed_field
    @property
    def spell_attack_bonus(self) -> Optional[int]:
        """Spell attack bonus: proficiency + spellcasting mod."""
        if self.spellcasting_ability is None:
            return None
        mod = self.abilities.get_modifier(self.spellcasting_ability)
        return self.proficiency_bonus + mod

    @computed_field
    @property
    def passive_perception(self) -> int:
        """Passive Perception: 10 + Wisdom mod + proficiency if proficient."""
        base = 10 + self.abilities.wis_mod
        if Skill.PERCEPTION in self.skill_proficiencies:
            base += self.proficiency_bonus
            if Skill.PERCEPTION in self.skill_expertise:
                base += self.proficiency_bonus
        return base

    @computed_field
    @property
    def is_concentrating(self) -> bool:
        """Whether the character is currently concentrating on a spell."""
        return self.concentration_spell_id is not None

    @computed_field
    @property
    def concentration_spell(self) -> Optional[str]:
        """Alias for concentration_spell_id for compatibility."""
        return self.concentration_spell_id

    def get_skill_modifier(self, skill: Skill) -> int:
        """Get total modifier for a skill check."""
        from .common import SKILL_ABILITIES

        ability = SKILL_ABILITIES[skill]
        mod = self.abilities.get_modifier(ability)

        if skill in self.skill_proficiencies:
            mod += self.proficiency_bonus
            if skill in self.skill_expertise:
                mod += self.proficiency_bonus

        return mod

    @property
    def ability_scores(self) -> dict[str, int]:
        """Compatibility property: return abilities as dict (matches Combatant interface)."""
        return {
            "str": self.abilities.strength,
            "dex": self.abilities.dexterity,
            "con": self.abilities.constitution,
            "int": self.abilities.intelligence,
            "wis": self.abilities.wisdom,
            "cha": self.abilities.charisma,
        }

    def get_save_modifier(self, ability: AbilityScore) -> int:
        """Get total modifier for a saving throw."""
        mod = self.abilities.get_modifier(ability)
        if ability in self.saving_throw_proficiencies:
            mod += self.proficiency_bonus
        return mod

    def has_condition(self, condition: Condition) -> bool:
        """Check if character has a specific condition."""
        return any(c.condition == condition for c in self.conditions)

    def get_exhaustion_level(self) -> int:
        """Get current exhaustion level (0-6)."""
        for c in self.conditions:
            if c.condition == Condition.EXHAUSTION:
                return c.stacks
        return 0

    def calculate_ac_from_equipment(self, equipped_armor: list[dict]) -> int:
        """Calculate AC from equipped armor/shield items.

        Args:
            equipped_armor: List of SRD equipment dicts with armor_class data.
                Each should have: armor_class.base, armor_class.dex_bonus, armor_category

        Returns:
            Calculated AC. Falls back to 10 + DEX (unarmored) if no armor.

        D&D 5e rules:
        - No armor: 10 + DEX modifier
        - Light armor: base + DEX modifier
        - Medium armor: base + DEX modifier (max +2)
        - Heavy armor: base (no DEX)
        - Shield: +2 (stacks with armor)
        """
        dex_mod = self.abilities.dex_mod
        base_ac = 10 + dex_mod  # Unarmored default
        shield_bonus = 0

        for item in equipped_armor:
            ac_data = item.get("armor_class", {})
            category = item.get("armor_category", "").lower()

            if category == "shield":
                shield_bonus = ac_data.get("base", 2)
                continue

            ac_base = ac_data.get("base", 10)
            has_dex = ac_data.get("dex_bonus", True)

            if not has_dex:
                # Heavy armor — no DEX bonus
                base_ac = ac_base
            elif category == "medium":
                # Medium armor — DEX bonus capped at +2
                base_ac = ac_base + min(dex_mod, 2)
            else:
                # Light armor — full DEX bonus
                base_ac = ac_base + dex_mod

        return base_ac + shield_bonus

    def summary(self) -> str:
        """Brief summary for context."""
        return (
            f"{self.name} (Level {self.level} {self.race_index} {self.class_index}) - "
            f"HP: {self.hp.current}/{self.hp.maximum}"
        )
