"""Combat and combatant data models."""

from datetime import datetime
from typing import Optional, Any
import uuid

from pydantic import BaseModel, Field

from .common import ChannelId, CharacterId, CombatId, CombatState, Condition


class TurnResources(BaseModel):
    """Available resources for a combatant's turn."""

    action: bool = True
    bonus_action: bool = True
    reaction: bool = True  # Resets at start of turn
    movement: int = Field(default=30, ge=0)  # In feet
    free_interaction: bool = True  # Draw weapon, open door, etc.

    def use_action(self) -> bool:
        """Use action. Returns False if already used."""
        if not self.action:
            return False
        self.action = False
        return True

    def use_bonus_action(self) -> bool:
        """Use bonus action. Returns False if already used."""
        if not self.bonus_action:
            return False
        self.bonus_action = False
        return True

    def use_reaction(self) -> bool:
        """Use reaction. Returns False if already used."""
        if not self.reaction:
            return False
        self.reaction = False
        return True

    def use_movement(self, feet: int) -> bool:
        """Use movement. Returns False if insufficient."""
        if feet > self.movement:
            return False
        self.movement -= feet
        return True

    def use_free_interaction(self) -> bool:
        """Use free object interaction. Returns False if already used."""
        if not self.free_interaction:
            return False
        self.free_interaction = False
        return True

    def reset_for_new_turn(self, speed: int) -> None:
        """Reset resources for a new turn."""
        self.action = True
        self.bonus_action = True
        self.reaction = True  # Reaction resets at START of turn
        self.movement = speed
        self.free_interaction = True


class DeathSaves(BaseModel):
    """Death saving throw tracking for player characters."""

    successes: int = Field(default=0, ge=0, le=3)
    failures: int = Field(default=0, ge=0, le=3)

    @property
    def is_stable(self) -> bool:
        """Check if character has stabilized (3 successes)."""
        return self.successes >= 3

    @property
    def is_dead(self) -> bool:
        """Check if character is dead (3 failures)."""
        return self.failures >= 3

    def add_success(self, count: int = 1) -> None:
        """Add success(es) to death saves."""
        self.successes = min(3, self.successes + count)

    def add_failure(self, count: int = 1) -> None:
        """Add failure(s) to death saves."""
        self.failures = min(3, self.failures + count)

    def reset(self) -> None:
        """Reset death saves (when healed above 0 HP)."""
        self.successes = 0
        self.failures = 0


class Combatant(BaseModel):
    """A participant in combat (player or NPC/monster)."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    combat_id: CombatId
    name: str

    # Identity
    is_player: bool
    character_id: Optional[CharacterId] = None  # For players
    monster_index: Optional[str] = None  # SRD monster index for NPCs

    # Initiative
    initiative_roll: Optional[int] = None
    initiative_bonus: int = Field(default=0)
    turn_order: Optional[int] = None  # Computed after all rolls

    # Combat stats
    hp_max: int = Field(ge=1)
    hp_current: int = Field(ge=0)
    hp_temp: int = Field(default=0, ge=0)  # Temporary HP
    armor_class: int = Field(default=10, ge=0)
    speed: int = Field(default=30, ge=0)

    # Ability scores (for monsters, populated from SRD; players use Character model)
    ability_scores: dict[str, int] = Field(default_factory=dict)
    # e.g., {"str": 27, "dex": 10, "con": 25, "int": 16, "wis": 13, "cha": 21}

    # Save bonuses (explicit proficient saves from SRD stat block)
    save_bonuses: dict[str, int] = Field(default_factory=dict)
    # e.g., {"dex": 6, "con": 13, "wis": 7, "cha": 11}

    # Proficiency bonus
    proficiency_bonus: int = Field(default=2)

    # Damage modifiers
    resistances: list[str] = Field(default_factory=list)  # Damage types with resistance
    immunities: list[str] = Field(default_factory=list)  # Damage types with immunity
    vulnerabilities: list[str] = Field(default_factory=list)  # Damage types with vulnerability

    # Recharge abilities (for monsters)
    recharge_abilities: list["RechargeAbility"] = Field(default_factory=list)

    # Cover status
    cover: str = "none"  # "none", "half", "three_quarters", "full"

    # Status
    is_active: bool = True  # False = dead/fled/removed
    is_surprised: bool = False
    is_stable: bool = False  # For unconscious but stable characters

    # Death saves (for player characters)
    death_saves: DeathSaves = Field(default_factory=DeathSaves)

    # Turn resources (for current combatant)
    turn_resources: TurnResources = Field(default_factory=TurnResources)

    # Active effects (conditions, buffs, debuffs, ongoing damage)
    effects: list["CombatEffect"] = Field(default_factory=list)

    # Help advantage (from Help action)
    has_help_advantage: bool = False

    @property
    def is_conscious(self) -> bool:
        """Check if combatant is conscious (HP > 0)."""
        return self.hp_current > 0

    @property
    def is_dead(self) -> bool:
        """Check if combatant is dead (HP <= 0 for monsters, or death saves for players)."""
        if not self.is_player:
            return self.hp_current <= 0
        # Players die with 3 failed death saves
        return self.death_saves.is_dead

    @property
    def is_dying(self) -> bool:
        """Check if player is making death saves (0 HP, not stable, not dead)."""
        return (
            self.is_player
            and self.hp_current == 0
            and not self.is_stable
            and not self.death_saves.is_dead
        )

    def get_ability_modifier(self, ability: str) -> int:
        """Get ability modifier from raw score. ability is short form: 'str', 'dex', etc."""
        score = self.ability_scores.get(ability, 10)
        return (score - 10) // 2

    def get_save_modifier(self, ability: str) -> int:
        """Get saving throw modifier. Uses explicit save_bonuses if available, else raw ability mod."""
        if ability in self.save_bonuses:
            return self.save_bonuses[ability]
        return self.get_ability_modifier(ability)

    def take_damage(
        self,
        amount: int,
        damage_type: Optional[str] = None,
        is_critical: bool = False,
    ) -> tuple[int, bool, str]:
        """
        Apply damage. Handles temp HP, unconscious hits, and resistance/immunity/vulnerability.

        Args:
            amount: Base damage amount
            damage_type: Type of damage (fire, slashing, etc.) for resistance checks
            is_critical: Whether this is critical hit damage

        Returns:
            (actual_damage, caused_instant_death, modifier_applied)
            modifier_applied: "none", "resistance", "immunity", or "vulnerability"
        """
        if amount <= 0:
            return (0, False, "none")

        modifier_applied = "none"

        # Apply damage type modifiers
        if damage_type:
            damage_type_lower = damage_type.lower()
            if damage_type_lower in [i.lower() for i in self.immunities]:
                return (0, False, "immunity")
            elif damage_type_lower in [r.lower() for r in self.resistances]:
                amount = amount // 2  # Round down
                modifier_applied = "resistance"
            elif damage_type_lower in [v.lower() for v in self.vulnerabilities]:
                amount = amount * 2
                modifier_applied = "vulnerability"

        remaining = amount
        instant_death = False

        # Temp HP absorbs damage first
        if self.hp_temp > 0:
            if remaining <= self.hp_temp:
                self.hp_temp -= remaining
                return (amount, False, modifier_applied)
            else:
                remaining -= self.hp_temp
                self.hp_temp = 0

        # Was already at 0 HP (unconscious player taking damage)
        if self.hp_current == 0 and self.is_player:
            # Damage while unconscious = auto death save failure
            failures = 2 if is_critical else 1
            self.death_saves.add_failure(failures)
            return (amount, self.death_saves.is_dead, modifier_applied)

        # Apply remaining damage to HP
        old_hp = self.hp_current
        self.hp_current = max(0, self.hp_current - remaining)

        # Check for instant death (remaining damage >= max HP)
        if self.hp_current == 0 and self.is_player:
            overflow = remaining - old_hp
            if overflow >= self.hp_max:
                instant_death = True
                self.death_saves.failures = 3

        return (amount, instant_death, modifier_applied)

    def heal(self, amount: int) -> int:
        """Heal. Returns actual HP restored. Clears death saves if brought above 0."""
        if amount <= 0:
            return 0

        old = self.hp_current
        self.hp_current = min(self.hp_max, self.hp_current + amount)

        # Regaining HP clears death saves and stable status
        if old == 0 and self.hp_current > 0:
            self.death_saves.reset()
            self.is_stable = False

        return self.hp_current - old

    def add_temp_hp(self, amount: int) -> int:
        """Add temporary HP. Only keeps higher value. Returns new temp HP total."""
        if amount <= 0:
            return self.hp_temp
        # Temp HP doesn't stack - keep the higher value
        self.hp_temp = max(self.hp_temp, amount)
        return self.hp_temp

    def stabilize(self) -> None:
        """Stabilize an unconscious creature (e.g., via Medicine check or spare the dying)."""
        if self.hp_current == 0:
            self.is_stable = True
            self.death_saves.reset()

    # ==================== Effect Management ====================

    def add_effect(self, effect: "CombatEffect") -> None:
        """Add an effect to this combatant."""
        # Initialize rounds_remaining if not set
        if effect.duration_rounds is not None and effect.rounds_remaining is None:
            effect.rounds_remaining = effect.duration_rounds
        self.effects.append(effect)

    def remove_effect(self, effect_id: str) -> bool:
        """Remove an effect by ID. Returns True if removed."""
        for i, e in enumerate(self.effects):
            if e.id == effect_id:
                self.effects.pop(i)
                return True
        return False

    def remove_effects_by_source(self, source_combatant_id: str, concentration_only: bool = True) -> list["CombatEffect"]:
        """Remove effects from a source (e.g., when concentration breaks)."""
        removed = []
        remaining = []
        for e in self.effects:
            if e.source_combatant_id == source_combatant_id:
                if not concentration_only or e.is_concentration:
                    removed.append(e)
                    continue
            remaining.append(e)
        self.effects = remaining
        return removed

    def get_active_conditions(self) -> list[Condition]:
        """Get all conditions currently applied by effects."""
        return [e.condition for e in self.effects if e.condition is not None]

    def has_effect_condition(self, condition: Condition) -> bool:
        """Check if any effect applies the given condition."""
        return condition in self.get_active_conditions()

    def get_bonus_dice(self, applies_to: str) -> list[str]:
        """Get bonus dice that apply to a roll type (attack, save, damage_on_hit)."""
        return [e.bonus_dice for e in self.effects if e.bonus_dice and applies_to in e.bonus_applies_to]

    def consume_help_advantage(self) -> bool:
        """Consume help advantage if available. Returns True if it was available."""
        if self.has_help_advantage:
            self.has_help_advantage = False
            return True
        return False

    # ==================== Recharge Abilities ====================

    def use_recharge_ability(self, ability_name: str) -> bool:
        """
        Use a recharge ability, marking it as unavailable.
        Returns True if the ability was available and is now used.
        """
        for ability in self.recharge_abilities:
            if ability.name.lower() == ability_name.lower():
                if ability.is_available:
                    ability.is_available = False
                    return True
                return False
        return False

    def get_recharge_ability(self, ability_name: str) -> Optional["RechargeAbility"]:
        """Get a recharge ability by name."""
        for ability in self.recharge_abilities:
            if ability.name.lower() == ability_name.lower():
                return ability
        return None

    def get_available_recharge_abilities(self) -> list["RechargeAbility"]:
        """Get all currently available recharge abilities."""
        return [a for a in self.recharge_abilities if a.is_available]

    def get_unavailable_recharge_abilities(self) -> list["RechargeAbility"]:
        """Get all currently unavailable (used) recharge abilities."""
        return [a for a in self.recharge_abilities if not a.is_available]


class RechargeAbility(BaseModel):
    """A recharge ability (like Dragon Breath) that needs to roll to recharge."""

    name: str
    recharge_on: int = Field(ge=1, le=6, default=5)  # Recharges on this roll or higher (e.g., 5 = 5-6)
    is_available: bool = True  # Whether the ability is currently available
    description: Optional[str] = None  # Ability description


class CombatEffect(BaseModel):
    """An active effect on a combatant during combat."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    name: str
    effect_type: str = "condition"  # condition, buff, debuff, ongoing_damage

    # Source tracking
    source_combatant_id: Optional[str] = None
    source_spell_index: Optional[str] = None
    is_concentration: bool = False

    # Duration
    duration_rounds: Optional[int] = None
    rounds_remaining: Optional[int] = None
    expires_at_end_of_turn: bool = True  # vs start of turn

    # What it applies
    condition: Optional[Condition] = None

    # Repeating save to end effect
    save_ability: Optional[str] = None  # "wisdom", "constitution", etc.
    save_dc: Optional[int] = None
    save_ends_on_success: bool = True
    save_at_end_of_turn: bool = True

    # Ongoing damage
    damage_dice: Optional[str] = None  # "2d6"
    damage_type: Optional[str] = None  # "fire"
    damage_save_for_half: bool = False

    # Bonus dice (for Bless, etc.)
    bonus_dice: Optional[str] = None
    bonus_applies_to: list[str] = Field(default_factory=list)

    # Metadata
    created_round: int = 1


class ReadiedAction(BaseModel):
    """A readied action waiting for a trigger."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    combatant_id: str
    combat_id: CombatId
    trigger_condition: str
    action_description: str
    spell_index: Optional[str] = None  # If readying a spell
    created_round: int


class Combat(BaseModel):
    """An active combat encounter."""

    id: CombatId = Field(default_factory=lambda: str(uuid.uuid4()))
    session_id: str
    channel_id: ChannelId

    # State machine
    state: CombatState = Field(default=CombatState.IDLE)

    # Round and turn tracking
    current_round: int = Field(default=1, ge=1)
    current_turn_index: int = Field(default=0, ge=0)

    # Combatants
    combatants: list[Combatant] = Field(default_factory=list)

    # Readied actions
    readied_actions: list[ReadiedAction] = Field(default_factory=list)

    # Description
    encounter_name: Optional[str] = None
    encounter_description: Optional[str] = None

    # Metadata
    created_at: datetime = Field(default_factory=datetime.utcnow)
    ended_at: Optional[datetime] = None

    # Valid state transitions
    VALID_TRANSITIONS: dict[CombatState, list[CombatState]] = {
        CombatState.IDLE: [CombatState.SETUP],
        CombatState.SETUP: [CombatState.ROLLING_INITIATIVE, CombatState.IDLE],
        CombatState.ROLLING_INITIATIVE: [CombatState.ACTIVE],
        CombatState.ACTIVE: [CombatState.AWAITING_ACTION, CombatState.COMBAT_END],
        CombatState.AWAITING_ACTION: [CombatState.RESOLVING_ACTION, CombatState.END_TURN],
        CombatState.RESOLVING_ACTION: [CombatState.AWAITING_ACTION, CombatState.END_TURN],
        CombatState.END_TURN: [CombatState.ACTIVE, CombatState.COMBAT_END],
        CombatState.COMBAT_END: [CombatState.IDLE],
    }

    def can_transition(self, new_state: CombatState) -> bool:
        """Check if transition to new state is valid."""
        return new_state in self.VALID_TRANSITIONS.get(self.state, [])

    def transition(self, new_state: CombatState) -> bool:
        """Transition to new state. Returns False if invalid."""
        if not self.can_transition(new_state):
            return False
        self.state = new_state
        return True

    def get_sorted_combatants(self) -> list[Combatant]:
        """Get combatants sorted by turn order."""
        return sorted(
            [c for c in self.combatants if c.turn_order is not None],
            key=lambda c: c.turn_order or 0,
        )

    def get_current_combatant(self) -> Optional[Combatant]:
        """Get the combatant whose turn it is."""
        sorted_combatants = self.get_sorted_combatants()
        if not sorted_combatants or self.current_turn_index >= len(sorted_combatants):
            return None
        return sorted_combatants[self.current_turn_index]

    def get_active_combatants(self) -> list[Combatant]:
        """Get all active (alive, not fled) combatants."""
        return [c for c in self.combatants if c.is_active and c.is_conscious]

    def add_combatant(self, combatant: Combatant) -> None:
        """Add a combatant to the encounter."""
        combatant.combat_id = self.id
        self.combatants.append(combatant)

    def remove_combatant(self, combatant_id: str) -> bool:
        """Mark a combatant as inactive. Returns False if not found."""
        for c in self.combatants:
            if c.id == combatant_id:
                c.is_active = False
                return True
        return False

    def roll_all_initiative(self) -> None:
        """Sort combatants by initiative (should be called after all have rolled)."""
        # Sort by: initiative roll (desc), then initiative bonus (desc), then name (asc)
        sorted_list = sorted(
            self.combatants,
            key=lambda c: (-(c.initiative_roll or 0), -c.initiative_bonus, c.name),
        )
        for i, combatant in enumerate(sorted_list):
            combatant.turn_order = i

    def next_turn(self) -> Optional[Combatant]:
        """Advance to next turn. Returns the new current combatant."""
        sorted_combatants = self.get_sorted_combatants()
        if not sorted_combatants:
            return None

        # Move to next combatant
        self.current_turn_index += 1

        # If we've gone through everyone, start a new round
        if self.current_turn_index >= len(sorted_combatants):
            self.current_turn_index = 0
            self.current_round += 1

        current = self.get_current_combatant()
        if current:
            current.turn_resources.reset_for_new_turn(current.speed)

        return current

    def is_combat_over(self) -> bool:
        """Check if combat should end (one side defeated)."""
        players_alive = any(
            c.is_active and c.is_conscious and c.is_player for c in self.combatants
        )
        enemies_alive = any(
            c.is_active and c.is_conscious and not c.is_player for c in self.combatants
        )
        return not players_alive or not enemies_alive
