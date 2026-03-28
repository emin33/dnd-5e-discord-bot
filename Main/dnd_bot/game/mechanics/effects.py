"""Effect duration and auto-expire system for combat."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional, Callable, Any
import uuid

from ...models.common import AbilityScore, Condition


class EffectTiming(str, Enum):
    """When an effect triggers or expires."""
    START_OF_TURN = "start_of_turn"
    END_OF_TURN = "end_of_turn"
    START_OF_CASTER_TURN = "start_of_caster_turn"
    END_OF_CASTER_TURN = "end_of_caster_turn"


class EffectType(str, Enum):
    """Type of effect."""
    CONDITION = "condition"  # Applies a condition
    BUFF = "buff"  # Positive effect (Bless, Bardic Inspiration)
    DEBUFF = "debuff"  # Negative effect
    ONGOING_DAMAGE = "ongoing_damage"  # Damage each round
    AURA = "aura"  # Affects nearby creatures


@dataclass
class RepeatingSave:
    """Configuration for a save that repeats each round."""
    ability: AbilityScore
    dc: int
    timing: EffectTiming = EffectTiming.END_OF_TURN
    ends_on_success: bool = True


@dataclass
class OngoingDamage:
    """Configuration for damage that occurs each round."""
    dice: str  # e.g., "1d6", "2d8"
    damage_type: str  # e.g., "fire", "acid"
    timing: EffectTiming = EffectTiming.START_OF_TURN
    save_ability: Optional[AbilityScore] = None
    save_dc: Optional[int] = None
    half_on_save: bool = True


@dataclass
class ActiveEffect:
    """An active effect on a combatant with duration tracking."""

    id: str = field(default_factory=lambda: str(uuid.uuid4()))
    name: str = ""
    effect_type: EffectType = EffectType.CONDITION

    # Source tracking
    source_combatant_id: Optional[str] = None  # Who applied this effect
    source_spell_index: Optional[str] = None  # From what spell
    is_concentration: bool = False  # Ends if caster loses concentration

    # Duration
    duration_rounds: Optional[int] = None  # None = indefinite / until condition met
    rounds_remaining: Optional[int] = None
    expires_timing: EffectTiming = EffectTiming.END_OF_TURN

    # What it applies
    condition: Optional[Condition] = None  # If this applies a condition

    # Repeating saves
    repeating_save: Optional[RepeatingSave] = None

    # Ongoing damage
    ongoing_damage: Optional[OngoingDamage] = None

    # Bonus tracking (for Bless, Bardic Inspiration, etc.)
    bonus_dice: Optional[str] = None  # e.g., "1d4" for Bless
    bonus_to: list[str] = field(default_factory=list)  # ["attack", "save", "check"]

    # Metadata
    created_at: datetime = field(default_factory=datetime.utcnow)
    created_round: int = 1

    def should_process_at(self, timing: EffectTiming, is_source_turn: bool = False) -> bool:
        """Check if this effect should process at the given timing."""
        if timing == EffectTiming.START_OF_CASTER_TURN or timing == EffectTiming.END_OF_CASTER_TURN:
            return is_source_turn and self._timing_matches(timing)
        return self._timing_matches(timing)

    def _timing_matches(self, timing: EffectTiming) -> bool:
        """Check if timing matches for this effect."""
        # For repeating saves
        if self.repeating_save and self.repeating_save.timing == timing:
            return True
        # For ongoing damage
        if self.ongoing_damage and self.ongoing_damage.timing == timing:
            return True
        # For expiration
        if self.expires_timing == timing:
            return True
        return False

    def decrement_duration(self) -> bool:
        """
        Decrement duration by 1 round.
        Returns True if effect should expire (duration reached 0).
        """
        if self.rounds_remaining is None:
            return False
        self.rounds_remaining -= 1
        return self.rounds_remaining <= 0

    def is_expired(self) -> bool:
        """Check if effect has expired."""
        if self.rounds_remaining is None:
            return False
        return self.rounds_remaining <= 0


@dataclass
class EffectProcessResult:
    """Result of processing an effect for a turn."""
    effect_id: str
    effect_name: str

    # What happened
    expired: bool = False
    save_required: bool = False
    save_result: Optional[int] = None  # The roll result
    save_dc: Optional[int] = None
    save_succeeded: bool = False
    ended_by_save: bool = False

    # Damage (if ongoing)
    damage_dealt: int = 0
    damage_type: Optional[str] = None
    damage_halved: bool = False  # If save reduced damage

    # Messages for narration
    messages: list[str] = field(default_factory=list)


class EffectTracker:
    """Manages active effects on a combatant."""

    def __init__(self):
        self.effects: list[ActiveEffect] = []

    def add_effect(self, effect: ActiveEffect) -> None:
        """Add a new effect."""
        # Initialize rounds_remaining from duration if not set
        if effect.duration_rounds is not None and effect.rounds_remaining is None:
            effect.rounds_remaining = effect.duration_rounds
        self.effects.append(effect)

    def remove_effect(self, effect_id: str) -> bool:
        """Remove an effect by ID. Returns True if found and removed."""
        for i, e in enumerate(self.effects):
            if e.id == effect_id:
                self.effects.pop(i)
                return True
        return False

    def remove_effects_by_source(self, source_combatant_id: str) -> list[ActiveEffect]:
        """Remove all effects from a specific source (e.g., when concentration breaks)."""
        removed = []
        remaining = []
        for e in self.effects:
            if e.source_combatant_id == source_combatant_id and e.is_concentration:
                removed.append(e)
            else:
                remaining.append(e)
        self.effects = remaining
        return removed

    def remove_effects_by_condition(self, condition: Condition) -> list[ActiveEffect]:
        """Remove all effects that apply a specific condition."""
        removed = []
        remaining = []
        for e in self.effects:
            if e.condition == condition:
                removed.append(e)
            else:
                remaining.append(e)
        self.effects = remaining
        return removed

    def get_effects_by_type(self, effect_type: EffectType) -> list[ActiveEffect]:
        """Get all effects of a specific type."""
        return [e for e in self.effects if e.effect_type == effect_type]

    def get_active_conditions(self) -> list[Condition]:
        """Get all conditions currently applied by effects."""
        return [e.condition for e in self.effects if e.condition is not None]

    def has_condition(self, condition: Condition) -> bool:
        """Check if any effect applies the given condition."""
        return condition in self.get_active_conditions()

    def get_bonus_dice(self, bonus_type: str) -> list[str]:
        """Get all bonus dice that apply to a type (attack, save, check)."""
        dice = []
        for e in self.effects:
            if e.bonus_dice and bonus_type in e.bonus_to:
                dice.append(e.bonus_dice)
        return dice

    def process_start_of_turn(
        self,
        current_round: int,
        is_source_turn_fn: Callable[[str], bool],
        roll_save_fn: Callable[[AbilityScore, int], tuple[int, bool]],
        roll_damage_fn: Callable[[str], int],
    ) -> list[EffectProcessResult]:
        """
        Process effects at start of turn.

        Args:
            current_round: The current combat round
            is_source_turn_fn: Function that takes combatant_id and returns True if it's their turn
            roll_save_fn: Function that takes (ability, dc) and returns (roll_result, succeeded)
            roll_damage_fn: Function that takes dice notation and returns damage

        Returns:
            List of results for each processed effect
        """
        return self._process_effects(
            EffectTiming.START_OF_TURN,
            current_round,
            is_source_turn_fn,
            roll_save_fn,
            roll_damage_fn,
        )

    def process_end_of_turn(
        self,
        current_round: int,
        is_source_turn_fn: Callable[[str], bool],
        roll_save_fn: Callable[[AbilityScore, int], tuple[int, bool]],
        roll_damage_fn: Callable[[str], int],
    ) -> list[EffectProcessResult]:
        """
        Process effects at end of turn.

        Args:
            current_round: The current combat round
            is_source_turn_fn: Function that takes combatant_id and returns True if it's their turn
            roll_save_fn: Function that takes (ability, dc) and returns (roll_result, succeeded)
            roll_damage_fn: Function that takes dice notation and returns damage

        Returns:
            List of results for each processed effect
        """
        return self._process_effects(
            EffectTiming.END_OF_TURN,
            current_round,
            is_source_turn_fn,
            roll_save_fn,
            roll_damage_fn,
        )

    def _process_effects(
        self,
        timing: EffectTiming,
        current_round: int,
        is_source_turn_fn: Callable[[str], bool],
        roll_save_fn: Callable[[AbilityScore, int], tuple[int, bool]],
        roll_damage_fn: Callable[[str], int],
    ) -> list[EffectProcessResult]:
        """Internal method to process effects at a specific timing."""
        results = []
        effects_to_remove = []

        for effect in self.effects:
            is_source_turn = (
                effect.source_combatant_id is not None
                and is_source_turn_fn(effect.source_combatant_id)
            )

            if not effect.should_process_at(timing, is_source_turn):
                continue

            result = EffectProcessResult(
                effect_id=effect.id,
                effect_name=effect.name,
            )

            # Process ongoing damage
            if effect.ongoing_damage and effect.ongoing_damage.timing == timing:
                damage = roll_damage_fn(effect.ongoing_damage.dice)
                result.damage_type = effect.ongoing_damage.damage_type

                # Check for save to reduce damage
                if effect.ongoing_damage.save_ability and effect.ongoing_damage.save_dc:
                    roll, succeeded = roll_save_fn(
                        effect.ongoing_damage.save_ability,
                        effect.ongoing_damage.save_dc,
                    )
                    result.save_required = True
                    result.save_result = roll
                    result.save_dc = effect.ongoing_damage.save_dc
                    result.save_succeeded = succeeded

                    if succeeded and effect.ongoing_damage.half_on_save:
                        damage = damage // 2
                        result.damage_halved = True

                result.damage_dealt = damage
                result.messages.append(
                    f"Takes {damage} {effect.ongoing_damage.damage_type} damage from {effect.name}"
                )

            # Process repeating save
            if effect.repeating_save and effect.repeating_save.timing == timing:
                roll, succeeded = roll_save_fn(
                    effect.repeating_save.ability,
                    effect.repeating_save.dc,
                )
                result.save_required = True
                result.save_result = roll
                result.save_dc = effect.repeating_save.dc
                result.save_succeeded = succeeded

                if succeeded and effect.repeating_save.ends_on_success:
                    result.ended_by_save = True
                    effects_to_remove.append(effect.id)
                    result.messages.append(
                        f"Succeeded on save against {effect.name} - effect ends!"
                    )
                elif not succeeded:
                    result.messages.append(
                        f"Failed save against {effect.name} - effect continues"
                    )

            # Check for duration expiration
            if effect.expires_timing == timing:
                if effect.decrement_duration():
                    result.expired = True
                    effects_to_remove.append(effect.id)
                    result.messages.append(f"{effect.name} has expired")

            results.append(result)

        # Remove expired/ended effects
        for effect_id in effects_to_remove:
            self.remove_effect(effect_id)

        return results

    def clear_all(self) -> None:
        """Clear all effects."""
        self.effects.clear()


# Factory functions for common effects

def create_hold_person_effect(
    source_combatant_id: str,
    caster_save_dc: int,
    current_round: int,
) -> ActiveEffect:
    """Create a Hold Person effect with end-of-turn saves."""
    return ActiveEffect(
        name="Hold Person",
        effect_type=EffectType.CONDITION,
        source_combatant_id=source_combatant_id,
        source_spell_index="hold-person",
        is_concentration=True,
        duration_rounds=10,  # 1 minute = 10 rounds
        expires_timing=EffectTiming.END_OF_CASTER_TURN,
        condition=Condition.PARALYZED,
        repeating_save=RepeatingSave(
            ability=AbilityScore.WISDOM,
            dc=caster_save_dc,
            timing=EffectTiming.END_OF_TURN,
            ends_on_success=True,
        ),
        created_round=current_round,
    )


def create_bless_effect(
    source_combatant_id: str,
    current_round: int,
) -> ActiveEffect:
    """Create a Bless effect (1d4 to attacks and saves)."""
    return ActiveEffect(
        name="Bless",
        effect_type=EffectType.BUFF,
        source_combatant_id=source_combatant_id,
        source_spell_index="bless",
        is_concentration=True,
        duration_rounds=10,  # 1 minute
        expires_timing=EffectTiming.END_OF_CASTER_TURN,
        bonus_dice="1d4",
        bonus_to=["attack", "save"],
        created_round=current_round,
    )


def create_hex_effect(
    source_combatant_id: str,
    current_round: int,
    bonus_damage_dice: str = "1d6",
) -> ActiveEffect:
    """Create a Hex effect (bonus necrotic damage on hits)."""
    return ActiveEffect(
        name="Hex",
        effect_type=EffectType.DEBUFF,
        source_combatant_id=source_combatant_id,
        source_spell_index="hex",
        is_concentration=True,
        duration_rounds=10,  # 1 hour at base, but tracked in combat rounds
        expires_timing=EffectTiming.END_OF_CASTER_TURN,
        bonus_dice=bonus_damage_dice,
        bonus_to=["damage_on_hit"],
        created_round=current_round,
    )


def create_burning_effect(
    source: str,
    damage_dice: str = "1d6",
    save_dc: int = 10,
) -> ActiveEffect:
    """Create a burning/on fire effect with saves to end."""
    return ActiveEffect(
        name="Burning",
        effect_type=EffectType.ONGOING_DAMAGE,
        duration_rounds=None,  # Lasts until save succeeds
        ongoing_damage=OngoingDamage(
            dice=damage_dice,
            damage_type="fire",
            timing=EffectTiming.START_OF_TURN,
        ),
        repeating_save=RepeatingSave(
            ability=AbilityScore.DEXTERITY,
            dc=save_dc,
            timing=EffectTiming.END_OF_TURN,
            ends_on_success=True,
        ),
    )


def create_moonbeam_effect(
    source_combatant_id: str,
    caster_save_dc: int,
    current_round: int,
) -> ActiveEffect:
    """Create a Moonbeam effect (damage at start of turn or when entering)."""
    return ActiveEffect(
        name="Moonbeam",
        effect_type=EffectType.ONGOING_DAMAGE,
        source_combatant_id=source_combatant_id,
        source_spell_index="moonbeam",
        is_concentration=True,
        duration_rounds=10,  # 1 minute
        expires_timing=EffectTiming.END_OF_CASTER_TURN,
        ongoing_damage=OngoingDamage(
            dice="2d10",
            damage_type="radiant",
            timing=EffectTiming.START_OF_TURN,
            save_ability=AbilityScore.CONSTITUTION,
            save_dc=caster_save_dc,
            half_on_save=True,
        ),
        created_round=current_round,
    )
