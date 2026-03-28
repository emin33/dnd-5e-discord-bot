"""Rest mechanics - short rest and long rest handling."""

from dataclasses import dataclass
from typing import Optional
import structlog

from ...models import Character, RestType, Condition
from .dice import get_roller, DiceRoll

logger = structlog.get_logger()


@dataclass
class HitDiceResult:
    """Result of spending a hit die during short rest."""

    roll: DiceRoll
    healing: int
    remaining_dice: int


@dataclass
class ShortRestResult:
    """Result of taking a short rest."""

    character_name: str
    hit_dice_spent: list[HitDiceResult]
    total_healing: int
    hp_before: int
    hp_after: int
    features_recovered: list[str]


@dataclass
class LongRestResult:
    """Result of taking a long rest."""

    character_name: str
    hp_restored: int
    hp_max: int
    spell_slots_restored: dict[int, int]  # level -> slots restored
    hit_dice_restored: int
    exhaustion_removed: int
    features_recovered: list[str]


class RestManager:
    """Manages rest mechanics for characters."""

    def __init__(self):
        self.roller = get_roller()

    def can_short_rest(self, character: Character) -> tuple[bool, str]:
        """Check if character can take a short rest."""
        # Check if already at full HP with no hit dice to spend
        if character.hp.current >= character.hp.maximum and character.hit_dice.remaining == 0:
            return False, "Already at full HP with no hit dice remaining"

        return True, ""

    def can_long_rest(self, character: Character) -> tuple[bool, str]:
        """Check if character can take a long rest."""
        # In a real implementation, would check if 24 hours have passed
        # For now, always allow
        return True, ""

    def short_rest(
        self,
        character: Character,
        hit_dice_to_spend: int = 0,
    ) -> ShortRestResult:
        """
        Take a short rest.

        - Spend hit dice to heal (optional)
        - Recover some class features (Warlock spell slots, Fighter Second Wind, etc.)
        """
        hp_before = character.hp.current
        hit_dice_results = []
        total_healing = 0

        # Spend hit dice
        dice_to_spend = min(hit_dice_to_spend, character.hit_dice.remaining)
        for _ in range(dice_to_spend):
            result = self._spend_hit_die(character)
            if result:
                hit_dice_results.append(result)
                total_healing += result.healing

        # Recover class features based on class
        features_recovered: list[str] = []
        features_recovered.extend(self._recover_short_rest_features(character))

        logger.info(
            "short_rest_taken",
            character=character.name,
            hit_dice_spent=len(hit_dice_results),
            healing=total_healing,
        )

        return ShortRestResult(
            character_name=character.name,
            hit_dice_spent=hit_dice_results,
            total_healing=total_healing,
            hp_before=hp_before,
            hp_after=character.hp.current,
            features_recovered=features_recovered,
        )

    def _recover_short_rest_features(self, character: Character) -> list[str]:
        """
        Recover class-specific features on short rest.

        D&D 5e short rest recoveries:
        - Warlock: All pact slots
        - Fighter: Second Wind (1/rest), Action Surge (2/rest at level 17+)
        - Monk: All ki points
        - Bard: Bardic Inspiration (at level 5+, uses = CHA mod)
        - Cleric (some domains): Channel Divinity
        - Druid: Wild Shape uses (some cases)
        """
        features = []
        class_index = character.class_index.lower() if character.class_index else ""
        level = character.level

        if class_index == "warlock":
            # Warlocks regain Pact Magic slots on short rest (levels 1-5 only).
            # IMPORTANT: Do NOT call restore_all() — that would also restore
            # multiclass slots from other classes (e.g., Cleric/Wizard).
            restored_any = False
            for slot_level in range(1, 6):
                current, max_slots = character.spell_slots.get_slots(slot_level)
                if max_slots > 0 and current < max_slots:
                    setattr(character.spell_slots, f"level_{slot_level}", (max_slots, max_slots))
                    restored_any = True
            if restored_any:
                features.append("Pact Magic slots")

        elif class_index == "fighter":
            # Second Wind: recover 1 use per short/long rest (level 1+)
            # Action Surge: recover 1 use per short/long rest (level 2+)
            if level >= 1:
                if not hasattr(character, "second_wind_used"):
                    character.second_wind_used = False
                if character.second_wind_used:
                    character.second_wind_used = False
                    features.append("Second Wind")

            if level >= 2:
                if not hasattr(character, "action_surge_used"):
                    character.action_surge_used = 0
                # At level 17+, fighters get 2 Action Surges
                max_surges = 2 if level >= 17 else 1
                if character.action_surge_used > 0:
                    character.action_surge_used = 0
                    features.append("Action Surge")

        elif class_index == "monk":
            # All ki points recovered on short rest
            if level >= 2:
                if not hasattr(character, "ki_points"):
                    character.ki_points = level  # Ki = monk level
                    character.ki_max = level
                else:
                    if character.ki_points < character.ki_max:
                        character.ki_points = character.ki_max
                        features.append(f"Ki points ({character.ki_max})")

        elif class_index == "bard":
            # Song of Rest happens automatically (healing bonus) - handled elsewhere
            # Bardic Inspiration: at level 5+, recharges on short rest
            if level >= 5:
                if not hasattr(character, "bardic_inspiration_uses"):
                    cha_mod = max(1, character.abilities.cha_mod)
                    character.bardic_inspiration_uses = cha_mod
                    character.bardic_inspiration_max = cha_mod
                else:
                    if character.bardic_inspiration_uses < character.bardic_inspiration_max:
                        character.bardic_inspiration_uses = character.bardic_inspiration_max
                        features.append("Bardic Inspiration")

        elif class_index == "cleric":
            # Channel Divinity: recover 1 use per short rest (level 2+)
            if level >= 2:
                if not hasattr(character, "channel_divinity_uses"):
                    max_uses = 1 if level < 6 else (2 if level < 18 else 3)
                    character.channel_divinity_uses = max_uses
                    character.channel_divinity_max = max_uses
                else:
                    if character.channel_divinity_uses < character.channel_divinity_max:
                        character.channel_divinity_uses = character.channel_divinity_max
                        features.append("Channel Divinity")

        elif class_index == "paladin":
            # Channel Divinity: recover 1 use per short rest (level 3+)
            if level >= 3:
                if not hasattr(character, "channel_divinity_uses"):
                    character.channel_divinity_uses = 1
                    character.channel_divinity_max = 1
                else:
                    if character.channel_divinity_uses < character.channel_divinity_max:
                        character.channel_divinity_uses = character.channel_divinity_max
                        features.append("Channel Divinity")

        elif class_index == "druid":
            # Wild Shape uses: 2 per short/long rest (level 2+)
            if level >= 2:
                if not hasattr(character, "wild_shape_uses"):
                    character.wild_shape_uses = 2
                    character.wild_shape_max = 2
                else:
                    if character.wild_shape_uses < character.wild_shape_max:
                        character.wild_shape_uses = character.wild_shape_max
                        features.append("Wild Shape")

        return features

    def _spend_hit_die(self, character: Character) -> Optional[HitDiceResult]:
        """Spend a single hit die and heal."""
        if character.hit_dice.remaining <= 0:
            return None

        if character.hp.current >= character.hp.maximum:
            return None

        # Spend the die
        character.hit_dice.spend(1)

        # Roll hit die + Constitution modifier
        die_type = character.hit_dice.die_type
        con_mod = character.abilities.con_mod

        roll = self.roller.roll(f"1d{die_type}+{con_mod}")

        # Minimum 1 HP restored
        healing = max(1, roll.total)

        # Apply healing
        actual = character.hp.heal(healing)

        return HitDiceResult(
            roll=roll,
            healing=actual,
            remaining_dice=character.hit_dice.remaining,
        )

    def long_rest(self, character: Character) -> LongRestResult:
        """
        Take a long rest.

        - Restore all HP
        - Restore all spell slots
        - Recover half hit dice (minimum 1)
        - Remove 1 level of exhaustion
        - Recover all class features
        """
        # Restore HP
        hp_restored = character.hp.maximum - character.hp.current
        character.hp.current = character.hp.maximum
        character.hp.temporary = 0  # Temp HP expires

        # Restore spell slots
        slots_restored: dict[int, int] = {}
        for level in range(1, 10):
            current, max_slots = character.spell_slots.get_slots(level)
            if max_slots > 0:
                restored = max_slots - current
                if restored > 0:
                    slots_restored[level] = restored

        character.spell_slots.restore_all()

        # Recover hit dice (half rounded up, minimum 1) — PHB p.186
        dice_to_recover = max(1, -(-character.hit_dice.total // 2))
        old_dice = character.hit_dice.remaining
        character.hit_dice.recover_long_rest()
        hit_dice_restored = character.hit_dice.remaining - old_dice

        # Remove exhaustion
        exhaustion_removed = 0
        for condition in character.conditions[:]:
            if condition.condition == Condition.EXHAUSTION:
                if condition.stacks > 1:
                    condition.stacks -= 1
                    exhaustion_removed = 1
                else:
                    character.conditions.remove(condition)
                    exhaustion_removed = 1
                break

        # Reset death saves
        character.death_saves.reset()

        # Break concentration
        character.concentration_spell_id = None

        # TODO: Recover class features
        features_recovered: list[str] = ["All class features"]

        logger.info(
            "long_rest_taken",
            character=character.name,
            hp_restored=hp_restored,
            slots_restored=slots_restored,
            hit_dice_restored=hit_dice_restored,
            exhaustion_removed=exhaustion_removed,
        )

        return LongRestResult(
            character_name=character.name,
            hp_restored=hp_restored,
            hp_max=character.hp.maximum,
            spell_slots_restored=slots_restored,
            hit_dice_restored=hit_dice_restored,
            exhaustion_removed=exhaustion_removed,
            features_recovered=features_recovered,
        )

    def add_exhaustion(self, character: Character, levels: int = 1) -> int:
        """
        Add exhaustion to a character.

        Returns new exhaustion level.
        """
        # Find existing exhaustion
        for condition in character.conditions:
            if condition.condition == Condition.EXHAUSTION:
                condition.stacks = min(6, condition.stacks + levels)
                if condition.stacks >= 6:
                    # 6 levels of exhaustion = death
                    logger.warning(
                        "character_died_exhaustion",
                        character=character.name,
                    )
                return condition.stacks

        # No existing exhaustion, add new
        from ...models import CharacterCondition

        exhaustion = CharacterCondition(
            condition=Condition.EXHAUSTION,
            source="Exhaustion",
            stacks=min(6, levels),
        )
        character.conditions.append(exhaustion)
        return exhaustion.stacks

    def remove_exhaustion(self, character: Character, levels: int = 1) -> int:
        """
        Remove exhaustion from a character.

        Returns new exhaustion level (0 if fully removed).
        """
        for condition in character.conditions:
            if condition.condition == Condition.EXHAUSTION:
                condition.stacks = max(0, condition.stacks - levels)
                if condition.stacks <= 0:
                    character.conditions.remove(condition)
                    return 0
                return condition.stacks

        return 0  # No exhaustion


# Singleton instance
_manager: Optional[RestManager] = None


def get_rest_manager() -> RestManager:
    """Get the singleton rest manager."""
    global _manager
    if _manager is None:
        _manager = RestManager()
    return _manager
