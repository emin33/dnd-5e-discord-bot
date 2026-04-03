"""Dice rolling engine for D&D 5e.

Supports standard dice notation (e.g., "2d6+3", "1d20", "4d6kh3")
with advantage/disadvantage handling.
"""

import random
import re
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class DiceRoll:
    """Result of a dice roll."""

    notation: str
    dice_results: list[int] = field(default_factory=list)
    kept_dice: list[int] = field(default_factory=list)
    dropped_dice: list[int] = field(default_factory=list)
    modifier: int = 0
    total: int = 0
    reason: str = ""

    # For d20 rolls
    natural_20: bool = False
    natural_1: bool = False

    # For advantage/disadvantage
    advantage_rolls: Optional[list[int]] = None
    disadvantage_rolls: Optional[list[int]] = None
    roll_type: str = "normal"  # "normal", "advantage", "disadvantage"

    @property
    def dice_sum(self) -> int:
        """Sum of kept dice (before modifier)."""
        return sum(self.kept_dice)

    def __str__(self) -> str:
        """Human-readable representation."""
        if self.advantage_rolls:
            rolls_str = f"[{self.advantage_rolls[0]}, {self.advantage_rolls[1]}] → {self.kept_dice[0]}"
        elif self.disadvantage_rolls:
            rolls_str = f"[{self.disadvantage_rolls[0]}, {self.disadvantage_rolls[1]}] → {self.kept_dice[0]}"
        elif self.dropped_dice:
            all_dice = self.kept_dice + self.dropped_dice
            kept_str = ", ".join(str(d) for d in self.kept_dice)
            dropped_str = ", ".join(f"~~{d}~~" for d in self.dropped_dice)
            rolls_str = f"[{kept_str}, {dropped_str}]"
        else:
            rolls_str = f"[{', '.join(str(d) for d in self.kept_dice)}]"

        if self.modifier > 0:
            return f"{rolls_str} + {self.modifier} = **{self.total}**"
        elif self.modifier < 0:
            return f"{rolls_str} - {abs(self.modifier)} = **{self.total}**"
        else:
            return f"{rolls_str} = **{self.total}**"


class DiceRoller:
    """Dice rolling engine with D&D notation support."""

    # Regex pattern for dice notation
    # Matches: XdY, XdYkh/klZ, XdY+/-Z
    DICE_PATTERN = re.compile(
        r"^(\d+)?d(\d+)"  # NdM (N optional, defaults to 1)
        r"(?:(kh|kl|dh|dl)(\d+))?"  # Keep/drop highest/lowest
        r"(?:([+-])(\d+))?$",  # Modifier
        re.IGNORECASE,
    )

    def __init__(self, rng: Optional[random.Random] = None):
        """Initialize with optional custom RNG for testing."""
        self.rng = rng or random.Random()

    def roll(
        self,
        notation: str,
        advantage: bool = False,
        disadvantage: bool = False,
        reason: str = "",
    ) -> DiceRoll:
        """
        Roll dice using standard D&D notation.

        Args:
            notation: Dice notation (e.g., "1d20+5", "2d6", "4d6kh3")
            advantage: Roll 2d20, take higher (only for d20 rolls)
            disadvantage: Roll 2d20, take lower (only for d20 rolls)
            reason: Description of what the roll is for

        Returns:
            DiceRoll with complete roll information
        """
        notation = notation.strip().lower()
        match = self.DICE_PATTERN.match(notation)

        if not match:
            raise ValueError(f"Invalid dice notation: {notation}")

        num_dice = int(match.group(1)) if match.group(1) else 1
        die_size = int(match.group(2))

        if num_dice <= 0 or die_size <= 0:
            raise ValueError(f"Invalid dice notation: {notation} (dice count and size must be positive)")
        keep_drop = match.group(3)  # kh, kl, dh, dl
        keep_drop_count = int(match.group(4)) if match.group(4) else None
        mod_sign = match.group(5)  # + or -
        mod_value = int(match.group(6)) if match.group(6) else 0

        modifier = mod_value if mod_sign != "-" else -mod_value

        # Handle advantage/disadvantage for d20 rolls
        if (advantage or disadvantage) and num_dice == 1 and die_size == 20:
            return self._roll_with_advantage_disadvantage(
                notation, modifier, advantage, reason
            )

        # Roll the dice
        dice_results = [self.rng.randint(1, die_size) for _ in range(num_dice)]

        # Handle keep/drop mechanics
        kept_dice, dropped_dice = self._apply_keep_drop(
            dice_results, keep_drop, keep_drop_count
        )

        total = sum(kept_dice) + modifier

        # Check for natural 20/1 on d20
        natural_20 = die_size == 20 and num_dice == 1 and kept_dice[0] == 20
        natural_1 = die_size == 20 and num_dice == 1 and kept_dice[0] == 1

        return DiceRoll(
            notation=notation,
            dice_results=dice_results,
            kept_dice=kept_dice,
            dropped_dice=dropped_dice,
            modifier=modifier,
            total=total,
            reason=reason,
            natural_20=natural_20,
            natural_1=natural_1,
        )

    def _roll_with_advantage_disadvantage(
        self,
        notation: str,
        modifier: int,
        advantage: bool,
        reason: str,
    ) -> DiceRoll:
        """Roll d20 with advantage or disadvantage."""
        roll1 = self.rng.randint(1, 20)
        roll2 = self.rng.randint(1, 20)

        if advantage:
            kept = max(roll1, roll2)
            roll_type = "advantage"
            adv_rolls = [roll1, roll2]
            dis_rolls = None
        else:
            kept = min(roll1, roll2)
            roll_type = "disadvantage"
            adv_rolls = None
            dis_rolls = [roll1, roll2]

        total = kept + modifier

        return DiceRoll(
            notation=notation,
            dice_results=[roll1, roll2],
            kept_dice=[kept],
            dropped_dice=[roll1 if roll1 != kept else roll2],
            modifier=modifier,
            total=total,
            reason=reason,
            natural_20=(kept == 20),
            natural_1=(kept == 1),
            advantage_rolls=adv_rolls,
            disadvantage_rolls=dis_rolls,
            roll_type=roll_type,
        )

    def _apply_keep_drop(
        self,
        dice: list[int],
        keep_drop: Optional[str],
        count: Optional[int],
    ) -> tuple[list[int], list[int]]:
        """Apply keep/drop mechanics to dice results."""
        if not keep_drop:
            return dice.copy(), []

        sorted_dice = sorted(dice, reverse=True)
        count = count or 1

        if keep_drop == "kh":  # Keep highest
            kept = sorted_dice[:count]
            dropped = sorted_dice[count:]
        elif keep_drop == "kl":  # Keep lowest
            kept = sorted_dice[-count:]
            dropped = sorted_dice[:-count]
        elif keep_drop == "dh":  # Drop highest
            kept = sorted_dice[count:]
            dropped = sorted_dice[:count]
        elif keep_drop == "dl":  # Drop lowest
            kept = sorted_dice[:-count] if count < len(sorted_dice) else []
            dropped = sorted_dice[-count:]
        else:
            kept = dice.copy()
            dropped = []

        return kept, dropped

    def roll_ability_scores_4d6(self) -> list[DiceRoll]:
        """Roll 6 ability scores using 4d6 drop lowest method."""
        return [self.roll("4d6kh3", reason=f"Ability Score {i + 1}") for i in range(6)]

    def roll_stat_block(self) -> list[int]:
        """Roll and return 6 ability score totals using 4d6 drop lowest."""
        rolls = self.roll_ability_scores_4d6()
        return [r.total for r in rolls]

    def roll_initiative(self, modifier: int = 0) -> DiceRoll:
        """Roll initiative (1d20 + modifier)."""
        notation = f"1d20+{modifier}" if modifier >= 0 else f"1d20{modifier}"
        return self.roll(notation, reason="Initiative")

    def roll_attack(
        self,
        modifier: int = 0,
        advantage: bool = False,
        disadvantage: bool = False,
    ) -> DiceRoll:
        """Roll an attack (1d20 + modifier with optional advantage/disadvantage)."""
        notation = f"1d20+{modifier}" if modifier >= 0 else f"1d20{modifier}"
        return self.roll(
            notation,
            advantage=advantage,
            disadvantage=disadvantage,
            reason="Attack Roll",
        )

    def roll_damage(self, notation: str, critical: bool = False) -> DiceRoll:
        """
        Roll damage dice.

        For critical hits, doubles the number of dice rolled.
        """
        if critical:
            # Double the dice for critical hits
            match = self.DICE_PATTERN.match(notation.lower())
            if match:
                num_dice = int(match.group(1)) if match.group(1) else 1
                die_size = match.group(2)
                rest = notation[match.end(2) :]
                notation = f"{num_dice * 2}d{die_size}{rest}"

        return self.roll(notation, reason="Damage")

    def roll_save(
        self,
        modifier: int = 0,
        advantage: bool = False,
        disadvantage: bool = False,
    ) -> DiceRoll:
        """Roll a saving throw."""
        notation = f"1d20+{modifier}" if modifier >= 0 else f"1d20{modifier}"
        return self.roll(
            notation,
            advantage=advantage,
            disadvantage=disadvantage,
            reason="Saving Throw",
        )

    def roll_check(
        self,
        modifier: int = 0,
        advantage: bool = False,
        disadvantage: bool = False,
    ) -> DiceRoll:
        """Roll an ability/skill check."""
        notation = f"1d20+{modifier}" if modifier >= 0 else f"1d20{modifier}"
        return self.roll(
            notation,
            advantage=advantage,
            disadvantage=disadvantage,
            reason="Ability Check",
        )

    def roll_hit_dice(self, die_type: int, con_modifier: int = 0) -> DiceRoll:
        """Roll a hit die for healing during short rest."""
        notation = f"1d{die_type}+{con_modifier}" if con_modifier >= 0 else f"1d{die_type}{con_modifier}"
        result = self.roll(notation, reason="Hit Die")
        # Minimum 1 HP healed
        if result.total < 1:
            result.total = 1
        return result

    def roll_death_save(self) -> DiceRoll:
        """Roll a death saving throw (unmodified d20)."""
        return self.roll("1d20", reason="Death Save")


# Global dice roller instance
_roller: Optional[DiceRoller] = None


def get_roller() -> DiceRoller:
    """Get the global dice roller instance."""
    global _roller
    if _roller is None:
        _roller = DiceRoller()
    return _roller


def roll(notation: str, **kwargs) -> DiceRoll:
    """Convenience function to roll dice."""
    return get_roller().roll(notation, **kwargs)
