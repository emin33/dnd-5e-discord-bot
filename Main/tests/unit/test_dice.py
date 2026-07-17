"""Tests for the dice rolling mechanics."""

import random

import pytest
import structlog.testing

from dnd_bot.game.mechanics.dice import DiceRoller


class ScriptedRNG(random.Random):
    """Deterministic RNG returning a scripted sequence from randint().

    Determinism pattern for dice tests: DiceRoller accepts an ``rng``
    (random.Random) in its constructor, so inject one of these with the
    exact die faces you want, e.g. ``DiceRoller(rng=ScriptedRNG([20]))``.
    """

    def __new__(cls, values: list[int]) -> "ScriptedRNG":
        # random.Random.__new__ tries to seed with the first arg; bypass it.
        return super().__new__(cls)

    def __init__(self, values: list[int]):
        super().__init__()
        self._values = list(values)

    def randint(self, a: int, b: int) -> int:  # noqa: ARG002
        return self._values.pop(0)


class TestDiceRoller:
    """Tests for DiceRoller class."""

    @pytest.fixture
    def roller(self):
        return DiceRoller()

    def test_simple_d20(self, roller):
        """Test rolling 1d20."""
        result = roller.roll("1d20")
        assert 1 <= result.total <= 20
        assert len(result.kept_dice) == 1
        assert result.modifier == 0

    def test_d20_with_modifier(self, roller):
        """Test rolling 1d20+5."""
        result = roller.roll("1d20+5")
        assert 6 <= result.total <= 25
        assert result.modifier == 5

    def test_d20_negative_modifier(self, roller):
        """Test rolling 1d20-3."""
        result = roller.roll("1d20-3")
        assert -2 <= result.total <= 17
        assert result.modifier == -3

    def test_multiple_dice(self, roller):
        """Test rolling 2d6."""
        result = roller.roll("2d6")
        assert 2 <= result.total <= 12
        assert len(result.kept_dice) == 2

    def test_multiple_dice_with_modifier(self, roller):
        """Test rolling 2d6+3."""
        result = roller.roll("2d6+3")
        assert 5 <= result.total <= 15
        assert result.modifier == 3

    def test_keep_highest(self, roller):
        """Test rolling 4d6kh3 (keep highest 3)."""
        result = roller.roll("4d6kh3")
        assert 3 <= result.total <= 18
        assert len(result.kept_dice) == 3
        assert len(result.dropped_dice) == 1
        # Dropped die should be <= minimum kept die
        assert result.dropped_dice[0] <= min(result.kept_dice)

    def test_keep_lowest(self, roller):
        """Test rolling 2d20kl1 (keep lowest)."""
        result = roller.roll("2d20kl1")
        assert 1 <= result.total <= 20
        assert len(result.kept_dice) == 1

    def test_advantage(self, roller):
        """Test rolling with advantage."""
        result = roller.roll("1d20", advantage=True)
        assert 1 <= result.total <= 20
        assert result.roll_type == "advantage"
        assert result.advantage_rolls is not None
        assert len(result.advantage_rolls) == 2
        # Result should be the higher of the two rolls
        assert result.kept_dice[0] == max(result.advantage_rolls)

    def test_disadvantage(self, roller):
        """Test rolling with disadvantage."""
        result = roller.roll("1d20", disadvantage=True)
        assert 1 <= result.total <= 20
        assert result.roll_type == "disadvantage"
        assert result.disadvantage_rolls is not None
        assert len(result.disadvantage_rolls) == 2
        # Result should be the lower of the two rolls
        assert result.kept_dice[0] == min(result.disadvantage_rolls)

    def test_natural_20_detection(self):
        """A rolled 20 on 1d20 sets natural_20 (deterministic via scripted RNG)."""
        roller = DiceRoller(rng=ScriptedRNG([20]))
        result = roller.roll("1d20")
        assert result.kept_dice[0] == 20
        assert result.natural_20 is True
        assert result.natural_1 is False

    def test_natural_20_not_set_on_other_rolls(self):
        """A rolled 19 must not set natural_20."""
        roller = DiceRoller(rng=ScriptedRNG([19]))
        result = roller.roll("1d20")
        assert result.natural_20 is False
        assert result.natural_1 is False

    def test_natural_1_detection(self):
        """A rolled 1 on 1d20 sets natural_1 (deterministic via scripted RNG)."""
        roller = DiceRoller(rng=ScriptedRNG([1]))
        result = roller.roll("1d20")
        assert result.kept_dice[0] == 1
        assert result.natural_1 is True
        assert result.natural_20 is False

    def test_natural_1_not_set_on_other_rolls(self):
        """A rolled 2 must not set natural_1."""
        roller = DiceRoller(rng=ScriptedRNG([2]))
        result = roller.roll("1d20")
        assert result.natural_1 is False
        assert result.natural_20 is False

    def test_advantage_crit_detection_deterministic(self):
        """Advantage keeps the higher die; a kept 20 sets natural_20."""
        roller = DiceRoller(rng=ScriptedRNG([7, 20]))
        result = roller.roll("1d20", advantage=True)
        assert result.kept_dice == [20]
        assert result.natural_20 is True

    def test_disadvantage_fumble_detection_deterministic(self):
        """Disadvantage keeps the lower die; a kept 1 sets natural_1."""
        roller = DiceRoller(rng=ScriptedRNG([1, 15]))
        result = roller.roll("1d20", disadvantage=True)
        assert result.kept_dice == [1]
        assert result.natural_1 is True

    def test_advantage_ignored_for_non_1d20_warns(self):
        """Adv/dis on non-1d20 notation is ignored but must warn, not be silent."""
        roller = DiceRoller(rng=ScriptedRNG([3, 4]))
        with structlog.testing.capture_logs() as logs:
            result = roller.roll("2d6", advantage=True)
        assert result.roll_type == "normal"
        assert len(result.kept_dice) == 2
        warnings = [
            log for log in logs
            if log["event"] == "advantage_flag_ignored_non_1d20"
        ]
        assert len(warnings) == 1
        assert warnings[0]["notation"] == "2d6"
        assert warnings[0]["log_level"] == "warning"

    def test_critical_damage(self, roller):
        """Test that critical damage doubles dice."""
        result = roller.roll_damage("2d6+3", critical=True)
        # Critical doubles dice: 4d6+3
        assert 7 <= result.total <= 27  # 4*1+3 to 4*6+3
        assert len(result.kept_dice) == 4

    def test_ability_score_roll(self, roller):
        """Test 4d6 drop lowest for ability scores."""
        results = roller.roll_ability_scores_4d6()
        assert len(results) == 6
        for result in results:
            assert 3 <= result.total <= 18
            assert len(result.kept_dice) == 3
            assert len(result.dropped_dice) == 1

    def test_initiative_roll(self, roller):
        """Test initiative roll."""
        result = roller.roll_initiative(modifier=3)
        assert 4 <= result.total <= 23
        assert result.modifier == 3

    def test_attack_roll(self, roller):
        """Test attack roll."""
        result = roller.roll_attack(modifier=5)
        assert 6 <= result.total <= 25
        assert result.modifier == 5

    def test_check_roll(self, roller):
        """Test ability check roll."""
        result = roller.roll_check(modifier=2)
        assert 3 <= result.total <= 22

    def test_save_roll(self, roller):
        """Test saving throw roll."""
        result = roller.roll_save(modifier=-1)
        assert 0 <= result.total <= 19

    def test_invalid_notation(self, roller):
        """Test that invalid notation raises ValueError."""
        with pytest.raises(ValueError):
            roller.roll("invalid")

    def test_zero_dice(self, roller):
        """Test that 0 dice raises ValueError."""
        with pytest.raises(ValueError):
            roller.roll("0d20")

    def test_reason_stored(self, roller):
        """Test that roll reason is stored."""
        result = roller.roll("1d20", reason="Attack roll")
        assert result.reason == "Attack roll"


class TestDiceRollResult:
    """Tests for DiceRoll dataclass."""

    def test_total_calculation(self):
        """Test that total is calculated correctly."""
        from dnd_bot.game.mechanics.dice import DiceRoll

        result = DiceRoll(
            notation="3d6+3",
            kept_dice=[4, 5, 6],
            modifier=3,
            total=18,
        )
        assert result.total == 18  # 4+5+6+3

    def test_total_with_negative_modifier(self):
        """Test total with negative modifier."""
        from dnd_bot.game.mechanics.dice import DiceRoll

        result = DiceRoll(
            notation="1d20-5",
            kept_dice=[10],
            modifier=-5,
            total=5,
        )
        assert result.total == 5
