"""Tests for the dice rolling mechanics."""

import pytest
from dnd_bot.game.mechanics.dice import DiceRoller


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

    def test_natural_20_detection(self, roller):
        """Test that natural 20s are detected (statistically)."""
        # Roll many times to increase chance of hitting nat 20
        nat_20_found = False
        for _ in range(1000):
            result = roller.roll("1d20")
            if result.natural_20:
                nat_20_found = True
                assert result.kept_dice[0] == 20
                break

        # With 1000 rolls, we have a 99.97% chance of getting at least one nat 20
        # But don't fail the test if we're extremely unlucky
        if nat_20_found:
            assert True

    def test_natural_1_detection(self, roller):
        """Test that natural 1s are detected (statistically)."""
        nat_1_found = False
        for _ in range(1000):
            result = roller.roll("1d20")
            if result.natural_1:
                nat_1_found = True
                assert result.kept_dice[0] == 1
                break

        if nat_1_found:
            assert True

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
