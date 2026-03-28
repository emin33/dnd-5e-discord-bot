"""Tests for the combat system."""

import pytest
from dnd_bot.models import Combat, Combatant, CombatState
from dnd_bot.game.combat.manager import CombatManager


class TestCombatant:
    """Tests for Combatant model."""

    @pytest.fixture
    def player_combatant(self):
        return Combatant(
            combat_id="test-combat",
            name="Test Fighter",
            is_player=True,
            hp_max=50,
            hp_current=50,
            armor_class=18,
            speed=30,
            initiative_bonus=2,
        )

    @pytest.fixture
    def monster_combatant(self):
        return Combatant(
            combat_id="test-combat",
            name="Goblin",
            is_player=False,
            hp_max=7,
            hp_current=7,
            armor_class=15,
            speed=30,
            initiative_bonus=2,
        )

    def test_take_damage(self, player_combatant):
        """Test taking damage."""
        actual, instant_death = player_combatant.take_damage(10)
        assert actual == 10
        assert player_combatant.hp_current == 40
        assert not instant_death

    def test_take_damage_to_zero(self, player_combatant):
        """Test taking damage to 0 HP."""
        actual, instant_death = player_combatant.take_damage(50)
        assert player_combatant.hp_current == 0
        assert not player_combatant.is_conscious
        assert not instant_death  # No instant death, just at 0

    def test_instant_death(self, player_combatant):
        """Test instant death from massive damage."""
        # Damage = current HP + max HP
        actual, instant_death = player_combatant.take_damage(100)  # 50 + 50
        assert player_combatant.hp_current == 0
        assert instant_death
        assert player_combatant.death_saves.is_dead

    def test_heal(self, player_combatant):
        """Test healing."""
        player_combatant.hp_current = 30
        actual = player_combatant.heal(15)
        assert actual == 15
        assert player_combatant.hp_current == 45

    def test_heal_no_overheal(self, player_combatant):
        """Test that healing doesn't exceed max."""
        player_combatant.hp_current = 45
        actual = player_combatant.heal(20)
        assert actual == 5  # Only healed 5 to reach max
        assert player_combatant.hp_current == 50

    def test_temp_hp(self, player_combatant):
        """Test temporary HP."""
        player_combatant.add_temp_hp(10)
        assert player_combatant.hp_temp == 10

        # Temp HP absorbs damage first
        actual, _ = player_combatant.take_damage(7)
        assert player_combatant.hp_temp == 3
        assert player_combatant.hp_current == 50

        # Finish off temp HP and some real HP
        actual, _ = player_combatant.take_damage(8)
        assert player_combatant.hp_temp == 0
        assert player_combatant.hp_current == 45

    def test_temp_hp_no_stack(self, player_combatant):
        """Test that temp HP doesn't stack."""
        player_combatant.add_temp_hp(10)
        player_combatant.add_temp_hp(5)  # Should keep 10
        assert player_combatant.hp_temp == 10

        player_combatant.add_temp_hp(15)  # Should update to 15
        assert player_combatant.hp_temp == 15

    def test_death_saves_player(self, player_combatant):
        """Test death saves for player at 0 HP."""
        player_combatant.hp_current = 0
        assert player_combatant.is_dying

        # Add successes
        player_combatant.death_saves.add_success()
        assert player_combatant.death_saves.successes == 1
        assert not player_combatant.death_saves.is_stable

        player_combatant.death_saves.add_success(2)
        assert player_combatant.death_saves.successes == 3
        assert player_combatant.death_saves.is_stable

    def test_death_saves_failure(self, player_combatant):
        """Test death save failures."""
        player_combatant.hp_current = 0
        player_combatant.death_saves.add_failure(3)
        assert player_combatant.death_saves.is_dead
        assert player_combatant.is_dead

    def test_heal_resets_death_saves(self, player_combatant):
        """Test that healing resets death saves."""
        player_combatant.hp_current = 0
        player_combatant.death_saves.add_failure(2)
        player_combatant.death_saves.add_success(1)

        player_combatant.heal(5)
        assert player_combatant.hp_current == 5
        assert player_combatant.death_saves.successes == 0
        assert player_combatant.death_saves.failures == 0

    def test_monster_death(self, monster_combatant):
        """Test that monsters die at 0 HP without death saves."""
        monster_combatant.take_damage(7)
        assert monster_combatant.hp_current == 0
        assert monster_combatant.is_dead


class TestCombat:
    """Tests for Combat model."""

    @pytest.fixture
    def combat(self):
        return Combat(
            id="test-combat",
            session_id="test-session",
            channel_id=12345,
            state=CombatState.SETUP,
        )

    def test_add_combatant(self, combat):
        """Test adding combatants."""
        combatant = Combatant(
            combat_id=combat.id,
            name="Fighter",
            is_player=True,
            hp_max=50,
            hp_current=50,
            armor_class=18,
        )
        combat.add_combatant(combatant)
        assert len(combat.combatants) == 1

    def test_state_transitions(self, combat):
        """Test valid state transitions."""
        assert combat.can_transition(CombatState.ROLLING_INITIATIVE)
        assert combat.transition(CombatState.ROLLING_INITIATIVE)
        assert combat.state == CombatState.ROLLING_INITIATIVE

        assert combat.can_transition(CombatState.ACTIVE)
        assert combat.transition(CombatState.ACTIVE)

    def test_invalid_transition(self, combat):
        """Test invalid state transition."""
        assert not combat.can_transition(CombatState.COMBAT_END)
        assert not combat.transition(CombatState.COMBAT_END)
        assert combat.state == CombatState.SETUP

    def test_initiative_sorting(self, combat):
        """Test initiative-based sorting."""
        c1 = Combatant(
            combat_id=combat.id,
            name="Slow",
            is_player=True,
            hp_max=10,
            hp_current=10,
            initiative_roll=5,
            initiative_bonus=0,
        )
        c2 = Combatant(
            combat_id=combat.id,
            name="Fast",
            is_player=True,
            hp_max=10,
            hp_current=10,
            initiative_roll=20,
            initiative_bonus=0,
        )
        combat.add_combatant(c1)
        combat.add_combatant(c2)
        combat.roll_all_initiative()

        sorted_list = combat.get_sorted_combatants()
        assert sorted_list[0].name == "Fast"
        assert sorted_list[1].name == "Slow"

    def test_combat_over_all_enemies_dead(self, combat):
        """Test combat ends when all enemies are dead."""
        player = Combatant(
            combat_id=combat.id,
            name="Hero",
            is_player=True,
            hp_max=50,
            hp_current=50,
        )
        enemy = Combatant(
            combat_id=combat.id,
            name="Villain",
            is_player=False,
            hp_max=20,
            hp_current=0,  # Dead
        )
        combat.add_combatant(player)
        combat.add_combatant(enemy)

        assert combat.is_combat_over()


class TestCombatManager:
    """Tests for CombatManager."""

    @pytest.fixture
    def manager(self):
        return CombatManager.create_encounter(
            session_id="test-session",
            channel_id=12345,
            name="Test Encounter",
        )

    def test_create_encounter(self, manager):
        """Test creating an encounter."""
        assert manager.combat.state == CombatState.SETUP
        assert manager.combat.encounter_name == "Test Encounter"

    def test_add_custom_combatant(self, manager):
        """Test adding a custom combatant."""
        combatant = manager.add_custom_combatant(
            name="Custom Enemy",
            hp=50,
            ac=15,
            initiative_bonus=2,
        )
        assert combatant.name == "Custom Enemy"
        assert combatant.hp_max == 50
        assert combatant.armor_class == 15

    def test_get_combatant_by_name(self, manager):
        """Test finding combatant by name."""
        manager.add_custom_combatant(name="Test Fighter", hp=50, ac=15)
        found = manager.get_combatant_by_name("Fighter")
        assert found is not None
        assert "Fighter" in found.name

    def test_roll_initiative(self, manager):
        """Test rolling initiative."""
        manager.add_custom_combatant(name="Fighter 1", hp=50, ac=15)
        manager.add_custom_combatant(name="Fighter 2", hp=50, ac=15)

        results = manager.roll_all_initiative()
        assert len(results) == 2
        for combatant, roll in results:
            assert combatant.initiative_roll is not None
            assert 1 <= roll.total <= 20

    def test_apply_damage(self, manager):
        """Test applying damage."""
        combatant = manager.add_custom_combatant(name="Target", hp=50, ac=15)
        actual, is_unconscious, instant_death = manager.apply_damage(combatant.id, 20)
        assert actual == 20
        assert combatant.hp_current == 30
        assert not is_unconscious

    def test_apply_healing(self, manager):
        """Test applying healing."""
        combatant = manager.add_custom_combatant(name="Target", hp=50, ac=15)
        combatant.hp_current = 30
        actual, was_revived = manager.apply_healing(combatant.id, 15)
        assert actual == 15
        assert combatant.hp_current == 45
        assert not was_revived
