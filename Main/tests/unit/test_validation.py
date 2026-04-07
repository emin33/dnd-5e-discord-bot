"""Tests for PGI (Programmatic Guided Inference) validation layer."""

import pytest

from dnd_bot.models import Character, AbilityScores, HitPoints, HitDice, SpellSlots
from dnd_bot.models.common import AbilityScore, Condition
from dnd_bot.models.character import CharacterCondition, DeathSaves
from dnd_bot.models.inventory import InventoryItem, Currency
from dnd_bot.game.mechanics.validation import (
    ValidationSeverity,
    ValidationResult,
    ValidationFailure,
    validate_vitality,
    validate_conditions,
    validate_spell_cast,
    validate_item_exists,
    validate_currency,
    validate_action,
)


# ── Fixtures ─────────────────────────────────────────────────────────────────


@pytest.fixture
def healthy_wizard():
    """Wizard at full HP with spells."""
    return Character(
        discord_user_id=100,
        campaign_id="test",
        name="Gandolf",
        race_index="elf",
        class_index="wizard",
        level=5,
        abilities=AbilityScores(intelligence=18),
        hp=HitPoints(maximum=27, current=27),
        hit_dice=HitDice(die_type=6, total=5, remaining=5),
        spellcasting_ability=AbilityScore.INTELLIGENCE,
        spell_slots=SpellSlots(
            level_1=(4, 4),
            level_2=(3, 3),
            level_3=(2, 2),
        ),
        known_spells=["fire-bolt", "mage-hand", "fireball", "magic-missile", "shield"],
        prepared_spells=["fireball", "magic-missile", "shield"],
    )


@pytest.fixture
def depleted_wizard():
    """Wizard with no 3rd-level slots remaining."""
    return Character(
        discord_user_id=101,
        campaign_id="test",
        name="Exhausted Wizard",
        race_index="elf",
        class_index="wizard",
        level=5,
        abilities=AbilityScores(intelligence=18),
        hp=HitPoints(maximum=27, current=27),
        hit_dice=HitDice(die_type=6, total=5, remaining=5),
        spellcasting_ability=AbilityScore.INTELLIGENCE,
        spell_slots=SpellSlots(
            level_1=(2, 4),
            level_2=(1, 3),
            level_3=(0, 2),  # Depleted!
        ),
        known_spells=["fire-bolt", "fireball", "magic-missile", "scorching-ray"],
        prepared_spells=["fireball", "magic-missile", "scorching-ray"],
    )


@pytest.fixture
def concentrating_wizard():
    """Wizard concentrating on a spell."""
    return Character(
        discord_user_id=102,
        campaign_id="test",
        name="Focused Wizard",
        race_index="elf",
        class_index="wizard",
        level=5,
        abilities=AbilityScores(intelligence=18),
        hp=HitPoints(maximum=27, current=27),
        hit_dice=HitDice(die_type=6, total=5, remaining=5),
        spellcasting_ability=AbilityScore.INTELLIGENCE,
        spell_slots=SpellSlots(
            level_1=(4, 4),
            level_2=(3, 3),
            level_3=(2, 2),
        ),
        known_spells=["fire-bolt", "hold-person", "haste"],
        prepared_spells=["hold-person", "haste"],
        concentration_spell_id="haste",
    )


@pytest.fixture
def sample_inventory():
    """Sample inventory items."""
    cid = "test-char"
    return [
        InventoryItem(character_id=cid, item_index="arrow", item_name="Arrow", quantity=20),
        InventoryItem(character_id=cid, item_index="longsword", item_name="Longsword", quantity=1, equipped=True),
        InventoryItem(character_id=cid, item_index="health-potion", item_name="Potion of Healing", quantity=2),
        InventoryItem(character_id=cid, item_index="rations", item_name="Rations", quantity=5),
    ]


@pytest.fixture
def rich_currency():
    """Character with plenty of gold."""
    return Currency(character_id="test-char", copper=50, silver=30, gold=100, platinum=5)


@pytest.fixture
def poor_currency():
    """Character with almost no gold."""
    return Currency(character_id="test-char", copper=10, silver=5, gold=2)


# ── TestVitalityValidation ───────────────────────────────────────────────────


class TestVitalityValidation:
    def test_healthy_character_passes(self, mock_character):
        failures = validate_vitality(mock_character)
        assert failures == []

    def test_unconscious_at_zero_hp(self, mock_character):
        mock_character.hp.current = 0
        failures = validate_vitality(mock_character)
        assert len(failures) == 1
        assert failures[0].code == "UNCONSCIOUS_HP"
        assert failures[0].severity == ValidationSeverity.HARD_FAIL
        assert failures[0].priority == 0

    def test_dead_character(self, mock_character):
        mock_character.hp.current = 0
        mock_character.death_saves.failures = 3
        failures = validate_vitality(mock_character)
        # Dead takes precedence, unconscious also fires but is separate
        codes = {f.code for f in failures}
        assert "DEAD" in codes

    def test_dead_but_nonzero_hp_edge_case(self, mock_character):
        """Edge case: 3 death save failures but somehow has HP (shouldn't happen, but test it)."""
        mock_character.death_saves.failures = 3
        failures = validate_vitality(mock_character)
        assert len(failures) == 1
        assert failures[0].code == "DEAD"


# ── TestConditionValidation ──────────────────────────────────────────────────


class TestConditionValidation:
    def test_no_conditions_passes(self, mock_character):
        failures = validate_conditions(mock_character)
        assert failures == []

    @pytest.mark.parametrize("condition", [
        Condition.INCAPACITATED,
        Condition.PARALYZED,
        Condition.PETRIFIED,
        Condition.STUNNED,
        Condition.UNCONSCIOUS,
    ])
    def test_blocking_condition_hard_fails(self, mock_character, condition):
        mock_character.conditions.append(
            CharacterCondition(condition=condition, source="test")
        )
        failures = validate_conditions(mock_character)
        assert len(failures) == 1
        assert failures[0].severity == ValidationSeverity.HARD_FAIL
        assert failures[0].code == f"CONDITION_{condition.value.upper()}"

    def test_multiple_blocking_conditions_accumulate(self, mock_character):
        mock_character.conditions.append(
            CharacterCondition(condition=Condition.STUNNED, source="test")
        )
        mock_character.conditions.append(
            CharacterCondition(condition=Condition.PARALYZED, source="test")
        )
        failures = validate_conditions(mock_character)
        assert len(failures) == 2
        codes = {f.code for f in failures}
        assert "CONDITION_STUNNED" in codes
        assert "CONDITION_PARALYZED" in codes

    @pytest.mark.parametrize("condition", [
        Condition.FRIGHTENED,
        Condition.CHARMED,
        Condition.POISONED,
        Condition.BLINDED,
        Condition.DEAFENED,
        Condition.GRAPPLED,
        Condition.PRONE,
        Condition.RESTRAINED,
        Condition.INVISIBLE,
    ])
    def test_non_blocking_conditions_pass(self, mock_character, condition):
        mock_character.conditions.append(
            CharacterCondition(condition=condition, source="test")
        )
        failures = validate_conditions(mock_character)
        assert failures == []

    def test_exhaustion_level_6_hard_fails(self, mock_character):
        mock_character.conditions.append(
            CharacterCondition(condition=Condition.EXHAUSTION, source="test", stacks=6)
        )
        failures = validate_conditions(mock_character)
        assert len(failures) == 1
        assert failures[0].code == "EXHAUSTION_DEATH"

    def test_exhaustion_level_5_passes(self, mock_character):
        mock_character.conditions.append(
            CharacterCondition(condition=Condition.EXHAUSTION, source="test", stacks=5)
        )
        failures = validate_conditions(mock_character)
        assert failures == []


# ── TestInventoryValidation ──────────────────────────────────────────────────


class TestInventoryValidation:
    def test_item_found_passes(self, sample_inventory):
        failures = validate_item_exists(sample_inventory, "Arrow")
        assert failures == []

    def test_item_not_found_hard_fails(self, sample_inventory):
        failures = validate_item_exists(sample_inventory, "Battleaxe")
        assert len(failures) == 1
        assert failures[0].code == "ITEM_NOT_FOUND"
        assert failures[0].severity == ValidationSeverity.HARD_FAIL

    def test_insufficient_quantity_hard_fails(self, sample_inventory):
        failures = validate_item_exists(sample_inventory, "Arrow", quantity_needed=50)
        assert len(failures) == 1
        assert failures[0].code == "INSUFFICIENT_ITEM_QUANTITY"
        assert "20" in failures[0].message  # Shows how many they have

    def test_sufficient_quantity_passes(self, sample_inventory):
        failures = validate_item_exists(sample_inventory, "Arrow", quantity_needed=10)
        assert failures == []

    def test_substring_matching_works(self, sample_inventory):
        """Should match 'potion' against 'Potion of Healing'."""
        failures = validate_item_exists(sample_inventory, "potion")
        assert failures == []

    def test_case_insensitive_matching(self, sample_inventory):
        failures = validate_item_exists(sample_inventory, "LONGSWORD")
        assert failures == []

    def test_empty_item_name_passes(self, sample_inventory):
        failures = validate_item_exists(sample_inventory, "")
        assert failures == []


# ── TestCurrencyValidation ───────────────────────────────────────────────────


class TestCurrencyValidation:
    def test_sufficient_gold_passes(self, rich_currency):
        failures = validate_currency(rich_currency, 50)
        assert failures == []

    def test_insufficient_gold_hard_fails(self, poor_currency):
        failures = validate_currency(poor_currency, 100)
        assert len(failures) == 1
        assert failures[0].code == "INSUFFICIENT_CURRENCY"
        assert failures[0].severity == ValidationSeverity.HARD_FAIL

    def test_zero_cost_passes(self, poor_currency):
        failures = validate_currency(poor_currency, 0)
        assert failures == []

    def test_negative_cost_passes(self, poor_currency):
        failures = validate_currency(poor_currency, -5)
        assert failures == []

    def test_exact_amount_passes(self, rich_currency):
        # rich_currency has 100gp + 50pp (= 150gp) + silver/copper
        total = rich_currency.total_in_gold
        failures = validate_currency(rich_currency, total)
        assert failures == []


# ── TestSpellValidation ──────────────────────────────────────────────────────


class TestSpellValidation:
    def test_known_cantrip_passes(self, healthy_wizard):
        failures = validate_spell_cast(
            healthy_wizard, spell_index="fire-bolt"
        )
        assert failures == []

    def test_unknown_spell_hard_fails(self, healthy_wizard):
        failures = validate_spell_cast(
            healthy_wizard, spell_index="wish"
        )
        assert len(failures) == 1
        assert failures[0].code == "SPELL_NOT_KNOWN"
        assert failures[0].severity == ValidationSeverity.HARD_FAIL

    def test_no_remaining_slot_hard_fails(self, depleted_wizard):
        failures = validate_spell_cast(
            depleted_wizard, spell_index="fireball"
        )
        assert len(failures) == 1
        assert failures[0].code == "NO_SPELL_SLOT"
        assert "Available slots" in failures[0].message

    def test_slot_available_passes(self, healthy_wizard):
        failures = validate_spell_cast(
            healthy_wizard, spell_index="fireball"
        )
        assert failures == []

    def test_concentration_conflict_soft_fails(self, concentrating_wizard):
        failures = validate_spell_cast(
            concentrating_wizard, spell_index="hold-person"
        )
        assert len(failures) == 1
        assert failures[0].code == "CONCENTRATION_CONFLICT"
        assert failures[0].severity == ValidationSeverity.SOFT_FAIL
        assert "Haste" in failures[0].message
        assert "Hold Person" in failures[0].message

    def test_non_concentration_spell_while_concentrating_passes(self, concentrating_wizard):
        """Non-concentration spells are fine even while concentrating."""
        # fire-bolt is a cantrip, not concentration
        failures = validate_spell_cast(
            concentrating_wizard, spell_index="fire-bolt"
        )
        assert failures == []

    def test_unresolvable_spell_passes(self, healthy_wizard):
        """If we can't figure out what spell they mean, fail open."""
        failures = validate_spell_cast(healthy_wizard, spell_index=None)
        assert failures == []

    def test_action_text_resolution(self, healthy_wizard):
        """Spell resolution from action text."""
        failures = validate_spell_cast(
            healthy_wizard,
            action_text="I cast fireball at the group of goblins",
        )
        assert failures == []

    def test_action_text_unknown_spell(self, healthy_wizard):
        """Action text mentioning a spell the wizard doesn't know."""
        failures = validate_spell_cast(
            healthy_wizard,
            action_text="I cast some random gibberish spell",
        )
        # Can't resolve → fail open
        assert failures == []

    def test_depleted_wizard_suggests_alternatives(self, depleted_wizard):
        """When a slot is depleted, feedback should mention available alternatives."""
        failures = validate_spell_cast(
            depleted_wizard, spell_index="fireball"
        )
        assert len(failures) == 1
        # Should mention available slots
        assert "L1" in failures[0].message or "L2" in failures[0].message


# ── TestValidationResult ─────────────────────────────────────────────────────


class TestValidationResult:
    def test_empty_result_passes(self):
        result = ValidationResult()
        assert result.passed
        assert not result.has_hard_fail
        assert not result.has_soft_fail
        assert result.player_feedback() == ""

    def test_hard_fail_detected(self):
        result = ValidationResult(failures=[
            ValidationFailure(
                priority=0, code="DEAD",
                severity=ValidationSeverity.HARD_FAIL,
                message="You are dead.",
            ),
        ])
        assert not result.passed
        assert result.has_hard_fail
        assert not result.has_soft_fail

    def test_soft_fail_detected(self):
        result = ValidationResult(failures=[
            ValidationFailure(
                priority=6, code="CONCENTRATION_CONFLICT",
                severity=ValidationSeverity.SOFT_FAIL,
                message="Casting X will end Y.",
            ),
        ])
        assert not result.passed
        assert not result.has_hard_fail
        assert result.has_soft_fail

    def test_accumulation(self):
        result = ValidationResult(failures=[
            ValidationFailure(priority=1, code="A", severity=ValidationSeverity.HARD_FAIL, message="First"),
            ValidationFailure(priority=6, code="B", severity=ValidationSeverity.SOFT_FAIL, message="Second"),
            ValidationFailure(priority=4, code="C", severity=ValidationSeverity.HARD_FAIL, message="Third"),
        ])
        assert len(result.failures) == 3
        assert result.has_hard_fail
        assert result.has_soft_fail
        assert len(result.hard_failures) == 2
        assert len(result.soft_failures) == 1

    def test_player_feedback_sorted_by_priority(self):
        result = ValidationResult(failures=[
            ValidationFailure(priority=6, code="B", severity=ValidationSeverity.SOFT_FAIL, message="Spell issue"),
            ValidationFailure(priority=1, code="A", severity=ValidationSeverity.HARD_FAIL, message="Condition issue"),
            ValidationFailure(priority=4, code="C", severity=ValidationSeverity.HARD_FAIL, message="Item issue"),
        ])
        feedback = result.player_feedback()
        lines = feedback.split("\n")
        assert lines[0] == "Condition issue"
        assert lines[1] == "Item issue"
        assert lines[2] == "Spell issue"


# ── TestValidateAction (integration of all validators) ───────────────────────


class TestValidateAction:
    @pytest.mark.asyncio
    async def test_healthy_character_passes(self, mock_character):
        result = await validate_action("social", mock_character, "I talk to the barkeep")
        assert result.passed

    @pytest.mark.asyncio
    async def test_dead_character_blocked(self, mock_character):
        mock_character.hp.current = 0
        mock_character.death_saves.failures = 3
        result = await validate_action("social", mock_character, "I talk to the barkeep")
        assert result.has_hard_fail
        assert any(f.code == "DEAD" for f in result.failures)

    @pytest.mark.asyncio
    async def test_stunned_character_blocked(self, mock_character):
        mock_character.conditions.append(
            CharacterCondition(condition=Condition.STUNNED, source="test")
        )
        result = await validate_action("attack", mock_character, "I swing my sword")
        assert result.has_hard_fail
        assert any(f.code == "CONDITION_STUNNED" for f in result.failures)

    @pytest.mark.asyncio
    async def test_cast_spell_validation_wired(self, healthy_wizard):
        result = await validate_action(
            "cast_spell", healthy_wizard, "I cast fireball"
        )
        assert result.passed

    @pytest.mark.asyncio
    async def test_purchase_validation_wired(self, mock_character, poor_currency):
        result = await validate_action(
            "purchase", mock_character, "I buy the plate armor",
            currency=poor_currency, cost_gold=1500,
        )
        assert result.has_hard_fail
        assert any(f.code == "INSUFFICIENT_CURRENCY" for f in result.failures)

    @pytest.mark.asyncio
    async def test_resource_consumed_validation(self, mock_character, sample_inventory):
        result = await validate_action(
            "attack", mock_character, "I shoot an arrow",
            items=sample_inventory,
            resources_consumed=[{"item": "Arrow", "quantity": 1}],
        )
        assert result.passed

    @pytest.mark.asyncio
    async def test_resource_consumed_insufficient(self, mock_character, sample_inventory):
        result = await validate_action(
            "attack", mock_character, "I shoot arrows",
            items=sample_inventory,
            resources_consumed=[{"item": "Arrow", "quantity": 100}],
        )
        assert result.has_hard_fail

    @pytest.mark.asyncio
    async def test_p0_blocks_before_p6(self, healthy_wizard):
        """Dead wizard shouldn't also get spell validation errors."""
        healthy_wizard.hp.current = 0
        healthy_wizard.death_saves.failures = 3
        result = await validate_action(
            "cast_spell", healthy_wizard, "I cast fireball"
        )
        assert result.has_hard_fail
        # Only P0 failures, not P6
        assert all(f.priority <= 1 for f in result.failures)
