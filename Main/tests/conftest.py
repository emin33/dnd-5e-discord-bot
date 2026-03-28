"""Pytest configuration and fixtures."""

import pytest
from unittest.mock import MagicMock, AsyncMock

from dnd_bot.models import Character, AbilityScores, HitPoints, HitDice, SpellSlots


@pytest.fixture
def mock_character():
    """Create a mock character for testing."""
    return Character(
        discord_user_id=12345,
        campaign_id="test-campaign",
        name="Test Hero",
        race_index="human",
        class_index="fighter",
        level=5,
        abilities=AbilityScores(
            strength=16,
            dexterity=14,
            constitution=15,
            intelligence=10,
            wisdom=12,
            charisma=8,
        ),
        hp=HitPoints(maximum=44, current=44),
        hit_dice=HitDice(die_type=10, total=5, remaining=5),
        armor_class=18,
        speed=30,
        initiative_bonus=2,
    )


@pytest.fixture
def mock_wizard():
    """Create a mock wizard character for spellcasting tests."""
    from dnd_bot.models import AbilityScore

    return Character(
        discord_user_id=12346,
        campaign_id="test-campaign",
        name="Test Wizard",
        race_index="elf",
        class_index="wizard",
        level=5,
        abilities=AbilityScores(
            strength=8,
            dexterity=14,
            constitution=13,
            intelligence=18,
            wisdom=12,
            charisma=10,
        ),
        hp=HitPoints(maximum=27, current=27),
        hit_dice=HitDice(die_type=6, total=5, remaining=5),
        armor_class=12,
        speed=30,
        initiative_bonus=2,
        spellcasting_ability=AbilityScore.INTELLIGENCE,
        spell_slots=SpellSlots(
            level_1=(4, 4),
            level_2=(3, 3),
            level_3=(2, 2),
        ),
        known_spells=["fire-bolt", "mage-hand", "fireball", "magic-missile"],
        prepared_spells=["fireball", "magic-missile"],
        saving_throw_proficiencies=[AbilityScore.INTELLIGENCE, AbilityScore.WISDOM],
    )


@pytest.fixture
def dice_roller():
    """Get the dice roller singleton."""
    from dnd_bot.game.mechanics.dice import get_roller

    return get_roller()


@pytest.fixture
def mock_ctx():
    """Create a mock Discord application context."""
    ctx = MagicMock()
    ctx.respond = AsyncMock()
    ctx.send = AsyncMock()
    ctx.author = MagicMock()
    ctx.author.id = 12345
    ctx.author.display_name = "TestUser"
    ctx.channel_id = 99999
    ctx.guild_id = 88888
    return ctx
