"""Pytest configuration and fixtures."""

import itertools

import pytest
from unittest.mock import MagicMock, AsyncMock

from dnd_bot.models import Character, AbilityScores, HitPoints, HitDice, SpellSlots

# The combat-manager / coordinator / turn-lock registries are module-level
# globals keyed by channel id, so hand-picked constants in different test
# files can silently collide and share state. This counter guarantees every
# test that asks gets a channel id unique across the whole run.
_unique_channel_ids = itertools.count(900_001)


@pytest.fixture
def unique_channel_ids():
    """Factory allocating run-unique Discord channel ids.

    Cleans the module-global combat registries for every id it handed out,
    so a failing test can't leave a combat/coordinator/turn-lock entry
    behind.
    """
    allocated: list[int] = []

    def _alloc() -> int:
        channel_id = next(_unique_channel_ids)
        allocated.append(channel_id)
        return channel_id

    yield _alloc

    from dnd_bot.game.combat.coordinator import clear_coordinator_by_key
    from dnd_bot.game.combat.manager import clear_combat_by_key

    for channel_id in allocated:
        key = f"discord:{channel_id}"
        clear_combat_by_key(key)
        clear_coordinator_by_key(key)


@pytest.fixture
def unique_channel_id(unique_channel_ids) -> int:
    """A Discord channel id no other test in this run uses."""
    return unique_channel_ids()


@pytest.fixture(scope="session", autouse=True)
def _llm_singletons_not_left_faked():
    """Seam-leak regression net: no test may leak an LLM fake into a
    module singleton.

    The dedup-judge / state-extractor / entity-extractor singletons capture
    ``get_llm_client()`` at first construction. A test that swaps a fake in
    (or constructs the singleton while the client seam is patched) without
    restoring it poisons every later test in the run. This net turns a
    reintroduced leak into a loud session-end failure instead of downstream
    weirdness.
    """
    import dnd_bot.llm.extractors.dedup_judge as dedup_judge_mod
    import dnd_bot.llm.extractors.entity_extractor as entity_extractor_mod
    import dnd_bot.llm.extractors.state_extractor as state_extractor_mod

    def _singletons() -> dict:
        return {
            "dedup_judge": dedup_judge_mod._JUDGE,
            "state_extractor": state_extractor_mod._state_extractor,
            "entity_extractor": entity_extractor_mod._extractor,
        }

    started_with = {
        name: type(getattr(instance, "client", None)).__name__
        for name, instance in _singletons().items()
        if instance is not None
    }

    yield

    from tests.fakes import FunctionBrain, ScriptedBrain

    leaked = {}
    for name, instance in _singletons().items():
        if instance is None:
            continue
        client = getattr(instance, "client", None)
        if isinstance(client, (FunctionBrain, ScriptedBrain)):
            leaked[name] = type(client).__name__
    assert not leaked, (
        f"Test fakes leaked into LLM extractor singletons: {leaked} "
        f"(client types at session start: {started_with or 'none built yet'}). "
        "A test replaced the singleton's client, or built the singleton "
        "under a patched get_llm_client, without restoring it."
    )


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
