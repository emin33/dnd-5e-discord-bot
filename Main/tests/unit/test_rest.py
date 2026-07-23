"""Regression tests for truthful, schema-safe rest recovery."""

from dnd_bot.game.mechanics.rest import RestManager
from dnd_bot.models import Character, HitDice, HitPoints, SpellSlots


def _character(class_index: str, *, level: int = 5) -> Character:
    return Character(
        discord_user_id=1,
        campaign_id="camp",
        name="Rest Tester",
        race_index="human",
        class_index=class_index,
        level=level,
        hp=HitPoints(maximum=30, current=12),
        hit_dice=HitDice(die_type=8, total=level, remaining=1),
    )


def test_short_rest_does_not_assign_unmodelled_fighter_counters():
    """A base Character has no feature-use fields; resting must not crash."""
    character = _character("fighter")

    result = RestManager().short_rest(character)

    assert result.features_recovered == []
    assert "second_wind_used" not in character.model_fields_set
    assert "action_surge_used" not in character.model_fields_set


def test_long_rest_does_not_claim_untracked_class_features():
    character = _character("fighter")

    result = RestManager().long_rest(character)

    assert character.hp.current == character.hp.maximum
    assert result.features_recovered == []
    assert "All class features" not in result.features_recovered


def test_warlock_short_rest_recovers_tracked_pact_slots():
    character = _character("warlock")
    character.spell_slots = SpellSlots(level_1=(0, 2), level_2=(1, 2))

    result = RestManager().short_rest(character)

    assert character.spell_slots.get_slots(1) == (2, 2)
    assert character.spell_slots.get_slots(2) == (2, 2)
    assert result.features_recovered == ["Pact Magic slots"]
