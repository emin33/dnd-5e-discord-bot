"""Durable class-feature resources: model, recovery semantics, persistence.

LONGFORM_READINESS blocker "add durable feature resources before claiming
class-feature rest recovery": rest recovery previously probed getattr()
attributes the persisted Character never has, so nothing but Warlock pact
slots ever actually recovered. These tests pin the durable
(character_id, resource_key, current, maximum, recharge_rule, source)
model, the rest-type recovery rules, and the migration + repository
round-trip.
"""

import pytest

from dnd_bot.game.mechanics.rest import RestManager
from dnd_bot.models import AbilityScores, Character, HitDice, HitPoints
from dnd_bot.models.feature_resource import (
    RECHARGE_LONG_REST,
    RECHARGE_SHORT_REST,
    FeatureResource,
    default_feature_resources,
)


def _character(class_index: str, *, level: int = 5, charisma: int = 16) -> Character:
    return Character(
        id="char-1",
        discord_user_id=1,
        campaign_id="camp-1",
        name="Tester",
        race_index="human",
        class_index=class_index,
        level=level,
        abilities=AbilityScores(charisma=charisma),
        hp=HitPoints(current=20, maximum=20),
        hit_dice=HitDice(die_type=10, total=level, remaining=level),
    )


def _resource(key: str, current: int, maximum: int, rule: str) -> FeatureResource:
    return FeatureResource(
        character_id="char-1",
        resource_key=key,
        name=key.replace("_", " ").title(),
        current=current,
        maximum=maximum,
        recharge_rule=rule,
    )


class TestDefaultFeatureResources:
    def test_fighter_low_level(self):
        resources = {
            r.resource_key: r for r in default_feature_resources(_character("fighter", level=1))
        }
        assert resources["second_wind"].maximum == 1
        assert resources["second_wind"].recharge_rule == RECHARGE_SHORT_REST
        assert "action_surge" not in resources

    def test_fighter_17_gets_two_action_surges(self):
        resources = {
            r.resource_key: r for r in default_feature_resources(_character("fighter", level=17))
        }
        assert resources["action_surge"].maximum == 2

    def test_monk_ki_scales_with_level(self):
        resources = default_feature_resources(_character("monk", level=7))
        assert resources[0].resource_key == "ki"
        assert resources[0].maximum == 7

    def test_bard_inspiration_rule_flips_at_five(self):
        low = default_feature_resources(_character("bard", level=4, charisma=16))
        high = default_feature_resources(_character("bard", level=5, charisma=16))
        assert low[0].maximum == 3  # CHA mod
        assert low[0].recharge_rule == RECHARGE_LONG_REST
        assert high[0].recharge_rule == RECHARGE_SHORT_REST

    def test_barbarian_rage_scales(self):
        assert default_feature_resources(_character("barbarian", level=1))[0].maximum == 2
        assert default_feature_resources(_character("barbarian", level=12))[0].maximum == 5

    def test_rows_carry_identity_and_source(self):
        resource = default_feature_resources(_character("druid", level=3))[0]
        assert resource.character_id == "char-1"
        assert resource.source == "class:druid"
        assert resource.current == resource.maximum == 2

    def test_warlock_seeds_nothing(self):
        # Pact Magic stays on SpellSlots, which already persists.
        assert default_feature_resources(_character("warlock", level=5)) == []


class TestRestRecovery:
    def test_short_rest_recovers_only_short_rest_rule(self):
        manager = RestManager()
        spent_wind = _resource("second_wind", 0, 1, RECHARGE_SHORT_REST)
        spent_rage = _resource("rage", 1, 3, RECHARGE_LONG_REST)

        result = manager.short_rest(
            _character("fighter"),
            feature_resources=[spent_wind, spent_rage],
        )

        assert "Second Wind" in result.features_recovered
        assert spent_wind.current == 1
        # Rage waits for a long rest.
        assert "Rage" not in result.features_recovered
        assert spent_rage.current == 1

    def test_long_rest_recovers_both_rules(self):
        manager = RestManager()
        spent_wind = _resource("second_wind", 0, 1, RECHARGE_SHORT_REST)
        spent_rage = _resource("rage", 1, 3, RECHARGE_LONG_REST)

        result = manager.long_rest(
            _character("barbarian"),
            feature_resources=[spent_wind, spent_rage],
        )

        assert "Second Wind" in result.features_recovered
        assert "Rage" in result.features_recovered
        assert spent_wind.current == 1
        assert spent_rage.current == 3

    def test_full_resources_are_not_claimed(self):
        # The recovered list is a receipt, not a wish list: a counter
        # already at maximum reports nothing.
        manager = RestManager()
        result = manager.long_rest(
            _character("fighter"),
            feature_resources=[_resource("second_wind", 1, 1, RECHARGE_SHORT_REST)],
        )
        assert result.features_recovered == []

    def test_rest_without_resources_stays_truthful(self):
        manager = RestManager()
        result = manager.long_rest(_character("fighter"))
        assert result.features_recovered == []

    def test_spend_and_restore_bounds(self):
        resource = _resource("ki", 3, 5, RECHARGE_SHORT_REST)
        assert resource.spend(2) is True
        assert resource.current == 1
        assert resource.spend(2) is False  # not enough — unchanged
        assert resource.current == 1
        assert resource.restore() == 4
        assert resource.current == 5


class TestFeatureResourcePersistence:
    @pytest.fixture
    async def db(self, tmp_path):
        from dnd_bot.data.database import Database

        database = Database(db_path=tmp_path / "test.db")
        await database.connect()
        # Parent rows for the FK chain (campaign -> character).
        await database.execute(
            "INSERT INTO campaign (id, guild_id, name, dm_user_id) VALUES (?, ?, ?, ?)",
            ("camp-1", 1, "Test Campaign", 1),
        )
        await database.execute(
            """
            INSERT INTO character
                (id, discord_user_id, campaign_id, name, race_index,
                 class_index, hp_max, hp_current, hit_dice_type,
                 hit_dice_total, hit_dice_remaining)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            ("char-1", 1, "camp-1", "Tester", "human", "fighter", 20, 20, 10, 5, 5),
        )
        yield database
        await database.disconnect()

    async def test_round_trip(self, db):
        from dnd_bot.data.repositories.feature_resource_repo import (
            FeatureResourceRepository,
        )

        repo = FeatureResourceRepository(db)
        seeded = default_feature_resources(_character("fighter", level=17))
        await repo.save_all(seeded)

        loaded = await repo.list_for_character("char-1")
        assert {r.resource_key for r in loaded} == {"second_wind", "action_surge"}
        surge = next(r for r in loaded if r.resource_key == "action_surge")
        assert (surge.current, surge.maximum) == (2, 2)
        assert surge.recharge_rule == RECHARGE_SHORT_REST
        assert surge.source == "class:fighter"

        # Upsert: a mutated counter overwrites, never duplicates.
        surge.current = 0
        await repo.save_all([surge])
        reloaded = await repo.list_for_character("char-1")
        assert len(reloaded) == 2
        assert next(
            r.current for r in reloaded if r.resource_key == "action_surge"
        ) == 0

    async def test_set_current_targets_one_row(self, db):
        from dnd_bot.data.repositories.feature_resource_repo import (
            FeatureResourceRepository,
        )

        repo = FeatureResourceRepository(db)
        await repo.save_all(default_feature_resources(_character("monk", level=5)))

        assert await repo.set_current("char-1", "ki", 2) is True
        loaded = await repo.list_for_character("char-1")
        assert loaded[0].current == 2
        assert await repo.set_current("char-1", "nonexistent", 1) is False

    async def test_character_delete_cascades(self, db):
        from dnd_bot.data.repositories.feature_resource_repo import (
            FeatureResourceRepository,
        )

        repo = FeatureResourceRepository(db)
        await repo.save_all(default_feature_resources(_character("druid")))
        assert await repo.list_for_character("char-1")

        await db.execute("DELETE FROM character WHERE id = ?", ("char-1",))
        assert await repo.list_for_character("char-1") == []
