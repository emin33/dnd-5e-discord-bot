"""Integration tests for character_repo against a real (tmp) SQLite DB.

These cover the actual save/load cycle that mocked unit tests miss. In particular,
this catches the row-indexing bug from audit P0 #1: writes go to the correct
columns via the INSERT statement, but reads come back from wrong column ordinals
when migrations and code drift apart.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from dnd_bot.data.database import Database
from dnd_bot.data.repositories.character_repo import CharacterRepository
from dnd_bot.models import (
    AbilityScore,
    AbilityScores,
    Character,
    HitDice,
    HitPoints,
    Skill,
    SpellSlots,
)


@pytest.fixture
async def db(tmp_path: Path):
    """A fresh Database pointed at an isolated tmp file. Runs all migrations."""
    db = Database(db_path=tmp_path / "test.db")
    await db.connect()

    # Seed a campaign row — character has a FK on campaign_id and the repo
    # methods that filter `WHERE is_active = 1` also need a real campaign.
    await db.execute(
        """
        INSERT INTO campaign (id, guild_id, name, dm_user_id)
        VALUES (?, ?, ?, ?)
        """,
        ("test-campaign", 99999, "Test Campaign", 12345),
    )
    await db.commit()

    yield db

    await db.disconnect()


@pytest.fixture
def repo(db: Database) -> CharacterRepository:
    return CharacterRepository(db=db)


def _make_character(**overrides) -> Character:
    """Build a character with every persisted field populated to non-default."""
    defaults = dict(
        discord_user_id=12345,
        campaign_id="test-campaign",
        name="Aria Stormwind",
        description="A tall half-elf bard with silver hair and storm-grey eyes.",
        portrait_url="https://example.com/aria.png",
        voice_id="elevenlabs:rachel",
        race_index="half-elf",
        class_index="bard",
        subclass_index="lore",
        level=5,
        experience=6500,
        background_index="entertainer",
        abilities=AbilityScores(
            strength=10,
            dexterity=14,
            constitution=13,
            intelligence=12,
            wisdom=11,
            charisma=18,
        ),
        armor_class=14,
        speed=30,
        initiative_bonus=2,
        hp=HitPoints(maximum=38, current=30, temporary=5),
        hit_dice=HitDice(die_type=8, total=5, remaining=3),
        spellcasting_ability=AbilityScore.CHARISMA,
        spell_slots=SpellSlots(level_1=(3, 4), level_2=(2, 3), level_3=(1, 2)),
        known_spells=["fireball", "vicious-mockery", "healing-word"],
        prepared_spells=["fireball", "healing-word"],
        saving_throw_proficiencies=[AbilityScore.DEXTERITY, AbilityScore.CHARISMA],
        skill_proficiencies=[Skill.PERFORMANCE, Skill.PERSUASION, Skill.DECEPTION],
        skill_expertise=[Skill.PERFORMANCE],
    )
    defaults.update(overrides)
    return Character(**defaults)


@pytest.mark.asyncio
async def test_create_then_get_by_id_round_trips_all_fields(repo: CharacterRepository):
    original = _make_character()
    await repo.create(original)

    loaded = await repo.get_by_id(original.id)
    assert loaded is not None

    # Identity
    assert loaded.id == original.id
    assert loaded.discord_user_id == original.discord_user_id
    assert loaded.campaign_id == original.campaign_id
    assert loaded.name == original.name

    # Immersion fields — these are the ones audit P0 #1 was scrambling.
    # Before the fix, description came back as a timestamp string,
    # portrait_url as a timestamp string, voice_id as the description text.
    assert loaded.description == original.description
    assert loaded.portrait_url == original.portrait_url
    assert loaded.voice_id == original.voice_id

    # Core info
    assert loaded.race_index == original.race_index
    assert loaded.class_index == original.class_index
    assert loaded.subclass_index == original.subclass_index
    assert loaded.level == original.level
    assert loaded.experience == original.experience
    assert loaded.background_index == original.background_index

    # Abilities
    assert loaded.abilities == original.abilities

    # Combat stats
    assert loaded.armor_class == original.armor_class
    assert loaded.speed == original.speed
    assert loaded.initiative_bonus == original.initiative_bonus

    # HP & hit dice
    assert loaded.hp == original.hp
    assert loaded.hit_dice == original.hit_dice

    # Spellcasting
    assert loaded.spellcasting_ability == original.spellcasting_ability
    assert loaded.spell_slots.get_slots(1) == (3, 4)
    assert loaded.spell_slots.get_slots(2) == (2, 3)
    assert loaded.spell_slots.get_slots(3) == (1, 2)
    assert set(loaded.known_spells) == set(original.known_spells)
    assert set(loaded.prepared_spells) == set(original.prepared_spells)

    # Proficiencies
    assert set(loaded.saving_throw_proficiencies) == set(original.saving_throw_proficiencies)
    assert set(loaded.skill_proficiencies) == set(original.skill_proficiencies)
    assert set(loaded.skill_expertise) == set(original.skill_expertise)


@pytest.mark.asyncio
async def test_get_by_user_and_campaign_returns_same_character(repo: CharacterRepository):
    original = _make_character()
    await repo.create(original)

    loaded = await repo.get_by_user_and_campaign(
        original.discord_user_id, original.campaign_id
    )
    assert loaded is not None
    assert loaded.id == original.id
    # Sanity-check the immersion fields again via this code path — the audit's
    # row-indexing bug affects every loader, not just get_by_id.
    assert loaded.description == original.description
    assert loaded.portrait_url == original.portrait_url
    assert loaded.voice_id == original.voice_id


@pytest.mark.asyncio
async def test_immersion_fields_persist_when_null_or_empty(repo: CharacterRepository):
    """Defaults (empty description, None portrait_url/voice_id) survive round-trip."""
    character = _make_character(
        description="",
        portrait_url=None,
        voice_id=None,
    )
    await repo.create(character)

    loaded = await repo.get_by_id(character.id)
    assert loaded is not None
    assert loaded.description == ""
    assert loaded.portrait_url is None
    assert loaded.voice_id is None


@pytest.mark.asyncio
async def test_get_by_id_returns_none_for_missing(repo: CharacterRepository):
    assert await repo.get_by_id("nonexistent-id") is None


@pytest.mark.asyncio
async def test_update_syncs_known_and_prepared_spells(repo: CharacterRepository):
    """Audit #94: update() must add/remove known spells and flip is_prepared.

    The original update() only inserted prepared_spells; spells the character
    "unlearned" stayed in the DB and known-but-not-prepared spells could never
    be persisted.
    """
    character = _make_character(
        known_spells=["fireball", "vicious-mockery", "healing-word"],
        prepared_spells=["fireball", "healing-word"],
    )
    await repo.create(character)

    # Unlearn one, learn a new one, swap which is prepared.
    character.known_spells = ["fireball", "thunderwave", "healing-word"]
    character.prepared_spells = ["thunderwave", "healing-word"]
    await repo.update(character)

    loaded = await repo.get_by_id(character.id)
    assert loaded is not None
    assert set(loaded.known_spells) == {"fireball", "thunderwave", "healing-word"}
    assert set(loaded.prepared_spells) == {"thunderwave", "healing-word"}
    assert "vicious-mockery" not in loaded.known_spells


@pytest.mark.asyncio
async def test_cache_returns_deep_copy_not_shared_reference(repo: CharacterRepository):
    """Audit #55: cache must not expose its stored object to callers.

    Caller mutations to a returned Character should NOT bleed into the next
    cached read (else stale-state-in-cache bugs become silent).
    """
    original = _make_character()
    await repo.create(original)

    loaded_1 = await repo.get_by_user_and_campaign(
        original.discord_user_id, original.campaign_id
    )
    assert loaded_1 is not None

    # Mutate the returned object — should not leak into the cache.
    loaded_1.name = "Tampered Name"
    loaded_1.hp.current = 1

    loaded_2 = await repo.get_by_user_and_campaign(
        original.discord_user_id, original.campaign_id
    )
    assert loaded_2 is not None
    assert loaded_2.name == original.name
    assert loaded_2.hp.current == original.hp.current
    assert loaded_2 is not loaded_1  # Distinct objects


@pytest.mark.asyncio
async def test_update_invalidates_cache(repo: CharacterRepository):
    """A write must invalidate the cache so the next read sees fresh state."""
    from dnd_bot.data.repositories import character_repo as cr_module

    original = _make_character()
    await repo.create(original)

    # Populate cache
    loaded = await repo.get_by_user_and_campaign(
        original.discord_user_id, original.campaign_id
    )
    assert loaded is not None
    assert (original.discord_user_id, original.campaign_id) in cr_module._get_cache

    # Mutate via update
    loaded.hp.current = 7
    await repo.update(loaded)

    # Cache should be empty
    assert cr_module._get_cache == {}

    # Next read fetches the new value
    refetched = await repo.get_by_user_and_campaign(
        original.discord_user_id, original.campaign_id
    )
    assert refetched is not None
    assert refetched.hp.current == 7


@pytest.mark.asyncio
async def test_spell_slot_and_concentration_persist(repo: CharacterRepository):
    """Audit #3: the targeted writes combat uses to persist slot/concentration
    expenditure must survive a fresh load (combat mutates a cached Character that
    the sync path doesn't save, so the coordinator calls these directly)."""
    caster = _make_character()  # wizard-ish with slots level_1=(3,4) etc.
    await repo.create(caster)

    # Expend a level-1 slot (3 -> 2) and start concentration, as combat does.
    await repo.update_spell_slot(caster.id, 1, 2)
    await repo.update_concentration(caster.id, "haste")

    reloaded = await repo.get_by_id(caster.id)
    assert reloaded.spell_slots.get_slots(1) == (2, 4)
    assert reloaded.concentration_spell_id == "haste"

    # Clearing concentration persists too.
    await repo.update_concentration(caster.id, None)
    reloaded2 = await repo.get_by_id(caster.id)
    assert reloaded2.concentration_spell_id is None


@pytest.mark.asyncio
async def test_migration_005_idempotent_on_rerun(tmp_path: Path):
    """Migration 005 must not fail when re-applied to a DB that already has the columns.

    Catches a regression in the migration runner's duplicate-column-name fallback.
    """
    db_path = tmp_path / "migration_replay.db"

    db1 = Database(db_path=db_path)
    await db1.connect()
    await db1.disconnect()

    # Wipe schema_migrations to simulate re-running every migration on a DB
    # that already has all the columns from a prior run.
    import sqlite3
    conn = sqlite3.connect(db_path)
    conn.execute("DELETE FROM schema_migrations")
    conn.commit()
    conn.close()

    # Reconnecting should re-run all migrations. Migration 005's ALTER TABLE
    # statements would normally throw "duplicate column name" — the runner's
    # fallback path must swallow those.
    db2 = Database(db_path=db_path)
    await db2.connect()  # would raise without the fallback

    # Confirm the columns are still there and addressable.
    row = await db2.fetch_one("PRAGMA table_info(character)")
    assert row is not None  # PRAGMA returns at least one row for an existing table
    await db2.disconnect()
