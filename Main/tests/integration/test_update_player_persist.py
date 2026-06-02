"""Integration test for audit #1: narrated update_player must reach the DB.

Before the Option-C fix, `_execute_update_player` only built a log dict and
returned `narrator_authoritative=True` while mutating nothing — so out-of-combat
damage / heal / loot / currency / conditions the narrator declared never hit the
character sheet. This is the regression guard for that whole class.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from dnd_bot.data.database import Database
from dnd_bot.data.repositories.character_repo import CharacterRepository
from dnd_bot.data.repositories.inventory_repo import InventoryRepository
from dnd_bot.data.repositories import character_repo as character_repo_module
from dnd_bot.llm.effects import EffectExecutor, ProposedEffect, EffectType
from dnd_bot.models import (
    AbilityScore, AbilityScores, Character, HitPoints, HitDice, SpellSlots,
)
from dnd_bot.models.common import Condition


@pytest.fixture
async def env(tmp_path: Path, monkeypatch):
    """Tmp DB + seeded caster, with the executor's char repo pointed at it."""
    db = Database(db_path=tmp_path / "test.db")
    await db.connect()
    await db.execute(
        "INSERT INTO campaign (id, guild_id, name, dm_user_id) VALUES (?, ?, ?, ?)",
        ("camp", 1, "Camp", 1),
    )
    await db.commit()

    char_repo = CharacterRepository(db=db)
    inv_repo = InventoryRepository(db=db)

    caster = Character(
        discord_user_id=42, campaign_id="camp", name="Elara",
        race_index="elf", class_index="wizard", level=5,
        abilities=AbilityScores(intelligence=18),
        hp=HitPoints(maximum=30, current=30, temporary=5),
        hit_dice=HitDice(die_type=6, total=5, remaining=5),
        spellcasting_ability=AbilityScore.INTELLIGENCE,
        spell_slots=SpellSlots(level_1=(4, 4), level_2=(3, 3)),
    )
    await char_repo.create(caster)

    # The executor imports get_character_repo at call time — point it at tmp.
    monkeypatch.setattr(character_repo_module, "get_character_repo",
                        lambda: _async_return(char_repo))

    session = SimpleNamespace(players={"p": SimpleNamespace(character=caster)})
    executor = EffectExecutor(session=session, inventory_repo=inv_repo)
    executor.acting_character_id = caster.id

    yield SimpleNamespace(
        db=db, char_repo=char_repo, inv_repo=inv_repo, caster=caster, executor=executor,
    )

    # Close the aiosqlite connection — leaving it open keeps a background
    # thread alive that prevents pytest from exiting (the whole-suite "hang on
    # exit" that orphaned processes for days was this fixture using `return`).
    await db.disconnect()


def _async_return(value):
    async def _coro():
        return value
    return _coro()


@pytest.mark.asyncio
async def test_narrated_damage_persists(env):
    eff = ProposedEffect(
        effect_type=EffectType.UPDATE_PLAYER,
        player_hp_delta=-8, player_damage_type="fire", player_hp_reason="trap",
    )
    result = await env.executor.execute(eff, idempotency_key="t1")
    assert result.success and result.details["persisted"] is True

    reloaded = await env.char_repo.get_by_id(env.caster.id)
    # 8 damage: 5 absorbed by temp HP, 3 off current (30 -> 27).
    assert reloaded.hp.temporary == 0
    assert reloaded.hp.current == 27


@pytest.mark.asyncio
async def test_narrated_heal_clamps_to_max(env):
    # First drop to 10, then over-heal.
    env.caster.hp.current = 10
    await env.char_repo.update(env.caster)
    eff = ProposedEffect(effect_type=EffectType.UPDATE_PLAYER, player_hp_delta=100)
    await env.executor.execute(eff, idempotency_key="t2")
    reloaded = await env.char_repo.get_by_id(env.caster.id)
    assert reloaded.hp.current == 30  # clamped to max


@pytest.mark.asyncio
async def test_narrated_loot_and_currency_persist(env):
    eff = ProposedEffect(
        effect_type=EffectType.UPDATE_PLAYER,
        player_currency_delta={"gp": 50},
        player_item_grant=[{"name": "Healing Potion", "quantity": 2}],
    )
    result = await env.executor.execute(eff, idempotency_key="t3")
    assert result.details["persisted"] is True

    currency = await env.inv_repo.get_currency(env.caster.id)
    assert currency.gold == 50
    items = await env.inv_repo.get_all_items(env.caster.id)
    assert any(i.item_name == "Healing Potion" and i.quantity == 2 for i in items)


@pytest.mark.asyncio
async def test_narrated_condition_and_slot_persist(env):
    eff = ProposedEffect(
        effect_type=EffectType.UPDATE_PLAYER,
        player_add_conditions=["poisoned"],
        player_spell_slot_used=1,
    )
    await env.executor.execute(eff, idempotency_key="t4")
    reloaded = await env.char_repo.get_by_id(env.caster.id)
    assert any(c.condition == Condition.POISONED for c in reloaded.conditions)
    assert reloaded.spell_slots.get_slots(1) == (3, 4)  # expended one of four


@pytest.mark.asyncio
async def test_idempotent_no_double_damage(env):
    """Same idempotency key must not apply HP delta twice (retry safety)."""
    eff = ProposedEffect(effect_type=EffectType.UPDATE_PLAYER, player_hp_delta=-5, player_damage_type="cold")
    await env.executor.execute(eff, idempotency_key="dupe")
    await env.executor.execute(eff, idempotency_key="dupe")  # retry
    reloaded = await env.char_repo.get_by_id(env.caster.id)
    # 5 damage once: 5 temp absorbs all → temp 0, current still 30.
    assert reloaded.hp.temporary == 0
    assert reloaded.hp.current == 30


@pytest.mark.asyncio
async def test_ambiguous_multiplayer_does_not_persist(env):
    """With 2 players and no acting hint, refuse to guess — log-only, no mutation."""
    second = Character(
        discord_user_id=99, campaign_id="camp", name="Thorin",
        race_index="dwarf", class_index="fighter",
        abilities=AbilityScores(), hp=HitPoints(maximum=40, current=40),
        hit_dice=HitDice(die_type=10, total=1, remaining=1),
    )
    env.executor.session.players["p2"] = SimpleNamespace(character=second)
    env.executor.acting_character_id = None  # no hint

    eff = ProposedEffect(effect_type=EffectType.UPDATE_PLAYER, player_hp_delta=-10, player_damage_type="fire")
    result = await env.executor.execute(eff, idempotency_key="amb")
    assert result.details["persisted"] is False

    reloaded = await env.char_repo.get_by_id(env.caster.id)
    assert reloaded.hp.current == 30  # untouched
