"""Player-state writes against the REAL repo + REAL executor.

The unit pins for the single-writer seam drive fakes, and the fakes were
wrong in the one way that mattered: they had no ``equipped`` semantics and
no ``get_item_by_index``. So a regression shipped in which every removal
addressed by display NAME resolved through
``get_item_by_index(char_id, slug(name))`` — a query that filters
``equipped = 0`` and matches ``item_index``. It therefore missed:

- every equipped row (starting equipment auto-equips all weapons/armor), and
- every row whose SRD index is not the slug of its display name
  ("Rations (1 day)" -> rations-1-day, all seven packs, "Thieves' Tools",
  "Crossbow, light" — 88 of 237 SRD equipment entries).

The executor still returned success with an empty ``applied``, so the
narration said "you drop the longsword", the world log recorded
"player lost: Longsword", and the row stayed. The receipt-vs-state gate
could not see it: a write that neither happened nor was receipted balances.

These run against a real migrated SQLite DB so the resolution semantics are
pinned where the fakes cannot lie.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnd_bot.data.database import Database
from dnd_bot.data.repositories.character_repo import CharacterRepository
from dnd_bot.data.repositories.inventory_repo import InventoryRepository
from dnd_bot.game.session import GameSession
from dnd_bot.game.world_state import WorldState
from dnd_bot.llm.effects import EffectExecutor, EffectType, ProposedEffect
from dnd_bot.models import (
    AbilityScores, Character, HitPoints, HitDice, InventoryItem,
)


@pytest.fixture
async def rig(tmp_path: Path):
    db = Database(db_path=tmp_path / "writes.db")
    await db.connect()
    await db.execute(
        "INSERT INTO campaign (id, guild_id, name, dm_user_id) VALUES (?, ?, ?, ?)",
        ("camp", 1, "Camp", 1),
    )
    await db.commit()

    char_repo = CharacterRepository(db=db)
    inv_repo = InventoryRepository(db=db)
    hero = Character(
        discord_user_id=7, campaign_id="camp", name="Kael",
        race_index="human", class_index="fighter", level=1,
        abilities=AbilityScores(),
        hp=HitPoints(maximum=11, current=11),
        hit_dice=HitDice(die_type=10, total=1, remaining=1),
    )
    await char_repo.create(hero)

    session = GameSession(id="s", channel_id=771, guild_id=1, campaign_id="camp")
    session.world_state = WorldState(current_location="Copper Finch")
    session.add_player(7, "Kael", hero)

    executor = EffectExecutor(
        scene_registry=None, session=session, inventory_repo=inv_repo,
    )
    executor.acting_character_id = hero.id
    try:
        yield executor, inv_repo, hero
    finally:
        await db.disconnect()


async def _add(inv_repo, hero, index, name, quantity=1, equipped=False):
    return await inv_repo.add_item(InventoryItem(
        character_id=hero.id, item_index=index, item_name=name,
        quantity=quantity, equipped=equipped,
    ))


async def _remove(executor, name, item_index="", quantity=1):
    entry = {"name": name, "quantity": quantity}
    if item_index:
        entry["item_index"] = item_index
    return await executor.execute(ProposedEffect(
        effect_type=EffectType.UPDATE_PLAYER, player_item_remove=[entry],
    ))


@pytest.mark.asyncio
async def test_equipped_row_is_actually_removed(rig):
    executor, inv_repo, hero = rig
    row = await _add(inv_repo, hero, "longsword", "Longsword", equipped=True)

    result = await _remove(executor, "Longsword", row.item_index)

    assert result.details["applied"].get("items_removed") == [
        {"name": "Longsword", "quantity": 1}
    ]
    assert await inv_repo.get_item_by_id(row.id) is None


@pytest.mark.asyncio
async def test_srd_indexed_row_is_actually_removed(rig):
    """item_index != slug(item_name) — 88/237 SRD equipment entries."""
    executor, inv_repo, hero = rig
    row = await _add(inv_repo, hero, "explorers-pack", "Explorer's Pack")

    result = await _remove(executor, "Explorer's Pack")

    assert result.details["applied"].get("items_removed") == [
        {"name": "Explorer's Pack", "quantity": 1}
    ]
    assert await inv_repo.get_item_by_id(row.id) is None


@pytest.mark.asyncio
async def test_partial_quantity_decrements_rather_than_deleting(rig):
    executor, inv_repo, hero = rig
    row = await _add(inv_repo, hero, "arrow", "Arrow", quantity=20)

    await _remove(executor, "Arrow", "arrow", quantity=2)

    assert (await inv_repo.get_item_by_id(row.id)).quantity == 18


@pytest.mark.asyncio
async def test_unresolvable_removal_is_reported_not_silently_dropped(rig):
    """The old path returned success with an empty receipt and no warning."""
    executor, inv_repo, hero = rig
    await _add(inv_repo, hero, "arrow", "Arrow", quantity=5)

    result = await _remove(executor, "Obsidian Key")

    applied = result.details["applied"]
    assert "items_removed" not in applied
    assert applied.get("items_remove_unresolved") == ["Obsidian Key"]


@pytest.mark.asyncio
async def test_exact_index_beats_a_same_named_row(rig):
    """Precision first: an exact index match is the strongest signal."""
    executor, inv_repo, hero = rig
    wielded = await _add(inv_repo, hero, "dagger", "Dagger", equipped=True)
    other = await _add(inv_repo, hero, "dagger-of-venom", "Dagger", quantity=1)

    await _remove(executor, "Dagger", "dagger")

    assert await inv_repo.get_item_by_id(wielded.id) is None
    assert await inv_repo.get_item_by_id(other.id) is not None


@pytest.mark.asyncio
async def test_unequipped_row_wins_when_only_the_name_matches(rig):
    """Tie-break among name matches: consume the spare, not the wielded one."""
    executor, inv_repo, hero = rig
    wielded = await _add(inv_repo, hero, "dagger-fine", "Dagger", equipped=True)
    spare = await _add(inv_repo, hero, "dagger-plain", "Dagger", quantity=1)

    await _remove(executor, "Dagger")

    assert await inv_repo.get_item_by_id(spare.id) is None
    assert await inv_repo.get_item_by_id(wielded.id) is not None


# ── Currency: coin breaking + receipts that match the write ──────────────────


async def _spend(executor, delta):
    return await executor.execute(ProposedEffect(
        effect_type=EffectType.UPDATE_PLAYER, player_currency_delta=delta,
    ))


@pytest.mark.asyncio
async def test_spending_breaks_larger_coins(rig):
    """A 2gp payment from a platinum purse must succeed."""
    executor, inv_repo, hero = rig
    currency = await inv_repo.get_currency(hero.id)
    currency.platinum, currency.gold = 5, 0
    await inv_repo.update_currency(currency)

    result = await _spend(executor, {"cp": -200})

    after = await inv_repo.get_currency(hero.id)
    assert after.total_in_copper == 5000 - 200
    # The receipt reports what MOVED, per denomination — replaying it over
    # the opening balance must reproduce the closing balance exactly.
    delta = result.details["applied"]["currency_delta"]
    replayed = {
        "cp": 0, "sp": 0, "ep": 0, "gp": 0, "pp": 5,
    }
    for code, amount in delta.items():
        replayed[code] += amount
    assert replayed["pp"] == after.platinum
    assert replayed["gp"] == after.gold
    assert replayed["sp"] == after.silver
    assert replayed["cp"] == after.copper


@pytest.mark.asyncio
async def test_insufficient_total_wealth_moves_nothing_and_receipts_nothing(rig):
    executor, inv_repo, hero = rig
    currency = await inv_repo.get_currency(hero.id)
    currency.gold = 1
    await inv_repo.update_currency(currency)

    result = await _spend(executor, {"cp": -5000})

    after = await inv_repo.get_currency(hero.id)
    assert after.total_in_copper == 100  # untouched
    # No movement, so no currency receipt — the old code clamped to zero and
    # still receipted the full requested delta.
    assert "currency_delta" not in result.details["applied"]


@pytest.mark.asyncio
async def test_gain_is_recorded_in_the_denomination_received(rig):
    executor, inv_repo, hero = rig

    result = await _spend(executor, {"sp": 5})

    after = await inv_repo.get_currency(hero.id)
    assert after.silver == 5
    assert result.details["applied"]["currency_delta"] == {"sp": 5}
