"""Integration tests for inventory_repo against a real (tmp) SQLite DB.

Covers audit P0 #3 (`transfer_item` atomicity) — exercises the savepoint wrap
and the inlined stack-merge logic.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from dnd_bot.data.database import Database
from dnd_bot.data.repositories.inventory_repo import InventoryRepository
from dnd_bot.models import InventoryItem


@pytest.fixture
async def db(tmp_path: Path):
    db = Database(db_path=tmp_path / "test.db")
    await db.connect()

    # Seed a campaign + two characters with the minimal columns required by FKs
    # and NOT NULL constraints.
    await db.execute(
        "INSERT INTO campaign (id, guild_id, name, dm_user_id) VALUES (?, ?, ?, ?)",
        ("test-campaign", 99999, "Test Campaign", 12345),
    )
    for char_id, user_id, name in [
        ("char-alice", 111, "Alice"),
        ("char-bob", 222, "Bob"),
    ]:
        await db.execute(
            """
            INSERT INTO character (
                id, discord_user_id, campaign_id, name,
                race_index, class_index, level,
                hp_max, hp_current, hit_dice_type, hit_dice_total, hit_dice_remaining
            ) VALUES (?, ?, 'test-campaign', ?, 'human', 'fighter', 1, 10, 10, 10, 1, 1)
            """,
            (char_id, user_id, name),
        )
    await db.commit()

    yield db

    await db.disconnect()


@pytest.fixture
def repo(db: Database) -> InventoryRepository:
    return InventoryRepository(db=db)


@pytest.mark.asyncio
async def test_full_stack_transfer_moves_item(repo: InventoryRepository):
    """Whole-stack transfer should change character_id without splitting."""
    item = InventoryItem(
        character_id="char-alice",
        item_index="potion-of-healing",
        item_name="Potion of Healing",
        quantity=5,
    )
    await repo.add_item(item)

    transferred = await repo.transfer_item(
        item.id, from_character_id="char-alice", to_character_id="char-bob", quantity=5
    )
    assert transferred is not None
    assert transferred.character_id == "char-bob"
    assert transferred.quantity == 5

    # Alice should now have zero potions, Bob should have 5
    alice_items = await repo.get_all_items("char-alice")
    bob_items = await repo.get_all_items("char-bob")
    assert alice_items == []
    assert len(bob_items) == 1
    assert bob_items[0].quantity == 5


@pytest.mark.asyncio
async def test_split_transfer_creates_new_recipient_stack(repo: InventoryRepository):
    """Split transfer when recipient has no existing stack of the same item."""
    item = InventoryItem(
        character_id="char-alice",
        item_index="arrow",
        item_name="Arrow",
        quantity=20,
    )
    await repo.add_item(item)

    transferred = await repo.transfer_item(
        item.id, from_character_id="char-alice", to_character_id="char-bob", quantity=7
    )
    assert transferred is not None
    assert transferred.character_id == "char-bob"
    assert transferred.quantity == 7

    alice_items = await repo.get_all_items("char-alice")
    bob_items = await repo.get_all_items("char-bob")
    assert len(alice_items) == 1
    assert alice_items[0].quantity == 13
    assert len(bob_items) == 1
    assert bob_items[0].quantity == 7


@pytest.mark.asyncio
async def test_split_transfer_merges_into_existing_recipient_stack(repo: InventoryRepository):
    """Split transfer should merge into recipient's existing stack of the same item_index.

    Audit #3 fix inlined the merge logic. Without the merge, Bob would end up
    with two arrow stacks; the test asserts a single merged stack.
    """
    alice_arrows = InventoryItem(
        character_id="char-alice", item_index="arrow", item_name="Arrow", quantity=20,
    )
    bob_arrows = InventoryItem(
        character_id="char-bob", item_index="arrow", item_name="Arrow", quantity=10,
    )
    await repo.add_item(alice_arrows)
    await repo.add_item(bob_arrows)

    await repo.transfer_item(
        alice_arrows.id, "char-alice", "char-bob", quantity=5,
    )

    alice_items = await repo.get_all_items("char-alice")
    bob_items = await repo.get_all_items("char-bob")
    assert len(alice_items) == 1
    assert alice_items[0].quantity == 15
    # Bob's existing stack should have absorbed the 5 — one row, not two
    assert len(bob_items) == 1
    assert bob_items[0].quantity == 15


@pytest.mark.asyncio
async def test_transfer_returns_none_for_wrong_source(repo: InventoryRepository):
    item = InventoryItem(
        character_id="char-alice", item_index="rope", item_name="Rope", quantity=1,
    )
    await repo.add_item(item)

    # Try to transfer from the wrong character — should refuse without mutating anything.
    result = await repo.transfer_item(
        item.id, from_character_id="char-bob", to_character_id="char-alice", quantity=1,
    )
    assert result is None

    alice_items = await repo.get_all_items("char-alice")
    assert len(alice_items) == 1
    assert alice_items[0].quantity == 1


@pytest.mark.asyncio
async def test_transfer_to_missing_character_returns_none(repo: InventoryRepository):
    item = InventoryItem(
        character_id="char-alice", item_index="rope", item_name="Rope", quantity=1,
    )
    await repo.add_item(item)

    result = await repo.transfer_item(
        item.id, "char-alice", "char-nonexistent", quantity=1,
    )
    assert result is None

    # Source unchanged
    alice_items = await repo.get_all_items("char-alice")
    assert len(alice_items) == 1
    assert alice_items[0].quantity == 1
