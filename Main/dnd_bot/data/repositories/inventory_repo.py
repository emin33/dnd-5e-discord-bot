"""Inventory repository for database operations."""

from datetime import datetime
from typing import Optional
import uuid

from ...models import InventoryItem, Currency
from ..database import Database, get_database


class InventoryRepository:
    """Repository for inventory database operations."""

    def __init__(self, db: Optional[Database] = None):
        self._db = db

    async def _get_db(self) -> Database:
        if self._db:
            return self._db
        return await get_database()

    # ==================== Item Operations ====================

    async def add_item(self, item: InventoryItem) -> InventoryItem:
        """Add an item to a character's inventory."""
        db = await self._get_db()

        # Check if item already exists (stack it)
        existing = await self.get_item_by_index(item.character_id, item.item_index)
        if existing and not item.equipped:
            # Stack items (except equipped ones)
            existing.quantity += item.quantity
            await self.update_item(existing)
            return existing

        await db.execute(
            """
            INSERT INTO character_inventory
            (id, character_id, item_index, item_name, quantity, equipped, attunement_required, attuned, notes, added_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                item.id,
                item.character_id,
                item.item_index,
                item.item_name,
                item.quantity,
                1 if item.equipped else 0,
                1 if item.attunement_required else 0,
                1 if item.attuned else 0,
                item.notes,
                item.added_at.isoformat(),
            ),
        )
        await db.commit()
        return item

    async def get_item_by_id(self, item_id: str) -> Optional[InventoryItem]:
        """Get an inventory item by its ID."""
        db = await self._get_db()

        row = await db.fetch_one(
            "SELECT * FROM character_inventory WHERE id = ?",
            (item_id,),
        )

        if not row:
            return None

        return self._row_to_item(row)

    async def get_item_by_index(
        self, character_id: str, item_index: str
    ) -> Optional[InventoryItem]:
        """Get an inventory item by character and item index."""
        db = await self._get_db()

        row = await db.fetch_one(
            "SELECT * FROM character_inventory WHERE character_id = ? AND item_index = ? AND equipped = 0",
            (character_id, item_index),
        )

        if not row:
            return None

        return self._row_to_item(row)

    async def get_all_items(self, character_id: str) -> list[InventoryItem]:
        """Get all items in a character's inventory."""
        db = await self._get_db()

        rows = await db.fetch_all(
            "SELECT * FROM character_inventory WHERE character_id = ? ORDER BY equipped DESC, item_name ASC",
            (character_id,),
        )

        return [self._row_to_item(row) for row in rows]

    async def get_equipped_items(self, character_id: str) -> list[InventoryItem]:
        """Get all equipped items for a character."""
        db = await self._get_db()

        rows = await db.fetch_all(
            "SELECT * FROM character_inventory WHERE character_id = ? AND equipped = 1",
            (character_id,),
        )

        return [self._row_to_item(row) for row in rows]

    async def update_item(self, item: InventoryItem) -> bool:
        """Update an inventory item."""
        db = await self._get_db()

        await db.execute(
            """
            UPDATE character_inventory
            SET quantity = ?, equipped = ?, attuned = ?, notes = ?
            WHERE id = ?
            """,
            (
                item.quantity,
                1 if item.equipped else 0,
                1 if item.attuned else 0,
                item.notes,
                item.id,
            ),
        )
        await db.commit()
        return True

    async def remove_item(self, item_id: str, quantity: int = 1) -> bool:
        """Remove an item or reduce its quantity."""
        db = await self._get_db()

        item = await self.get_item_by_id(item_id)
        if not item:
            return False

        if quantity >= item.quantity:
            # Remove entirely
            await db.execute(
                "DELETE FROM character_inventory WHERE id = ?",
                (item_id,),
            )
        else:
            # Reduce quantity
            await db.execute(
                "UPDATE character_inventory SET quantity = quantity - ? WHERE id = ?",
                (quantity, item_id),
            )

        await db.commit()
        return True

    async def equip_item(self, item_id: str) -> bool:
        """Equip an item."""
        db = await self._get_db()

        await db.execute(
            "UPDATE character_inventory SET equipped = 1 WHERE id = ?",
            (item_id,),
        )
        await db.commit()
        return True

    async def unequip_item(self, item_id: str) -> bool:
        """Unequip an item."""
        db = await self._get_db()

        await db.execute(
            "UPDATE character_inventory SET equipped = 0 WHERE id = ?",
            (item_id,),
        )
        await db.commit()
        return True

    async def attune_item(self, item_id: str) -> bool:
        """Attune to a magic item."""
        db = await self._get_db()

        await db.execute(
            "UPDATE character_inventory SET attuned = 1 WHERE id = ?",
            (item_id,),
        )
        await db.commit()
        return True

    async def unattune_item(self, item_id: str) -> bool:
        """End attunement to a magic item."""
        db = await self._get_db()

        await db.execute(
            "UPDATE character_inventory SET attuned = 0 WHERE id = ?",
            (item_id,),
        )
        await db.commit()
        return True

    async def get_attuned_count(self, character_id: str) -> int:
        """Get the number of attuned items (max 3 normally)."""
        db = await self._get_db()

        row = await db.fetch_one(
            "SELECT COUNT(*) FROM character_inventory WHERE character_id = ? AND attuned = 1",
            (character_id,),
        )

        return row[0] if row else 0

    async def transfer_item(
        self, item_id: str, from_character_id: str, to_character_id: str, quantity: int = 1
    ) -> Optional[InventoryItem]:
        """Transfer an item between characters."""
        db = await self._get_db()

        item = await self.get_item_by_id(item_id)
        if not item or item.character_id != from_character_id:
            return None

        # Validate target character exists
        target_exists = await db.fetch_one(
            "SELECT 1 FROM character WHERE id = ?", (to_character_id,)
        )
        if not target_exists:
            return None

        if item.equipped:
            # Unequip first
            item.equipped = False
            item.attuned = False

        if quantity >= item.quantity:
            # Transfer entire stack
            item.character_id = to_character_id
            await db.execute(
                "UPDATE character_inventory SET character_id = ?, equipped = 0, attuned = 0 WHERE id = ?",
                (to_character_id, item_id),
            )
        else:
            # Split stack
            item.quantity -= quantity
            await self.update_item(item)

            # Create new item for recipient
            new_item = InventoryItem(
                character_id=to_character_id,
                item_index=item.item_index,
                item_name=item.item_name,
                quantity=quantity,
                attunement_required=item.attunement_required,
            )
            await self.add_item(new_item)
            item = new_item

        await db.commit()
        return item

    def _row_to_item(self, row) -> InventoryItem:
        """Convert a database row to an InventoryItem."""
        added_at = datetime.utcnow()
        if row[9]:
            try:
                added_at = datetime.fromisoformat(row[9])
            except (ValueError, TypeError):
                pass

        return InventoryItem(
            id=row[0],
            character_id=row[1],
            item_index=row[2],
            item_name=row[3],
            quantity=row[4],
            equipped=bool(row[5]),
            attunement_required=bool(row[6]),
            attuned=bool(row[7]),
            notes=row[8],
            added_at=added_at,
        )

    # ==================== Currency Operations ====================

    async def get_currency(self, character_id: str) -> Currency:
        """Get a character's currency."""
        db = await self._get_db()

        row = await db.fetch_one(
            "SELECT * FROM character_currency WHERE character_id = ?",
            (character_id,),
        )

        if not row:
            # Create default currency record
            currency = Currency(character_id=character_id)
            await db.execute(
                """
                INSERT INTO character_currency (character_id, copper, silver, electrum, gold, platinum)
                VALUES (?, 0, 0, 0, 0, 0)
                """,
                (character_id,),
            )
            await db.commit()
            return currency

        return Currency(
            character_id=row[0],
            copper=row[1],
            silver=row[2],
            electrum=row[3],
            gold=row[4],
            platinum=row[5],
        )

    async def update_currency(self, currency: Currency) -> bool:
        """Update a character's currency."""
        db = await self._get_db()

        await db.execute(
            """
            UPDATE character_currency
            SET copper = ?, silver = ?, electrum = ?, gold = ?, platinum = ?
            WHERE character_id = ?
            """,
            (
                currency.copper,
                currency.silver,
                currency.electrum,
                currency.gold,
                currency.platinum,
                currency.character_id,
            ),
        )
        await db.commit()
        return True

    async def add_gold(self, character_id: str, amount: int) -> Currency:
        """Add gold to a character."""
        currency = await self.get_currency(character_id)
        currency.gold += amount
        await self.update_currency(currency)
        return currency

    async def remove_gold(self, character_id: str, amount: int) -> tuple[bool, Currency]:
        """Remove gold from a character. Returns (success, currency)."""
        currency = await self.get_currency(character_id)

        # Convert to copper for easier math
        amount_copper = amount * 100
        if currency.total_in_copper < amount_copper:
            return False, currency

        currency.remove_currency(amount_copper)
        await self.update_currency(currency)
        return True, currency


# Global repository instance
_repo: Optional[InventoryRepository] = None


async def get_inventory_repo() -> InventoryRepository:
    """Get the global inventory repository."""
    global _repo
    if _repo is None:
        _repo = InventoryRepository()
    return _repo
