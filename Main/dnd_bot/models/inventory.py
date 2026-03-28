"""Inventory and item data models."""

from datetime import datetime
from enum import Enum
from typing import Optional
import uuid

from pydantic import BaseModel, Field


class ItemRarity(str, Enum):
    """Item rarity levels."""
    COMMON = "common"
    UNCOMMON = "uncommon"
    RARE = "rare"
    VERY_RARE = "very_rare"
    LEGENDARY = "legendary"
    ARTIFACT = "artifact"


class ItemCategory(str, Enum):
    """Item categories."""
    WEAPON = "weapon"
    ARMOR = "armor"
    ADVENTURING_GEAR = "adventuring-gear"
    TOOL = "tool"
    POTION = "potion"
    SCROLL = "scroll"
    WONDROUS_ITEM = "wondrous-item"
    RING = "ring"
    ROD = "rod"
    STAFF = "staff"
    WAND = "wand"
    AMMUNITION = "ammunition"
    MOUNT = "mount"
    VEHICLE = "vehicle"
    TREASURE = "treasure"
    OTHER = "other"


class InventoryItem(BaseModel):
    """An item in a character's inventory."""

    id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    character_id: str
    item_index: str  # SRD index (e.g., "longsword", "potion-of-healing")
    item_name: str  # Display name
    quantity: int = Field(default=1, ge=0)

    # Equipment state
    equipped: bool = False
    attunement_required: bool = False
    attuned: bool = False

    # Custom notes
    notes: Optional[str] = None

    # Metadata
    added_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def is_equipped(self) -> bool:
        """Check if item is currently equipped."""
        return self.equipped

    @property
    def can_attune(self) -> bool:
        """Check if item requires and allows attunement."""
        return self.attunement_required and not self.attuned


class Currency(BaseModel):
    """Character currency holdings."""

    character_id: str
    copper: int = Field(default=0, ge=0)
    silver: int = Field(default=0, ge=0)
    electrum: int = Field(default=0, ge=0)
    gold: int = Field(default=0, ge=0)
    platinum: int = Field(default=0, ge=0)

    @property
    def total_in_gold(self) -> float:
        """Get total value in gold pieces."""
        return (
            self.copper / 100
            + self.silver / 10
            + self.electrum / 2
            + self.gold
            + self.platinum * 10
        )

    @property
    def total_in_copper(self) -> int:
        """Get total value in copper pieces."""
        return (
            self.copper
            + self.silver * 10
            + self.electrum * 50
            + self.gold * 100
            + self.platinum * 1000
        )

    def add_currency(self, copper: int = 0, silver: int = 0, electrum: int = 0, gold: int = 0, platinum: int = 0) -> None:
        """Add currency."""
        self.copper += copper
        self.silver += silver
        self.electrum += electrum
        self.gold += gold
        self.platinum += platinum

    def remove_currency(self, amount_copper: int) -> bool:
        """
        Remove currency by total copper value.
        Uses largest denominations first.
        Returns True if successful, False if insufficient funds.
        """
        if amount_copper > self.total_in_copper:
            return False

        remaining = amount_copper

        # Use platinum first
        plat_needed = min(self.platinum, remaining // 1000)
        self.platinum -= plat_needed
        remaining -= plat_needed * 1000

        # Gold
        gold_needed = min(self.gold, remaining // 100)
        self.gold -= gold_needed
        remaining -= gold_needed * 100

        # Electrum
        elec_needed = min(self.electrum, remaining // 50)
        self.electrum -= elec_needed
        remaining -= elec_needed * 50

        # Silver
        silv_needed = min(self.silver, remaining // 10)
        self.silver -= silv_needed
        remaining -= silv_needed * 10

        # Copper
        if remaining > self.copper:
            # Need to break larger coin
            # This is a simplification - just use copper
            self.copper = self.total_in_copper - amount_copper
            self.silver = 0
            self.electrum = 0
            self.gold = 0
            self.platinum = 0
        else:
            self.copper -= remaining

        return True


class ItemInfo(BaseModel):
    """Parsed item information from SRD."""

    index: str
    name: str
    category: ItemCategory
    cost_copper: int = 0
    weight: float = 0.0
    description: str = ""

    # Weapon stats
    damage_dice: Optional[str] = None
    damage_type: Optional[str] = None
    weapon_range: Optional[str] = None  # "melee" or "ranged"
    properties: list[str] = Field(default_factory=list)

    # Armor stats
    armor_class: Optional[int] = None
    armor_type: Optional[str] = None  # "light", "medium", "heavy", "shield"
    strength_requirement: Optional[int] = None
    stealth_disadvantage: bool = False

    # Magic item info
    rarity: ItemRarity = ItemRarity.COMMON
    requires_attunement: bool = False
    attunement_requirements: Optional[str] = None
