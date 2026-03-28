"""Starting equipment assignment for new characters.

D&D 5e characters start with class-based equipment and gold.
This module handles automatic assignment of starting gear.
"""

from typing import Optional
import structlog

from ...data.srd import get_srd
from ...data.repositories import get_inventory_repo
from ...models import InventoryItem

logger = structlog.get_logger()


# Default starting gold by class (in gold pieces)
# Based on D&D 5e PHB starting wealth averages
STARTING_GOLD_BY_CLASS = {
    "barbarian": 40,   # 2d4 x 10 gp average
    "bard": 100,       # 5d4 x 10 gp average
    "cleric": 100,     # 5d4 x 10 gp average
    "druid": 40,       # 2d4 x 10 gp average
    "fighter": 100,    # 5d4 x 10 gp average
    "monk": 10,        # 5d4 gp average (no x10)
    "paladin": 100,    # 5d4 x 10 gp average
    "ranger": 100,     # 5d4 x 10 gp average
    "rogue": 80,       # 4d4 x 10 gp average
    "sorcerer": 60,    # 3d4 x 10 gp average
    "warlock": 80,     # 4d4 x 10 gp average
    "wizard": 80,      # 4d4 x 10 gp average
}

# Default equipment choice selections by class
# When class has options, pick sensible defaults
DEFAULT_EQUIPMENT_CHOICES = {
    "barbarian": {
        # (a) greataxe or (b) any martial melee weapon → greataxe
        0: 0,  # first option (greataxe)
        # (a) two handaxes or (b) any simple weapon → two handaxes
        1: 0,
    },
    "bard": {
        # (a) rapier, (b) longsword, or (c) any simple weapon → rapier
        0: 0,
        # (a) diplomat's pack or (b) entertainer's pack → entertainer's pack
        1: 1,
        # (a) lute or (b) any other musical instrument → lute
        2: 0,
    },
    "cleric": {
        # (a) mace or (b) warhammer → mace
        0: 0,
        # (a) scale mail, (b) leather armor, or (c) chain mail → scale mail
        1: 0,
        # (a) light crossbow and bolts or (b) any simple weapon → light crossbow
        2: 0,
        # (a) priest's pack or (b) explorer's pack → priest's pack
        3: 0,
    },
    "druid": {
        # (a) wooden shield or (b) any simple weapon → wooden shield
        0: 0,
        # (a) scimitar or (b) any simple melee weapon → scimitar
        1: 0,
    },
    "fighter": {
        # (a) chain mail or (b) leather armor, longbow, arrows → chain mail
        0: 0,
        # (a) martial weapon and shield or (b) two martial weapons → sword and shield
        1: 0,
        # (a) light crossbow and bolts or (b) two handaxes → light crossbow
        2: 0,
        # (a) dungeoneer's pack or (b) explorer's pack → explorer's pack
        3: 1,
    },
    "monk": {
        # (a) shortsword or (b) any simple weapon → shortsword
        0: 0,
        # (a) dungeoneer's pack or (b) explorer's pack → explorer's pack
        1: 1,
    },
    "paladin": {
        # (a) martial weapon and shield or (b) two martial weapons → sword and shield
        0: 0,
        # (a) five javelins or (b) any simple melee weapon → five javelins
        1: 0,
        # (a) priest's pack or (b) explorer's pack → explorer's pack
        2: 1,
    },
    "ranger": {
        # (a) scale mail or (b) leather armor → leather armor
        0: 1,
        # (a) two shortswords or (b) two simple melee weapons → two shortswords
        1: 0,
        # (a) dungeoneer's pack or (b) explorer's pack → explorer's pack
        2: 1,
    },
    "rogue": {
        # (a) rapier or (b) shortsword → rapier
        0: 0,
        # (a) shortbow and arrows or (b) shortsword → shortbow
        1: 0,
        # (a) burglar's pack, (b) dungeoneer's pack, or (c) explorer's pack → burglar's pack
        2: 0,
    },
    "sorcerer": {
        # (a) light crossbow and bolts or (b) any simple weapon → light crossbow
        0: 0,
        # (a) component pouch or (b) arcane focus → component pouch
        1: 0,
        # (a) dungeoneer's pack or (b) explorer's pack → explorer's pack
        2: 1,
    },
    "warlock": {
        # (a) light crossbow and bolts or (b) any simple weapon → light crossbow
        0: 0,
        # (a) component pouch or (b) arcane focus → arcane focus
        1: 1,
        # (a) scholar's pack or (b) dungeoneer's pack → scholar's pack
        2: 0,
    },
    "wizard": {
        # (a) quarterstaff or (b) dagger → quarterstaff
        0: 0,
        # (a) component pouch or (b) arcane focus → arcane focus
        1: 1,
        # (a) scholar's pack or (b) explorer's pack → scholar's pack
        2: 0,
    },
}


async def assign_starting_equipment(
    character_id: str,
    class_index: str,
) -> dict:
    """
    Assign starting equipment and gold to a new character.

    Args:
        character_id: The character's database ID
        class_index: The character's class (e.g., "ranger", "fighter")

    Returns:
        Dict with assigned items and gold amount
    """
    srd = get_srd()
    inv_repo = await get_inventory_repo()

    class_data = srd.get_class(class_index.lower())
    if not class_data:
        logger.warning("class_not_found_for_equipment", class_index=class_index)
        return {"items": [], "gold": 0}

    assigned_items = []

    # 1. Add guaranteed starting equipment
    for entry in class_data.get("starting_equipment", []):
        equipment = entry.get("equipment", {})
        quantity = entry.get("quantity", 1)

        item_index = equipment.get("index", "")
        item_name = equipment.get("name", item_index)

        if item_index:
            item = InventoryItem(
                character_id=character_id,
                item_index=item_index,
                item_name=item_name,
                quantity=quantity,
            )
            await inv_repo.add_item(item)
            assigned_items.append({"name": item_name, "quantity": quantity})

            logger.debug(
                "starting_item_added",
                character_id=character_id,
                item=item_name,
                quantity=quantity,
            )

    # 2. Add equipment from default choices
    class_choices = DEFAULT_EQUIPMENT_CHOICES.get(class_index.lower(), {})
    options = class_data.get("starting_equipment_options", [])

    for option_idx, option in enumerate(options):
        choice_idx = class_choices.get(option_idx, 0)  # Default to first option
        items_to_add = _resolve_equipment_option(option, choice_idx, srd)

        for item_data in items_to_add:
            item = InventoryItem(
                character_id=character_id,
                item_index=item_data["index"],
                item_name=item_data["name"],
                quantity=item_data.get("quantity", 1),
            )
            await inv_repo.add_item(item)
            assigned_items.append({"name": item_data["name"], "quantity": item_data.get("quantity", 1)})

    # 2.5. Auto-equip weapons and armor
    all_items = await inv_repo.get_all_items(character_id)
    for item in all_items:
        equip_data = srd.get_equipment(item.item_index)
        if equip_data:
            cat = equip_data.get("equipment_category", {}).get("index", "")
            if cat in ("weapon", "armor"):
                item.equipped = True
                await inv_repo.update_item(item)

    # 3. Add starting gold
    starting_gold = STARTING_GOLD_BY_CLASS.get(class_index.lower(), 50)
    await inv_repo.add_gold(character_id, starting_gold)

    logger.info(
        "starting_equipment_assigned",
        character_id=character_id,
        class_index=class_index,
        item_count=len(assigned_items),
        gold=starting_gold,
    )

    return {
        "items": assigned_items,
        "gold": starting_gold,
    }


def _resolve_equipment_option(option: dict, choice_idx: int, srd) -> list[dict]:
    """
    Resolve a single equipment option to concrete items.

    Args:
        option: The option dict from SRD data
        choice_idx: Which choice to select (0-indexed)
        srd: SRD data loader

    Returns:
        List of items to add: [{"index": "...", "name": "...", "quantity": N}, ...]
    """
    items = []

    from_data = option.get("from", {})
    options_array = from_data.get("options", [])

    if not options_array:
        return items

    # Clamp choice to valid range
    choice_idx = min(choice_idx, len(options_array) - 1)
    chosen = options_array[choice_idx]

    option_type = chosen.get("option_type", "")

    if option_type == "counted_reference":
        # Simple: N copies of a specific item
        count = chosen.get("count", 1)
        of = chosen.get("of", {})
        if of.get("index"):
            items.append({
                "index": of["index"],
                "name": of.get("name", of["index"]),
                "quantity": count,
            })

    elif option_type == "multiple":
        # Multiple items in one choice
        for sub_item in chosen.get("items", []):
            sub_type = sub_item.get("option_type", "")
            if sub_type == "counted_reference":
                of = sub_item.get("of", {})
                if of.get("index"):
                    items.append({
                        "index": of["index"],
                        "name": of.get("name", of["index"]),
                        "quantity": sub_item.get("count", 1),
                    })

    elif option_type == "choice":
        # Nested choice - pick a default from the sub-category
        sub_choice = chosen.get("choice", {})
        from_data = sub_choice.get("from", {})

        # If it's an equipment category, pick a sensible default
        if from_data.get("option_set_type") == "equipment_category":
            category = from_data.get("equipment_category", {}).get("index", "")
            count = sub_choice.get("choose", 1)

            # Pick defaults based on category
            default_items = _get_default_from_category(category, count, srd)
            items.extend(default_items)

    return items


def _get_default_from_category(category: str, count: int, srd) -> list[dict]:
    """Get default items from an equipment category."""
    # Sensible defaults for common categories
    defaults = {
        "simple-melee-weapons": [
            {"index": "handaxe", "name": "Handaxe", "quantity": 1},
        ],
        "simple-weapons": [
            {"index": "javelin", "name": "Javelin", "quantity": 1},
        ],
        "martial-melee-weapons": [
            {"index": "longsword", "name": "Longsword", "quantity": 1},
        ],
        "martial-weapons": [
            {"index": "longsword", "name": "Longsword", "quantity": 1},
        ],
        "simple-ranged-weapons": [
            {"index": "shortbow", "name": "Shortbow", "quantity": 1},
        ],
        "musical-instruments": [
            {"index": "lute", "name": "Lute", "quantity": 1},
        ],
        "artisans-tools": [
            {"index": "smiths-tools", "name": "Smith's Tools", "quantity": 1},
        ],
    }

    if category in defaults:
        items = defaults[category][:count]
        # Ensure we have enough items
        while len(items) < count:
            items.append(items[0].copy())
        return items

    return []


# Export for use in character creation
__all__ = ["assign_starting_equipment", "STARTING_GOLD_BY_CLASS"]
