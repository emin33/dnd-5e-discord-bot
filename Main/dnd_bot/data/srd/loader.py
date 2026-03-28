"""SRD (System Reference Document) data loader.

Loads D&D 5e SRD data from the 5e-database JSON files into memory
for fast access during gameplay.
"""

import json
from pathlib import Path
from typing import Any, Optional
import structlog

from ...config import get_settings

logger = structlog.get_logger()


class SRDDataLoader:
    """Loads and caches SRD data from local JSON files."""

    # Mapping of category names to JSON file names
    DATA_FILES = {
        "ability_scores": "5e-SRD-Ability-Scores.json",
        "alignments": "5e-SRD-Alignments.json",
        "backgrounds": "5e-SRD-Backgrounds.json",
        "classes": "5e-SRD-Classes.json",
        "conditions": "5e-SRD-Conditions.json",
        "damage_types": "5e-SRD-Damage-Types.json",
        "equipment_categories": "5e-SRD-Equipment-Categories.json",
        "equipment": "5e-SRD-Equipment.json",
        "feats": "5e-SRD-Feats.json",
        "features": "5e-SRD-Features.json",
        "languages": "5e-SRD-Languages.json",
        "levels": "5e-SRD-Levels.json",
        "magic_items": "5e-SRD-Magic-Items.json",
        "magic_schools": "5e-SRD-Magic-Schools.json",
        "monsters": "5e-SRD-Monsters.json",
        "proficiencies": "5e-SRD-Proficiencies.json",
        "races": "5e-SRD-Races.json",
        "rule_sections": "5e-SRD-Rule-Sections.json",
        "rules": "5e-SRD-Rules.json",
        "skills": "5e-SRD-Skills.json",
        "spells": "5e-SRD-Spells.json",
        "subclasses": "5e-SRD-Subclasses.json",
        "subraces": "5e-SRD-Subraces.json",
        "traits": "5e-SRD-Traits.json",
        "weapon_properties": "5e-SRD-Weapon-Properties.json",
    }

    def __init__(self, srd_path: Optional[Path] = None):
        settings = get_settings()
        self.srd_path = srd_path or settings.srd_path
        self._cache: dict[str, dict[str, Any]] = {}
        self._loaded = False

    def load_all(self) -> None:
        """Load all SRD data files into cache."""
        if self._loaded:
            return

        logger.info("loading_srd_data", path=str(self.srd_path))

        for category, filename in self.DATA_FILES.items():
            filepath = self.srd_path / filename
            if not filepath.exists():
                logger.warning("srd_file_not_found", category=category, path=str(filepath))
                continue

            try:
                with open(filepath, "r", encoding="utf-8") as f:
                    data = json.load(f)

                # Index by 'index' field for O(1) lookup
                self._cache[category] = {item["index"]: item for item in data}

                logger.debug(
                    "srd_category_loaded",
                    category=category,
                    count=len(self._cache[category]),
                )
            except json.JSONDecodeError as e:
                logger.error("srd_json_error", category=category, error=str(e))
            except KeyError as e:
                logger.error("srd_missing_index", category=category, error=str(e))

        self._loaded = True
        total_items = sum(len(cat) for cat in self._cache.values())
        logger.info("srd_data_loaded", categories=len(self._cache), total_items=total_items)

    def get(self, category: str, index: str) -> Optional[dict[str, Any]]:
        """Get a specific item from a category by its index."""
        if not self._loaded:
            self.load_all()
        return self._cache.get(category, {}).get(index)

    def get_all(self, category: str) -> dict[str, Any]:
        """Get all items in a category."""
        if not self._loaded:
            self.load_all()
        return self._cache.get(category, {})

    def search(self, category: str, name: str) -> list[dict[str, Any]]:
        """Search for items by name (case-insensitive partial match)."""
        if not self._loaded:
            self.load_all()

        items = self._cache.get(category, {})
        name_lower = name.lower()
        return [
            item for item in items.values()
            if name_lower in item.get("name", "").lower()
        ]

    # Convenience methods for common lookups

    def get_class(self, index: str) -> Optional[dict[str, Any]]:
        """Get a class by index (e.g., 'fighter', 'wizard')."""
        return self.get("classes", index)

    def get_race(self, index: str) -> Optional[dict[str, Any]]:
        """Get a race by index (e.g., 'dwarf', 'elf')."""
        return self.get("races", index)

    def get_subrace(self, index: str) -> Optional[dict[str, Any]]:
        """Get a subrace by index (e.g., 'hill-dwarf')."""
        return self.get("subraces", index)

    def get_spell(self, index: str) -> Optional[dict[str, Any]]:
        """Get a spell by index (e.g., 'fireball')."""
        return self.get("spells", index)

    def get_monster(self, index: str) -> Optional[dict[str, Any]]:
        """Get a monster by index (e.g., 'goblin')."""
        return self.get("monsters", index)

    def get_equipment(self, index: str) -> Optional[dict[str, Any]]:
        """Get equipment by index (e.g., 'longsword')."""
        return self.get("equipment", index)

    def get_magic_item(self, index: str) -> Optional[dict[str, Any]]:
        """Get a magic item by index."""
        return self.get("magic_items", index)

    def get_condition(self, index: str) -> Optional[dict[str, Any]]:
        """Get a condition by index (e.g., 'blinded')."""
        return self.get("conditions", index)

    def get_feature(self, index: str) -> Optional[dict[str, Any]]:
        """Get a class feature by index."""
        return self.get("features", index)

    def get_trait(self, index: str) -> Optional[dict[str, Any]]:
        """Get a racial trait by index."""
        return self.get("traits", index)

    def get_background(self, index: str) -> Optional[dict[str, Any]]:
        """Get a background by index."""
        return self.get("backgrounds", index)

    def get_skill(self, index: str) -> Optional[dict[str, Any]]:
        """Get a skill by index (e.g., 'athletics')."""
        return self.get("skills", index)

    def get_proficiency(self, index: str) -> Optional[dict[str, Any]]:
        """Get a proficiency by index."""
        return self.get("proficiencies", index)

    def get_level_data(self, class_index: str, level: int) -> Optional[dict[str, Any]]:
        """Get class progression data for a specific level."""
        # Levels are stored as "class-level" format, e.g., "fighter-1"
        level_index = f"{class_index}-{level}"
        return self.get("levels", level_index)

    def get_all_races(self) -> list[dict[str, Any]]:
        """Get all playable races."""
        return list(self.get_all("races").values())

    def get_all_classes(self) -> list[dict[str, Any]]:
        """Get all playable classes."""
        return list(self.get_all("classes").values())

    def get_all_backgrounds(self) -> list[dict[str, Any]]:
        """Get all backgrounds."""
        return list(self.get_all("backgrounds").values())

    def get_spells_by_class(self, class_index: str) -> list[dict[str, Any]]:
        """Get all spells available to a class."""
        spells = self.get_all("spells")
        return [
            spell for spell in spells.values()
            if any(
                cls.get("index") == class_index
                for cls in spell.get("classes", [])
            )
        ]

    def get_spells_by_level(self, level: int) -> list[dict[str, Any]]:
        """Get all spells of a specific level (0 for cantrips)."""
        spells = self.get_all("spells")
        return [spell for spell in spells.values() if spell.get("level") == level]

    def get_equipment_by_category(self, category: str) -> list[dict[str, Any]]:
        """Get equipment by category (e.g., 'weapon', 'armor')."""
        equipment = self.get_all("equipment")
        return [
            item for item in equipment.values()
            if item.get("equipment_category", {}).get("index") == category
        ]

    def fuzzy_match_monster(self, name: str, threshold: float = 0.6) -> Optional[dict[str, Any]]:
        """
        Fuzzy match a name to an SRD monster.

        Tries in order:
        1. Exact index match (e.g., "guard" -> "guard")
        2. Exact name match (case-insensitive)
        3. Best partial match above threshold using SequenceMatcher

        Returns the monster data dict, or None if no good match.
        """
        import difflib
        import re

        if not self._loaded:
            self.load_all()

        monsters = self._cache.get("monsters", {})
        if not monsters:
            return None

        # Normalize: lowercase, strip articles, replace spaces with hyphens
        normalized = re.sub(r'\b(a|an|the)\b', '', name.lower()).strip()
        normalized = re.sub(r'\s+', ' ', normalized).strip()
        index_form = normalized.replace(' ', '-')

        # 1. Exact index match
        if index_form in monsters:
            return monsters[index_form]

        # 2. Exact name match (case-insensitive)
        for monster in monsters.values():
            if monster.get("name", "").lower() == normalized:
                return monster

        # 3. Best fuzzy match
        best_score = 0.0
        best_match = None

        for monster in monsters.values():
            monster_name = monster.get("name", "").lower()
            score = difflib.SequenceMatcher(None, normalized, monster_name).ratio()
            if score > best_score:
                best_score = score
                best_match = monster

        if best_score >= threshold and best_match:
            logger.debug(
                "fuzzy_monster_match",
                query=name,
                match=best_match.get("name"),
                score=best_score,
            )
            return best_match

        return None


# Global SRD loader instance
_srd: Optional[SRDDataLoader] = None


def get_srd() -> SRDDataLoader:
    """Get the global SRD loader instance."""
    global _srd
    if _srd is None:
        _srd = SRDDataLoader()
        _srd.load_all()
    return _srd
