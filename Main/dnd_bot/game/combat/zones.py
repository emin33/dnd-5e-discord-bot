"""Abstract zone-based positioning for combat.

Instead of grid-based tracking, we use abstract zones:
- Melee pairs: Who is in melee combat with whom
- Ranged: Not in melee with anyone

This enables:
- Opportunity attack triggers when leaving melee
- Ranged disadvantage when in melee
- Movement as zone transitions (engage/disengage)
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional
import structlog

logger = structlog.get_logger()


class CombatZone(str, Enum):
    """Abstract positioning zones."""
    MELEE = "melee"    # In close combat with at least one enemy
    RANGED = "ranged"  # At distance, not in melee
    COVER_HALF = "cover_half"  # Behind half cover (+2 AC)
    COVER_THREE_QUARTERS = "cover_three_quarters"  # Behind 3/4 cover (+5 AC)
    COVER_FULL = "cover_full"  # Behind full cover (untargetable)


@dataclass
class ZoneTransition:
    """Result of a zone transition (movement)."""
    combatant_id: str
    from_zone: CombatZone
    to_zone: CombatZone
    opportunity_attackers: list[str] = field(default_factory=list)
    movement_used: int = 0
    description: str = ""


class ZoneTracker:
    """
    Tracks abstract positioning of combatants.

    Key concept: Position is RELATIONAL, not absolute.
    - "Thorin is in melee with Goblin 1 and Goblin 2"
    - "Elara is at range from all enemies"

    This avoids grid complexity while preserving tactical decisions:
    - Engaging enemies (move into melee)
    - Disengaging (move out, risking OAs unless using Disengage)
    - Ranged attacks in melee (disadvantage)
    """

    def __init__(self):
        # Set of (combatant_a, combatant_b) pairs in melee
        # Stored as frozensets for bidirectional lookup
        self._melee_pairs: set[frozenset[str]] = set()

        # Cover status per combatant
        self._cover: dict[str, CombatZone] = {}

        # Combatants who took Disengage action this turn
        self._disengaged: set[str] = set()

        # Combatants who took Dodge action this turn
        self._dodging: set[str] = set()

    # ==================== Melee Tracking ====================

    def is_in_melee_with(self, combatant_a: str, combatant_b: str) -> bool:
        """Check if two combatants are in melee with each other."""
        return frozenset([combatant_a, combatant_b]) in self._melee_pairs

    def is_in_melee(self, combatant_id: str) -> bool:
        """Check if combatant is in melee with anyone."""
        return any(combatant_id in pair for pair in self._melee_pairs)

    def get_melee_targets(self, combatant_id: str) -> list[str]:
        """Get all combatants this one is in melee with."""
        targets = []
        for pair in self._melee_pairs:
            if combatant_id in pair:
                other = next(c for c in pair if c != combatant_id)
                targets.append(other)
        return targets

    def engage_melee(self, combatant_a: str, combatant_b: str) -> None:
        """Put two combatants in melee with each other."""
        pair = frozenset([combatant_a, combatant_b])
        if pair not in self._melee_pairs:
            self._melee_pairs.add(pair)
            logger.debug(
                "melee_engaged",
                combatant_a=combatant_a,
                combatant_b=combatant_b,
            )

    def disengage_from(self, combatant_id: str, target_id: str) -> bool:
        """
        Remove combatant from melee with a specific target.

        Returns True if they were in melee.
        """
        pair = frozenset([combatant_id, target_id])
        if pair in self._melee_pairs:
            self._melee_pairs.remove(pair)
            logger.debug(
                "melee_disengaged",
                combatant=combatant_id,
                from_target=target_id,
            )
            return True
        return False

    def disengage_all(self, combatant_id: str) -> list[str]:
        """
        Remove combatant from melee with all targets.

        Returns list of targets they were in melee with.
        """
        to_remove = []
        former_targets = []

        for pair in self._melee_pairs:
            if combatant_id in pair:
                to_remove.append(pair)
                other = next(c for c in pair if c != combatant_id)
                former_targets.append(other)

        for pair in to_remove:
            self._melee_pairs.remove(pair)

        if former_targets:
            logger.debug(
                "melee_disengaged_all",
                combatant=combatant_id,
                from_targets=former_targets,
            )

        return former_targets

    # ==================== Opportunity Attacks ====================

    def get_opportunity_attackers(self, leaving_combatant: str) -> list[str]:
        """
        Get combatants who can make opportunity attacks against someone leaving melee.

        Returns empty list if the combatant used Disengage.
        """
        if leaving_combatant in self._disengaged:
            return []  # Disengage prevents OAs

        return self.get_melee_targets(leaving_combatant)

    def mark_disengaged(self, combatant_id: str) -> None:
        """Mark that a combatant used the Disengage action."""
        self._disengaged.add(combatant_id)
        logger.debug("combatant_disengaged", combatant=combatant_id)

    def is_disengaged(self, combatant_id: str) -> bool:
        """Check if combatant has taken Disengage action."""
        return combatant_id in self._disengaged

    # ==================== Dodge ====================

    def mark_dodging(self, combatant_id: str) -> None:
        """Mark that a combatant used the Dodge action."""
        self._dodging.add(combatant_id)
        logger.debug("combatant_dodging", combatant=combatant_id)

    def is_dodging(self, combatant_id: str) -> bool:
        """Check if combatant has taken Dodge action (attacks against have disadvantage)."""
        return combatant_id in self._dodging

    # ==================== Cover ====================

    def set_cover(self, combatant_id: str, cover_type: CombatZone) -> None:
        """Set cover status for a combatant."""
        if cover_type in (CombatZone.COVER_HALF, CombatZone.COVER_THREE_QUARTERS, CombatZone.COVER_FULL):
            self._cover[combatant_id] = cover_type
            logger.debug("cover_set", combatant=combatant_id, cover=cover_type.value)

    def get_cover(self, combatant_id: str) -> Optional[CombatZone]:
        """Get cover status for a combatant."""
        return self._cover.get(combatant_id)

    def clear_cover(self, combatant_id: str) -> None:
        """Clear cover status for a combatant."""
        self._cover.pop(combatant_id, None)

    def get_cover_ac_bonus(self, combatant_id: str) -> int:
        """Get AC bonus from cover."""
        cover = self._cover.get(combatant_id)
        if cover == CombatZone.COVER_HALF:
            return 2
        elif cover == CombatZone.COVER_THREE_QUARTERS:
            return 5
        return 0

    def can_target(self, attacker_id: str, target_id: str) -> tuple[bool, str]:
        """
        Check if attacker can target a combatant.

        Returns (can_target, reason).
        """
        cover = self._cover.get(target_id)
        if cover == CombatZone.COVER_FULL:
            return False, f"Target has full cover"
        return True, ""

    # ==================== Zone Queries ====================

    def get_zone(self, combatant_id: str) -> CombatZone:
        """Get the current zone of a combatant."""
        if self.is_in_melee(combatant_id):
            return CombatZone.MELEE

        cover = self._cover.get(combatant_id)
        if cover:
            return cover

        return CombatZone.RANGED

    def get_ranged_attack_disadvantage(self, attacker_id: str, target_id: str) -> tuple[bool, str]:
        """
        Check if a ranged attack has disadvantage.

        Ranged attacks have disadvantage if:
        - Attacker is in melee with anyone (not just target)
        - Target has cover (doesn't give disadvantage, but noted)
        """
        if self.is_in_melee(attacker_id):
            return True, "Attacker is in melee"
        return False, ""

    def get_melee_attack_validity(self, attacker_id: str, target_id: str) -> tuple[bool, str]:
        """
        Check if a melee attack is valid.

        Melee attacks require being in melee with the target.
        """
        if self.is_in_melee_with(attacker_id, target_id):
            return True, ""

        # Could engage if has movement (caller should check)
        return False, "Not in melee range - need to move to engage"

    # ==================== Turn Management ====================

    def on_turn_start(self, combatant_id: str) -> None:
        """Reset turn-based states for a combatant."""
        self._disengaged.discard(combatant_id)
        self._dodging.discard(combatant_id)

    def on_turn_end(self, combatant_id: str) -> None:
        """Handle end of turn for a combatant."""
        pass  # Currently nothing to do

    def on_combat_end(self) -> None:
        """Reset all state when combat ends."""
        self._melee_pairs.clear()
        self._cover.clear()
        self._disengaged.clear()
        self._dodging.clear()

    # ==================== Combat Setup ====================

    def setup_initial_positions(
        self,
        player_ids: list[str],
        enemy_ids: list[str],
        melee_engaged: bool = False,
    ) -> None:
        """
        Set up initial positions at combat start.

        If melee_engaged is True, puts enemies in melee with nearest player.
        Otherwise, everyone starts at range.
        """
        if melee_engaged and player_ids and enemy_ids:
            # Simple heuristic: first enemy engages first player
            # More sophisticated would consider who triggered combat
            self.engage_melee(player_ids[0], enemy_ids[0])

    # ==================== State Export ====================

    def get_state_summary(self) -> str:
        """Get a human-readable summary of current positions."""
        if not self._melee_pairs:
            return "All combatants at range"

        lines = ["Melee engagements:"]
        seen_pairs = set()
        for pair in self._melee_pairs:
            pair_tuple = tuple(sorted(pair))
            if pair_tuple not in seen_pairs:
                seen_pairs.add(pair_tuple)
                a, b = pair_tuple
                lines.append(f"  - {a} <-> {b}")

        return "\n".join(lines)

    def remove_combatant(self, combatant_id: str) -> None:
        """Remove a combatant from all tracking (death, flee, etc.)."""
        # Remove from all melee pairs
        self._melee_pairs = {
            pair for pair in self._melee_pairs
            if combatant_id not in pair
        }
        # Remove from other tracking
        self._cover.pop(combatant_id, None)
        self._disengaged.discard(combatant_id)
        self._dodging.discard(combatant_id)

        logger.debug("combatant_removed_from_zones", combatant=combatant_id)
