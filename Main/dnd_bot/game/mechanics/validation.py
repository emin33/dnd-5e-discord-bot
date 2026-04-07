"""PGI (Programmatic Guided Inference) validation layer.

Deterministic game-rule validation that runs BEFORE narrator LLM inference.
Invalid actions are hard-rejected with immediate feedback — no turn consumed,
no narrator tokens spent.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import structlog

from ...models.character import Character
from ...models.common import Condition
from ...models.inventory import InventoryItem, Currency
from ..magic.spellcasting import SpellcastingManager

logger = structlog.get_logger()


# ── Data Structures ──────────────────────────────────────────────────────────


class ValidationSeverity(str, Enum):
    """How severe a validation failure is."""

    HARD_FAIL = "hard_fail"  # Mechanical impossibility → immediate feedback, no LLM
    SOFT_FAIL = "soft_fail"  # Dramatic failure → narrator gets modified payload


@dataclass
class ValidationFailure:
    """A single validation failure."""

    priority: int  # 0-9, lower = more severe, determines display order
    code: str  # Machine-readable: "NO_SPELL_SLOT", "UNCONSCIOUS", etc.
    severity: ValidationSeverity
    message: str  # Player-facing feedback
    details: dict = field(default_factory=dict)


@dataclass
class ValidationResult:
    """Accumulated result from all validators."""

    failures: list[ValidationFailure] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return len(self.failures) == 0

    @property
    def has_hard_fail(self) -> bool:
        return any(f.severity == ValidationSeverity.HARD_FAIL for f in self.failures)

    @property
    def has_soft_fail(self) -> bool:
        return any(f.severity == ValidationSeverity.SOFT_FAIL for f in self.failures)

    @property
    def hard_failures(self) -> list[ValidationFailure]:
        return [f for f in self.failures if f.severity == ValidationSeverity.HARD_FAIL]

    @property
    def soft_failures(self) -> list[ValidationFailure]:
        return [f for f in self.failures if f.severity == ValidationSeverity.SOFT_FAIL]

    def player_feedback(self) -> str:
        """Format all failures into a player-facing message, sorted by priority."""
        if not self.failures:
            return ""
        sorted_failures = sorted(self.failures, key=lambda f: f.priority)
        lines = []
        for f in sorted_failures:
            lines.append(f.message)
        return "\n".join(lines)


# ── Conditions that block all actions ────────────────────────────────────────

_BLOCKING_CONDITIONS = frozenset({
    Condition.INCAPACITATED,
    Condition.PARALYZED,
    Condition.PETRIFIED,
    Condition.STUNNED,
    Condition.UNCONSCIOUS,
})

_CONDITION_MESSAGES = {
    Condition.INCAPACITATED: "You are incapacitated and cannot take actions.",
    Condition.PARALYZED: "You are paralyzed and cannot take actions or move.",
    Condition.PETRIFIED: "You are petrified and cannot take actions or move.",
    Condition.STUNNED: "You are stunned and cannot take actions.",
    Condition.UNCONSCIOUS: "You are unconscious and cannot take actions.",
}


# ── P0: Vitality ─────────────────────────────────────────────────────────────


def validate_vitality(character: Character) -> list[ValidationFailure]:
    """P0: Check if character is alive and conscious enough to act."""
    failures = []

    if character.death_saves.is_dead:
        failures.append(ValidationFailure(
            priority=0,
            code="DEAD",
            severity=ValidationSeverity.HARD_FAIL,
            message="You have died and cannot take actions.",
        ))

    if character.hp.is_unconscious and not character.death_saves.is_dead:
        failures.append(ValidationFailure(
            priority=0,
            code="UNCONSCIOUS_HP",
            severity=ValidationSeverity.HARD_FAIL,
            message="You are at 0 HP and unconscious. You can only make death saving throws.",
        ))

    return failures


# ── P1: Conditions ───────────────────────────────────────────────────────────


def validate_conditions(character: Character) -> list[ValidationFailure]:
    """P1: Check if active conditions block all actions."""
    failures = []
    active = {c.condition for c in character.conditions}

    for condition in _BLOCKING_CONDITIONS:
        if condition in active:
            failures.append(ValidationFailure(
                priority=1,
                code=f"CONDITION_{condition.value.upper()}",
                severity=ValidationSeverity.HARD_FAIL,
                message=_CONDITION_MESSAGES[condition],
                details={"condition": condition.value},
            ))

    # Exhaustion level 6 = death
    exhaustion = character.get_exhaustion_level()
    if exhaustion >= 6:
        failures.append(ValidationFailure(
            priority=1,
            code="EXHAUSTION_DEATH",
            severity=ValidationSeverity.HARD_FAIL,
            message="You have 6 levels of exhaustion and have died.",
        ))

    return failures


# ── P4: Inventory ────────────────────────────────────────────────────────────


def validate_item_exists(
    items: list[InventoryItem],
    item_name: str,
    quantity_needed: int = 1,
) -> list[ValidationFailure]:
    """P4: Check if the character has an item in their inventory."""
    if not item_name:
        return []

    # Substring match (same pattern as orchestrator._consume_resources)
    item_lower = item_name.lower()
    matching = [i for i in items if item_lower in i.item_name.lower()]

    if not matching:
        return [ValidationFailure(
            priority=4,
            code="ITEM_NOT_FOUND",
            severity=ValidationSeverity.HARD_FAIL,
            message=f"You don't have '{item_name}' in your inventory.",
            details={"item": item_name},
        )]

    total_qty = sum(i.quantity for i in matching)
    if total_qty < quantity_needed:
        return [ValidationFailure(
            priority=4,
            code="INSUFFICIENT_ITEM_QUANTITY",
            severity=ValidationSeverity.HARD_FAIL,
            message=(
                f"You only have {total_qty} '{item_name}' "
                f"but need {quantity_needed}."
            ),
            details={"item": item_name, "have": total_qty, "need": quantity_needed},
        )]

    return []


def validate_currency(
    currency: Currency,
    cost_gold: float,
) -> list[ValidationFailure]:
    """P4: Check if the character has enough currency."""
    if cost_gold <= 0:
        return []

    have = currency.total_in_gold
    if have < cost_gold:
        return [ValidationFailure(
            priority=4,
            code="INSUFFICIENT_CURRENCY",
            severity=ValidationSeverity.HARD_FAIL,
            message=(
                f"You need {cost_gold:.0f} gp but only have {have:.0f} gp."
            ),
            details={"cost_gold": cost_gold, "have_gold": have},
        )]

    return []


# ── P6: Spell Casting ────────────────────────────────────────────────────────


def _resolve_spell_from_action(
    action_text: str,
    character: Character,
) -> Optional[tuple[str, bool]]:
    """Try to resolve a spell index from the player's action text.

    Matches against the character's known/prepared spells first, then falls
    back to SRD spell list to catch "unknown spell" cases.

    Returns:
        (spell_index, is_known) or None if unresolvable.
        is_known=True means the character knows/has prepared the spell.
        is_known=False means the spell exists in SRD but character doesn't know it.
    """
    if not action_text:
        return None

    action_lower = action_text.lower()

    # Tier 1: Match against character's known/prepared spells
    all_spells = set(character.known_spells) | set(character.prepared_spells)
    for spell_index in all_spells:
        display_name = spell_index.replace("-", " ")
        if display_name in action_lower:
            return (spell_index, True)

    # Tier 2: Match against SRD spell list (catches "I cast wish" when wizard doesn't know it)
    try:
        from ...data.srd import get_srd
        srd = get_srd()
        srd_spells = srd.get_all("spells")  # dict of index → spell data
        # Sort by name length descending to match longest first ("cure wounds" before "cure")
        sorted_indices = sorted(srd_spells.keys(), key=len, reverse=True)
        for index in sorted_indices:
            display_name = index.replace("-", " ")
            if len(display_name) >= 3 and display_name in action_lower:
                return (index, False)
    except Exception:
        # SRD unavailable — fail open
        pass

    return None


def validate_spell_cast(
    character: Character,
    spell_index: Optional[str] = None,
    spell_level: Optional[int] = None,
    action_text: Optional[str] = None,
) -> list[ValidationFailure]:
    """P6: Validate spell casting prerequisites.

    Checks: known/prepared, slot availability, concentration conflict.
    Reuses SpellcastingManager.can_cast() for the core known+slot check.
    """
    # Try to resolve spell if not provided
    if spell_index is None and action_text:
        resolved = _resolve_spell_from_action(action_text, character)
        if resolved:
            spell_index, is_known = resolved
            # SRD spell that character doesn't know → hard fail immediately
            if not is_known:
                spell_name = spell_index.replace("-", " ").title()
                return [ValidationFailure(
                    priority=6,
                    code="SPELL_NOT_KNOWN",
                    severity=ValidationSeverity.HARD_FAIL,
                    message=f"You don't know the spell {spell_name}.",
                    details={"spell": spell_index, "reason": "not_in_known_or_prepared"},
                )]

    # Can't resolve → fail open, let narrator handle ambiguity
    if spell_index is None:
        return []

    failures = []
    mgr = SpellcastingManager()

    # Core check: known/prepared + slot availability
    can_cast, reason = mgr.can_cast(character, spell_index, spell_level)

    if not can_cast:
        # Build helpful context about what IS available
        available_slots = []
        for level in range(1, 10):
            current, max_slots = character.spell_slots.get_slots(level)
            if current > 0:
                available_slots.append(f"L{level} ({current})")

        slot_info = ", ".join(available_slots) if available_slots else "none"

        # Determine specific failure code
        spell_info = mgr.get_spell_info(spell_index)
        spell_name = spell_info.name if spell_info else spell_index.replace("-", " ").title()

        if "don't know" in reason or "haven't prepared" in reason:
            code = "SPELL_NOT_KNOWN"
        elif "requires at least" in reason:
            code = "SPELL_SLOT_TOO_LOW"
        else:
            code = "NO_SPELL_SLOT"

        msg = f"{reason}\nAvailable slots: {slot_info}"

        # For slot exhaustion, suggest available prepared spells at remaining levels
        if code == "NO_SPELL_SLOT" and available_slots and spell_info:
            usable = []
            prepared = set(character.prepared_spells) | set(character.known_spells)
            for idx in prepared:
                info = mgr.get_spell_info(idx)
                if info and info.level > 0:
                    if character.spell_slots.has_slot(info.level):
                        usable.append(f"{info.name} (L{info.level})")
            if usable:
                msg += f"\nPrepared spells you can still cast: {', '.join(usable[:5])}"

        failures.append(ValidationFailure(
            priority=6,
            code=code,
            severity=ValidationSeverity.HARD_FAIL,
            message=msg,
            details={"spell": spell_index, "reason": reason},
        ))
        return failures

    # Concentration conflict check (soft fail — player might choose to drop)
    spell_info = mgr.get_spell_info(spell_index)
    if spell_info and spell_info.concentration and character.is_concentrating:
        current_spell = character.concentration_spell_id or "a spell"
        current_name = current_spell.replace("-", " ").title()
        new_name = spell_info.name

        failures.append(ValidationFailure(
            priority=6,
            code="CONCENTRATION_CONFLICT",
            severity=ValidationSeverity.SOFT_FAIL,
            message=(
                f"Casting {new_name} will end your concentration on {current_name}."
            ),
            details={
                "current_concentration": current_spell,
                "new_spell": spell_index,
            },
        ))

    return failures


# ── Orchestrator Entry Point ─────────────────────────────────────────────────


async def validate_action(
    action_type: str,
    character: Character,
    action_text: str,
    items: Optional[list[InventoryItem]] = None,
    currency: Optional[Currency] = None,
    resources_consumed: Optional[list[dict]] = None,
    item_name: Optional[str] = None,
    cost_gold: float = 0,
) -> ValidationResult:
    """Run all applicable PGI validators for an action.

    Args:
        action_type: From TriageResult.action_type
        character: The acting character
        action_text: The player's original message text
        items: Pre-fetched inventory items (for inventory/resource checks)
        currency: Pre-fetched currency (for purchase checks)
        resources_consumed: From TriageResult.resources_consumed
        item_name: Specific item name from triage (for inventory actions)
        cost_gold: Gold cost from triage (for purchase actions)
    """
    result = ValidationResult()

    # ── P0: Vitality ──
    result.failures.extend(validate_vitality(character))

    # ── P1: Conditions ──
    result.failures.extend(validate_conditions(character))

    # Early return if character can't act at all
    if result.has_hard_fail:
        logger.info(
            "pgi_blocked_p0_p1",
            player=character.name,
            action_type=action_type,
            codes=[f.code for f in result.failures],
        )
        return result

    # ── Action-specific validators ──

    if action_type == "cast_spell":
        result.failures.extend(validate_spell_cast(
            character=character,
            action_text=action_text,
        ))

    elif action_type == "inventory" and item_name and items is not None:
        result.failures.extend(validate_item_exists(items, item_name))

    elif action_type == "purchase" and currency is not None and cost_gold > 0:
        result.failures.extend(validate_currency(currency, cost_gold))

    # Validate resources consumed (ammunition, consumables) for any action type.
    # Filter out spell slot references — triage brains often emit
    # {"item": "L3 Spell Slot", "quantity": 1} which isn't an inventory item.
    # Spell slot validation is already handled by validate_spell_cast().
    if resources_consumed and items is not None:
        for resource in resources_consumed:
            r_item = resource.get("item", "")
            r_qty = resource.get("quantity", 1)
            if not r_item:
                continue
            # Skip spell slot references (handled by P6 spell validation)
            if "spell slot" in r_item.lower() or "spell_slot" in r_item.lower():
                logger.debug("pgi_skip_spell_slot_resource", item=r_item)
                continue
            result.failures.extend(validate_item_exists(items, r_item, r_qty))

    if result.failures:
        logger.info(
            "pgi_validation_failures",
            player=character.name,
            action_type=action_type,
            codes=[f.code for f in result.failures],
            severities=[f.severity.value for f in result.failures],
        )

    return result
