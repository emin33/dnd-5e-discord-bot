"""Durable class-feature resource counters (Second Wind, ki, rage, ...).

The shape LONGFORM_READINESS_2026_07.md's rest section calls for:
``(character_id, resource_key, current, maximum, recharge_rule, source)``.
Rest recovery operates on these persisted rows instead of probing
attributes the base Character model never had — which is why, before
this, nothing but Warlock pact slots ever actually recovered.
"""

from typing import TYPE_CHECKING

from pydantic import BaseModel, Field

if TYPE_CHECKING:
    from .character import Character

RECHARGE_SHORT_REST = "short_rest"
RECHARGE_LONG_REST = "long_rest"


class FeatureResource(BaseModel):
    """One durable per-character feature counter."""

    character_id: str
    resource_key: str
    name: str = ""
    current: int = Field(default=0, ge=0)
    maximum: int = Field(default=0, ge=0)
    # short_rest resources also recover on a long rest; long_rest
    # resources recover only there.
    recharge_rule: str = RECHARGE_LONG_REST
    source: str = ""

    def spend(self, amount: int = 1) -> bool:
        """Consume uses. False (and no change) if not enough remain."""
        if amount <= 0 or self.current < amount:
            return False
        self.current -= amount
        return True

    def restore(self) -> int:
        """Refill to maximum; returns how many uses came back."""
        restored = self.maximum - self.current
        self.current = self.maximum
        return restored


def default_feature_resources(character: "Character") -> list[FeatureResource]:
    """SRD feature counters for a character's class and level.

    The seeding shape: called lazily by the rest flow when a character has
    no persisted rows yet, then persisted, so existing characters need no
    backfill migration. Warlock Pact Magic deliberately stays on
    ``SpellSlots`` — it already persists and recovers there.
    """
    from .common import AbilityScore

    class_index = (character.class_index or "").lower()
    level = int(character.level or 1)
    resources: list[FeatureResource] = []

    def add(key: str, name: str, maximum: int, rule: str) -> None:
        resources.append(FeatureResource(
            character_id=character.id,
            resource_key=key,
            name=name,
            current=maximum,
            maximum=maximum,
            recharge_rule=rule,
            source=f"class:{class_index}",
        ))

    if class_index == "fighter":
        add("second_wind", "Second Wind", 1, RECHARGE_SHORT_REST)
        if level >= 2:
            add(
                "action_surge", "Action Surge",
                2 if level >= 17 else 1, RECHARGE_SHORT_REST,
            )
    elif class_index == "monk":
        if level >= 2:
            add("ki", "Ki points", level, RECHARGE_SHORT_REST)
    elif class_index == "bard":
        charisma_mod = character.abilities.get_modifier(AbilityScore.CHARISMA)
        # Font of Inspiration (level 5) moves recovery to short rests.
        add(
            "bardic_inspiration", "Bardic Inspiration",
            max(1, charisma_mod),
            RECHARGE_SHORT_REST if level >= 5 else RECHARGE_LONG_REST,
        )
    elif class_index == "cleric":
        if level >= 2:
            uses = 3 if level >= 18 else (2 if level >= 6 else 1)
            add("channel_divinity", "Channel Divinity", uses, RECHARGE_SHORT_REST)
    elif class_index == "paladin":
        if level >= 3:
            add("channel_divinity", "Channel Divinity", 1, RECHARGE_SHORT_REST)
    elif class_index == "druid":
        if level >= 2:
            add("wild_shape", "Wild Shape", 2, RECHARGE_SHORT_REST)
    elif class_index == "barbarian":
        if level >= 17:
            rages = 6
        elif level >= 12:
            rages = 5
        elif level >= 6:
            rages = 4
        elif level >= 3:
            rages = 3
        else:
            rages = 2
        add("rage", "Rage", rages, RECHARGE_LONG_REST)

    return resources
