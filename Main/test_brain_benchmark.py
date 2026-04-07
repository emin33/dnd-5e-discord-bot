"""Brain + PGI Pipeline Benchmark — Triage accuracy & validation coverage.

Seeds a mock character sheet and runs player actions through the full
triage → PGI validation pipeline on Gemma. Tests both that triage classifies
correctly AND that PGI catches invalid actions before narrator fires.

Usage:
    python test_brain_benchmark.py                    # Full benchmark
    python test_brain_benchmark.py --category PGI     # PGI cases only
    python test_brain_benchmark.py --category TRIAGE  # Triage cases only
    python test_brain_benchmark.py --model gemma4:26b # Override model
"""

import asyncio
import argparse
import json
import time
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from dataclasses import dataclass, field, asdict
from typing import Optional

from dnd_bot.models import Character, AbilityScores, HitPoints, HitDice, SpellSlots
from dnd_bot.models.common import AbilityScore, Condition, Skill
from dnd_bot.models.character import CharacterCondition, DeathSaves
from dnd_bot.models.inventory import InventoryItem, Currency
from dnd_bot.game.mechanics.validation import (
    validate_action,
    ValidationResult,
    ValidationSeverity,
)


# ── Character Variants ───────────────────────────────────────────────────────
# Each variant is a specific character state that PGI test cases run against.


def _make_wizard(
    name: str = "Elara",
    level: int = 5,
    slots: tuple = ((4, 4), (3, 3), (2, 2)),
    known: list = None,
    prepared: list = None,
    concentration: str = None,
    hp_current: int = 27,
    conditions: list = None,
    death_failures: int = 0,
) -> Character:
    """Build a wizard with specific state."""
    if known is None:
        known = [
            "fire-bolt", "mage-hand", "prestidigitation",  # cantrips
            "magic-missile", "shield", "detect-magic",      # L1
            "scorching-ray", "misty-step",                  # L2
            "fireball", "counterspell", "haste",            # L3
        ]
    if prepared is None:
        prepared = [
            "magic-missile", "shield", "detect-magic",
            "scorching-ray", "misty-step",
            "fireball", "counterspell", "haste",
        ]

    slot_1, slot_2, slot_3 = slots
    char = Character(
        discord_user_id=100,
        campaign_id="benchmark",
        name=name,
        race_index="half-elf",
        class_index="wizard",
        level=level,
        abilities=AbilityScores(
            strength=8, dexterity=14, constitution=13,
            intelligence=18, wisdom=12, charisma=10,
        ),
        hp=HitPoints(maximum=27, current=hp_current),
        hit_dice=HitDice(die_type=6, total=level, remaining=level),
        armor_class=12,
        speed=30,
        spellcasting_ability=AbilityScore.INTELLIGENCE,
        spell_slots=SpellSlots(
            level_1=slot_1, level_2=slot_2, level_3=slot_3,
        ),
        known_spells=known,
        prepared_spells=prepared,
        concentration_spell_id=concentration,
        saving_throw_proficiencies=[AbilityScore.INTELLIGENCE, AbilityScore.WISDOM],
        skill_proficiencies=[Skill.ARCANA, Skill.INVESTIGATION, Skill.HISTORY],
        conditions=conditions or [],
    )
    if death_failures > 0:
        char.death_saves.failures = death_failures
    return char


def _make_ranger(
    name: str = "Theron",
    hp_current: int = 28,
    conditions: list = None,
    death_failures: int = 0,
) -> Character:
    """Build a ranger (martial, no spellcasting focus)."""
    char = Character(
        discord_user_id=101,
        campaign_id="benchmark",
        name=name,
        race_index="elf",
        class_index="ranger",
        level=3,
        abilities=AbilityScores(
            strength=14, dexterity=16, constitution=14,
            intelligence=10, wisdom=14, charisma=10,
        ),
        hp=HitPoints(maximum=28, current=hp_current),
        hit_dice=HitDice(die_type=10, total=3, remaining=3),
        armor_class=15,
        speed=30,
        skill_proficiencies=[
            Skill.PERCEPTION, Skill.STEALTH, Skill.SURVIVAL,
            Skill.NATURE, Skill.ANIMAL_HANDLING,
        ],
        conditions=conditions or [],
    )
    if death_failures > 0:
        char.death_saves.failures = death_failures
    return char


# Named character variants for test cases
CHARACTER_VARIANTS = {
    # Healthy wizard, full resources
    "wizard_full": lambda: _make_wizard(),

    # Wizard with L3 slots depleted (L1 and L2 partially used)
    "wizard_depleted_l3": lambda: _make_wizard(
        slots=((2, 4), (1, 3), (0, 2)),
    ),

    # Wizard with ALL slots depleted
    "wizard_empty": lambda: _make_wizard(
        slots=((0, 4), (0, 3), (0, 2)),
    ),

    # Wizard concentrating on Haste
    "wizard_concentrating": lambda: _make_wizard(concentration="haste"),

    # Wizard at 0 HP (unconscious)
    "wizard_unconscious": lambda: _make_wizard(hp_current=0),

    # Dead wizard (0 HP + 3 death save failures)
    "wizard_dead": lambda: _make_wizard(hp_current=0, death_failures=3),

    # Wizard who is stunned
    "wizard_stunned": lambda: _make_wizard(conditions=[
        CharacterCondition(condition=Condition.STUNNED, source="Mind Flayer"),
    ]),

    # Wizard who is paralyzed
    "wizard_paralyzed": lambda: _make_wizard(conditions=[
        CharacterCondition(condition=Condition.PARALYZED, source="Hold Person"),
    ]),

    # Healthy ranger, full resources
    "ranger_full": lambda: _make_ranger(),

    # Ranger who is frightened (non-blocking, should still act)
    "ranger_frightened": lambda: _make_ranger(conditions=[
        CharacterCondition(condition=Condition.FRIGHTENED, source="Dragon Fear"),
    ]),

    # Ranger who is petrified
    "ranger_petrified": lambda: _make_ranger(conditions=[
        CharacterCondition(condition=Condition.PETRIFIED, source="Basilisk Gaze"),
    ]),
}


# ── Inventory Seeds ──────────────────────────────────────────────────────────

def _make_inventory(character_id: str = "bench-char") -> list[InventoryItem]:
    return [
        InventoryItem(character_id=character_id, item_index="longbow", item_name="Longbow", quantity=1, equipped=True),
        InventoryItem(character_id=character_id, item_index="shortsword", item_name="Shortsword", quantity=1),
        InventoryItem(character_id=character_id, item_index="arrow", item_name="Arrow", quantity=18),
        InventoryItem(character_id=character_id, item_index="potion-of-healing", item_name="Potion of Healing", quantity=2),
        InventoryItem(character_id=character_id, item_index="rations", item_name="Rations", quantity=5),
        InventoryItem(character_id=character_id, item_index="rope-silk", item_name="Silk Rope (50 ft)", quantity=1),
        InventoryItem(character_id=character_id, item_index="thieves-tools", item_name="Thieves' Tools", quantity=1),
    ]


def _make_currency(character_id: str = "bench-char") -> Currency:
    return Currency(character_id=character_id, gold=45, silver=30, copper=80)


def _make_poor_currency(character_id: str = "bench-char") -> Currency:
    return Currency(character_id=character_id, gold=1, silver=5, copper=10)


# ── Test Cases ───────────────────────────────────────────────────────────────
#
# category:
#   TRIAGE   - Tests triage classification accuracy (existing)
#   PGI      - Tests PGI validation catches invalid actions
#   PIPELINE - Tests triage → PGI end-to-end (triage must classify correctly
#              for PGI to validate the right thing)
#
# expected_pgi: "pass" | "hard_fail" | "soft_fail" | None (skip PGI check)
# expected_pgi_code: Specific failure code if expected_pgi is hard/soft fail

TEST_CASES = [
    # ═══════════════════════════════════════════════════════════════════════
    # TRIAGE — Classification accuracy (same as before, Gemma-only)
    # ═══════════════════════════════════════════════════════════════════════
    {
        "action": "I try to pick the lock on the chest.",
        "expected_type": "skill_check",
        "expected_skill": "sleight_of_hand",
        "expected_needs_roll": True,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I search the room for hidden doors or compartments.",
        "expected_type": "skill_check",
        "expected_skill": "investigation",
        "expected_needs_roll": True,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I try to convince the guard to let us pass.",
        "expected_type": "skill_check",
        "expected_skill": "persuasion",
        "expected_needs_roll": True,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I attempt to climb the slippery cliff face.",
        "expected_type": "skill_check",
        "expected_skill": "athletics",
        "expected_needs_roll": True,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I try to sneak past the sleeping guard.",
        "expected_type": "skill_check",
        "expected_skill": "stealth",
        "expected_needs_roll": True,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I intimidate the bandit into dropping his weapon.",
        "expected_type": "skill_check",
        "expected_skill": "intimidation",
        "expected_needs_roll": True,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I try to identify what type of potion this is.",
        "expected_type": "skill_check",
        "expected_skill": "arcana",
        "expected_needs_roll": True,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I listen at the door to hear if anyone is inside.",
        "expected_type": "skill_check",
        "expected_skill": "perception",
        "expected_needs_roll": True,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I try to calm the spooked horse before it bolts.",
        "expected_type": "skill_check",
        "expected_skill": "animal_handling",
        "expected_needs_roll": True,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I attempt to forge the nobleman's signature on the letter.",
        "expected_type": "skill_check",
        "expected_skill": "deception",
        "expected_needs_roll": True,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    # No-roll actions
    {
        "action": "I walk over to the bar and sit down.",
        "expected_type": "movement",
        "expected_needs_roll": False,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I say hello to the innkeeper and ask what's on the menu.",
        "expected_type": "social",
        "expected_needs_roll": False,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I open the unlocked door and walk through.",
        "expected_type": "movement",
        "expected_needs_roll": False,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I take a seat by the fire and warm my hands.",
        "expected_type": "roleplay",
        "expected_needs_roll": False,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I read the sign posted on the wall.",
        "expected_type": "exploration",
        "expected_needs_roll": False,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I draw my sword and announce that I'm ready.",
        "expected_type": "roleplay",
        "expected_needs_roll": False,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I look around the tavern to see who's here.",
        "expected_type": "exploration",
        "expected_needs_roll": False,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I pick up the gold coins from the table.",
        "expected_type": "inventory",
        "expected_needs_roll": False,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    # Attack / spell / purchase
    {
        "action": "I attack the goblin with my longsword.",
        "expected_type": "attack",
        "expected_needs_roll": False,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I shoot an arrow at the wolf.",
        "expected_type": "attack",
        "expected_needs_roll": False,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I cast Detect Magic to scan the room.",
        "expected_type": "cast_spell",
        "expected_needs_roll": False,
        "category": "TRIAGE",
        "variant": "wizard_full",
    },
    {
        "action": "I'd like to buy a healing potion from the shopkeeper.",
        "expected_type": "purchase",
        "expected_needs_roll": False,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    # Edge cases
    {
        "action": "I carefully examine the ancient runes carved into the altar.",
        "expected_type": "skill_check",
        "expected_skill": "arcana",
        "expected_needs_roll": True,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I try to jump across the 10-foot gap.",
        "expected_type": "skill_check",
        "expected_skill": "athletics",
        "expected_needs_roll": True,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I tell the barmaid about my adventures on the road.",
        "expected_type": "social",
        "expected_needs_roll": False,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },
    {
        "action": "I look at the night sky to figure out which direction is north.",
        "expected_type": "exploration",
        "expected_needs_roll": False,
        "category": "TRIAGE",
        "variant": "ranger_full",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # PGI — Validation catches (triage result is mocked, tests PGI only)
    # ═══════════════════════════════════════════════════════════════════════

    # -- P0: Vitality --
    {
        "action": "I cast magic missile at the skeleton.",
        "expected_type": "cast_spell",
        "category": "PGI",
        "variant": "wizard_dead",
        "expected_pgi": "hard_fail",
        "expected_pgi_code": "DEAD",
        "notes": "Dead wizard can't cast — PGI blocks at P0",
    },
    {
        "action": "I try to stand up and fight.",
        "expected_type": "skill_check",
        "category": "PGI",
        "variant": "wizard_unconscious",
        "expected_pgi": "hard_fail",
        "expected_pgi_code": "UNCONSCIOUS_HP",
        "notes": "0 HP wizard can't act — PGI blocks at P0",
    },

    # -- P1: Conditions --
    {
        "action": "I cast fireball at the enemies.",
        "expected_type": "cast_spell",
        "category": "PGI",
        "variant": "wizard_stunned",
        "expected_pgi": "hard_fail",
        "expected_pgi_code": "CONDITION_STUNNED",
        "notes": "Stunned wizard can't take actions",
    },
    {
        "action": "I try to run away.",
        "expected_type": "movement",
        "category": "PGI",
        "variant": "wizard_paralyzed",
        "expected_pgi": "hard_fail",
        "expected_pgi_code": "CONDITION_PARALYZED",
        "notes": "Paralyzed wizard can't take actions or move",
    },
    {
        "action": "I attack the goblin with my sword.",
        "expected_type": "attack",
        "category": "PGI",
        "variant": "ranger_petrified",
        "expected_pgi": "hard_fail",
        "expected_pgi_code": "CONDITION_PETRIFIED",
        "notes": "Petrified ranger can't take actions",
    },
    {
        "action": "I try to sneak past the guard.",
        "expected_type": "skill_check",
        "category": "PGI",
        "variant": "ranger_frightened",
        "expected_pgi": "pass",
        "notes": "Frightened is non-blocking — PGI should pass",
    },

    # -- P6: Spell casting --
    {
        "action": "I cast fireball at the group of orcs.",
        "expected_type": "cast_spell",
        "category": "PGI",
        "variant": "wizard_depleted_l3",
        "expected_pgi": "hard_fail",
        "expected_pgi_code": "NO_SPELL_SLOT",
        "notes": "No L3 slots — PGI blocks, suggests alternatives",
    },
    {
        "action": "I cast magic missile at the goblin.",
        "expected_type": "cast_spell",
        "category": "PGI",
        "variant": "wizard_depleted_l3",
        "expected_pgi": "pass",
        "notes": "L1 slots available — magic missile should pass",
    },
    {
        "action": "I cast fire bolt at the zombie.",
        "expected_type": "cast_spell",
        "category": "PGI",
        "variant": "wizard_empty",
        "expected_pgi": "pass",
        "notes": "Cantrip doesn't need slots — always passes",
    },
    {
        "action": "I cast fireball at them.",
        "expected_type": "cast_spell",
        "category": "PGI",
        "variant": "wizard_empty",
        "expected_pgi": "hard_fail",
        "expected_pgi_code": "NO_SPELL_SLOT",
        "notes": "All slots depleted — fireball blocked",
    },
    {
        "action": "I cast wish to restore my allies.",
        "expected_type": "cast_spell",
        "category": "PGI",
        "variant": "wizard_full",
        "expected_pgi": "hard_fail",
        "expected_pgi_code": "SPELL_NOT_KNOWN",
        "notes": "Wish not in known spells — blocked",
    },
    {
        "action": "I cast fireball at the full wizard resources.",
        "expected_type": "cast_spell",
        "category": "PGI",
        "variant": "wizard_full",
        "expected_pgi": "pass",
        "notes": "Wizard has L3 slots and knows fireball — passes",
    },

    # -- P6: Concentration conflicts --
    {
        "action": "I cast haste on myself.",
        "expected_type": "cast_spell",
        "category": "PGI",
        "variant": "wizard_concentrating",
        "expected_pgi": "soft_fail",
        "expected_pgi_code": "CONCENTRATION_CONFLICT",
        "notes": "Already concentrating on Haste — soft fail, warn player",
    },
    {
        "action": "I cast scorching ray at the enemy.",
        "expected_type": "cast_spell",
        "category": "PGI",
        "variant": "wizard_concentrating",
        "expected_pgi": "pass",
        "notes": "Scorching ray is not concentration — no conflict",
    },

    # ═══════════════════════════════════════════════════════════════════════
    # PIPELINE — End-to-end: triage classifies, then PGI validates
    # These test that triage correctly identifies the action AND PGI catches it.
    # ═══════════════════════════════════════════════════════════════════════
    {
        "action": "I cast fireball at the bandits surrounding us.",
        "expected_type": "cast_spell",
        "expected_needs_roll": False,
        "category": "PIPELINE",
        "variant": "wizard_depleted_l3",
        "expected_pgi": "hard_fail",
        "expected_pgi_code": "NO_SPELL_SLOT",
        "notes": "Triage should classify cast_spell, PGI should catch depleted L3",
    },
    {
        "action": "I cast shield to block the incoming attack.",
        "expected_type": "cast_spell",
        "expected_needs_roll": False,
        "category": "PIPELINE",
        "variant": "wizard_full",
        "expected_pgi": "pass",
        "notes": "Triage classifies spell, PGI passes — shield is L1, slots available",
    },
    {
        "action": "I cast magic missile at the skeleton.",
        "expected_type": "cast_spell",
        "expected_needs_roll": False,
        "category": "PIPELINE",
        "variant": "wizard_stunned",
        "expected_pgi": "hard_fail",
        "expected_pgi_code": "CONDITION_STUNNED",
        "notes": "Triage classifies spell but PGI blocks — stunned",
    },
    {
        "action": "I sneak past the guards and into the castle.",
        "expected_type": "skill_check",
        "expected_skill": "stealth",
        "expected_needs_roll": True,
        "category": "PIPELINE",
        "variant": "ranger_full",
        "expected_pgi": "pass",
        "notes": "Healthy ranger, skill check — both triage and PGI pass",
    },
    {
        "action": "I cast detect magic to find hidden enchantments.",
        "expected_type": "cast_spell",
        "expected_needs_roll": False,
        "category": "PIPELINE",
        "variant": "wizard_concentrating",
        "expected_pgi": "soft_fail",
        "expected_pgi_code": "CONCENTRATION_CONFLICT",
        "notes": "Detect magic is concentration — conflicts with active Haste",
    },
    {
        "action": "I cast fire bolt at the goblin.",
        "expected_type": "cast_spell",
        "expected_needs_roll": False,
        "category": "PIPELINE",
        "variant": "wizard_depleted_l3",
        "expected_pgi": "pass",
        "notes": "Cantrip — slots irrelevant, should pass",
    },
]


# ── Scene Context (injected into triage prompt) ──────────────────────────────

def build_scene_context(character: Character) -> str:
    """Build scene context string with character capabilities baked in."""
    # Spell info
    spell_info = ""
    if character.known_spells:
        cantrips = [s.replace("-", " ").title() for s in character.known_spells
                    if s in ("fire-bolt", "mage-hand", "prestidigitation")]
        leveled = [s.replace("-", " ").title() for s in character.known_spells
                   if s not in ("fire-bolt", "mage-hand", "prestidigitation")]

        slot_parts = []
        for level in range(1, 10):
            current, maximum = character.spell_slots.get_slots(level)
            if maximum > 0:
                slot_parts.append(f"L{level}: {current}/{maximum}")

        spell_info = f"""
  Cantrips: {', '.join(cantrips) if cantrips else 'None'}
  Known Spells: {', '.join(leveled) if leveled else 'None'}
  Spell Slots: {', '.join(slot_parts) if slot_parts else 'None'}"""
        if character.is_concentrating:
            conc_name = (character.concentration_spell_id or "").replace("-", " ").title()
            spell_info += f"\n  Concentrating on: {conc_name}"

    # Conditions
    condition_info = ""
    if character.conditions:
        cond_names = [c.condition.value.title() for c in character.conditions]
        condition_info = f"\n  Conditions: {', '.join(cond_names)}"

    # Skills
    skills = ""
    if character.skill_proficiencies:
        skill_names = [s.value.replace("_", " ").title() if hasattr(s, 'value') else str(s)
                       for s in character.skill_proficiencies]
        skills = f"\n  Proficiencies: {', '.join(skill_names)}"

    return f"""## Current Scene
You are in the Rusty Flagon tavern in the town of Millhaven. The common room
is warm and busy - a fire crackles in the hearth, several patrons drink at
wooden tables, and a barmaid moves between them. Behind the bar, the innkeeper
Gorm polishes mugs. A locked door leads to the back storeroom. Outside, the
cobblestone streets of Millhaven stretch toward the market square. To the north,
the Whisperwood Forest looms dark and foreboding.

## Entities Present
- Gorm (innkeeper, human, friendly) - behind the bar
- Barmaid (human, neutral) - serving tables
- 3 patrons (commoners) - drinking at tables
- 1 hooded figure (unknown) - sitting alone in the corner

## Party
- {character.name} ({character.race_index.title()} {character.class_index.title()}, Level {character.level}) - HP {character.hp.current}/{character.hp.maximum}, AC {character.armor_class}{skills}{spell_info}{condition_info}
  Equipment: Longbow, Shortsword, Explorer's Pack, Thieves' Tools

## Player Action
"""


# ── Scoring ──────────────────────────────────────────────────────────────────

@dataclass
class CaseResult:
    case_index: int
    action: str
    category: str
    variant: str
    expected_type: str
    expected_skill: Optional[str] = None
    expected_needs_roll: bool = False
    notes: str = ""

    # Triage results
    actual_type: str = ""
    actual_skill: Optional[str] = None
    actual_needs_roll: bool = False
    actual_dc: Optional[int] = None
    actual_reasoning: str = ""

    # Triage scoring
    type_correct: bool = False
    roll_correct: bool = False
    skill_correct: bool = False
    dc_reasonable: bool = True
    json_valid: bool = True
    parse_error: str = ""

    # PGI results
    expected_pgi: Optional[str] = None  # "pass", "hard_fail", "soft_fail"
    expected_pgi_code: Optional[str] = None
    actual_pgi: Optional[str] = None
    actual_pgi_codes: list = field(default_factory=list)
    pgi_correct: bool = True
    pgi_code_correct: bool = True
    pgi_feedback: str = ""

    elapsed: float = 0.0


@dataclass
class BenchmarkResult:
    model: str
    cases: list[CaseResult] = field(default_factory=list)
    total_time: float = 0.0
    json_failures: int = 0

    @property
    def triage_cases(self) -> list[CaseResult]:
        return [c for c in self.cases if c.category in ("TRIAGE", "PIPELINE")]

    @property
    def pgi_cases(self) -> list[CaseResult]:
        return [c for c in self.cases if c.expected_pgi is not None]

    @property
    def type_accuracy(self) -> float:
        cases = self.triage_cases
        return sum(1 for c in cases if c.type_correct) / len(cases) if cases else 0

    @property
    def roll_accuracy(self) -> float:
        cases = self.triage_cases
        return sum(1 for c in cases if c.roll_correct) / len(cases) if cases else 0

    @property
    def skill_accuracy(self) -> float:
        cases = [c for c in self.triage_cases if c.expected_skill]
        return sum(1 for c in cases if c.skill_correct) / len(cases) if cases else 0

    @property
    def pgi_accuracy(self) -> float:
        cases = self.pgi_cases
        return sum(1 for c in cases if c.pgi_correct) / len(cases) if cases else 0

    @property
    def pgi_code_accuracy(self) -> float:
        cases = [c for c in self.pgi_cases if c.expected_pgi_code]
        return sum(1 for c in cases if c.pgi_code_correct) / len(cases) if cases else 0


# ── Runner ───────────────────────────────────────────────────────────────────

async def run_benchmark(model: str, category_filter: Optional[str] = None) -> BenchmarkResult:
    """Run test cases through triage + PGI on Gemma."""
    from dnd_bot.llm.client import OllamaClient
    from dnd_bot.llm.orchestrator import DMOrchestrator
    from dnd_bot.llm.brains.base import BrainContext

    client = OllamaClient(model=model, num_ctx=8000)

    orchestrator = DMOrchestrator.__new__(DMOrchestrator)
    orchestrator.client = client
    orchestrator._current_session = None
    orchestrator._scene_registry = None

    # Filter cases
    cases = TEST_CASES
    if category_filter:
        cases = [c for c in cases if c["category"] == category_filter.upper()]

    result = BenchmarkResult(model=model)
    t0_total = time.monotonic()

    inventory = _make_inventory()
    currency = _make_currency()

    for i, case in enumerate(cases):
        variant_name = case.get("variant", "ranger_full")
        character = CHARACTER_VARIANTS[variant_name]()
        player_name = character.name
        scene = build_scene_context(character)

        context = BrainContext(
            player_action=case["action"],
            player_name=player_name,
            current_scene=scene,
        )

        t0 = time.monotonic()
        case_result = CaseResult(
            case_index=i,
            action=case["action"],
            category=case["category"],
            variant=variant_name,
            expected_type=case["expected_type"],
            expected_skill=case.get("expected_skill"),
            expected_needs_roll=case.get("expected_needs_roll", False),
            expected_pgi=case.get("expected_pgi"),
            expected_pgi_code=case.get("expected_pgi_code"),
            notes=case.get("notes", ""),
        )

        # ── Step 1: Triage ──
        triage_ok = False
        triage = None
        try:
            triage = await orchestrator._triage_action(
                action=case["action"],
                player_name=player_name,
                context=context,
            )
            case_result.actual_type = triage.action_type
            case_result.actual_skill = triage.skill
            case_result.actual_needs_roll = triage.needs_roll
            case_result.actual_dc = triage.dc
            case_result.actual_reasoning = triage.reasoning

            # Score triage
            expected_types = [case["expected_type"]]
            NO_ROLL_TYPES = {"social", "roleplay", "exploration", "movement", "inventory"}
            if case["expected_type"] in NO_ROLL_TYPES:
                expected_types = list(NO_ROLL_TYPES)
            if case["expected_type"] == "skill_check":
                expected_types = ["skill_check", "ability_check"]

            case_result.type_correct = triage.action_type in expected_types
            case_result.roll_correct = triage.needs_roll == case.get("expected_needs_roll", False)

            if case.get("expected_skill") and triage.skill:
                actual = triage.skill.lower().replace(" ", "_")
                expected = case["expected_skill"].lower().replace(" ", "_")
                case_result.skill_correct = (
                    actual == expected or expected in actual or actual in expected
                )

            if triage.dc is not None:
                case_result.dc_reasonable = 5 <= triage.dc <= 20

            triage_ok = True

        except Exception as e:
            case_result.json_valid = False
            case_result.parse_error = str(e)[:200]
            result.json_failures += 1

        # ── Step 2: PGI Validation ──
        if case.get("expected_pgi") is not None:
            # For PGI-only cases, use the expected type if triage failed
            action_type = triage.action_type if triage else case["expected_type"]

            pgi = await validate_action(
                action_type=action_type,
                character=character,
                action_text=case["action"],
                items=inventory,
                currency=currency,
                resources_consumed=triage.resources_consumed if triage else None,
                item_name=triage.item_name if triage else None,
                cost_gold=float(triage.item_cost) if triage and triage.item_cost else 0,
            )

            # Determine actual PGI outcome
            if pgi.has_hard_fail:
                case_result.actual_pgi = "hard_fail"
            elif pgi.has_soft_fail:
                case_result.actual_pgi = "soft_fail"
            else:
                case_result.actual_pgi = "pass"

            case_result.actual_pgi_codes = [f.code for f in pgi.failures]
            case_result.pgi_feedback = pgi.player_feedback()

            # Score PGI
            case_result.pgi_correct = case_result.actual_pgi == case["expected_pgi"]

            if case.get("expected_pgi_code"):
                case_result.pgi_code_correct = (
                    case["expected_pgi_code"] in case_result.actual_pgi_codes
                )
            else:
                case_result.pgi_code_correct = True

        case_result.elapsed = time.monotonic() - t0
        result.cases.append(case_result)

        # Progress indicator
        triage_status = "OK" if case_result.type_correct else "MISS"
        pgi_status = ""
        if case.get("expected_pgi"):
            pgi_status = " PGI:OK" if case_result.pgi_correct else " PGI:MISS"
        status = f"T:{triage_status:4s}{pgi_status}"

        print(
            f"  [{i+1:2d}/{len(cases)}] {status:14s} | "
            f"{case['action'][:50]:50s} | "
            f"type={case_result.actual_type:15s} | "
            f"{case_result.elapsed:.1f}s"
        )

    result.total_time = time.monotonic() - t0_total
    return result


def print_report(result: BenchmarkResult):
    """Print benchmark report."""
    print(f"\n{'=' * 80}")
    print(f"BRAIN + PGI PIPELINE BENCHMARK — {result.model}")
    print(f"{'=' * 80}")

    # Triage accuracy
    triage = result.triage_cases
    if triage:
        print(f"\n── Triage Accuracy ──")
        print(f"  Action type:    {result.type_accuracy:.0%} ({sum(1 for c in triage if c.type_correct)}/{len(triage)})")
        print(f"  Roll decision:  {result.roll_accuracy:.0%} ({sum(1 for c in triage if c.roll_correct)}/{len(triage)})")
        skill_cases = [c for c in triage if c.expected_skill]
        if skill_cases:
            print(f"  Skill selection:{result.skill_accuracy:.0%} ({sum(1 for c in skill_cases if c.skill_correct)}/{len(skill_cases)})")
        print(f"  JSON failures:  {result.json_failures}")

    # PGI accuracy
    pgi = result.pgi_cases
    if pgi:
        print(f"\n── PGI Validation ──")
        print(f"  Outcome accuracy: {result.pgi_accuracy:.0%} ({sum(1 for c in pgi if c.pgi_correct)}/{len(pgi)})")
        code_cases = [c for c in pgi if c.expected_pgi_code]
        if code_cases:
            print(f"  Code accuracy:    {result.pgi_code_accuracy:.0%} ({sum(1 for c in code_cases if c.pgi_code_correct)}/{len(code_cases)})")

        # Breakdown by expected outcome
        for outcome in ("pass", "hard_fail", "soft_fail"):
            subset = [c for c in pgi if c.expected_pgi == outcome]
            if subset:
                correct = sum(1 for c in subset if c.pgi_correct)
                print(f"    {outcome:10s}: {correct}/{len(subset)} correct")

    # Timing
    print(f"\n── Timing ──")
    print(f"  Total: {result.total_time:.1f}s | Avg: {result.total_time/len(result.cases):.1f}s/case")

    # Misses detail
    triage_misses = [c for c in triage if not c.type_correct or not c.roll_correct]
    if triage_misses:
        print(f"\n── Triage Misses ──")
        for c in triage_misses:
            issues = []
            if not c.type_correct:
                issues.append(f"type: want={c.expected_type} got={c.actual_type}")
            if not c.roll_correct:
                issues.append(f"roll: want={c.expected_needs_roll} got={c.actual_needs_roll}")
            if c.expected_skill and not c.skill_correct:
                issues.append(f"skill: want={c.expected_skill} got={c.actual_skill}")
            print(f"  [{c.case_index+1:2d}] {c.action[:60]}")
            print(f"       {' | '.join(issues)}")

    pgi_misses = [c for c in pgi if not c.pgi_correct or not c.pgi_code_correct]
    if pgi_misses:
        print(f"\n── PGI Misses ──")
        for c in pgi_misses:
            print(f"  [{c.case_index+1:2d}] {c.action[:60]}")
            print(f"       want={c.expected_pgi}({c.expected_pgi_code}) "
                  f"got={c.actual_pgi}({c.actual_pgi_codes})")
            print(f"       variant={c.variant} | {c.notes}")
            if c.pgi_feedback:
                print(f"       feedback: {c.pgi_feedback[:120]}")

    # Save results
    output = {
        "model": result.model,
        "test_cases": len(result.cases),
        "triage_accuracy": result.type_accuracy,
        "roll_accuracy": result.roll_accuracy,
        "skill_accuracy": result.skill_accuracy,
        "pgi_accuracy": result.pgi_accuracy,
        "pgi_code_accuracy": result.pgi_code_accuracy,
        "total_time": result.total_time,
        "cases": [asdict(c) for c in result.cases],
    }

    os.makedirs("data/benchmark_logs", exist_ok=True)
    outpath = "data/benchmark_logs/brain_benchmark_latest.json"
    with open(outpath, "w") as f:
        json.dump(output, f, indent=2, default=str)
    print(f"\nResults saved to {outpath}")


async def main():
    parser = argparse.ArgumentParser(description="Brain + PGI Pipeline Benchmark")
    parser.add_argument(
        "--model", default="gemma4:26b",
        help="Model to benchmark (default: gemma4:26b)",
    )
    parser.add_argument(
        "--category", default=None, choices=["TRIAGE", "PGI", "PIPELINE"],
        help="Run only a specific category of test cases",
    )
    args = parser.parse_args()

    cases = TEST_CASES
    if args.category:
        cases = [c for c in cases if c["category"] == args.category.upper()]

    variants_used = {c.get("variant", "ranger_full") for c in cases}

    print(f"Brain + PGI Pipeline Benchmark")
    print(f"Model: {args.model}")
    print(f"Cases: {len(cases)} ({args.category or 'ALL'})")
    print(f"Variants: {', '.join(sorted(variants_used))}")
    print()

    result = await run_benchmark(args.model, args.category)
    print_report(result)


if __name__ == "__main__":
    asyncio.run(main())
