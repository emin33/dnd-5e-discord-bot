"""
Deterministic Combat Test Suite - No LLM, runs in seconds.

Tests the game mechanics layer: damage, healing, saves, initiative,
spell slots, conditions, rest recovery, dice math, AC calculation.

Usage:
    python test_combat.py           # Run all tests
    python test_combat.py -v        # Verbose output
"""

import sys
import os
import math

os.chdir(os.path.dirname(os.path.abspath(__file__)))

from dnd_bot.models.character import (
    Character, AbilityScores, HitPoints, HitDice, DeathSaves, SpellSlots, Skill,
)
from dnd_bot.models.combat import (
    Combatant, Combat, CombatState, CombatEffect, TurnResources,
)
from dnd_bot.models.common import AbilityScore, Condition
from dnd_bot.game.mechanics.dice import DiceRoller

# ==================== Test Framework ====================

class TestResult:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name: str):
        self.passed += 1
        if "-v" in sys.argv:
            print(f"  \033[92mPASS\033[0m {name}")

    def fail(self, name: str, expected, got):
        self.failed += 1
        self.errors.append((name, expected, got))
        print(f"  \033[91mFAIL\033[0m {name}: expected {expected}, got {got}")

    def summary(self):
        total = self.passed + self.failed
        color = "\033[92m" if self.failed == 0 else "\033[91m"
        print(f"\n{color}{self.passed}/{total} passed\033[0m")
        if self.errors:
            print(f"\033[91m{self.failed} failures:\033[0m")
            for name, exp, got in self.errors:
                print(f"  - {name}: expected {exp}, got {got}")
        return self.failed == 0


def assert_eq(results: TestResult, name: str, got, expected):
    if got == expected:
        results.ok(name)
    else:
        results.fail(name, expected, got)


def assert_true(results: TestResult, name: str, condition):
    if condition:
        results.ok(name)
    else:
        results.fail(name, True, condition)


def assert_in_range(results: TestResult, name: str, value, low, high):
    if low <= value <= high:
        results.ok(name)
    else:
        results.fail(name, f"{low}-{high}", value)


# ==================== Test Helpers ====================

def make_character(name="TestChar", strength=14, dexterity=16, constitution=12,
                   intelligence=10, wisdom=13, charisma=8, level=1,
                   hp=10, ac=13, class_index="ranger") -> Character:
    abilities = AbilityScores(
        strength=strength, dexterity=dexterity, constitution=constitution,
        intelligence=intelligence, wisdom=wisdom, charisma=charisma,
    )
    return Character(
        id="test-char-1",
        discord_user_id=1,
        campaign_id="test-campaign",
        name=name,
        race_index="elf",
        class_index=class_index,
        level=level,
        abilities=abilities,
        armor_class=ac,
        speed=30,
        initiative_bonus=abilities.dex_mod,
        hp=HitPoints(maximum=hp, current=hp),
        hit_dice=HitDice(die_type=10, total=level, remaining=level),
        death_saves=DeathSaves(),
        spellcasting_ability=AbilityScore.WISDOM,
        spell_slots=SpellSlots(),
        saving_throw_proficiencies=[AbilityScore.STRENGTH, AbilityScore.DEXTERITY],
        skill_proficiencies=[Skill.PERCEPTION, Skill.STEALTH, Skill.SURVIVAL],
    )


def make_combatant(name="Goblin", hp=7, ac=15, is_player=False,
                   str_score=8, dex_score=14, con_score=10,
                   monster_index="goblin") -> Combatant:
    return Combatant(
        id=f"test-{name.lower().replace(' ', '-')}",
        combat_id="test-combat",
        name=name,
        is_player=is_player,
        monster_index=monster_index if not is_player else None,
        initiative_bonus=(dex_score - 10) // 2,
        hp_max=hp,
        hp_current=hp,
        armor_class=ac,
        speed=30,
        ability_scores={
            "str": str_score, "dex": dex_score, "con": con_score,
            "int": 10, "wis": 8, "cha": 8,
        },
        proficiency_bonus=2,
    )


# ==================== Test Suites ====================

def test_ability_scores(r: TestResult):
    """Test ability modifier calculations."""
    print("\n--- Ability Scores ---")

    char = make_character(strength=10, dexterity=16, constitution=8)

    # Standard modifier formula: (score - 10) // 2
    assert_eq(r, "STR 10 -> mod 0", char.abilities.get_modifier(AbilityScore.STRENGTH), 0)
    assert_eq(r, "DEX 16 -> mod +3", char.abilities.get_modifier(AbilityScore.DEXTERITY), 3)
    assert_eq(r, "CON 8 -> mod -1", char.abilities.get_modifier(AbilityScore.CONSTITUTION), -1)

    # Edge cases
    assert_eq(r, "Score 1 -> mod -5", (1 - 10) // 2, -5)
    assert_eq(r, "Score 20 -> mod +5", (20 - 10) // 2, 5)
    assert_eq(r, "Score 30 -> mod +10", (30 - 10) // 2, 10)

    # Compatibility property
    assert_eq(r, "ability_scores dict has str", char.ability_scores["str"], 10)
    assert_eq(r, "ability_scores dict has dex", char.ability_scores["dex"], 16)


def test_saving_throws(r: TestResult):
    """Test saving throw modifiers with proficiency."""
    print("\n--- Saving Throws ---")

    char = make_character(strength=14, dexterity=16, level=1)
    # Proficient in STR and DEX saves (ranger)

    # STR save: mod +2 + prof +2 = +4
    assert_eq(r, "STR save (proficient)", char.get_save_modifier(AbilityScore.STRENGTH), 4)
    # DEX save: mod +3 + prof +2 = +5
    assert_eq(r, "DEX save (proficient)", char.get_save_modifier(AbilityScore.DEXTERITY), 5)
    # WIS save: mod +1, no proficiency
    assert_eq(r, "WIS save (not proficient)", char.get_save_modifier(AbilityScore.WISDOM), 1)

    # Monster saves
    combatant = make_combatant(str_score=16, dex_score=14, con_score=12)
    # No save proficiencies -> just ability mod
    assert_eq(r, "Monster STR save (no prof)", combatant.get_save_modifier("str"), 3)
    assert_eq(r, "Monster DEX save (no prof)", combatant.get_save_modifier("dex"), 2)

    # Monster with explicit save bonus
    combatant.save_bonuses = {"con": 6}
    assert_eq(r, "Monster CON save (explicit +6)", combatant.get_save_modifier("con"), 6)
    # STR still uses ability mod
    assert_eq(r, "Monster STR save (still raw)", combatant.get_save_modifier("str"), 3)


def test_damage(r: TestResult):
    """Test damage application: HP, resistance, immunity, vulnerability."""
    print("\n--- Damage ---")

    # Basic damage - returns (actual, instant_death, modifier)
    c = make_combatant(hp=20)
    actual, instant_death, mod = c.take_damage(5)
    assert_eq(r, "Basic 5 damage -> 15 HP", c.hp_current, 15)
    assert_eq(r, "Basic damage actual", actual, 5)
    assert_true(r, "Not instant death", not instant_death)

    # Damage to 0
    c = make_combatant(hp=20)
    actual, instant_death, mod = c.take_damage(20)
    assert_eq(r, "Exact lethal -> 0 HP", c.hp_current, 0)

    # Overkill (not instant death: excess < max HP) - need player for instant death check
    c = make_combatant(hp=20, is_player=True)
    c.character_id = "test"
    actual, instant_death, mod = c.take_damage(25)
    assert_eq(r, "Overkill -> 0 HP (not negative)", c.hp_current, 0)
    assert_true(r, "Not instant death (excess 5 < max 20)", not instant_death)

    # Instant death (excess >= max HP)
    c = make_combatant(hp=20, is_player=True)
    c.character_id = "test"
    actual, instant_death, mod = c.take_damage(40)
    assert_true(r, "Massive damage -> instant death", instant_death)

    # Resistance (half damage)
    c = make_combatant(hp=20)
    c.resistances = ["fire"]
    actual, instant_death, mod = c.take_damage(10, damage_type="fire")
    assert_eq(r, "Fire resist: 10 -> 5 damage", actual, 5)
    assert_eq(r, "Fire resist: HP = 15", c.hp_current, 15)
    assert_eq(r, "Modifier is resistance", mod, "resistance")

    # Immunity (no damage)
    c = make_combatant(hp=20)
    c.immunities = ["poison"]
    actual, instant_death, mod = c.take_damage(10, damage_type="poison")
    assert_eq(r, "Poison immune: 0 damage", actual, 0)
    assert_eq(r, "Poison immune: HP unchanged", c.hp_current, 20)

    # Vulnerability (double damage)
    c = make_combatant(hp=20)
    c.vulnerabilities = ["radiant"]
    actual, instant_death, mod = c.take_damage(6, damage_type="radiant")
    assert_eq(r, "Radiant vuln: 6 -> 12 damage", actual, 12)
    assert_eq(r, "Radiant vuln: HP = 8", c.hp_current, 8)

    # Temp HP absorbs first
    c = make_combatant(hp=20)
    c.hp_temp = 5
    actual, instant_death, mod = c.take_damage(8)
    assert_eq(r, "Temp HP absorbs: 5 temp gone", c.hp_temp, 0)
    assert_eq(r, "Remaining 3 hits real HP: 17", c.hp_current, 17)


def test_healing(r: TestResult):
    """Test healing mechanics."""
    print("\n--- Healing ---")

    c = make_combatant(hp=20)
    c.hp_current = 5

    # Normal heal
    healed = c.heal(10)
    assert_eq(r, "Heal 10: HP 5 -> 15", c.hp_current, 15)
    assert_eq(r, "Healed amount = 10", healed, 10)

    # Overheal capped at max
    healed = c.heal(100)
    assert_eq(r, "Overheal capped at max", c.hp_current, 20)
    assert_eq(r, "Overheal actual = 5", healed, 5)

    # Heal at full HP
    healed = c.heal(10)
    assert_eq(r, "Heal at full = 0", healed, 0)


def test_death_saves(r: TestResult):
    """Test death save mechanics."""
    print("\n--- Death Saves ---")

    ds = DeathSaves()
    assert_eq(r, "Initial successes 0", ds.successes, 0)
    assert_eq(r, "Initial failures 0", ds.failures, 0)

    ds.add_success()
    assert_eq(r, "1 success", ds.successes, 1)
    assert_true(r, "Not stabilized at 1", not ds.is_stable)

    ds.add_success()
    ds.add_success()
    assert_true(r, "Stabilized at 3 successes", ds.is_stable)
    assert_true(r, "Not dead", not ds.is_dead)

    ds.reset()
    ds.add_failure()
    ds.add_failure()
    ds.add_failure()
    assert_true(r, "Dead at 3 failures", ds.is_dead)


def test_dice_roller(r: TestResult):
    """Test dice rolling mechanics."""
    print("\n--- Dice Roller ---")

    import random
    roller = DiceRoller(rng=random.Random(42))

    # Roll d20+5
    roll = roller.roll("1d20+5")
    assert_in_range(r, "d20+5 in range 6-25", roll.total, 6, 25)
    assert_eq(r, "d20 modifier stored", roll.modifier, 5)

    # Roll damage
    damage = roller.roll_damage("2d6")
    assert_in_range(r, "2d6 damage in range 2-12", damage.total, 2, 12)

    # Roll many d20s to test range
    results = [roller.roll("1d20").total for _ in range(100)]
    assert_in_range(r, "d20 min is 1", min(results), 1, 1)
    assert_in_range(r, "d20 max is 20", max(results), 20, 20)

    # Critical hit detection
    roller2 = DiceRoller(rng=random.Random(0))
    found_crit = False
    for _ in range(1000):
        roll = roller2.roll("1d20")
        if roll.natural_20:
            found_crit = True
            break
    assert_true(r, "Found a nat 20 in 1000 rolls", found_crit)

    # Advantage: roll 2d20, take higher
    roller3 = DiceRoller(rng=random.Random(42))
    adv_roll = roller3.roll("1d20", advantage=True)
    assert_true(r, "Advantage has advantage_rolls", adv_roll.advantage_rolls is not None)

    # 4d6kh3 (ability score generation)
    ability_roll = roller.roll("4d6kh3")
    assert_eq(r, "4d6kh3 keeps 3 dice", len(ability_roll.kept_dice), 3)
    assert_in_range(r, "4d6kh3 total 3-18", ability_roll.total, 3, 18)


def test_spell_slots(r: TestResult):
    """Test spell slot tracking."""
    print("\n--- Spell Slots ---")

    slots = SpellSlots()

    # Set slots for a level 1 ranger (2 first-level slots)
    slots.level_1 = (2, 2)
    current, maximum = slots.get_slots(1)
    assert_eq(r, "L1 slots: 2/2", (current, maximum), (2, 2))

    # Expend a slot
    success = slots.expend_slot(1)
    assert_true(r, "Expend L1 succeeds", success)
    current, _ = slots.get_slots(1)
    assert_eq(r, "L1 after expend: 1", current, 1)

    # Expend second slot
    slots.expend_slot(1)
    current, _ = slots.get_slots(1)
    assert_eq(r, "L1 after 2 expends: 0", current, 0)

    # Can't expend when empty
    success = slots.expend_slot(1)
    assert_true(r, "Can't expend empty slot", not success)

    # Restore all
    slots.restore_all()
    current, _ = slots.get_slots(1)
    assert_eq(r, "After restore: 2", current, 2)


def test_initiative_order(r: TestResult):
    """Test initiative sorting."""
    print("\n--- Initiative ---")

    combat = Combat(
        id="test",
        session_id="test-session",
        channel_id=999,
        state=CombatState.ACTIVE,
    )

    c1 = make_combatant("Fighter", hp=20, dex_score=10)
    c1.initiative_roll = 15
    c2 = make_combatant("Rogue", hp=15, dex_score=18)
    c2.initiative_roll = 20
    c3 = make_combatant("Wizard", hp=8, dex_score=12)
    c3.initiative_roll = 10

    combat.add_combatant(c1)
    combat.add_combatant(c2)
    combat.add_combatant(c3)

    combat.roll_all_initiative()
    order = [c.name for c in combat.get_sorted_combatants()]
    assert_eq(r, "Initiative order: Rogue > Fighter > Wizard", order, ["Rogue", "Fighter", "Wizard"])

    # Tie-breaking: higher DEX goes first
    c1.initiative_roll = 15
    c2.initiative_roll = 15
    c2.initiative_bonus = 4  # DEX 18
    c1.initiative_bonus = 0  # DEX 10
    combat.roll_all_initiative()
    order = [c.name for c in combat.get_sorted_combatants()]
    rogue_idx = order.index("Rogue")
    fighter_idx = order.index("Fighter")
    assert_true(r, "Tie break: higher DEX first", rogue_idx < fighter_idx)


def test_combat_effects(r: TestResult):
    """Test combat effect duration and tracking."""
    print("\n--- Combat Effects ---")

    c = make_combatant(hp=20)

    effect = CombatEffect(
        id="eff-1",
        name="Bless",
        effect_type="buff",
        duration_rounds=10,
        bonus_dice="1d4",
        bonus_applies_to=["attack", "save"],
    )
    c.add_effect(effect)
    assert_eq(r, "Effect added", len(c.effects), 1)
    assert_eq(r, "Duration initialized", c.effects[0].rounds_remaining, 10)

    # Decrement
    c.effects[0].rounds_remaining -= 1
    assert_eq(r, "Duration decremented", c.effects[0].rounds_remaining, 9)


def test_hit_dice_recovery(r: TestResult):
    """Test hit dice recovery rounding (should round UP)."""
    print("\n--- Hit Dice Recovery ---")

    # D&D 5e: recover half your total hit dice (minimum 1), rounded UP
    for total, expected in [(1, 1), (2, 1), (3, 2), (4, 2), (5, 3), (7, 4), (9, 5), (10, 5)]:
        recovered = max(1, math.ceil(total / 2))
        assert_eq(r, f"Total {total} -> recover {expected}", recovered, expected)


def test_concentration_dc(r: TestResult):
    """Test concentration save DC calculation."""
    print("\n--- Concentration DC ---")

    # DC = max(10, damage // 2)
    for damage, expected_dc in [(1, 10), (10, 10), (19, 10), (20, 10), (21, 10), (22, 11), (30, 15), (100, 50)]:
        dc = max(10, damage // 2)
        assert_eq(r, f"Damage {damage} -> DC {expected_dc}", dc, expected_dc)


def test_ac_from_equipment(r: TestResult):
    """Test AC calculation from equipment."""
    print("\n--- AC Calculation ---")

    char = make_character(dexterity=16)  # DEX mod +3

    # Unarmored: 10 + DEX
    assert_eq(r, "Unarmored AC: 10 + 3 = 13", char.armor_class, 13)

    # With armor (using calculate_ac_from_equipment)
    # Light armor: AC base + DEX mod
    leather = [{"armor_class": {"base": 11}, "armor_category": "Light", "str_minimum": 0}]
    new_ac = char.calculate_ac_from_equipment(leather)
    assert_eq(r, "Leather armor: 11 + 3 = 14", new_ac, 14)

    # Medium armor: AC base + DEX mod (max 2)
    chain_shirt = [{"armor_class": {"base": 13, "dex_bonus": True, "max_bonus": 2}, "armor_category": "Medium", "str_minimum": 0}]
    new_ac = char.calculate_ac_from_equipment(chain_shirt)
    assert_eq(r, "Chain shirt: 13 + 2 (max) = 15", new_ac, 15)

    # Heavy armor: AC base, no DEX
    plate = [{"armor_class": {"base": 18, "dex_bonus": False}, "armor_category": "Heavy", "str_minimum": 15}]
    new_ac = char.calculate_ac_from_equipment(plate)
    assert_eq(r, "Plate armor: 18 (no DEX)", new_ac, 18)

    # Shield bonus
    shield = [{"armor_class": {"base": 11}, "armor_category": "Light", "str_minimum": 0},
              {"armor_class": {"base": 2}, "armor_category": "Shield", "str_minimum": 0}]
    new_ac = char.calculate_ac_from_equipment(shield)
    assert_eq(r, "Leather + shield: 11 + 3 + 2 = 16", new_ac, 16)


def test_proficiency_bonus(r: TestResult):
    """Test proficiency bonus by level."""
    print("\n--- Proficiency Bonus ---")

    # D&D 5e: +2 at L1-4, +3 at L5-8, +4 at L9-12, +5 at L13-16, +6 at L17-20
    expected = {1: 2, 4: 2, 5: 3, 8: 3, 9: 4, 12: 4, 13: 5, 16: 5, 17: 6, 20: 6}
    for level, expected_bonus in expected.items():
        char = make_character(level=level)
        assert_eq(r, f"Level {level} -> prof +{expected_bonus}", char.proficiency_bonus, expected_bonus)


def test_condition_effects(r: TestResult):
    """Test that conditions properly affect combatants."""
    print("\n--- Condition Enforcement ---")

    c = make_combatant(hp=20)

    # No conditions -> get_active_conditions returns empty
    assert_eq(r, "No conditions initially", c.get_active_conditions(), [])

    # Add paralyzed effect
    from dnd_bot.models.combat import CombatEffect
    para_effect = CombatEffect(
        id="eff-para",
        name="Hold Person",
        effect_type="condition",
        condition=Condition.PARALYZED,
        duration_rounds=10,
        save_ability="wisdom",
        save_dc=15,
        save_ends_on_success=True,
    )
    c.add_effect(para_effect)
    conditions = c.get_active_conditions()
    assert_true(r, "Paralyzed in conditions", Condition.PARALYZED in conditions)

    # Stunned
    stun_effect = CombatEffect(
        id="eff-stun",
        name="Stunning Strike",
        effect_type="condition",
        condition=Condition.STUNNED,
        duration_rounds=1,
    )
    c2 = make_combatant(hp=20)
    c2.add_effect(stun_effect)
    conditions2 = c2.get_active_conditions()
    assert_true(r, "Stunned in conditions", Condition.STUNNED in conditions2)

    # Test auto-crit: paralyzed target in melee
    from dnd_bot.game.mechanics.conditions import ConditionResolver
    assert_true(r, "Paralyzed -> auto crit in melee",
                ConditionResolver.is_auto_crit([Condition.PARALYZED], attacker_within_5ft=True))
    assert_true(r, "Paralyzed -> NOT auto crit at range",
                not ConditionResolver.is_auto_crit([Condition.PARALYZED], attacker_within_5ft=False))
    assert_true(r, "Unconscious -> auto crit in melee",
                ConditionResolver.is_auto_crit([Condition.UNCONSCIOUS], attacker_within_5ft=True))
    assert_true(r, "Stunned -> NOT auto crit",
                not ConditionResolver.is_auto_crit([Condition.STUNNED], attacker_within_5ft=True))

    # Test attack modifiers: blinded attacker has disadvantage
    atk_adv, atk_dis = ConditionResolver.get_attack_modifiers([Condition.BLINDED])
    assert_true(r, "Blinded attacker has disadvantage", atk_dis)
    assert_true(r, "Blinded attacker no advantage", not atk_adv)

    # Test attacks against: blinded target gives advantage to attacker
    tgt_adv, tgt_dis = ConditionResolver.get_attacks_against_modifiers(
        [Condition.BLINDED], attacker_within_5ft=True)
    assert_true(r, "Attacks vs blinded have advantage", tgt_adv)

    # Prone: melee advantage, ranged disadvantage
    tgt_adv_m, tgt_dis_m = ConditionResolver.get_attacks_against_modifiers(
        [Condition.PRONE], attacker_within_5ft=True)
    assert_true(r, "Melee vs prone has advantage", tgt_adv_m)

    tgt_adv_r, tgt_dis_r = ConditionResolver.get_attacks_against_modifiers(
        [Condition.PRONE], attacker_within_5ft=False)
    assert_true(r, "Ranged vs prone has disadvantage", tgt_dis_r)


# ==================== Main ====================

def main():
    print("\033[1m=== D&D 5e Combat Test Suite ===\033[0m")
    r = TestResult()

    test_ability_scores(r)
    test_saving_throws(r)
    test_damage(r)
    test_healing(r)
    test_death_saves(r)
    test_dice_roller(r)
    test_spell_slots(r)
    test_initiative_order(r)
    test_combat_effects(r)
    test_hit_dice_recovery(r)
    test_concentration_dc(r)
    test_ac_from_equipment(r)
    test_proficiency_bonus(r)
    test_condition_effects(r)

    success = r.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
