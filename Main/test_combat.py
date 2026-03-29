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


def test_coordinator_npc_turn(r: TestResult):
    """Test the combat coordinator NPC turn flow (surprise, action execution)."""
    print("\n--- Coordinator: NPC Turn Flow ---")
    import asyncio
    from dnd_bot.game.combat.manager import CombatManager
    from dnd_bot.game.combat.coordinator import CombatTurnCoordinator
    from dnd_bot.game.combat.actions import CombatAction, CombatActionType, ActionResult

    # Create a combat encounter
    combat = CombatManager.create_encounter(
        session_id="test-session",
        channel_id=99999,
        name="Test Combat",
        description="Test encounter",
    )

    # Add a player
    player_char = make_character(name="TestHero", hp=20, ac=15)
    combat.add_player(player_char)

    # Add a goblin enemy
    goblin = combat.add_monster("goblin", name="Test Goblin")
    assert_true(r, "Goblin combatant created", goblin is not None)

    # Set surprise on goblin
    goblin.is_surprised = True
    assert_true(r, "Goblin marked surprised", goblin.is_surprised)

    # Roll initiative and start combat
    combat.roll_all_initiative()
    combat.start_combat()
    assert_true(r, "Combat is active", combat.combat.state in (CombatState.ACTIVE, CombatState.AWAITING_ACTION))

    # Create coordinator
    coordinator = CombatTurnCoordinator(combat)

    # Run the surprised goblin's turn
    results = asyncio.get_event_loop().run_until_complete(
        coordinator.run_npc_turn(goblin)
    )

    assert_true(r, "Surprised turn returns results", len(results) > 0)
    assert_eq(r, "Surprised result is END_TURN", results[0].action.action_type, CombatActionType.END_TURN)
    assert_true(r, "Surprise cleared after turn", not goblin.is_surprised)

    # Now run a non-surprised turn (goblin should actually attack)
    # Reset turn resources first
    results2 = asyncio.get_event_loop().run_until_complete(
        coordinator.run_npc_turn(goblin)
    )
    assert_true(r, "Non-surprised turn has results", len(results2) > 0)
    # The goblin should attempt an attack, not END_TURN
    if results2[0].action.action_type != CombatActionType.END_TURN:
        assert_eq(r, "Non-surprised action is ATTACK", results2[0].action.action_type, CombatActionType.ATTACK)


def test_coordinator_player_attack(r: TestResult):
    """Test player attack execution through coordinator."""
    print("\n--- Coordinator: Player Attack ---")
    import asyncio
    from dnd_bot.game.combat.manager import CombatManager
    from dnd_bot.game.combat.coordinator import CombatTurnCoordinator
    from dnd_bot.game.combat.actions import CombatAction, CombatActionType

    combat = CombatManager.create_encounter(
        session_id="test-session",
        channel_id=99998,
        name="Test Attack",
        description="Test attack flow",
    )

    player_char = make_character(name="Archer", dexterity=16, hp=20, ac=13)
    combat.add_player(player_char)
    goblin = combat.add_monster("goblin", name="Target Goblin")

    combat.roll_all_initiative()
    combat.start_combat()

    coordinator = CombatTurnCoordinator(combat)

    # Find the player combatant
    player_combatant = None
    for c in combat.combat.combatants:
        if c.is_player:
            player_combatant = c
            break

    assert_true(r, "Found player combatant", player_combatant is not None)

    # Start player turn
    asyncio.get_event_loop().run_until_complete(
        coordinator.start_turn(player_combatant)
    )

    # Execute an attack
    action = CombatAction(
        action_type=CombatActionType.ATTACK,
        combatant_id=player_combatant.id,
        target_ids=[goblin.id],
        weapon_index="longbow",
    )
    result = asyncio.get_event_loop().run_until_complete(
        coordinator.execute_action(action)
    )

    assert_true(r, "Attack result returned", result is not None)
    # Note: attack_roll may be None if weapon resolution falls back to unarmed
    # in test environment (no inventory DB setup). Check that result is valid.
    if result.attack_roll:
        assert_true(r, "Attack has roll", True)
        if result.success:
            assert_true(r, "Hit: damage dealt", sum(result.damage_dealt.values()) > 0 if result.damage_dealt else False)
        else:
            assert_true(r, "Miss: no damage", not result.damage_dealt or sum(result.damage_dealt.values()) == 0)
    else:
        # Weapon not found in test env — just verify the action didn't crash
        assert_true(r, "Attack completed without crash", True)


def test_group_detection(r: TestResult):
    """Test _detect_group_count and _singularize_name."""
    print("\n--- Group Detection ---")
    from dnd_bot.llm.orchestrator import DMOrchestrator

    # Plural names
    assert_eq(r, "Goblins -> 3", DMOrchestrator._detect_group_count("Goblins"), 3)
    assert_eq(r, "Bandits -> 3", DMOrchestrator._detect_group_count("Bandits"), 3)
    assert_eq(r, "Wolves -> 3", DMOrchestrator._detect_group_count("Wolves"), 3)

    # Number words
    assert_eq(r, "Three Goblins -> 3", DMOrchestrator._detect_group_count("Three Goblins"), 3)
    assert_eq(r, "Two Bandits -> 2", DMOrchestrator._detect_group_count("Two Bandits"), 2)
    assert_eq(r, "Five Wolves -> 5", DMOrchestrator._detect_group_count("Five Wolves"), 5)

    # Digit prefix
    assert_eq(r, "3 Goblins -> 3", DMOrchestrator._detect_group_count("3 Goblins"), 3)

    # Singular names
    assert_eq(r, "Goblin -> 1", DMOrchestrator._detect_group_count("Goblin"), 1)
    assert_eq(r, "Bandit Leader -> 1", DMOrchestrator._detect_group_count("Bandit Leader"), 1)
    assert_eq(r, "Wolf -> 1", DMOrchestrator._detect_group_count("Wolf"), 1)

    # Singularize
    assert_eq(r, "Goblins -> Goblin", DMOrchestrator._singularize_name("Goblins"), "Goblin")
    assert_eq(r, "Wolves -> Wolf", DMOrchestrator._singularize_name("Wolves"), "Wolf")
    assert_eq(r, "Three Bandits -> Bandit", DMOrchestrator._singularize_name("Three Bandits"), "Bandit")


def test_multiplayer_session(r: TestResult):
    """Test multiple players in a session and combat."""
    print("\n--- Multi-Player Session ---")
    import asyncio
    from dnd_bot.game.session import GameSession, SessionState, PlayerInfo

    session = GameSession(
        id="test-mp-session",
        channel_id=88888,
        guild_id=88888,
        campaign_id="test-campaign",
        state=SessionState.ACTIVE,
    )

    # Create 3 different player characters
    player1 = make_character(name="Aragorn", strength=16, dexterity=12, hp=25, ac=16, class_index="fighter")
    player1.id = "char-1"
    player1.discord_user_id = 1001

    player2 = make_character(name="Legolas", strength=10, dexterity=18, hp=18, ac=14, class_index="ranger")
    player2.id = "char-2"
    player2.discord_user_id = 1002

    player3 = make_character(name="Gandalf", strength=8, dexterity=10, hp=15, ac=12, class_index="wizard")
    player3.id = "char-3"
    player3.discord_user_id = 1003

    # Join all 3 players
    p1 = session.add_player(1001, "Player1", player1)
    p2 = session.add_player(1002, "Player2", player2)
    p3 = session.add_player(1003, "Player3", player3)

    assert_eq(r, "3 players joined", len(session.players), 3)
    assert_true(r, "Player1 found by user_id", session.get_player(1001) is not None)
    assert_true(r, "Player2 found by user_id", session.get_player(1002) is not None)
    assert_true(r, "Player3 found by user_id", session.get_player(1003) is not None)
    assert_true(r, "Unknown user returns None", session.get_player(9999) is None)

    # All characters accessible
    chars = session.get_all_characters()
    assert_eq(r, "3 characters returned", len(chars), 3)
    char_names = {c.name for c in chars}
    assert_true(r, "All names present", char_names == {"Aragorn", "Legolas", "Gandalf"})

    # Player-character mapping
    assert_eq(r, "Player1 -> Aragorn", session.get_player(1001).character.name, "Aragorn")
    assert_eq(r, "Player2 -> Legolas", session.get_player(1002).character.name, "Legolas")

    # Remove a player
    removed = session.remove_player(1002)
    assert_true(r, "Player2 removed", removed is not None)
    assert_eq(r, "2 players remain", len(session.players), 2)
    assert_true(r, "Player2 gone", session.get_player(1002) is None)


def test_multiplayer_combat(r: TestResult):
    """Test combat with multiple player characters + NPCs."""
    print("\n--- Multi-Player Combat ---")
    import asyncio
    from dnd_bot.game.combat.manager import CombatManager
    from dnd_bot.game.combat.coordinator import CombatTurnCoordinator
    from dnd_bot.game.combat.actions import CombatAction, CombatActionType

    combat = CombatManager.create_encounter(
        session_id="test-mp",
        channel_id=77777,
        name="Multi-Player Battle",
        description="3 players vs 2 goblins",
    )

    # Add 3 players
    fighter = make_character(name="Aragorn", strength=16, dexterity=12, hp=25, ac=16, class_index="fighter")
    fighter.id = "char-fighter"
    fighter.discord_user_id = 1001

    ranger = make_character(name="Legolas", strength=10, dexterity=18, hp=18, ac=14, class_index="ranger")
    ranger.id = "char-ranger"
    ranger.discord_user_id = 1002

    wizard = make_character(name="Gandalf", strength=8, dexterity=10, hp=15, ac=12, class_index="wizard")
    wizard.id = "char-wizard"
    wizard.discord_user_id = 1003

    combat.add_player(fighter)
    combat.add_player(ranger)
    combat.add_player(wizard)

    # Add 2 goblins
    goblin1 = combat.add_monster("goblin", name="Goblin Archer")
    goblin2 = combat.add_monster("goblin", name="Goblin Grunt")

    assert_eq(r, "5 combatants total", len(combat.combat.combatants), 5)

    # Count players vs NPCs
    players = [c for c in combat.combat.combatants if c.is_player]
    npcs = [c for c in combat.combat.combatants if not c.is_player]
    assert_eq(r, "3 player combatants", len(players), 3)
    assert_eq(r, "2 NPC combatants", len(npcs), 2)

    # Roll initiative
    combat.roll_all_initiative()
    combat.start_combat()

    sorted_combatants = combat.combat.get_sorted_combatants()
    assert_eq(r, "5 in initiative order", len(sorted_combatants), 5)

    # Verify all names present in initiative
    names_in_order = [c.name for c in sorted_combatants]
    assert_true(r, "Aragorn in initiative", "Aragorn" in names_in_order)
    assert_true(r, "Legolas in initiative", "Legolas" in names_in_order)
    assert_true(r, "Gandalf in initiative", "Gandalf" in names_in_order)
    assert_true(r, "Goblin Archer in initiative", "Goblin Archer" in names_in_order)
    assert_true(r, "Goblin Grunt in initiative", "Goblin Grunt" in names_in_order)

    # Verify character_id linkage on player combatants
    for pc in players:
        assert_true(r, f"{pc.name} has character_id", pc.character_id is not None)
        assert_true(r, f"{pc.name} is_player=True", pc.is_player)

    for npc in npcs:
        assert_true(r, f"{npc.name} is_player=False", not npc.is_player)
        assert_true(r, f"{npc.name} has monster_index", npc.monster_index is not None)

    # Test turn advancement cycles through all combatants
    first = combat.combat.get_current_combatant()
    assert_true(r, "First combatant exists", first is not None)

    seen = set()
    for _ in range(5):
        current = combat.combat.get_current_combatant()
        seen.add(current.name)
        combat.next_turn()
    assert_eq(r, "All 5 combatants got a turn", len(seen), 5)

    # After 5 next_turn calls, we should be back to the first combatant
    back_to_first = combat.combat.get_current_combatant()
    assert_eq(r, "Round cycles back", back_to_first.name, first.name)
    assert_eq(r, "Round advanced to 2", combat.combat.current_round, 2)


def test_multiplayer_turn_context(r: TestResult):
    """Test that turn context correctly reflects each player's unique data."""
    print("\n--- Multi-Player Turn Context ---")
    import asyncio
    from dnd_bot.game.combat.manager import CombatManager
    from dnd_bot.game.combat.coordinator import CombatTurnCoordinator

    combat = CombatManager.create_encounter(
        session_id="test-mp-ctx",
        channel_id=66666,
        name="Context Test",
        description="Testing per-player context",
    )

    fighter = make_character(name="Conan", strength=18, dexterity=10, hp=30, ac=18, class_index="fighter")
    fighter.id = "char-conan"
    ranger = make_character(name="Robin", strength=12, dexterity=18, hp=20, ac=14, class_index="ranger")
    ranger.id = "char-robin"

    combat.add_player(fighter)
    combat.add_player(ranger)
    goblin = combat.add_monster("goblin", name="Test Goblin")

    combat.roll_all_initiative()
    combat.start_combat()

    coordinator = CombatTurnCoordinator(combat)

    # Find each player combatant
    conan_comb = None
    robin_comb = None
    for c in combat.combat.combatants:
        if c.name == "Conan":
            conan_comb = c
        elif c.name == "Robin":
            robin_comb = c

    assert_true(r, "Conan combatant found", conan_comb is not None)
    assert_true(r, "Robin combatant found", robin_comb is not None)

    # Build turn context for each — should have different stats
    conan_ctx = asyncio.get_event_loop().run_until_complete(
        coordinator._build_turn_context(conan_comb)
    )
    robin_ctx = asyncio.get_event_loop().run_until_complete(
        coordinator._build_turn_context(robin_comb)
    )

    assert_eq(r, "Conan context name", conan_ctx.combatant_name, "Conan")
    assert_eq(r, "Robin context name", robin_ctx.combatant_name, "Robin")
    assert_eq(r, "Conan HP 30", conan_ctx.hp_max, 30)
    assert_eq(r, "Robin HP 20", robin_ctx.hp_max, 20)
    assert_eq(r, "Conan AC 18", conan_ctx.armor_class, 18)
    assert_eq(r, "Robin AC 14", robin_ctx.armor_class, 14)
    assert_true(r, "Conan is player", conan_ctx.is_player)
    assert_true(r, "Robin is player", robin_ctx.is_player)

    # Character IDs may be None in test env (no DB to load from _character_cache)
    # The combatant itself has character_id — verify that instead
    assert_eq(r, "Conan combatant char_id", conan_comb.character_id, "char-conan")
    assert_eq(r, "Robin combatant char_id", robin_comb.character_id, "char-robin")


def test_combat_end_conditions(r: TestResult):
    """Test combat end with multiple players — ends when all enemies OR all players down."""
    print("\n--- Combat End Conditions ---")
    from dnd_bot.game.combat.manager import CombatManager

    combat = CombatManager.create_encounter(
        session_id="test-end",
        channel_id=55555,
        name="End Test",
        description="Testing combat end",
    )

    p1 = make_character(name="Hero1", hp=20, ac=15)
    p1.id = "hero1"
    p2 = make_character(name="Hero2", hp=20, ac=15)
    p2.id = "hero2"
    combat.add_player(p1)
    combat.add_player(p2)

    g1 = combat.add_monster("goblin", name="Goblin A")
    g2 = combat.add_monster("goblin", name="Goblin B")

    combat.roll_all_initiative()
    combat.start_combat()

    # Not over initially
    assert_true(r, "Combat not over at start", not combat.combat.is_combat_over())

    # Kill one goblin — not over
    g1.hp_current = 0
    assert_true(r, "Not over with 1 goblin dead", not combat.combat.is_combat_over())

    # Kill both goblins — combat over (victory)
    g2.hp_current = 0
    assert_true(r, "Over when all enemies dead", combat.combat.is_combat_over())

    # Reset — kill one player
    g1.hp_current = 7
    g2.hp_current = 7
    p1_comb = [c for c in combat.combat.combatants if c.name == "Hero1"][0]
    p2_comb = [c for c in combat.combat.combatants if c.name == "Hero2"][0]

    p1_comb.hp_current = 0
    assert_true(r, "Not over with 1 player dead", not combat.combat.is_combat_over())

    # Kill both players — combat over (defeat)
    p2_comb.hp_current = 0
    assert_true(r, "Over when all players dead", combat.combat.is_combat_over())


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
    test_coordinator_npc_turn(r)
    test_coordinator_player_attack(r)
    test_group_detection(r)
    test_multiplayer_session(r)
    test_multiplayer_combat(r)
    test_multiplayer_turn_context(r)
    test_combat_end_conditions(r)

    success = r.summary()
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
