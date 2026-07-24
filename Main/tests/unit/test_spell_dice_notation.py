"""SRD dice-notation crash net (2026-07-24 adversarial-review finding).

SRD heal/damage strings arrive in shapes the DiceRoller rejected outright:

- ``"1d8 + MOD"``      — heal strings; the old ``replace("+ mod", ...)`` was
  case-sensitive so the MOD token survived into the roller (cure-wounds and
  EVERY leveled heal crashed with ``Invalid dice notation``)
- ``"10d6 + 40"``      — spaced flat modifier (disintegrate, finger-of-death)
- ``"4d6 + 5d6"``      — compound dice sums (flame-strike, ice-storm)
- ``"20"``             — flat integers (guardian-of-faith, heal)

These pins exercise the roller directly with the real SRD strings, the
SpellcastingManager heal/save paths through a stub SRD (the
test_combat_round pattern — no dependence on the external 5e-database
checkout), and the CombatTurnCoordinator._execute_spell seam where the
crash was confirmed.
"""

import pytest

from dnd_bot.game.combat.actions import CombatAction, CombatActionType
from dnd_bot.game.combat.coordinator import CombatTurnCoordinator
from dnd_bot.game.combat.manager import CombatManager
from dnd_bot.game.mechanics.dice import DiceRoll, DiceRoller
from dnd_bot.game.session import GameSession, SessionState
from dnd_bot.models import (
    AbilityScore,
    AbilityScores,
    Character,
    HitDice,
    HitPoints,
    SpellSlots,
)
from tests.unit.test_dice import ScriptedRNG

# ── Canned SRD rows (real strings, verbatim from 5e-SRD-Spells.json) ─────────

_CURE_WOUNDS = {
    "index": "cure-wounds",
    "name": "Cure Wounds",
    "level": 1,
    "school": {"index": "evocation"},
    "casting_time": "1 action",
    "range": "Touch",
    "components": ["V", "S"],
    "duration": "Instantaneous",
    "concentration": False,
    "ritual": False,
    "desc": ["A creature you touch regains hit points."],
    "heal_at_slot_level": {
        "1": "1d8 + MOD",
        "2": "2d8 + MOD",
        "3": "3d8 + MOD",
        "4": "4d8 + MOD",
    },
}

_HEAL = {
    "index": "heal",
    "name": "Heal",
    "level": 6,
    "school": {"index": "evocation"},
    "casting_time": "1 action",
    "range": "60 feet",
    "components": ["V", "S"],
    "duration": "Instantaneous",
    "concentration": False,
    "ritual": False,
    "desc": ["A creature regains a flat amount of hit points."],
    "heal_at_slot_level": {"6": "70", "7": "80", "8": "90", "9": "100"},
}

_FLAME_STRIKE = {
    "index": "flame-strike",
    "name": "Flame Strike",
    "level": 5,
    "school": {"index": "evocation"},
    "casting_time": "1 action",
    "range": "60 feet",
    "components": ["V", "S", "M"],
    "duration": "Instantaneous",
    "concentration": False,
    "ritual": False,
    "desc": ["A vertical column of divine fire roars down."],
    "dc": {"dc_type": {"index": "dex"}},
    "damage": {
        "damage_type": {"name": "Fire"},
        "damage_at_slot_level": {
            "5": "4d6 + 4d6",
            "6": "4d6 + 5d6",
            "7": "4d6 + 6d6",
        },
    },
}

_DISINTEGRATE = {
    "index": "disintegrate",
    "name": "Disintegrate",
    "level": 6,
    "school": {"index": "transmutation"},
    "casting_time": "1 action",
    "range": "60 feet",
    "components": ["V", "S", "M"],
    "duration": "Instantaneous",
    "concentration": False,
    "ritual": False,
    "desc": ["A thin green ray springs from your finger."],
    "dc": {"dc_type": {"index": "dex"}},
    "damage": {
        "damage_type": {"name": "Force"},
        "damage_at_slot_level": {"6": "10d6 + 40"},
    },
}

_FINGER_OF_DEATH = {
    "index": "finger-of-death",
    "name": "Finger of Death",
    "level": 7,
    "school": {"index": "necromancy"},
    "casting_time": "1 action",
    "range": "60 feet",
    "components": ["V", "S"],
    "duration": "Instantaneous",
    "concentration": False,
    "ritual": False,
    "desc": ["You send negative energy coursing through a creature."],
    "dc": {"dc_type": {"index": "con"}},
    "damage": {
        "damage_type": {"name": "Necrotic"},
        "damage_at_slot_level": {"7": "7d8 + 30"},
    },
}

_ICE_STORM = {
    "index": "ice-storm",
    "name": "Ice Storm",
    "level": 4,
    "school": {"index": "evocation"},
    "casting_time": "1 action",
    "range": "300 feet",
    "components": ["V", "S", "M"],
    "duration": "Instantaneous",
    "concentration": False,
    "ritual": False,
    "desc": ["A hail of rock-hard ice pounds to the ground."],
    "dc": {"dc_type": {"index": "dex"}},
    "damage": {
        "damage_type": {"name": "Bludgeoning"},
        "damage_at_slot_level": {"4": "2d8 + 4d6", "5": "3d8 + 4d6"},
    },
}

_GUARDIAN_OF_FAITH = {
    "index": "guardian-of-faith",
    "name": "Guardian of Faith",
    "level": 4,
    "school": {"index": "conjuration"},
    "casting_time": "1 action",
    "range": "30 feet",
    "components": ["V"],
    "duration": "8 hours",
    "concentration": False,
    "ritual": False,
    "desc": ["A Large spectral guardian appears."],
    "dc": {"dc_type": {"index": "dex"}},
    "damage": {
        "damage_type": {"name": "Radiant"},
        "damage_at_slot_level": {"4": "20"},
    },
}

_SPELLS = {
    row["index"]: row
    for row in (
        _CURE_WOUNDS,
        _HEAL,
        _FLAME_STRIKE,
        _DISINTEGRATE,
        _FINGER_OF_DEATH,
        _ICE_STORM,
        _GUARDIAN_OF_FAITH,
    )
}


class _StubSRD:
    """Serves exactly the canned spell rows the tests reference."""

    def get_spell(self, index):
        return _SPELLS.get(index)

    def get_equipment(self, index):
        return None

    def get_monster(self, index):
        return None


# ── Fixtures / helpers ────────────────────────────────────────────────────────


def _make_cleric(wisdom: int = 16) -> Character:
    """Level-9 cleric: WIS 16 → +3 mod, proficiency +4, save DC 15."""
    return Character(
        discord_user_id=77001,
        campaign_id="spell-dice-test",
        name="Test Cleric",
        race_index="human",
        class_index="cleric",
        level=9,
        abilities=AbilityScores(
            strength=10,
            dexterity=12,
            constitution=14,
            intelligence=10,
            wisdom=wisdom,
            charisma=13,
        ),
        hp=HitPoints(maximum=60, current=60),
        hit_dice=HitDice(die_type=8, total=9, remaining=9),
        armor_class=16,
        speed=30,
        initiative_bonus=1,
        spellcasting_ability=AbilityScore.WISDOM,
        spell_slots=SpellSlots(
            level_1=(2, 2),
            level_2=(2, 2),
            level_3=(2, 2),
            level_4=(1, 1),
            level_5=(1, 1),
            level_6=(1, 1),
            level_7=(1, 1),
        ),
        known_spells=list(_SPELLS),
        prepared_spells=list(_SPELLS),
    )


def _make_manager(monkeypatch, rng_values):
    """SpellcastingManager with a stub SRD and a deterministic roller."""
    import dnd_bot.game.magic.spellcasting as spellcasting_mod

    roller = DiceRoller(rng=ScriptedRNG(list(rng_values)))
    monkeypatch.setattr(spellcasting_mod, "get_roller", lambda: roller)
    monkeypatch.setattr(spellcasting_mod, "get_srd", lambda: _StubSRD())
    return spellcasting_mod.SpellcastingManager()


class _FakeInventoryRepo:
    async def get_equipped_items(self, character_id):
        return []


class _FakeCharacterRepo:
    """Records the targeted slot/concentration writes the coordinator issues."""

    def __init__(self):
        self.slot_calls: list[tuple] = []
        self.concentration_calls: list[tuple] = []

    async def update_spell_slot(self, character_id, slot_level, current):
        self.slot_calls.append((character_id, slot_level, current))
        return True

    async def update_concentration(self, character_id, spell_id):
        self.concentration_calls.append((character_id, spell_id))
        return True


class _ScriptedCheckRoller:
    """Coordinator-roller stand-in for target saving throws (roll_check)."""

    def __init__(self, faces):
        self.faces = list(faces)
        self.calls: list[tuple] = []

    def roll_check(self, modifier=0, advantage=False, disadvantage=False):
        self.calls.append(("check", modifier, advantage, disadvantage))
        face = self.faces.pop(0)
        return DiceRoll(
            notation="1d20",
            dice_results=[face],
            kept_dice=[face],
            modifier=modifier,
            total=face + modifier,
        )


# ── The roller itself, fed the real SRD strings ──────────────────────────────


class TestRollerSRDNotation:
    """DiceRoller must accept every shape the SRD actually ships."""

    def test_spaced_flat_modifier(self):
        """"1d8 + 3" — the shape a substituted heal string arrives in."""
        roller = DiceRoller(rng=ScriptedRNG([5]))
        result = roller.roll("1d8 + 3")
        assert result.kept_dice == [5]
        assert result.modifier == 3
        assert result.total == 8

    def test_disintegrate_spaced_modifier(self):
        """"10d6 + 40" — disintegrate's SRD damage string, verbatim."""
        roller = DiceRoller(rng=ScriptedRNG([3] * 10))
        result = roller.roll("10d6 + 40")
        assert len(result.kept_dice) == 10
        assert result.modifier == 40
        assert result.total == 70

    def test_finger_of_death_spaced_modifier(self):
        """"7d8 + 30" — finger-of-death's SRD damage string, verbatim."""
        roller = DiceRoller(rng=ScriptedRNG([2] * 7))
        result = roller.roll("7d8 + 30")
        assert result.total == 44

    def test_flame_strike_compound_dice(self):
        """"4d6 + 5d6" — upcast flame-strike: two dice groups, no modifier."""
        roller = DiceRoller(rng=ScriptedRNG([1, 2, 3, 4, 5, 6, 1, 2, 3]))
        result = roller.roll("4d6 + 5d6")
        assert len(result.kept_dice) == 9
        assert result.modifier == 0
        assert result.total == 27

    def test_ice_storm_mixed_compound_dice(self):
        """"2d8 + 4d6" — ice-storm mixes die sizes across groups."""
        roller = DiceRoller(rng=ScriptedRNG([8, 7, 1, 2, 3, 4]))
        result = roller.roll("2d8 + 4d6")
        assert len(result.kept_dice) == 6
        assert result.total == 25

    def test_flat_integer(self):
        """"20" — guardian-of-faith deals a flat 20; no dice to roll."""
        result = DiceRoller(rng=ScriptedRNG([])).roll("20")
        assert result.kept_dice == []
        assert result.total == 20

    def test_flat_integer_heal(self):
        """"70" — the heal spell restores a flat 70."""
        result = DiceRoller(rng=ScriptedRNG([])).roll("70")
        assert result.total == 70

    def test_meteor_swarm_large_compound(self):
        """"20d6 + 20d6" — the largest compound string in the SRD."""
        result = DiceRoller().roll("20d6 + 20d6")
        assert len(result.kept_dice) == 40
        assert 40 <= result.total <= 240

    def test_unsubstituted_mod_token_still_loud(self):
        """A MOD token that escapes substitution must stay a loud error."""
        with pytest.raises(ValueError):
            DiceRoller().roll("1d8 + MOD")

    def test_garbage_still_rejected(self):
        with pytest.raises(ValueError):
            DiceRoller().roll("not dice")

    def test_subtracted_dice_group_rejected(self):
        """The SRD never subtracts dice; keep the grammar strict."""
        with pytest.raises(ValueError):
            DiceRoller().roll("4d6 - 1d4")

    def test_critical_doubles_spaced_notation(self):
        roller = DiceRoller(rng=ScriptedRNG([4, 5]))
        result = roller.roll_damage("1d8 + 3", critical=True)
        assert len(result.kept_dice) == 2
        assert result.modifier == 3
        assert result.total == 12

    def test_critical_doubles_every_compound_group(self):
        """A crit doubles ALL the attack's damage dice, per group."""
        roller = DiceRoller(rng=ScriptedRNG([1] * 16))
        result = roller.roll_damage("4d6 + 4d6", critical=True)
        assert len(result.kept_dice) == 16
        assert result.total == 16


# ── SpellcastingManager heal/save paths ──────────────────────────────────────


class TestCastHealingSpell:
    """cast_healing_spell must survive every SRD heal string."""

    def test_cure_wounds_substitutes_uppercase_mod(self, monkeypatch):
        """"1d8 + MOD" + WIS 16 → 1d8+3. The old replace() missed uppercase."""
        manager = _make_manager(monkeypatch, [5])
        cleric = _make_cleric()
        spell = manager.get_spell_info("cure-wounds")

        result = manager.cast_healing_spell(cleric, spell, slot_level=1)

        assert result.healing_amount == 8
        assert result.healing_roll is not None
        assert result.healing_roll.modifier == 3

    def test_cure_wounds_upcast_scales_dice(self, monkeypatch):
        """"3d8 + MOD" at slot 3 → three dice plus the modifier."""
        manager = _make_manager(monkeypatch, [5, 6, 7])
        cleric = _make_cleric()
        spell = manager.get_spell_info("cure-wounds")

        result = manager.cast_healing_spell(cleric, spell, slot_level=3)

        assert result.healing_amount == 21  # 5+6+7 dice + 3 mod

    def test_negative_modifier_substitutes_with_sign(self, monkeypatch):
        """WIS 8 → -1: "1d8 + MOD" must become 1d8-1, not "1d8+-1"."""
        manager = _make_manager(monkeypatch, [1])
        cleric = _make_cleric(wisdom=8)
        spell = manager.get_spell_info("cure-wounds")

        result = manager.cast_healing_spell(cleric, spell, slot_level=1)

        assert result.healing_roll is not None
        assert result.healing_roll.modifier == -1
        assert result.healing_amount == 0  # max(0, 1 - 1)

    def test_no_spellcasting_ability_defaults_mod_to_zero(self, monkeypatch):
        """A caster without a spellcasting ability heals with MOD=0, not a crash."""
        manager = _make_manager(monkeypatch, [4])
        cleric = _make_cleric()
        cleric.spellcasting_ability = None
        spell = manager.get_spell_info("cure-wounds")

        result = manager.cast_healing_spell(cleric, spell, slot_level=1)

        assert result.healing_amount == 4

    def test_flat_heal_string(self, monkeypatch):
        """The heal spell's "70" is a flat amount — no dice at all."""
        manager = _make_manager(monkeypatch, [])
        cleric = _make_cleric()
        spell = manager.get_spell_info("heal")

        result = manager.cast_healing_spell(cleric, spell, slot_level=6)

        assert result.healing_amount == 70


class TestCastSaveSpell:
    """cast_save_spell must survive spaced, compound, and flat SRD strings."""

    def test_flame_strike_compound_base_slot(self, monkeypatch):
        manager = _make_manager(monkeypatch, [1, 2, 3, 4, 5, 6, 1, 2])
        cleric = _make_cleric()
        spell = manager.get_spell_info("flame-strike")

        result = manager.cast_save_spell(cleric, spell, slot_level=5)

        assert result.damage_dealt == 24
        assert result.damage_type == "fire"
        assert result.save_dc == 15
        assert result.save_ability == AbilityScore.DEXTERITY

    def test_flame_strike_upcast_adds_a_group_die(self, monkeypatch):
        """Slot 6 → "4d6 + 5d6": nine dice, not eight."""
        manager = _make_manager(monkeypatch, [1] * 9)
        cleric = _make_cleric()
        spell = manager.get_spell_info("flame-strike")

        result = manager.cast_save_spell(cleric, spell, slot_level=6)

        assert result.damage_roll is not None
        assert len(result.damage_roll.kept_dice) == 9

    def test_disintegrate_spaced_modifier(self, monkeypatch):
        manager = _make_manager(monkeypatch, [3] * 10)
        cleric = _make_cleric()
        spell = manager.get_spell_info("disintegrate")

        result = manager.cast_save_spell(cleric, spell, slot_level=6)

        assert result.damage_dealt == 70  # 10×3 + 40

    def test_finger_of_death_spaced_modifier(self, monkeypatch):
        manager = _make_manager(monkeypatch, [2] * 7)
        cleric = _make_cleric()
        spell = manager.get_spell_info("finger-of-death")

        result = manager.cast_save_spell(cleric, spell, slot_level=7)

        assert result.damage_dealt == 44  # 7×2 + 30

    def test_ice_storm_mixed_compound(self, monkeypatch):
        manager = _make_manager(monkeypatch, [8, 7, 1, 2, 3, 4])
        cleric = _make_cleric()
        spell = manager.get_spell_info("ice-storm")

        result = manager.cast_save_spell(cleric, spell, slot_level=4)

        assert result.damage_dealt == 25

    def test_guardian_of_faith_flat_damage(self, monkeypatch):
        manager = _make_manager(monkeypatch, [])
        cleric = _make_cleric()
        spell = manager.get_spell_info("guardian-of-faith")

        result = manager.cast_save_spell(cleric, spell, slot_level=4)

        assert result.damage_dealt == 20


# ── The confirmed crash seam: CombatTurnCoordinator._execute_spell ───────────


class TestCombatSpellNotation:
    """The in-combat casts that crashed, end-to-end through execute_action."""

    def _setup(self, monkeypatch, unique_channel_id, cleric, spell_rng):
        manager = CombatManager.create_encounter(
            session_id="spell-dice-session",
            channel_id=unique_channel_id,
            name="Spell Dice Test",
        )
        manager.add_player(cleric)
        goblin = manager.add_custom_combatant(name="Goblin", hp=30, ac=13)
        manager.start_combat()
        player = next(c for c in manager.combat.combatants if c.is_player)
        player.turn_order = 0
        goblin.turn_order = 1
        manager.combat.current_turn_index = 0

        session = GameSession(
            id="spell-dice-session",
            channel_id=unique_channel_id,
            guild_id=1,
            campaign_id="spell-dice-campaign",
            state=SessionState.COMBAT,
        )
        session.add_player(cleric.discord_user_id, "Tester", cleric)
        coordinator = CombatTurnCoordinator(manager, session)

        # Deterministic spellcasting seams (stub SRD, scripted dice).
        spell_roller = DiceRoller(rng=ScriptedRNG(list(spell_rng)))
        monkeypatch.setattr(
            "dnd_bot.game.magic.spellcasting.get_roller", lambda: spell_roller
        )
        monkeypatch.setattr(
            "dnd_bot.game.magic.spellcasting.get_srd", lambda: _StubSRD()
        )

        repo = _FakeCharacterRepo()

        async def _get_char_repo():
            return repo

        inventory = _FakeInventoryRepo()

        async def _get_inv_repo():
            return inventory

        monkeypatch.setattr(
            "dnd_bot.game.combat.coordinator.get_character_repo", _get_char_repo
        )
        monkeypatch.setattr(
            "dnd_bot.game.combat.coordinator.get_inventory_repo", _get_inv_repo
        )

        return coordinator, player, goblin, repo

    async def test_cure_wounds_heals_in_combat(
        self, unique_channel_id, monkeypatch
    ):
        """The headline crash: a leveled heal cast through _execute_spell."""
        cleric = _make_cleric()
        coordinator, player, _, repo = self._setup(
            monkeypatch, unique_channel_id, cleric, spell_rng=[5]
        )
        player.hp_current = 40

        await coordinator.start_turn(player)
        result = await coordinator.execute_action(
            CombatAction(
                action_type=CombatActionType.CAST_SPELL,
                combatant_id=player.id,
                spell_index="cure-wounds",
                slot_level=1,
                target_ids=[player.id],
            )
        )

        assert result.success is True
        assert result.error is None
        assert result.healing_done == {player.id: 8}  # die 5 + WIS mod 3
        assert player.hp_current == 48
        # The slot was expended exactly once and persisted.
        assert cleric.spell_slots.get_slots(1) == (1, 2)
        assert repo.slot_calls == [(cleric.id, 1, 1)]
        assert player.turn_resources.action is False

    async def test_flame_strike_damages_in_combat(
        self, unique_channel_id, monkeypatch
    ):
        """A compound-notation save spell cast through _execute_spell."""
        cleric = _make_cleric()
        coordinator, player, goblin, repo = self._setup(
            monkeypatch, unique_channel_id, cleric,
            spell_rng=[1, 2, 3, 4, 5, 6, 1, 2],
        )
        # Goblin fails its DEX save (face 3 vs DC 15) → full damage.
        coordinator.roller = _ScriptedCheckRoller(faces=[3])

        await coordinator.start_turn(player)
        result = await coordinator.execute_action(
            CombatAction(
                action_type=CombatActionType.CAST_SPELL,
                combatant_id=player.id,
                spell_index="flame-strike",
                slot_level=5,
                target_ids=[goblin.id],
            )
        )

        assert result.success is True
        assert result.damage_dealt == {goblin.id: 24}
        assert goblin.hp_current == 6
        assert cleric.spell_slots.get_slots(5) == (0, 1)
        assert repo.slot_calls == [(cleric.id, 5, 0)]
