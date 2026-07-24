"""REFACTOR_PLAN Step-3 prerequisite net: pin a full combat round at the
game layer, through ``CombatTurnCoordinator``.

coordinator.py (1586 lines) had ZERO direct combat-round coverage — the
existing nets pin teardown (test_combat_teardown), locking
(test_combat_turn_lock) and the combat-over edge (test_combat_over), but
nobody pinned the happy path Step 3 is about to restructure:

    start_turn -> execute_action -> end_turn -> NPC turn -> combat-over

Determinism (the plan's rule: scrub rolls, pin structure):
- Initiative is pinned by assigning turn_order directly (player 0, goblin 1).
- The coordinator's roller is replaced by ``_ScriptedRoller``, which pops
  scripted d20 FACES / damage sums and RECORDS every request — so the pins
  assert the coordinator's own arithmetic (attack modifier, damage notation,
  advantage flags) without pinning randomness.
- The SRD loader is replaced by a stub (no dependence on the external
  5e-database checkout) and the inventory repo by a fake (no DB).

Teardown through GameSessionManager.end_combat is deliberately NOT
re-pinned here — that is test_combat_teardown.py's contract.
"""

from types import SimpleNamespace

import pytest

from dnd_bot.game.combat.actions import CombatAction, CombatActionType
from dnd_bot.game.combat.coordinator import CombatTurnCoordinator
from dnd_bot.game.combat.manager import CombatManager
from dnd_bot.game.mechanics.dice import DiceRoll, DiceRoller
from dnd_bot.game.session import GameSession, SessionState
from dnd_bot.models import CombatState, Condition
from dnd_bot.models.combat import CombatEffect

# Channel ids come from the run-unique ``unique_channel_id`` fixture — the
# combat, coordinator, and turn-lock registries are module-level globals.


# ── Deterministic collaborators ───────────────────────────────────────────────

# Canned SRD rows so the net never reads the external 5e-database checkout.
_LONGSWORD = {
    "index": "longsword",
    "name": "Longsword",
    "equipment_category": {"index": "weapon"},
    "weapon_range": "Melee",
    "damage": {"damage_dice": "1d8", "damage_type": {"name": "Slashing"}},
    "properties": [{"index": "versatile"}],
}

_GOBLIN = {
    "index": "goblin",
    "name": "Goblin",
    "actions": [
        {
            "name": "Scimitar",
            "attack_bonus": 4,
            "damage": [
                {"damage_dice": "1d6", "damage_type": {"name": "Slashing"}}
            ],
            "desc": "Melee Weapon Attack: +4 to hit, reach 5 ft., one target.",
        }
    ],
}

_HOLD_PERSON = {
    "index": "hold-person",
    "name": "Hold Person",
    "level": 2,
    "school": {"index": "enchantment"},
    "casting_time": "1 action",
    "range": "60 feet",
    "components": ["V", "S", "M"],
    "material": "A small, straight piece of iron.",
    "duration": "Up to 1 minute",
    "concentration": True,
    "ritual": False,
    "desc": ["Choose a humanoid... paralyzed for the duration."],
    "dc": {"dc_type": {"index": "wis"}},
}


class _StubSRD:
    """Serves exactly the canned entries the tests reference."""

    def get_equipment(self, index):
        return _LONGSWORD if index == "longsword" else None

    def get_monster(self, index):
        return _GOBLIN if index == "goblin" else None

    def get_spell(self, index):
        return _HOLD_PERSON if index == "hold-person" else None


class _ScriptedRoller:
    """DiceRoller stand-in: scripted d20 faces / damage sums, recorded calls.

    ``attack_faces`` are raw d20 faces — total = face + the modifier the
    coordinator computed, so modifier math stays real and pinnable.
    ``damage_sums`` are the dice-only sums — the coordinator adds the
    ability modifier on top, so damage math stays real and pinnable.
    ``save_faces`` are raw d20 faces for saving throws (default face 10).
    """

    def __init__(self, attack_faces=(), damage_sums=(), save_faces=()):
        self.attack_faces = list(attack_faces)
        self.damage_sums = list(damage_sums)
        self.save_faces = list(save_faces)
        self.calls: list[tuple] = []

    def roll_attack(self, modifier=0, advantage=False, disadvantage=False):
        self.calls.append(("attack", modifier, advantage, disadvantage))
        face = self.attack_faces.pop(0)
        return DiceRoll(
            notation="1d20",
            dice_results=[face],
            kept_dice=[face],
            modifier=modifier,
            total=face + modifier,
            natural_20=face == 20,
            natural_1=face == 1,
        )

    def roll_damage(self, notation, critical=False):
        self.calls.append(("damage", notation, critical))
        total = self.damage_sums.pop(0)
        return DiceRoll(
            notation=notation, dice_results=[total], kept_dice=[total], total=total
        )

    def roll_check(self, modifier=0, advantage=False, disadvantage=False):
        self.calls.append(("check", modifier, advantage, disadvantage))
        return DiceRoll(
            notation="1d20", dice_results=[10], kept_dice=[10],
            modifier=modifier, total=10 + modifier,
        )

    def roll_save(self, modifier=0, advantage=False, disadvantage=False):
        self.calls.append(("save", modifier, advantage, disadvantage))
        face = self.save_faces.pop(0) if self.save_faces else 10
        return DiceRoll(
            notation="1d20", dice_results=[face], kept_dice=[face],
            modifier=modifier, total=face + modifier,
        )

    def roll(self, notation, advantage=False, disadvantage=False, reason=""):
        self.calls.append(("roll", notation))
        return DiceRoll(notation=notation, kept_dice=[1], total=1)


class _RealDamageRoller(_ScriptedRoller):
    """Scripted attack faces, REAL damage roller — used to pin what the
    production dice engine actually does with a weapon's damage notation."""

    def roll_damage(self, notation, critical=False):
        self.calls.append(("damage", notation, critical))
        return DiceRoller().roll_damage(notation, critical=critical)


class _ScriptedNpcBrain:
    """decide_action pops scripted CombatActions, then falls back to END_TURN."""

    def __init__(self, actions=()):
        self._actions = list(actions)
        self.decide_calls = 0

    def roll_recharge(self, combatant):
        return []

    async def decide_action(self, combatant, combat_state, zones):
        self.decide_calls += 1
        if self._actions:
            return self._actions.pop(0)
        return CombatAction(
            action_type=CombatActionType.END_TURN, combatant_id=combatant.id
        )


class _FakeInventoryRepo:
    def __init__(self, equipped):
        self._equipped = list(equipped)

    async def get_equipped_items(self, character_id):
        return list(self._equipped)


class _FakeCharacterRepo:
    """Records the targeted concentration/slot writes the coordinator issues."""

    def __init__(self):
        self.concentration_calls: list[tuple] = []
        self.slot_calls: list[tuple] = []

    async def update_concentration(self, character_id, spell_id):
        self.concentration_calls.append((character_id, spell_id))
        return True

    async def update_spell_slot(self, character_id, slot_level, current):
        self.slot_calls.append((character_id, slot_level, current))
        return True


# ── Fixtures / helpers ────────────────────────────────────────────────────────


def _make_combat(channel_id: int, character, goblin_hp: int = 12) -> CombatManager:
    """Mid-combat encounter with a pinned order: player first, goblin second.

    The goblin is a custom combatant carrying ``monster_index='goblin'`` so
    weapon lookup exercises the coordinator's monster-attack parsing against
    the stub SRD (the same shape ``_trigger_combat``'s add_monster path
    produces).
    """
    manager = CombatManager.create_encounter(
        session_id="combat-round-test-session",
        channel_id=channel_id,
        name="Combat Round Test",
    )
    manager.add_player(character)
    goblin = manager.add_custom_combatant(name="Goblin", hp=goblin_hp, ac=13)
    goblin.monster_index = "goblin"
    manager.start_combat()
    player = next(c for c in manager.combat.combatants if c.is_player)
    player.turn_order = 0
    goblin.turn_order = 1
    manager.combat.current_turn_index = 0  # player is acting
    return manager


def _make_session(channel_id: int, character) -> GameSession:
    session = GameSession(
        id="combat-round-session",
        channel_id=channel_id,
        guild_id=1,
        campaign_id="combat-round-campaign",
        state=SessionState.COMBAT,
    )
    session.add_player(character.discord_user_id, "Tester", character)
    return session


@pytest.fixture
def equipped_longsword(monkeypatch):
    """Route the coordinator's inventory reads to a fake repo with a longsword."""
    repo = _FakeInventoryRepo([SimpleNamespace(item_index="longsword")])

    async def _get_repo():
        return repo

    monkeypatch.setattr(
        "dnd_bot.game.combat.coordinator.get_inventory_repo", _get_repo
    )
    return repo


def _coordinator(manager, session, roller) -> CombatTurnCoordinator:
    coordinator = CombatTurnCoordinator(manager, session)
    coordinator.roller = roller
    coordinator.srd = _StubSRD()
    return coordinator


def _attack(attacker_id: str, target_id: str) -> CombatAction:
    return CombatAction(
        action_type=CombatActionType.ATTACK,
        combatant_id=attacker_id,
        target_ids=[target_id],
    )


class TestFullCombatRound:
    """The golden-master trajectory: one full round plus the finishing blow."""

    async def test_full_round_trajectory(
        self, mock_character, unique_channel_id, monkeypatch, equipped_longsword
    ):
        manager = _make_combat(unique_channel_id, mock_character, goblin_hp=12)
        player = next(c for c in manager.combat.combatants if c.is_player)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)
        session = _make_session(unique_channel_id, mock_character)
        roller = _ScriptedRoller(attack_faces=[12, 19, 15], damage_sums=[4, 3, 2])
        coordinator = _coordinator(manager, session, roller)

        brain = _ScriptedNpcBrain([_attack(goblin.id, player.id)])
        monkeypatch.setattr(
            "dnd_bot.game.combat.npc_brain.get_npc_brain", lambda: brain
        )

        # ── Round 1, player turn: start_turn ──
        ctx = await coordinator.start_turn(player)
        assert ctx.combat_over is False
        assert ctx.combatant_id == player.id
        assert ctx.is_player is True
        assert (ctx.has_action, ctx.has_bonus_action, ctx.has_reaction) == (
            True, True, True,
        )
        assert ctx.movement_remaining == 30
        assert (ctx.hp_current, ctx.hp_max, ctx.armor_class) == (44, 44, 18)
        assert ctx.conditions == []
        # Character data resolved session-first (no repo hit) + fake inventory
        assert ctx.character_id == mock_character.id
        assert [w.name for w in ctx.equipped_weapons] == ["Longsword"]
        assert ctx.equipped_weapons[0].damage_dice == "1d8"
        assert ctx.is_concentrating is False

        # ── execute_action: longsword attack, face 12 ──
        # Modifier pinned below via roller.calls: STR +3 (16) + prof +3
        # (level 5) + weapon +0 = 6 -> total 18 vs AC 13 = hit.
        result = await coordinator.execute_action(_attack(player.id, goblin.id))
        assert result.success is True
        assert result.attack_roll.total == 18
        assert result.target_ac == 13
        assert (result.critical_hit, result.critical_miss) == (False, False)
        # Damage = scripted dice sum 4 + STR mod 3
        assert result.damage_dealt == {goblin.id: 7}
        assert result.damage_type == "slashing"
        assert goblin.hp_current == 5
        assert result.unconscious_targets == []
        assert result.killed_targets == []
        # The attack consumed the action
        assert player.turn_resources.action is False

        # ── end_turn: hands off to the goblin, same round ──
        end = await coordinator.end_turn(player)
        assert end.combat_over is False
        assert end.next_combatant_id == goblin.id
        assert end.next_is_player is False
        assert end.round_advanced is False
        assert end.new_round == 1
        assert end.effect_messages == []

        # ── NPC turn: scripted brain swings the scimitar once ──
        results = await coordinator.run_npc_turn(goblin)
        assert brain.decide_calls == 1
        assert len(results) == 1
        npc_result = results[0]
        assert npc_result.success is True
        # Monster modifier comes straight from the stat block (+4);
        # face 19 + 4 = 23 vs the player's AC 18 = hit.
        assert npc_result.attack_roll.total == 23
        assert npc_result.target_ac == 18
        # Monster damage adds NO ability mod (baked into the stat block).
        assert npc_result.damage_dealt == {player.id: 3}
        assert player.hp_current == 41
        # The NPC turn ended itself: initiative wrapped, new round, player up.
        assert manager.combat.current_round == 2
        assert manager.combat.get_current_combatant() is player
        assert manager.combat.state == CombatState.AWAITING_ACTION

        # ── Round 2, player turn: the finishing blow ──
        await coordinator.start_turn(player)
        kill = await coordinator.execute_action(_attack(player.id, goblin.id))
        assert kill.success is True
        assert kill.damage_dealt == {goblin.id: 5}  # dice 2 + STR 3
        assert goblin.hp_current == 0
        # Current behavior: a monster dropped to 0 is reported UNCONSCIOUS,
        # not killed (Combatant.take_damage only flags instant_death for
        # players; ``killed_targets`` needs overflow >= hp_max). Pinned as-is.
        assert kill.unconscious_targets == ["Goblin"]
        assert kill.killed_targets == []

        # ── end_turn on a decided encounter: first-class combat-over ──
        over = await coordinator.end_turn(player)
        assert over.combat_over is True
        assert over.next_combatant_id == ""
        # The advance wrapped initiative (bumping the round) BEFORE seeing
        # is_combat_over — new_round echoes that bump. Pinned exact.
        assert over.new_round == 3
        assert manager.combat.state == CombatState.COMBAT_END
        assert manager.combat.ended_at is not None

        # ── The whole round's dice trajectory, exact ──
        assert roller.calls == [
            ("attack", 6, False, False),   # player longsword: STR 3 + prof 3
            ("damage", "1d8", False),
            ("attack", 4, False, False),   # goblin scimitar: stat-block +4
            ("damage", "1d6", False),
            ("attack", 6, False, False),   # player longsword, round 2
            ("damage", "1d8", False),
        ]


class TestActionEdges:
    """Focused pins on the execute_action edges Step 3 must not disturb."""

    async def test_miss_consumes_action_but_deals_no_damage(
        self, mock_character, unique_channel_id, equipped_longsword
    ):
        manager = _make_combat(unique_channel_id, mock_character)
        player = next(c for c in manager.combat.combatants if c.is_player)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)
        session = _make_session(unique_channel_id, mock_character)
        roller = _ScriptedRoller(attack_faces=[3])  # 3 + 6 = 9 < AC 13
        coordinator = _coordinator(manager, session, roller)

        await coordinator.start_turn(player)
        result = await coordinator.execute_action(_attack(player.id, goblin.id))

        assert result.success is False
        assert result.error is None  # a miss is not an error
        assert result.attack_roll.total == 9
        assert result.damage_dealt == {}
        assert goblin.hp_current == 12
        # The action is spent on a miss (D&D rules), and no damage was rolled.
        assert player.turn_resources.action is False
        assert roller.calls == [("attack", 6, False, False)]

    async def test_unarmed_hit_deals_flat_one_plus_str(
        self, mock_character, unique_channel_id, monkeypatch
    ):
        """FLIPPED (was PINNED BROKEN): the unarmed-strike fallback used to
        declare damage_dice '1', which DiceRoller rejects ('1' is not dice
        notation), so every unarmed HIT died in _execute_action_locked's
        except and came back failed with the action already consumed. The
        fallback (in BOTH _get_weapon_for_attack and _get_equipped_weapons)
        now says '1d1' — a real roll that always totals 1 — and the melee
        damage math adds STR on top: 1 + STR 3 = 4 bludgeoning -> hp 8.
        """
        # No equipped items -> WeaponStats fallback "Unarmed Strike".
        repo = _FakeInventoryRepo([])

        async def _get_repo():
            return repo

        monkeypatch.setattr(
            "dnd_bot.game.combat.coordinator.get_inventory_repo", _get_repo
        )

        manager = _make_combat(unique_channel_id, mock_character)
        player = next(c for c in manager.combat.combatants if c.is_player)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)
        session = _make_session(unique_channel_id, mock_character)
        roller = _RealDamageRoller(attack_faces=[12])  # 12 + 6 = 18: a HIT
        coordinator = _coordinator(manager, session, roller)

        ctx = await coordinator.start_turn(player)
        # Pin the SECOND fallback site too: _get_equipped_weapons feeds
        # TurnContext.equipped_weapons, and its unarmed fallback must also
        # say '1d1' (a partial revert to '1' would only break this seam).
        assert [w.damage_dice for w in ctx.equipped_weapons] == ["1d1"]
        result = await coordinator.execute_action(_attack(player.id, goblin.id))

        assert result.success is True
        assert result.error is None
        assert result.damage_type == "bludgeoning"
        # Flat 1 (the REAL roller resolved "1d1") + STR mod 3.
        assert result.damage_dealt == {goblin.id: 4}
        assert goblin.hp_current == 8
        assert player.turn_resources.action is False
        assert roller.calls == [
            ("attack", 6, False, False),  # unarmed is melee: STR 3 + prof 3
            ("damage", "1d1", False),
        ]

    async def test_blocking_condition_rejects_action_before_resources(
        self, mock_character, unique_channel_id, equipped_longsword
    ):
        manager = _make_combat(unique_channel_id, mock_character)
        player = next(c for c in manager.combat.combatants if c.is_player)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)
        session = _make_session(unique_channel_id, mock_character)
        roller = _ScriptedRoller()
        coordinator = _coordinator(manager, session, roller)

        await coordinator.start_turn(player)
        player.effects.append(
            CombatEffect(name="paralyzed", condition=Condition.PARALYZED)
        )
        result = await coordinator.execute_action(_attack(player.id, goblin.id))

        assert result.success is False
        assert result.error == "Cannot act while paralyzed"
        # The block fires BEFORE resource consumption — action retained.
        assert player.turn_resources.action is True
        assert roller.calls == []


class TestNpcTurnEdges:
    async def test_surprised_npc_skips_its_action_and_turn_advances(
        self, mock_character, unique_channel_id, monkeypatch, equipped_longsword
    ):
        """The surprise leg _trigger_combat(player_initiated=True) relies on:
        a surprised NPC takes no action, surprise clears at end of its turn,
        and initiative still advances."""
        manager = _make_combat(unique_channel_id, mock_character)
        player = next(c for c in manager.combat.combatants if c.is_player)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)
        goblin.is_surprised = True
        manager.combat.current_turn_index = 1  # goblin is acting
        session = _make_session(unique_channel_id, mock_character)
        roller = _ScriptedRoller()
        coordinator = _coordinator(manager, session, roller)

        brain = _ScriptedNpcBrain([_attack(goblin.id, player.id)])
        monkeypatch.setattr(
            "dnd_bot.game.combat.npc_brain.get_npc_brain", lambda: brain
        )

        results = await coordinator.run_npc_turn(goblin)

        # One informative skip result; the brain never got to decide.
        assert brain.decide_calls == 0
        assert len(results) == 1
        assert results[0].success is True
        assert "caught off guard" in (results[0].error or "")
        assert player.hp_current == 44
        assert roller.calls == []
        # Surprise ended with the turn; initiative moved on to the player.
        assert goblin.is_surprised is False
        assert manager.combat.get_current_combatant() is player
        assert manager.combat.current_round == 2


class TestConcentrationBreak:
    """DF-8 surviving leg: breaking concentration must reach ALL THREE stores.

    ``manager.break_concentration`` only strips concentration CombatEffects;
    before the fix a failed CON save never cleared
    ``Character.concentration_spell_id`` nor the DB row, so the spell came
    back on resume/reload. These pin the coordinator's full break: manager
    effects stripped + Character cleared + ``repo.update_concentration(id,
    None)`` issued (mirroring the set-path used on cast).
    """

    def _rig(self, mock_character, unique_channel_id, monkeypatch, roller):
        """Goblin about to attack a player who is concentrating on Bless."""
        mock_character.concentration_spell_id = "bless"
        manager = _make_combat(unique_channel_id, mock_character)
        player = next(c for c in manager.combat.combatants if c.is_player)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)
        # The manager-side footprint of that concentration, as
        # _execute_spell would have attached it.
        player.effects.append(
            CombatEffect(
                name="Bless",
                effect_type="buff",
                source_combatant_id=player.id,
                is_concentration=True,
            )
        )
        manager.combat.current_turn_index = 1  # goblin is acting
        session = _make_session(unique_channel_id, mock_character)
        coordinator = _coordinator(manager, session, roller)

        char_repo = _FakeCharacterRepo()

        async def _get_repo():
            return char_repo

        monkeypatch.setattr(
            "dnd_bot.game.combat.coordinator.get_character_repo", _get_repo
        )
        return player, goblin, coordinator, char_repo

    async def test_failed_con_save_clears_character_and_persists(
        self, mock_character, unique_channel_id, monkeypatch
    ):
        # Scimitar face 19 + 4 = 23 vs AC 18: hit for 3. CON save face 2
        # + CON mod 2 = 4 vs DC 10: FAILED -> full break.
        roller = _ScriptedRoller(
            attack_faces=[19], damage_sums=[3], save_faces=[2]
        )
        player, goblin, coordinator, char_repo = self._rig(
            mock_character, unique_channel_id, monkeypatch, roller
        )

        result = await coordinator.execute_action(_attack(goblin.id, player.id))

        assert result.success is True
        assert player.hp_current == 41
        assert ("save", 2, False, False) in roller.calls
        assert result.concentration_broken is True
        # Manager leg: the concentration effect is stripped...
        assert all(e.name != "Bless" for e in player.effects)
        # ...AND the Character + DB row are cleared (the DF-8 gap).
        assert mock_character.concentration_spell_id is None
        assert char_repo.concentration_calls == [(mock_character.id, None)]

    async def test_dropping_to_zero_hp_breaks_concentration_without_a_save(
        self, mock_character, unique_channel_id, monkeypatch
    ):
        # Overwhelming hit: 44 HP -> 0 (no instant death, overflow < max).
        # 0 HP ends concentration outright (PHB p.203) — no save is rolled.
        roller = _ScriptedRoller(attack_faces=[19], damage_sums=[50])
        player, goblin, coordinator, char_repo = self._rig(
            mock_character, unique_channel_id, monkeypatch, roller
        )

        result = await coordinator.execute_action(_attack(goblin.id, player.id))

        assert result.success is True
        assert player.hp_current == 0
        assert not any(call[0] == "save" for call in roller.calls)
        assert result.concentration_broken is True
        assert all(e.name != "Bless" for e in player.effects)
        assert mock_character.concentration_spell_id is None
        assert char_repo.concentration_calls == [(mock_character.id, None)]


class TestConcentrationRecast:
    """Casting a NEW concentration spell while already concentrating.

    Regression pin: ``manager.break_concentration`` strips ALL concentration
    CombatEffects sourced by the caster, so the old-spell break must run
    BEFORE the new spell's mechanical execution. When it ran after, the
    save-branch's just-applied effect (Hold Person -> Paralyzed) was
    silently deleted while the slot stayed spent and ``conditions_applied``
    still reported PARALYZED.
    """

    async def test_recast_keeps_new_effect_and_strips_old(
        self, mock_character, unique_channel_id, monkeypatch, equipped_longsword
    ):
        from dnd_bot.models import AbilityScore, SpellSlots

        # A WIS caster (DC 8 + prof 3 + WIS 1 = 12) concentrating on Bless.
        mock_character.spellcasting_ability = AbilityScore.WISDOM
        mock_character.spell_slots = SpellSlots(level_2=(3, 3))
        mock_character.prepared_spells = ["hold-person"]
        mock_character.concentration_spell_id = "bless"

        manager = _make_combat(unique_channel_id, mock_character)
        player = next(c for c in manager.combat.combatants if c.is_player)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)
        # Manager-side footprint of the OLD concentration (self-Bless).
        player.effects.append(
            CombatEffect(
                name="Bless",
                effect_type="buff",
                source_combatant_id=player.id,
                is_concentration=True,
            )
        )
        session = _make_session(unique_channel_id, mock_character)
        # Goblin's WIS save: roll_check face 10 + mod 0 = 10 < DC 12: FAILED.
        roller = _ScriptedRoller()
        coordinator = _coordinator(manager, session, roller)

        # SpellcastingManager reads the SRD through its own seam.
        monkeypatch.setattr(
            "dnd_bot.game.magic.spellcasting.get_srd", lambda: _StubSRD()
        )
        char_repo = _FakeCharacterRepo()

        async def _get_repo():
            return char_repo

        monkeypatch.setattr(
            "dnd_bot.game.combat.coordinator.get_character_repo", _get_repo
        )

        await coordinator.start_turn(player)
        result = await coordinator.execute_action(
            CombatAction(
                action_type=CombatActionType.CAST_SPELL,
                combatant_id=player.id,
                target_ids=[goblin.id],
                spell_index="hold-person",
                slot_level=2,
            )
        )

        assert result.success is True
        # Old concentration (Bless) is broken end-to-end...
        assert result.concentration_broken is True
        assert all(e.name != "Bless" for e in player.effects)
        # ...and the NEW spell's just-applied effect SURVIVES. (The
        # regression: breaking AFTER the save branch deleted it, leaving
        # the slot spent and the goblin un-paralyzed.)
        assert any(
            e.source_spell_index == "hold-person"
            and e.condition == Condition.PARALYZED
            for e in goblin.effects
        )
        assert result.conditions_applied == {goblin.id: [Condition.PARALYZED]}
        # New concentration is set + persisted; the slot stayed spent.
        assert mock_character.concentration_spell_id == "hold-person"
        assert char_repo.concentration_calls[-1] == (
            mock_character.id,
            "hold-person",
        )
        assert mock_character.spell_slots.get_slots(2)[0] == 2
        assert char_repo.slot_calls == [(mock_character.id, 2, 2)]


class TestSpellConditionMap:
    def test_command_applies_no_automatic_condition(self):
        """C11: 'command' must NOT auto-apply Prone on a failed save — only
        the Grovel variant knocks prone (1 of 6 SRD variants), and
        CombatAction carries no variant argument to gate on. The outcome is
        the narrator's to describe. The map is consulted generically in
        _execute_cast_spell, so its contents ARE the behavior."""
        from dnd_bot.game.combat.coordinator import SPELL_CONDITION_MAP

        assert "command" not in SPELL_CONDITION_MAP
        # Canary: unambiguous entries are untouched.
        assert SPELL_CONDITION_MAP["hold-person"][0] == Condition.PARALYZED


class TestActionEconomyRefund:
    """AQ-ERR-03: an action that never happened gives its economy back.

    The refund keys on the handler contract: success=False WITH an error
    string is a precondition rejection (no dice, no damage, no slot), so
    the action/bonus/reaction comes back and a retry works; success=False
    with NO error is an outcome failure (a miss, a failed stealth roll)
    and the cost stays. Pre-fix, economy was consumed before routing and
    never refunded - a targeting mistake or handler crash burned the
    whole turn, and a failed cast burned the spell slot too.
    """

    def _setup(self, channel_id, character):
        manager = _make_combat(channel_id, character)
        player = next(c for c in manager.combat.combatants if c.is_player)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)
        session = _make_session(channel_id, character)
        return manager, player, goblin, session

    async def test_attack_at_unknown_target_refunds_the_action(
        self, mock_character, unique_channel_id
    ):
        manager, player, goblin, session = self._setup(
            unique_channel_id, mock_character
        )
        coordinator = _coordinator(manager, session, _ScriptedRoller())

        result = await coordinator.execute_action(
            _attack(player.id, "no-such-combatant")
        )

        assert result.success is False
        assert result.error == "Target not found"
        # Pre-fix the action was consumed before routing, so this turn was
        # spent on a targeting mistake and the retry died with
        # "No action available".
        assert player.turn_resources.action is True

    async def test_missed_attack_keeps_the_action_spent(
        self, mock_character, unique_channel_id, equipped_longsword
    ):
        manager, player, goblin, session = self._setup(
            unique_channel_id, mock_character
        )
        # Natural 1: guaranteed miss that genuinely executed.
        coordinator = _coordinator(
            manager, session, _ScriptedRoller(attack_faces=[1])
        )

        result = await coordinator.execute_action(_attack(player.id, goblin.id))

        assert result.success is False
        assert not result.error
        assert player.turn_resources.action is False

    async def test_handler_crash_refunds_the_action(
        self, mock_character, unique_channel_id, monkeypatch
    ):
        manager, player, goblin, session = self._setup(
            unique_channel_id, mock_character
        )
        coordinator = _coordinator(manager, session, _ScriptedRoller())

        async def _boom(action, combatant):
            raise RuntimeError("handler exploded")

        monkeypatch.setattr(coordinator, "_execute_attack", _boom)

        result = await coordinator.execute_action(_attack(player.id, goblin.id))

        assert result.success is False
        assert "handler exploded" in (result.error or "")
        assert player.turn_resources.action is True

    async def test_bonus_action_rejection_refunds_the_consumed_action(
        self, mock_character, unique_channel_id
    ):
        manager, player, goblin, session = self._setup(
            unique_channel_id, mock_character
        )
        coordinator = _coordinator(manager, session, _ScriptedRoller())
        player.turn_resources.bonus_action = False  # already spent

        result = await coordinator.execute_action(CombatAction(
            action_type=CombatActionType.DASH,
            combatant_id=player.id,
            uses_action=True,
            uses_bonus_action=True,
        ))

        assert result.error == "No bonus action available"
        # The action consumed a moment earlier comes back with it.
        assert player.turn_resources.action is True

    async def test_precondition_rejection_does_not_burn_recharge(
        self, mock_character, unique_channel_id
    ):
        from dnd_bot.models.combat import RechargeAbility

        manager, player, goblin, session = self._setup(
            unique_channel_id, mock_character
        )
        goblin.recharge_abilities.append(RechargeAbility(name="Fire Breath"))
        coordinator = _coordinator(manager, session, _ScriptedRoller())

        result = await coordinator.execute_action(CombatAction(
            action_type=CombatActionType.ATTACK,
            combatant_id=goblin.id,
            target_ids=[],
            ability_name="Fire Breath",
        ))

        assert result.success is False
        assert result.error == "No target specified"
        # Pre-fix the 5-6 recharge was marked used on a rejected attack.
        assert goblin.get_recharge_ability("Fire Breath").is_available is True

    async def test_missed_ability_attack_still_burns_recharge(
        self, mock_character, unique_channel_id
    ):
        from dnd_bot.models.combat import RechargeAbility

        manager, player, goblin, session = self._setup(
            unique_channel_id, mock_character
        )
        goblin.recharge_abilities.append(RechargeAbility(name="Fire Breath"))
        coordinator = _coordinator(
            manager, session, _ScriptedRoller(attack_faces=[1])
        )

        result = await coordinator.execute_action(CombatAction(
            action_type=CombatActionType.ATTACK,
            combatant_id=goblin.id,
            target_ids=[player.id],
            ability_name="Fire Breath",
        ))

        # A miss is an executed action: the breath was used, it just missed.
        assert result.success is False
        assert not result.error
        assert goblin.get_recharge_ability("Fire Breath").is_available is False

    async def test_failed_cast_leaves_slot_and_concentration_intact(
        self, mock_character, unique_channel_id, monkeypatch
    ):
        from dnd_bot.models import AbilityScore, SpellSlots

        mock_character.spellcasting_ability = AbilityScore.WISDOM
        mock_character.spell_slots = SpellSlots(level_2=(3, 3))
        mock_character.prepared_spells = ["hold-person"]
        mock_character.concentration_spell_id = "bless"

        manager, player, goblin, session = self._setup(
            unique_channel_id, mock_character
        )
        player.effects.append(CombatEffect(
            name="Bless",
            effect_type="buff",
            source_combatant_id=player.id,
            is_concentration=True,
        ))
        coordinator = _coordinator(manager, session, _ScriptedRoller())
        monkeypatch.setattr(
            "dnd_bot.game.magic.spellcasting.get_srd", lambda: _StubSRD()
        )
        char_repo = _FakeCharacterRepo()

        async def _get_repo():
            return char_repo

        monkeypatch.setattr(
            "dnd_bot.game.combat.coordinator.get_character_repo", _get_repo
        )

        result = await coordinator.execute_action(CombatAction(
            action_type=CombatActionType.CAST_SPELL,
            combatant_id=player.id,
            target_ids=["vanished-combatant"],
            spell_index="hold-person",
            slot_level=2,
        ))

        assert result.success is False
        assert result.error == "No valid targets for spell"
        # Pre-fix: the slot was expended AND persisted, the old
        # concentration (Bless) was broken, and a retry burned a second
        # slot. All of it must be untouched.
        assert mock_character.spell_slots.get_slots(2)[0] == 3
        assert char_repo.slot_calls == []
        assert mock_character.concentration_spell_id == "bless"
        assert any(e.name == "Bless" for e in player.effects)
        assert player.turn_resources.action is True

    async def test_attack_spell_with_no_target_keeps_the_slot(
        self, mock_character, unique_channel_id, monkeypatch
    ):
        from types import SimpleNamespace as _NS

        from dnd_bot.game.magic.spellcasting import SpellcastingManager
        from dnd_bot.models import SpellSlots

        mock_character.spell_slots = SpellSlots(level_2=(3, 3))
        stub_bolt = _NS(
            name="Stub Bolt",
            level=2,
            attack_type="ranged",
            save_dc_ability=None,
            concentration=False,
            heal_at_slot_level=None,
        )
        monkeypatch.setattr(
            SpellcastingManager, "get_spell_info", lambda self, idx: stub_bolt
        )
        monkeypatch.setattr(
            SpellcastingManager,
            "can_cast",
            lambda self, character, idx, lvl: (True, ""),
        )

        manager, player, goblin, session = self._setup(
            unique_channel_id, mock_character
        )
        coordinator = _coordinator(manager, session, _ScriptedRoller())
        char_repo = _FakeCharacterRepo()

        async def _get_repo():
            return char_repo

        monkeypatch.setattr(
            "dnd_bot.game.combat.coordinator.get_character_repo", _get_repo
        )

        result = await coordinator.execute_action(CombatAction(
            action_type=CombatActionType.CAST_SPELL,
            combatant_id=player.id,
            target_ids=[],
            spell_index="stub-bolt",
            slot_level=2,
        ))

        assert result.success is False
        assert result.error == "No target for attack spell"
        assert mock_character.spell_slots.get_slots(2)[0] == 3
        assert char_repo.slot_calls == []
        assert player.turn_resources.action is True

    async def test_crash_inside_cast_leaves_the_slot_intact(
        self, mock_character, unique_channel_id, monkeypatch
    ):
        from types import SimpleNamespace as _NS

        from dnd_bot.game.magic.spellcasting import SpellcastingManager
        from dnd_bot.models import SpellSlots

        mock_character.spell_slots = SpellSlots(level_1=(2, 2))
        stub_cure = _NS(
            name="Stub Cure",
            level=1,
            attack_type=None,
            save_dc_ability=None,
            concentration=False,
            heal_at_slot_level={1: "1d8"},
        )
        monkeypatch.setattr(
            SpellcastingManager, "get_spell_info", lambda self, idx: stub_cure
        )
        monkeypatch.setattr(
            SpellcastingManager,
            "can_cast",
            lambda self, character, idx, lvl: (True, ""),
        )

        def _crash(self, **kwargs):
            raise ValueError("Invalid dice notation")

        monkeypatch.setattr(SpellcastingManager, "cast_healing_spell", _crash)

        manager, player, goblin, session = self._setup(
            unique_channel_id, mock_character
        )
        coordinator = _coordinator(manager, session, _ScriptedRoller())
        char_repo = _FakeCharacterRepo()

        async def _get_repo():
            return char_repo

        monkeypatch.setattr(
            "dnd_bot.game.combat.coordinator.get_character_repo", _get_repo
        )

        result = await coordinator.execute_action(CombatAction(
            action_type=CombatActionType.CAST_SPELL,
            combatant_id=player.id,
            target_ids=[player.id],
            spell_index="stub-cure",
            slot_level=1,
        ))

        assert result.success is False
        assert "Invalid dice notation" in (result.error or "")
        # The expend runs AFTER the mechanical branches, so a crash inside
        # a cast_* call leaves the slot untouched — which is what makes the
        # crash-path action refund honest. Pre-reorder, the slot was
        # expended AND persisted first, so the refunded action invited
        # same-turn retries that each drained another slot for zero effect.
        assert mock_character.spell_slots.get_slots(1)[0] == 2
        assert char_repo.slot_calls == []
        assert player.turn_resources.action is True
