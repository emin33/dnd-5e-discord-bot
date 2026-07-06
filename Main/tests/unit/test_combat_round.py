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


class _StubSRD:
    """Serves exactly the two canned entries the tests reference."""

    def get_equipment(self, index):
        return _LONGSWORD if index == "longsword" else None

    def get_monster(self, index):
        return _GOBLIN if index == "goblin" else None


class _ScriptedRoller:
    """DiceRoller stand-in: scripted d20 faces / damage sums, recorded calls.

    ``attack_faces`` are raw d20 faces — total = face + the modifier the
    coordinator computed, so modifier math stays real and pinnable.
    ``damage_sums`` are the dice-only sums — the coordinator adds the
    ability modifier on top, so damage math stays real and pinnable.
    """

    def __init__(self, attack_faces=(), damage_sums=()):
        self.attack_faces = list(attack_faces)
        self.damage_sums = list(damage_sums)
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
        return DiceRoll(
            notation="1d20", dice_results=[10], kept_dice=[10],
            modifier=modifier, total=10 + modifier,
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

    async def test_unarmed_hit_is_broken_by_its_damage_notation(
        self, mock_character, unique_channel_id, monkeypatch
    ):
        """PINNED BROKEN: the unarmed-strike fallback declares damage_dice
        '1', which DiceRoller rejects ('1' is not dice notation), so every
        unarmed HIT dies in _execute_action_locked's except and comes back
        as a failed action WITH the action already consumed. Flips when the
        fallback gets a rollable notation (or flat damage handling):
        success True + 1 bludgeoning damage -> hp 11. NOTE: the fallback
        exists in TWO places — _get_weapon_for_attack (pinned here) and
        _get_equipped_weapons — fix both.
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

        await coordinator.start_turn(player)
        result = await coordinator.execute_action(_attack(player.id, goblin.id))

        assert result.success is False                      # -> True when fixed
        assert "Invalid dice notation: 1" in (result.error or "")  # -> None
        assert goblin.hp_current == 12                      # -> 11 when fixed
        assert player.turn_resources.action is False  # spent, then it crashed

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
