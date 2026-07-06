"""Pin the combat narration stack THROUGH the coordinator (pin-first rule).

``NarratorBrain.narrate_outcome`` (driven by ``coordinator.narrate_result``
and ``narrate_turn_results``) is the FOURTH narration path — the one Step 2
deliberately left out of scope. Before it migrates onto the Step-2
NarrationSpec/NarrationStrategy these pins capture, at the narrator-client
seam, exactly what the stack sends and returns today:

- message shape: basic (non-bookend) build — [system persona + ## Party +
  ## Combat State, user player_action, system instruction]
- the player_action decoration: ``[]: <summary>\n\n[MECHANICAL RESULT: …]``
  (the ``[]:`` prefix is real — the coordinator never sets player_name)
- chat kwargs, exact: temperature / max_tokens=1500 / think=False —
  NO tools, NO anti-repetition penalties, NO streaming
- response post-processing: code-fence + PROSE:/INTENTS strip, empty-prose
  fallback
- tool_calls in the response are IGNORED: no effects, and critically NO
  second (followup) LLM call
- narrate_turn_results batches N results into ONE call

Prose itself is never pinned (plan rule) — assertions are on structure,
kwargs, and the decorated inputs.
"""

import pytest

from dnd_bot.game.combat.actions import (
    ActionResult,
    CombatAction,
    CombatActionType,
)
from dnd_bot.game.combat.coordinator import CombatTurnCoordinator
from dnd_bot.game.combat.manager import CombatManager
from dnd_bot.game.mechanics.dice import DiceRoll
from dnd_bot.llm.brains.narrator import NARRATOR_SYSTEM_PROMPT, NarratorBrain
from tests.fakes import ScriptedBrain, narration_response

# Channel ids come from the run-unique ``unique_channel_id`` fixture — the
# combat, coordinator, and turn-lock registries are module-level globals.


def _make_combat(channel_id: int, character) -> CombatManager:
    """Mid-combat encounter with a pinned order: player first, goblin second."""
    manager = CombatManager.create_encounter(
        session_id="combat-narration-test-session",
        channel_id=channel_id,
        name="Combat Narration Test",
    )
    manager.add_player(character)
    manager.add_custom_combatant(name="Goblin", hp=12, ac=13)
    manager.start_combat()
    player = next(c for c in manager.combat.combatants if c.is_player)
    goblin = next(c for c in manager.combat.combatants if not c.is_player)
    player.turn_order = 0
    goblin.turn_order = 1
    manager.combat.current_turn_index = 0  # player is acting
    return manager


def _narrating_coordinator(manager, responses):
    """Coordinator + NarratorBrain wired to a ScriptedBrain client."""
    client = ScriptedBrain(responses)
    narrator = NarratorBrain(client=client)
    coordinator = CombatTurnCoordinator(manager)
    coordinator.set_narrator(narrator)
    return coordinator, narrator, client


def _hit_result(player, goblin) -> ActionResult:
    """A deterministic longsword hit: 18 vs AC, 7 slashing damage."""
    return ActionResult(
        action=CombatAction(
            action_type=CombatActionType.ATTACK,
            combatant_id=player.id,
            target_ids=[goblin.id],
            weapon_index="longsword",
        ),
        success=True,
        attack_roll=DiceRoll(
            notation="1d20", dice_results=[12], kept_dice=[12],
            modifier=6, total=18,
        ),
        damage_dealt={goblin.id: 7},
        damage_type="slashing",
        target_ac=13,
    )


def _dash_result(player) -> ActionResult:
    return ActionResult(
        action=CombatAction(
            action_type=CombatActionType.DASH,
            combatant_id=player.id,
        ),
        success=True,
        zone_changes=["Movement increased by 30ft"],
    )


# The instruction the stack appends after the built messages — spec.prompt
# data once migrated; content must not change.
_OUTCOME_INSTRUCTION = (
    "Narrate the mechanical result above in a dramatic, engaging way. "
    "Do NOT change or add to the mechanical outcome - just describe it vividly. "
    "Output ONLY the narrative prose directly. Do NOT use PROSE:/INTENTS: "
    "format headers or code blocks. Just write the narrative."
)

_COMBAT_SYSTEM_SUFFIX = (
    "\n\n## Party\nTest Hero: 100% HP"
    "\n\n## Combat State\nRound: 1"
    "\nInitiative Order:"
    "\n- Test Hero (HP: 44/44)"
    "\n- Goblin (HP: 12/12)"
    "\nCurrent Turn: Test Hero"
)


class TestNarrateResult:
    async def test_prompt_shape_decoration_and_kwargs(
        self, mock_character, unique_channel_id
    ):
        manager = _make_combat(unique_channel_id, mock_character)
        player = next(c for c in manager.combat.combatants if c.is_player)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)
        coordinator, narrator, client = _narrating_coordinator(
            manager, [narration_response("The blade lands true.")]
        )

        narrative = await coordinator.narrate_result(_hit_result(player, goblin))

        assert narrative == "The blade lands true."
        assert len(client.calls) == 1
        call = client.calls[0]
        assert call["method"] == "chat"  # combat narration never streams

        # Message shape: basic (non-bookend) build + the appended instruction.
        assert [m["role"] for m in call["messages"]] == ["system", "user", "system"]

        # System = persona + ## Party + ## Combat State (no world state,
        # memory, scene, or quests — the coordinator-built context has none).
        assert call["messages"][0]["content"] == (
            NARRATOR_SYSTEM_PROMPT + _COMBAT_SYSTEM_SUFFIX
        )

        # player_action decoration: get_summary() + [MECHANICAL RESULT: …].
        # The "[]:" prefix is real — the coordinator never sets player_name.
        assert call["messages"][1]["content"] == (
            "[]: attack: SUCCESS | Attack 18 = HIT | Damage: 7"
            "\n\n[MECHANICAL RESULT: Test Hero's longsword strikes Goblin. "
            "Hit for 7 slashing damage.]"
        )

        assert call["messages"][2]["content"] == _OUTCOME_INSTRUCTION

        # kwargs, exact: no tools, no anti-repetition penalties (drift vs
        # the orchestrator's three paths — migration verdict pending),
        # thinking disabled (Qwen3 truncation).
        assert call["kwargs"] == {
            "temperature": narrator.temperature,
            "max_tokens": 1500,
            "think": False,
        }

    async def test_tool_calls_ignored_no_followup_call(
        self, mock_character, unique_channel_id
    ):
        """Combat effects are owned by the combat engine: narrator tool
        calls on this path are dropped, and no tool-followup leg fires."""
        manager = _make_combat(unique_channel_id, mock_character)
        player = next(c for c in manager.combat.combatants if c.is_player)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)
        coordinator, _, client = _narrating_coordinator(
            manager,
            [
                narration_response(
                    "Steel bites deep.",
                    tool_calls=[{
                        "name": "update_player",
                        "arguments": {"player_name": "Test Hero", "hp_delta": -3},
                    }],
                )
            ],
        )

        narrative = await coordinator.narrate_result(_hit_result(player, goblin))

        assert narrative == "Steel bites deep."
        assert len(client.calls) == 1  # exactly one call — NO followup leg

    async def test_strips_fences_prose_header_and_intents_block(
        self, mock_character, unique_channel_id
    ):
        manager = _make_combat(unique_channel_id, mock_character)
        player = next(c for c in manager.combat.combatants if c.is_player)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)
        coordinator, _, _ = _narrating_coordinator(
            manager,
            [
                narration_response(
                    "```\nPROSE: The goblin reels backward.\n"
                    "INTENTS: apply_damage \"goblin\" 7\n```"
                )
            ],
        )

        narrative = await coordinator.narrate_result(_hit_result(player, goblin))

        assert narrative == "The goblin reels backward."

    async def test_empty_prose_falls_back_to_placeholder(
        self, mock_character, unique_channel_id
    ):
        manager = _make_combat(unique_channel_id, mock_character)
        player = next(c for c in manager.combat.combatants if c.is_player)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)
        coordinator, _, _ = _narrating_coordinator(
            manager, [narration_response("")]
        )

        narrative = await coordinator.narrate_result(_hit_result(player, goblin))

        assert narrative == "*The action unfolds dramatically...*"


class TestNarrateTurnResults:
    async def test_empty_results_no_call(self, mock_character, unique_channel_id):
        manager = _make_combat(unique_channel_id, mock_character)
        coordinator, _, client = _narrating_coordinator(
            manager, [narration_response("unused")]
        )

        assert await coordinator.narrate_turn_results([]) == ""
        assert client.calls == []

    async def test_single_result_uses_the_standard_path(
        self, mock_character, unique_channel_id
    ):
        manager = _make_combat(unique_channel_id, mock_character)
        player = next(c for c in manager.combat.combatants if c.is_player)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)
        coordinator, _, client = _narrating_coordinator(
            manager, [narration_response("A single stroke.")]
        )

        narrative = await coordinator.narrate_turn_results(
            [_hit_result(player, goblin)]
        )

        assert narrative == "A single stroke."
        assert len(client.calls) == 1
        # Same decoration as narrate_result — not the batched "Action N:" form.
        assert "[MECHANICAL RESULT: Test Hero's longsword strikes Goblin. " in (
            client.calls[0]["messages"][1]["content"]
        )

    async def test_batches_multiple_results_into_one_call(
        self, mock_character, unique_channel_id
    ):
        manager = _make_combat(unique_channel_id, mock_character)
        player = next(c for c in manager.combat.combatants if c.is_player)
        goblin = next(c for c in manager.combat.combatants if not c.is_player)
        coordinator, narrator, client = _narrating_coordinator(
            manager, [narration_response("A flurry of motion.")]
        )

        narrative = await coordinator.narrate_turn_results(
            [_hit_result(player, goblin), _dash_result(player)]
        )

        assert narrative == "A flurry of motion."
        assert len(client.calls) == 1  # ONE batched call, not one per result
        call = client.calls[0]

        assert [m["role"] for m in call["messages"]] == ["system", "user", "system"]
        # Context comes from the LAST result; the combined description lists
        # every action. multiaction outcomes get no extra mechanical suffix.
        assert call["messages"][1]["content"] == (
            "[]: dash: SUCCESS"
            "\n\n[MECHANICAL RESULT: "
            "Action 1: Test Hero's longsword strikes Goblin."
            "\nAction 2: Test Hero dashes, doubling their movement speed.]"
        )
        assert call["messages"][2]["content"] == _OUTCOME_INSTRUCTION
        assert call["kwargs"] == {
            "temperature": narrator.temperature,
            "max_tokens": 1500,
            "think": False,
        }
