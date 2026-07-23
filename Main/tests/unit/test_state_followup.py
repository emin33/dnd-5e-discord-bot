"""Unit tests for extractor-coordinated targeted tool recovery signals.

The detector mirrors the long-form audit's tool-omission observer: it must
fire only when the extractor's claim is literally grounded in narration and
not already covered by a proposed tool, and abstain on every ambiguity.
"""

import pytest

from dnd_bot.game.world_state import NPCState, NPCUpdate, StateDelta, WorldState
from dnd_bot.llm.effects import EffectType, ProposedEffect
from dnd_bot.llm.state_followup import (
    MAX_SIGNALS_PER_TURN,
    uncovered_state_signals,
)


def _world(npcs: dict[str, NPCState] | None = None) -> WorldState:
    return WorldState(npcs=npcs or {})


def _signals(delta, *, before="", narrative="", effects=(), world=None, player=""):
    return uncovered_state_signals(
        delta,
        before_location=before,
        narrative=narrative,
        proposed_effects=list(effects),
        world_state=world or _world(),
        player_name=player,
    )


class TestLocationSignal:
    def test_genuine_move_fires(self):
        delta = StateDelta(location_change="Tallow Rows")
        signals = _signals(
            delta,
            before="Harrow's Drippings - Upstairs",
            narrative="You step out into the grey light of the Tallow Rows.",
        )
        assert [s.kind for s in signals] == ["location"]
        assert signals[0].tool_name == "change_location"
        assert '"Tallow Rows"' in signals[0].instruction

    def test_sub_scene_refinement_abstains(self):
        delta = StateDelta(location_change="Tallow Rows")
        signals = _signals(
            delta,
            before="Tallow Rows alley",
            narrative="You guide him out of the alley into the Tallow Rows.",
        )
        assert signals == []

    def test_covered_by_proposed_change_location(self):
        delta = StateDelta(location_change="Tallow Rows")
        proposed = ProposedEffect(
            effect_type=EffectType.CHANGE_LOCATION,
            location_name="Tallow Rows",
        )
        signals = _signals(
            delta,
            before="Harrow's Drippings",
            narrative="You walk into the Tallow Rows.",
            effects=[proposed],
        )
        assert signals == []

    def test_name_absent_from_narration_abstains(self):
        delta = StateDelta(location_change="Tallow Rows")
        signals = _signals(
            delta,
            before="Harrow's Drippings",
            narrative="You descend the stairs into the dim shop below.",
        )
        assert signals == []

    def test_sentence_label_abstains(self):
        delta = StateDelta(
            location_change="the narrow street behind the rendering district"
        )
        signals = _signals(
            delta,
            before="Harrow's Drippings",
            narrative=(
                "You slip into the narrow street behind the rendering "
                "district."
            ),
        )
        assert signals == []


class TestNewNpcSignal:
    def test_materialized_proper_npc_requests_ref(self):
        npc = NPCState(name="Lena Harker")
        delta = StateDelta(new_npcs=[npc])
        world = _world({npc.id: npc})
        signals = _signals(
            delta,
            narrative="A woman introduces herself: Lena Harker, at your service.",
            world=world,
        )
        assert [s.kind for s in signals] == ["new_npc"]
        assert signals[0].tool_name == "ref_entity"
        assert npc.id in signals[0].instruction

    def test_generic_label_abstains(self):
        npc = NPCState(name="the hooded figure")
        delta = StateDelta(new_npcs=[npc])
        signals = _signals(
            delta,
            narrative="The hooded figure watches you from the doorway.",
            world=_world({npc.id: npc}),
        )
        assert signals == []

    def test_covered_by_ref_alias(self):
        npc = NPCState(name="Lena Harker")
        delta = StateDelta(new_npcs=[npc])
        ref = ProposedEffect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id="some-other-id",
            ref_alias_used="Lena",
        )
        signals = _signals(
            delta,
            narrative="Lena Harker bars the door behind you.",
            effects=[ref],
            world=_world({npc.id: npc}),
        )
        assert signals == []

    def test_covered_by_add_npc_name(self):
        npc = NPCState(name="Lena Harker")
        delta = StateDelta(new_npcs=[npc])
        add = ProposedEffect(
            effect_type=EffectType.ADD_NPC,
            npc_name="Lena Harker",
        )
        signals = _signals(
            delta,
            narrative="Lena Harker bars the door behind you.",
            effects=[add],
            world=_world({npc.id: npc}),
        )
        assert signals == []

    def test_unmaterialized_npc_abstains(self):
        npc = NPCState(name="Lena Harker")
        delta = StateDelta(new_npcs=[npc])
        signals = _signals(
            delta,
            narrative="Lena Harker bars the door behind you.",
            world=_world({}),  # rejected/never applied
        )
        assert signals == []

    def test_name_not_in_prose_abstains(self):
        npc = NPCState(name="Lena Harker")
        delta = StateDelta(new_npcs=[npc])
        signals = _signals(
            delta,
            narrative="The shopkeeper's sister watches from the stairs.",
            world=_world({npc.id: npc}),
        )
        assert signals == []

    def test_player_name_abstains(self):
        npc = NPCState(name="Kael")
        delta = StateDelta(new_npcs=[npc])
        signals = _signals(
            delta,
            narrative="Kael steps into the light.",
            world=_world({npc.id: npc}),
            player="Kael",
        )
        assert signals == []

    def test_merged_id_resolves_by_name(self):
        # The delta NPC was merged onto a canonical roster id before apply;
        # resolution must find the applied entity by name, not the stale id.
        roster_npc = NPCState(name="Lena Harker")
        delta_npc = NPCState(name="Lena Harker")  # different uuid
        delta = StateDelta(new_npcs=[delta_npc])
        signals = _signals(
            delta,
            narrative="Lena Harker bars the door.",
            world=_world({roster_npc.id: roster_npc}),
        )
        assert len(signals) == 1
        assert roster_npc.id in signals[0].instruction


class TestNpcUpdateSignal:
    def test_death_requires_update_entity(self):
        npc = NPCState(name="Doran")
        delta = StateDelta(npc_updates=[NPCUpdate(id=npc.id, alive=False)])
        signals = _signals(
            delta,
            narrative="Doran slumps against the wall and does not move again.",
            world=_world({npc.id: npc}),
        )
        assert [s.kind for s in signals] == ["npc_update"]
        assert signals[0].tool_name == "update_entity"
        assert npc.id in signals[0].instruction

    def test_covered_by_proposed_update(self):
        npc = NPCState(name="Doran")
        delta = StateDelta(npc_updates=[NPCUpdate(id=npc.id, alive=False)])
        update = ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id=npc.id,
        )
        signals = _signals(
            delta,
            narrative="Doran slumps against the wall.",
            effects=[update],
            world=_world({npc.id: npc}),
        )
        assert signals == []

    def test_disposition_needs_narration_grammar(self):
        npc = NPCState(name="Doran")
        delta = StateDelta(
            npc_updates=[NPCUpdate(id=npc.id, disposition="friendly")]
        )
        # No becomes/turns/now/swears cue near a disposition word.
        signals = _signals(
            delta,
            narrative="Doran hands you the bottle without a word.",
            world=_world({npc.id: npc}),
        )
        assert signals == []

    def test_disposition_shift_with_grammar_fires(self):
        npc = NPCState(name="Doran")
        delta = StateDelta(
            npc_updates=[NPCUpdate(id=npc.id, disposition="friendly")]
        )
        signals = _signals(
            delta,
            narrative="Something eases in his face; Doran becomes a friend.",
            world=_world({npc.id: npc}),
        )
        assert [s.kind for s in signals] == ["npc_update"]

    def test_same_turn_add_npc_covers_update(self):
        npc = NPCState(name="Lena Harker")
        delta = StateDelta(
            npc_updates=[
                NPCUpdate(name="Lena Harker", add_inventory=["ledger"])
            ]
        )
        add = ProposedEffect(
            effect_type=EffectType.ADD_NPC,
            npc_name="Lena Harker",
        )
        signals = _signals(
            delta,
            narrative="Lena Harker takes the ledger.",
            effects=[add],
            world=_world({npc.id: npc}),
        )
        assert signals == []


class TestSignalCap:
    def test_capped_per_turn(self):
        npcs = {}
        new_npcs = []
        names = ["Aldric Vane", "Bess Marrow", "Corvin Ash", "Dela Thorn",
                 "Edmun Grey", "Fara Quill"]
        for name in names:
            npc = NPCState(name=name)
            npcs[npc.id] = npc
            new_npcs.append(npc)
        delta = StateDelta(new_npcs=new_npcs)
        narrative = "They arrive together: " + ", ".join(names) + "."
        signals = _signals(delta, narrative=narrative, world=_world(npcs))
        assert len(signals) == MAX_SIGNALS_PER_TURN


class TestTargetedStateFollowupMethod:
    @pytest.mark.asyncio
    async def test_returns_structurally_valid_effects(self):
        from dnd_bot.llm.narration import NarrationStrategy
        from dnd_bot.llm.state_followup import StateFollowupSignal

        class _Response:
            def __init__(self, tool_calls):
                self.tool_calls = tool_calls
                self.content = ""

        class _Client:
            def __init__(self):
                self.requests = []

            async def chat(self, messages, **kwargs):
                self.requests.append((messages, kwargs))
                return _Response([
                    {
                        "name": "change_location",
                        "arguments": {"location_name": "Tallow Rows"},
                    },
                    # Off-menu call must be filtered out.
                    {
                        "name": "add_npc",
                        "arguments": {"name": "Invented Person"},
                    },
                ])

        class _Narrator:
            def __init__(self):
                self.client = _Client()

        strategy = NarrationStrategy.__new__(NarrationStrategy)
        strategy.last_diagnostics = {}
        narrator = _Narrator()
        strategy._get_narrator = lambda: narrator
        strategy._get_tools = lambda: [
            {"function": {"name": "change_location"}},
            {"function": {"name": "add_npc"}},
        ]

        signals = [
            StateFollowupSignal(
                kind="location",
                tool_name="change_location",
                instruction=(
                    'The party moved to "Tallow Rows". Call '
                    'change_location(location_name="Tallow Rows").'
                ),
            )
        ]
        effects = await strategy.targeted_state_followup(
            "You step into the Tallow Rows.", signals
        )
        assert [e.effect_type for e in effects] == [EffectType.CHANGE_LOCATION]
        assert effects[0].location_name == "Tallow Rows"
        assert strategy.last_diagnostics["state_followup_attempted"] is True
        assert strategy.last_diagnostics["state_followup_effects"] == 1
        # Only the signalled tool family was advertised.
        _, kwargs = narrator.client.requests[0]
        advertised = {
            tool["function"]["name"] for tool in kwargs["tools"]
        }
        assert advertised == {"change_location"}

    @pytest.mark.asyncio
    async def test_no_signals_no_request(self):
        from dnd_bot.llm.narration import NarrationStrategy

        strategy = NarrationStrategy.__new__(NarrationStrategy)
        strategy.last_diagnostics = {}
        assert await strategy.targeted_state_followup("prose", []) == []
