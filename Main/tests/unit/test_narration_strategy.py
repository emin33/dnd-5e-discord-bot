"""Unit tests for the single narration path (REFACTOR_PLAN.md Step 2).

NarrationStrategy is exercised in isolation with fake collaborators — the
integration pins in tests/integration/test_process_action.py cover the real
wiring through process_action; here we pin the strategy's own contract:

- context union via dataclasses.replace (only player_action/player_name
  overridden; every other field carried untouched),
- bookend-vs-basic builder branch on world_state_yaml,
- per-path prompt + tool reminder appended (in that order, last),
- exact chat / chat_stream / followup kwargs,
- the followup policy (runs only when the primary returned no effects),
- the empty-prose policies (bail-with-fallback vs substitute-and-continue),
- the truncated-ending ellipsis fix.

No prose is asserted anywhere; scripted fakes only.
"""

from __future__ import annotations

from dataclasses import replace

import pytest

from dnd_bot.game.world_state import NPCState
from dnd_bot.llm.brains.base import BrainContext
from dnd_bot.llm.continuity import NarrativeGovernance
from dnd_bot.llm.effects import EffectType, ProposedEffect
from dnd_bot.llm.narration import (
    NarrationSpec,
    NarrationStrategy,
    strip_repair_meta_preamble,
)
from dnd_bot.llm.narrator_tools import tool_calls_to_effects

from tests.fakes import ScriptedBrain, narration_response


# ── Fakes ─────────────────────────────────────────────────────────────────────

def _extract(response, action):
    """Minimal stand-in for the orchestrator's _extract_prose_and_effects:
    tool calls become effects; content is prose (INTENTS fallback not needed
    here — that branch belongs to the extractor's own tests)."""
    content = (response.content or "").strip()
    if response.tool_calls:
        return content, tool_calls_to_effects(response.tool_calls)
    return content, []


class _RecordingNarrator:
    """Stands in for NarratorBrain: builders record the context they got and
    return a distinguishable message stack."""

    def __init__(self, client, temperature=0.55):
        self.client = client
        self.temperature = temperature
        self.bookend_contexts: list[BrainContext] = []
        self.basic_contexts: list[BrainContext] = []

    def build_bookend_messages(self, context):
        self.bookend_contexts.append(context)
        return [
            {"role": "system", "content": "PERSONA"},
            {
                "role": "user",
                "content": f"<player_action>[{context.player_name}]: "
                           f"{context.player_action}</player_action>",
            },
        ]

    def build_messages(self, context):
        self.basic_contexts.append(context)
        return [{"role": "system", "content": "PERSONA-BASIC"}]


class _ChatOnlyClient:
    """A client WITHOUT chat_stream — streaming must fall back to chat."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.calls: list[dict] = []

    async def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs, "method": "chat"})
        return self._responses[len(self.calls) - 1]


class _FlakyFollowupClient:
    """First chat succeeds; the second (the followup) raises."""

    def __init__(self, first_response):
        self._first = first_response
        self.calls: list[dict] = []

    async def chat(self, messages, **kwargs):
        self.calls.append({"messages": messages, "kwargs": kwargs, "method": "chat"})
        if len(self.calls) > 1:
            raise RuntimeError("followup provider exploded")
        return self._first


class _Harness:
    """Strategy + recording collaborators, mirroring the orchestrator wiring."""

    def __init__(self, client, on_token=None, governance=None):
        self.narrator = _RecordingNarrator(client)
        self.select_calls: list[tuple] = []
        self.tools = [{"type": "function", "function": {"name": "ref_entity"}}]

        def _select(action, triage, context):
            self.select_calls.append((action, triage, context))
            return self.narrator.client

        def _reminder(messages):
            messages.append({"role": "system", "content": "TOOL-REMINDER"})

        self.strategy = NarrationStrategy(
            get_narrator=lambda: self.narrator,
            select_client=_select,
            get_tools=lambda: self.tools,
            append_tool_reminder=_reminder,
            extract_prose_and_effects=_extract,
            get_on_token=lambda: on_token,
            get_governance=(lambda: governance) if governance else None,
        )


def _spec(**overrides) -> NarrationSpec:
    base = dict(
        action="I greet the barkeep",
        player_name="Elara",
        player_action="I greet the barkeep\n\n[NARRATIVE DIRECTION: calm]",
        prompt="###INSTRUCTION###\nNarrate.",
        prompt_role="system",
    )
    base.update(overrides)
    return NarrationSpec(**base)


def _context(**overrides) -> BrainContext:
    base = dict(
        campaign_id="camp",
        session_id="sess",
        party_members="party",
        current_scene="scene",
        active_quests="quests",
        memory_context="memory",
        message_history=[{"role": "user", "content": "history"}],
        session_summary="summary",
        character_stats="stats",
        world_state_yaml=(
            "location: tavern\n"
            "npcs_here:\n"
            "- id: barkeep\n"
            "  name: barkeep\n"
            "  type: npc\n"
        ),
        kg_context_yaml="kg",
        narrative_memory="past prose",
        last_turn_trace="trace",
        player_action="I greet the barkeep",
        player_name="Elara",
    )
    base.update(overrides)
    return BrainContext(**base)


_REF_TOOL_CALL = {"name": "ref_entity", "arguments": {"entity_id": "barkeep"}}


def _add_npc_tool_call(name: str) -> dict:
    return {
        "name": "add_npc",
        "arguments": {
            "npc_id": "stranger_1",
            "name": name,
            "disposition": "neutral",
            "gender": "female",
            "description": "A watchful woman with close-cropped dark hair.",
        },
    }


# ── Context union + message assembly ─────────────────────────────────────────

@pytest.mark.asyncio
async def test_union_context_message_order_and_chat_kwargs():
    client = ScriptedBrain([
        narration_response("The barkeep nods.", tool_calls=[_REF_TOOL_CALL]),
    ])
    h = _Harness(client)
    spec, context = _spec(), _context()

    prose, effects = await h.strategy.run(spec, context, triage="TRIAGE")

    assert prose == "The barkeep nods."
    assert [e.effect_type for e in effects] == [EffectType.REF_ENTITY]

    # Tier selection: once, with the RAW action and the ORIGINAL context.
    assert h.select_calls == [("I greet the barkeep", "TRIAGE", context)]

    # The builder received the FULL union: replace() carries every upstream
    # field; only player_action/player_name are overridden.
    assert h.narrator.bookend_contexts == [
        replace(context, player_action=spec.player_action, player_name="Elara")
    ]
    assert h.narrator.basic_contexts == []

    # Message order: built stack, then the per-path prompt, then the tool
    # reminder LAST (freshest instruction in the attention window).
    msgs = client.calls[0]["messages"]
    assert msgs[0] == {"role": "system", "content": "PERSONA"}
    assert msgs[-2] == {"role": "system", "content": "###INSTRUCTION###\nNarrate."}
    assert msgs[-1] == {"role": "system", "content": "TOOL-REMINDER"}

    # Exact primary chat contract.
    assert client.calls[0]["method"] == "chat"
    assert client.calls[0]["kwargs"] == {
        "temperature": 0.55,
        "max_tokens": 1500,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.3,
        "tools": h.tools,
        "tool_choice": "auto",
    }


@pytest.mark.asyncio
async def test_basic_builder_when_no_world_state():
    client = ScriptedBrain([
        narration_response("Words happen.", tool_calls=[_REF_TOOL_CALL]),
    ])
    h = _Harness(client)

    await h.strategy.run(_spec(), _context(world_state_yaml=""), triage=None)

    assert h.narrator.bookend_contexts == []
    assert len(h.narrator.basic_contexts) == 1
    assert client.calls[0]["messages"][0] == {
        "role": "system", "content": "PERSONA-BASIC",
    }


@pytest.mark.asyncio
async def test_prompt_role_user_for_mechanical_results():
    client = ScriptedBrain([
        narration_response("Coin changes hands.", tool_calls=[_REF_TOOL_CALL]),
    ])
    h = _Harness(client)

    await h.strategy.run(
        _spec(prompt="The player attempted…", prompt_role="user"),
        _context(),
        triage=None,
    )

    assert client.calls[0]["messages"][-2] == {
        "role": "user", "content": "The player attempted…",
    }


# ── Streaming policy ──────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_streaming_when_spec_allows_and_callback_wired():
    client = ScriptedBrain([
        narration_response("The hearth crackles softly."),   # streamed leg
        narration_response("", tool_calls=[_REF_TOOL_CALL]),  # followup leg
    ])
    tokens: list[str] = []

    async def on_token(t):
        tokens.append(t)

    h = _Harness(client, on_token=on_token)
    prose, effects = await h.strategy.run(_spec(allow_streaming=True), _context(), None)

    stream, followup = client.calls
    assert stream["method"] == "chat_stream"
    assert tokens == ["The hearth crackles softly."]
    # Streaming carries NO tools kwargs at all — the pinned hole the
    # followup leg exists to cover.
    assert stream["kwargs"] == {
        "temperature": 0.55,
        "max_tokens": 1500,
        "frequency_penalty": 0.4,
        "presence_penalty": 0.3,
    }
    assert followup["method"] == "chat"
    assert [e.effect_type for e in effects] == [EffectType.REF_ENTITY]
    assert prose == "The hearth crackles softly."


@pytest.mark.asyncio
async def test_no_streaming_when_spec_disallows():
    client = ScriptedBrain([
        narration_response("Quiet.", tool_calls=[_REF_TOOL_CALL]),
    ])
    tokens: list[str] = []

    async def on_token(t):
        tokens.append(t)

    h = _Harness(client, on_token=on_token)
    await h.strategy.run(_spec(allow_streaming=False), _context(), None)

    assert client.calls[0]["method"] == "chat"
    assert tokens == []


@pytest.mark.asyncio
async def test_no_streaming_when_client_lacks_chat_stream():
    client = _ChatOnlyClient([
        narration_response("Plain chat.", tool_calls=[_REF_TOOL_CALL]),
    ])

    async def on_token(t):  # pragma: no cover - must never fire
        raise AssertionError("token callback must not be used")

    h = _Harness(client, on_token=on_token)
    await h.strategy.run(_spec(allow_streaming=True), _context(), None)

    assert client.calls[0]["method"] == "chat"
    assert "tools" in client.calls[0]["kwargs"]


# ── Followup policy ───────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_general_tool_followup_hides_unobligated_npc_creation():
    client = ScriptedBrain([
        narration_response("The injured courier slumps against the gate."),
        narration_response("", tool_calls=[_REF_TOOL_CALL]),
    ])
    h = _Harness(client)
    h.tools = [
        {"type": "function", "function": {"name": "ref_entity"}},
        {"type": "function", "function": {"name": "add_npc"}},
        {"type": "function", "function": {"name": "update_entity"}},
    ]

    await h.strategy.run(_spec(), _context(), None)

    followup_names = {
        tool["function"]["name"] for tool in client.calls[1]["kwargs"]["tools"]
    }
    assert "add_npc" not in followup_names
    assert {"ref_entity", "update_entity"}.issubset(followup_names)


@pytest.mark.asyncio
async def test_general_followup_drops_unadvertised_npc_creation_call():
    client = ScriptedBrain([
        narration_response("The injured courier slumps against the gate."),
        narration_response("", tool_calls=[{
            "name": "add_npc",
            "arguments": {"name": "Sorra Venn", "disposition": "neutral"},
        }]),
    ])
    h = _Harness(client)
    h.tools = [
        {"type": "function", "function": {"name": "ref_entity"}},
        {"type": "function", "function": {"name": "add_npc"}},
    ]

    _, effects = await h.strategy.run(_spec(), _context(), None)

    assert effects == []
    assert h.strategy.last_diagnostics["tool_policy_suppressed_effects"] == 1
    assert h.strategy.last_diagnostics["tool_invalid_effects_dropped"] == 0


@pytest.mark.asyncio
async def test_followup_drops_reference_id_absent_from_authoritative_roster():
    client = ScriptedBrain([
        narration_response("The gaunt man watches from the alley."),
        narration_response("", tool_calls=[{
            "name": "ref_entity",
            "arguments": {"entity_id": "gaunt_man"},
        }]),
    ])
    h = _Harness(client)

    _, effects = await h.strategy.run(
        _spec(),
        _context(world_state_yaml=(
            "npcs:\n"
            "  - id: mara-id\n"
            "    name: Mara\n"
            "    type: npc\n"
        )),
        None,
    )

    assert effects == []
    assert h.strategy.last_diagnostics["tool_unknown_roster_refs_dropped"] == 1


@pytest.mark.asyncio
async def test_empty_roster_drops_false_existing_npc_claim_but_keeps_mutation():
    client = ScriptedBrain([
        narration_response("The courier's breathing steadies."),
        narration_response("", tool_calls=[
            {
                "name": "ref_entity",
                "arguments": {"entity_id": "courier"},
            },
            {
                "name": "update_entity",
                "arguments": {
                    "entity_id": "courier",
                    "description_addition": "breathing has steadied",
                },
            },
        ]),
    ])
    h = _Harness(client)
    h.tools.append({
        "type": "function",
        "function": {"name": "update_entity"},
    })

    _, effects = await h.strategy.run(
        _spec(action="I examine the collapsed courier."),
        _context(
            player_action="I examine the collapsed courier.",
            world_state_yaml="turn: 1\nphase: exploration\n",
            kg_context_yaml="",
        ),
        None,
    )

    assert [effect.effect_type for effect in effects] == [
        EffectType.UPDATE_ENTITY
    ]
    assert h.strategy.last_diagnostics["tool_unknown_roster_refs_dropped"] == 1


def test_world_location_and_scene_items_are_authoritative_roster_refs():
    rows = NarrationStrategy._roster_refs_from_context(_context(
        world_state_yaml="""
location: The Silver Needle
scene_items:
- 'living-brass-compass: A warm compass with a restless needle.'
""",
        kg_context_yaml="",
    ))

    assert ("the-silver-needle", "The Silver Needle", ()) in rows
    assert ("living-brass-compass", "living-brass-compass", ()) in rows


@pytest.mark.asyncio
async def test_self_introduction_obligation_opens_npc_creation_tool():
    client = ScriptedBrain([
        narration_response('She coughs. "I\'m Elara. Elara Venn."'),
        narration_response("", tool_calls=[{
            "name": "add_npc",
            "arguments": {
                "name": "Elara Venn",
                "description": "An injured courier.",
                "disposition": "neutral",
            },
        }]),
    ])
    h = _Harness(client)
    h.tools = [
        {"type": "function", "function": {"name": "ref_entity"}},
        {"type": "function", "function": {"name": "add_npc"}},
    ]

    _, effects = await h.strategy.run(_spec(), _context(), None)

    followup_names = {
        tool["function"]["name"] for tool in client.calls[1]["kwargs"]["tools"]
    }
    assert "add_npc" in followup_names
    assert [effect.effect_type for effect in effects] == [EffectType.ADD_NPC]


@pytest.mark.asyncio
async def test_followup_skipped_when_primary_returned_effects():
    client = ScriptedBrain([
        narration_response("Done deal.", tool_calls=[_REF_TOOL_CALL]),
    ])
    h = _Harness(client)

    await h.strategy.run(_spec(), _context(), None)

    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_reference_only_state_transition_gets_mutation_followup():
    client = ScriptedBrain([
        narration_response(
            "Mara flees through the kitchen door.",
            tool_calls=[{"name": "ref_entity", "arguments": {"entity_id": "mara"}}],
        ),
        narration_response("", tool_calls=[{
            "name": "update_entity",
            "arguments": {"entity_id": "mara", "status": "fled"},
        }]),
    ])
    h = _Harness(client)

    _, effects = await h.strategy.run(
        _spec(action="I warn Mara and she flees immediately."),
        _context(world_state_yaml="""
npcs_here:
- id: mara
  name: Mara
  type: npc
"""),
        None,
    )

    assert len(client.calls) == 2
    assert [effect.effect_type for effect in effects] == [
        EffectType.REF_ENTITY,
        EffectType.UPDATE_ENTITY,
    ]
    assert h.strategy.last_diagnostics["tool_followup_for_mutation"] is True
    assert "previous calls only referenced entities" in (
        client.calls[1]["messages"][-1]["content"]
    )


@pytest.mark.asyncio
async def test_no_tool_state_transition_gets_mutation_specific_followup():
    client = ScriptedBrain([
        narration_response("Mara becomes your ally and shows you her scar."),
        narration_response("", tool_calls=[{
            "name": "update_entity",
            "arguments": {"entity_id": "mara", "disposition": "allied"},
        }]),
    ])
    h = _Harness(client)

    _, effects = await h.strategy.run(
        _spec(action="Mara becomes my ally and reveals her scar."),
        _context(),
        None,
    )

    assert [effect.effect_type for effect in effects] == [EffectType.UPDATE_ENTITY]
    assert h.strategy.last_diagnostics["tool_followup_for_mutation"] is True


@pytest.mark.asyncio
async def test_historical_recollection_does_not_replay_stale_mutation_tools():
    action = (
        "I call out for Old Bram, though I know he died. I listen for no "
        "living answer and instead recall his last warning about the Ash Gate."
    )
    client = ScriptedBrain([
        narration_response(
            "No answer comes. You remember Old Bram's last warning before he died."
        ),
    ])
    h = _Harness(client)

    prose, effects = await h.strategy.run(
        _spec(action=action, player_action=action),
        _context(player_action=action),
        None,
    )

    assert prose.startswith("No answer comes.")
    assert effects == []
    assert len(client.calls) == 1
    assert h.strategy.last_diagnostics["tool_followup_for_mutation"] is False


@pytest.mark.asyncio
async def test_travel_wording_gets_location_mutation_followup():
    client = ScriptedBrain([
        narration_response("You walk east and enter the Tallow Ward."),
        narration_response("", tool_calls=[{
            "name": "change_location",
            "arguments": {"location_name": "Tallow Ward"},
        }]),
    ])
    h = _Harness(client)

    _, effects = await h.strategy.run(
        _spec(action="I cross the bridge and walk into a distant district."),
        _context(),
        None,
    )

    assert [effect.effect_type for effect in effects] == [EffectType.CHANGE_LOCATION]
    assert h.strategy.last_diagnostics["tool_followup_for_mutation"] is True


@pytest.mark.asyncio
async def test_completed_narrated_travel_gets_narrow_terminal_location_repair():
    client = ScriptedBrain([
        narration_response(
            "You step out into the Tallow Rows and reach a public crossroads.",
            tool_calls=[_REF_TOOL_CALL],
        ),
        narration_response("", tool_calls=[_REF_TOOL_CALL]),
        narration_response("", tool_calls=[{
            "name": "change_location",
            "arguments": {"location_name": "Tallow Rows"},
        }]),
    ])
    h = _Harness(client)
    h.tools.append({
        "type": "function",
        "function": {"name": "change_location"},
    })
    action = (
        "I leave this scene and ask at the nearest public crossroads where "
        "I can find Archivist Valerius, then follow the first credible "
        "direction toward them."
    )

    _, effects = await h.strategy.run(
        _spec(action=action, player_action=action),
        _context(player_action=action),
        None,
    )

    assert [effect.effect_type for effect in effects] == [
        EffectType.REF_ENTITY,
        EffectType.CHANGE_LOCATION,
    ]
    assert len(client.calls) == 3
    terminal_tool_names = {
        tool["function"]["name"] for tool in client.calls[-1]["kwargs"]["tools"]
    }
    assert terminal_tool_names == {"change_location"}
    assert h.strategy.last_diagnostics[
        "effect_obligation_terminal_repair_succeeded"
    ] is True


@pytest.mark.asyncio
async def test_reference_only_nonmutating_dialogue_skips_followup():
    client = ScriptedBrain([
        narration_response(
            "The barkeep answers your question.",
            tool_calls=[_REF_TOOL_CALL],
        ),
    ])
    h = _Harness(client)

    await h.strategy.run(_spec(action="I ask the barkeep a question."), _context(), None)

    assert len(client.calls) == 1
    assert h.strategy.last_diagnostics["tool_followup_for_mutation"] is False


@pytest.mark.asyncio
async def test_followup_reuses_stack_and_uses_followup_kwargs():
    client = ScriptedBrain([
        narration_response("The tavern hums quietly."),       # no tool calls
        narration_response("", tool_calls=[_REF_TOOL_CALL]),   # forced tools
    ])
    h = _Harness(client)

    prose, effects = await h.strategy.run(_spec(), _context(), None)

    primary, followup = client.calls
    n = len(primary["messages"])
    # The followup REUSES the full primary stack (audit #20 contract)…
    assert followup["messages"][:n] == primary["messages"]
    # …then appends the assistant prose + the tool-only instruction.
    assert followup["messages"][n] == {
        "role": "assistant", "content": "The tavern hums quietly.",
    }
    assert followup["messages"][n + 1]["role"] == "user"
    assert followup["messages"][n + 1]["content"].startswith(
        "Now call a tool for everything you narrated above"
    )
    assert len(followup["messages"]) == n + 2

    # Followup kwargs: deterministic, capped, tools REQUIRED, same tier set.
    assert followup["kwargs"] == {
        "temperature": 0,
        "max_tokens": 500,
        "think": False,
        "tools": h.tools,
        "tool_choice": "required",
    }

    assert prose == "The tavern hums quietly."
    assert [e.effect_type for e in effects] == [EffectType.REF_ENTITY]
    assert h.strategy.last_diagnostics["tool_followup_attempted"] is True
    assert h.strategy.last_diagnostics["tool_followup_effects"] == 1
    assert h.strategy.last_diagnostics["final_effects"] == 1


@pytest.mark.asyncio
async def test_followup_failure_swallowed_and_returns_no_effects():
    client = _FlakyFollowupClient(narration_response("The tavern hums quietly."))
    h = _Harness(client)

    prose, effects = await h.strategy.run(_spec(), _context(), None)

    assert len(client.calls) == 2  # the followup WAS attempted
    assert prose == "The tavern hums quietly."
    assert effects == []


@pytest.mark.asyncio
async def test_invalid_primary_tool_arguments_are_repaired_once():
    client = ScriptedBrain([
        narration_response(
            "The barkeep nods.",
            tool_calls=[{"name": "ref_entity", "arguments": {}}],
        ),
        narration_response("", tool_calls=[_REF_TOOL_CALL]),
    ])
    h = _Harness(client)

    prose, effects = await h.strategy.run(
        _spec(),
        _context(),
        None,
    )

    assert prose == "The barkeep nods."
    assert len(client.calls) == 2
    assert effects[0].ref_entity_id == "barkeep"
    assert "previous tool arguments were invalid" in client.calls[1]["messages"][-1]["content"]


@pytest.mark.asyncio
async def test_add_npc_name_must_be_grounded_in_visible_prose():
    client = ScriptedBrain([
        narration_response(
            'The woman says, "Pitfall. That is my name."',
            tool_calls=[_add_npc_tool_call("Sera Vhen")],
        ),
        narration_response("", tool_calls=[_add_npc_tool_call("Pitfall")]),
    ])
    h = _Harness(client)

    prose, effects = await h.strategy.run(
        _spec(),
        _context(),
        None,
    )

    assert prose == 'The woman says, "Pitfall. That is my name."'
    assert len(client.calls) == 2
    assert [effect.npc_name for effect in effects] == ["Pitfall"]
    repair_prompt = client.calls[1]["messages"][-1]["content"]
    assert "npc_name must appear exactly in narrator prose" in repair_prompt


@pytest.mark.asyncio
async def test_add_npc_can_use_full_name_grounded_in_current_player_action():
    client = ScriptedBrain([
        narration_response(
            "Elara closes the ledger and studies you.",
            tool_calls=[_add_npc_tool_call("Elara Vex")],
        ),
    ])
    h = _Harness(client)

    prose, effects = await h.strategy.run(
        _spec(action="I present the vial to Elara Vex"),
        _context(),
        None,
    )

    assert prose == "Elara closes the ledger and studies you."
    assert [effect.npc_name for effect in effects] == ["Elara Vex"]
    assert len(client.calls) == 1


@pytest.mark.asyncio
async def test_ref_entity_alias_must_be_grounded_in_visible_prose():
    client = ScriptedBrain([
        narration_response(
            "Elara closes the ledger.",
            tool_calls=[{
                "name": "ref_entity",
                "arguments": {"entity_id": "lys", "alias_used": "Lys"},
            }],
        ),
        narration_response("", tool_calls=[]),
    ])
    h = _Harness(client)

    prose, effects = await h.strategy.run(
        _spec(),
        _context(world_state_yaml="""
npcs_here:
- id: lys
  name: Lys
  type: npc
"""),
        None,
    )

    assert prose == "Elara closes the ledger."
    assert effects == []
    assert len(client.calls) == 2
    repair_prompt = client.calls[1]["messages"][-1]["content"]
    assert "alias_used must appear exactly in narrator prose" in repair_prompt


@pytest.mark.asyncio
async def test_invalid_followup_reference_is_recovered_without_model_repair():
    client = ScriptedBrain([
        narration_response("The barkeep nods."),
        narration_response("", tool_calls=[{"name": "ref_entity", "arguments": {}}]),
        narration_response("", tool_calls=[_REF_TOOL_CALL]),
    ])
    h = _Harness(client)

    _, effects = await h.strategy.run(
        _spec(),
        _context(world_state_yaml="""
npcs_here:
- id: barkeep
  name: barkeep
"""),
        None,
    )

    assert len(client.calls) == 2
    assert effects[0].ref_entity_id == "barkeep"
    assert h.strategy.last_diagnostics["tool_repair_attempted"] is False
    assert h.strategy.last_diagnostics[
        "tool_followup_structural_error_details"
    ] == ["ref_entity: ref_entity requires entity_id from the roster"]


@pytest.mark.asyncio
async def test_ref_alias_is_shortened_to_exact_grounded_name_without_repair():
    client = ScriptedBrain([
        narration_response("Sorin lowers his voice.", tool_calls=[{
            "name": "ref_entity",
            "arguments": {"entity_id": "courier", "alias_used": "Sorin Vex"},
        }]),
    ])
    h = _Harness(client)

    _, effects = await h.strategy.run(
        _spec(action="I listen."),
        _context(
            player_action="I listen.",
            world_state_yaml="""
npcs_here:
- id: courier
  name: Sorin
  type: npc
""",
        ),
        None,
    )

    assert len(client.calls) == 1
    assert effects[0].ref_entity_id == "courier"
    assert effects[0].ref_alias_used == "Sorin"


@pytest.mark.asyncio
async def test_roster_reference_strips_alias_belonging_to_a_different_character():
    client = ScriptedBrain([
        narration_response(
            "Elena Voss watches while the Tollman closes the gate.",
            tool_calls=[{
                "name": "ref_entity",
                "arguments": {
                    "entity_id": "elena-id",
                    "alias_used": "the Tollman",
                },
            }],
        ),
    ])
    h = _Harness(client)

    _, effects = await h.strategy.run(
        _spec(action="I watch Elena and the Tollman."),
        _context(world_state_yaml="""
npcs_here:
- id: elena-id
  name: Elena Voss
  type: npc
- id: tollman-id
  name: the Tollman
  type: npc
"""),
        None,
    )

    assert len(client.calls) == 1
    assert effects[0].ref_entity_id == "elena-id"
    assert effects[0].ref_alias_used is None
    assert h.strategy.last_diagnostics[
        "tool_ref_alias_mismatches_removed"
    ] == 1


@pytest.mark.asyncio
async def test_generic_roster_identity_can_take_explicitly_introduced_name():
    client = ScriptedBrain([
        narration_response(
            'The hooded woman lowers her voice. "I\'m Mira."',
            tool_calls=[{
                "name": "ref_entity",
                "arguments": {
                    "entity_id": "hooded-woman-id",
                    "alias_used": "Mira",
                },
            }],
        ),
    ])
    h = _Harness(client)

    _, effects = await h.strategy.run(
        _spec(action="I ask the hooded woman her name."),
        _context(world_state_yaml="""
npcs_here:
- id: hooded-woman-id
  name: hooded woman
  type: npc
"""),
        None,
    )

    assert effects[0].ref_alias_used == "Mira"
    assert h.strategy.last_diagnostics[
        "tool_ref_alias_mismatches_removed"
    ] == 0


@pytest.mark.asyncio
async def test_bare_article_cannot_ground_an_unrelated_roster_reference():
    client = ScriptedBrain([
        narration_response("You enter the Rusty Hinge.", tool_calls=[{
            "name": "ref_entity",
            "arguments": {"entity_id": "old-alley", "alias_used": "the"},
        }]),
        narration_response(""),
    ])
    h = _Harness(client)

    _, effects = await h.strategy.run(
        _spec(action="I enter the tavern."),
        _context(
            player_action="I enter the tavern.",
            kg_context_yaml="""
known_entities:
- id: old-alley
  name: Old Alley
  type: location
""",
        ),
        None,
    )

    assert effects == []
    assert "too generic" in h.strategy.last_diagnostics[
        "primary_structural_error_details"
    ][0]


def test_duplicate_same_name_object_creation_keeps_richer_single_effect():
    sparse = ProposedEffect(
        effect_type=EffectType.SPAWN_OBJECT,
        object_name="charcoal map rubbing",
    )
    rich = ProposedEffect(
        effect_type=EffectType.SPAWN_OBJECT,
        object_name="Charcoal Map-Rubbing",
        object_description="A brittle map of the lower bridges.",
    )

    effects, collapsed = NarrationStrategy._collapse_duplicate_creations([
        sparse,
        rich,
    ])

    assert effects == [rich]
    assert collapsed == 1


@pytest.mark.asyncio
async def test_npc_held_spawned_object_gets_narrowed_entity_update_repair():
    prose = (
        "Dorn reaches below the stall. When his hand comes back up, it's "
        "holding a rusted iron key."
    )
    client = ScriptedBrain([
        narration_response(prose, tool_calls=[
            {"name": "ref_entity", "arguments": {"entity_id": "dorn-id"}},
            {
                "name": "spawn_object",
                "arguments": {
                    "name": "rusted iron key",
                    "description": "A long, age-blackened key.",
                },
            },
        ]),
        narration_response("", tool_calls=[{
            "name": "update_entity",
            "arguments": {
                "entity_id": "dorn-id",
                "add_items": ["rusted iron key"],
            },
        }]),
    ])
    h = _Harness(client)
    h.tools.append({
        "type": "function",
        "function": {"name": "update_entity"},
    })

    _, effects = await h.strategy.run(
        _spec(action="I ask Dorn for a route below."),
        _context(
            player_action="I ask Dorn for a route below.",
            world_state_yaml="""
npcs_here:
- id: dorn-id
  name: Dorn
  type: npc
""",
        ),
        None,
    )

    assert len(client.calls) == 2
    assert {effect.effect_type for effect in effects} == {
        EffectType.REF_ENTITY,
        EffectType.SPAWN_OBJECT,
        EffectType.UPDATE_ENTITY,
    }
    assert h.strategy.last_diagnostics["effect_obligation_missing_final"] == []


@pytest.mark.asyncio
async def test_empty_ref_call_recovers_all_named_authoritative_roster_entities():
    client = ScriptedBrain([
        narration_response("Orin warns Lira about the road."),
        narration_response("", tool_calls=[{
            "name": "ref_entity",
            "arguments": {},
        }]),
    ])
    h = _Harness(client)
    world_yaml = """
location: lower terraces
npcs_here:
- id: orin-id
  name: Orin
  aliases: [the hunched figure]
- id: lira-id
  name: Lira
"""

    _, effects = await h.strategy.run(
        _spec(action="I listen."),
        _context(player_action="I listen.", world_state_yaml=world_yaml),
        None,
    )

    assert len(client.calls) == 2
    assert {effect.ref_entity_id for effect in effects} == {"orin-id", "lira-id"}
    assert h.strategy.last_diagnostics["tool_repair_attempted"] is False
    assert set(h.strategy.last_diagnostics["tool_ref_deterministic_recoveries"]) == {
        "orin-id",
        "lira-id",
    }


@pytest.mark.asyncio
async def test_empty_ref_call_recovers_unique_grounded_first_name():
    client = ScriptedBrain([
        narration_response("Liraen warns that the ink is spreading."),
        narration_response("", tool_calls=[{
            "name": "ref_entity",
            "arguments": {},
        }]),
    ])
    h = _Harness(client)
    world_yaml = """
npcs_here:
- id: liraen-id
  name: Liraen Vex
- id: dorn-id
  name: Dorn Bale
"""

    _, effects = await h.strategy.run(
        _spec(action="I listen."),
        _context(player_action="I listen.", world_state_yaml=world_yaml),
        None,
    )

    assert len(client.calls) == 2
    assert effects[0].ref_entity_id == "liraen-id"
    assert effects[0].ref_alias_used == "Liraen"


@pytest.mark.asyncio
async def test_generic_add_npc_is_rejected_before_execution_surface():
    client = ScriptedBrain([
        narration_response(
            "The courier stirs.",
            tool_calls=[_add_npc_tool_call("courier")],
        ),
        narration_response(""),
    ])
    h = _Harness(client)

    _, effects = await h.strategy.run(_spec(action="I wait."), _context(), None)

    assert effects == []
    assert h.strategy.last_diagnostics["primary_structural_errors"] == 1
    assert "generic role" in h.strategy.last_diagnostics[
        "primary_structural_error_details"
    ][0]


@pytest.mark.asyncio
async def test_tool_repair_preserves_valid_calls_and_can_abstain_from_bad_ones():
    client = ScriptedBrain([
        narration_response("Marta nods while the cloaked woman watches."),
        narration_response("", tool_calls=[
            _REF_TOOL_CALL,
            _add_npc_tool_call("Wool Cloak Woman"),
        ]),
        narration_response(""),  # compliant abstention: anonymous NPC is omitted
    ])
    h = _Harness(client)

    _, effects = await h.strategy.run(_spec(), _context(), None)

    assert [effect.effect_type for effect in effects] == [EffectType.REF_ENTITY]
    diagnostics = h.strategy.last_diagnostics
    assert diagnostics["tool_followup_structural_errors"] == 0
    assert diagnostics["tool_repair_structural_errors"] == 0
    assert len(client.calls) == 2
    assert diagnostics["tool_invalid_effects_dropped"] == 0
    assert diagnostics["tool_policy_suppressed_effects"] == 1
    assert diagnostics["tool_repair_failed_closed"] is False


@pytest.mark.asyncio
async def test_invalid_bounded_repair_fails_closed():
    client = ScriptedBrain([
        narration_response("The barkeep nods."),
        narration_response("", tool_calls=[{"name": "ref_entity", "arguments": {}}]),
        narration_response("", tool_calls=[{"name": "ref_entity", "arguments": {}}]),
    ])
    h = _Harness(client)

    _, effects = await h.strategy.run(
        _spec(),
        _context(
            world_state_yaml="turn: 1\nphase: exploration\n",
            kg_context_yaml="",
        ),
        None,
    )

    assert len(client.calls) == 2
    assert effects == []
    diagnostics = h.strategy.last_diagnostics
    assert diagnostics["tool_followup_structural_errors"] == 1
    assert diagnostics["tool_repair_structural_errors"] == 0
    assert diagnostics["tool_invalid_effects_dropped"] == 1
    assert diagnostics["tool_repair_failed_closed"] is False


@pytest.mark.asyncio
async def test_wrong_mutation_family_cannot_mask_required_remove_entity():
    action = (
        "This is an established automatic trigger with no roll: its charge "
        "destroys the sealed reliquary completely."
    )
    client = ScriptedBrain([
        narration_response(
            "The reliquary detonates and is destroyed completely.",
            tool_calls=[{
                "name": "update_player",
                "arguments": {
                    "hp_delta": -2,
                    "damage_type": "fire",
                    "hp_reason": "reliquary blast",
                },
            }],
        ),
        narration_response("", tool_calls=[{
            "name": "remove_entity",
            "arguments": {
                "entity_id": "sealed-reliquary",
                "reason": "destroyed",
            },
        }]),
    ])
    h = _Harness(client)

    prose, effects = await h.strategy.run(
        _spec(action=action, player_action=action),
        _context(player_action=action),
        None,
    )

    assert prose == "The reliquary detonates and is destroyed completely."
    assert [effect.effect_type for effect in effects] == [
        EffectType.UPDATE_PLAYER,
        EffectType.REMOVE_ENTITY,
    ]
    assert len(client.calls) == 2
    primary_contract = "\n".join(
        message["content"] for message in client.calls[0]["messages"]
    )
    assert "RESOLVED OUTCOME CONTRACT" in primary_contract
    followup_prompt = client.calls[1]["messages"][-1]["content"]
    assert "REQUIRED effect obligations: remove_entity" in followup_prompt
    diagnostics = h.strategy.last_diagnostics
    assert diagnostics["effect_obligation_missing_initial"] == ["remove_entity"]
    assert diagnostics["effect_obligation_missing_final"] == []
    assert diagnostics["effect_obligation_repair_succeeded"] is True


@pytest.mark.asyncio
async def test_resolved_transfer_contradiction_repairs_prose_and_tools_together():
    action = (
        "This is an uncontested item transfer with no roll: I hand my brass "
        "compass to Mara Venn, she accepts it, and it is now in her coat "
        "rather than my pack."
    )
    client = ScriptedBrain([
        narration_response(
            "Your pack is empty. You lost the compass; Mara never had it.",
            tool_calls=[_REF_TOOL_CALL],
        ),
        narration_response(
            "You hand the brass compass to Mara Venn. She accepts it and "
            "tucks it securely into her coat.",
            tool_calls=[
                {
                    "name": "update_entity",
                    "arguments": {
                        "entity_id": "mara-venn",
                        "add_items": ["brass compass"],
                    },
                },
                {
                    "name": "update_player",
                    "arguments": {
                        "item_remove": [{
                            "name": "brass compass",
                            "destination": "npc:mara-venn",
                        }],
                    },
                },
            ],
        ),
    ])
    h = _Harness(client)

    prose, effects = await h.strategy.run(
        _spec(action=action, player_action=action),
        _context(player_action=action),
        None,
    )

    assert prose.startswith("You hand the brass compass to Mara Venn.")
    assert {effect.effect_type for effect in effects} == {
        EffectType.UPDATE_ENTITY,
        EffectType.UPDATE_PLAYER,
    }
    assert len(client.calls) == 2
    repair_prompt = client.calls[1]["messages"][-1]["content"]
    assert "RESOLVED OUTCOME REPAIR REQUIRED" in repair_prompt
    diagnostics = h.strategy.last_diagnostics
    assert diagnostics["resolved_outcome_repair_attempted"] is True
    assert diagnostics["resolved_outcome_repair_succeeded"] is True
    assert diagnostics["resolved_outcome_failed_closed"] is False
    assert diagnostics["effect_obligation_missing_final"] == []


@pytest.mark.asyncio
async def test_transfer_with_correct_prose_but_no_tools_uses_combined_repair():
    action = (
        "This is an uncontested item transfer with no roll: I hand my brass "
        "compass to Mara Venn, she accepts it, and it is now in her coat."
    )
    client = ScriptedBrain([
        narration_response(
            "You hand the brass compass to Mara Venn. She accepts it and "
            "tucks it into her coat."
        ),
        narration_response(
            "You hand the brass compass to Mara Venn. She accepts it and "
            "tucks it into her coat.",
            tool_calls=[
                {
                    "name": "update_entity",
                    "arguments": {
                        "entity_id": "mara-venn",
                        "add_items": ["brass compass"],
                    },
                },
                {
                    "name": "update_player",
                    "arguments": {
                        "item_remove": [{"name": "brass compass"}],
                    },
                },
            ],
        ),
    ])
    h = _Harness(client)

    _, effects = await h.strategy.run(
        _spec(action=action, player_action=action),
        _context(
            player_action=action,
            world_state_yaml="""
npcs_here:
- id: mara
  name: Mara
  type: npc
""",
        ),
        None,
    )

    assert {effect.effect_type for effect in effects} == {
        EffectType.UPDATE_ENTITY,
        EffectType.UPDATE_PLAYER,
    }
    assert len(client.calls) == 2
    assert "omitted one or both sides" in client.calls[1]["messages"][-1]["content"]
    diagnostics = h.strategy.last_diagnostics
    assert diagnostics["resolved_outcome_contradictions"] == []
    assert diagnostics["resolved_outcome_repair_attempted"] is True
    assert diagnostics["resolved_outcome_repair_succeeded"] is True


@pytest.mark.asyncio
async def test_terminal_repair_narrows_surface_to_still_missing_family():
    action = "Mara reveals her scar and swears to become my ally."
    client = ScriptedBrain([
        narration_response("Mara shows her scar and swears to become your ally."),
        narration_response("", tool_calls=[{
            "name": "ref_entity",
            "arguments": {"entity_id": "mara"},
        }]),
        narration_response("", tool_calls=[{
            "name": "update_entity",
            "arguments": {
                "entity_id": "mara",
                "disposition": "allied",
                "description_addition": "bears a fresh crescent scar",
            },
        }]),
    ])
    h = _Harness(client)
    h.tools = [{
        "type": "function",
        "function": {"name": "update_entity", "parameters": {"type": "object"}},
    }]

    _, effects = await h.strategy.run(
        _spec(action=action, player_action=action),
        _context(
            player_action=action,
            world_state_yaml="""
npcs_here:
- id: mara
  name: Mara
  type: npc
""",
        ),
        None,
    )

    assert [effect.effect_type for effect in effects] == [
        EffectType.REF_ENTITY,
        EffectType.UPDATE_ENTITY,
    ]
    assert len(client.calls) == 3
    terminal_tools = client.calls[2]["kwargs"]["tools"]
    assert [tool["function"]["name"] for tool in terminal_tools] == [
        "update_entity"
    ]
    diagnostics = h.strategy.last_diagnostics
    assert diagnostics["effect_obligation_terminal_repair_attempted"] is True
    assert diagnostics["effect_obligation_terminal_repair_succeeded"] is True
    assert diagnostics["effect_obligation_missing_final"] == []


@pytest.mark.asyncio
async def test_unrepaired_resolved_outcome_contradiction_fails_closed():
    action = (
        "This is an uncontested item transfer with no roll: I hand my brass "
        "compass to Mara Venn, she accepts it, and it is now in her coat."
    )
    contradiction = "You lost the compass. Mara never had it."
    client = ScriptedBrain([
        narration_response(contradiction, tool_calls=[_REF_TOOL_CALL]),
        narration_response(contradiction, tool_calls=[_REF_TOOL_CALL]),
    ])
    h = _Harness(client)

    prose, effects = await h.strategy.run(
        _spec(action=action, player_action=action),
        _context(player_action=action),
        None,
    )

    assert "no world-state change is committed" in prose
    assert effects == []
    assert len(client.calls) == 2
    diagnostics = h.strategy.last_diagnostics
    assert diagnostics["resolved_outcome_repair_attempted"] is True
    assert diagnostics["resolved_outcome_repair_succeeded"] is False
    assert diagnostics["resolved_outcome_failed_closed"] is True
    assert set(diagnostics["effect_obligation_missing_final"]) == {
        "update_entity",
        "update_player",
    }


# ── Empty-prose policies ──────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_empty_prose_bails_with_fallback_and_no_followup():
    client = ScriptedBrain([narration_response("")])
    h = _Harness(client)

    prose, effects = await h.strategy.run(
        _spec(
            empty_prose_fallback="*Elara's action unfolds...*",
            empty_prose_warn_event="narrator_returned_empty_for_action",
        ),
        _context(),
        None,
    )

    assert prose == "*Elara's action unfolds...*"
    assert effects == []
    assert len(client.calls) == 1  # bail: no followup on placeholder prose


@pytest.mark.asyncio
async def test_empty_prose_bail_discards_primary_effects():
    # Empty content WITH tool calls: the bail path returns no effects —
    # preserved pre-Step-2 behavior of the action/outcome paths.
    client = ScriptedBrain([narration_response("", tool_calls=[_REF_TOOL_CALL])])
    h = _Harness(client)

    prose, effects = await h.strategy.run(
        _spec(empty_prose_fallback="*fallback...*"), _context(), None,
    )

    assert prose == "*fallback...*"
    assert effects == []


@pytest.mark.asyncio
async def test_empty_prose_continue_substitutes_hint_and_runs_followup():
    client = ScriptedBrain([
        narration_response(""),                               # empty primary
        narration_response("", tool_calls=[_REF_TOOL_CALL]),   # followup
    ])
    h = _Harness(client)

    prose, effects = await h.strategy.run(
        _spec(
            empty_prose_fallback="The purchase lands.",
            continue_on_empty_prose=True,
        ),
        _context(),
        None,
    )

    # The mech hint substituted AND fed to the followup as the prose turn.
    assert prose == "The purchase lands."
    followup = client.calls[1]
    assert {"role": "assistant", "content": "The purchase lands."} in followup["messages"]
    assert [e.effect_type for e in effects] == [EffectType.REF_ENTITY]


# ── Truncation fix ────────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_truncated_prose_gets_ellipsis():
    client = ScriptedBrain([
        narration_response("The rain falls", tool_calls=[_REF_TOOL_CALL]),
    ])
    h = _Harness(client)

    prose, _ = await h.strategy.run(_spec(), _context(), None)

    assert prose == "The rain falls..."


# ── Context-budget warning ────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_context_near_cap_warns_when_num_ctx_known():
    from structlog.testing import capture_logs

    client = ScriptedBrain([
        narration_response("Fine.", tool_calls=[_REF_TOOL_CALL]),
    ])
    # Tiny declared context: budget = 8000 - 1500 - 5000 = 1500 tokens,
    # so ~8k chars of assembled messages must trip the warning.
    client.num_ctx = 8000
    h = _Harness(client)

    with capture_logs() as logs:
        await h.strategy.run(_spec(prompt="p" * 8000), _context(), None)

    events = [l for l in logs if l["event"] == "narration_context_near_cap"]
    assert len(events) == 1
    assert events[0]["num_ctx"] == 8000
    assert events[0]["token_budget"] == 1500
    assert events[0]["estimated_prompt_tokens"] > 1500


@pytest.mark.asyncio
async def test_no_near_cap_warning_without_num_ctx():
    from structlog.testing import capture_logs

    # Cloud-style client: no num_ctx attribute — no known hard cap, no noise.
    client = ScriptedBrain([
        narration_response("Fine.", tool_calls=[_REF_TOOL_CALL]),
    ])
    h = _Harness(client)

    with capture_logs() as logs:
        await h.strategy.run(_spec(prompt="p" * 60000), _context(), None)

    assert not [l for l in logs if l["event"] == "narration_context_near_cap"]


# -- Immutable continuity governance -----------------------------------------

def _dead_bram_governance() -> NarrativeGovernance:
    return NarrativeGovernance([
        NPCState(id="bram", name="Old Bram", alive=False),
    ])


@pytest.mark.asyncio
async def test_continuity_violation_is_rewritten_and_primary_effects_discarded():
    client = ScriptedBrain([
        narration_response(
            "Old Bram enters and smiles.",
            tool_calls=[{"name": "ref_entity", "arguments": {"entity_id": "bram"}}],
        ),
        narration_response(
            "Old Bram's corpse remains beneath the cairn.",
            tool_calls=[_REF_TOOL_CALL],
        ),
    ])
    h = _Harness(client, governance=_dead_bram_governance())

    prose, effects = await h.strategy.run(_spec(), _context(), None)

    assert prose == "Old Bram's corpse remains beneath the cairn."
    assert [effect.ref_entity_id for effect in effects] == ["barkeep"]
    assert len(client.calls) == 2
    repair = client.calls[1]
    assert repair["method"] == "chat"
    assert repair["kwargs"]["temperature"] == 0
    assert "CONTINUITY REPAIR REQUIRED" in repair["messages"][-1]["content"]
    assert "Old Bram enters and smiles." in repair["messages"][-2]["content"]
    assert h.strategy.last_diagnostics["continuity_violations"] == 1
    assert h.strategy.last_diagnostics["continuity_repair_succeeded"] is True
    assert h.strategy.last_diagnostics["continuity_failed_closed"] is False


@pytest.mark.asyncio
async def test_repeated_continuity_violation_fails_closed_without_tool_followup():
    client = ScriptedBrain([
        narration_response(
            "Old Bram enters and smiles.",
            tool_calls=[{"name": "ref_entity", "arguments": {"entity_id": "bram"}}],
        ),
        narration_response(
            "Old Bram laughs and offers you a drink.",
            tool_calls=[{"name": "ref_entity", "arguments": {"entity_id": "bram"}}],
        ),
    ])
    h = _Harness(client, governance=_dead_bram_governance())

    prose, effects = await h.strategy.run(_spec(), _context(), None)

    assert "Old Bram remains dead" in prose
    assert effects == []
    # Primary + one repair only. Fail-closed prose must not trigger a tool leg.
    assert len(client.calls) == 2
    assert h.strategy.last_diagnostics["continuity_failed_closed"] is True


@pytest.mark.asyncio
async def test_streaming_is_buffered_when_immutable_rules_are_active():
    client = ScriptedBrain([
        narration_response("The rain falls.", tool_calls=[_REF_TOOL_CALL]),
    ])
    streamed: list[str] = []

    async def on_token(token: str) -> None:
        streamed.append(token)

    h = _Harness(
        client,
        on_token=on_token,
        governance=_dead_bram_governance(),
    )

    prose, _ = await h.strategy.run(
        _spec(allow_streaming=True), _context(), None
    )

    assert prose == "The rain falls."
    assert streamed == []
    assert client.calls[0]["method"] == "chat"


# -- Repair meta-preamble strip ----------------------------------------------
#
# Live defect 2026-07-23 (turn log a04069e1, turn 2): the resolved-outcome
# repair returned prose OPENING with "You're right. I apologize for the
# contradiction. Let me correct this." — player-visible assistant meta-talk
# the grader flagged as a severe contradiction.

class TestStripRepairMetaPreamble:
    def test_live_case_apology_opener_is_stripped(self):
        prose = (
            "You're right. I apologize for the contradiction. Let me correct "
            "this. You hand the brass compass to Mara Venn. She accepts it."
        )
        assert strip_repair_meta_preamble(prose) == (
            "You hand the brass compass to Mara Venn. She accepts it."
        )

    def test_correction_header_line_is_stripped(self):
        prose = "Here is the corrected narration:\nThe rain falls on the cairn."
        assert strip_repair_meta_preamble(prose) == "The rain falls on the cairn."

    def test_all_meta_prose_strips_to_empty(self):
        prose = "My apologies. Let me rewrite the narration to fix this."
        assert strip_repair_meta_preamble(prose) == ""

    def test_clean_fiction_is_untouched(self):
        prose = "Mara studies the compass, then tucks it into her coat."
        assert strip_repair_meta_preamble(prose) == prose

    def test_quoted_dialogue_opener_is_fiction_not_meta(self):
        prose = '"You\'re right," Mara says. "The road forks at the cairn."'
        assert strip_repair_meta_preamble(prose) == prose

    def test_meta_vocabulary_after_fiction_start_is_kept(self):
        prose = (
            "Mara frowns at the ledger. The contradiction in the guild's "
            "accounts is plain to see."
        )
        assert strip_repair_meta_preamble(prose) == prose


@pytest.mark.asyncio
async def test_resolved_outcome_repair_strips_meta_preamble_from_prose():
    action = (
        "This is an uncontested item transfer with no roll: I hand my brass "
        "compass to Mara Venn, she accepts it, and it is now in her coat "
        "rather than my pack."
    )
    client = ScriptedBrain([
        narration_response(
            "Your pack is empty. You lost the compass; Mara never had it.",
            tool_calls=[_REF_TOOL_CALL],
        ),
        narration_response(
            "You're right. I apologize for the contradiction. Let me correct "
            "this. You hand the brass compass to Mara Venn. She accepts it "
            "and tucks it securely into her coat.",
            tool_calls=[
                {
                    "name": "update_entity",
                    "arguments": {
                        "entity_id": "mara-venn",
                        "add_items": ["brass compass"],
                    },
                },
                {
                    "name": "update_player",
                    "arguments": {
                        "item_remove": [{
                            "name": "brass compass",
                            "destination": "npc:mara-venn",
                        }],
                    },
                },
            ],
        ),
    ])
    h = _Harness(client)

    prose, effects = await h.strategy.run(
        _spec(action=action, player_action=action),
        _context(player_action=action),
        None,
    )

    assert prose.startswith("You hand the brass compass to Mara Venn.")
    assert "apologize" not in prose
    assert "contradiction" not in prose
    assert {effect.effect_type for effect in effects} == {
        EffectType.UPDATE_ENTITY,
        EffectType.UPDATE_PLAYER,
    }
    repair_prompt = client.calls[1]["messages"][-1]["content"]
    assert "never apologize" in repair_prompt
    diagnostics = h.strategy.last_diagnostics
    assert diagnostics["repair_meta_preamble_stripped"] is True
    assert diagnostics["resolved_outcome_repair_succeeded"] is True
    assert diagnostics["resolved_outcome_failed_closed"] is False


@pytest.mark.asyncio
async def test_resolved_outcome_repair_of_pure_meta_fails_closed():
    action = (
        "This is an uncontested item transfer with no roll: I hand my brass "
        "compass to Mara Venn, she accepts it, and it is now in her coat "
        "rather than my pack."
    )
    client = ScriptedBrain([
        narration_response(
            "Your pack is empty. You lost the compass; Mara never had it.",
            tool_calls=[_REF_TOOL_CALL],
        ),
        narration_response(
            "You're right. I apologize for the contradiction. Let me correct "
            "this."
        ),
    ])
    h = _Harness(client)

    prose, effects = await h.strategy.run(
        _spec(action=action, player_action=action),
        _context(player_action=action),
        None,
    )

    assert "could not be resolved consistently" in prose
    assert "apologize" not in prose
    assert effects == []
    # Fail-closed prose must not enter the tool-followup leg.
    assert len(client.calls) == 2
    diagnostics = h.strategy.last_diagnostics
    assert diagnostics["repair_meta_preamble_stripped"] is True
    assert diagnostics["resolved_outcome_repair_succeeded"] is False
    assert diagnostics["resolved_outcome_failed_closed"] is True


@pytest.mark.asyncio
async def test_continuity_repair_strips_meta_preamble_from_prose():
    client = ScriptedBrain([
        narration_response(
            "Old Bram enters and smiles.",
            tool_calls=[{"name": "ref_entity", "arguments": {"entity_id": "bram"}}],
        ),
        narration_response(
            "My mistake. Here is the corrected narration: Old Bram's corpse "
            "remains beneath the cairn.",
            tool_calls=[_REF_TOOL_CALL],
        ),
    ])
    h = _Harness(client, governance=_dead_bram_governance())

    prose, effects = await h.strategy.run(_spec(), _context(), None)

    assert prose == "Old Bram's corpse remains beneath the cairn."
    assert [effect.ref_entity_id for effect in effects] == ["barkeep"]
    repair_prompt = client.calls[1]["messages"][-1]["content"]
    assert "never apologize" in repair_prompt
    diagnostics = h.strategy.last_diagnostics
    assert diagnostics["repair_meta_preamble_stripped"] is True
    assert diagnostics["continuity_repair_succeeded"] is True
    assert diagnostics["continuity_failed_closed"] is False


@pytest.mark.asyncio
async def test_continuity_repair_of_pure_meta_falls_back_closed():
    client = ScriptedBrain([
        narration_response(
            "Old Bram enters and smiles.",
            tool_calls=[{"name": "ref_entity", "arguments": {"entity_id": "bram"}}],
        ),
        narration_response(
            "You're right, there is a contradiction. Let me rewrite the "
            "narration."
        ),
    ])
    h = _Harness(client, governance=_dead_bram_governance())

    prose, effects = await h.strategy.run(_spec(), _context(), None)

    assert "Old Bram remains dead" in prose
    assert "apologize" not in prose and "rewrite" not in prose
    assert effects == []
    assert len(client.calls) == 2
    diagnostics = h.strategy.last_diagnostics
    assert diagnostics["repair_meta_preamble_stripped"] is True
    assert diagnostics["continuity_repair_succeeded"] is False
    assert diagnostics["continuity_failed_closed"] is True


class TestProseFreshnessHint:
    def _history_context(self, history):
        from dnd_bot.llm.brains.base import BrainContext

        return BrainContext(message_history=history)

    def test_lists_recent_assistant_openings(self):
        from dnd_bot.llm.narration import NarrationStrategy

        hint = NarrationStrategy._prose_freshness_hint(self._history_context([
            {"role": "user", "content": "I open the door."},
            {"role": "assistant",
             "content": "Elara's lips part in a slow smile as she watches."},
            {"role": "user", "content": "I ask about the map."},
            {"role": "assistant",
             "content": "Elara's fingers drum the table twice before she speaks."},
        ]))
        assert "Elara's fingers drum the table" in hint
        assert "Elara's lips part in" in hint
        assert "different subject" in hint

    def test_fewer_than_two_narrations_no_hint(self):
        from dnd_bot.llm.narration import NarrationStrategy

        hint = NarrationStrategy._prose_freshness_hint(self._history_context([
            {"role": "assistant", "content": "Only one reply so far."},
        ]))
        assert hint == ""

    def test_empty_history_no_hint(self):
        from dnd_bot.llm.narration import NarrationStrategy

        assert NarrationStrategy._prose_freshness_hint(
            self._history_context([])
        ) == ""
