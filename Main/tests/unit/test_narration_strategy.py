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

from dnd_bot.llm.brains.base import BrainContext
from dnd_bot.llm.effects import EffectType
from dnd_bot.llm.narration import NarrationSpec, NarrationStrategy
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

    def _build_bookend_messages(self, context):
        self.bookend_contexts.append(context)
        return [
            {"role": "system", "content": "PERSONA"},
            {
                "role": "user",
                "content": f"<player_action>[{context.player_name}]: "
                           f"{context.player_action}</player_action>",
            },
        ]

    def _build_messages(self, context):
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

    def __init__(self, client, on_token=None):
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
        world_state_yaml="location: tavern",
        kg_context_yaml="kg",
        narrative_memory="past prose",
        last_turn_trace="trace",
        player_action="I greet the barkeep",
        player_name="Elara",
    )
    base.update(overrides)
    return BrainContext(**base)


_REF_TOOL_CALL = {"name": "ref_entity", "arguments": {"entity_id": "barkeep"}}


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
async def test_followup_skipped_when_primary_returned_effects():
    client = ScriptedBrain([
        narration_response("Done deal.", tool_calls=[_REF_TOOL_CALL]),
    ])
    h = _Harness(client)

    await h.strategy.run(_spec(), _context(), None)

    assert len(client.calls) == 1


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


@pytest.mark.asyncio
async def test_followup_failure_swallowed_and_returns_no_effects():
    client = _FlakyFollowupClient(narration_response("The tavern hums quietly."))
    h = _Harness(client)

    prose, effects = await h.strategy.run(_spec(), _context(), None)

    assert len(client.calls) == 2  # the followup WAS attempted
    assert prose == "The tavern hums quietly."
    assert effects == []


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
