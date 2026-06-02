"""Tests for TurnLogReader — assertion API over per-session JSONL logs."""

import json
from pathlib import Path

import pytest


@pytest.fixture
def synthetic_log(tmp_path):
    """Build a minimal synthetic JSONL log with the structure the
    orchestrator's TurnRecord writes. Models the user's seed-and-callback
    scenario: player gives relic to innkeeper at turn 3, returns at turn 25.
    """
    session_id = "test_session_42"
    log_dir = tmp_path
    path = log_dir / f"{session_id}.jsonl"

    records = [
        # Turn 1 — player enters the inn
        {
            "session_id": session_id,
            "turn": 1,
            "timestamp": "2026-04-26T12:00:00",
            "world_state": {
                "before": "turn: 0\nlocation: ''\n",
                "after": (
                    "turn: 1\n"
                    "phase: exploration\n"
                    "time_of_day: evening\n"
                    "location: hearthwood inn\n"
                    "npcs_here:\n"
                    "  - name: bron\n"
                    "    disposition: friendly\n"
                    "    desc: A barrel-chested innkeeper with a graying beard\n"
                ),
            },
            "narrator_response": {
                "raw": "You step into the warmth of the Hearthwood Inn. Bron the innkeeper looks up.",
                "format": "ollama",
                "reprompted": False,
            },
            "knowledge_graph": {
                "nodes_total": 2,
                "edges_total": 1,
                "seed_entities": [],
                "context_injected": False,
                "context_yaml": "",
                "narrative_chunks": [],
                "text_match_seeds": [],
                "scene_seeds": [],
                "vector_match_seeds": [],
            },
            "effects": {
                "proposed": [{"effect_type": "ref_entity"}],
                "executed": [{"effect_type": "ref_entity"}],
                "rejected": [],
            },
            "narrator_routing": {
                "tier": "standard",
                "provider": "ollama",
                "model": "qwen3.5:27b",
                "significance": "routine",
                "phase_b_veto": False,
            },
            "prompt": {
                "messages": [
                    {"role": "system", "content": "narrator prompt"},
                    {"role": "user", "content": "I enter the inn."},
                ],
            },
        },
        # Turn 3 — player gives relic to innkeeper for safekeeping
        {
            "session_id": session_id,
            "turn": 3,
            "timestamp": "2026-04-26T12:05:00",
            "world_state": {
                "before": "turn: 2\n",
                "after": (
                    "turn: 3\n"
                    "phase: exploration\n"
                    "location: hearthwood inn\n"
                    "npcs_here:\n"
                    "  - name: bron\n"
                    "    disposition: friendly\n"
                    "    desc: A barrel-chested innkeeper\n"
                    "    holds: [ancient relic]\n"
                ),
            },
            "narrator_response": {
                "raw": "Bron accepts the ancient relic with reverence.",
                "format": "ollama",
                "reprompted": False,
            },
            "knowledge_graph": {
                "nodes_total": 4,
                "edges_total": 3,
                "seed_entities": ["bron"],
                "context_injected": True,
                "context_yaml": "bron:\n  disposition: friendly\n",
                "narrative_chunks": [],
                "text_match_seeds": ["bron"],
                "scene_seeds": ["bron"],
                "vector_match_seeds": [],
            },
            "effects": {
                "proposed": [
                    {"effect_type": "update_player"},
                    {"effect_type": "update_entity"},
                ],
                "executed": [
                    {"effect_type": "update_player", "details": {"items_removed": [{"name": "ancient relic", "destination": "npc:bron"}]}},
                    {"effect_type": "update_entity", "details": {"entity_id": "bron", "applied": {"items_added": ["ancient relic"]}}},
                ],
                "rejected": [],
            },
            "narrator_routing": {
                "tier": "premium",
                "provider": "ollama",
                "model": "qwen3.5:27b",
                "significance": "notable",
                "phase_b_veto": False,
            },
            "prompt": {
                "messages": [
                    {"role": "user", "content": "I hand the ancient relic to Bron for safekeeping."},
                ],
            },
        },
        # Turn 25 — player returns to inn after long absence
        {
            "session_id": session_id,
            "turn": 25,
            "timestamp": "2026-04-26T13:00:00",
            "world_state": {
                "before": "turn: 24\n",
                "after": (
                    "turn: 25\n"
                    "phase: exploration\n"
                    "location: hearthwood inn\n"
                    "npcs_here:\n"
                    "  - name: bron\n"
                    "    disposition: friendly\n"
                    "    desc: A barrel-chested innkeeper\n"
                    "    holds: [ancient relic]\n"
                ),
            },
            "narrator_response": {
                "raw": "Bron's face lights up — he returns the ancient relic to your hand.",
                "format": "ollama",
                "reprompted": False,
            },
            "knowledge_graph": {
                "nodes_total": 12,
                "edges_total": 18,
                "seed_entities": ["bron", "ancient_relic"],
                "context_injected": True,
                "context_yaml": (
                    "bron:\n  disposition: friendly\n  holds: [ancient relic]\n"
                    "ancient_relic:\n  held_by: bron\n"
                ),
                "narrative_chunks": [
                    {"chunk_id": "c123", "preview": "Bron accepts the ancient relic with reverence", "turn": 3, "entity_ids": ["bron"]},
                ],
                "text_match_seeds": ["bron", "ancient_relic"],
                "scene_seeds": ["bron"],
                "vector_match_seeds": [],
            },
            "effects": {
                "proposed": [{"effect_type": "ref_entity"}, {"effect_type": "update_player"}],
                "executed": [{"effect_type": "ref_entity"}, {"effect_type": "update_player"}],
                "rejected": [],
            },
            "narrator_routing": {
                "tier": "standard",
                "provider": "ollama",
                "model": "qwen3.5:27b",
                "significance": "notable",
                "phase_b_veto": False,
            },
            "prompt": {
                "messages": [
                    {"role": "user", "content": "I return to the inn and ask Bron for my relic back."},
                ],
            },
        },
    ]

    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            f.write(json.dumps(r) + "\n")

    return session_id, log_dir


# ── Loading ────────────────────────────────────────────────────────────


class TestLoading:

    def test_load_returns_reader_with_records(self, synthetic_log):
        from dnd_bot.llm.turn_log_reader import TurnLogReader
        session_id, log_dir = synthetic_log
        reader = TurnLogReader.load(session_id, log_dir=log_dir)
        assert len(reader) == 3
        assert reader.turns() == [1, 3, 25]

    def test_missing_session_raises(self, tmp_path):
        from dnd_bot.llm.turn_log_reader import TurnLogReader
        with pytest.raises(FileNotFoundError):
            TurnLogReader.load("nonexistent", log_dir=tmp_path)

    def test_malformed_lines_skipped(self, tmp_path):
        """Reader should be defensive against partial writes."""
        from dnd_bot.llm.turn_log_reader import TurnLogReader
        path = tmp_path / "broken.jsonl"
        with open(path, "w") as f:
            f.write('{"turn": 1, "session_id": "x"}\n')
            f.write("not json garbage\n")
            f.write('{"turn": 2, "session_id": "x"}\n')
        reader = TurnLogReader.load("broken", log_dir=tmp_path)
        assert len(reader) == 2
        assert reader.turns() == [1, 2]


# ── WorldState snapshot queries ───────────────────────────────────────


class TestWorldStateSnapshot:

    def test_npc_holds_item_on_seed_turn(self, synthetic_log):
        from dnd_bot.llm.turn_log_reader import TurnLogReader
        session_id, log_dir = synthetic_log
        log = TurnLogReader.load(session_id, log_dir=log_dir)

        ws = log.world_state_after(turn=3)
        bron = ws.npc("bron")
        assert bron.name == "bron"
        assert bron.holds("ancient relic")
        assert "ancient relic" in bron.inventory

    def test_npc_still_holds_item_on_callback_turn(self, synthetic_log):
        """The user's headline use case: 22 turns later, innkeeper still has relic."""
        from dnd_bot.llm.turn_log_reader import TurnLogReader
        session_id, log_dir = synthetic_log
        log = TurnLogReader.load(session_id, log_dir=log_dir)

        ws_after = log.world_state_after(turn=25)
        assert ws_after.npc("bron").holds("ancient relic")

    def test_npc_lookup_returns_empty_for_unknown(self, synthetic_log):
        from dnd_bot.llm.turn_log_reader import TurnLogReader
        session_id, log_dir = synthetic_log
        log = TurnLogReader.load(session_id, log_dir=log_dir)

        ws = log.world_state_after(turn=3)
        ghost = ws.npc("ghost_who_never_existed")
        assert not ghost  # falsy
        assert ghost.name == ""

    def test_location_and_phase_accessible(self, synthetic_log):
        from dnd_bot.llm.turn_log_reader import TurnLogReader
        session_id, log_dir = synthetic_log
        log = TurnLogReader.load(session_id, log_dir=log_dir)

        ws = log.world_state_after(turn=1)
        assert ws.location == "hearthwood inn"
        assert ws.phase == "exploration"
        assert ws.turn == 1


# ── KG context queries ────────────────────────────────────────────────


class TestKGContextSnapshot:

    def test_kg_context_mentions_the_relic_at_callback_turn(self, synthetic_log):
        """When the player asks for the relic 22 turns later, the KG
        context should surface the relic + bron — that's how memory works."""
        from dnd_bot.llm.turn_log_reader import TurnLogReader
        session_id, log_dir = synthetic_log
        log = TurnLogReader.load(session_id, log_dir=log_dir)

        ctx = log.kg_context_for(turn=25)
        assert ctx.mentions("ancient relic")
        assert "bron" in ctx.seed_entities

    def test_narrative_chunks_recalled(self, synthetic_log):
        from dnd_bot.llm.turn_log_reader import TurnLogReader
        session_id, log_dir = synthetic_log
        log = TurnLogReader.load(session_id, log_dir=log_dir)

        ctx = log.kg_context_for(turn=25)
        chunks = ctx.chunks_mentioning("relic")
        assert len(chunks) == 1
        assert chunks[0]["turn"] == 3  # the seed turn

    def test_kg_context_empty_for_turn_with_no_kg(self, synthetic_log):
        from dnd_bot.llm.turn_log_reader import TurnLogReader
        session_id, log_dir = synthetic_log
        log = TurnLogReader.load(session_id, log_dir=log_dir)

        ctx = log.kg_context_for(turn=1)  # KG was empty on turn 1
        assert not ctx.mentions("anything")
        assert ctx.seed_entities == []


# ── Narrator response + effects ────────────────────────────────────────


class TestNarratorAndEffects:

    def test_narrator_response_text(self, synthetic_log):
        from dnd_bot.llm.turn_log_reader import TurnLogReader
        session_id, log_dir = synthetic_log
        log = TurnLogReader.load(session_id, log_dir=log_dir)

        resp = log.narrator_response(turn=3)
        assert resp.references("ancient relic")
        assert resp.char_length > 0

    def test_executed_effects(self, synthetic_log):
        from dnd_bot.llm.turn_log_reader import TurnLogReader
        session_id, log_dir = synthetic_log
        log = TurnLogReader.load(session_id, log_dir=log_dir)

        effs = log.effects_at(turn=3)
        assert effs.has_executed("update_entity")
        assert effs.has_executed("update_player")
        assert len(effs.executed_of_type("update_entity")) == 1

    def test_narrator_routing(self, synthetic_log):
        from dnd_bot.llm.turn_log_reader import TurnLogReader
        session_id, log_dir = synthetic_log
        log = TurnLogReader.load(session_id, log_dir=log_dir)

        routing = log.narrator_routing(turn=3)
        assert routing is not None
        assert routing.tier == "premium"
        assert routing.significance == "notable"


# ── Trajectory / multi-turn queries ────────────────────────────────────


class TestTrajectoryQueries:

    def test_entity_history_traces_npc_across_turns(self, synthetic_log):
        from dnd_bot.llm.turn_log_reader import TurnLogReader
        session_id, log_dir = synthetic_log
        log = TurnLogReader.load(session_id, log_dir=log_dir)

        history = log.entity_history("bron")
        # Bron appears in all 3 turns of our synthetic log
        assert len(history) == 3
        turns = [t for t, _ in history]
        assert turns == [1, 3, 25]

    def test_turns_mentioning_finds_seed_and_callback(self, synthetic_log):
        from dnd_bot.llm.turn_log_reader import TurnLogReader
        session_id, log_dir = synthetic_log
        log = TurnLogReader.load(session_id, log_dir=log_dir)

        relic_turns = log.turns_mentioning("relic")
        assert relic_turns == [3, 25]

    def test_turns_with_kg_mention(self, synthetic_log):
        from dnd_bot.llm.turn_log_reader import TurnLogReader
        session_id, log_dir = synthetic_log
        log = TurnLogReader.load(session_id, log_dir=log_dir)

        # KG mentioned the relic at turn 25 (and only there in our fixture)
        kg_turns = log.turns_with_kg_mention("relic")
        assert 25 in kg_turns


# ── Integration: the seed-and-callback assertion pattern ──────────────


class TestSeedAndCallbackPattern:
    """End-to-end demonstration of the assertion API supporting the
    long-horizon test pattern."""

    def test_seed_planted_and_held_through_callback(self, synthetic_log):
        from dnd_bot.llm.turn_log_reader import TurnLogReader
        session_id, log_dir = synthetic_log
        log = TurnLogReader.load(session_id, log_dir=log_dir)

        # 1. The seed planted at turn 3 took effect
        assert log.world_state_after(turn=3).npc("bron").holds("ancient relic")

        # 2. State held through to the callback turn (22 turns later)
        assert log.world_state_after(turn=25).npc("bron").holds("ancient relic")

        # 3. The KG surfaced the relic when the player asked for it
        assert log.kg_context_for(turn=25).mentions("ancient relic")

        # 4. The narrator's prose at the callback referenced it
        assert log.narrator_response(turn=25).references("ancient relic")

        # 5. The effects log shows the update_entity that handed it over
        assert log.effects_at(turn=3).has_executed("update_entity")
