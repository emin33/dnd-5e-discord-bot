"""Tests for EntityDedupJudge — the paraphrase-fragmentation fix.

Mocks the brain client so the judge logic is tested in isolation. The
real brain integration is exercised by the long-horizon test in
test_long_horizon.py.
"""

from __future__ import annotations

import json
from dataclasses import dataclass

import pytest

from dnd_bot.llm.extractors.dedup_judge import (
    EntityDedupJudge,
    DedupDecision,
    _format_registry_slice,
)
from dnd_bot.game.world_state import NPCState


@dataclass
class _MockResponse:
    content: str = ""


class _MockBrain:
    """Stand-in for the brain client. Returns a pre-canned content string."""

    def __init__(self, response_content: str):
        self.response_content = response_content
        self.last_messages: list[dict] | None = None
        self.last_kwargs: dict | None = None

    async def chat(self, messages, **kwargs):
        self.last_messages = messages
        self.last_kwargs = kwargs
        return _MockResponse(content=self.response_content)


# ── Registry slice formatting ──────────────────────────────────────────


class TestRegistryFormatting:

    def test_empty_registry_returns_marker(self):
        assert _format_registry_slice([], current_turn=5) == "(registry empty)"

    def test_recency_window_filters_old_entities(self):
        old = NPCState(
            id="x", name="Forgotten", description="long ago", last_seen_turn=1, important=False
        )
        recent = NPCState(
            id="y", name="Marta", description="herbalist", last_seen_turn=18, important=False
        )
        out = _format_registry_slice([old, recent], current_turn=20, recency_window=5)
        assert "Marta" in out
        assert "Forgotten" not in out

    def test_important_entities_kept_outside_recency_window(self):
        """``important`` NPCs survive the recency cutoff — quest givers
        stay surfaced even after long gaps."""
        old_important = NPCState(
            id="q", name="Questmaster", description="key story figure",
            last_seen_turn=1, important=True,
        )
        out = _format_registry_slice([old_important], current_turn=50, recency_window=5)
        assert "Questmaster" in out

    def test_dead_entities_excluded(self):
        dead = NPCState(id="d", name="Slain", description="rip", alive=False, last_seen_turn=10)
        out = _format_registry_slice([dead], current_turn=10, recency_window=15)
        assert "Slain" not in out

    def test_aliases_and_inventory_included(self):
        npc = NPCState(
            id="b", name="Bron", description="innkeeper",
            aliases=["Old Bron"], inventory=["jade relic"],
            last_seen_turn=5,
        )
        out = _format_registry_slice([npc], current_turn=6)
        assert "Old Bron" in out
        assert "jade relic" in out


# ── Judge decision parsing ─────────────────────────────────────────────


@pytest.mark.asyncio
class TestJudgeDecisionParsing:

    async def test_empty_registry_fast_path(self):
        """No NPCs in world → judge skips brain call and accepts."""
        judge = EntityDedupJudge(client=_MockBrain('{"action": "accept"}'))
        decision = await judge.judge_add_npc(
            proposed_name="Marta",
            proposed_description="herbalist",
            narrator_prose="Marta greets you",
            existing_npcs=[],
            current_turn=1,
        )
        assert decision.action == "accept"
        # Brain wasn't called (empty registry → fast path)
        assert judge.client.last_messages is None

    async def test_rewrite_returns_target_id_and_alias(self):
        brain = _MockBrain(
            '{"action": "rewrite", "target_id": "bron-uuid", "alias": "Old Bron"}'
        )
        judge = EntityDedupJudge(client=brain)
        existing = NPCState(id="bron-uuid", name="Bron", last_seen_turn=5)
        decision = await judge.judge_add_npc(
            proposed_name="Old Bron",
            proposed_description="innkeeper",
            narrator_prose="Old Bron leaned over the bar",
            existing_npcs=[existing],
            current_turn=6,
        )
        assert decision.is_rewrite
        assert decision.target_id == "bron-uuid"
        assert decision.alias == "Old Bron"

    async def test_accept_when_judge_says_accept(self):
        brain = _MockBrain('{"action": "accept"}')
        judge = EntityDedupJudge(client=brain)
        existing = NPCState(id="bron-uuid", name="Bron", last_seen_turn=5)
        decision = await judge.judge_add_npc(
            proposed_name="Stranger",
            proposed_description="hooded figure on the road",
            narrator_prose="A stranger approaches",
            existing_npcs=[existing],
            current_turn=6,
        )
        assert decision.action == "accept"
        assert not decision.is_rewrite

    async def test_default_safe_on_brain_error(self):
        """Judge defaults to ACCEPT when the brain call raises."""
        class _BrokenBrain:
            async def chat(self, **kwargs):
                raise RuntimeError("simulated brain failure")
        judge = EntityDedupJudge(client=_BrokenBrain())
        existing = NPCState(id="bron-uuid", name="Bron", last_seen_turn=5)
        decision = await judge.judge_add_npc(
            proposed_name="Marta",
            proposed_description="herbalist",
            narrator_prose="...",
            existing_npcs=[existing],
            current_turn=6,
        )
        assert decision.action == "accept"

    async def test_default_safe_on_unparseable_response(self):
        """Garbage response from brain → accept."""
        brain = _MockBrain("definitely not json {{{{")
        judge = EntityDedupJudge(client=brain)
        existing = NPCState(id="bron-uuid", name="Bron", last_seen_turn=5)
        decision = await judge.judge_add_npc(
            proposed_name="X",
            proposed_description="Y",
            narrator_prose="prose",
            existing_npcs=[existing],
            current_turn=6,
        )
        assert decision.action == "accept"

    async def test_rewrite_without_target_id_treated_as_accept(self):
        """Malformed rewrite (no target_id) defaults safe → accept."""
        brain = _MockBrain('{"action": "rewrite"}')
        judge = EntityDedupJudge(client=brain)
        existing = NPCState(id="bron-uuid", name="Bron", last_seen_turn=5)
        decision = await judge.judge_add_npc(
            proposed_name="Bron",
            proposed_description="innkeeper",
            narrator_prose="prose",
            existing_npcs=[existing],
            current_turn=6,
        )
        assert decision.action == "accept"

    async def test_extra_text_around_json_still_parses(self):
        """Some models add prose around the JSON — extract it anyway."""
        brain = _MockBrain(
            'Sure, here is my answer: {"action": "rewrite", "target_id": "bron-uuid", "alias": "old bron"} okay'
        )
        judge = EntityDedupJudge(client=brain)
        existing = NPCState(id="bron-uuid", name="Bron", last_seen_turn=5)
        decision = await judge.judge_add_npc(
            proposed_name="old bron",
            proposed_description="innkeeper",
            narrator_prose="prose",
            existing_npcs=[existing],
            current_turn=6,
        )
        assert decision.is_rewrite
        assert decision.target_id == "bron-uuid"

    async def test_brain_receives_registry_in_prompt(self):
        """Sanity: the judge sends both the proposed NPC AND the registry
        slice to the brain. Without the registry, the brain has nothing
        to dedup against."""
        brain = _MockBrain('{"action": "accept"}')
        judge = EntityDedupJudge(client=brain)
        existing = NPCState(id="b-1", name="Bron", description="innkeeper", last_seen_turn=5)
        await judge.judge_add_npc(
            proposed_name="Old Bron",
            proposed_description="aged innkeeper",
            narrator_prose="Old Bron grunted",
            existing_npcs=[existing],
            current_turn=6,
        )
        # Verify the prompt contained both
        user_msg = next(m for m in brain.last_messages if m["role"] == "user")
        assert "Old Bron" in user_msg["content"]
        assert "Bron" in user_msg["content"]
        assert "b-1" in user_msg["content"]
        # Verify json_mode was requested (Gemini-style structured output)
        assert brain.last_kwargs.get("json_mode") is True
