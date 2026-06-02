"""Tests for Phase 4: state extractor + dedup judge integration.

Two surfaces:

1. ``EXTRACTION_PROMPT`` schema — instructs the brain about the new
   ``NPCUpdate`` fields (``id``, ``new_name``, ``add_aliases``) and
   steers it toward updates over new_npcs when prose paraphrases an
   already-registered entity.

2. ``DMOrchestrator._dedup_extractor_new_npcs`` — runs each
   ``delta.new_npcs`` entry through ``EntityDedupJudge`` before the
   delta is applied. High-confidence dedup → drop the entry from
   ``new_npcs`` and append an ``NPCUpdate(id=..., add_aliases=[...])``
   so the alias is recorded against the existing record. Default safe
   on any judge error.
"""
from __future__ import annotations
from dataclasses import dataclass
from unittest.mock import MagicMock

import pytest

from dnd_bot.game.world_state import (
    WorldState, NPCState, NPCUpdate, StateDelta,
)
from dnd_bot.llm.extractors.state_extractor import EXTRACTION_PROMPT
from dnd_bot.llm.extractors import dedup_judge as dedup_judge_module
from dnd_bot.llm.extractors.dedup_judge import EntityDedupJudge
from dnd_bot.llm.orchestrator import DMOrchestrator


# ── Mock brain client (mirrors the pattern in test_dedup_judge.py) ──


@dataclass
class _MockResp:
    content: str = ""


class _MockBrain:
    def __init__(self, response_content: str):
        self.response_content = response_content
        self.call_count = 0

    async def chat(self, messages, **kwargs):
        self.call_count += 1
        return _MockResp(content=self.response_content)


# ── Part C1: Prompt schema ─────────────────────────────────────────


class TestExtractionPromptSchema:
    """Phase 4 Part A: the prompt teaches the brain about id-keyed
    updates. These tests pin the prompt content so a regression that
    silently drops the new fields is caught."""

    def test_prompt_mentions_id_keyed_updates(self):
        # The npc_updates section must mention id-based lookup
        assert "npc_updates" in EXTRACTION_PROMPT
        # Locate the npc_updates block and verify it teaches id usage
        idx = EXTRACTION_PROMPT.find("npc_updates")
        assert idx >= 0
        block = EXTRACTION_PROMPT[idx:idx + 1500]
        assert "id" in block
        assert "UUID" in block or "uuid" in block

    def test_prompt_mentions_new_name(self):
        """Rename workflow: prose establishes a new identity for an
        existing record. ``new_name`` moves the old name to aliases."""
        assert "new_name" in EXTRACTION_PROMPT

    def test_prompt_mentions_add_aliases(self):
        """Paraphrase workflow: identity unchanged, alias recorded."""
        assert "add_aliases" in EXTRACTION_PROMPT

    def test_prompt_steers_prefer_updates_over_new_npcs(self):
        """The brain must be told to prefer npc_updates over new_npcs
        when prose paraphrases an entity that's already registered."""
        text = EXTRACTION_PROMPT.lower()
        assert "prefer" in text
        # The instruction explicitly cites paraphrase as the canonical
        # case for using add_aliases over new_npcs
        assert "paraphras" in text


# ── Part C2 + C3: Orchestrator dedup pass ──────────────────────────


def _make_orchestrator() -> DMOrchestrator:
    """Build a DMOrchestrator with all heavy deps mocked.

    The dedup pass only touches the dedup_judge module-level singleton,
    so we mock the four constructor deps to skip ollama init."""
    return DMOrchestrator(
        narrator=MagicMock(),
        adjudicator=MagicMock(),
        rules=MagicMock(),
        client=MagicMock(),
    )


@pytest.fixture
def reset_dedup_singleton():
    """Restore the dedup_judge module singleton after each test so a
    test's mock doesn't leak into later cases."""
    saved = dedup_judge_module._JUDGE
    yield
    dedup_judge_module._JUDGE = saved


@pytest.mark.asyncio
class TestExtractorDedupRewrite:
    """Phase 4 Part B: orchestrator runs extractor's new_npcs through
    the brain dedup judge BEFORE apply_delta."""

    async def test_rewrite_drops_new_npc_and_appends_alias_update(
        self, reset_dedup_singleton
    ):
        """Canonical paraphrase scenario:
        - World has Bram (uuid bram-uuid)
        - Extractor proposed Old Bram as a fresh NPC
        - Judge confidently says rewrite → bram-uuid, alias Old Bram
        Expected: new_npcs is empty; npc_updates contains exactly one
        NPCUpdate(id=bram-uuid, add_aliases=[Old Bram]).
        """
        ws = WorldState(turn=2, current_location="village square")
        ws.npcs["bram-uuid"] = NPCState(
            id="bram-uuid", name="Bram",
            description="turnip cart owner",
            location="village square",
            last_seen_turn=2,
        )
        delta = StateDelta(
            new_npcs=[NPCState(name="Old Bram", description="aged turnip vendor")]
        )
        dedup_judge_module._JUDGE = EntityDedupJudge(client=_MockBrain(
            '{"action": "rewrite", "target_id": "bram-uuid", "alias": "Old Bram"}'
        ))

        orch = _make_orchestrator()
        result = await orch._dedup_extractor_new_npcs(
            delta, narrator_prose="Old Bram leaned on his cart, eyeing the square.",
            world_state=ws,
        )

        assert result.new_npcs == [], "rewritten entry must be dropped from new_npcs"
        assert len(result.npc_updates) == 1
        update = result.npc_updates[0]
        assert isinstance(update, NPCUpdate)
        assert update.id == "bram-uuid"
        assert update.add_aliases == ["Old Bram"]

    async def test_after_apply_delta_world_has_one_record_with_alias(
        self, reset_dedup_singleton
    ):
        """End-to-end through apply_delta: the world state ends with
        exactly one NPC, alias accumulated. No fragmentation."""
        ws = WorldState(turn=2)
        ws.npcs["bram-uuid"] = NPCState(
            id="bram-uuid", name="Bram",
            description="turnip cart owner", last_seen_turn=2,
        )
        delta = StateDelta(
            new_npcs=[NPCState(name="Old Bram", description="aged turnip vendor")]
        )
        dedup_judge_module._JUDGE = EntityDedupJudge(client=_MockBrain(
            '{"action": "rewrite", "target_id": "bram-uuid", "alias": "Old Bram"}'
        ))

        orch = _make_orchestrator()
        delta = await orch._dedup_extractor_new_npcs(delta, "prose", ws)
        rejections = ws.apply_delta(delta)

        assert rejections == []
        assert len(ws.npcs) == 1
        survivor = ws.npcs["bram-uuid"]
        assert survivor.name == "Bram"
        assert "Old Bram" in survivor.aliases

    async def test_accept_preserves_genuinely_new_npc(self, reset_dedup_singleton):
        """Judge accept (genuinely new) → new_npcs survives, no
        alias-update side effect on existing entities."""
        ws = WorldState(turn=2)
        ws.npcs["existing-uuid"] = NPCState(
            id="existing-uuid", name="Tomas",
            description="wandering bard", last_seen_turn=2,
        )
        delta = StateDelta(
            new_npcs=[NPCState(name="Marta", description="herbalist")]
        )
        dedup_judge_module._JUDGE = EntityDedupJudge(
            client=_MockBrain('{"action": "accept"}')
        )

        orch = _make_orchestrator()
        result = await orch._dedup_extractor_new_npcs(
            delta, "Marta tended her herb stall.", ws,
        )

        assert len(result.new_npcs) == 1
        assert result.new_npcs[0].name == "Marta"
        # No spurious updates appended for the existing entity
        assert all(u.id != "existing-uuid" for u in result.npc_updates)

    async def test_judge_target_not_in_world_state_keeps_original(
        self, reset_dedup_singleton
    ):
        """Hallucinated target_id (judge invented an id that doesn't
        exist) → default-safe → keep the original new_npcs entry."""
        ws = WorldState(turn=2)
        ws.npcs["real-uuid"] = NPCState(id="real-uuid", name="Real", last_seen_turn=2)
        delta = StateDelta(
            new_npcs=[NPCState(name="Stranger", description="hooded figure")]
        )
        dedup_judge_module._JUDGE = EntityDedupJudge(client=_MockBrain(
            '{"action": "rewrite", "target_id": "ghost-uuid", "alias": "X"}'
        ))

        orch = _make_orchestrator()
        result = await orch._dedup_extractor_new_npcs(delta, "prose", ws)

        assert len(result.new_npcs) == 1
        assert result.new_npcs[0].name == "Stranger"

    async def test_judge_error_keeps_original(self, reset_dedup_singleton):
        """Brain client raising → default safe → new_npcs preserved."""
        class _BrokenBrain:
            async def chat(self, messages, **kwargs):
                raise RuntimeError("simulated brain failure")

        ws = WorldState(turn=2)
        ws.npcs["x"] = NPCState(id="x", name="Existing", last_seen_turn=2)
        delta = StateDelta(
            new_npcs=[NPCState(name="Newcomer", description="fresh face")]
        )
        dedup_judge_module._JUDGE = EntityDedupJudge(client=_BrokenBrain())

        orch = _make_orchestrator()
        result = await orch._dedup_extractor_new_npcs(delta, "prose", ws)

        assert len(result.new_npcs) == 1
        assert result.new_npcs[0].name == "Newcomer"

    async def test_mixed_batch_some_rewritten_some_accepted(
        self, reset_dedup_singleton
    ):
        """When extractor emits multiple new_npcs in one turn, each is
        judged independently. Verify the judge is called per-entry and
        the surviving / rewritten split is correct.

        Brain returns the SAME canned response for every call, so this
        test pins both: the rewrite path fires (bram-uuid for first
        entry, fictional or harmless second result for second). Use a
        mock that swaps responses by call count.
        """
        ws = WorldState(turn=2)
        ws.npcs["bram-uuid"] = NPCState(
            id="bram-uuid", name="Bram", description="vendor",
            last_seen_turn=2,
        )

        # Two new_npcs proposed by the extractor in one turn
        delta = StateDelta(
            new_npcs=[
                NPCState(name="Old Bram", description="aged vendor"),
                NPCState(name="Marta", description="herbalist"),
            ]
        )

        # Stateful mock: rewrite first, accept second
        responses = [
            '{"action": "rewrite", "target_id": "bram-uuid", "alias": "Old Bram"}',
            '{"action": "accept"}',
        ]
        class _SeqBrain:
            def __init__(self):
                self.i = 0
            async def chat(self, messages, **kwargs):
                content = responses[self.i] if self.i < len(responses) else responses[-1]
                self.i += 1
                return _MockResp(content=content)

        dedup_judge_module._JUDGE = EntityDedupJudge(client=_SeqBrain())

        orch = _make_orchestrator()
        result = await orch._dedup_extractor_new_npcs(delta, "prose", ws)

        # Marta survives; Old Bram becomes alias-update on bram-uuid
        assert len(result.new_npcs) == 1
        assert result.new_npcs[0].name == "Marta"
        assert len(result.npc_updates) == 1
        assert result.npc_updates[0].id == "bram-uuid"
        assert result.npc_updates[0].add_aliases == ["Old Bram"]

    async def test_preserves_existing_npc_updates(self, reset_dedup_singleton):
        """If the extractor already produced npc_updates entries (e.g.,
        a separate disposition change for a different NPC), the dedup
        pass must not clobber them — appended-only semantics."""
        ws = WorldState(turn=2)
        ws.npcs["bram-uuid"] = NPCState(
            id="bram-uuid", name="Bram", last_seen_turn=2,
        )
        ws.npcs["other-uuid"] = NPCState(
            id="other-uuid", name="Other", last_seen_turn=2,
        )
        delta = StateDelta(
            new_npcs=[NPCState(name="Old Bram", description="aged")],
            npc_updates=[NPCUpdate(id="other-uuid", disposition="hostile")],
        )
        dedup_judge_module._JUDGE = EntityDedupJudge(client=_MockBrain(
            '{"action": "rewrite", "target_id": "bram-uuid", "alias": "Old Bram"}'
        ))

        orch = _make_orchestrator()
        result = await orch._dedup_extractor_new_npcs(delta, "prose", ws)

        # Both updates present
        assert len(result.npc_updates) == 2
        ids = [u.id for u in result.npc_updates]
        assert "other-uuid" in ids
        assert "bram-uuid" in ids
        # Pre-existing update unmodified
        other_update = next(u for u in result.npc_updates if u.id == "other-uuid")
        assert other_update.disposition == "hostile"
