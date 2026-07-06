"""End-to-end stress test for the NPC dedup refactor (Phases 1-7).

The unit suite already covers each layer in isolation
(``test_world_state.py``, ``test_knowledge_graph.py``, ``test_dedup_judge.py``,
``test_state_extractor.py``). This file fills the remaining gaps:

* Narrator-side ``_dedup_rewrite`` (the primary hook for the fix)
* ``_sync_effect_to_world_state`` REF_ENTITY + UPDATE_ENTITY recency branches
* Idempotency-key invariance across rewrite (research-grounded invariant)
* ``WorldState.to_yaml()`` id-first rendering with ``aliases`` /
  ``last_seen_turn`` (the surface the narrator actually reads)
* ``_find_npc`` alias-fallback resolution chain
* ``apply_delta`` rename → moves old name into aliases
* The full paraphrase pile-up scenario — the regression fence for the
  original "9 add_npc calls for one hooded character" bug

Mocks the brain client so every test runs in <100ms and survives in CI.
"""
from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import yaml

from dnd_bot.game.world_store import WorldStateStore
from dnd_bot.game.world_state import (
    WorldState, NPCState, NPCUpdate, StateDelta,
)
from dnd_bot.llm.effects import (
    EffectType, ProposedEffect, build_effect_idempotency_key,
)
from dnd_bot.llm.extractors import dedup_judge as dedup_judge_module
from dnd_bot.llm.extractors.dedup_judge import EntityDedupJudge
from dnd_bot.llm.orchestrator import DMOrchestrator


# ── Mock infrastructure ────────────────────────────────────────────


@dataclass
class _MockResp:
    content: str = ""


class _MockBrain:
    """Brain client stub with canned content."""
    def __init__(self, response_content: str):
        self.response_content = response_content
        self.call_count = 0

    async def chat(self, messages, **kwargs):
        self.call_count += 1
        return _MockResp(content=self.response_content)


class _SeqBrain:
    """Brain client stub that cycles a sequence of responses."""
    def __init__(self, responses: list[str]):
        self.responses = responses
        self.i = 0

    async def chat(self, messages, **kwargs):
        content = self.responses[self.i] if self.i < len(self.responses) else self.responses[-1]
        self.i += 1
        return _MockResp(content=content)


class _BrokenBrain:
    async def chat(self, messages, **kwargs):
        raise RuntimeError("simulated brain failure")


def _make_orchestrator(world_state: WorldState | None = None) -> DMOrchestrator:
    """Build a DMOrchestrator with all heavy deps mocked and an
    attached mock session carrying a real WorldState."""
    orch = DMOrchestrator(
        narrator=MagicMock(),
        adjudicator=MagicMock(),
        client=MagicMock(),
    )
    if world_state is not None:
        orch._current_session = SimpleNamespace(world_state=world_state)
    return orch


@pytest.fixture
def reset_dedup_singleton():
    """Restore the module-level ``_JUDGE`` after each test so a test's
    mock can't leak into later cases."""
    saved = dedup_judge_module._JUDGE
    yield
    dedup_judge_module._JUDGE = saved


# ── 1-4: Narrator-side _dedup_rewrite ──────────────────────────────


@pytest.mark.asyncio
class TestDedupRewrite:
    """Phase 6's primary hook. ADD_NPC effects routed through the brain
    dedup judge before validation. The whole bug fix lives here."""

    async def test_rewrite_converts_add_npc_to_ref_entity(self, reset_dedup_singleton):
        ws = WorldState(turn=6)
        ws.npcs["bron-uuid"] = NPCState(
            id="bron-uuid", name="Bron", description="innkeeper",
            last_seen_turn=5,
        )
        effect = ProposedEffect(
            effect_type=EffectType.ADD_NPC,
            npc_name="Old Bron",
            npc_description="aged innkeeper",
            dialogue_indices=[1, 2],
            dialogue_emotions=["wry", "tired"],
        )
        dedup_judge_module._JUDGE = EntityDedupJudge(client=_MockBrain(
            '{"action": "rewrite", "target_id": "bron-uuid", "alias": "Old Bron"}'
        ))

        orch = _make_orchestrator(ws)
        rewritten = await orch._dedup_rewrite(
            effect, "Old Bron leaned on the bar", ws,
        )

        # Effect type flipped, identity points at the existing entity
        assert rewritten.effect_type == EffectType.REF_ENTITY
        assert rewritten.ref_entity_id == "bron-uuid"
        assert rewritten.ref_alias_used == "Old Bron"
        # Dialogue tracking carried through — those quotes still belong to Bron
        assert rewritten.dialogue_indices == [1, 2]
        assert rewritten.dialogue_emotions == ["wry", "tired"]
        # The judge's alias was stashed on the existing NPC so future
        # paraphrases match faster
        assert "Old Bron" in ws.npcs["bron-uuid"].aliases

    async def test_accept_leaves_effect_unchanged(self, reset_dedup_singleton):
        ws = WorldState(turn=6)
        ws.npcs["bron-uuid"] = NPCState(
            id="bron-uuid", name="Bron", description="innkeeper",
            last_seen_turn=5,
        )
        effect = ProposedEffect(
            effect_type=EffectType.ADD_NPC,
            npc_name="Marta",
            npc_description="herbalist with a wagon of dried roots",
        )
        dedup_judge_module._JUDGE = EntityDedupJudge(
            client=_MockBrain('{"action": "accept"}')
        )

        orch = _make_orchestrator(ws)
        result = await orch._dedup_rewrite(effect, "Marta waves hello.", ws)

        # Identity preserved — same object reference, no mutation
        assert result is effect
        assert result.effect_type == EffectType.ADD_NPC
        assert result.npc_name == "Marta"
        # No spurious alias added to the unrelated existing NPC
        assert ws.npcs["bron-uuid"].aliases == []

    async def test_default_safe_on_judge_error(self, reset_dedup_singleton):
        ws = WorldState(turn=6)
        ws.npcs["bron-uuid"] = NPCState(id="bron-uuid", name="Bron", last_seen_turn=5)
        effect = ProposedEffect(
            effect_type=EffectType.ADD_NPC,
            npc_name="Stranger",
            npc_description="hooded figure",
        )
        dedup_judge_module._JUDGE = EntityDedupJudge(client=_BrokenBrain())

        orch = _make_orchestrator(ws)
        result = await orch._dedup_rewrite(effect, "prose", ws)

        # Brain blew up → keep the original effect. False negatives are
        # recoverable; false positives are not.
        assert result is effect
        assert result.effect_type == EffectType.ADD_NPC

    async def test_default_safe_on_hallucinated_target_id(self, reset_dedup_singleton):
        """The judge may invent a target_id that isn't in the registry.
        Treat as 'accept' rather than blindly trusting it (this would
        produce a REF_ENTITY pointing at nothing, then a downstream
        bridge error)."""
        ws = WorldState(turn=6)
        ws.npcs["real-uuid"] = NPCState(id="real-uuid", name="Real", last_seen_turn=5)
        effect = ProposedEffect(
            effect_type=EffectType.ADD_NPC,
            npc_name="Newcomer",
            npc_description="fresh face",
        )
        dedup_judge_module._JUDGE = EntityDedupJudge(client=_MockBrain(
            '{"action": "rewrite", "target_id": "ghost-uuid", "alias": "X"}'
        ))

        orch = _make_orchestrator(ws)
        result = await orch._dedup_rewrite(effect, "prose", ws)

        assert result is effect
        assert result.effect_type == EffectType.ADD_NPC


# ── 5: Idempotency key invariance under rewrite ────────────────────


class TestIdempotencyKeyInvariance:
    """Research-grounded invariant (Airbyte / Graphiti): idempotency
    keys must come from deterministic position, never from LLM output.
    If the rewrite produced a different key, retries after a rewrite
    would create duplicate effects."""

    def test_key_format_does_not_depend_on_effect_content(self):
        e_add = ProposedEffect(
            effect_type=EffectType.ADD_NPC,
            npc_name="Hooded Figure",
        )
        e_ref = ProposedEffect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id="bron-uuid",
            ref_alias_used="Old Bron",
        )

        # Same (campaign, message, index) → same key, regardless of
        # whether the effect is the original add_npc or the rewrite.
        key_before = build_effect_idempotency_key("camp-1", "msg-42", 3)
        key_after = build_effect_idempotency_key("camp-1", "msg-42", 3)
        assert key_before == key_after
        assert key_before == "camp-1:msg-42:3"
        # The function takes ints/strings, not the effect — so the LLM's
        # creative paraphrase can't smuggle entropy into the key
        _ = e_add, e_ref  # used for narrative; behavior is in the key itself

    def test_keys_differ_across_effect_positions(self):
        """Two effects emitted in the same turn must get distinct keys
        so each can fire independently — even if one is rewritten."""
        k0 = build_effect_idempotency_key("c", "m", 0)
        k1 = build_effect_idempotency_key("c", "m", 1)
        assert k0 != k1


# ── 6-7: _sync_effect_to_world_state recency branches ──────────────


class TestSyncEffectRecencyBranches:
    """Phase 5: when narrator emits REF_ENTITY or UPDATE_ENTITY, the
    sync layer bumps ``last_seen_turn`` so relevance-windowed roster
    selection keeps the entity surfaced for follow-up turns."""

    def test_ref_entity_bumps_recency_and_accumulates_alias(self):
        ws = WorldState(turn=10)
        ws.npcs["bron-uuid"] = NPCState(
            id="bron-uuid", name="Bron", last_seen_turn=5,
        )
        orch = _make_orchestrator(ws)
        effect = ProposedEffect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id="bron-uuid",
            ref_alias_used="the innkeeper",
        )

        WorldStateStore(ws).apply_effect(effect)

        npc = ws.npcs["bron-uuid"]
        assert npc.last_seen_turn == 10
        assert "the innkeeper" in npc.aliases

    def test_ref_entity_alias_not_duplicated_on_repeat(self):
        ws = WorldState(turn=10)
        ws.npcs["bron-uuid"] = NPCState(
            id="bron-uuid", name="Bron",
            aliases=["the innkeeper"],
            last_seen_turn=5,
        )
        orch = _make_orchestrator(ws)
        effect = ProposedEffect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id="bron-uuid",
            ref_alias_used="the innkeeper",
        )

        WorldStateStore(ws).apply_effect(effect)
        WorldStateStore(ws).apply_effect(effect)

        assert ws.npcs["bron-uuid"].aliases == ["the innkeeper"]

    def test_ref_entity_alias_matching_canonical_name_not_added(self):
        ws = WorldState(turn=10)
        ws.npcs["bron-uuid"] = NPCState(
            id="bron-uuid", name="Bron", last_seen_turn=5,
        )
        orch = _make_orchestrator(ws)
        effect = ProposedEffect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id="bron-uuid",
            ref_alias_used="Bron",  # same as canonical name
        )

        WorldStateStore(ws).apply_effect(effect)

        # Recency still bumps but no alias spam
        assert ws.npcs["bron-uuid"].last_seen_turn == 10
        assert ws.npcs["bron-uuid"].aliases == []

    def test_update_entity_bumps_recency(self):
        ws = WorldState(turn=12)
        ws.npcs["bron-uuid"] = NPCState(
            id="bron-uuid", name="Bron",
            disposition="neutral",
            last_seen_turn=3,
        )
        orch = _make_orchestrator(ws)
        effect = ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="bron-uuid",
            update_disposition="hostile",
        )

        WorldStateStore(ws).apply_effect(effect)

        npc = ws.npcs["bron-uuid"]
        assert npc.last_seen_turn == 12
        assert npc.disposition == "hostile"


# ── 8: apply_delta rename moves old name → aliases ─────────────────


class TestApplyDeltaRename:
    """Phase 1 invariant: ``NPCUpdate.new_name`` performs a rename and
    moves the previous name into ``aliases``. Pins the contract that the
    extractor's prompt is steered toward (and that misfired on the
    "Marta — bowl for the elf" turn — the regression we want to catch
    if the prompt drifts back)."""

    def test_new_name_moves_old_name_to_aliases(self):
        ws = WorldState(turn=4)
        bram = NPCState(id="bram-uuid", name="Bram", last_seen_turn=2)
        ws.npcs[bram.id] = bram

        delta = StateDelta(
            npc_updates=[NPCUpdate(id="bram-uuid", new_name="Bram Hornwood")]
        )
        rejections = ws.apply_delta(delta)

        assert rejections == []
        assert ws.npcs["bram-uuid"].name == "Bram Hornwood"
        assert "Bram" in ws.npcs["bram-uuid"].aliases

    def test_new_name_is_idempotent_when_unchanged(self):
        ws = WorldState(turn=4)
        ws.npcs["x"] = NPCState(id="x", name="Bram", last_seen_turn=2)
        delta = StateDelta(npc_updates=[NPCUpdate(id="x", new_name="Bram")])
        ws.apply_delta(delta)
        assert ws.npcs["x"].name == "Bram"
        assert ws.npcs["x"].aliases == []


# ── 9: _find_npc resolution chain ──────────────────────────────────


class TestFindNpcResolutionChain:
    """Phase 1: ``_find_npc`` resolves by id → name → alias. The alias
    path is the one used by the matcher and the orchestrator's effect
    sync when prose paraphrases an entity. If it regresses, paraphrase
    matching silently falls through to None and the bug returns."""

    def test_id_match(self):
        ws = WorldState()
        ws.npcs["bram-uuid"] = NPCState(id="bram-uuid", name="Bram")
        assert ws._find_npc("bram-uuid") is ws.npcs["bram-uuid"]

    def test_name_match_case_insensitive(self):
        ws = WorldState()
        ws.npcs["bram-uuid"] = NPCState(id="bram-uuid", name="Bram")
        assert ws._find_npc("bram") is ws.npcs["bram-uuid"]
        assert ws._find_npc("BRAM") is ws.npcs["bram-uuid"]

    def test_alias_match_case_insensitive(self):
        ws = WorldState()
        ws.npcs["x"] = NPCState(
            id="x", name="Bram", aliases=["Old Bram", "the dwarf"],
        )
        assert ws._find_npc("the dwarf") is ws.npcs["x"]
        assert ws._find_npc("The Dwarf") is ws.npcs["x"]
        assert ws._find_npc("OLD BRAM") is ws.npcs["x"]

    def test_id_preferred_over_name_collision(self):
        """If id collides with a different NPC's name (pathological but
        possible if someone names an NPC after a UUID), id wins."""
        ws = WorldState()
        ws.npcs["alpha"] = NPCState(id="alpha", name="Alpha")
        ws.npcs["other"] = NPCState(id="other", name="alpha")  # name == "alpha"
        # Looking up "alpha" hits the id-keyed dict first
        assert ws._find_npc("alpha") is ws.npcs["alpha"]

    def test_no_match_returns_none(self):
        ws = WorldState()
        ws.npcs["x"] = NPCState(id="x", name="Bram")
        assert ws._find_npc("Nobody") is None
        assert ws._find_npc("") is None


# ── 10: YAML id-first rendering ────────────────────────────────────


class TestWorldStateYamlIdFirst:
    """Phase 2: the YAML the narrator reads must surface ``id``,
    ``aliases``, and ``last_seen_turn`` for paraphrase resolution to
    work from the narrator side. ``test_world_state.py`` only pins
    ``name`` / ``location`` — these tests pin the new fields."""

    def test_npcs_here_includes_id_aliases_last_seen(self):
        ws = WorldState(turn=10, current_location="tavern")
        ws.npcs["bron-uuid"] = NPCState(
            id="bron-uuid", name="Bron",
            location="tavern",
            aliases=["Old Bron"],
            description="innkeeper",
            last_seen_turn=10,
        )

        data = yaml.safe_load(ws.to_yaml())
        entries = data["npcs_here"]
        bron = next(e for e in entries if e["name"] == "Bron")

        assert bron["id"] == "bron-uuid"
        assert bron["aliases"] == ["Old Bron"]
        assert bron["last_seen_turn"] == 10
        assert "desc" in bron  # description present

    def test_key_npcs_elsewhere_includes_id_aliases_last_seen(self):
        ws = WorldState(turn=10, current_location="tavern")
        ws.npcs["king-uuid"] = NPCState(
            id="king-uuid", name="King Aldric",
            location="castle",
            important=True,
            aliases=["His Majesty"],
            last_seen_turn=7,
        )

        data = yaml.safe_load(ws.to_yaml())
        entries = data["key_npcs_elsewhere"]
        king = next(e for e in entries if e["name"] == "King Aldric")

        assert king["id"] == "king-uuid"
        assert king["aliases"] == ["His Majesty"]
        assert king["last_seen_turn"] == 7
        assert king["location"] == "castle"

    def test_zero_last_seen_omitted(self):
        """``last_seen_turn=0`` (default for freshly minted entities at
        turn 0) is omitted to keep the snapshot tight."""
        ws = WorldState(turn=0, current_location="tavern")
        ws.npcs["x"] = NPCState(
            id="x", name="Fresh", location="tavern", last_seen_turn=0,
        )
        data = yaml.safe_load(ws.to_yaml())
        entry = data["npcs_here"][0]
        assert "last_seen_turn" not in entry


# ── 11: Full paraphrase pile-up — the regression fence ─────────────


@pytest.mark.asyncio
class TestParaphrasePileUpRegression:
    """The pre-fix bug: 9 ``add_npc`` calls across 4 paraphrases of one
    hooded character. The fix routes paraphrases through ``_dedup_rewrite``
    → REF_ENTITY → ``_sync_effect_to_world_state`` accumulates aliases
    instead of fragmenting.

    This test simulates that exact failure path with mocked judge
    decisions and asserts a single NPC survives with all paraphrases
    captured as aliases. If this test fails, the bug is back."""

    async def test_three_paraphrases_collapse_to_one_npc_with_aliases(
        self, reset_dedup_singleton
    ):
        ws = WorldState(turn=1)
        orch = _make_orchestrator(ws)

        # ── Turn 1: first introduction (registry empty, judge fast-path
        # accepts, sync mints the NPCState).
        dedup_judge_module._JUDGE = EntityDedupJudge(
            client=_MockBrain('{"action": "accept"}')
        )
        ws.turn = 1
        first = ProposedEffect(
            effect_type=EffectType.ADD_NPC,
            npc_name="Hooded Figure",
            npc_description="cloaked stranger by the fire",
        )
        # The orchestrator only calls _dedup_rewrite when registry is
        # non-empty; simulate that gating here.
        if ws.npcs:
            first = await orch._dedup_rewrite(first, "prose-1", ws)
        WorldStateStore(ws).apply_effect(first)

        # One NPC now exists, fresh UUID
        assert len(ws.npcs) == 1
        original_id = next(iter(ws.npcs))
        assert ws.npcs[original_id].name == "Hooded Figure"

        # ── Turn 2: paraphrase "Cloaked Figure" — judge rewrites.
        ws.turn = 2
        dedup_judge_module._JUDGE = EntityDedupJudge(client=_MockBrain(
            f'{{"action": "rewrite", "target_id": "{original_id}", "alias": "Cloaked Figure"}}'
        ))
        second = ProposedEffect(
            effect_type=EffectType.ADD_NPC,
            npc_name="Cloaked Figure",
            npc_description="hooded figure by the hearth",
        )
        if ws.npcs:
            second = await orch._dedup_rewrite(second, "prose-2", ws)
        WorldStateStore(ws).apply_effect(second)

        assert second.effect_type == EffectType.REF_ENTITY
        assert len(ws.npcs) == 1, "rewrite must NOT create a new entity"

        # ── Turn 3: another paraphrase "Shadow Figure".
        ws.turn = 3
        dedup_judge_module._JUDGE = EntityDedupJudge(client=_MockBrain(
            f'{{"action": "rewrite", "target_id": "{original_id}", "alias": "Shadow Figure"}}'
        ))
        third = ProposedEffect(
            effect_type=EffectType.ADD_NPC,
            npc_name="Shadow Figure",
            npc_description="the same hooded one",
        )
        if ws.npcs:
            third = await orch._dedup_rewrite(third, "prose-3", ws)
        WorldStateStore(ws).apply_effect(third)

        # ── Verify the regression fence.
        assert len(ws.npcs) == 1, (
            f"Expected 1 NPC after 3 paraphrases; got {len(ws.npcs)}. "
            f"This is the original bug — paraphrases fragmenting into "
            f"separate NPCState records."
        )
        survivor = ws.npcs[original_id]
        assert survivor.name == "Hooded Figure"
        # All paraphrases captured as aliases — order not guaranteed but
        # both must be present
        assert "Cloaked Figure" in survivor.aliases
        assert "Shadow Figure" in survivor.aliases
        # Recency tracked across the pile-up
        assert survivor.last_seen_turn == 3

    async def test_accept_in_pile_up_creates_distinct_npc(
        self, reset_dedup_singleton
    ):
        """When a paraphrase-looking ADD_NPC is actually a genuinely
        different character, the judge must accept and a second NPC
        appears. False positives (merging distinct characters) are
        worse than false negatives, so this path is the safety valve."""
        ws = WorldState(turn=1)
        orch = _make_orchestrator(ws)

        # Seed: Hooded Figure
        dedup_judge_module._JUDGE = EntityDedupJudge(
            client=_MockBrain('{"action": "accept"}')
        )
        first = ProposedEffect(
            effect_type=EffectType.ADD_NPC, npc_name="Hooded Figure",
        )
        WorldStateStore(ws).apply_effect(first)
        assert len(ws.npcs) == 1

        # Turn 2: actually new — old woman by the well, not the hooded one
        ws.turn = 2
        dedup_judge_module._JUDGE = EntityDedupJudge(
            client=_MockBrain('{"action": "accept"}')
        )
        second = ProposedEffect(
            effect_type=EffectType.ADD_NPC,
            npc_name="Marta",
            npc_description="old woman drawing water",
        )
        second_after = await orch._dedup_rewrite(second, "prose", ws)
        WorldStateStore(ws).apply_effect(second_after)

        assert second_after.effect_type == EffectType.ADD_NPC
        assert len(ws.npcs) == 2
        names = {npc.name for npc in ws.npcs.values()}
        assert names == {"Hooded Figure", "Marta"}
