"""Unit tests for fact supersession (the append-only ledger fix).

Motivating live case: narrative-grader severe contradiction in run
20260723_003823 T22 — "wax soft and warm" coexisted with "wax dry and
brittle" because facts never retired.
"""

from types import SimpleNamespace

import pytest

from dnd_bot.game.fact_supersession import (
    FactSupersessionJudge,
    MAX_CANDIDATES,
    candidate_indices,
    fact_anchor_words,
)
from dnd_bot.game.world_state import StateDelta, WorldState
from dnd_bot.game.world_store import WorldStateStore


class TestAnchors:
    def test_stopwords_and_short_words_excluded(self):
        anchors = fact_anchor_words("The wax seal on it is soft and warm")
        assert "wax" in anchors and "seal" in anchors and "soft" in anchors
        assert "the" not in anchors and "is" not in anchors
        assert "on" not in anchors and "it" not in anchors

    def test_shared_subject_word_pairs_candidates(self):
        established = [
            "Sera Vellik rests at the Gutter-Step Market.",
            "The eastern gate is a quarter mile away.",
            "The wax seal on the letter is soft and warm.",
        ]
        indices = candidate_indices(
            "The wax smudge is dry and brittle.", established
        )
        assert indices == [2]

    def test_no_shared_anchor_no_candidates(self):
        assert candidate_indices(
            "Rain falls upward over Veyr.", ["Kael owns a shortbow."]
        ) == []

    def test_recent_first_and_capped(self):
        established = [f"The bell number {i} tolls at dusk." for i in range(20)]
        indices = candidate_indices("The bell cracked at dawn.", established)
        assert len(indices) == MAX_CANDIDATES
        assert indices[0] == 19  # most recent first


class _FakeClient:
    def __init__(self, content):
        self._content = content
        self.calls = []

    async def chat(self, **kwargs):
        self.calls.append(kwargs)
        return SimpleNamespace(content=self._content)


class TestJudge:
    @pytest.mark.asyncio
    async def test_valid_indices_retire_candidates(self):
        judge = FactSupersessionJudge(client=_FakeClient('{"supersedes": [2]}'))
        retired = await judge.judge("new", ["keep me", "retire me"])
        assert retired == ["retire me"]

    @pytest.mark.asyncio
    async def test_empty_response_keeps_all(self):
        judge = FactSupersessionJudge(client=_FakeClient(""))
        assert await judge.judge("new", ["a"]) == []

    @pytest.mark.asyncio
    async def test_out_of_range_and_junk_filtered(self):
        judge = FactSupersessionJudge(
            client=_FakeClient('{"supersedes": [0, 3, "x", 1]}')
        )
        assert await judge.judge("new", ["a", "b"]) == ["a"]

    @pytest.mark.asyncio
    async def test_client_error_keeps_all(self):
        class _Boom:
            async def chat(self, **kwargs):
                raise RuntimeError("down")

        judge = FactSupersessionJudge(client=_Boom())
        assert await judge.judge("new", ["a"]) == []

    @pytest.mark.asyncio
    async def test_thinking_disabled_and_json_mode(self):
        client = _FakeClient('{"supersedes": []}')
        await FactSupersessionJudge(client=client).judge("new", ["a"])
        kwargs = client.calls[0]
        assert kwargs["think"] is False
        assert kwargs["json_mode"] is True


class TestRetireFact:
    def test_moves_to_archive_with_provenance(self):
        state = WorldState(turn=22, established_facts=["old truth"])
        assert state.retire_fact("old truth", superseded_by="new truth")
        assert state.established_facts == []
        assert state.superseded_facts == [{
            "fact": "old truth", "superseded_by": "new truth", "turn": 22,
        }]

    def test_missing_fact_is_noop(self):
        state = WorldState()
        assert state.retire_fact("never existed", superseded_by="x") is False
        assert state.superseded_facts == []


class TestStoreSeam:
    @pytest.mark.asyncio
    async def test_wax_case_retires_stale_state(self, monkeypatch):
        state = WorldState(turn=22, established_facts=[
            "Sera Vellik rests at the Gutter-Step Market.",
            "The wax seal on the letter is soft and warm.",
        ])

        class _FakeJudge:
            def __init__(self, client=None):
                pass

            async def judge(self, new_fact, candidates):
                return [c for c in candidates if "soft and warm" in c]

        monkeypatch.setattr(
            "dnd_bot.game.fact_supersession.FactSupersessionJudge", _FakeJudge
        )
        delta = StateDelta(new_facts=["The wax smudge is dry and brittle."])
        await WorldStateStore(state).apply_delta(delta)

        assert "The wax smudge is dry and brittle." in state.established_facts
        assert "The wax seal on the letter is soft and warm." not in (
            state.established_facts
        )
        assert state.superseded_facts[0]["superseded_by"] == (
            "The wax smudge is dry and brittle."
        )
        # Unrelated fact untouched.
        assert "Sera Vellik rests at the Gutter-Step Market." in (
            state.established_facts
        )

    @pytest.mark.asyncio
    async def test_no_candidates_never_consults_judge(self, monkeypatch):
        state = WorldState(established_facts=["Kael owns a shortbow."])

        class _Exploding:
            def __init__(self, client=None):
                raise AssertionError("judge must not be constructed")

        monkeypatch.setattr(
            "dnd_bot.game.fact_supersession.FactSupersessionJudge", _Exploding
        )
        delta = StateDelta(new_facts=["Rain falls upward over Veyr."])
        await WorldStateStore(state).apply_delta(delta)
        assert len(state.established_facts) == 2

    @pytest.mark.asyncio
    async def test_judge_init_failure_fails_open(self, monkeypatch):
        state = WorldState(established_facts=["The bell tolls at dusk."])

        class _Broken:
            def __init__(self, client=None):
                raise RuntimeError("no brain configured")

        monkeypatch.setattr(
            "dnd_bot.game.fact_supersession.FactSupersessionJudge", _Broken
        )
        delta = StateDelta(new_facts=["The bell cracked at dawn."])
        await WorldStateStore(state).apply_delta(delta)
        # Both facts kept — fail-open.
        assert len(state.established_facts) == 2
        assert state.superseded_facts == []
