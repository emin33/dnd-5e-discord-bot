"""Pins for the 2026-07-16 narrator effect-pipeline fix sweep.

Four defects, one pin each (every test fails if its fix is reverted):

A (DF-12)  _process_proposed_effects gated post-execution work on
           ``result.success`` only, so idempotency hits
           (``was_duplicate=True``) re-ran ``world_store.apply_effect``
           and re-appended to ``_last_executed_effects`` — double-applying
           to WorldState/KG on retries.
B          The drop-item branch of _handle_inventory returned narrative
           success without ever touching the inventory row.
C          Effects with ``requires_confirmation`` were logged then
           ``continue``d — silently discarded (no confirmation UI exists;
           the tool path never sets the flag). Parity: execute with a
           warning that confirmation was auto-approved.
D (DF-22)  set_session never reset the narrator scratchpad, so session B
           inherited session A's tensions/moods on the process-wide
           singleton. Cleared only when the session key actually changes.
"""

import pytest

from dnd_bot.llm import orchestrator as orchestrator_module
from dnd_bot.llm.brains.base import BrainContext
from dnd_bot.llm.effects import EffectExecutionResult, EffectType, ProposedEffect
from dnd_bot.llm.orchestrator import DMOrchestrator, TriageResult
from dnd_bot.game.session import GameSession
from dnd_bot.game.world_state import WorldState
from dnd_bot.models import InventoryItem


def _async_return(value):
    async def _coro():
        return value
    return _coro()


class _StubExecutor:
    """Succeeds every effect; flags duplicates from the second call on.

    Simulates the real executor's idempotency store: same effect retried
    within a turn executes as a no-op with ``was_duplicate=True``.
    """

    def __init__(self):
        self.executed: list[ProposedEffect] = []
        self.acting_character_id = None

    async def execute(self, effect, idempotency_key=None):
        self.executed.append(effect)
        return EffectExecutionResult(
            effect=effect,
            success=True,
            was_duplicate=len(self.executed) > 1,
        )


@pytest.fixture
def orch() -> DMOrchestrator:
    session = GameSession(id="sess", channel_id=901_101, guild_id=1, campaign_id="camp")
    session.world_state = WorldState(current_location="Tavern")
    orchestrator = DMOrchestrator()
    orchestrator.set_session(session)
    return orchestrator


def _spawn_effect(**overrides) -> ProposedEffect:
    kwargs = dict(
        effect_type=EffectType.SPAWN_OBJECT,
        object_name="Rusty Key",
        object_description="a rusty iron key",
    )
    kwargs.update(overrides)
    return ProposedEffect(**kwargs)


# ── A (DF-12): idempotency hits must not double-apply ────────────────────────


@pytest.mark.asyncio
async def test_duplicate_effect_does_not_reapply_to_world_state():
    session = GameSession(id="sess", channel_id=901_102, guild_id=1, campaign_id="camp")
    session.world_state = WorldState(current_location="Tavern")
    orch = DMOrchestrator()
    orch.set_session(session)
    orch._effect_executor = _StubExecutor()
    context = BrainContext(campaign_id="camp", session_id="sess")

    # First pass: the effect lands in WorldState once.
    await orch._process_proposed_effects([_spawn_effect()], context, "msg-1")
    assert session.world_state.scene_items == {"Rusty Key": "a rusty iron key"}
    assert session.world_state.recent_transfers == [
        "a rusty iron key appeared in the scene"
    ]
    assert [e.effect_type for e in orch._last_executed_effects] == [
        EffectType.SPAWN_OBJECT
    ]

    # Retry (executor reports the idempotency hit): NO second WorldState
    # apply, NO transfer-log repeat, NO KG-bridge append.
    await orch._process_proposed_effects([_spawn_effect()], context, "msg-1")
    assert session.world_state.recent_transfers == [
        "a rusty iron key appeared in the scene"
    ]
    assert orch._last_executed_effects == []


# ── B: drop must resolve the item and actually remove it ─────────────────────


class _FakeInventoryRepo:
    def __init__(self, items: list[InventoryItem]):
        self.items = items
        self.removed: list[tuple[str, int]] = []

    async def get_all_items(self, character_id: str) -> list[InventoryItem]:
        return [i for i in self.items if i.character_id == character_id]

    async def remove_item(self, item_id: str, quantity: int = 1) -> bool:
        self.removed.append((item_id, quantity))
        return True


@pytest.fixture
def drop_setup(mock_character, monkeypatch):
    session = GameSession(id="sess", channel_id=901_103, guild_id=1, campaign_id="camp")
    session.add_player(12345, "Test Hero", mock_character)
    orch = DMOrchestrator()
    orch.set_session(session)

    lantern = InventoryItem(
        character_id=mock_character.id,
        item_index="brass-lantern",
        item_name="Brass Lantern",
        quantity=2,
    )
    repo = _FakeInventoryRepo([lantern])
    monkeypatch.setattr(
        orchestrator_module, "get_inventory_repo", lambda: _async_return(repo)
    )
    return orch, repo, lantern


@pytest.mark.asyncio
async def test_drop_removes_resolved_item_from_inventory(drop_setup):
    orch, repo, lantern = drop_setup
    triage = TriageResult(
        action_type="inventory", reasoning="", item_name="brass lantern",
    )

    result = await orch._handle_inventory(
        triage, "Test Hero",
        BrainContext(campaign_id="camp", session_id="sess"),
        "I drop the brass lantern",
    )

    assert result["operation"] == "drop"
    assert result["success"] is True
    assert result["item"] == "Brass Lantern"
    assert result["quantity"] == 1
    # The inventory row was actually touched — the reverted handler never
    # called the repo at all.
    assert repo.removed == [(lantern.id, 1)]
    assert result["tool_calls"][0]["name"] == "remove_item"


@pytest.mark.asyncio
async def test_drop_of_item_not_in_inventory_fails_honestly(drop_setup):
    orch, repo, _ = drop_setup
    triage = TriageResult(
        action_type="inventory", reasoning="", item_name="ruby amulet",
    )

    result = await orch._handle_inventory(
        triage, "Test Hero",
        BrainContext(campaign_id="camp", session_id="sess"),
        "I drop the ruby amulet",
    )

    assert result["operation"] == "drop"
    # Reverted handler claimed success for items the character never had.
    assert result["success"] is False
    assert "not found in inventory" in result["error"]
    assert repo.removed == []


# ── C: requires_confirmation executes (auto-approved), not discarded ─────────


@pytest.mark.asyncio
async def test_requires_confirmation_effect_executes_with_warning(orch):
    executor = _StubExecutor()
    orch._effect_executor = executor
    effect = _spawn_effect(
        requires_confirmation=True,
        confirmation_prompt="Accept the rusty key?",
    )

    await orch._process_proposed_effects(
        [effect], BrainContext(campaign_id="camp", session_id="sess"), "msg-1"
    )

    # Reverted behavior `continue`d before execution: executed == [] and
    # nothing reached WorldState while the narration claimed it happened.
    assert len(executor.executed) == 1
    assert executor.executed[0].requires_confirmation is True
    assert orch._current_session.world_state.scene_items == {
        "Rusty Key": "a rusty iron key"
    }
    assert [e.effect_type for e in orch._last_executed_effects] == [
        EffectType.SPAWN_OBJECT
    ]


# ── D (DF-22): scratchpad resets when the session key changes ─────────────────


def test_scratchpad_cleared_when_session_key_changes():
    orch = DMOrchestrator()
    session_a = GameSession(id="a", channel_id=901_104, guild_id=1, campaign_id="camp")
    session_b = GameSession(id="b", channel_id=901_105, guild_id=1, campaign_id="camp")

    orch.set_session(session_a)
    orch.scratchpad_note("tension", "The innkeeper is hiding something.")
    assert orch.scratchpad_context() != ""

    orch.set_session(session_b)
    assert orch._scratchpad == []
    assert orch.scratchpad_context() == ""


def test_scratchpad_survives_same_session_reset():
    orch = DMOrchestrator()
    session_a = GameSession(id="a", channel_id=901_106, guild_id=1, campaign_id="camp")

    orch.set_session(session_a)
    orch.scratchpad_note("npc_mood", "The guard is suspicious of Elara.")

    # Re-set of the SAME session (and a transient detach) keeps continuity.
    orch.set_session(session_a)
    assert len(orch._scratchpad) == 1
    orch.set_session(None)
    orch.set_session(session_a)
    assert len(orch._scratchpad) == 1
