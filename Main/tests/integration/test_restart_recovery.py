"""Crash-recovery round trip: the restart checkpoint's live seam, offline.

Runs the real TestSession against run-unique isolated storage, seeds
canonical state through the production stores (the tool preflight's own
fixture: world roster, knowledge graph, scene registry, inventory),
persists the per-turn snapshot the pipeline writes each turn, crashes the
in-process state, recovers via GameSessionManager.recover_sessions, and
requires the projection-convergence assertions to hold. No LLM calls —
this is the deterministic pin behind test_long_horizon --restart-at-turn.
"""

import pytest

from test_harness import TestSession as HarnessSession
from test_long_horizon import evaluate_restart_convergence
from test_tool_reliability import _seed_scene


@pytest.mark.asyncio
async def test_process_restart_recovers_convergent_projections():
    session = HarnessSession(
        isolated_storage=True,
        world_setting="A rain-dark tavern town beside the forbidden Ash Gate.",
    )
    assert await session.setup()
    try:
        live = session.manager.get_session(session.channel_id)
        mara_id, _ = await _seed_scene(session)
        live.world_state.turn = 3  # a mid-run turn counter must round-trip

        # The production pipeline persists this snapshot at the end of every
        # processed turn (session.py Step "persist the live world"); the
        # offline test stands in for that seam directly.
        await session.manager._persist_world_snapshot(live)

        pre = await session.capture_projection_state()
        restart = await session.simulate_process_restart()
        post = await session.capture_projection_state()

        assert restart["recovered"], restart
        results = evaluate_restart_convergence(
            {"turn": 3, "pre": pre, "post": post, "restart": restart}
        )
        failures = [f"{r.name}: {r.detail}" for r in results if not r.passed]
        assert not failures, failures

        # The recovered stores are live objects, not just matching captures.
        recovered = session.manager.get_session(session.channel_id)
        assert recovered is not live
        assert recovered.world_state.turn == 3
        assert mara_id in recovered.world_state.npcs
        assert recovered.knowledge_graph is not None
        assert recovered.knowledge_graph.get_entity(mara_id) is not None
    finally:
        await session.cleanup()
