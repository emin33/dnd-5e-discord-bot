"""Pin evaluate_restart_convergence — the restart checkpoint's verdict.

The checkpoint's captures are plain dicts, so every divergence class the
matrix cares about (roster drift, graph drift, ledger drift, a rebuilt
registry that lost a canonical join, an unindexed described entity, a
dirty audit, a failed recovery) is pinned here synthetically; the live
seam itself is exercised by tests/integration/test_restart_recovery.py.
"""

from test_long_horizon import evaluate_restart_convergence


def _capture(**overrides):
    base = {
        "session_id": "session-1",
        "session_key": "discord:999999",
        "world_turn": 12,
        "current_location": "Copper Finch",
        "roster": {"id-mara": "Mara Venn", "id-orris": "Orris"},
        "alive_roster_ids": ["id-mara", "id-orris"],
        "dead_roster": {"id-bram": "Old Bram"},
        "established_facts": ["fact a", "fact b"],
        "superseded_facts": ["old fact"],
        "kg_node_ids": ["copper-finch", "id-mara", "id-orris"],
        "described_kg_ids": ["id-mara", "id-orris"],
        "indexed_entity_ids": ["id-mara", "id-orris"],
        "scene_npc_links": ["id-mara", "id-orris"],
        "audit": {"passed": True, "violations": [], "coverage": {}, "counts": {}},
    }
    base.update(overrides)
    return base


def _checkpoint(**overrides):
    checkpoint = {
        "turn": 12,
        "pre": _capture(),
        "post": _capture(),
        "restart": {
            "recovered": True,
            "recovered_count": 1,
            "session_id": "session-1",
        },
    }
    checkpoint.update(overrides)
    return checkpoint


def _by_name(results):
    return {r.name: r for r in results}


def test_converged_checkpoint_passes_both_gates():
    results = _by_name(evaluate_restart_convergence(_checkpoint()))
    assert set(results) == {
        "restart_recovery_succeeded", "restart_projection_convergence",
    }
    assert results["restart_recovery_succeeded"].passed
    assert results["restart_projection_convergence"].passed
    assert "converged" in results["restart_projection_convergence"].detail


def test_missing_checkpoint_fails_both_gates():
    # The run never reached its restart turn (crash before the boundary).
    results = _by_name(evaluate_restart_convergence({"turn": 40}))
    assert not results["restart_recovery_succeeded"].passed
    assert not results["restart_projection_convergence"].passed


def test_failed_recovery_fails_even_when_captures_match():
    results = _by_name(evaluate_restart_convergence(_checkpoint(
        restart={"recovered": False, "recovered_count": 0, "error": "no snapshot"},
    )))
    assert not results["restart_recovery_succeeded"].passed
    assert not results["restart_projection_convergence"].passed


def test_multiple_recovered_sessions_fail_recovery_gate():
    results = _by_name(evaluate_restart_convergence(_checkpoint(
        restart={"recovered": True, "recovered_count": 2},
    )))
    assert not results["restart_recovery_succeeded"].passed


def test_session_identity_swap_fails_recovery_gate():
    results = _by_name(evaluate_restart_convergence(_checkpoint(
        post=_capture(session_id="session-2"),
    )))
    assert not results["restart_recovery_succeeded"].passed


def test_lost_roster_npc_is_named():
    results = _by_name(evaluate_restart_convergence(_checkpoint(
        post=_capture(
            roster={"id-mara": "Mara Venn"},
            scene_npc_links=["id-mara"],
        ),
    )))
    gate = results["restart_projection_convergence"]
    assert not gate.passed
    assert "id-orris" in gate.detail


def test_renamed_roster_npc_is_named():
    results = _by_name(evaluate_restart_convergence(_checkpoint(
        post=_capture(roster={"id-mara": "Mara Venn", "id-orris": "Orris Vane"}),
    )))
    gate = results["restart_projection_convergence"]
    assert not gate.passed
    assert "renamed" in gate.detail and "id-orris" in gate.detail


def test_world_turn_or_location_drift_fails():
    for overrides in ({"world_turn": 11}, {"current_location": "Ash Gate"}):
        results = _by_name(evaluate_restart_convergence(_checkpoint(
            post=_capture(**overrides),
        )))
        assert not results["restart_projection_convergence"].passed


def test_fact_ledger_drift_fails():
    results = _by_name(evaluate_restart_convergence(_checkpoint(
        post=_capture(established_facts=["fact a"]),
    )))
    gate = results["restart_projection_convergence"]
    assert not gate.passed
    assert "established_facts" in gate.detail


def test_kg_node_drift_names_the_node():
    results = _by_name(evaluate_restart_convergence(_checkpoint(
        post=_capture(kg_node_ids=["copper-finch", "id-mara"]),
    )))
    gate = results["restart_projection_convergence"]
    assert not gate.passed
    assert "id-orris" in gate.detail


def test_registry_missing_canonical_join_fails():
    results = _by_name(evaluate_restart_convergence(_checkpoint(
        post=_capture(scene_npc_links=["id-mara"]),
    )))
    gate = results["restart_projection_convergence"]
    assert not gate.passed
    assert "scene-registry" in gate.detail and "id-orris" in gate.detail


def test_unindexed_described_entity_fails():
    results = _by_name(evaluate_restart_convergence(_checkpoint(
        post=_capture(indexed_entity_ids=["id-mara"]),
    )))
    gate = results["restart_projection_convergence"]
    assert not gate.passed
    assert "unindexed" in gate.detail and "id-orris" in gate.detail


def test_unreadable_vector_index_fails_closed():
    results = _by_name(evaluate_restart_convergence(_checkpoint(
        post=_capture(indexed_entity_ids=None),
    )))
    gate = results["restart_projection_convergence"]
    assert not gate.passed
    assert "vector index unreadable" in gate.detail


def test_dirty_audit_on_either_side_fails():
    dirty = {"passed": False, "violations": ["pinned_fact_is_superseded: x"]}
    for side in ("pre", "post"):
        results = _by_name(evaluate_restart_convergence(_checkpoint(
            **{side: _capture(audit=dirty)},
        )))
        gate = results["restart_projection_convergence"]
        assert not gate.passed
        assert f"{side}-restart consistency audit" in gate.detail


def test_dead_roster_rebuild_is_not_asserted():
    # Dead rows sync to the npc DB only at graceful end_session, so the
    # recovered dead-roster union legitimately differs after a crash.
    results = _by_name(evaluate_restart_convergence(_checkpoint(
        post=_capture(dead_roster={}),
    )))
    assert results["restart_projection_convergence"].passed


def test_dead_roster_npc_needs_no_registry_join():
    # Recovery deliberately refuses to re-register dead roster NPCs
    # (session.py "don't resurrect the corpse"): a pre-restart death stays
    # in world.npcs (alive=False) yet must not fail the join expectation.
    # Adversarial review of 5d9ccaf, confirmed HIGH.
    dead_included = {
        "roster": {"id-mara": "Mara Venn", "id-orris": "Orris",
                   "id-slain": "Slain Bravo"},
        "alive_roster_ids": ["id-mara", "id-orris"],
    }
    results = _by_name(evaluate_restart_convergence(_checkpoint(
        pre=_capture(**dead_included),
        post=_capture(**dead_included),  # scene_npc_links still alive-only
    )))
    assert results["restart_projection_convergence"].passed, (
        results["restart_projection_convergence"].detail
    )


def test_post_audit_scene_link_dangling_is_tolerated_other_violations_fatal():
    # Recovery preloads the registry from the whole alive campaign DB while
    # world.npcs stays scene-scoped, so H4 scene_link_dangling is an
    # expected post-restart transient (closed by the next move's rescope).
    # Adversarial review of 5d9ccaf, confirmed HIGH.
    dangling_only = {
        "passed": False,
        "violations": ["scene_link_dangling: Distant Ferrier -> id-far"],
    }
    results = _by_name(evaluate_restart_convergence(_checkpoint(
        post=_capture(audit=dangling_only),
    )))
    assert results["restart_projection_convergence"].passed, (
        results["restart_projection_convergence"].detail
    )

    # The same violation on the PRE side is a live-state defect and fails.
    results = _by_name(evaluate_restart_convergence(_checkpoint(
        pre=_capture(audit=dangling_only),
    )))
    assert not results["restart_projection_convergence"].passed

    # Mixed post-side violations keep failing on the non-dangling class.
    mixed = {
        "passed": False,
        "violations": [
            "scene_link_dangling: Distant Ferrier -> id-far",
            "pinned_fact_is_superseded: x",
        ],
    }
    results = _by_name(evaluate_restart_convergence(_checkpoint(
        post=_capture(audit=mixed),
    )))
    gate = results["restart_projection_convergence"]
    assert not gate.passed
    assert "pinned_fact_is_superseded" in gate.detail
