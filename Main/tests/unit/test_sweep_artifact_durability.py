"""Pin: a sweep run always leaves an artifact, even if teardown kills it.

Two live runs (20260725 #3 and #7) completed every turn and then died
silently during session teardown — exit 1, no traceback, empty stderr, and
NO artifact. That is the worst possible failure shape for the promotion
matrix: `soak_gate.py` computes its pass rate over the artifacts it is
handed, so a run that never writes one drops out of the denominator
entirely and *raises* the apparent pass rate. A configuration that crashed
half the time could report 100%.

The report is therefore built and persisted BEFORE `harness.cleanup()`, and
an evaluation failure still writes a crash artifact that soak_gate scores as
a failed run.
"""

from __future__ import annotations

import json

import pytest

import test_tool_reliability as sweep
from soak_gate import load_run_summary


def test_crash_report_is_scored_as_a_failed_run(tmp_path):
    report = sweep._crash_report(
        "prof", "player_state_sweep", "sess-1", 13,
        errors=[{"turn": 8, "error": "no response"}],
        error="RuntimeError: boom", tb="Traceback...",
    )
    path = tmp_path / "20260725_999999_player_state_sweep_prof.json"
    path.write_text(json.dumps(report), encoding="utf-8")

    summary = load_run_summary(path)

    # Recognized as a tool_reliability run (gates + usage), counted FAILED —
    # not skipped, not "unreadable".
    assert summary.kind == "tool_reliability"
    assert summary.passed is False
    assert report["gates"]["run_completed"] is False
    assert any("boom" in str(e) for e in report["errors"])


def test_build_report_runs_standalone_on_an_empty_session():
    """_build_report must not depend on run()'s function-local imports.

    Extracting it from run() left TurnLogReader/NarrativeGovernance/
    resolve_unique_identity behind as locals, so every live run raised
    NameError inside the report builder — caught only because the crash
    artifact recorded the traceback. Exercising the REAL builder (the
    durability test below stubs it) pins the dependency.
    """
    # Reaching the turn-log read proves every name the body needs resolved.
    # A missing log is the expected outcome here; a NameError is the bug.
    with pytest.raises(FileNotFoundError):
        sweep._build_report(
            profile="prof",
            scenario="player_state_sweep",
            session_id="no-such-session",
            actions=["I wait."],
            expected_effects={},
            harness=type("H", (), {"action_log": []})(),
            responses={},
            errors=[],
            world_snapshot={},
            bram_id="bram",
            initial_player_state={},
            final_player_state={},
            elapsed=1.0,
        )


@pytest.mark.asyncio
async def test_artifact_is_written_before_teardown(tmp_path, monkeypatch):
    """The ordering guarantee: cleanup can die and the artifact survives."""
    artifact = tmp_path / "run.json"
    order: list[str] = []

    class _DyingHarness:
        channel_id = 1
        action_log: list = []

        class _Char:
            id = "char-1"
        character = _Char()

        class _Manager:
            def get_session(self, channel_id):
                return None
        manager = _Manager()

        async def cleanup(self):
            order.append("cleanup")
            raise RuntimeError("native teardown crash")

    def _fake_build_report(**kwargs):
        order.append("build_report")
        return {
            "profile": kwargs["profile"], "scenario": kwargs["scenario"],
            "usage": {"cost_usd": 0.0, "cost_complete": True},
            "gates": {"all_turns_returned": True}, "turn_rows": [],
        }

    monkeypatch.setattr(sweep, "_build_report", _fake_build_report)

    # Drive just the finally-block contract: build+persist, then tear down.
    harness = _DyingHarness()
    errors: list[dict] = []
    report = None
    try:
        raise RuntimeError("turn loop died")  # worst case: mid-loop failure
    except RuntimeError:
        pass
    finally:
        report = sweep._build_report(
            profile="p", scenario="player_state_sweep", session_id="s",
            actions=[], expected_effects={}, harness=harness, responses={},
            errors=errors, world_snapshot={}, bram_id="b",
            initial_player_state={}, final_player_state={}, elapsed=1.0,
        )
        artifact.write_text(json.dumps(report), encoding="utf-8")
        try:
            await harness.cleanup()
        except Exception:
            order.append("cleanup_failed")

    assert order == ["build_report", "cleanup", "cleanup_failed"]
    assert artifact.exists()
    assert json.loads(artifact.read_text(encoding="utf-8"))["gates"]
