"""Pin the multi-run threshold gate (soak_gate.py) on synthetic manifests.

The gate is the matrix's soak/reliability row: stable pass rate, bounded
cost, bounded p95 turn latency across N runs, exit nonzero naming the
violating run/turn. Both manifest dialects are covered, plus the observed
schema-drift shapes (unfinalized manifests, artifacts without per-turn
timing, unpriced-model cost).
"""

import json

import pytest

from soak_gate import (
    GateThresholds,
    evaluate_gate,
    load_run_summary,
    main,
    percentile,
)


def _write_long_horizon(
    tmp_path,
    stem="20260724_000000_prof",
    verdict="PASS",
    cost=0.25,
    cost_complete=True,
    elapsed=(10.0, 20.0, 30.0),
    with_jsonl=True,
    p95_ms=30000.0,
    finalized=True,
):
    doc = {
        "stem": stem,
        "profile": "prof",
        "scenario": "deep_seeded_callback",
        "session_id": "session-1",
        "n_turns": len(elapsed),
        "jsonl": str(tmp_path / f"{stem}.jsonl"),
    }
    if finalized:
        doc["verdict"] = verdict
        doc["report"] = {
            "total_cost_usd": cost,
            "cost_complete": cost_complete,
            "latency_turn_ms": {"p50": 20000.0, "p95": p95_ms},
        }
    path = tmp_path / f"{stem}.manifest.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    if with_jsonl:
        lines = [
            json.dumps({"turn": i + 1, "elapsed": e})
            for i, e in enumerate(elapsed)
        ]
        (tmp_path / f"{stem}.jsonl").write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_tool_reliability(
    tmp_path,
    name="20260724_000001_prof",
    gates=None,
    cost=0.02,
    cost_complete=True,
    elapsed=(5.0, 6.0, 7.0),
    with_turn_timing=True,
):
    doc = {
        "profile": "prof",
        "scenario": "baseline",
        "turns": len(elapsed),
        "usage": {"cost_usd": cost, "cost_complete": cost_complete},
        "gates": dict(gates) if gates is not None else {"a": True, "b": True},
        "turn_rows": [
            {"turn": i + 1}
            | ({"elapsed_seconds": e} if with_turn_timing else {})
            for i, e in enumerate(elapsed)
        ],
    }
    path = tmp_path / f"{name}.json"
    path.write_text(json.dumps(doc), encoding="utf-8")
    return path


def test_percentile_edges():
    assert percentile([], 0.95) is None
    assert percentile([7.0], 0.95) == 7.0
    assert percentile([1.0, 2.0, 3.0, 4.0], 0.5) == pytest.approx(2.5)


def test_all_thresholds_pass_on_green_mixed_runs(tmp_path):
    runs = [
        load_run_summary(_write_long_horizon(tmp_path)),
        load_run_summary(_write_tool_reliability(tmp_path)),
    ]
    report = evaluate_gate(runs, GateThresholds(
        min_pass_rate=1.0, max_cost_usd=0.50, max_p95_turn_s=60.0, min_runs=2,
    ))
    assert report.ok, report.violations
    assert report.pass_rate == 1.0


def test_pass_rate_violation_names_the_failing_run(tmp_path):
    runs = [
        load_run_summary(_write_long_horizon(tmp_path, stem="run_pass")),
        load_run_summary(_write_long_horizon(
            tmp_path, stem="run_fail", verdict="FAIL",
        )),
    ]
    report = evaluate_gate(runs, GateThresholds(min_pass_rate=0.9))
    assert not report.ok
    assert any("run_fail" in v and "FAIL" in v for v in report.violations)


def test_untrusted_and_invalid_verdicts_count_as_failed(tmp_path):
    for verdict in ("UNTRUSTED", "INVALID(player-error)"):
        run = load_run_summary(_write_long_horizon(
            tmp_path, stem=f"run_{verdict[:4]}", verdict=verdict,
        ))
        assert not run.passed
        assert run.verdict == verdict


def test_tool_reliability_pass_is_all_gates(tmp_path):
    green = load_run_summary(_write_tool_reliability(
        tmp_path, name="tr_green", gates={"a": True, "b": True},
    ))
    red = load_run_summary(_write_tool_reliability(
        tmp_path, name="tr_red", gates={"a": True, "b": False},
    ))
    assert green.passed
    assert not red.passed
    assert "b" in red.verdict  # the failed gate is named


def test_cost_over_budget_names_run(tmp_path):
    run = load_run_summary(_write_long_horizon(
        tmp_path, stem="run_pricey", cost=0.90,
    ))
    report = evaluate_gate([run], GateThresholds(max_cost_usd=0.40))
    assert not report.ok
    assert any("run_pricey" in v and "0.9000" in v for v in report.violations)


def test_incomplete_cost_fails_closed_and_is_waivable(tmp_path):
    run = load_run_summary(_write_long_horizon(
        tmp_path, stem="run_unpriced", cost=0.10, cost_complete=False,
    ))
    strict = evaluate_gate([run], GateThresholds(max_cost_usd=0.40))
    assert any("incomplete" in v for v in strict.violations)
    waived = evaluate_gate([run], GateThresholds(
        max_cost_usd=0.40, allow_incomplete_cost=True,
    ))
    assert waived.ok


def test_p95_violation_names_run_and_offending_turns(tmp_path):
    # 18 fast turns and two 400s outliers: sorted sample 18 of 20 is already
    # an outlier, so interpolated p95 is 400s — far above a 60s bound — and
    # turns 19/20 are the ones over it.
    elapsed = tuple([10.0] * 18 + [400.0, 400.0])
    run = load_run_summary(_write_long_horizon(
        tmp_path, stem="run_slow", elapsed=elapsed,
    ))
    report = evaluate_gate([run], GateThresholds(max_p95_turn_s=60.0))
    assert not report.ok
    violation = next(v for v in report.violations if "run_slow" in v)
    assert "p95" in violation and "19" in violation and "20" in violation


def test_missing_latency_fails_closed_and_is_waivable(tmp_path):
    run = load_run_summary(_write_tool_reliability(
        tmp_path, name="tr_untimed", with_turn_timing=False,
    ))
    assert run.turn_latencies == []
    strict = evaluate_gate([run], GateThresholds(max_p95_turn_s=60.0))
    assert any("tr_untimed" in v and "latency" in v for v in strict.violations)
    waived = evaluate_gate([run], GateThresholds(
        max_p95_turn_s=60.0, allow_missing_latency=True,
    ))
    assert waived.ok


def test_long_horizon_missing_jsonl_falls_back_to_manifest_p95(tmp_path):
    run = load_run_summary(_write_long_horizon(
        tmp_path, stem="run_nojsonl", with_jsonl=False, p95_ms=45000.0,
    ))
    assert run.p95_turn_s == pytest.approx(45.0)
    assert any("manifest p95" in note for note in run.notes)


def test_unfinalized_manifest_counts_as_failed(tmp_path):
    run = load_run_summary(_write_long_horizon(
        tmp_path, stem="run_crashed", finalized=False,
    ))
    assert not run.passed
    assert run.verdict == "UNFINALIZED"


def test_unreadable_manifest_counts_as_failed(tmp_path):
    path = tmp_path / "garbage.json"
    path.write_text("{not json", encoding="utf-8")
    run = load_run_summary(path)
    assert not run.passed
    report = evaluate_gate([run], GateThresholds(min_pass_rate=1.0))
    assert not report.ok


def test_min_runs_guard(tmp_path):
    run = load_run_summary(_write_long_horizon(tmp_path))
    report = evaluate_gate([run], GateThresholds(min_pass_rate=0.5, min_runs=3))
    assert not report.ok
    assert any("at least 3" in v for v in report.violations)


def test_main_exit_codes_and_threshold_requirement(tmp_path, capsys):
    green = _write_long_horizon(tmp_path, stem="run_green")
    red = _write_long_horizon(tmp_path, stem="run_red", verdict="FAIL")

    assert main(["--pass-rate", "1.0", str(green)]) == 0
    assert main(["--pass-rate", "1.0", str(green), str(red)]) == 1
    out = capsys.readouterr().out
    assert "GATE: FAIL" in out and "run_red" in out

    with pytest.raises(SystemExit):
        main([str(green)])  # no thresholds given
