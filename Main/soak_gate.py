"""Multi-run threshold gate over harness manifests (the matrix's soak row).

One green run proves a path; the promotion matrix requires a stable pass
rate, a bounded cost, and a bounded p95 turn latency across repeated runs.
This aggregator takes N manifest paths — ``test_long_horizon`` manifests
and/or ``test_tool_reliability`` reports, freely mixed — and enforces:

- pass rate  >= ``--pass-rate``  (long_horizon: ``verdict == "PASS"``;
  tool_reliability: every gate true — those artifacts store no verdict);
- per-run cost <= ``--max-cost-usd`` (fails closed on incomplete pricing
  unless ``--allow-incomplete-cost``);
- per-run p95 whole-turn latency <= ``--max-p95-turn-s`` (fails closed on
  missing per-turn samples unless ``--allow-missing-latency``).

Exit 0 only when every enforced threshold holds; violations name the run
and, for latency, the offending turns. Deliberately stdlib-only so the
gate runs anywhere the artifacts do.

Usage::

    python soak_gate.py --pass-rate 0.8 --max-cost-usd 0.40 \
        --max-p95-turn-s 60 data/long_horizon/*.manifest.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional


def percentile(values: list[float], p: float) -> Optional[float]:
    """Linear interpolation on the sorted sample; None on empty input.

    Mirrors test_long_horizon._percentiles (kept local so this gate stays
    stdlib-only and importable without the harness's dotenv/session deps).
    """
    vals = sorted(v for v in values if v is not None)
    if not vals:
        return None
    if len(vals) == 1:
        return vals[0]
    idx = p * (len(vals) - 1)
    lo = int(idx)
    hi = min(lo + 1, len(vals) - 1)
    return vals[lo] + (vals[hi] - vals[lo]) * (idx - lo)


@dataclass
class RunSummary:
    path: str
    kind: str                      # "long_horizon" | "tool_reliability"
    run_id: str
    passed: bool
    verdict: str                   # PASS/FAIL/UNTRUSTED/... or gates:n/m
    turns: Optional[int]
    cost_usd: Optional[float]
    cost_complete: bool
    # (turn_number, seconds) whole-turn samples; empty when unavailable.
    turn_latencies: list[tuple[int, float]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    @property
    def p95_turn_s(self) -> Optional[float]:
        return percentile([seconds for _, seconds in self.turn_latencies], 0.95)


def _long_horizon_latencies(manifest_path: Path, doc: dict) -> tuple[list[tuple[int, float]], list[str]]:
    """Per-turn samples from the sibling .jsonl; manifest p95 as fallback."""
    notes: list[str] = []
    candidates = []
    stem = doc.get("stem")
    if stem:
        candidates.append(manifest_path.parent / f"{stem}.jsonl")
    recorded = doc.get("jsonl")
    if recorded:
        recorded_path = Path(str(recorded))
        candidates.append(recorded_path)
        candidates.append(manifest_path.parent / recorded_path.name)

    for candidate in candidates:
        try:
            if not candidate.exists():
                continue
            samples: list[tuple[int, float]] = []
            with candidate.open(encoding="utf-8") as handle:
                for line in handle:
                    line = line.strip()
                    if not line:
                        continue
                    record = json.loads(line)
                    elapsed = record.get("elapsed")
                    if isinstance(elapsed, (int, float)):
                        samples.append((int(record.get("turn") or 0), float(elapsed)))
            if samples:
                return samples, notes
        except (OSError, json.JSONDecodeError, ValueError) as e:
            notes.append(f"jsonl unreadable ({candidate.name}: {e})")

    p95_ms = ((doc.get("report") or {}).get("latency_turn_ms") or {}).get("p95")
    if isinstance(p95_ms, (int, float)):
        notes.append("per-turn jsonl missing; using manifest p95 as a single sample")
        return [(0, float(p95_ms) / 1000.0)], notes
    notes.append("no per-turn latency data")
    return [], notes


def load_run_summary(path: Path) -> RunSummary:
    """Parse one manifest into the gate's normalized shape.

    Unreadable or unrecognizable files load as failed runs rather than
    raising: a soak run that produced a corrupt artifact is a failed run,
    and the gate must count it against the pass rate, not skip it.
    """
    try:
        doc = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(doc, dict):
            raise ValueError("manifest root is not an object")
    except (OSError, json.JSONDecodeError, ValueError) as e:
        return RunSummary(
            path=str(path), kind="unreadable", run_id=path.stem,
            passed=False, verdict=f"UNREADABLE({e})", turns=None,
            cost_usd=None, cost_complete=False,
            notes=[f"unreadable manifest: {e}"],
        )

    if isinstance(doc.get("gates"), dict) and isinstance(doc.get("usage"), dict):
        gates = {name: bool(ok) for name, ok in doc["gates"].items()}
        usage = doc["usage"]
        latencies = [
            (int(row.get("turn") or 0), float(row["elapsed_seconds"]))
            for row in doc.get("turn_rows") or []
            if isinstance(row, dict)
            and isinstance(row.get("elapsed_seconds"), (int, float))
        ]
        notes = [] if latencies else [
            "no per-turn latency data (pre-2026-07-24 tool_reliability artifact)"
        ]
        failed = sorted(name for name, ok in gates.items() if not ok)
        return RunSummary(
            path=str(path),
            kind="tool_reliability",
            run_id=path.stem,
            passed=bool(gates) and all(gates.values()),
            verdict=(
                f"gates:{sum(gates.values())}/{len(gates)}"
                + (f" failed={failed[:3]}" if failed else "")
            ),
            turns=doc.get("turns"),
            cost_usd=usage.get("cost_usd"),
            cost_complete=bool(usage.get("cost_complete")),
            turn_latencies=latencies,
            notes=notes,
        )

    if "scenario" in doc and ("n_turns" in doc or "stem" in doc):
        report = doc.get("report") or {}
        verdict = str(doc.get("verdict") or "")
        notes: list[str] = []
        if not verdict:
            # Seed-pick-only manifest: the run crashed before finalize.
            notes.append("manifest never finalized (no verdict) — counted as failed")
            verdict = "UNFINALIZED"
        latencies, latency_notes = _long_horizon_latencies(path, doc)
        notes.extend(latency_notes)
        return RunSummary(
            path=str(path),
            kind="long_horizon",
            run_id=str(doc.get("stem") or path.stem),
            passed=verdict == "PASS",
            verdict=verdict,
            turns=doc.get("n_turns"),
            cost_usd=report.get("total_cost_usd"),
            cost_complete=bool(report.get("cost_complete")),
            turn_latencies=latencies,
            notes=notes,
        )

    return RunSummary(
        path=str(path), kind="unknown", run_id=path.stem,
        passed=False, verdict="UNRECOGNIZED", turns=None,
        cost_usd=None, cost_complete=False,
        notes=["unrecognized manifest shape — counted as failed"],
    )


@dataclass
class GateThresholds:
    min_pass_rate: Optional[float] = None
    max_cost_usd: Optional[float] = None
    max_p95_turn_s: Optional[float] = None
    min_runs: int = 1
    allow_incomplete_cost: bool = False
    allow_missing_latency: bool = False


@dataclass
class GateReport:
    runs: list[RunSummary]
    thresholds: GateThresholds
    violations: list[str] = field(default_factory=list)

    @property
    def pass_rate(self) -> Optional[float]:
        if not self.runs:
            return None
        return sum(run.passed for run in self.runs) / len(self.runs)

    @property
    def ok(self) -> bool:
        return not self.violations


def evaluate_gate(runs: list[RunSummary], thresholds: GateThresholds) -> GateReport:
    report = GateReport(runs=runs, thresholds=thresholds)
    violations = report.violations

    if len(runs) < max(1, thresholds.min_runs):
        violations.append(
            f"only {len(runs)} run(s) supplied; gate requires at least "
            f"{max(1, thresholds.min_runs)}"
        )

    # A corrupt or unrecognized artifact is never a healthy gate input:
    # violate unconditionally, so a latency-only invocation with
    # --allow-missing-latency cannot exit 0 over garbage (adversarial
    # review finding).
    for run in runs:
        if run.kind in ("unreadable", "unknown"):
            violations.append(f"{run.run_id}: {run.verdict} manifest at {run.path}")

    if thresholds.min_pass_rate is not None and runs:
        rate = report.pass_rate or 0.0
        if rate < thresholds.min_pass_rate:
            failed = [f"{run.run_id} ({run.verdict})" for run in runs if not run.passed]
            violations.append(
                f"pass rate {rate:.2f} < {thresholds.min_pass_rate:.2f}; "
                f"failed runs: {', '.join(failed[:6])}"
                + (f" (+{len(failed) - 6} more)" if len(failed) > 6 else "")
            )

    for run in runs:
        if thresholds.max_cost_usd is not None:
            if run.cost_usd is None:
                violations.append(f"{run.run_id}: no cost recorded")
            else:
                if not run.cost_complete and not thresholds.allow_incomplete_cost:
                    violations.append(
                        f"{run.run_id}: cost incomplete (unpriced models); "
                        "pass --allow-incomplete-cost to waive"
                    )
                if run.cost_usd > thresholds.max_cost_usd:
                    violations.append(
                        f"{run.run_id}: cost ${run.cost_usd:.4f} > "
                        f"budget ${thresholds.max_cost_usd:.4f}"
                    )

        if thresholds.max_p95_turn_s is not None:
            p95 = run.p95_turn_s
            if p95 is None:
                if not thresholds.allow_missing_latency:
                    violations.append(
                        f"{run.run_id}: no per-turn latency samples; "
                        "pass --allow-missing-latency to waive"
                    )
            elif p95 > thresholds.max_p95_turn_s:
                slow_turns = sorted(
                    (turn for turn, seconds in run.turn_latencies
                     if seconds > thresholds.max_p95_turn_s),
                )
                violations.append(
                    f"{run.run_id}: p95 turn latency {p95:.1f}s > "
                    f"{thresholds.max_p95_turn_s:.1f}s "
                    f"(turns over bound: {slow_turns[:8]}"
                    + (f" +{len(slow_turns) - 8} more)" if len(slow_turns) > 8 else ")")
                )

    return report


def render_report(report: GateReport) -> None:
    thresholds = report.thresholds
    print("=" * 72)
    print("SOAK GATE")
    print("=" * 72)
    enforced = []
    if thresholds.min_pass_rate is not None:
        enforced.append(f"pass-rate>={thresholds.min_pass_rate:.2f}")
    if thresholds.max_cost_usd is not None:
        enforced.append(f"cost<=${thresholds.max_cost_usd:.4f}")
    if thresholds.max_p95_turn_s is not None:
        enforced.append(f"p95-turn<={thresholds.max_p95_turn_s:.1f}s")
    print(f"thresholds: {', '.join(enforced) or '(none)'}; "
          f"min-runs={max(1, thresholds.min_runs)}")
    print()
    for run in report.runs:
        p95 = run.p95_turn_s
        cost = f"${run.cost_usd:.4f}" if run.cost_usd is not None else "-"
        if not run.cost_complete and run.cost_usd is not None:
            cost += "*"
        print(
            f"  [{'PASS' if run.passed else 'FAIL'}] {run.run_id}"
            f"  kind={run.kind} turns={run.turns if run.turns is not None else '-'}"
            f" cost={cost}"
            f" p95={f'{p95:.1f}s' if p95 is not None else '-'}"
            f" ({run.verdict})"
        )
        for note in run.notes:
            print(f"         note: {note}")
    rate = report.pass_rate
    print(f"\npass rate: {f'{rate:.2f}' if rate is not None else '-'} "
          f"({sum(r.passed for r in report.runs)}/{len(report.runs)})")
    pooled = percentile(
        [seconds for run in report.runs for _, seconds in run.turn_latencies],
        0.95,
    )
    if pooled is not None:
        print(f"pooled p95 turn latency: {pooled:.1f}s")
    if report.violations:
        print("\nVIOLATIONS:")
        for violation in report.violations:
            print(f"  - {violation}")
    print(f"\nGATE: {'PASS' if report.ok else 'FAIL'}")


def main(argv: Optional[list[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("manifests", nargs="+", metavar="MANIFEST",
                        help="long_horizon manifest / tool_reliability report paths")
    parser.add_argument("--pass-rate", type=float, dest="pass_rate",
                        help="Minimum pass rate across runs, 0..1")
    parser.add_argument("--max-cost-usd", type=float, dest="max_cost_usd",
                        help="Per-run cost budget in USD")
    parser.add_argument("--max-p95-turn-s", type=float, dest="max_p95_turn_s",
                        help="Per-run p95 whole-turn latency bound in seconds")
    parser.add_argument("--min-runs", type=int, default=1,
                        help="Fail unless at least this many manifests load")
    parser.add_argument("--allow-incomplete-cost", action="store_true",
                        help="Do not fail runs whose cost excludes unpriced models")
    parser.add_argument("--allow-missing-latency", action="store_true",
                        help="Do not fail runs without per-turn latency samples")
    args = parser.parse_args(argv)

    if args.pass_rate is None and args.max_cost_usd is None and args.max_p95_turn_s is None:
        parser.error(
            "no thresholds given; pass at least one of --pass-rate, "
            "--max-cost-usd, --max-p95-turn-s"
        )

    # PowerShell/cmd pass glob patterns through literally; expand them here
    # so the documented `data/long_horizon/*.manifest.json` invocation works
    # on the project's primary shell. A pattern matching nothing stays
    # literal and loads as an unreadable (failing) run.
    manifest_paths: list[str] = []
    for raw in args.manifests:
        if any(ch in raw for ch in "*?["):
            import glob as _glob
            matches = sorted(_glob.glob(raw))
            manifest_paths.extend(matches or [raw])
        else:
            manifest_paths.append(raw)

    runs = [load_run_summary(Path(p)) for p in manifest_paths]
    thresholds = GateThresholds(
        min_pass_rate=args.pass_rate,
        max_cost_usd=args.max_cost_usd,
        max_p95_turn_s=args.max_p95_turn_s,
        min_runs=args.min_runs,
        allow_incomplete_cost=args.allow_incomplete_cost,
        allow_missing_latency=args.allow_missing_latency,
    )
    report = evaluate_gate(runs, thresholds)
    render_report(report)
    return 0 if report.ok else 1


if __name__ == "__main__":
    sys.exit(main())
