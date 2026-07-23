"""Offline narrative-quality grader for long-form turn logs.

The tool and memory pillars have hard gates (test_long_horizon.py,
test_tool_reliability.py); this closes the third pillar. It reads a
completed run's turn log, computes deterministic prose metrics, then
grades narration in rolling windows with an INDEPENDENT judge (Gemini via
the production client — never the DeepSeek narrator grading itself) on a
five-dimension rubric, and enforces the hard thresholds the readiness plan
committed to: overall average >= 4.0, no dimension mean below 3.0, zero
severe contradictions.

Usage:
    python test_narrative_grader.py --turn-log data/turn_logs/<id>.jsonl
    python test_narrative_grader.py --turn-log <id>            # bare id ok
    python test_narrative_grader.py --turn-log <id> --judge-model gemini-2.5-flash

Exit codes: 0 = all gates pass, 1 = any gate fails or the grade is
incomplete (missing judge responses fail closed).
"""

from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

WINDOW_SIZE = 6
NARRATION_EXCERPT_CHARS = 2600
DIMENSIONS = (
    "continuity",
    "contradiction_free",
    "npc_voice",
    "prose_freshness",
    "player_agency",
)

# Hard gates (LONGFORM_READINESS: "narrative dimensions averaging >=4/5
# with no dimension below 3, zero severe contradictions").
GATE_OVERALL_AVG = 4.0
GATE_DIMENSION_MIN_MEAN = 3.0
# Deterministic repetition ceiling: fraction of 8-grams in a narration
# that already appeared in the previous three narrations. Calibrated
# against the three 2026-07 soaks (healthy runs sit well under this).
GATE_REPEAT_RATIO_MAX = 0.35

JUDGE_SYSTEM = """You are a strict editorial judge for a D&D campaign's \
narration. You grade the DUNGEON MASTER's prose only — the player's \
actions are context, not the subject. Be calibrated: 5 = professional \
published-module quality, 4 = solid with minor nits, 3 = serviceable but \
flawed, 2 = clearly problematic, 1 = broken. Most competent narration \
earns 3-4; reserve 5 for genuinely excellent windows. Ground every flag \
in a specific turn number."""

JUDGE_PROMPT = """## Story so far (grader's running summary)
{story_so_far}

## Window: turns {first_turn}-{last_turn}
{window_block}

Grade THIS WINDOW of DM narration on five dimensions, integer 1-5 each:

- continuity: do scenes, geography, time, and cause-effect follow from \
the story so far and from turn to turn?
- contradiction_free: does any narration contradict established facts, \
prior narration, or itself? List every contradiction found in \
"contradictions" with turn number and severity "minor" (cosmetic slip) \
or "severe" (breaks story logic or retcons established state).
- npc_voice: do named NPCs keep consistent personality, knowledge, and \
manner of speech?
- prose_freshness: is the prose varied in structure and imagery, or \
does it recycle the same sentence shapes, phrases, and beats?
- player_agency: does the narration honor what the player actually \
attempted, without hijacking their action or inventing player intent?

Return ONLY one JSON object:
{{
  "scores": {{"continuity": N, "contradiction_free": N, "npc_voice": N, \
"prose_freshness": N, "player_agency": N}},
  "contradictions": [{{"turn": N, "severity": "minor|severe", \
"detail": "..."}}],
  "flags": ["T<n>: <specific issue>", ...],
  "window_summary": "2-3 sentences a future grader needs: where the \
party is, active NPCs, open threads, key facts established this window."
}}"""


@dataclass
class TurnRow:
    turn: int
    action: str
    narration: str


@dataclass
class WindowGrade:
    first_turn: int
    last_turn: int
    scores: dict[str, int]
    contradictions: list[dict]
    flags: list[str]
    window_summary: str


@dataclass
class GradeReport:
    turn_log: str
    judge_model: str
    windows: list[WindowGrade] = field(default_factory=list)
    repeat_ratios: dict[int, float] = field(default_factory=dict)
    opening_bigrams: dict[str, int] = field(default_factory=dict)
    failed_windows: int = 0

    def dimension_means(self) -> dict[str, float]:
        means = {}
        for dim in DIMENSIONS:
            values = [w.scores.get(dim, 0) for w in self.windows if w.scores.get(dim)]
            means[dim] = round(sum(values) / len(values), 2) if values else 0.0
        return means

    def overall_average(self) -> float:
        means = [m for m in self.dimension_means().values() if m]
        return round(sum(means) / len(means), 2) if means else 0.0

    def severe_contradictions(self) -> list[dict]:
        return [
            c
            for w in self.windows
            for c in w.contradictions
            if str(c.get("severity", "")).lower() == "severe"
        ]

    def worst_repeat_ratio(self) -> float:
        return round(max(self.repeat_ratios.values(), default=0.0), 3)

    def gates(self) -> list[tuple[str, bool, str]]:
        means = self.dimension_means()
        severe = self.severe_contradictions()
        return [
            (
                "judge_coverage_complete",
                self.failed_windows == 0 and bool(self.windows),
                f"windows graded={len(self.windows)}; failed={self.failed_windows}",
            ),
            (
                "overall_average",
                self.overall_average() >= GATE_OVERALL_AVG,
                f"avg={self.overall_average()} (gate >= {GATE_OVERALL_AVG})",
            ),
            (
                "no_dimension_below_minimum",
                bool(means) and all(m >= GATE_DIMENSION_MIN_MEAN for m in means.values()),
                f"means={means} (gate >= {GATE_DIMENSION_MIN_MEAN})",
            ),
            (
                "zero_severe_contradictions",
                not severe,
                f"severe={severe[:5]}",
            ),
            (
                "prose_repetition_bounded",
                self.worst_repeat_ratio() <= GATE_REPEAT_RATIO_MAX,
                f"worst 8-gram repeat ratio={self.worst_repeat_ratio()} "
                f"(gate <= {GATE_REPEAT_RATIO_MAX})",
            ),
        ]

    def passed(self) -> bool:
        return all(ok for _, ok, _ in self.gates())


def load_turn_rows(path: Path) -> list[TurnRow]:
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        record = json.loads(line)
        narration = str((record.get("narrator_response") or {}).get("raw") or "")
        if not narration:
            continue
        rows.append(TurnRow(
            turn=int(record.get("turn") or 0),
            action=str(record.get("action") or ""),
            narration=narration,
        ))
    rows.sort(key=lambda r: r.turn)
    return rows


def _ngrams(text: str, n: int = 8) -> set[tuple[str, ...]]:
    words = re.findall(r"[a-z0-9']+", text.lower())
    return {tuple(words[i:i + n]) for i in range(max(0, len(words) - n + 1))}


def repeat_ratios(rows: list[TurnRow], lookback: int = 3) -> dict[int, float]:
    """Per-turn fraction of narration 8-grams seen in recent narrations.

    Catches the failure the judge can miss inside one window: a narrator
    settling into recycled paragraphs across turns.
    """
    ratios: dict[int, float] = {}
    for index, row in enumerate(rows):
        grams = _ngrams(row.narration)
        if not grams:
            continue
        recent: set[tuple[str, ...]] = set()
        for prior in rows[max(0, index - lookback):index]:
            recent |= _ngrams(prior.narration)
        if not recent:
            continue
        ratios[row.turn] = len(grams & recent) / len(grams)
    return ratios


def opening_bigrams(rows: list[TurnRow]) -> dict[str, int]:
    """How often each two-word narration opener recurs (variety signal)."""
    counts: Counter[str] = Counter()
    for row in rows:
        words = re.findall(r"[A-Za-z']+", row.narration)[:2]
        if len(words) == 2:
            counts[" ".join(w.lower() for w in words)] += 1
    return dict(counts.most_common(8))


def _excerpt(text: str, limit: int) -> str:
    """Trim at a sentence boundary and SAY SO, so the judge never grades
    the grader's own cutoff as a narration defect (first live run flagged
    excerpt boundaries as 'DM ends mid-sentence')."""
    if len(text) <= limit:
        return text
    cut = text[:limit]
    boundary = max(cut.rfind(". "), cut.rfind('."'), cut.rfind("!\n"), cut.rfind(".\n"))
    if boundary > limit // 2:
        cut = cut[:boundary + 1]
    return cut + " [EXCERPT TRUNCATED BY GRADER — do not grade the cutoff]"


def _window_block(rows: list[TurnRow]) -> str:
    parts = []
    for row in rows:
        parts.append(
            f"### Turn {row.turn}\n"
            f"PLAYER: {_excerpt(row.action, 400)}\n"
            f"DM: {_excerpt(row.narration, NARRATION_EXCERPT_CHARS)}"
        )
    return "\n\n".join(parts)


async def grade_run(
    turn_log: Path,
    judge_model: str = "gemini-2.5-flash",
    window_size: int = WINDOW_SIZE,
) -> GradeReport:
    from dnd_bot.llm.client import GeminiClient
    from dnd_bot.llm.json_extract import extract_json_object

    rows = load_turn_rows(turn_log)
    if not rows:
        raise SystemExit(f"No narrated turns found in {turn_log}")

    report = GradeReport(turn_log=str(turn_log), judge_model=judge_model)
    report.repeat_ratios = repeat_ratios(rows)
    report.opening_bigrams = opening_bigrams(rows)

    judge = GeminiClient(model=judge_model)
    story_so_far = "(campaign start — nothing established yet)"

    for start in range(0, len(rows), window_size):
        window = rows[start:start + window_size]
        prompt = JUDGE_PROMPT.format(
            story_so_far=story_so_far,
            first_turn=window[0].turn,
            last_turn=window[-1].turn,
            window_block=_window_block(window),
        )
        try:
            response = await judge.chat(
                messages=[
                    {"role": "system", "content": JUDGE_SYSTEM},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.1,
                max_tokens=1200,
                json_mode=True,
                think=False,
            )
            data, _warnings = extract_json_object(response.content or "")
        except Exception as e:  # judge availability failures fail closed
            print(f"  window T{window[0].turn}-{window[-1].turn}: judge error: {e}")
            data = None
        scores = (data or {}).get("scores") or {}
        if not data or not all(
            isinstance(scores.get(d), int) and 1 <= scores[d] <= 5
            for d in DIMENSIONS
        ):
            report.failed_windows += 1
            continue
        grade = WindowGrade(
            first_turn=window[0].turn,
            last_turn=window[-1].turn,
            scores={d: scores[d] for d in DIMENSIONS},
            contradictions=list(data.get("contradictions") or []),
            flags=[str(f) for f in (data.get("flags") or [])],
            window_summary=str(data.get("window_summary") or ""),
        )
        report.windows.append(grade)
        if grade.window_summary:
            story_so_far = grade.window_summary
        print(
            f"  T{grade.first_turn}-{grade.last_turn}: "
            + " ".join(f"{d}={grade.scores[d]}" for d in DIMENSIONS)
        )
    return report


def _resolve_turn_log(raw: str) -> Path:
    path = Path(raw)
    if path.exists():
        return path
    candidate = Path("data/turn_logs") / f"{raw.removesuffix('.jsonl')}.jsonl"
    if candidate.exists():
        return candidate
    raise SystemExit(f"Turn log not found: {raw}")


def main() -> int:
    # Windows consoles default to cp1252; judge flags quote em-dash prose.
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--turn-log", required=True)
    parser.add_argument("--judge-model", default="gemini-2.5-flash")
    parser.add_argument("--window-size", type=int, default=WINDOW_SIZE)
    args = parser.parse_args()

    turn_log = _resolve_turn_log(args.turn_log)
    print(f"Grading {turn_log} with {args.judge_model}...")
    report = asyncio.run(grade_run(
        turn_log, judge_model=args.judge_model, window_size=args.window_size,
    ))

    print("\n=== NARRATIVE QUALITY REPORT ===")
    print(f"windows: {len(report.windows)} | dimension means: {report.dimension_means()}")
    print(f"overall: {report.overall_average()} | worst repeat ratio: {report.worst_repeat_ratio()}")
    print(f"common openers: {report.opening_bigrams}")
    all_flags = [f for w in report.windows for f in w.flags]
    for flag in all_flags[:12]:
        print(f"  flag: {flag}")
    verdict = True
    for name, ok, detail in report.gates():
        print(f"  [{'PASS' if ok else 'FAIL'}] {name}: {detail}")
        verdict = verdict and ok

    out_dir = Path("data/narrative_quality")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{turn_log.stem}.json"
    out_path.write_text(json.dumps({
        "turn_log": report.turn_log,
        "judge_model": report.judge_model,
        "verdict": "PASS" if verdict else "FAIL",
        "overall_average": report.overall_average(),
        "dimension_means": report.dimension_means(),
        "gates": [
            {"name": n, "passed": ok, "detail": d} for n, ok, d in report.gates()
        ],
        "worst_repeat_ratio": report.worst_repeat_ratio(),
        "opening_bigrams": report.opening_bigrams,
        "windows": [
            {
                "turns": [w.first_turn, w.last_turn],
                "scores": w.scores,
                "contradictions": w.contradictions,
                "flags": w.flags,
                "summary": w.window_summary,
            }
            for w in report.windows
        ],
    }, indent=2), encoding="utf-8")
    print(f"\nVerdict: {'PASS' if verdict else 'FAIL'} — artifact: {out_path}")
    return 0 if verdict else 1


if __name__ == "__main__":
    sys.exit(main())
