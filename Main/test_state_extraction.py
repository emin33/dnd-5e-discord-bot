"""State-extraction eval: does the brain produce correct StateDelta from
narrator prose given prior world state?

State extraction is the trickiest brain task because the output is a deeply
structured delta with ~13 optional fields, and bad outputs corrupt long-term
world consistency. The biggest production risks are:

- Location names that violate the format ("behind the bar inside the tavern")
- NPC location strings that invent sub-locations not in current state
- Returning a string where a list is expected (post-processing has to coerce)
- Inventing new_npcs for entities already in the world state
- Extracting atmosphere as new_facts ("the wind is cold")

We don't try to score every field exactly — too brittle. We score the
high-leverage axes that have caused real bugs:
1. Location format compliance (no "behind the X inside the Y")
2. Field-type compliance (lists are arrays not strings)
3. NPC location anchored to current world state location
4. No new_npcs for entities already tracked
5. Empty extraction on pure atmospheric prose

Usage:
    python test_state_extraction.py
    python test_state_extraction.py --profile haiku_immersive_dreamshaper
    python test_state_extraction.py --names location_format
"""

from __future__ import annotations

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.chdir(Path(__file__).parent)
from dotenv import load_dotenv
load_dotenv()


class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    CYAN = "\033[96m"
    MAGENTA = "\033[95m"


def header(text: str):
    print(f"\n{C.BOLD}{C.CYAN}{'=' * 72}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {text}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'=' * 72}{C.RESET}")


# =============================================================================
# Scenarios
# =============================================================================

@dataclass
class Scenario:
    name: str
    narrative: str
    world_state_yaml: str = ""

    # ── Expected fields (each optional; only scored if set) ──
    expected_empty_delta: bool = False  # True = atmospheric prose, no extraction expected
    expected_location_change: Optional[str] = None  # exact match (case-insensitive)
    expected_new_npc_names: list[str] = field(default_factory=list)  # substring matches
    forbidden_new_npc_names: list[str] = field(default_factory=list)  # MUST NOT show as new_npcs
    expected_npc_location_for: Optional[tuple[str, str]] = None  # (npc_name_substring, location)
    notes: str = ""


SCENARIOS: list[Scenario] = [
    # ── Pure atmosphere — should produce no delta ──────────────────────────
    Scenario(
        name="atmosphere_only",
        narrative=(
            "The wind howls through the broken stones. Your torchlight "
            "flickers in the cold draft. Far below, water drips into a "
            "deep well. The smell of damp earth fills your nose."
        ),
        world_state_yaml=(
            "location: ruined keep\n"
            "time_of_day: evening\n"
            "phase: exploration\n"
        ),
        expected_empty_delta=True,
        notes="Pure atmosphere, no entities or state changes. Brain should "
              "return mostly-empty delta. Tests anti-extraction discipline.",
    ),

    # ── Location naming compliance (the high-frequency failure) ────────────
    Scenario(
        name="location_format_short_name_available",
        narrative=(
            "You step through the doorway and into the Rusty Compass tavern. "
            "Smoke and warmth wash over you. The common room is half-full "
            "tonight."
        ),
        world_state_yaml=(
            "location: Thornfield Road\n"
            "time_of_day: evening\n"
        ),
        expected_location_change="the rusty compass",  # case-insensitive
        notes="Brand-name location available — should be used directly.",
    ),
    Scenario(
        name="location_format_invent_short_name",
        narrative=(
            "The party emerges into a wide clearing dominated by an ancient "
            "stone shrine. Moss covers half the carvings. Crows perch on "
            "the toppled pillars."
        ),
        world_state_yaml=(
            "location: the wood path\n"
            "time_of_day: morning\n"
        ),
        # Brain should invent a 2-3 word name like "shrine clearing" or
        # "ancient shrine". We can't predict exactly which, so just check
        # format: short, lowercase or proper-noun, no commas / "behind" /
        # "inside" / sentences.
        notes="No name given — brain should invent a 2-3 word name like "
              "'shrine clearing' or 'ancient shrine', NOT 'a wide clearing "
              "with a stone shrine'.",
    ),

    # ── NPC location must come from current state, not invented ────────────
    Scenario(
        name="npc_location_anchored_to_current_state",
        narrative=(
            "Behind the bar, a dwarf named Borin polishes mugs. \"What'll "
            "it be?\" he asks. He nods toward the corner table."
        ),
        world_state_yaml=(
            "location: the rusty compass\n"
            "time_of_day: evening\n"
            "npcs_here: []\n"
        ),
        expected_new_npc_names=["borin"],
        # Brain MUST anchor borin's location to "the rusty compass" — NOT
        # invent "behind the bar inside the tavern" or similar.
        expected_npc_location_for=("borin", "the rusty compass"),
        notes="Borin is described as 'behind the bar' but his location field "
              "MUST anchor to the current world state location, not invent "
              "a sub-location.",
    ),

    # ── Don't re-create existing NPCs ──────────────────────────────────────
    Scenario(
        name="existing_npc_not_recreated",
        narrative=(
            "Aldric leans on the counter, tired-looking. \"You're back. "
            "Find what you were looking for?\""
        ),
        world_state_yaml=(
            "location: the rusty compass\n"
            "time_of_day: night\n"
            "npcs_here:\n"
            "- name: Aldric\n"
            "  disposition: friendly\n"
            "  desc: The innkeeper, broad-shouldered\n"
        ),
        forbidden_new_npc_names=["aldric"],
        notes="Aldric is already in the world state. Brain MUST NOT add him "
              "to new_npcs. Tests dedup discipline.",
    ),

    # ── New NPC introduction ───────────────────────────────────────────────
    Scenario(
        name="introduce_new_named_npc",
        narrative=(
            "A weather-beaten ranger named Sela approaches your table. She "
            "wears a green cloak and a bow across her back. \"You're the "
            "ones from the south road, aren't you?\""
        ),
        world_state_yaml=(
            "location: the rusty compass\n"
            "time_of_day: evening\n"
            "npcs_here:\n"
            "- name: Aldric\n"
            "  disposition: friendly\n"
        ),
        expected_new_npc_names=["sela"],
        expected_npc_location_for=("sela", "the rusty compass"),
        forbidden_new_npc_names=["aldric"],  # already tracked
        notes="New NPC introduced; Aldric stays known. Sela's location should "
              "anchor to current state.",
    ),

    # ── Multiple location changes within one narrative (only one wins) ─────
    Scenario(
        name="location_change_via_movement",
        narrative=(
            "You leave the warmth of the inn and step into the cold street. "
            "The town square sprawls before you, its central fountain dark "
            "and silent. A few late stragglers hurry home."
        ),
        world_state_yaml=(
            "location: the rusty compass\n"
            "time_of_day: night\n"
        ),
        expected_location_change="town square",  # case-insensitive substring
        notes="Movement to a new named location. Brain should set "
              "location_change to a short form of 'town square'.",
    ),

    # ── Time of day change ─────────────────────────────────────────────────
    Scenario(
        name="time_change_to_night",
        narrative=(
            "By the time you leave the magistrate's office, the sun has "
            "fully set. Lamps glow in the windows along the lane. Night "
            "has settled over the town."
        ),
        world_state_yaml=(
            "location: magistrate's office\n"
            "time_of_day: evening\n"
        ),
        notes="Time shifted from evening to night. Brain should emit "
              "time_change=night in the delta. (We don't strictly assert "
              "this — model freedom is fine — but a complete miss flags an "
              "issue.)",
    ),

    # ── Field type — list-vs-string fragility ──────────────────────────────
    Scenario(
        name="single_fact_should_be_list",
        narrative=(
            "Aldric leans close. \"The bridge over the Thorn River was "
            "destroyed last week. Only the ferry runs now.\""
        ),
        world_state_yaml=(
            "location: the rusty compass\n"
            "time_of_day: evening\n"
        ),
        notes="One fact established. The new_facts field should be a list "
              "containing this fact, NOT a bare string. (Validated as "
              "list-type in the post-process check.)",
    ),

    # ── Don't extract atmosphere as facts ──────────────────────────────────
    Scenario(
        name="weather_is_not_a_fact",
        narrative=(
            "Rain has been falling steadily for hours. You shiver under "
            "your cloak as you walk. The road has turned to mud."
        ),
        world_state_yaml=(
            "location: the wood path\n"
            "time_of_day: afternoon\n"
        ),
        notes="Atmospheric description (rain, mud). Should NOT generate "
              "new_facts — these are passing weather, not campaign facts. "
              "Tests anti-extraction discipline on atmosphere.",
    ),
]


# =============================================================================
# Runner
# =============================================================================

@dataclass
class Result:
    scenario: Scenario
    delta: dict  # parsed StateDelta.model_dump()
    elapsed: float
    error: Optional[str] = None


async def run_one(scenario: Scenario) -> Result:
    from dnd_bot.llm.extractors.state_extractor import StateExtractor

    extractor = StateExtractor()
    t0 = time.monotonic()
    try:
        delta = await extractor.extract(
            narrative_text=scenario.narrative,
            world_state_yaml=scenario.world_state_yaml,
        )
        return Result(
            scenario=scenario,
            delta=delta.model_dump() if hasattr(delta, "model_dump") else {},
            elapsed=time.monotonic() - t0,
        )
    except Exception as e:
        return Result(
            scenario=scenario,
            delta={},
            elapsed=time.monotonic() - t0,
            error=str(e),
        )


# =============================================================================
# Scoring
# =============================================================================

@dataclass
class CheckResult:
    name: str
    passed: bool
    detail: str = ""


# Reject patterns for location strings — these violate the format we taught.
_BAD_LOCATION_PATTERNS = [
    re.compile(r"\bbehind\s+(the\s+)?\w+\b", re.IGNORECASE),
    re.compile(r"\binside\s+(the\s+)?\w+\b", re.IGNORECASE),
    re.compile(r"\bnear\s+(the\s+)?\w+", re.IGNORECASE),
]


def _looks_like_bad_location(loc: Optional[str]) -> Optional[str]:
    """Return the offending pattern if loc violates our format, else None."""
    if not loc:
        return None
    # Words count
    if len(loc.split()) > 5:
        return f"too many words ({len(loc.split())})"
    # Sentence-y: ends with punctuation, has commas
    if "," in loc or loc.endswith("."):
        return "contains comma or period — not a name"
    for pattern in _BAD_LOCATION_PATTERNS:
        if pattern.search(loc):
            return f"contains forbidden pattern '{pattern.pattern}'"
    return None


def _check_empty_delta(result: Result) -> CheckResult:
    """For atmospheric scenarios — most fields should be empty/None."""
    delta = result.delta
    fields_set: list[str] = []
    if delta.get("location_change"):
        fields_set.append("location_change")
    if delta.get("new_npcs"):
        fields_set.append(f"new_npcs[{len(delta['new_npcs'])}]")
    if delta.get("new_facts"):
        fields_set.append(f"new_facts[{len(delta['new_facts'])}]")
    if delta.get("new_quests"):
        fields_set.append(f"new_quests[{len(delta['new_quests'])}]")

    # time_change and small-flag changes are OK; we're checking "did the
    # brain over-extract from atmosphere?"
    passed = not fields_set
    return CheckResult(
        name="empty_delta",
        passed=passed,
        detail=f"unexpected fields populated: {fields_set}" if not passed else "",
    )


def _check_location_change(result: Result) -> Optional[CheckResult]:
    expected = result.scenario.expected_location_change
    if expected is None:
        return None
    actual = (result.delta.get("location_change") or "").lower()
    expected_lower = expected.lower()
    passed = expected_lower in actual
    return CheckResult(
        name="location_change",
        passed=passed,
        detail=f"got='{actual}', expected substring='{expected_lower}'" if not passed else "",
    )


def _check_location_format(result: Result) -> list[CheckResult]:
    """Check ALL location strings emitted in the delta for format violations."""
    checks: list[CheckResult] = []
    delta = result.delta

    # location_change
    loc = delta.get("location_change")
    if loc:
        bad = _looks_like_bad_location(loc)
        checks.append(CheckResult(
            name="location_change.format",
            passed=bad is None,
            detail=f"'{loc}' — {bad}" if bad else "",
        ))

    # new_npcs locations
    for i, npc in enumerate(delta.get("new_npcs") or []):
        loc = npc.get("location") if isinstance(npc, dict) else None
        if loc:
            bad = _looks_like_bad_location(loc)
            checks.append(CheckResult(
                name=f"new_npcs[{i}].location.format",
                passed=bad is None,
                detail=f"'{loc}' — {bad}" if bad else "",
            ))

    # new_connections (each should be short)
    for i, conn in enumerate(delta.get("new_connections") or []):
        if isinstance(conn, str):
            bad = _looks_like_bad_location(conn)
            checks.append(CheckResult(
                name=f"new_connections[{i}].format",
                passed=bad is None,
                detail=f"'{conn}' — {bad}" if bad else "",
            ))

    return checks


def _check_field_types(result: Result) -> list[CheckResult]:
    """Verify list fields are actually lists (not strings)."""
    list_fields = ("new_connections", "new_facts", "new_events",
                   "new_npcs", "new_quests", "npc_updates",
                   "removed_npcs", "quest_updates")
    checks: list[CheckResult] = []
    for f in list_fields:
        val = result.delta.get(f)
        if val is None:
            continue
        if isinstance(val, list):
            checks.append(CheckResult(name=f"types.{f}", passed=True))
        else:
            checks.append(CheckResult(
                name=f"types.{f}",
                passed=False,
                detail=f"expected list, got {type(val).__name__}: {val!r:.80}",
            ))
    return checks


def _check_expected_npcs(result: Result) -> list[CheckResult]:
    checks: list[CheckResult] = []
    new_npcs = result.delta.get("new_npcs") or []

    for expected_name in result.scenario.expected_new_npc_names:
        needle = expected_name.lower()
        found = any(needle in (n.get("name") or "").lower() for n in new_npcs)
        checks.append(CheckResult(
            name=f"new_npcs[{expected_name}].present",
            passed=found,
            detail=f"not found in new_npcs (got: {[n.get('name') for n in new_npcs]})" if not found else "",
        ))
    return checks


def _check_forbidden_npcs(result: Result) -> list[CheckResult]:
    checks: list[CheckResult] = []
    new_npcs = result.delta.get("new_npcs") or []
    for forbidden in result.scenario.forbidden_new_npc_names:
        needle = forbidden.lower()
        offending = [n.get("name") for n in new_npcs if needle in (n.get("name") or "").lower()]
        checks.append(CheckResult(
            name=f"forbid_new_npcs[{forbidden}]",
            passed=not offending,
            detail=f"forbidden new_npc(s): {offending}" if offending else "",
        ))
    return checks


def _check_npc_location(result: Result) -> Optional[CheckResult]:
    expected = result.scenario.expected_npc_location_for
    if expected is None:
        return None
    npc_substring, location_substring = expected
    npc_substring = npc_substring.lower()
    location_substring = location_substring.lower()

    new_npcs = result.delta.get("new_npcs") or []
    matches = [n for n in new_npcs if npc_substring in (n.get("name") or "").lower()]

    if not matches:
        return CheckResult(
            name=f"npc_location[{npc_substring}]",
            passed=False,
            detail=f"NPC '{npc_substring}' not found in new_npcs",
        )

    actual_loc = (matches[0].get("location") or "").lower()
    passed = location_substring in actual_loc
    return CheckResult(
        name=f"npc_location[{npc_substring}]",
        passed=passed,
        detail=f"got location='{actual_loc}', expected substring='{location_substring}'" if not passed else "",
    )


def _all_checks(result: Result) -> list[CheckResult]:
    if result.error:
        return [CheckResult(name="error", passed=False, detail=result.error)]

    checks: list[CheckResult] = []

    if result.scenario.expected_empty_delta:
        checks.append(_check_empty_delta(result))

    loc_check = _check_location_change(result)
    if loc_check is not None:
        checks.append(loc_check)

    checks.extend(_check_location_format(result))
    checks.extend(_check_field_types(result))
    checks.extend(_check_expected_npcs(result))
    checks.extend(_check_forbidden_npcs(result))

    npc_loc = _check_npc_location(result)
    if npc_loc is not None:
        checks.append(npc_loc)

    return checks


# =============================================================================
# Reporting
# =============================================================================

def _print_per_scenario(results: list[Result], show_misses_only: bool):
    print(f"\n  {C.BOLD}{'Scenario':<42} {'Pass':>6} {'Time':>6}{C.RESET}")
    print(f"  {'─' * 65}")
    for r in results:
        checks = _all_checks(r)
        passed = sum(1 for c in checks if c.passed)
        total = len(checks)
        all_pass = passed == total

        if show_misses_only and all_pass:
            continue

        marker = "✓" if all_pass else "✗"
        color = C.GREEN if all_pass else C.RED
        ratio = f"{passed}/{total}"
        print(
            f"  {marker} {r.scenario.name:<40} "
            f"{color}{ratio:>6}{C.RESET} "
            f"{C.DIM}{r.elapsed:>5.1f}s{C.RESET}"
        )
        if not all_pass:
            for c in checks:
                if not c.passed:
                    print(f"    {C.YELLOW}✗ {c.name}{C.RESET}: {C.DIM}{c.detail}{C.RESET}")
            if r.scenario.notes:
                print(f"    {C.MAGENTA}note:{C.RESET} {C.DIM}{r.scenario.notes[:200]}{C.RESET}")


def _print_summary(results: list[Result], elapsed_total: float):
    total = len(results)
    fully_passing = sum(1 for r in results if all(c.passed for c in _all_checks(r)))
    full_pct = fully_passing / total if total else 0
    full_color = C.GREEN if full_pct >= 0.85 else C.YELLOW if full_pct >= 0.7 else C.RED

    # Per-axis aggregation
    location_format_attempted = 0
    location_format_clean = 0
    types_attempted = 0
    types_clean = 0
    npc_present_attempted = 0
    npc_present_clean = 0
    forbidden_attempted = 0
    forbidden_clean = 0
    location_change_attempted = 0
    location_change_correct = 0
    npc_loc_attempted = 0
    npc_loc_correct = 0
    empty_delta_attempted = 0
    empty_delta_clean = 0

    for r in results:
        for c in _all_checks(r):
            if ".format" in c.name and "location" in c.name or "connections" in c.name:
                location_format_attempted += 1
                if c.passed:
                    location_format_clean += 1
            elif c.name.startswith("types."):
                types_attempted += 1
                if c.passed:
                    types_clean += 1
            elif c.name.startswith("new_npcs[") and ".present" in c.name:
                npc_present_attempted += 1
                if c.passed:
                    npc_present_clean += 1
            elif c.name.startswith("forbid_new_npcs["):
                forbidden_attempted += 1
                if c.passed:
                    forbidden_clean += 1
            elif c.name == "location_change":
                location_change_attempted += 1
                if c.passed:
                    location_change_correct += 1
            elif c.name.startswith("npc_location["):
                npc_loc_attempted += 1
                if c.passed:
                    npc_loc_correct += 1
            elif c.name == "empty_delta":
                empty_delta_attempted += 1
                if c.passed:
                    empty_delta_clean += 1

    def _color_pct(pct: Optional[float]) -> str:
        if pct is None:
            return C.DIM
        return C.GREEN if pct >= 0.95 else C.YELLOW if pct >= 0.85 else C.RED

    print(f"\n  {C.BOLD}SUMMARY{C.RESET}")
    print(f"  Scenarios fully passing: {full_color}{fully_passing}/{total} = {full_pct:.0%}{C.RESET}")
    print(f"  Total time          : {elapsed_total:.1f}s")
    print(f"  Avg per scenario    : {elapsed_total / total if total else 0:.1f}s")

    print(f"\n  {C.BOLD}Per-axis accuracy:{C.RESET}")
    if location_format_attempted:
        pct = location_format_clean / location_format_attempted
        print(f"    location format compliance: {_color_pct(pct)}{location_format_clean}/{location_format_attempted} = {pct:.0%}{C.RESET}")
    if types_attempted:
        pct = types_clean / types_attempted
        print(f"    field-type compliance:      {_color_pct(pct)}{types_clean}/{types_attempted} = {pct:.0%}{C.RESET}")
    if location_change_attempted:
        pct = location_change_correct / location_change_attempted
        print(f"    location_change accuracy:   {_color_pct(pct)}{location_change_correct}/{location_change_attempted} = {pct:.0%}{C.RESET}")
    if npc_present_attempted:
        pct = npc_present_clean / npc_present_attempted
        print(f"    expected NPCs extracted:    {_color_pct(pct)}{npc_present_clean}/{npc_present_attempted} = {pct:.0%}{C.RESET}")
    if npc_loc_attempted:
        pct = npc_loc_correct / npc_loc_attempted
        print(f"    NPC location anchored:      {_color_pct(pct)}{npc_loc_correct}/{npc_loc_attempted} = {pct:.0%}{C.RESET}")
    if forbidden_attempted:
        pct = forbidden_clean / forbidden_attempted
        print(f"    no-dup of existing NPCs:    {_color_pct(pct)}{forbidden_clean}/{forbidden_attempted} = {pct:.0%}{C.RESET}")
    if empty_delta_attempted:
        pct = empty_delta_clean / empty_delta_attempted
        print(f"    no extraction on atmosphere:{_color_pct(pct)}{empty_delta_clean}/{empty_delta_attempted} = {pct:.0%}{C.RESET}")


def _save_results(results: list[Result], elapsed_total: float, profile_name: str):
    out_dir = Path("data/state_extraction_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": profile_name,
        "elapsed_total_s": elapsed_total,
        "scenarios": [
            {
                "name": r.scenario.name,
                "delta": r.delta,
                "elapsed_s": r.elapsed,
                "error": r.error,
                "checks": [
                    {"name": c.name, "passed": c.passed, "detail": c.detail}
                    for c in _all_checks(r)
                ],
            }
            for r in results
        ],
    }
    ts = int(time.time())
    (out_dir / f"state_extraction_{ts}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (out_dir / "state_extraction_latest.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  {C.DIM}Log saved: data/state_extraction_logs/state_extraction_{ts}.json{C.RESET}")


# =============================================================================
# Main
# =============================================================================

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="State-extraction eval")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--names", nargs="+", default=None)
    parser.add_argument("--show-misses-only", action="store_true")
    parser.add_argument("--list-scenarios", action="store_true")
    args = parser.parse_args()

    if args.list_scenarios:
        for i, s in enumerate(SCENARIOS):
            print(f"  [{i:>2}] {s.name}")
        return

    if args.profile:
        from dnd_bot.config import set_profile
        set_profile(args.profile)
        profile_name = args.profile
    else:
        from dnd_bot.config import get_profile
        profile_name = get_profile().name

    scenarios = list(SCENARIOS)
    if args.names:
        wanted = list(args.names)
        scenarios = [s for s in scenarios if any(w in s.name for w in wanted)]
    if not scenarios:
        print(f"  {C.RED}No scenarios match.{C.RESET}")
        return

    header(f"STATE EXTRACTION EVAL")
    from dnd_bot.config import get_profile
    profile = get_profile()
    print(f"  Profile : {profile_name}")
    print(f"  Brain   : {profile.brain.provider}/{profile.brain.model}")
    print(f"  Scenarios: {len(scenarios)}\n")

    sys.stdout.write("  Warmup  ... ")
    sys.stdout.flush()
    from dnd_bot.llm.client import get_llm_client
    brain = get_llm_client()
    t_warm = time.monotonic()
    try:
        await brain.chat(
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            temperature=0,
            max_tokens=8,
            think=False,
        )
        print(f"{C.GREEN}{time.monotonic() - t_warm:.1f}s{C.RESET}\n")
    except Exception as e:
        print(f"{C.YELLOW}skipped ({e}){C.RESET}\n")

    results: list[Result] = []
    t_start = time.monotonic()

    for idx, scen in enumerate(scenarios):
        sys.stdout.write(f"  [{idx+1:>2}/{len(scenarios)}] {scen.name:<40} ")
        sys.stdout.flush()

        result = await run_one(scen)
        results.append(result)

        if result.error:
            print(f"{C.RED}ERR{C.RESET} ({result.elapsed:.1f}s)")
            continue

        checks = _all_checks(result)
        passed = sum(1 for c in checks if c.passed)
        total = len(checks)
        if passed == total:
            print(f"{C.GREEN}{passed}/{total}{C.RESET} ({result.elapsed:.1f}s)")
        else:
            print(f"{C.YELLOW}{passed}/{total}{C.RESET} ({result.elapsed:.1f}s)")

    elapsed_total = time.monotonic() - t_start

    _print_per_scenario(results, args.show_misses_only)
    _print_summary(results, elapsed_total)

    is_filtered = bool(args.names)
    if not is_filtered:
        _save_results(results, elapsed_total, profile_name)
    else:
        print(f"\n  {C.DIM}Filtered subset run — log NOT saved.{C.RESET}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Interrupted.{C.RESET}")
