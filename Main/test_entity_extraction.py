"""Entity-extraction eval: does the brain correctly identify NPCs / creatures /
objects from narrator prose AND classify their disposition + combat_initiated?

Sends curated narrator prose snippets through ``EntityExtractor`` and grades
against expected entity sets and the high-stakes ``combat_initiated`` boolean.
The combat_initiated flag is what gates whether the orchestrator initiates a
combat encounter — false negatives let combat slip past, false positives kick
off combat unprompted. Both have real game impact.

Usage:
    python test_entity_extraction.py
    python test_entity_extraction.py --profile haiku_immersive_dreamshaper
    python test_entity_extraction.py --names tense_standoff
    python test_entity_extraction.py --show-misses-only
"""

from __future__ import annotations

import asyncio
import json
import os
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
#
# Each scenario provides a narrator prose snippet and the expected
# extraction result. We score:
#  - combat_initiated (high-stakes boolean — false neg means missed combat)
#  - entity count (within tolerance — exact name match is too brittle)
#  - disposition for any named entity (the squishy classification we improved)
#  - that no spell-effect or atmosphere is mistakenly extracted as entity

@dataclass
class ExpectedEntity:
    """An entity we expect the brain to extract."""
    name_substring: str  # case-insensitive substring match against extracted names
    entity_type: str  # npc | creature | object
    disposition: Optional[str] = None  # if set, must match exactly
    monster_index_should_be_set: Optional[bool] = None  # if combat-capable, expect non-null
    notes: str = ""


@dataclass
class Scenario:
    name: str
    narrative: str
    current_scene: str = ""
    existing_entities: list[str] = field(default_factory=list)

    expected_combat_initiated: bool = False
    expected_entities: list[ExpectedEntity] = field(default_factory=list)
    forbidden_substrings: list[str] = field(default_factory=list)  # things that should NOT show up
    notes: str = ""


SCENARIOS: list[Scenario] = [
    # ── ROUTINE INTRODUCTIONS ─────────────────────────────────────────────
    Scenario(
        name="innkeeper_greeting",
        narrative=(
            "Aldric, the broad-shouldered innkeeper, looks up from the tankard "
            "he's drying. \"Welcome back. Stew's hot, ale's cold.\" He nods at "
            "the empty stools by the fire."
        ),
        current_scene="The Rusty Compass tavern",
        expected_combat_initiated=False,
        expected_entities=[
            ExpectedEntity(
                name_substring="aldric",
                entity_type="npc",
                disposition="friendly",
                monster_index_should_be_set=True,  # commoner is the right map
            ),
        ],
        notes="Friendly NPC introduction, no tension.",
    ),
    Scenario(
        name="forge_tension",
        narrative=(
            "A burly dwarf scowls at you from behind the forge. Soot streaks "
            "his apron. He doesn't speak — just glares as he hammers a "
            "red-hot blade into shape."
        ),
        current_scene="The blacksmith's forge",
        expected_combat_initiated=False,
        expected_entities=[
            ExpectedEntity(
                name_substring="dwarf",
                entity_type="npc",
                disposition="unfriendly",
                monster_index_should_be_set=True,
            ),
        ],
        notes="Unfriendly NPC — glares but not attacking. Tests "
              "hostile-vs-unfriendly boundary.",
    ),

    # ── CRITICAL: HOSTILE-vs-UNFRIENDLY BOUNDARY ──────────────────────────
    Scenario(
        name="tense_standoff",
        narrative=(
            "The three guards block the alley. Their captain — a scarred "
            "woman in chainmail — draws her shortsword and levels it at "
            "Thorn. \"Drop your weapons. Now.\""
        ),
        current_scene="A dim alley in the merchant district",
        expected_combat_initiated=False,  # weapon drawn but not yet attacking
        expected_entities=[
            ExpectedEntity(name_substring="captain", entity_type="npc",
                           disposition="unfriendly",  # blocking + threatening = unfriendly per prompt
                           monster_index_should_be_set=True),
        ],
        forbidden_substrings=[],
        notes="Critical boundary: weapon drawn + threat issued, but no attack "
              "yet. Per the rewritten extraction prompt: drawing a weapon and "
              "issuing threats = unfriendly (not hostile). Hostile is reserved "
              "for actively attacking. Combat NOT initiated until first blow.",
    ),
    Scenario(
        name="actual_attack_starts_combat",
        narrative=(
            "The bandit lunges at you with a wicked dagger, slashing across "
            "your guard. Two more bandits charge from the shadows, blades "
            "drawn and screaming."
        ),
        current_scene="The forest path",
        expected_combat_initiated=True,
        expected_entities=[
            ExpectedEntity(name_substring="bandit", entity_type="creature",
                           disposition="hostile",
                           monster_index_should_be_set=True),
        ],
        notes="Active attack — lunge + slash. Combat MUST initiate.",
    ),
    Scenario(
        name="goblin_arrows_fly",
        narrative=(
            "An arrow whistles past your ear and thuds into the tree behind "
            "you. Six goblins emerge from the brush, bows drawn, more arrows "
            "already nocked."
        ),
        current_scene="A forest clearing",
        expected_combat_initiated=True,
        expected_entities=[
            ExpectedEntity(name_substring="goblin", entity_type="creature",
                           disposition="hostile",
                           monster_index_should_be_set=True),
        ],
        notes="Arrow already in flight — combat has begun even though you're "
              "not yet wounded.",
    ),
    Scenario(
        name="threat_no_attack",
        narrative=(
            "The cult fanatics line up in a half-circle, hands raised in "
            "complex sigils, eyes burning with fervor. Their leader speaks: "
            "\"Leave this place, intruders. Or join the host.\""
        ),
        current_scene="The ritual chamber",
        expected_combat_initiated=False,
        expected_entities=[
            ExpectedEntity(name_substring="fanatic", entity_type="creature",
                           disposition="unfriendly",
                           monster_index_should_be_set=True),
        ],
        notes="Hands raised in sigils LOOKS like spellcasting prep, but no "
              "spell has launched yet. Cult is unfriendly (threatening) not "
              "hostile (attacking). Combat NOT initiated.",
    ),

    # ── NPC INTRODUCTION ──────────────────────────────────────────────────
    Scenario(
        name="merchant_in_shop",
        narrative=(
            "Behind the counter, a wiry leatherworker named Pelo arranges "
            "boots in neat rows. He nods politely as you enter, his calloused "
            "hands pausing in their work."
        ),
        current_scene="The leatherworker's shop",
        expected_combat_initiated=False,
        expected_entities=[
            ExpectedEntity(name_substring="pelo", entity_type="npc",
                           disposition="friendly",
                           monster_index_should_be_set=True),
        ],
        notes="Calm friendly NPC, named.",
    ),

    # ── OBJECT EXTRACTION ─────────────────────────────────────────────────
    Scenario(
        name="locked_chest_in_room",
        narrative=(
            "At the back of the room sits a heavy iron-bound chest, its "
            "padlock corroded and ancient. Strange runes are scratched into "
            "the lid."
        ),
        current_scene="The abandoned study",
        expected_combat_initiated=False,
        expected_entities=[
            ExpectedEntity(name_substring="chest", entity_type="object",
                           monster_index_should_be_set=False),  # objects don't fight
        ],
        notes="Notable interactable object. Should be extracted as object, "
              "not creature.",
    ),

    # ── DO-NOT-EXTRACT (anti-pattern test) ────────────────────────────────
    Scenario(
        name="atmosphere_only_no_entities",
        narrative=(
            "The wind howls through the broken stones. Your torchlight "
            "flickers in the cold draft. Somewhere far off, water drips. "
            "The smell of damp earth fills your nose."
        ),
        current_scene="A ruined keep",
        expected_combat_initiated=False,
        expected_entities=[],
        forbidden_substrings=["wind", "torchlight", "draft", "smell"],
        notes="Pure atmospheric description. NOTHING should be extracted as "
              "an entity. Tests anti-pattern of treating atmosphere as entities.",
    ),
    Scenario(
        name="spell_effect_not_creature",
        narrative=(
            "Writhing shadow tendrils erupt from the ground, lashing at the "
            "party. Thorn dives aside as the dark energy snaps at his ankles."
        ),
        current_scene="Combat arena",
        expected_combat_initiated=True,  # spell IS attacking
        expected_entities=[],  # no creatures — just a spell effect
        forbidden_substrings=["tendril", "shadow", "energy"],
        notes="Spell effects are NOT entities. Combat IS initiated (the "
              "effect is attacking) but no creature should be added to the "
              "scene roster.",
    ),

    # ── MONSTER_INDEX MAPPING ─────────────────────────────────────────────
    Scenario(
        name="hooded_stranger_is_bandit",
        narrative=(
            "A hooded stranger steps from the alley, hand resting casually "
            "on a dagger at his belt. \"You've taken a wrong turn.\""
        ),
        current_scene="A dim alley",
        expected_combat_initiated=False,
        expected_entities=[
            ExpectedEntity(name_substring="stranger", entity_type="npc",
                           disposition="unfriendly",
                           monster_index_should_be_set=True),
        ],
        notes="Hooded stranger with weapon, threatening — should map to a "
              "combat-capable monster_index (bandit/thug/spy).",
    ),
    Scenario(
        name="wolf_pack",
        narrative=(
            "Three wolves circle the camp, their lean bodies low to the "
            "ground. One snarls, exposing yellow teeth, but they don't yet "
            "press in."
        ),
        current_scene="A campsite at dusk",
        expected_combat_initiated=False,  # circling but not attacking
        expected_entities=[
            ExpectedEntity(name_substring="wolves", entity_type="creature",
                           disposition="hostile",
                           monster_index_should_be_set=True),
        ],
        notes="Animals showing aggressive posture — disposition hostile but "
              "combat not yet initiated. Tests that animals get a "
              "monster_index (wolf).",
    ),

    # ── BIG NPC + small NPCs together ─────────────────────────────────────
    Scenario(
        name="court_audience",
        narrative=(
            "Lord Marwen sits on the dais, flanked by two silent guards in "
            "ceremonial armor. He gestures for you to approach. The guards "
            "do not react."
        ),
        current_scene="The throne room",
        expected_combat_initiated=False,
        expected_entities=[
            ExpectedEntity(name_substring="marwen", entity_type="npc",
                           disposition="neutral",
                           monster_index_should_be_set=True),
            # Guards may or may not get extracted; we don't assert on them
        ],
        notes="A named noble + background guards. Marwen must extract, "
              "guards optional.",
    ),

    # ── EDGE: AMBIGUOUS DISPOSITION ───────────────────────────────────────
    Scenario(
        name="suspicious_priest",
        narrative=(
            "Brother Cael smiles too warmly as he greets you, his eyes "
            "darting toward the chapel door. \"Of course, of course. The "
            "abbot will be most pleased to see you.\""
        ),
        current_scene="The monastery courtyard",
        expected_combat_initiated=False,
        expected_entities=[
            ExpectedEntity(name_substring="cael", entity_type="npc",
                           # Disposition genuinely ambiguous — could be
                           # friendly (he's smiling, polite) or unfriendly
                           # (suspicious cues). Don't assert disposition.
                           monster_index_should_be_set=True),
        ],
        notes="Acting friendly but body language is suspicious. Tests "
              "ambiguous disposition handling.",
    ),
]


# =============================================================================
# Runner
# =============================================================================

@dataclass
class Result:
    scenario: Scenario
    extracted: dict  # the parsed ExtractionResult.model_dump()
    elapsed: float
    error: Optional[str] = None


async def run_one(scenario: Scenario) -> Result:
    """Send the narrative through the EntityExtractor."""
    from dnd_bot.llm.extractors.entity_extractor import EntityExtractor

    extractor = EntityExtractor()
    t0 = time.monotonic()
    try:
        result = await extractor.extract(
            narrative_text=scenario.narrative,
            current_scene=scenario.current_scene,
            existing_entities=scenario.existing_entities,
        )
        return Result(
            scenario=scenario,
            extracted=result.model_dump() if hasattr(result, "model_dump") else {},
            elapsed=time.monotonic() - t0,
        )
    except Exception as e:
        return Result(
            scenario=scenario,
            extracted={},
            elapsed=time.monotonic() - t0,
            error=str(e),
        )


# =============================================================================
# Scoring
# =============================================================================

@dataclass
class CheckResult:
    name: str  # "combat_initiated" | "expected_entities" | "forbidden_substrings"
    passed: bool
    detail: str = ""


def _check_combat_initiated(result: Result) -> CheckResult:
    actual = result.extracted.get("combat_initiated", False)
    expected = result.scenario.expected_combat_initiated
    passed = actual == expected
    return CheckResult(
        name="combat_initiated",
        passed=passed,
        detail=f"got={actual}, expected={expected}" if not passed else "",
    )


def _find_entity_by_substring(extracted_entities: list[dict], substring: str) -> Optional[dict]:
    """Find an extracted entity whose name contains the substring (case insensitive)."""
    needle = substring.lower()
    for e in extracted_entities:
        name = (e.get("name") or "").lower()
        if needle in name:
            return e
    return None


def _check_expected_entities(result: Result) -> list[CheckResult]:
    """One CheckResult per expected entity."""
    checks: list[CheckResult] = []
    extracted = result.extracted.get("entities", [])
    for exp in result.scenario.expected_entities:
        found = _find_entity_by_substring(extracted, exp.name_substring)
        if found is None:
            checks.append(CheckResult(
                name=f"entity[{exp.name_substring}]",
                passed=False,
                detail=f"not found in extraction (extracted: "
                       f"{[e.get('name') for e in extracted]})",
            ))
            continue

        # Type check
        if found.get("entity_type") != exp.entity_type:
            checks.append(CheckResult(
                name=f"entity[{exp.name_substring}].type",
                passed=False,
                detail=f"got type={found.get('entity_type')}, "
                       f"expected {exp.entity_type}",
            ))
        else:
            checks.append(CheckResult(
                name=f"entity[{exp.name_substring}].type",
                passed=True,
            ))

        # Disposition check (only if expected)
        if exp.disposition is not None:
            actual_disp = found.get("disposition")
            checks.append(CheckResult(
                name=f"entity[{exp.name_substring}].disposition",
                passed=actual_disp == exp.disposition,
                detail=f"got disposition={actual_disp}, "
                       f"expected {exp.disposition}" if actual_disp != exp.disposition else "",
            ))

        # monster_index check (only if expected)
        if exp.monster_index_should_be_set is not None:
            actual_idx = found.get("monster_index")
            should_be_set = exp.monster_index_should_be_set
            is_set = actual_idx is not None and actual_idx != ""
            checks.append(CheckResult(
                name=f"entity[{exp.name_substring}].monster_index",
                passed=is_set == should_be_set,
                detail=f"got monster_index={actual_idx} "
                       f"(should_be_set={should_be_set})" if is_set != should_be_set else "",
            ))
    return checks


def _check_forbidden_substrings(result: Result) -> list[CheckResult]:
    """One CheckResult per forbidden substring — fails if it appears as an entity."""
    checks: list[CheckResult] = []
    extracted_names = [
        (e.get("name") or "").lower()
        for e in result.extracted.get("entities", [])
    ]
    for forbidden in result.scenario.forbidden_substrings:
        needle = forbidden.lower()
        offending = [n for n in extracted_names if needle in n]
        checks.append(CheckResult(
            name=f"forbid[{forbidden}]",
            passed=not offending,
            detail=f"forbidden substring '{forbidden}' found in: "
                   f"{offending}" if offending else "",
        ))
    return checks


def _all_checks(result: Result) -> list[CheckResult]:
    if result.error:
        return [CheckResult(name="error", passed=False, detail=result.error)]
    checks = [_check_combat_initiated(result)]
    checks.extend(_check_expected_entities(result))
    checks.extend(_check_forbidden_substrings(result))
    return checks


# =============================================================================
# Reporting
# =============================================================================

def _print_per_scenario(results: list[Result], show_misses_only: bool):
    print(f"\n  {C.BOLD}{'Scenario':<38} {'Pass':>6} {'Time':>6}{C.RESET}")
    print(f"  {'─' * 60}")
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
            f"  {marker} {r.scenario.name:<36} "
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
    total_scenarios = len(results)
    fully_passing = 0
    combat_correct = 0
    combat_attempted = 0
    disp_correct = 0
    disp_attempted = 0
    monster_idx_correct = 0
    monster_idx_attempted = 0
    forbidden_clean = 0
    forbidden_attempted = 0

    for r in results:
        checks = _all_checks(r)
        if all(c.passed for c in checks):
            fully_passing += 1
        for c in checks:
            if c.name == "combat_initiated":
                combat_attempted += 1
                if c.passed:
                    combat_correct += 1
            elif ".disposition" in c.name:
                disp_attempted += 1
                if c.passed:
                    disp_correct += 1
            elif ".monster_index" in c.name:
                monster_idx_attempted += 1
                if c.passed:
                    monster_idx_correct += 1
            elif c.name.startswith("forbid["):
                forbidden_attempted += 1
                if c.passed:
                    forbidden_clean += 1

    def _pct(c, n):
        if n == 0:
            return None
        return c / n

    def _color_pct(pct: Optional[float]) -> str:
        if pct is None:
            return C.DIM
        return C.GREEN if pct >= 0.95 else C.YELLOW if pct >= 0.85 else C.RED

    full_pct = fully_passing / total_scenarios if total_scenarios else 0
    full_color = _color_pct(full_pct)

    print(f"\n  {C.BOLD}SUMMARY{C.RESET}")
    print(f"  Total scenarios fully passing: {full_color}{fully_passing}/{total_scenarios} = {full_pct:.0%}{C.RESET}")
    print(f"  Total time          : {elapsed_total:.1f}s")
    print(f"  Avg per scenario    : {elapsed_total / total_scenarios if total_scenarios else 0:.1f}s")

    print(f"\n  {C.BOLD}Per-axis accuracy:{C.RESET}")
    if combat_attempted:
        pct = _pct(combat_correct, combat_attempted)
        col = _color_pct(pct)
        print(f"    combat_initiated  {col}{combat_correct}/{combat_attempted} = {pct:.0%}{C.RESET}  (the high-stakes boolean)")
    if disp_attempted:
        pct = _pct(disp_correct, disp_attempted)
        col = _color_pct(pct)
        print(f"    disposition       {col}{disp_correct}/{disp_attempted} = {pct:.0%}{C.RESET}")
    if monster_idx_attempted:
        pct = _pct(monster_idx_correct, monster_idx_attempted)
        col = _color_pct(pct)
        print(f"    monster_index     {col}{monster_idx_correct}/{monster_idx_attempted} = {pct:.0%}{C.RESET}  (set-vs-null check)")
    if forbidden_attempted:
        pct = _pct(forbidden_clean, forbidden_attempted)
        col = _color_pct(pct)
        print(f"    no spurious extr. {col}{forbidden_clean}/{forbidden_attempted} = {pct:.0%}{C.RESET}  (atmosphere not extracted as entities)")


def _save_results(results: list[Result], elapsed_total: float, profile_name: str):
    out_dir = Path("data/entity_extraction_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": profile_name,
        "elapsed_total_s": elapsed_total,
        "scenarios": [
            {
                "name": r.scenario.name,
                "extracted": r.extracted,
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
    (out_dir / f"entity_extraction_{ts}.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    (out_dir / "entity_extraction_latest.json").write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  {C.DIM}Log saved: data/entity_extraction_logs/entity_extraction_{ts}.json{C.RESET}")


# =============================================================================
# Main
# =============================================================================

async def main():
    import argparse
    parser = argparse.ArgumentParser(description="Entity-extraction eval")
    parser.add_argument("--profile", type=str, default=None)
    parser.add_argument("--names", nargs="+", default=None,
                        help="Run specific scenarios by name (substring match)")
    parser.add_argument("--show-misses-only", action="store_true",
                        help="Only display scenarios that failed at least one check")
    parser.add_argument("--list-scenarios", action="store_true")
    args = parser.parse_args()

    if args.list_scenarios:
        print(f"\n  Available scenarios:")
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

    # Filter scenarios
    scenarios = list(SCENARIOS)
    if args.names:
        wanted = list(args.names)
        scenarios = [s for s in scenarios if any(w in s.name for w in wanted)]

    if not scenarios:
        print(f"  {C.RED}No scenarios match the filters.{C.RESET}")
        return

    header(f"ENTITY EXTRACTION EVAL")
    from dnd_bot.config import get_profile
    profile = get_profile()
    print(f"  Profile : {profile_name}")
    print(f"  Brain   : {profile.brain.provider}/{profile.brain.model}")
    print(f"  Scenarios: {len(scenarios)}\n")

    # Warmup ping
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
        sys.stdout.write(f"  [{idx+1:>2}/{len(scenarios)}] {scen.name:<36} ")
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
