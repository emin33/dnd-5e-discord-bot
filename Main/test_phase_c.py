"""Phase C eval: does the brain correctly classify narrative_significance?

Sends curated D&D scenarios through the triage brain and grades the
``narrative_significance`` classification (routine / notable / climactic).
Reports per-class accuracy, confusion matrix, and per-scenario detail.

This is an evaluation harness, NOT a unit test. It hits a real brain via
your active profile (or --profile override) and costs whatever your brain
provider charges per triage call.

Usage:
    python test_phase_c.py                          # active profile's brain
    python test_phase_c.py --profile production     # use a specific profile's brain
    python test_phase_c.py --quick                  # only 6 scenarios (smoke test)
    python test_phase_c.py --scenarios 0 5 12       # specific scenarios

Targets the highest-performing local brain (Gemma 4 26b) by default if no
profile override is set.
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


# =============================================================================
# Colors
# =============================================================================

class C:
    RESET = "\033[0m"
    BOLD = "\033[1m"
    DIM = "\033[2m"
    RED = "\033[91m"
    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    BLUE = "\033[94m"
    MAGENTA = "\033[95m"
    CYAN = "\033[96m"


def header(text: str):
    print(f"\n{C.BOLD}{C.CYAN}{'=' * 72}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {text}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'=' * 72}{C.RESET}")


# =============================================================================
# Scenario definitions
# =============================================================================
#
# Each scenario provides a synthetic triage context (the kind of input the
# real brain sees in production) and an expected narrative_significance.
#
# Coverage targets:
# - 7 routine: ordinary turns that should NOT promote to premium
# - 7 notable: moments that deserve polish (first NPC, scene change, etc.)
# - 6 climactic: peak narrative beats (boss fights, deaths, betrayals)
# - 5 edge cases: boundary scenarios designed to test the brain's discernment
#
# Each scenario is self-contained — we don't share state across scenarios.

@dataclass
class Scenario:
    name: str
    expected: str  # brain classification: routine | notable | climactic
    action: str
    context: str
    notes: str = ""

    # Phase B context: would the orchestrator's deterministic veto fire?
    # The orchestrator computes definitely_standard from BrainContext.in_combat
    # plus triage.action_type in {"inventory", ...}. We only need in_combat here
    # because the brain provides action_type from its own output.
    in_combat: bool = False

    # Optional: expected triage classifications. None means we don't score
    # that field for this scenario. These let us regression-test the broader
    # triage prompt (action_type, needs_roll) alongside narrative_significance.
    expected_action_type: Optional[str] = None
    expected_needs_roll: Optional[bool] = None


SCENARIOS: list[Scenario] = [
    # ── ROUTINE ────────────────────────────────────────────────────────────
    Scenario(
        name="trash_mob_attack_round_3",
        expected="routine",
        action="I swing my longsword at the goblin in front of me",
        context=(
            "## Combat State\n"
            "Round 3 of combat against 3 goblins. Two are wounded.\n"
            "## Party\nThorn (fighter, HP 14/22), Lyra (cleric, HP 18/20)\n"
            "## Scene\nThe party is in a forest clearing, mid-fight."
        ),
        notes="Routine combat swing. Phase B veto fires (in_combat) so brain "
              "over-promotion would still route to standard.",
        in_combat=True,
        expected_action_type="attack",
        expected_needs_roll=False,  # combat resolves attack rolls separately
    ),
    Scenario(
        name="ask_innkeeper_price",
        expected="routine",
        action="I ask the innkeeper how much for a room and a meal",
        context=(
            "## Scene Entities\n- Aldric the innkeeper (neutral)\n"
            "## Scene\nThe Rusty Compass tavern, mid-afternoon. The party "
            "has been here before; Aldric is a known NPC."
        ),
        notes="Mundane shopping interaction with a known NPC.",
        expected_action_type="social",
        expected_needs_roll=False,
    ),
    Scenario(
        name="walk_to_market",
        expected="routine",
        action="I head to the market square to look for a leatherworker",
        context=(
            "## Scene\nVilltown's main road. Familiar territory — the party "
            "has shopped here twice before.\n## Party\nFull HP, no time pressure."
        ),
        notes="Simple movement between familiar locations. No stakes.",
        expected_action_type="movement",
        expected_needs_roll=False,
    ),
    Scenario(
        name="check_pack_for_rope",
        expected="routine",
        action="I dig through my pack to find my coil of rope",
        context=(
            "## Scene\nA quiet riverbank, mid-morning. The party is preparing "
            "to ford the stream.\n## Party\nAll healthy, time available."
        ),
        notes="Inventory action, no opposition, no narrative weight.",
        expected_action_type="inventory",
        expected_needs_roll=False,
    ),
    Scenario(
        name="small_talk_with_guard",
        expected="routine",
        action="I greet the gate guard and comment on the weather",
        context=(
            "## Scene Entities\n- Town gate guard (neutral, bored)\n"
            "## Scene\nApproaching the south gate at midday. The party is "
            "leaving town on a known errand."
        ),
        notes="Casual social with a low-stakes, undescribed background NPC.",
        expected_action_type="social",
        expected_needs_roll=False,
    ),
    Scenario(
        name="step_into_known_room",
        expected="routine",
        action="I walk back into the common room of the inn",
        context=(
            "## Scene\nThe Rusty Compass — the party rented rooms here last "
            "night. Aldric is behind the bar as usual."
        ),
        notes="Moving back into a previously-described familiar location.",
        expected_action_type="movement",
        expected_needs_roll=False,
    ),
    Scenario(
        name="pay_for_drinks",
        expected="routine",
        action="I drop two silver on the bar to pay for our ales",
        context=(
            "## Scene Entities\n- Aldric the innkeeper (neutral, friendly)\n"
            "## Scene\nThe Rusty Compass, finishing dinner."
        ),
        notes="Currency transaction with a known NPC. No stakes.",
        # action_type ambiguous: could be roleplay, social, or inventory
        expected_needs_roll=False,
    ),

    # ── NOTABLE ────────────────────────────────────────────────────────────
    Scenario(
        name="first_meeting_with_named_mentor",
        expected="notable",
        action="I introduce myself to Master Caldwell and explain why we're here",
        context=(
            "## Scene\nThe library of the Silver Spire. The party has been "
            "directed here by the council to seek out Master Caldwell — a "
            "renowned scholar who has never met them before.\n"
            "## Note\nThis is the first interaction with Caldwell, who will "
            "become a recurring quest-giver."
        ),
        notes="First meeting with a major NPC. Their first impression matters.",
    ),
    Scenario(
        name="enter_new_dungeon",
        expected="notable",
        action="I push open the heavy iron door and step inside",
        context=(
            "## Scene\nThe sealed entrance to the Hollowdeep crypt. The party "
            "has just unlocked the door for the first time. The crypt has "
            "never been described."
        ),
        notes="First entry into a new dungeon. Atmosphere matters.",
        expected_action_type="movement",
        expected_needs_roll=False,
    ),
    Scenario(
        name="find_important_letter",
        expected="notable",
        action="I open the desk drawer and pull out the sealed letter",
        context=(
            "## Scene\nThe study of the missing magistrate. The party has "
            "been investigating his disappearance.\n"
            "## Triage hint\nThis letter advances the central mystery."
        ),
        notes="Finding a major plot clue. Worth dwelling on.",
    ),
    Scenario(
        name="npc_breaks_down_crying",
        expected="notable",
        action="I gently ask Marta what really happened that night",
        context=(
            "## Scene Entities\n- Marta (anxious, knows more than she's said)\n"
            "## Scene\nMarta's kitchen. The party has earned a measure of "
            "trust through previous turns. She is on the edge of confessing."
        ),
        notes="An NPC reacting strongly. The reveal is meaningful but not the climax.",
    ),
    Scenario(
        name="strange_omen_appears",
        expected="notable",
        action="I look up at the sky to see what the villagers are pointing at",
        context=(
            "## Scene\nThe village square at dusk. Several villagers are "
            "gathered, pointing at the sky. The horizon is bruised purple "
            "with something unnatural moving across it."
        ),
        notes="An atmospheric scene change with mystery. Notable, not climactic.",
    ),
    Scenario(
        name="discover_body_of_quest_giver",
        expected="notable",
        action="I check Captain Halloran's pulse",
        context=(
            "## Scene\nThe captain's quarters. The party arrived to deliver "
            "their report and found the door ajar, the captain slumped at "
            "his desk.\n## Note\nHalloran was a recurring NPC but not in the "
            "inner circle."
        ),
        notes="Death of a secondary recurring NPC. Notable weight, but not the climax.",
    ),
    Scenario(
        name="sealed_vault_opens",
        expected="climactic",
        action="I work the final mechanism and step back as the vault grinds open",
        context=(
            "## Scene\nThe sealed vault of the Grimstone clan. The party has "
            "spent three sessions pursuing this. The mechanism just clicked. "
            "The vault has not yet been described."
        ),
        notes="A 3-session payoff. Setup-payoff signal at peak intensity = "
              "climactic. (Originally labeled notable; relabeled after both "
              "26b and e4b independently judged it climactic and the rules "
              "support that read.)",
    ),

    # ── CLIMACTIC ──────────────────────────────────────────────────────────
    Scenario(
        name="boss_fight_starts",
        expected="climactic",
        action="I draw my sword and charge Lady Veshra",
        context=(
            "## Scene Entities\n- Lady Veshra (BBEG, hostile)\n"
            "## Scene\nThe great hall of the Obsidian Throne. The party has "
            "pursued Veshra across three campaigns and just confronted her "
            "directly. Combat has not yet begun."
        ),
        notes="The campaign's BBEG fight begins. Peak narrative moment.",
        expected_action_type="attack",
        expected_needs_roll=False,
    ),
    Scenario(
        name="character_dies",
        expected="climactic",
        action="I take the killing blow for Pip and collapse on top of him",
        context=(
            "## Combat State\n"
            "Final round of the dragon fight. Thorn is at 2 HP. Pip the "
            "halfling scout is unconscious and dying. The dragon is about "
            "to bite Pip. Thorn declares this self-sacrificing action.\n"
            "## Note\nThorn is a player character."
        ),
        notes="Player character death moment. As climactic as it gets. "
              "Phase B in_combat fires but climactic overrides.",
        in_combat=True,
    ),
    Scenario(
        name="villain_reveals_identity",
        expected="climactic",
        action="I demand to know who is behind the mask",
        context=(
            "## Scene Entities\n- The masked figure (cornered)\n"
            "## Scene\nThe rooftop chase has ended. The party has the masked "
            "antagonist cornered. They have been pursuing this person for a "
            "full season.\n## Note\nThe reveal is about to happen."
        ),
        notes="Major identity reveal. Recontextualizes the campaign.",
    ),
    Scenario(
        name="betrayal_moment",
        expected="climactic",
        action="I confront Brother Kael — was it him all along?",
        context=(
            "## Scene Entities\n- Brother Kael (party ally for 2 seasons, "
            "now revealed traitor)\n"
            "## Scene\nThe ritual chamber. Kael's robes have just dropped to "
            "show the cult symbol. The party realizes their friend has been "
            "the inside man the entire time."
        ),
        notes="Long-running ally revealed as the traitor. Defining beat.",
    ),
    Scenario(
        name="final_choice",
        expected="climactic",
        action="I refuse the artifact's offer and shatter it against the altar",
        context=(
            "## Scene\nThe heart of the temple. The artifact has just offered "
            "the player ultimate power in exchange for their soul. This is "
            "the campaign's final moral choice. The decision will determine "
            "the world's fate."
        ),
        notes="Campaign-defining moral choice. Peak weight.",
    ),
    Scenario(
        name="dragon_emerges",
        expected="climactic",
        action="I look up as the mountain itself begins to move",
        context=(
            "## Scene\nThe summit plateau. What the party thought was a peak "
            "is unfolding into the wings of an ancient red dragon — the one "
            "the campaign has been about. This is its first full reveal."
        ),
        notes="The campaign's eponymous dragon reveals itself. Climactic.",
    ),

    # ── EDGE CASES — boundary tests ────────────────────────────────────────
    Scenario(
        name="edge_skill_check_with_real_stakes",
        expected="notable",
        action="I try to disarm the trap without setting off the gas",
        context=(
            "## Scene\nA narrow corridor in the crypt. A pressure plate is "
            "visible. Failure means the entire corridor fills with poisonous "
            "gas — a TPK risk."
        ),
        notes="Skill check with REAL stakes (TPK risk). Borderline notable, "
              "not yet climactic.",
        expected_action_type="skill_check",
        expected_needs_roll=True,
    ),
    Scenario(
        name="edge_routine_combat_in_climactic_arc",
        expected="routine",
        action="I attack the third minion with my crossbow",
        context=(
            "## Combat State\nMid-fight against Veshra and her 4 minions. "
            "Round 3. The party is grinding through her elite guards before "
            "reaching her.\n## Note\nThe overall encounter is climactic, but "
            "this single round is just attrition."
        ),
        notes="Routine swing inside a climactic encounter. Tests whether the "
              "brain over-promotes when context is dramatic. Phase B in_combat "
              "veto catches over-promotion to notable but not to climactic.",
        in_combat=True,
        expected_action_type="attack",
        expected_needs_roll=False,
    ),
    Scenario(
        name="edge_first_time_in_minor_shop",
        expected="routine",
        action="I browse the leatherworker's wares",
        context=(
            "## Scene\nThe leatherworker's stall, never visited before. The "
            "leatherworker is unnamed and unimportant — just a vendor."
        ),
        notes="Tests whether 'first time in a new place' gets over-promoted "
              "when the place is mundane.",
    ),
    Scenario(
        name="edge_dramatic_action_low_stakes",
        expected="routine",
        action="I dramatically draw my sword and point it at the merchant",
        context=(
            "## Scene Entities\n- Stallholder (frightened, nameless)\n"
            "## Scene\nA crowded market. The merchant is haggling poorly and "
            "the player decides to be theatrical for fun.\n## Note\nThis is "
            "a player joke, not a real threat."
        ),
        notes="Tests whether dramatic verbiage tricks the brain. Should stay "
              "routine because there's no real narrative weight.",
    ),
    Scenario(
        name="edge_minor_npc_dies_offscreen",
        expected="routine",
        action="I read the report about the courier's death",
        context=(
            "## Scene\nA briefing room. The party receives news that an "
            "unnamed courier was killed delivering a message. The courier "
            "was never an on-screen NPC."
        ),
        notes="Tests whether 'death' alone promotes. Should stay routine — "
              "the courier was never a character.",
    ),
]


# =============================================================================
# Brain runner
# =============================================================================

async def classify_scenario(brain_client, scenario: Scenario) -> tuple[Optional[dict], float, Optional[str]]:
    """Send the scenario through the triage brain. Returns (parsed_json, elapsed_s, error)."""
    from dnd_bot.llm.orchestrator import TRIAGE_SYSTEM_PROMPT

    # Construct the user message in the same shape the orchestrator does
    user_msg = f"## Player Action\n{scenario.action}\n\n{scenario.context}"

    messages = [
        {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
        {"role": "user", "content": user_msg},
    ]

    t0 = time.monotonic()
    try:
        response = await brain_client.chat(
            messages=messages,
            temperature=0,
            max_tokens=600,
            json_mode=True,
            think=False,
        )
    except Exception as e:
        return None, time.monotonic() - t0, str(e)

    elapsed = time.monotonic() - t0
    raw = response.content if hasattr(response, "content") else str(response)

    # Inline minimal parse — strip fences, extract JSON object, json.loads.
    # Mirrors the orchestrator's _parse_triage_json fallbacks but without
    # needing a full orchestrator instance.
    text = raw.strip()
    import re
    fence = re.match(r"^```(?:json)?\s*\n?(.*?)```\s*$", text, re.DOTALL)
    if fence:
        text = fence.group(1).strip()
    if "{" in text:
        start = text.find("{")
        end = text.rfind("}") + 1
        if end > start:
            text = text[start:end]

    try:
        parsed = json.loads(text)
        return parsed, elapsed, None
    except json.JSONDecodeError as e:
        return None, elapsed, f"json_parse: {e} | raw: {raw[:200]}"


# =============================================================================
# Reporting
# =============================================================================

@dataclass
class Result:
    scenario: Scenario
    predicted_significance: Optional[str]
    predicted_action_type: Optional[str]
    elapsed: float
    error: Optional[str]
    raw: Optional[dict] = None


# Phase B mundane action types that trigger the standard-veto.
# Mirrors orchestrator.Orchestrator._PHASE_B_MUNDANE_ACTION_TYPES.
_PHASE_B_MUNDANE_ACTION_TYPES = frozenset({"inventory"})


def _expected_tier(scenario: Scenario) -> str:
    """The tier the production router SHOULD pick for this scenario.

    Mirrors narrative_signals.select_narrator_tier semantics applied to the
    scenario's expected (correct) classification + Phase B context. This is
    what we ACTUALLY care about — final routing, not raw brain output.
    """
    # Phase B veto: in_combat forces standard unless climactic override
    if scenario.in_combat:
        return "premium" if scenario.expected == "climactic" else "standard"
    if scenario.expected == "climactic":
        return "premium"
    if scenario.expected == "notable":
        return "premium"
    return "standard"


def _predicted_tier(result: Result) -> str:
    """The tier the production router WOULD pick given the brain's actual output."""
    sig = result.predicted_significance
    action_type = result.predicted_action_type
    in_combat = result.scenario.in_combat

    # Phase B veto signal
    definitely_standard = bool(
        in_combat
        or (action_type and action_type in _PHASE_B_MUNDANE_ACTION_TYPES)
    )

    # Run the same priority chain as production
    if definitely_standard:
        if sig == "climactic":
            return "premium"
        return "standard"
    if sig in ("notable", "climactic"):
        return "premium"
    return "standard"


def _is_brain_correct(result: Result) -> bool:
    """Did the brain classify the significance correctly?"""
    return (
        result.predicted_significance is not None
        and result.predicted_significance == result.scenario.expected
    )


def _is_action_type_correct(result: Result) -> Optional[bool]:
    """Did the brain classify action_type correctly?

    Returns None if the scenario doesn't have an expected action_type
    (i.e., we deliberately don't score that field for this scenario).
    """
    if result.scenario.expected_action_type is None:
        return None
    return result.predicted_action_type == result.scenario.expected_action_type


def _is_needs_roll_correct(result: Result) -> Optional[bool]:
    """Did the brain emit the correct needs_roll boolean?

    Treats omitted needs_roll as False (matches the TriageSchema default).
    Returns None if the scenario doesn't have an expected_needs_roll.
    """
    if result.scenario.expected_needs_roll is None:
        return None
    if result.raw is None:
        return False
    predicted = result.raw.get("needs_roll", False)  # default to False if absent
    return predicted == result.scenario.expected_needs_roll


def _is_routing_correct(result: Result) -> bool:
    """Would production route this turn correctly?

    This is the metric that actually matters. Phase B can correct for some
    brain over-promotions (combat, inventory) so the brain doesn't have to
    be perfect to route correctly.
    """
    if result.predicted_significance is None:
        return False
    return _predicted_tier(result) == _expected_tier(result.scenario)


# Backwards-compatible alias used in older report functions
_is_correct = _is_brain_correct


def _print_per_scenario(results: list[Result], show_all_reasoning: bool = False):
    """Per-scenario breakdown — shows both brain classification and the
    final routing tier (what production actually serves).

    A "brain miss" that routes correctly because Phase B caught it is
    flagged with [P-B-veto]. Those are NOT real production errors.
    """
    print(
        f"\n  {C.BOLD}{'Scenario':<38} "
        f"{'Brain':>10} {'Tier':>10} {'Time':>6}{C.RESET}"
    )
    print(f"  {'─' * 70}")
    for r in results:
        brain_ok = _is_brain_correct(r)
        route_ok = _is_routing_correct(r)
        sig = r.predicted_significance or "ERROR"
        tier = _predicted_tier(r) if r.predicted_significance else "—"
        expected_tier = _expected_tier(r.scenario)

        sig_color = C.GREEN if brain_ok else (C.YELLOW if r.predicted_significance else C.RED)
        tier_color = C.GREEN if route_ok else C.RED

        # Marker: ✓ if routing correct (the thing that matters in production),
        # ⚠ if brain miss but routing correct (Phase B caught it),
        # ✗ if routing wrong.
        if route_ok and brain_ok:
            marker = "✓"
        elif route_ok and not brain_ok:
            marker = "⚠"
        else:
            marker = "✗"

        suffix = ""
        if route_ok and not brain_ok:
            suffix = f" {C.DIM}[Phase B veto saved it]{C.RESET}"

        print(
            f"  {marker} {r.scenario.name:<36} "
            f"{sig_color}{sig:>10}{C.RESET} "
            f"{tier_color}{tier:>10}{C.RESET} "
            f"{C.DIM}{r.elapsed:>5.1f}s{C.RESET}{suffix}"
        )
        if r.error:
            print(f"    {C.RED}{r.error[:200]}{C.RESET}")
            continue

        # Show reasoning when routing is wrong, or when brain missed
        # (even if Phase B saved it — useful for prompt-tuning visibility).
        # Or always if --show-reasoning was passed.
        show_reasoning = show_all_reasoning or not brain_ok
        if show_reasoning and r.raw:
            reasoning = r.raw.get("reasoning") or "(no reasoning emitted)"
            print(f"    {C.CYAN}brain says:{C.RESET} {C.DIM}{reasoning[:240]}{C.RESET}")
        if not route_ok:
            print(
                f"    {C.MAGENTA}routing miss:{C.RESET} "
                f"{C.DIM}got {tier}, expected {expected_tier}{C.RESET}"
            )
        if not brain_ok and r.scenario.notes:
            print(f"    {C.MAGENTA}label intent:{C.RESET} {C.DIM}{r.scenario.notes[:240]}{C.RESET}")


def _print_confusion_matrix(results: list[Result]):
    """Build a 3x3 confusion matrix on the three significance tiers."""
    tiers = ["routine", "notable", "climactic"]
    matrix = {(e, p): 0 for e in tiers for p in tiers}
    errors = 0
    for r in results:
        if r.predicted_significance is None or r.predicted_significance not in tiers:
            errors += 1
            continue
        matrix[(r.scenario.expected, r.predicted_significance)] += 1

    print(f"\n  {C.BOLD}CONFUSION MATRIX{C.RESET}  (rows=expected, cols=predicted)")
    print(f"\n  {'':>14} " + "".join(f"{t:>10}" for t in tiers))
    for e in tiers:
        row = "".join(
            (f"{C.GREEN if e == p else C.YELLOW}{matrix[(e, p)]:>10}{C.RESET}")
            for p in tiers
        )
        print(f"  {e:>14} " + row)
    if errors:
        print(f"\n  {C.RED}Parse / unknown errors: {errors}{C.RESET}")


def _print_summary(results: list[Result], elapsed_total: float):
    total = len(results)
    brain_correct = sum(1 for r in results if _is_brain_correct(r))
    routing_correct = sum(1 for r in results if _is_routing_correct(r))
    brain_acc = brain_correct / total if total else 0
    routing_acc = routing_correct / total if total else 0
    saved_by_veto = routing_correct - brain_correct

    print(f"\n  {C.BOLD}SUMMARY{C.RESET}")

    # Routing accuracy is the headline — it's what production actually serves.
    route_color = C.GREEN if routing_acc >= 0.95 else C.YELLOW if routing_acc >= 0.85 else C.RED
    print(f"  {C.BOLD}Routing accuracy{C.RESET} : {route_color}{routing_correct}/{total} = {routing_acc:.0%}{C.RESET}  (the metric that matters)")

    # Brain accuracy — sub-metric for prompt iteration / model selection
    brain_color = C.GREEN if brain_acc >= 0.95 else C.YELLOW if brain_acc >= 0.85 else C.RED
    print(f"  Brain classification: {brain_color}{brain_correct}/{total} = {brain_acc:.0%}{C.RESET}  (raw classifier accuracy)")

    if saved_by_veto > 0:
        print(f"  {C.DIM}Phase B veto rescued {saved_by_veto} brain miss(es){C.RESET}")

    print(f"  Total time          : {elapsed_total:.1f}s")
    avg = elapsed_total / total if total else 0
    print(f"  Avg per scenario    : {avg:.1f}s")

    # Per-tier brain accuracy (for prompt iteration debugging)
    by_tier: dict[str, list[Result]] = {}
    for r in results:
        by_tier.setdefault(r.scenario.expected, []).append(r)

    print(f"\n  Per-tier brain accuracy (raw classifier):")
    for tier in ("routine", "notable", "climactic"):
        bucket = by_tier.get(tier, [])
        if not bucket:
            continue
        c = sum(1 for r in bucket if _is_brain_correct(r))
        n = len(bucket)
        pct = c / n if n else 0
        col = C.GREEN if pct >= 0.85 else C.YELLOW if pct >= 0.7 else C.RED
        print(f"    {tier:<12} {col}{c}/{n} = {pct:.0%}{C.RESET}")

    # Other triage fields (only scenarios with expected values are counted).
    # These regression-test the broader TRIAGE_SYSTEM_PROMPT changes.
    action_type_results = [(_is_action_type_correct(r), r) for r in results]
    action_type_scored = [(ok, r) for ok, r in action_type_results if ok is not None]
    if action_type_scored:
        c = sum(1 for ok, _ in action_type_scored if ok)
        n = len(action_type_scored)
        pct = c / n if n else 0
        col = C.GREEN if pct >= 0.95 else C.YELLOW if pct >= 0.85 else C.RED
        print(f"\n  {C.BOLD}Other triage fields (regression coverage):{C.RESET}")
        print(f"    action_type:  {col}{c}/{n} = {pct:.0%}{C.RESET}  (scenarios with expected value)")

        # Show misses for diagnosis
        misses = [r for ok, r in action_type_scored if not ok]
        for r in misses:
            print(
                f"      {C.YELLOW}✗ {r.scenario.name}{C.RESET}: "
                f"got {C.DIM}{r.predicted_action_type}{C.RESET}, "
                f"expected {C.DIM}{r.scenario.expected_action_type}{C.RESET}"
            )

    needs_roll_results = [(_is_needs_roll_correct(r), r) for r in results]
    needs_roll_scored = [(ok, r) for ok, r in needs_roll_results if ok is not None]
    if needs_roll_scored:
        c = sum(1 for ok, _ in needs_roll_scored if ok)
        n = len(needs_roll_scored)
        pct = c / n if n else 0
        col = C.GREEN if pct >= 0.95 else C.YELLOW if pct >= 0.85 else C.RED
        print(f"    needs_roll:   {col}{c}/{n} = {pct:.0%}{C.RESET}")

        misses = [r for ok, r in needs_roll_scored if not ok]
        for r in misses:
            actual = r.raw.get("needs_roll") if r.raw else None
            print(
                f"      {C.YELLOW}✗ {r.scenario.name}{C.RESET}: "
                f"got {C.DIM}{actual}{C.RESET}, "
                f"expected {C.DIM}{r.scenario.expected_needs_roll}{C.RESET}"
            )


def _save_results(results: list[Result], elapsed_total: float, profile_name: str):
    out_dir = Path("data/phase_c_logs")
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "profile": profile_name,
        "elapsed_total_s": elapsed_total,
        "scenarios": [
            {
                "name": r.scenario.name,
                "expected": r.scenario.expected,
                "predicted": r.predicted_significance,
                "predicted_action_type": r.predicted_action_type,
                "elapsed_s": r.elapsed,
                "error": r.error,
                "raw": r.raw,
                "notes": r.scenario.notes,
            }
            for r in results
        ],
    }
    ts = int(time.time())
    path = out_dir / f"phase_c_{ts}.json"
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    latest = out_dir / "phase_c_latest.json"
    latest.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    print(f"\n  {C.DIM}Log saved: {path}{C.RESET}")


# =============================================================================
# Main
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(
        description="Phase C eval: brain classifies narrative_significance"
    )
    parser.add_argument("--profile", type=str, default=None,
                        help="Profile name to use (overrides ACTIVE_PROFILE)")
    parser.add_argument("--quick", action="store_true",
                        help="Run only 6 scenarios (smoke test)")
    parser.add_argument("--scenarios", nargs="+", type=int, default=None,
                        help="Run specific scenario indices (0-based)")
    parser.add_argument("--names", nargs="+", default=None,
                        help="Run specific scenarios by name (substring match). "
                             "Useful for re-testing one scenario after a prompt edit.")
    parser.add_argument("--tier", type=str, default=None,
                        choices=["routine", "notable", "climactic"],
                        help="Run only scenarios with this expected tier")
    parser.add_argument("--only-failing", action="store_true",
                        help="Re-run only the scenarios that failed in the last "
                             "saved log (data/phase_c_logs/phase_c_latest.json). "
                             "Use this to iterate on TRIAGE_SYSTEM_PROMPT — edit "
                             "the prompt, run --only-failing to see if the misses "
                             "got fixed without slowly re-running all 25.")
    parser.add_argument("--show-reasoning", action="store_true",
                        help="Show brain's reasoning field for ALL scenarios, "
                             "not just misses (default: only misses). Useful for "
                             "spotting borderline calls that happened to be correct.")
    parser.add_argument("--list-scenarios", action="store_true",
                        help="List all scenarios with index, name, expected tier — exit")
    args = parser.parse_args()

    if args.list_scenarios:
        print(f"\n  Available scenarios:")
        for i, s in enumerate(SCENARIOS):
            print(f"  [{i:>2}] {s.expected:>10}  {s.name}")
        return

    if args.profile:
        from dnd_bot.config import set_profile
        set_profile(args.profile)
        profile_name = args.profile
    else:
        from dnd_bot.config import get_profile
        profile_name = get_profile().name

    # Load the brain client via the standard path
    from dnd_bot.llm.client import get_llm_client
    brain = get_llm_client()

    header(f"PHASE C EVAL — narrative_significance classification")
    from dnd_bot.config import get_profile
    profile = get_profile()
    print(f"  Profile : {profile_name}")
    print(f"  Brain   : {profile.brain.provider}/{profile.brain.model}")

    # Warmup ping — large local models cold-start on first call (Gemma 26b can
    # take 60-120s to load into VRAM). Send a throwaway request first so the
    # actual scenarios all hit a warm model.
    sys.stdout.write("  Warmup  ... ")
    sys.stdout.flush()
    t_warm = time.monotonic()
    try:
        await brain.chat(
            messages=[{"role": "user", "content": "Reply with exactly: ok"}],
            temperature=0,
            max_tokens=8,
            think=False,
        )
        print(f"{C.GREEN}{time.monotonic() - t_warm:.1f}s{C.RESET}")
    except Exception as e:
        print(f"{C.YELLOW}skipped ({e}){C.RESET}")

    # Pick scenarios to run. Filters compose: --tier narrows the set, then
    # --names / --scenarios / --only-failing further narrows it.
    scenarios: list[Scenario] = list(SCENARIOS)

    # --only-failing: reload last log, keep only the scenarios that missed
    if args.only_failing:
        latest_path = Path("data/phase_c_logs/phase_c_latest.json")
        if not latest_path.exists():
            print(f"  {C.RED}No previous log found at {latest_path}.{C.RESET}")
            print(f"  {C.RED}Run a full eval first, then --only-failing to iterate.{C.RESET}")
            return
        try:
            prev = json.loads(latest_path.read_text(encoding="utf-8"))
        except Exception as e:
            print(f"  {C.RED}Failed to load previous log: {e}{C.RESET}")
            return

        prev_misses = {
            s["name"]
            for s in prev.get("scenarios", [])
            if s.get("predicted") != s.get("expected")
            or s.get("error")
        }
        if not prev_misses:
            print(f"  {C.GREEN}Previous run had 0 misses. Nothing to re-test.{C.RESET}")
            return
        scenarios = [s for s in scenarios if s.name in prev_misses]
        print(f"  {C.YELLOW}Re-running {len(scenarios)} previously failing scenarios:{C.RESET}")
        for s in scenarios:
            print(f"    - {s.name} (expected {s.expected})")

    # --tier: keep only scenarios with this expected significance
    if args.tier:
        scenarios = [s for s in scenarios if s.expected == args.tier]

    # --names: substring match on scenario name
    if args.names:
        wanted = list(args.names)
        scenarios = [s for s in scenarios if any(w in s.name for w in wanted)]

    # --scenarios: explicit index list
    if args.scenarios:
        # Apply against the ORIGINAL SCENARIOS list, then intersect with
        # whatever filters above already produced
        by_idx = [SCENARIOS[i] for i in args.scenarios if 0 <= i < len(SCENARIOS)]
        scenarios = [s for s in scenarios if s in by_idx]

    # --quick: 2 of each tier (overridden by anything more specific)
    if args.quick and not (args.scenarios or args.names or args.only_failing or args.tier):
        scenarios = [SCENARIOS[i] for i in (0, 4, 7, 11, 14, 18)]

    if not scenarios:
        print(f"  {C.RED}No scenarios match the filters.{C.RESET}")
        return

    print(f"  Scenarios: {len(scenarios)}\n")

    results: list[Result] = []
    t_start = time.monotonic()

    for idx, scen in enumerate(scenarios):
        sys.stdout.write(f"  [{idx+1:>2}/{len(scenarios)}] {scen.name:<40} ")
        sys.stdout.flush()

        parsed, elapsed, err = await classify_scenario(brain, scen)

        sig = None
        action_type = None
        if parsed:
            sig_raw = parsed.get("narrative_significance")
            sig = sig_raw.strip().lower() if isinstance(sig_raw, str) else None
            action_type = parsed.get("action_type")

        result = Result(
            scenario=scen,
            predicted_significance=sig,
            predicted_action_type=action_type,
            elapsed=elapsed,
            error=err,
            raw=parsed,
        )
        results.append(result)

        ok = _is_correct(result)
        if err:
            print(f"{C.RED}ERR{C.RESET} ({elapsed:.1f}s)")
        elif sig is None:
            print(f"{C.RED}MISSING{C.RESET} ({elapsed:.1f}s)")
        elif ok:
            print(f"{C.GREEN}{sig}{C.RESET} ({elapsed:.1f}s)")
        else:
            print(f"{C.YELLOW}{sig}{C.RESET} (expected {scen.expected}, {elapsed:.1f}s)")

    elapsed_total = time.monotonic() - t_start

    _print_per_scenario(results, show_all_reasoning=args.show_reasoning)
    _print_confusion_matrix(results)
    _print_summary(results, elapsed_total)

    # Skip saving when running a filtered subset — we don't want to overwrite
    # a full-suite log with partial results. The full log stays canonical
    # for --only-failing iteration.
    is_filtered_subset = bool(
        args.only_failing or args.tier or args.names or args.scenarios or args.quick
    )
    if not is_filtered_subset:
        _save_results(results, elapsed_total, profile_name)
    else:
        print(f"\n  {C.DIM}Filtered subset run — log NOT overwritten.{C.RESET}")
        print(f"  {C.DIM}(Last full log kept as the regression baseline for --only-failing.){C.RESET}")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Interrupted.{C.RESET}")
