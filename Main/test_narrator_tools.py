"""Narrator tool-calling benchmark.

Sends curated D&D narration scenarios through the active profile's narrator
client and grades whether it emits the right tool calls. Measures per-tool
recall (did expected tools fire?), precision (did forbidden tools fire?),
and basic argument correctness.

Unlike test_phase_c.py (which grades a triage classification), this targets
the narrator's tool-emission path — the same code that runs in production
when a profile sets `tools: full` (or `core_plus`/`core`).

Usage:
    python test_narrator_tools.py                     # active profile narrator
    python test_narrator_tools.py --profile qwen36_local
    python test_narrator_tools.py --quick             # 6-scenario smoke test
    python test_narrator_tools.py --tier full         # override tool tier
    python test_narrator_tools.py --scenarios 0 5 12  # specific scenarios
"""

from __future__ import annotations

import argparse
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


# ── Colors ─────────────────────────────────────────────────────────────

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


# ── Scenario definitions ───────────────────────────────────────────────

@dataclass
class Scenario:
    name: str
    description: str       # what this scenario is testing
    roster: str            # synthetic roster YAML the narrator sees
    scene: str             # current location/scene description
    recent: str            # recent narrative context (last 1-2 turns)
    player_action: str     # what the player just did
    expected_tools: dict   # tool_name → expected args (subset; None for "must fire, args don't matter")
    forbidden_tools: list  # tool names that MUST NOT fire
    expected_min_count: int = 0   # minimum number of tool calls (e.g., 0 for "no tools needed")


SCENARIOS: list[Scenario] = [
    # ── Single-tool: ref_entity ───────────────────────────────────────
    Scenario(
        name="ref_entity_basic",
        description="Mentioning a roster NPC by name should emit ref_entity.",
        roster="- [id: marta] Marta the herbalist (friendly, important)",
        scene="The herbalist's hut, late afternoon.",
        recent="Marta showed you her drying racks last turn.",
        player_action="I ask Marta about the strange mushrooms growing near the river.",
        expected_tools={"ref_entity": {"entity_id": "marta"}},
        forbidden_tools=["add_npc", "update_entity", "change_location"],
        expected_min_count=1,
    ),

    # ── Single-tool: add_npc ──────────────────────────────────────────
    Scenario(
        name="add_npc_first_time",
        description="A new NPC entering the scene should fire add_npc, not ref_entity.",
        roster="(no entities)",
        scene="The Rusty Compass tavern, dimly lit, a dozen patrons.",
        recent="You just sat at the bar.",
        player_action="I look for someone who might know about the missing shipment.",
        expected_tools={"add_npc": None},
        forbidden_tools=["ref_entity", "update_entity"],
        expected_min_count=1,
    ),

    # ── Negative: no tools needed ─────────────────────────────────────
    Scenario(
        name="atmospheric_no_tools",
        description="Pure atmospheric narration with no roster entity, no new NPC, no state change should fire no tools.",
        roster="(no entities)",
        scene="A windswept moor, dusk approaching.",
        recent="You crested the hill an hour ago.",
        player_action="I keep walking, watching the horizon.",
        expected_tools={},
        forbidden_tools=["ref_entity", "add_npc", "update_entity", "change_location",
                         "spawn_object", "start_combat", "update_player"],
        expected_min_count=0,
    ),

    # ── change_location: party moves ──────────────────────────────────
    Scenario(
        name="change_location_enter_building",
        description="Stepping into a new named location should fire change_location.",
        roster="(no entities)",
        scene="The market square, midday.",
        recent="You spotted the tavern across the plaza.",
        player_action="I push through the door of the Rusty Compass.",
        expected_tools={"change_location": {"location_name_contains": "compass"}},
        forbidden_tools=["spawn_object", "add_npc"],
        expected_min_count=1,
    ),

    # ── change_location: NEGATIVE (in-place action) ───────────────────
    Scenario(
        name="change_location_negative_in_place",
        description="An in-place action without movement should NOT fire change_location.",
        roster="- [id: korin] Korin Ironeye, blacksmith (neutral)",
        scene="Korin's forge, the air thick with smoke.",
        recent="Korin is hammering a glowing horseshoe.",
        player_action="I watch him work for a moment, admiring the craft.",
        expected_tools={"ref_entity": {"entity_id": "korin"}},
        forbidden_tools=["change_location", "add_npc"],
        expected_min_count=1,
    ),

    # ── update_entity: disposition shift ──────────────────────────────
    Scenario(
        name="update_entity_disposition_shift",
        description="Roster NPC turning hostile mid-scene should fire update_entity AND ref_entity.",
        roster="- [id: kael] Brother Kael, friendly priest (friendly)",
        scene="A side chapel, candles guttering.",
        recent="Kael has been guiding you through the catacombs.",
        player_action="I show Kael the cult symbol I found on the altar.",
        # The narrator's likely response: Kael's mask drops, he becomes hostile.
        expected_tools={
            "ref_entity": {"entity_id": "kael"},
            "update_entity": {"entity_id": "kael", "disposition": "hostile"},
        },
        forbidden_tools=["add_npc", "start_combat"],  # not yet attacking
        expected_min_count=2,
    ),

    # ── update_entity: NEGATIVE (just referencing) ────────────────────
    Scenario(
        name="update_entity_negative_just_reference",
        description="Mentioning roster entity without state change should ONLY fire ref_entity.",
        roster="- [id: marta] Marta the herbalist (friendly)",
        scene="Marta's hut.",
        recent="",
        player_action="I greet Marta and ask how her day has been.",
        expected_tools={"ref_entity": {"entity_id": "marta"}},
        forbidden_tools=["update_entity", "add_npc"],
        expected_min_count=1,
    ),

    # ── start_combat: combat initiation this turn ────────────────────
    Scenario(
        name="start_combat_active_attack",
        description="Combat begins THIS turn (player initiates) — narrator should fire start_combat for the moment-zero declaration.",
        roster="- [id: kael] Brother Kael (hostile)",
        scene="A side chapel, candles guttering.",
        recent=(
            "Kael's mask drops to reveal the cult symbol on his chest. "
            "He pulls a curved dagger from beneath his robes, his face "
            "twisted with hate."
        ),
        player_action="I lunge at Kael with my sword, swinging for his throat.",
        expected_tools={"start_combat": None, "ref_entity": {"entity_id": "kael"}},
        forbidden_tools=[],
        expected_min_count=1,
    ),

    # ── start_combat: NEGATIVE (tension only) ─────────────────────────
    Scenario(
        name="start_combat_negative_tension",
        description="Drawing weapons / hostile posture without an actual attack must NOT fire start_combat.",
        roster="- [id: bandits] Three bandits (hostile)",
        scene="A wooded path. Three bandits step from the trees, blades drawn.",
        recent="They demanded your purse.",
        player_action="I tell them I'm not handing anything over and ready my own blade.",
        expected_tools={"ref_entity": {"entity_id": "bandits"}},
        forbidden_tools=["start_combat"],
        expected_min_count=1,
    ),

    # ── update_player: item pickup ────────────────────────────────────
    Scenario(
        name="update_player_pickup",
        description="Player taking a scene item should fire update_player with item_grant.",
        roster="(no entities)",
        scene="An abandoned shrine. A jade dagger rests on the altar [scene:jade_dagger_1].",
        recent="You spotted the dagger glinting.",
        player_action="I take the jade dagger.",
        expected_tools={"update_player": None},  # any update_player call counts
        forbidden_tools=["spawn_object"],
        expected_min_count=1,
    ),

    # ── update_player: NEGATIVE (offer not yet accepted) ──────────────
    Scenario(
        name="update_player_negative_offer_pending",
        description="NPC offering an item the player hasn't accepted yet must NOT fire update_player.",
        roster="- [id: captain] Captain Halloran (friendly)",
        scene="The deck of the captain's ship.",
        recent="The captain finished her story.",
        player_action="I ask if she has any maps that might help me.",
        # Captain likely pulls out / offers a chart but the player hasn't accepted yet.
        expected_tools={"ref_entity": {"entity_id": "captain"}},
        forbidden_tools=["update_player"],
        expected_min_count=1,
    ),

    # ── update_player: currency exchange ──────────────────────────────
    Scenario(
        name="update_player_currency",
        description="NPC paying the player should fire update_player with currency_delta.",
        roster="- [id: mayor] Mayor Brennan (friendly, important)",
        scene="The mayor's office.",
        recent="You returned the stolen seal.",
        player_action="I hand the seal to the mayor.",
        # Likely: mayor presses gold pieces into the player's palm in gratitude.
        expected_tools={"update_player": None},
        forbidden_tools=["spawn_object"],
        expected_min_count=1,
    ),

    # ── update_entity: NPC inventory (the innkeeper-with-relic case) ──
    Scenario(
        name="update_entity_npc_holds_item",
        description="Player handing item to NPC for safekeeping should fire update_entity (NPC gains the item) AND update_player (player loses it).",
        roster="- [id: innkeeper] Bron the innkeeper (friendly, important)",
        scene="The taproom of the Hearthwood Inn.",
        recent="You spent the night with Bron's hospitality. He treated you to a free room.",
        player_action="I take the ancient relic from my pack and ask Bron to keep it safe for me while I'm away.",
        # Two-tool fire: update_entity adds relic to innkeeper, update_player removes from player
        expected_tools={
            "update_entity": {"entity_id": "innkeeper"},
            "ref_entity": {"entity_id": "innkeeper"},
        },
        forbidden_tools=[],
        expected_min_count=2,
    ),

    # ── spawn_object: discoverable item ───────────────────────────────
    Scenario(
        name="spawn_object_treasure",
        description="Discovering a specific interactable object should fire spawn_object.",
        roster="(no entities)",
        scene="A dusty crypt, the stone slabs cracked.",
        recent="You shoved aside the heavy lid.",
        player_action="I peer inside the sarcophagus.",
        # Likely response: a goblet/blade/skeleton with a ring.
        expected_tools={"spawn_object": None},
        forbidden_tools=["update_player"],  # they haven't picked it up yet
        expected_min_count=1,
    ),

    # ── update_player: environmental damage (formerly apply_damage) ───
    Scenario(
        name="update_player_trap_damage",
        description="Trap firing on the player should fire update_player with hp_delta. Triggering event in recent context — narrator must narrate the consequence.",
        roster="(no entities)",
        scene="A narrow corridor, dust thick on the floor.",
        recent=(
            "Halfway down the corridor your boot lands on a stone tile "
            "that sinks with a heavy click. Hidden mechanisms grind to "
            "life in the walls — a pair of poisoned darts whistle out "
            "of holes you didn't see and slam into your shoulder."
        ),
        player_action="I yank the darts free and keep moving.",
        expected_tools={"update_player": None},  # hp_delta < 0 + damage_type
        forbidden_tools=["start_combat"],
        expected_min_count=1,
    ),

    # ── request_roll: uncertain action ────────────────────────────────
    Scenario(
        name="request_roll_perception",
        description="Player asking to spot something hidden should fire request_roll.",
        roster="- [id: merchant] A traveling merchant (neutral)",
        scene="The merchant's wagon.",
        recent="The merchant has been chatting amiably.",
        player_action="I try to read the merchant — is he hiding something?",
        expected_tools={"request_roll": {"target": "player"}},
        forbidden_tools=["update_entity"],  # don't preempt the result
        expected_min_count=1,
    ),

    # ── Combination: add_npc with dialogue ────────────────────────────
    Scenario(
        name="add_npc_with_dialogue",
        description="New NPC speaking should fire add_npc with dialogue_indices set.",
        roster="(no entities)",
        scene="The market square.",
        recent="",
        player_action="I ask if anyone has seen the wagon driver.",
        # Likely: a stranger answers, fitting for add_npc with dialogue_indices=[1].
        expected_tools={"add_npc": None},
        forbidden_tools=["ref_entity"],
        expected_min_count=1,
    ),
]


# ── Grading ────────────────────────────────────────────────────────────

@dataclass
class ScenarioResult:
    scenario: Scenario
    raw_calls: list           # tool calls returned by the model
    prose: str
    expected_hits: dict       # tool_name → (fired, args_match)
    forbidden_violations: list  # forbidden tools that fired
    elapsed_s: float
    error: Optional[str] = None

    @property
    def all_expected_fired(self) -> bool:
        return all(h[0] for h in self.expected_hits.values())

    @property
    def args_correct(self) -> bool:
        return all(h[1] for h in self.expected_hits.values() if h[0])

    @property
    def no_forbidden_fired(self) -> bool:
        return len(self.forbidden_violations) == 0

    @property
    def passed(self) -> bool:
        # Pass if (1) all expected tools fired, (2) args broadly match, (3) no forbidden fired
        return (
            self.all_expected_fired
            and self.args_correct
            and self.no_forbidden_fired
            and not self.error
        )


def _arg_match(expected: dict | None, actual: dict) -> bool:
    """Loose match: every expected key (or _contains variant) must match the actual call."""
    if expected is None:
        return True
    for key, val in expected.items():
        if key.endswith("_contains"):
            real_key = key[: -len("_contains")]
            actual_val = actual.get(real_key, "")
            if not isinstance(actual_val, str) or val.lower() not in actual_val.lower():
                return False
        else:
            actual_val = actual.get(key)
            if isinstance(val, str) and isinstance(actual_val, str):
                if val.lower() != actual_val.lower():
                    return False
            elif actual_val != val:
                return False
    return True


def grade(scenario: Scenario, raw_calls: list, prose: str, elapsed_s: float, error: Optional[str] = None) -> ScenarioResult:
    expected_hits: dict[str, tuple[bool, bool]] = {}
    for tool_name, expected_args in scenario.expected_tools.items():
        # Did this tool fire at all?
        matching = [c for c in raw_calls if c.get("name") == tool_name]
        if not matching:
            expected_hits[tool_name] = (False, False)
            continue
        # Take the best match for args
        any_args_match = any(_arg_match(expected_args, c.get("arguments", {})) for c in matching)
        expected_hits[tool_name] = (True, any_args_match)

    forbidden_violations = [c["name"] for c in raw_calls if c.get("name") in scenario.forbidden_tools]

    return ScenarioResult(
        scenario=scenario,
        raw_calls=raw_calls,
        prose=prose,
        expected_hits=expected_hits,
        forbidden_violations=forbidden_violations,
        elapsed_s=elapsed_s,
        error=error,
    )


# ── Runner ─────────────────────────────────────────────────────────────

NARRATOR_SYSTEM = """\
# Role
You are the D&D 5e narrator. Every turn you produce TWO outputs together:
- Prose: 2-3 paragraphs of vivid second-person narration.
- Tool calls: structured signals for every state change your prose describes.

# Core rule
If your prose names a roster entity, introduces a new NPC, moves the party, \
spawns a tangible object, transfers an item or coin, shifts an entity's \
state, calls for a roll, deals environmental damage, or starts combat — you \
MUST emit the matching tool call in the same turn. Silent state changes \
break the game. Equally: do NOT fire tools the prose did not earn — \
over-firing is just as broken.

# Decision checklist (run before sending)
1. Did I name a roster entity in prose? → ref_entity for each (always)
2. Did I introduce a brand-new NPC? → add_npc (NOT ref_entity — they aren't on the roster yet)
3. Did the party MOVE to a new named area? → change_location (NOT for in-place actions)
4. Did I describe a discoverable, interactable scene object (treasure, lever, body, chest)? → spawn_object
5. Did the PLAYER's state change — picked up / lost / spent items, gained / spent coin, took \
   environmental damage, was healed, gained / lost a condition? → update_player \
   (NOT for offers the player hasn't accepted yet — those are pure prose)
6. Did an existing roster entity's disposition / status / importance / inventory shift? \
   → update_entity (NOT for plain references — that's ref_entity)
7. Did the action's outcome depend on a roll? → request_roll, then stop describing the outcome
8. Did combat ACTUALLY start (situation escalated to a fight, hostile committed, player attacked)? → start_combat \
   (NOT for tension, drawn weapons, or threats)
9. Did the WORLD (trap, fire, fall, poison) hurt the player? → update_player(hp_delta < 0) \
   (NOT for combat attacks — the engine handles those)

If none of the above apply, narrate freely and emit ZERO tool calls. \
That is a valid turn.

# Worked examples

## Example 1 — multi-tool fire (prose + tools together)
ROSTER: [id: marta] Marta the herbalist (friendly)
PLAYER: I greet Marta and ask about the strange mushrooms.
PROSE: Marta looks up from her drying racks, pleased to see you. "Those? \
I'd stay away from those, dear — caps-of-shadow, we call them," she says \
warmly, gesturing at the wicker basket by the door.
TOOLS: ref_entity({entity_id: "marta", dialogue_indices: [1], dialogue_emotions: ["calm"]})

## Example 2 — legitimate empty turn (atmosphere only)
ROSTER: (no entities)
PLAYER: I keep walking, watching the horizon.
PROSE: The wind howls across the moor, your cloak snapping at your legs. \
The sun sinks behind the ridge, casting long shadows across the heather. \
Somewhere far off, a curlew cries.
TOOLS: (none — pure atmosphere, nothing trackable)

## Example 3 — precision (don't over-fire)
ROSTER: [id: bandits] Three bandits (hostile)
PLAYER: I tell them I'm not handing anything over and ready my blade.
PROSE: The bandits' leader spits and tightens his grip on his crossbow. \
Blades come up on both sides of the road. No one moves. Not yet.
TOOLS: ref_entity({entity_id: "bandits"})
NOT start_combat — no blow landed. NOT update_entity — disposition was \
already hostile. Tension is just prose.

# Reminder (re-read before sending)
1. Re-run the 9-item checklist against your prose.
2. Match every state change to a tool call. Skip tools that don't fit.
3. Output prose AND tool calls in the same response."""


def build_messages(scenario: Scenario) -> list[dict]:
    user_prompt = f"""ROSTER (existing tracked entities you may reference):
{scenario.roster}

CURRENT SCENE:
{scenario.scene}

RECENT NARRATION:
{scenario.recent or "(scene just started)"}

PLAYER ACTION:
{scenario.player_action}

Narrate the result. Use tools as appropriate per their descriptions."""
    return [
        {"role": "system", "content": NARRATOR_SYSTEM},
        {"role": "user", "content": user_prompt},
    ]


async def run_scenario(client, tools: list[dict], scenario: Scenario, think: bool = False) -> ScenarioResult:
    """Run one scenario.

    NOTE on ``think``: defaults to False.

    The empty-content regression (ollama#10976) reproduces specifically
    on Qwen3 MoE models (qwen3:30b-a3b, qwen3.6:latest). Dense models
    like qwen3.5:27b don't trip it — they produce prose with think=False
    just fine, and skip the wasted thinking-pass latency.

    For MoE Qwen3 narrators, override with --think on the CLI.
    """
    messages = build_messages(scenario)
    start = time.time()
    try:
        # Base kwargs that every client supports.
        # No max_tokens cap — context window has ~23k tokens of headroom
        # after the ~9k-token prompt, and Qwen3's thinking pass + prose
        # + tool emission needs every token of that. Any cap risks the
        # model running out mid-prose.
        chat_kwargs = dict(
            messages=messages,
            temperature=0.7,
            tools=tools,
            think=think,
        )

        # Qwen3-specific sampling: applies ONLY when the underlying model
        # is in the Qwen3 family (per Qwen/Qwen3 model card). These values
        # are tuned for Qwen3's behavior; applying them to DeepSeek /
        # Sonnet / Gemini would mistune those models. Detect by model
        # name on the OllamaClient — cloud providers run different
        # families and stay on their own defaults.
        from dnd_bot.llm.client import OllamaClient
        if isinstance(client, OllamaClient):
            model_lower = (client.model or "").lower()
            if "qwen3" in model_lower:
                chat_kwargs["top_p"] = 0.8
                chat_kwargs["top_k"] = 20
                chat_kwargs["min_p"] = 0.0
                chat_kwargs["presence_penalty"] = 1.0

        response = await client.chat(**chat_kwargs)
        elapsed = time.time() - start
        raw_calls = response.tool_calls or []
        prose = response.content or ""
        return grade(scenario, raw_calls, prose, elapsed)
    except Exception as e:
        return grade(scenario, [], "", time.time() - start, error=str(e))


def render_result(idx: int, result: ScenarioResult):
    sc = result.scenario
    if result.error:
        status = f"{C.RED}ERROR{C.RESET}"
    elif result.passed:
        status = f"{C.GREEN}PASS{C.RESET}"
    else:
        status = f"{C.RED}FAIL{C.RESET}"
    print(f"\n[{idx:02d}] {status}  {C.BOLD}{sc.name}{C.RESET}  ({result.elapsed_s:.1f}s)")
    print(f"     {C.DIM}{sc.description}{C.RESET}")
    if result.error:
        print(f"     {C.RED}error: {result.error}{C.RESET}")
        return
    fired_names = [c.get("name", "?") for c in result.raw_calls]
    print(f"     fired: {fired_names or '(none)'}")
    for tool, (fired, args_ok) in result.expected_hits.items():
        if fired and args_ok:
            print(f"       {C.GREEN}✓{C.RESET} {tool} (args ok)")
        elif fired and not args_ok:
            print(f"       {C.YELLOW}~{C.RESET} {tool} (fired but args off)")
        else:
            print(f"       {C.RED}✗{C.RESET} {tool} (missed)")
    if result.forbidden_violations:
        print(f"     {C.RED}forbidden fired: {result.forbidden_violations}{C.RESET}")


def render_summary(results: list[ScenarioResult], model_name: str, tier: str):
    header(f"SUMMARY — {model_name} (tier: {tier})")
    n = len(results)
    passed = sum(r.passed for r in results)
    errors = sum(1 for r in results if r.error)
    print(f"  Total      : {n}")
    print(f"  {C.GREEN}Passed     : {passed}/{n} ({100*passed/n:.0f}%){C.RESET}")
    print(f"  {C.RED}Errors     : {errors}{C.RESET}")
    print(f"  Avg latency: {sum(r.elapsed_s for r in results)/max(n,1):.1f}s")

    # Per-tool recall
    print(f"\n  {C.BOLD}Per-tool recall (expected → fired):{C.RESET}")
    tool_stats: dict[str, list[bool]] = {}
    for r in results:
        for tool, (fired, _) in r.expected_hits.items():
            tool_stats.setdefault(tool, []).append(fired)
    for tool, hits in sorted(tool_stats.items()):
        rate = sum(hits) / len(hits)
        color = C.GREEN if rate >= 0.8 else (C.YELLOW if rate >= 0.5 else C.RED)
        print(f"    {tool:20s}  {color}{sum(hits)}/{len(hits)}  ({100*rate:.0f}%){C.RESET}")

    # Forbidden firings
    print(f"\n  {C.BOLD}Forbidden-tool violations:{C.RESET}")
    forbidden_by_tool: dict[str, int] = {}
    for r in results:
        for v in r.forbidden_violations:
            forbidden_by_tool[v] = forbidden_by_tool.get(v, 0) + 1
    if not forbidden_by_tool:
        print(f"    {C.GREEN}none{C.RESET}")
    else:
        for tool, count in sorted(forbidden_by_tool.items(), key=lambda x: -x[1]):
            print(f"    {C.RED}{tool}{C.RESET}: fired {count}x when forbidden")


# ── Main ───────────────────────────────────────────────────────────────

async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--profile", help="Profile to load")
    parser.add_argument("--tier", default="full", choices=["core", "core_plus", "full"])
    parser.add_argument("--quick", action="store_true", help="Run 6-scenario smoke set")
    parser.add_argument("--scenarios", type=int, nargs="+", help="Specific scenario indices")
    parser.add_argument("--think", action="store_true", help="Enable thinking mode (Qwen3 etc.)")
    args = parser.parse_args()

    if args.profile:
        os.environ["ACTIVE_PROFILE"] = args.profile

    from dnd_bot.config import get_profile
    from dnd_bot.llm.client import get_narrator_client_for
    from dnd_bot.llm.narrator_tools import get_narrator_tools_for_tier

    profile = get_profile()
    model_name = f"{profile.narrator.provider}/{profile.narrator.model}"
    header(f"NARRATOR TOOL-CALLING BENCHMARK — {model_name} ({args.tier} tier)")

    client = get_narrator_client_for("standard")
    tools = get_narrator_tools_for_tier(args.tier)

    print(f"  Profile : {profile.name}")
    print(f"  Model   : {model_name}")
    print(f"  Tier    : {args.tier} ({len(tools)} tools available)")
    print(f"  Tools   : {[t['function']['name'] for t in tools]}")

    # Scenario selection
    if args.scenarios:
        selected = [SCENARIOS[i] for i in args.scenarios if 0 <= i < len(SCENARIOS)]
    elif args.quick:
        selected = SCENARIOS[:6]
    else:
        selected = SCENARIOS

    print(f"  Running {len(selected)} scenarios...\n")

    results: list[ScenarioResult] = []
    for i, sc in enumerate(selected):
        print(f"  [{i+1}/{len(selected)}] {sc.name}...", end=" ", flush=True)
        r = await run_scenario(client, tools, sc, think=args.think)
        if r.error:
            print(f"{C.RED}error{C.RESET}")
        elif r.passed:
            print(f"{C.GREEN}pass{C.RESET}")
        else:
            print(f"{C.RED}fail{C.RESET}")
        results.append(r)

    header("DETAILED RESULTS")
    for i, r in enumerate(results):
        render_result(i, r)

    render_summary(results, model_name, args.tier)


if __name__ == "__main__":
    asyncio.run(main())
