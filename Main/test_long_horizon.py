"""Long-horizon memory + KG test (emergent-callback design).

Drives the production orchestrator through N turns of a Gemini-Flash
player. The narrator opens with whatever scene it likes — we don't fight
it. After a few turns, the framework asks Gemini to identify ONE
concrete element from the established prose (an NPC, item, or location)
that the player will deliberately return to many turns later. The rest
of the run pursues a "do filler, then come back" arc with the seed
chosen organically.

This tests what we actually care about: does the architecture retain
arbitrary established state across many turns? Does the KG surface it
when the player references it later? Does the narrator's recall work?

Usage::

    python test_long_horizon.py                              # default profile
    python test_long_horizon.py --profile deepseek_v4_flash
    python test_long_horizon.py --turns 22
    python test_long_horizon.py --scripted                   # scripted, no API key needed
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
from typing import Callable, Optional

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


# ── Scenario / phase / seed ────────────────────────────────────────────


@dataclass
class Phase:
    """One stretch of turns the player should pursue a sub-goal.

    The instruction can include ``{seed_name}`` and ``{seed_type}``
    placeholders — they'll be filled in once the framework has picked
    the seed (after the explore phase). Phases that run before the
    seed pick must not reference these placeholders.
    """
    name: str
    turn_range: tuple[int, int]
    instruction: str


@dataclass
class Seed:
    """A concrete element identified from the narrator's prose, that
    the player will deliberately come back to in later turns."""
    type: str    # "npc" | "item" | "place"
    name: str    # the actual name or short phrase
    reason: str  # why memorable (for logging only)
    chosen_after_turn: int  # the framework picked it after this turn


@dataclass
class Scenario:
    name: str
    description: str
    base_goal: str               # initial goal text (no seed yet)
    phases: list[Phase]
    total_turns: int
    seed_pick_after_turn: int    # framework picks seed after this turn
    seed_pick_prompt: str        # framework's prompt to Gemini for seed selection


@dataclass
class AssertionResult:
    name: str
    passed: bool
    description: str
    detail: str = ""


# ── The flagship scenario: emergent callback ──────────────────────────


SCENARIOS: dict[str, Scenario] = {
    "emergent_callback": Scenario(
        name="emergent_callback",
        description=(
            "Narrator opens freely; after the explore phase, the framework "
            "asks Gemini to pick one concrete element from the emerged scene "
            "to come back to later. Player does filler, then references the "
            "seed at the callback. Assertions check whether the architecture "
            "retained the seed across the gap."
        ),
        base_goal=(
            "You are a D&D adventurer. Engage with whatever the DM "
            "establishes. Be curious about the scene — items, NPCs, "
            "places. Later in the session, you'll deliberately return "
            "to ONE memorable element you encountered early on."
        ),
        phases=[
            Phase(
                name="explore",
                turn_range=(1, 5),
                instruction=(
                    "Engage with the scene naturally. Look around, "
                    "interact, talk to NPCs if any are present. Stay "
                    "curious — the early scene matters because you'll "
                    "come back to part of it later."
                ),
            ),
            Phase(
                name="filler",
                turn_range=(6, 16),
                instruction=(
                    "Do unrelated mundane adventuring — explore new "
                    "places, take small jobs, talk to other people. "
                    "Do NOT mention or return to the {seed_type} "
                    "called \"{seed_name}\" during this phase. Drift "
                    "the story elsewhere."
                ),
            ),
            Phase(
                name="callback",
                turn_range=(17, 22),
                instruction=(
                    "Now return to or specifically reference the {seed_type} "
                    "\"{seed_name}\" from earlier. Ask about it, go back to "
                    "it, mention it by name to anyone who might know. The "
                    "key thing: SAY THE NAME OF \"{seed_name}\"."
                ),
            ),
        ],
        total_turns=22,
        seed_pick_after_turn=5,
        seed_pick_prompt=(
            "You are the player in a D&D session. Below are the first "
            "few turns of narration. Identify ONE concrete, named element "
            "from what the DM established that you will deliberately return "
            "to in many turns. Pick something with a clear name, not vague "
            "atmosphere.\n\n"
            "Output ONLY a single JSON object on one line, no prose:\n"
            '{"type": "npc"|"item"|"place", "name": "<exact name>", "reason": "<why it stands out>"}\n\n'
            "Examples:\n"
            '  {"type": "npc", "name": "Marta", "reason": "old herbalist who hinted at a hidden cave"}\n'
            '  {"type": "item", "name": "jade serpent", "reason": "carved relic on a stone altar"}\n'
            '  {"type": "place", "name": "the moss-slick altar", "reason": "central scene piece"}\n'
            "\n"
            "Recent narration:\n"
        ),
    ),
}


# ── Player abstractions ────────────────────────────────────────────────


class ScriptedPlayer:
    """Plays a fixed list of actions, one per turn. Useful for
    framework debugging without burning Gemini quota."""

    def __init__(self, actions: list[str]):
        self.actions = actions
        self.turn = 0

    async def next_action(self, narrator_response: str, phase: Phase, seed: Optional[Seed]) -> str:
        if self.turn >= len(self.actions):
            return "I look around and consider my next move."
        action = self.actions[self.turn]
        self.turn += 1
        return action

    async def pick_seed(self, narration_history: list[str], scenario: Scenario) -> Seed:
        return Seed(type="item", name="lantern", reason="(scripted fallback)", chosen_after_turn=scenario.seed_pick_after_turn)


class GeminiFlashPlayer:
    """Plays a D&D character via Gemini Flash with persistent memory of
    the goal + history of recent narrator responses. Also handles the
    framework's seed-identification call."""

    def __init__(self, scenario: Scenario, history_window: int = 6):
        from dnd_bot.llm.client import GeminiClient
        self.client = GeminiClient(model="gemini-2.5-flash")
        self.scenario = scenario
        self.history_window = history_window
        self.history: list[tuple[str, str]] = []  # (action, narrator_response)

    async def next_action(self, narrator_response: str, phase: Phase, seed: Optional[Seed]) -> str:
        # Append the previous narrator response to history (for context)
        if self.history:
            self.history[-1] = (self.history[-1][0], narrator_response)

        # Resolve phase instruction with seed substitutions
        instruction = phase.instruction
        if seed:
            instruction = instruction.replace("{seed_name}", seed.name).replace("{seed_type}", seed.type)
        elif "{seed_" in instruction:
            instruction = (
                "Engage with the scene the narrator establishes. Do not "
                "get attached to specific items yet."
            )

        # Build prompt
        recent = self.history[-self.history_window:]
        history_block = ""
        if recent:
            lines = []
            for i, (action, response) in enumerate(recent):
                lines.append(f"  Turn {i+1}: I said \"{action}\"")
                if response:
                    lines.append(f"    DM: {response[:200]}")
            history_block = "Recent turns:\n" + "\n".join(lines) + "\n\n"

        seed_note = ""
        if seed:
            seed_note = (
                f"\nMEMORY: Earlier you noted a {seed.type} called "
                f"\"{seed.name}\" ({seed.reason}). Use this in the "
                f"appropriate phase per your instruction.\n"
            )

        system = f"""You are role-playing a D&D 5e character. Speak in first-person ("I do X").

OVERALL GOAL:
{self.scenario.base_goal}

CURRENT PHASE ({phase.name}, turns {phase.turn_range[0]}-{phase.turn_range[1]}):
{instruction}
{seed_note}

Output rules:
- Reply with ONE concrete first-person action sentence.
- Do NOT narrate the world or NPCs — only YOUR character's intent.
- No markdown, no headings, no quotes, no commentary.
- Keep it to 15-30 words. Be specific and actionable."""

        user = (
            history_block
            + f"DM's last response: {narrator_response[:400] if narrator_response else '(scene begins)'}\n\n"
            + "What does your character do next?"
        )

        response = await self.client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
            max_tokens=200,  # generous; scenarios with truncated actions hurt the test
        )
        action = (response.content or "").strip()

        # Strip leading bullets/quotes the model occasionally adds
        for prefix in ('"', "'", "- ", "* ", "I:", "Player:", "Action:"):
            if action.startswith(prefix):
                action = action[len(prefix):].strip()
        if action.endswith('"') or action.endswith("'"):
            action = action[:-1].strip()

        if not action or len(action) < 5:
            action = "I look around and take stock of my surroundings."
        self.history.append((action, ""))
        return action

    async def pick_seed(self, narration_history: list[str], scenario: Scenario) -> Seed:
        """Ask Gemini to identify ONE concrete element from the early
        narration to come back to later.

        Uses ``json_mode=True`` so Gemini's ``response_mime_type`` is set
        to ``application/json`` — without it Gemini Flash truncates the
        emitted JSON mid-token (observed at 23 chars in early runs).
        """
        narr_text = "\n---\n".join(
            f"Turn {i+1}:\n{n}" for i, n in enumerate(narration_history) if n
        )

        response = await self.client.chat(
            messages=[
                {"role": "system", "content": (
                    "You analyze D&D narration and identify the most "
                    "memorable concrete element to return to later. "
                    "Respond with EXACTLY ONE JSON object."
                )},
                {"role": "user", "content": scenario.seed_pick_prompt + narr_text},
            ],
            temperature=0.3,
            max_tokens=400,
            json_mode=True,  # Gemini's structured-output mode — fixes truncation
        )
        raw = (response.content or "").strip()

        # Extract the JSON object even if the model added extra text
        seed_data = None
        # Try to find {...} substring
        start = raw.find("{")
        end = raw.rfind("}")
        if start >= 0 and end > start:
            try:
                seed_data = json.loads(raw[start:end + 1])
            except json.JSONDecodeError:
                pass

        if not seed_data:
            print(f"  {C.YELLOW}seed_pick_parse_failed: {raw[:200]!r}{C.RESET}")
            # Fallback: pick a generic seed from the most recent narration
            seed_data = {"type": "place", "name": "the scene", "reason": "(fallback)"}

        return Seed(
            type=str(seed_data.get("type", "place")),
            name=str(seed_data.get("name", "the scene")).strip(),
            reason=str(seed_data.get("reason", "")),
            chosen_after_turn=scenario.seed_pick_after_turn,
        )


def _phase_for_turn(scenario: Scenario, turn: int) -> Phase:
    for phase in scenario.phases:
        lo, hi = phase.turn_range
        if lo <= turn <= hi:
            return phase
    return scenario.phases[-1]


# ── Driver ─────────────────────────────────────────────────────────────


async def run_long_horizon(
    scenario: Scenario,
    use_gemini: bool = True,
    profile: Optional[str] = None,
    turn_override: Optional[int] = None,
) -> dict:
    """Run the scenario end-to-end. Returns {session_id, seed, turn_records}."""
    if profile:
        os.environ["ACTIVE_PROFILE"] = profile

    from test_harness import TestSession

    header(f"LONG-HORIZON TEST — {scenario.name}")
    print(f"  Description : {scenario.description[:120]}...")
    print(f"  Profile     : {os.environ.get('ACTIVE_PROFILE', '(default)')}")
    print(f"  Player      : {'Gemini Flash (LLM)' if use_gemini else 'Scripted'}")
    n_turns = turn_override or scenario.total_turns
    print(f"  Turns       : {n_turns}")
    print(f"  Seed pick   : after turn {scenario.seed_pick_after_turn}\n")

    # Player
    if use_gemini:
        player = GeminiFlashPlayer(scenario=scenario)
    else:
        # Generic scripted actions — work with whatever the narrator establishes
        scripted = [
            "I look around and take in the scene.",
            "I check what's nearby that catches my eye.",
            "I move closer to whatever stands out and study it.",
            "I think about what this could mean.",
            "I take a moment to remember this place.",
        ] + ["I continue exploring the area."] * (n_turns - 5)
        player = ScriptedPlayer(actions=scripted)

    # Harness session
    session = TestSession()
    await session.setup()
    # GameSessionManager stores sessions keyed by ``f"discord:{channel_id}"``
    # (the session_key). Going through ``get_session`` builds the right key.
    ws_session = session.manager.get_session(session.channel_id)
    session_id = ws_session.id if ws_session else None
    print(f"  Session ID  : {session_id or '(unknown)'}\n")

    last_response_text = ""
    narration_history: list[str] = []  # for seed picking
    seed: Optional[Seed] = None
    turn_records: list[dict] = []

    for turn in range(1, n_turns + 1):
        # Pick the seed AFTER the explore phase but before the next action
        if seed is None and turn == scenario.seed_pick_after_turn + 1:
            try:
                seed = await player.pick_seed(narration_history, scenario)
                print(f"\n{C.MAGENTA}{C.BOLD}  >>> SEED IDENTIFIED <<< {C.RESET}")
                print(f"  {C.MAGENTA}type:   {seed.type}{C.RESET}")
                print(f"  {C.MAGENTA}name:   {seed.name}{C.RESET}")
                print(f"  {C.MAGENTA}reason: {seed.reason}{C.RESET}\n")
            except Exception as e:
                print(f"  {C.RED}seed_pick_error: {e}{C.RESET}")
                seed = Seed(type="place", name="the scene", reason="(error fallback)",
                            chosen_after_turn=scenario.seed_pick_after_turn)

        phase = _phase_for_turn(scenario, turn)
        try:
            action = await player.next_action(last_response_text, phase, seed)
        except Exception as e:
            print(f"  {C.RED}player_error_turn_{turn}: {e}{C.RESET}")
            action = "I pause and look around."

        print(f"\n{C.DIM}--- Turn {turn} [{phase.name}] ---{C.RESET}")
        print(f"  {C.CYAN}Player:{C.RESET} {action}")

        start = time.time()
        try:
            response = await session.send_action(action)
        except Exception as e:
            print(f"  {C.RED}orchestrator_error: {e}{C.RESET}")
            turn_records.append({"turn": turn, "action": action, "error": str(e)})
            continue
        elapsed = time.time() - start

        last_response_text = (response.narrative or "") if response else ""
        narration_history.append(last_response_text)
        preview = last_response_text[:200].replace("\n", " ")
        print(f"  {C.GREEN}Narrator ({elapsed:.1f}s):{C.RESET} {preview}...")

        turn_records.append({
            "turn": turn,
            "action": action,
            "elapsed": elapsed,
            "narrative_chars": len(last_response_text),
        })

    await session.cleanup()

    return {
        "scenario": scenario.name,
        "session_id": session_id,
        "seed": seed.__dict__ if seed else None,
        "turn_records": turn_records,
        "total_turns": n_turns,
    }


# ── Assertions (parameterized on the seed) ─────────────────────────────


def run_assertions(scenario: Scenario, session_id: str, seed: Seed) -> list[AssertionResult]:
    """Load the turn log and run scenario assertions, parameterized on
    the seed Gemini chose at the explore boundary."""
    from dnd_bot.llm.turn_log_reader import TurnLogReader

    try:
        log = TurnLogReader.load(session_id)
    except FileNotFoundError as e:
        return [AssertionResult("__load__", False, f"Could not load log: {e}")]

    print(f"\n  Log loaded: {len(log)} turn records")
    print(f"  Turns      : {log.turns()}\n")

    seed_name = seed.name.lower()
    seed_first_word = seed_name.split()[0] if seed_name else ""

    callback_phase = next(p for p in scenario.phases if p.name == "callback")
    cb_lo, cb_hi = callback_phase.turn_range
    callback_turns = [t for t in log.turns() if cb_lo <= t <= cb_hi]

    explore_phase = next(p for p in scenario.phases if p.name == "explore")
    ex_lo, ex_hi = explore_phase.turn_range
    explore_turns = [t for t in log.turns() if ex_lo <= t <= ex_hi]

    filler_phase = next(p for p in scenario.phases if p.name == "filler")
    fi_lo, fi_hi = filler_phase.turn_range
    filler_turns = [t for t in log.turns() if fi_lo <= t <= fi_hi]

    results: list[AssertionResult] = []

    # 1. The seed actually appeared in the explore phase narration
    appeared_in_explore = any(
        seed_name in log.narrator_response(t).text.lower()
        or seed_first_word in log.narrator_response(t).text.lower()
        for t in explore_turns
    )
    results.append(AssertionResult(
        name="seed_appears_in_explore",
        passed=appeared_in_explore,
        description=f"Seed '{seed.name}' appeared in narrator prose during explore phase.",
        detail=f"Searched turns {explore_turns}",
    ))

    # 2. Player references seed at the callback (they're instructed to)
    player_referenced_at_callback = any(
        seed_name in log.player_action(t).lower()
        or seed_first_word in log.player_action(t).lower()
        for t in callback_turns
    )
    results.append(AssertionResult(
        name="player_references_seed_at_callback",
        passed=player_referenced_at_callback,
        description=f"Player's input mentioned '{seed.name}' during callback phase.",
        detail=f"Searched turns {callback_turns}",
    ))

    # 3. Narrator's response at callback acknowledges the seed
    narrator_referenced_at_callback = any(
        seed_name in log.narrator_response(t).text.lower()
        or seed_first_word in log.narrator_response(t).text.lower()
        for t in callback_turns
    )
    results.append(AssertionResult(
        name="narrator_references_seed_at_callback",
        passed=narrator_referenced_at_callback,
        description=f"Narrator's prose at callback referenced '{seed.name}'.",
        detail=f"Searched turns {callback_turns}",
    ))

    # 4. KG context at callback contained the seed (memory retrieval worked)
    kg_surfaced = any(
        log.kg_context_for(t).mentions(seed_name)
        or log.kg_context_for(t).mentions(seed_first_word)
        for t in callback_turns
    )
    results.append(AssertionResult(
        name="kg_surfaced_seed_at_callback",
        passed=kg_surfaced,
        description=f"KG context injected into narrator at callback mentioned '{seed.name}'.",
        detail=f"Searched turns {callback_turns}",
    ))

    # 5. WorldState retained something about the seed across the gap
    # (NPC inventory / NPC presence / scene_items / facts / location)
    retained_in_world_state = False
    last_filler = max(filler_turns) if filler_turns else None
    if last_filler:
        ws_yaml = log.world_state_after(last_filler).raw_yaml.lower()
        retained_in_world_state = (seed_name in ws_yaml) or (seed_first_word and seed_first_word in ws_yaml)
    results.append(AssertionResult(
        name="world_state_retained_seed_through_filler",
        passed=retained_in_world_state,
        description=f"WorldState YAML at end of filler still contained '{seed.name}'.",
        detail=f"Checked turn {last_filler}",
    ))

    # 6. SOMETHING was emitted around the seed at explore phase (a tool fire)
    # — proves the architecture saw the element as worth tracking
    seed_tool_fired = False
    for t in explore_turns:
        effs = log.effects_at(t)
        for e in effs.effects:
            blob = json.dumps(e).lower()
            if seed_name in blob or seed_first_word in blob:
                seed_tool_fired = True
                break
        if seed_tool_fired:
            break
    results.append(AssertionResult(
        name="tool_fired_for_seed_in_explore",
        passed=seed_tool_fired,
        description=f"At least one tool call referenced '{seed.name}' during explore phase.",
        detail=f"Checked turns {explore_turns}",
    ))

    return results


def render_results(scenario: Scenario, seed: Seed, results: list[AssertionResult]):
    header(f"ASSERTIONS — {scenario.name}")
    print(f"  Seed       : {C.MAGENTA}{seed.type} '{seed.name}'{C.RESET}")
    print(f"  Reason     : {C.DIM}{seed.reason}{C.RESET}\n")

    passed = 0
    for r in results:
        marker = f"{C.GREEN}PASS{C.RESET}" if r.passed else f"{C.RED}FAIL{C.RESET}"
        print(f"  [{marker}] {C.BOLD}{r.name}{C.RESET}")
        print(f"         {C.DIM}{r.description}{C.RESET}")
        if r.detail:
            print(f"         {C.DIM}({r.detail}){C.RESET}")
        if r.passed:
            passed += 1

    total = len(results)
    color = C.GREEN if passed == total else (C.YELLOW if passed >= total // 2 else C.RED)
    print(f"\n  {color}{C.BOLD}{passed}/{total} assertions passed{C.RESET}")


# ── Main ───────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="emergent_callback",
                        choices=list(SCENARIOS.keys()))
    parser.add_argument("--profile", help="Profile to use (overrides ACTIVE_PROFILE)")
    parser.add_argument("--turns", type=int, help="Override total turn count")
    parser.add_argument("--scripted", action="store_true",
                        help="Use scripted actions instead of Gemini Flash")
    args = parser.parse_args()

    scenario = SCENARIOS[args.scenario]

    result = await run_long_horizon(
        scenario=scenario,
        use_gemini=not args.scripted,
        profile=args.profile,
        turn_override=args.turns,
    )

    if not result.get("session_id"):
        print(f"\n  {C.RED}No session_id captured — cannot run assertions{C.RESET}")
        return

    if not result.get("seed"):
        print(f"\n  {C.RED}No seed identified — cannot run assertions{C.RESET}")
        return

    seed = Seed(**result["seed"])
    assertions = run_assertions(scenario, result["session_id"], seed)
    render_results(scenario, seed, assertions)


if __name__ == "__main__":
    asyncio.run(main())
