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
    python test_long_horizon.py --assert-only data/long_horizon/<stem>.manifest.json

On every run the harness writes two crash-survivable artifacts under
``data/long_horizon/``:

- ``{YYYYMMDD_HHMMSS}_{profile}.jsonl`` — one flushed line per turn with the
  action, elapsed time, narrative size, and the LLM-usage delta for that turn.
- ``{same stem}.manifest.json`` — session id + seed + phase config written at
  seed-pick time, finalized in the ``finally`` block with the assertion results
  and the end-of-run usage/latency/cost roll-up.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import statistics
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


# ── Artifacts / pricing constants ──────────────────────────────────────

LONG_HORIZON_DIR = Path("data/long_horizon")

# Leading words stripped from a seed name before matching (FIX-2). A seed
# like "the moss-slick altar" must still match narration that says just
# "altar", so we match on the cleaned name OR its longest content token.
SEED_STOPWORDS = {"the", "a", "an", "of", "to", "in"}

# Degenerate seeds the framework falls back to when Gemini's seed-pick call
# fails or returns garbage (see GeminiFlashPlayer.pick_seed / the seed-pick
# except handler). A run seeded with one of these can't be trusted to prove
# recall, so its verdict is stamped UNTRUSTED rather than green (FIX-2).
FALLBACK_SEED_NAMES = {"the scene", "scene", ""}

# Per-1M-token USD prices keyed (provider, model-prefix); longest matching
# prefix wins. Unknown (provider, model) -> cost None + a warning line, never
# a crash. Deepseek/Groq/Ollama numbers per PROMPT_CACHING_2026_07.md §2;
# Gemini + Anthropic looked up 2026-07-17 (ai.google.dev/gemini-api/docs/pricing,
# platform.claude.com/docs/en/about-claude/pricing) — NOT guessed.
PRICING: dict[tuple[str, str], dict[str, float]] = {
    # DeepSeek narrator tiers (automatic on-disk cache; cache HIT is ~1/50 of miss)
    ("deepseek", "deepseek-v4-flash"): {"in": 0.14, "cached_in": 0.0028, "out": 0.28},
    ("deepseek", "deepseek-v4-pro"): {"in": 0.435, "cached_in": 0.003625, "out": 0.87},
    # Legacy names deprecate 2026-07-24; they currently alias v4-flash.
    ("deepseek", "deepseek-chat"): {"in": 0.14, "cached_in": 0.0028, "out": 0.28},
    ("deepseek", "deepseek-reasoner"): {"in": 0.14, "cached_in": 0.0028, "out": 0.28},
    # Groq qwen — no cache discount exists for this model (cached_in == in).
    ("groq", "qwen/qwen3-32b"): {"in": 0.29, "cached_in": 0.29, "out": 0.59},
    # Ollama local — everything is free. Empty prefix matches any local model.
    ("ollama", ""): {"in": 0.0, "cached_in": 0.0, "out": 0.0},
    # Gemini player model (+ pro), verified 2026-07-17.
    ("gemini", "gemini-2.5-flash"): {"in": 0.30, "cached_in": 0.03, "out": 2.50},
    ("gemini", "gemini-2.5-pro"): {"in": 1.25, "cached_in": 0.125, "out": 10.00},
    # Anthropic narrators, verified 2026-07-17. cached_in == cache-read price
    # (0.1x base); the anthropic cost branch derives write premium from `in`.
    ("anthropic", "claude-sonnet-4"): {"in": 3.0, "cached_in": 0.30, "out": 15.0},
    ("anthropic", "claude-sonnet-5"): {"in": 2.0, "cached_in": 0.20, "out": 10.0},
    ("anthropic", "claude-opus-4"): {"in": 5.0, "cached_in": 0.50, "out": 25.0},
    ("anthropic", "claude-haiku-4"): {"in": 1.0, "cached_in": 0.10, "out": 5.0},
}

# Providers whose reported prompt_tokens INCLUDE the cached slice (so the
# uncached input = prompt - cache_read). Anthropic is the exception: its
# input_tokens EXCLUDE cache read/write, which are billed on top.
_PROMPT_INCLUDES_CACHE = {"deepseek", "gemini", "groq", "openrouter", "ollama"}


def _price_for(provider: str, model: str) -> Optional[dict[str, float]]:
    """Longest-prefix price lookup. None when unknown (caller reports it)."""
    candidates = [
        (prefix, price)
        for (prov, prefix), price in PRICING.items()
        if prov == provider and (prefix == "" or model.startswith(prefix))
    ]
    if not candidates:
        return None
    return max(candidates, key=lambda kv: len(kv[0]))[1]


def _event_cost(ev) -> Optional[float]:
    """USD cost of one usage event, provider-aware. None if price unknown.

    Provider-aware cost math (prompt-token semantics differ by provider):
      deepseek/gemini/groq/openrouter/ollama — prompt_tokens INCLUDES the
        cached slice, so:
          cost = (prompt - cache_read)*in + cache_read*cached_in + completion*out
      anthropic — input_tokens EXCLUDE cached read/write, billed on top:
          cost = prompt*in + cache_read*(0.1*in) + cache_write*(1.25*in) + completion*out
    All rates are USD per 1M tokens.
    """
    price = _price_for(ev.provider, ev.model)
    if price is None:
        return None
    in_, cached_in, out = price["in"], price["cached_in"], price["out"]
    if ev.provider == "anthropic":
        cost = (
            ev.prompt_tokens * in_
            + ev.cache_read_tokens * (0.1 * in_)
            + ev.cache_write_tokens * (1.25 * in_)
            + ev.completion_tokens * out
        )
    else:
        uncached = max(ev.prompt_tokens - ev.cache_read_tokens, 0)
        cost = (
            uncached * in_
            + ev.cache_read_tokens * cached_in
            + ev.completion_tokens * out
        )
    return cost / 1_000_000.0


def _event_prompt_denom(ev) -> int:
    """Denominator for this event's cache-hit ratio, provider-aware.

    Matches the cost math: anthropic prompt EXCLUDES cache, everyone else's
    prompt INCLUDES it.
    """
    if ev.provider == "anthropic":
        return ev.prompt_tokens + ev.cache_read_tokens + ev.cache_write_tokens
    return ev.prompt_tokens


def _percentiles(values: list[float], ps=(0.50, 0.95)) -> dict[float, Optional[float]]:
    """p50/p95 by linear interpolation on the sorted sample. No new deps.

    (statistics.quantiles(n=20) needs >=2 points and picks fixed cut points;
    manual interpolation degrades cleanly to the single value when n==1.)
    """
    vals = sorted(v for v in values if v is not None)
    out: dict[float, Optional[float]] = {p: None for p in ps}
    if not vals:
        return out
    if len(vals) == 1:
        return {p: vals[0] for p in ps}
    for p in ps:
        idx = p * (len(vals) - 1)
        lo = int(idx)
        frac = idx - lo
        hi = min(lo + 1, len(vals) - 1)
        out[p] = vals[lo] + (vals[hi] - vals[lo]) * frac
    return out


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
        # The player client is built directly (NOT via _create_client) so it
        # is NOT auto-instrumented by the usage recorder — we record its usage
        # by hand after each chat() call below.
        from dnd_bot.llm import usage_recorder
        self._usage = usage_recorder
        self.client = GeminiClient(model="gemini-2.5-flash")
        self.scenario = scenario
        self.history_window = history_window
        self.history: list[tuple[str, str]] = []  # (action, narrator_response)

    def _record_usage(self, response, elapsed_ms: float) -> None:
        """Feed one player/seed-pick LLMResponse into the usage recorder."""
        try:
            self._usage.record(
                provider="gemini",
                model=getattr(self.client, "model", "gemini-2.5-flash"),
                stage="player",
                prompt_tokens=getattr(response, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(response, "completion_tokens", 0) or 0,
                cache_read_tokens=getattr(response, "cache_read_tokens", 0) or 0,
                cache_write_tokens=getattr(response, "cache_write_tokens", 0) or 0,
                elapsed_ms=elapsed_ms,
            )
        except Exception:
            pass  # telemetry must never break a run

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

        _t0 = time.perf_counter()
        response = await self.client.chat(
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=0.7,
            max_tokens=200,  # generous; scenarios with truncated actions hurt the test
        )
        self._record_usage(response, (time.perf_counter() - _t0) * 1000.0)
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

        _t0 = time.perf_counter()
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
            # Gemini 2.5 Flash has adaptive thinking ON by default; those
            # reasoning tokens draw from the same output budget. At 400 the
            # thinking consumed it all and the JSON truncated mid-object
            # ('{"type":') — the fallback seed then made every recall verdict
            # UNTRUSTED. 2048 leaves room for thinking AND the small object.
            max_tokens=2048,
            json_mode=True,  # response_mime_type=application/json
        )
        self._record_usage(response, (time.perf_counter() - _t0) * 1000.0)
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


# ── Seed cleaning / trust (FIX-2) ──────────────────────────────────────


def _clean_seed(seed: Seed) -> tuple[str, str]:
    """Return ``(cleaned_name, content_token)`` for matching (FIX-2).

    ``cleaned_name`` is the lowercased seed name with leading articles/
    stopwords stripped ("the moss-slick altar" -> "moss-slick altar").
    ``content_token`` is the longest non-stopword token of length >= 4
    ("altar"), or "" if none qualifies. Assertions match narration against
    the cleaned name OR the content token, so a seed named with a leading
    article still matches prose that drops it.
    """
    tokens = seed.name.lower().split()
    while tokens and tokens[0] in SEED_STOPWORDS:
        tokens.pop(0)
    cleaned = " ".join(tokens)
    content = [t for t in tokens if t not in SEED_STOPWORDS and len(t) >= 4]
    token = max(content, key=len) if content else ""
    return cleaned, token


def _seed_is_fallback(seed: Optional[Seed]) -> bool:
    """True when the seed is a degenerate framework fallback (FIX-2)."""
    if seed is None:
        return True
    return seed.name.strip().lower() in FALLBACK_SEED_NAMES


def _seed_matches(text: str, cleaned: str, token: str) -> bool:
    """Does ``text`` mention the seed by cleaned name or content token?"""
    low = text.lower()
    if cleaned and cleaned in low:
        return True
    if token and token in low:
        return True
    return False


# ── Driver ─────────────────────────────────────────────────────────────


def _resolve_profile_label(profile: Optional[str]) -> str:
    """Human/filesystem-safe profile name for artifact stems."""
    label = profile or os.environ.get("ACTIVE_PROFILE") or ""
    if not label:
        try:
            from dnd_bot.config import get_settings
            label = get_settings().active_profile
        except Exception:
            label = "default"
    return "".join(ch if (ch.isalnum() or ch in "-_.") else "_" for ch in label)


def _defensive_session_id(ws_session) -> str:
    """Session id via the defensive getattr chain (test_long_horizon_claude.py:129-135)."""
    return str(
        getattr(ws_session, "id", None)
        or getattr(ws_session, "session_id", None)
        or getattr(getattr(ws_session, "campaign", None), "id", None)
        or "unknown"
    )


def _aggregate_events(events) -> tuple[dict, dict]:
    """Sum a list of UsageEvents into a total dict + per-(provider/model/stage)
    breakdown dict."""
    total = {
        "llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
        "cache_read_tokens": 0, "cache_write_tokens": 0,
    }
    breakdown: dict[str, dict] = {}
    for ev in events:
        for bucket in (total, breakdown.setdefault(
            f"{ev.provider}/{ev.model}/{ev.stage}",
            {"llm_calls": 0, "prompt_tokens": 0, "completion_tokens": 0,
             "cache_read_tokens": 0, "cache_write_tokens": 0},
        )):
            bucket["llm_calls"] += 1
            bucket["prompt_tokens"] += ev.prompt_tokens
            bucket["completion_tokens"] += ev.completion_tokens
            bucket["cache_read_tokens"] += ev.cache_read_tokens
            bucket["cache_write_tokens"] += ev.cache_write_tokens
    return total, breakdown


async def run_long_horizon(
    scenario: Scenario,
    use_gemini: bool = True,
    profile: Optional[str] = None,
    turn_override: Optional[int] = None,
) -> dict:
    """Run the scenario end-to-end. Returns a result dict with session_id,
    seed, turn records, assertion results, verdict, and artifact paths."""
    if profile:
        os.environ["ACTIVE_PROFILE"] = profile

    from dnd_bot.llm import usage_recorder
    # Fresh capture for this run.
    usage_recorder.enable()
    usage_recorder.reset()

    from test_harness import TestSession

    header(f"LONG-HORIZON TEST — {scenario.name}")
    print(f"  Description : {scenario.description[:120]}...")
    print(f"  Profile     : {os.environ.get('ACTIVE_PROFILE', '(default)')}")
    print(f"  Player      : {'Gemini Flash (LLM)' if use_gemini else 'Scripted'}")
    n_turns = turn_override or scenario.total_turns
    print(f"  Turns       : {n_turns}")
    print(f"  Seed pick   : after turn {scenario.seed_pick_after_turn}\n")

    profile_label = _resolve_profile_label(profile)
    stem = f"{time.strftime('%Y%m%d_%H%M%S')}_{profile_label}"
    LONG_HORIZON_DIR.mkdir(parents=True, exist_ok=True)
    jsonl_path = LONG_HORIZON_DIR / f"{stem}.jsonl"
    manifest_path = LONG_HORIZON_DIR / f"{stem}.manifest.json"
    started_at = time.strftime("%Y-%m-%dT%H:%M:%S")

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
    manifest_written = False

    def _write_manifest(finalized: dict | None = None) -> None:
        """(Re)write the manifest. Called at seed-pick and in finally."""
        doc = {
            "stem": stem,
            "profile": profile_label,
            "scenario": scenario.name,
            "session_id": session_id,
            "started_at": started_at,
            "n_turns": n_turns,
            "player": "gemini" if use_gemini else "scripted",
            "seed": seed.__dict__ if seed else None,
            "phase_config": [
                {"name": p.name, "turn_range": list(p.turn_range)}
                for p in scenario.phases
            ],
            "seed_pick_after_turn": scenario.seed_pick_after_turn,
            "jsonl": str(jsonl_path),
        }
        if finalized:
            doc.update(finalized)
        try:
            manifest_path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
        except Exception as e:
            print(f"  {C.YELLOW}manifest_write_failed: {e}{C.RESET}")

    jsonl_f = open(jsonl_path, "w", encoding="utf-8")

    def _append_turn(record: dict) -> None:
        turn_records.append(record)
        try:
            jsonl_f.write(json.dumps(record, default=str) + "\n")
            jsonl_f.flush()
        except Exception:
            pass

    run_error: Optional[str] = None

    try:
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
                # Write the manifest as soon as the seed exists (crash-survivable).
                _write_manifest()
                manifest_written = True

            phase = _phase_for_turn(scenario, turn)
            player_error: Optional[str] = None
            fallback_action = False
            try:
                action = await player.next_action(last_response_text, phase, seed)
            except Exception as e:
                # FIX-4: a player-side failure (usually Gemini quota) means the
                # action this turn is a stand-in, not a real callback attempt.
                print(f"  {C.RED}player_error_turn_{turn}: {e}{C.RESET}")
                action = "I pause and look around."
                player_error = str(e)
                fallback_action = True

            print(f"\n{C.DIM}--- Turn {turn} [{phase.name}] ---{C.RESET}")
            print(f"  {C.CYAN}Player:{C.RESET} {action}")

            n0 = len(usage_recorder.events())
            start = time.time()
            orchestrator_error: Optional[str] = None
            try:
                response = await session.send_action(action)
            except Exception as e:
                elapsed = time.time() - start
                print(f"  {C.RED}orchestrator_error: {e}{C.RESET}")
                # FIX-6: keep narration_history aligned with turn numbers so the
                # seed-pick "Turn N" labels don't drift after an error turn.
                narration_history.append("")
                delta = usage_recorder.events()[n0:]
                utot, ubreak = _aggregate_events(delta)
                _append_turn({
                    "turn": turn, "phase": phase.name, "action": action,
                    "elapsed": elapsed, "narrative_chars": 0,
                    "player_error": player_error, "fallback_action": fallback_action,
                    "orchestrator_error": str(e),
                    **utot, "usage_breakdown": ubreak,
                })
                continue
            elapsed = time.time() - start

            last_response_text = (response.narrative or "") if response else ""
            narration_history.append(last_response_text)
            preview = last_response_text[:200].replace("\n", " ")
            print(f"  {C.GREEN}Narrator ({elapsed:.1f}s):{C.RESET} {preview}...")

            delta = usage_recorder.events()[n0:]
            utot, ubreak = _aggregate_events(delta)
            _append_turn({
                "turn": turn, "phase": phase.name, "action": action,
                "elapsed": elapsed, "narrative_chars": len(last_response_text),
                "player_error": player_error, "fallback_action": fallback_action,
                "orchestrator_error": None,
                **utot, "usage_breakdown": ubreak,
            })
    except Exception as e:
        # Unexpected loop-level failure — still finalize below.
        run_error = f"{type(e).__name__}: {e}"
        print(f"  {C.RED}run_error: {run_error}{C.RESET}")
    finally:
        # FIX-3: always clean up, finalize artifacts, and run assertions over
        # whatever turns landed — even on a partial/crashed run.
        try:
            await session.cleanup()
        except Exception:
            pass
        try:
            jsonl_f.close()
        except Exception:
            pass

        events_all = usage_recorder.events()

        # Verdict (FIX-2 / FIX-4).
        callback_phase = next((p for p in scenario.phases if p.name == "callback"), None)
        cb_lo, cb_hi = callback_phase.turn_range if callback_phase else (0, 0)
        callback_fallback = any(
            r.get("fallback_action") and cb_lo <= r.get("turn", -1) <= cb_hi
            for r in turn_records
        )
        verdict_trusted = not _seed_is_fallback(seed)

        results: list[AssertionResult] = []
        if seed is not None and session_id:
            try:
                results = run_assertions(scenario, session_id, seed)
                render_results(scenario, seed, results, verdict_trusted)
            except Exception as e:
                print(f"  {C.YELLOW}assertion_error: {e}{C.RESET}")
        else:
            print(f"\n  {C.YELLOW}Assertions skipped "
                  f"(seed={'yes' if seed else 'no'}, session_id={'yes' if session_id else 'no'}){C.RESET}")

        passed = sum(1 for r in results if r.passed)
        total = len(results)
        if callback_fallback:
            verdict = "INVALID(player-quota)"
        elif not verdict_trusted:
            verdict = "UNTRUSTED"
        elif total and passed == total:
            verdict = "PASS"
        else:
            verdict = "FAIL"

        report = build_report(events_all, turn_records, session_id, profile_label, scenario, n_turns)
        render_report(report, verdict, seed)

        # Ensure a manifest exists even if we crashed before seed pick.
        if not manifest_written:
            _write_manifest()
        _write_manifest(finalized={
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_error": run_error,
            "verdict": verdict,
            "verdict_trusted": verdict_trusted,
            "callback_player_quota": callback_fallback,
            "assertions": [r.__dict__ for r in results],
            "assertions_passed": passed,
            "assertions_total": total,
            "report": report,
        })
        print(f"\n  {C.DIM}Artifacts:{C.RESET}")
        print(f"    {C.DIM}{jsonl_path}{C.RESET}")
        print(f"    {C.DIM}{manifest_path}{C.RESET}")

    return {
        "scenario": scenario.name,
        "session_id": session_id,
        "seed": seed.__dict__ if seed else None,
        "turn_records": turn_records,
        "total_turns": n_turns,
        "verdict": verdict,
        "verdict_trusted": verdict_trusted,
        "assertions": [r.__dict__ for r in results],
        "manifest": str(manifest_path),
        "jsonl": str(jsonl_path),
        "report": report,
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

    # FIX-2: match on the stopword-stripped name OR the longest content token.
    seed_clean, seed_token = _clean_seed(seed)

    def _matches(text: str) -> bool:
        return _seed_matches(text, seed_clean, seed_token)

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
        _matches(log.narrator_response(t).text) for t in explore_turns
    )
    results.append(AssertionResult(
        name="seed_appears_in_explore",
        passed=appeared_in_explore,
        description=f"Seed '{seed.name}' appeared in narrator prose during explore phase.",
        detail=f"Searched turns {explore_turns}",
    ))

    # 2. Player references seed at the callback (they're instructed to)
    player_referenced_at_callback = any(
        _matches(log.player_action(t)) for t in callback_turns
    )
    results.append(AssertionResult(
        name="player_references_seed_at_callback",
        passed=player_referenced_at_callback,
        description=f"Player's input mentioned '{seed.name}' during callback phase.",
        detail=f"Searched turns {callback_turns}",
    ))

    # 3. Narrator's response at callback acknowledges the seed
    narrator_referenced_at_callback = any(
        _matches(log.narrator_response(t).text) for t in callback_turns
    )
    results.append(AssertionResult(
        name="narrator_references_seed_at_callback",
        passed=narrator_referenced_at_callback,
        description=f"Narrator's prose at callback referenced '{seed.name}'.",
        detail=f"Searched turns {callback_turns}",
    ))

    # 4. KG context at callback contained the seed (memory retrieval worked)
    kg_surfaced = any(
        log.kg_context_for(t).mentions(seed_clean)
        or (seed_token and log.kg_context_for(t).mentions(seed_token))
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
        ws_yaml = log.world_state_after(last_filler).raw_yaml
        retained_in_world_state = _matches(ws_yaml)
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
            if _matches(json.dumps(e)):
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


def render_results(scenario: Scenario, seed: Seed, results: list[AssertionResult],
                   verdict_trusted: bool = True):
    header(f"ASSERTIONS — {scenario.name}")
    print(f"  Seed       : {C.MAGENTA}{seed.type} '{seed.name}'{C.RESET}")
    print(f"  Reason     : {C.DIM}{seed.reason}{C.RESET}")
    if not verdict_trusted:
        print(f"  {C.YELLOW}{C.BOLD}UNTRUSTED — seed is a framework fallback; "
              f"green results below are NOT meaningful.{C.RESET}")
    print()

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
    if not verdict_trusted:
        print(f"\n  {C.YELLOW}{C.BOLD}{passed}/{total} assertions passed "
              f"(UNTRUSTED — fallback seed){C.RESET}")
    else:
        color = C.GREEN if passed == total else (C.YELLOW if passed >= total // 2 else C.RED)
        print(f"\n  {color}{C.BOLD}{passed}/{total} assertions passed{C.RESET}")


# ── End-of-run usage / latency / cost report ───────────────────────────


def _stage_latencies(session_id: Optional[str]) -> dict[str, list[float]]:
    """Per-stage (triage/narrate/state_extract) elapsed_ms read post-hoc from
    the per-session turn log via TurnLogReader (orchestrator timings)."""
    out: dict[str, list[float]] = {"triage": [], "narrate": [], "state_extract": []}
    if not session_id:
        return out
    try:
        from dnd_bot.llm.turn_log_reader import TurnLogReader
        log = TurnLogReader.load(session_id)
    except Exception:
        return out
    for rec in log.records:
        timings = rec.get("timings") or {}
        for stage in out:
            v = timings.get(stage)
            if isinstance(v, (int, float)):
                out[stage].append(float(v))
    return out


def _role_sets() -> tuple[set, set]:
    """(narrator (provider,model) set, brain (provider,model) set) for the
    active profile — used to bucket events into the §8 Narrator/Brain columns."""
    narrator: set = set()
    brain: set = set()
    try:
        from dnd_bot.config import get_profile
        prof = get_profile()
        for cfg in (getattr(prof, "narrator", None),
                    getattr(prof, "narrator_premium", None),
                    getattr(prof, "narrator_opening", None)):
            if cfg is not None:
                narrator.add((cfg.provider, cfg.model))
        b = getattr(prof, "brain", None)
        if b is not None:
            brain.add((b.provider, b.model))
    except Exception:
        pass
    return narrator, brain


def _role_of(ev, narrator_set: set, brain_set: set) -> str:
    if ev.stage == "player":
        return "player"
    key = (ev.provider, ev.model)
    if key in brain_set and key not in narrator_set:
        return "brain"
    if key in narrator_set and key not in brain_set:
        return "narrator"
    if key in narrator_set:  # overlap (same model both roles) -> narrator
        return "narrator"
    if ev.provider == "gemini":
        return "player"
    return "narrator"  # unknown non-player defaults to narrator bucket


def build_report(events, turn_records: list[dict], session_id: Optional[str],
                 profile_label: str, scenario: Scenario, n_turns: int) -> dict:
    """Assemble the machine-readable end-of-run roll-up (also stored in the
    manifest). Degrades to zeros/None when usage is absent."""
    narrator_set, brain_set = _role_sets()

    # Per (provider, model, stage) rows.
    rows: dict[tuple[str, str, str], dict] = {}
    for ev in events:
        key = (ev.provider, ev.model, ev.stage)
        row = rows.setdefault(key, {
            "provider": ev.provider, "model": ev.model, "stage": ev.stage,
            "calls": 0, "in": 0, "out": 0, "cache_read": 0, "cache_write": 0,
            "denom": 0, "cost": 0.0, "cost_known": True,
        })
        row["calls"] += 1
        row["in"] += ev.prompt_tokens
        row["out"] += ev.completion_tokens
        row["cache_read"] += ev.cache_read_tokens
        row["cache_write"] += ev.cache_write_tokens
        row["denom"] += _event_prompt_denom(ev)
        c = _event_cost(ev)
        if c is None:
            row["cost_known"] = False
        else:
            row["cost"] += c

    unpriced: set[tuple[str, str]] = set()
    row_list = []
    for key, row in sorted(rows.items()):
        hit = (row["cache_read"] / row["denom"]) if row["denom"] else 0.0
        if not row["cost_known"]:
            unpriced.add((row["provider"], row["model"]))
        row_list.append({
            **{k: row[k] for k in ("provider", "model", "stage", "calls", "in", "out", "cache_read", "cache_write")},
            "cache_hit_pct": round(hit * 100, 1),
            "cost_usd": round(row["cost"], 6) if row["cost_known"] else None,
        })

    # Role roll-up (Narrator/Brain/Player) for the §8 single row.
    roles: dict[str, dict] = {r: {"in": 0, "out": 0, "cache_read": 0, "cache_write": 0, "denom": 0}
                              for r in ("narrator", "brain", "player")}
    for ev in events:
        r = _role_of(ev, narrator_set, brain_set)
        b = roles[r]
        b["in"] += ev.prompt_tokens
        b["out"] += ev.completion_tokens
        b["cache_read"] += ev.cache_read_tokens
        b["cache_write"] += ev.cache_write_tokens
        b["denom"] += _event_prompt_denom(ev)

    def _role_hit(b) -> Optional[float]:
        return round(b["cache_read"] / b["denom"] * 100, 1) if b["denom"] else None

    total_cost = sum((r["cost_usd"] or 0.0) for r in row_list)
    cost_complete = all(r["cost_usd"] is not None for r in row_list)

    # Latency: whole-turn (harness) + per stage (turn log).
    turn_elapsed = [r.get("elapsed") for r in turn_records
                    if isinstance(r.get("elapsed"), (int, float))]
    turn_pct = _percentiles([e * 1000.0 for e in turn_elapsed])  # ms
    stage_lat = _stage_latencies(session_id)
    stage_pct = {s: _percentiles(v) for s, v in stage_lat.items()}

    return {
        "profile": profile_label,
        "turns": n_turns,
        "rows": row_list,
        "roles": {
            r: {
                "in": roles[r]["in"], "out": roles[r]["out"],
                "cache_read": roles[r]["cache_read"], "cache_write": roles[r]["cache_write"],
                "cache_hit_pct": _role_hit(roles[r]),
            } for r in roles
        },
        "total_cost_usd": round(total_cost, 6),
        "cost_complete": cost_complete,
        "unpriced_models": sorted(f"{p}/{m}" for p, m in unpriced),
        "latency_turn_ms": {"p50": turn_pct[0.50], "p95": turn_pct[0.95]},
        "latency_stage_ms": {
            s: {"p50": stage_pct[s][0.50], "p95": stage_pct[s][0.95]} for s in stage_pct
        },
    }


def _fmt_ms(v: Optional[float]) -> str:
    return f"{v/1000:.1f}s" if isinstance(v, (int, float)) else "—"


def _fmt_tok(row: dict, key: str) -> str:
    return f"{row.get(key, 0):,}"


def render_report(report: dict, verdict: str, seed: Optional[Seed]):
    header("END-OF-RUN REPORT")

    vcolor = (C.GREEN if verdict == "PASS"
              else C.YELLOW if verdict in ("UNTRUSTED", "INVALID(player-quota)")
              else C.RED)
    print(f"  Verdict    : {vcolor}{C.BOLD}{verdict}{C.RESET}")
    print(f"  Profile    : {report['profile']}    Turns: {report['turns']}")
    if seed:
        print(f"  Seed       : {C.MAGENTA}{seed.type} '{seed.name}'{C.RESET}")

    rows = report["rows"]
    if not rows:
        print(f"\n  {C.YELLOW}No LLM usage recorded (offline / no keys) — "
              f"report degrades to zero.{C.RESET}")
    else:
        print(f"\n  {C.BOLD}Per (provider, model, stage):{C.RESET}")
        print(f"    {'provider/model/stage':<44} {'calls':>5} {'in':>9} {'out':>8} "
              f"{'cread':>8} {'hit%':>6} {'cost$':>10}")
        for r in rows:
            pms = f"{r['provider']}/{r['model']}/{r['stage'] or '-'}"
            cost = f"{r['cost_usd']:.5f}" if r["cost_usd"] is not None else "excl."
            print(f"    {pms[:44]:<44} {r['calls']:>5} {_fmt_tok(r,'in'):>9} "
                  f"{_fmt_tok(r,'out'):>8} {_fmt_tok(r,'cache_read'):>8} "
                  f"{r['cache_hit_pct']:>6} {cost:>10}")

    if report["unpriced_models"]:
        print(f"\n  {C.YELLOW}Unpriced (cost excluded): "
              f"{', '.join(report['unpriced_models'])}{C.RESET}")

    tc = report["total_cost_usd"]
    suffix = "" if report["cost_complete"] else f" (excl. {', '.join(report['unpriced_models'])})"
    print(f"\n  {C.BOLD}Grand total cost:{C.RESET} ${tc:.5f}{suffix}")

    lt = report["latency_turn_ms"]
    print(f"\n  {C.BOLD}Latency (p50/p95):{C.RESET}")
    print(f"    whole-turn   : {_fmt_ms(lt['p50'])} / {_fmt_ms(lt['p95'])}")
    for stage, pc in report["latency_stage_ms"].items():
        print(f"    {stage:<13}: {_fmt_ms(pc['p50'])} / {_fmt_ms(pc['p95'])}")

    # §8 "Measured results" paste row (PROMPT_CACHING_2026_07.md column order):
    # Profile | Turns | Narrator in/out | Narrator cache-hit % | Brain in/out |
    # Brain cache-hit % | Player in/out | Est cost | p50/p95 turn | p50/p95 narrate | Notes
    roles = report["roles"]

    def io(r):
        return f"{roles[r]['in']}/{roles[r]['out']}"

    def hp(r):
        v = roles[r]["cache_hit_pct"]
        return f"{v}%" if v is not None else "—"

    narrate = report["latency_stage_ms"].get("narrate", {})
    cost_cell = f"${tc:.4f}" + ("" if report["cost_complete"] else "*")
    notes = "offline/no-usage" if not rows else ("cost-excl-some" if not report["cost_complete"] else "")
    paste = " | ".join([
        report["profile"], str(report["turns"]),
        io("narrator"), hp("narrator"),
        io("brain"), hp("brain"),
        io("player"),
        cost_cell,
        f"{_fmt_ms(lt['p50'])}/{_fmt_ms(lt['p95'])}",
        f"{_fmt_ms(narrate.get('p50'))}/{_fmt_ms(narrate.get('p95'))}",
        notes,
    ])
    print(f"\n  {C.BOLD}§8 paste row:{C.RESET}")
    print(f"    | {paste} |")


# ── --assert-only reload path ──────────────────────────────────────────


def run_assert_only(manifest_path_str: str) -> int:
    """Reload a manifest, re-run assertions against its turn log, and rewrite
    the manifest with fresh results. No new session. Returns a process exit
    code (0 ok, non-zero on a usage error)."""
    manifest_path = Path(manifest_path_str)
    if not manifest_path.exists():
        print(f"  {C.RED}--assert-only: manifest not found: {manifest_path}{C.RESET}")
        return 2
    try:
        doc = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as e:
        print(f"  {C.RED}--assert-only: could not read manifest: {e}{C.RESET}")
        return 2

    scenario_name = doc.get("scenario", "emergent_callback")
    scenario = SCENARIOS.get(scenario_name)
    if scenario is None:
        print(f"  {C.RED}--assert-only: unknown scenario '{scenario_name}'{C.RESET}")
        return 2

    session_id = doc.get("session_id")
    seed_data = doc.get("seed")
    if not session_id:
        print(f"  {C.RED}--assert-only: manifest has no session_id{C.RESET}")
        return 2
    if not seed_data:
        print(f"  {C.RED}--assert-only: manifest has no seed — cannot run assertions{C.RESET}")
        return 2

    seed = Seed(**seed_data)
    verdict_trusted = not _seed_is_fallback(seed)

    print(f"  Reloaded manifest: {manifest_path}")
    print(f"  Session ID  : {session_id}")
    results = run_assertions(scenario, str(session_id), seed)
    render_results(scenario, seed, results, verdict_trusted)

    passed = sum(1 for r in results if r.passed)
    total = len(results)
    verdict = ("UNTRUSTED" if not verdict_trusted
               else "PASS" if total and passed == total else "FAIL")
    doc.update({
        "reasserted_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "verdict": verdict,
        "verdict_trusted": verdict_trusted,
        "assertions": [r.__dict__ for r in results],
        "assertions_passed": passed,
        "assertions_total": total,
    })
    try:
        manifest_path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
        print(f"\n  {C.DIM}Rewrote {manifest_path}{C.RESET}")
    except OSError as e:
        print(f"  {C.YELLOW}--assert-only: manifest rewrite failed: {e}{C.RESET}")
    return 0


# ── Main ───────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="emergent_callback",
                        choices=list(SCENARIOS.keys()))
    parser.add_argument("--profile", help="Profile to use (overrides ACTIVE_PROFILE)")
    parser.add_argument("--turns", type=int, help="Override total turn count")
    parser.add_argument("--scripted", action="store_true",
                        help="Use scripted actions instead of Gemini Flash")
    parser.add_argument("--assert-only", dest="assert_only", metavar="MANIFEST",
                        help="Reload a manifest and re-run assertions only (no new session)")
    args = parser.parse_args()

    if args.assert_only:
        sys.exit(run_assert_only(args.assert_only))

    scenario = SCENARIOS[args.scenario]

    # FIX-5: a --turns override below the callback phase's end would run a
    # scenario whose callback assertions can never fire. Refuse it up front.
    if args.turns is not None:
        callback_phase = next((p for p in scenario.phases if p.name == "callback"), None)
        min_turns = callback_phase.turn_range[1] if callback_phase else scenario.total_turns
        if args.turns < min_turns:
            print(f"{C.RED}--turns {args.turns} is below the callback phase end "
                  f"({min_turns}). The callback assertions could never fire. "
                  f"Use --turns >= {min_turns}.{C.RESET}")
            sys.exit(2)

    result = await run_long_horizon(
        scenario=scenario,
        use_gemini=not args.scripted,
        profile=args.profile,
        turn_override=args.turns,
    )

    if not result.get("session_id"):
        print(f"\n  {C.RED}No session_id captured — assertions could not run{C.RESET}")
    if not result.get("seed"):
        print(f"\n  {C.RED}No seed identified — assertions could not run{C.RESET}")


if __name__ == "__main__":
    asyncio.run(main())
