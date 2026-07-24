"""Long-horizon memory + KG test (emergent-callback design).

Drives the production orchestrator through N turns of a Gemini Flash-Lite
player. Blank-slate scenarios let the narrator open freely; seeded scenarios
provide a world premise without pre-populating the graph or domain database.
After a few turns, the framework asks Gemini to identify ONE
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
    python test_long_horizon.py --scenario deep_seeded_callback --profile deepseek_v4_flash_qwen9b
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
import re
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

if sys.platform == "win32":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")

os.chdir(Path(__file__).parent)
from dotenv import load_dotenv  # noqa: E402
load_dotenv()

from dnd_bot.game.identity import (  # noqa: E402
    is_generic_npc_label,
    locations_equivalent,
)
from dnd_bot.llm.continuity import NarrativeGovernance  # noqa: E402


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
# "altar", so we match on the cleaned name and meaningful component tokens.
SEED_STOPWORDS = {"the", "a", "an", "of", "to", "in"}

# When the explore act ends with no eligible graph-backed seed (an NPC-less,
# object-focused story — a legal narrator outcome, observed 4x on 2026-07-23),
# extend explore with forced name-eliciting turns instead of aborting. Kept
# small so a late pick still leaves cool-off room before memory_silence starts.
SEED_PICK_MAX_RETRY_TURNS = 3

# Degenerate seeds the framework falls back to when Gemini's seed-pick call
# fails or returns garbage (see GeminiFlashPlayer.pick_seed / the seed-pick
# except handler). A run seeded with one of these can't be trusted to prove
# recall, so its verdict is stamped UNTRUSTED rather than green (FIX-2).
FALLBACK_SEED_NAMES = {"the scene", "scene", ""}

# Common narrator/tool labels are not stable identities and make substring
# leak checks noisy. A candidate containing only these words is descriptive,
# even when a tool call title-cases it (for example ``Beggar``).
GENERIC_NPC_SEED_TERMS = {
    "beggar", "captain", "child", "collapsed", "courier", "dockworker",
    "figure", "guard", "injured", "keeper", "man", "masked", "merchant",
    "observer", "priest", "priestess", "stranger", "unidentified", "unknown",
    "unseen", "vendor", "voice", "woman", "worker",
}

# A callback place needs a distinctive identity, not a generic scene category.
# Generic labels recur naturally and turn the washout into a substring-leak
# test (the current 80-turn soak exposed ``depot`` as exactly this failure).
GENERIC_PLACE_SEED_TERMS = {
    "alley", "area", "building", "corridor", "depot", "district", "door",
    "gate", "hall", "house", "inn", "market", "path", "road", "room",
    "sector", "shop", "square", "street", "tavern", "temple", "tower",
}

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
    # Gemini player models (+ pro), verified 2026-07-21. Keep Flash-Lite
    # before Flash: pricing lookup uses the longest matching model prefix.
    ("gemini", "gemini-2.5-flash-lite"): {"in": 0.10, "cached_in": 0.01, "out": 0.40},
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


DEFAULT_PLAYER_PERSONA = (
    "You are Kael Windrunner, an elven ranger with dry humor, practical "
    "courage, and a weakness for impossible maps. You protect people before "
    "institutions, distrust easy prophecies, and sometimes commit too quickly "
    "when someone vulnerable is being treated as expendable. Let choices "
    "reveal these traits; do not merely describe them."
)


@dataclass
class Scenario:
    name: str
    description: str
    base_goal: str               # initial goal text (no seed yet)
    phases: list[Phase]
    total_turns: int
    seed_pick_after_turn: int    # framework picks seed after this turn
    seed_pick_prompt: str        # framework's prompt to Gemini for seed selection
    world_setting: Optional[str] = None
    opening_situation: str = ""
    player_persona: str = DEFAULT_PLAYER_PERSONA
    memory_silence_range: Optional[tuple[int, int]] = None
    creativity_gate: bool = False
    tool_coverage_gate: bool = False
    seed_exclusions: tuple[str, ...] = ()
    seed_description_exclusions: tuple[str, ...] = ()
    required_seed_type: Optional[str] = None


@dataclass
class AssertionResult:
    name: str
    passed: bool
    description: str
    detail: str = ""


def _determine_verdict(
    *,
    run_error: Optional[str],
    run_complete: bool,
    orchestrator_failures: list[int],
    combat_policy_failed: bool,
    fallback_turns: list[int],
    verdict_trusted: bool,
    passed: int,
    total: int,
) -> str:
    """Return the gate verdict; only a complete, trusted pass is green."""
    if run_error or not run_complete or orchestrator_failures or combat_policy_failed:
        return "FAIL"
    if fallback_turns:
        return "INVALID(player-error)"
    if not verdict_trusted:
        return "UNTRUSTED"
    if total and passed == total:
        return "PASS"
    return "FAIL"


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

# The promotion/soak trajectory uses the same emergent-seed contract but puts
# more than sixty turns between establishment and callback. The middle phase
# explicitly invites concrete, tool-worthy state changes while keeping the
# chosen seed out of recency context.
SCENARIOS["deep_emergent_callback"] = Scenario(
    name="deep_emergent_callback",
    description=(
        "An 80-turn narrative, tool, and memory soak. Establish a concrete "
        "seed in the opening, sustain a separate multi-step story for more "
        "than sixty turns, then return to the seed and its consequences."
    ),
    base_goal=(
        "Play a curious, decisive D&D adventurer through a coherent long-form "
        "story. Build relationships, travel, make consequential choices, and "
        "interact with concrete people and objects. Avoid initiating combat. "
        "Much later you will deliberately return to one early detail."
    ),
    phases=[
        Phase(
            name="explore",
            turn_range=(1, 8),
            instruction=(
                "Explore the opening naturally and establish several concrete "
                "NPCs, objects, or places. Ask follow-up questions and make "
                "choices that give the scene specific, memorable details."
            ),
        ),
        Phase(
            name="filler",
            turn_range=(9, 70),
            instruction=(
                "Pursue a coherent but unrelated multi-step adventure. Change "
                "locations, deepen NPC relationships, make promises and "
                "discoveries, and acquire or transfer concrete objects when "
                "the fiction supports it. Invite real state/tool changes, not "
                "a checklist. Do NOT mention or return to the {seed_type} "
                "called \"{seed_name}\" anywhere in this phase. Avoid combat."
            ),
        ),
        Phase(
            name="callback",
            turn_range=(71, 80),
            instruction=(
                "Return to the early {seed_type} \"{seed_name}\" and SAY ITS "
                "NAME. Investigate what changed, connect it to the intervening "
                "story, and pursue the consequences for several turns rather "
                "than making a single passing reference."
            ),
        ),
    ],
    total_turns=80,
    seed_pick_after_turn=8,
    seed_pick_prompt=SCENARIOS["emergent_callback"].seed_pick_prompt,
)


# A high-potential creative trajectory. The static premise gives the narrator
# strong toys, factions, and pressure, but the callback seed must still be a
# NEW element invented during play. That keeps KG/DB accumulation a blank-slate
# test rather than letting the always-on world-setting block answer the recall.
_GLASSWAKE_WORLD = (
    "Veyr is a vertical city suspended above an endless lightning storm by "
    "seven colossal chains threaded through the fossil ribs of a dead sky-"
    "leviathan. Gravity tides make rain fall upward and open temporary roads "
    "across walls and ceilings. Once a year, during the Night of Unburdening, "
    "each household surrenders one memory to the Glass Archive; the city "
    "claims those memories keep the chains from failing. Memories are also "
    "contraband, medicine, evidence, and currency. Three public factions pull "
    "at the city: the Choir of Anchors protects the ritual, Ragpicker smugglers "
    "return stolen memories to their owners, and the Cartographers of the "
    "Unfallen map impossible gravity roads. Tonight upward rain whispers other "
    "people's secrets, the Archive's sealed inner door has opened from the "
    "inside, and a masked courier has collapsed at Kael Windrunner's feet at "
    "Saint Orra's Wake. The courier carries a living brass compass and a sealed "
    "letter in Kael's own handwriting dated thirty years from now: 'Before "
    "dawn, choose what the city must forget.' There is no predetermined "
    "culprit or correct choice. Invent specific NPC agendas, costs, locations, "
    "objects, and reversals that grow naturally from player decisions."
)

_GLASSWAKE_SEED_PROMPT = (
    "You are the player in a D&D session. From the first eight narrated turns, "
    "choose ONE discrete, concrete element invented by the DM during play that "
    "can plausibly stay off-screen for more than fifty turns and matter when it "
    "returns. It must have an exact name. Do NOT choose any element supplied by "
    "the fixed campaign premise: Veyr, the Night of Unburdening, Glass Archive, "
    "Saint Orra's Wake, the masked courier, living brass compass, future letter, "
    "Choir of Anchors, Ragpickers, Cartographers of the Unfallen, sky-leviathan, "
    "or the chains. Prefer a newly invented NPC, object, or localized place with "
    "an unresolved desire, danger, promise, or contradiction.\n\n"
    "Output ONLY one JSON object on one line, no prose:\n"
    '{"type": "npc"|"item"|"place", "name": "<exact name>", "reason": "<unresolved dramatic potential>"}\n\n'
    "Recent narration:\n"
)

SCENARIOS["deep_seeded_callback"] = Scenario(
    name="deep_seeded_callback",
    description=(
        "An 80-turn creative-continuity, tool, and memory soak in Veyr. A rich "
        "static premise drives play, while the long callback targets a separate "
        "element invented after the graph and database started empty."
    ),
    base_goal=(
        "Play Kael as a protagonist with changing loyalties, personal stakes, "
        "and a point of view. Follow the mystery of Veyr's stolen memories, but "
        "choose people and costly commitments over passive clue collection. "
        "Build a coherent dramatic arc without initiating combat. Much later, "
        "return to one emergent detail from the opening and make it consequential."
    ),
    phases=[
        Phase(
            name="explore",
            turn_range=(1, 8),
            instruction=(
                "Respond to the crisis at Saint Orra's Wake. Make decisive use "
                "of concrete objects and surroundings, speak to people with a "
                "specific agenda, and follow at least one uncomfortable lead. "
                "Create opportunities for the DM to establish new named NPCs, "
                "objects, and localized places beyond the fixed premise."
            ),
        ),
        Phase(
            name="breakaway",
            turn_range=(9, 14),
            instruction=(
                "Choose a lead that decisively carries you away from the early "
                "{seed_type} called {seed_name}. Do not name it in your action. "
                "Accept a cost, obligation, or ally that can drive the next act."
            ),
        ),
        Phase(
            name="mirror_market",
            turn_range=(15, 28),
            instruction=(
                "Pursue the memory trade through a vivid new district. Bargain, "
                "confide, deceive, investigate, acquire or give away a concrete "
                "item, and let an NPC relationship change. Do not mention the "
                "sealed callback {seed_type} {seed_name}; it is off-screen."
            ),
        ),
        Phase(
            name="leviathan_descent",
            turn_range=(29, 44),
            instruction=(
                "Follow a consequence into the sky-leviathan's forbidden civic "
                "underworks. Solve obstacles through choices, tools, skills, and "
                "relationships rather than starting combat. Discover something "
                "that changes your theory. Do not mention {seed_name}."
            ),
        ),
        Phase(
            name="divided_alliance",
            turn_range=(45, 58),
            instruction=(
                "Take a side provisionally, make or break a promise, and force "
                "two NPC agendas into direct tension. Use established possessions "
                "and places so state changes are earned by the fiction. Do not "
                "mention the off-screen {seed_type} {seed_name}."
            ),
        ),
        Phase(
            name="price_of_memory",
            turn_range=(59, 70),
            instruction=(
                "Drive toward a costly decision about what Veyr remembers. Let "
                "earlier bargains and relationships produce consequences; change "
                "your mind if events justify it. Do not mention {seed_name}. End "
                "this act with a reason to revisit unfinished opening business."
            ),
        ),
        Phase(
            name="callback",
            turn_range=(71, 80),
            instruction=(
                "Return to the emergent {seed_type} {seed_name} and say its exact "
                "name. Pursue its unresolved dramatic potential for several turns, "
                "connect it to choices made in Veyr, and let the callback change "
                "the ending rather than serving as a cameo."
            ),
        ),
    ],
    total_turns=80,
    seed_pick_after_turn=8,
    seed_pick_prompt=_GLASSWAKE_SEED_PROMPT,
    world_setting=_GLASSWAKE_WORLD,
    opening_situation=(
        "The Night of Unburdening has gone wrong at Saint Orra's Wake: upward "
        "rain is whispering stolen secrets, a masked courier has collapsed at "
        "your feet, and a future letter demands that you choose what Veyr forgets."
    ),
    player_persona=(
        "You are Kael Windrunner, an elven ranger who once guided refugees over "
        "a gravity road officials insisted did not exist. You have dry humor, "
        "hate systems that turn people into acceptable losses, and cannot resist "
        "an impossible map. You want the truth, but secretly fear that one of "
        "your defining memories was purchased rather than lived. Form opinions, "
        "attachments, suspicions, and regrets—and allow them to evolve."
    ),
    memory_silence_range=(15, 70),
    creativity_gate=True,
    tool_coverage_gate=True,
    seed_exclusions=(
        "Veyr",
        "Night of Unburdening",
        "Glass Archive",
        "Saint Orra's Wake",
        "masked courier",
        "living brass compass",
        "future letter",
        "Choir of Anchors",
        "Ragpickers",
        "Cartographers of the Unfallen",
        "sky-leviathan",
        "chains",
        "Kael Windrunner",
    ),
    # The opening courier is premise-provided even if the narrator assigns a
    # proper name during play. Do not let that rename masquerade as a genuinely
    # emergent callback entity.
    seed_description_exclusions=("masked courier",),
    required_seed_type="npc",
)


# Cheap promotion gate for retrieval changes. It preserves the creative seeded
# premise and a meaningful washout while compressing the 80-turn soak into a
# run that can be used before spending on the full suite.
_DEEP_SEEDED = SCENARIOS["deep_seeded_callback"]
SCENARIOS["targeted_relevance_callback"] = Scenario(
    name="targeted_relevance_callback",
    description=(
        "A 30-turn relevance, creativity, and tool-reliability gate in Veyr: "
        "establish an emergent detail, redact it from the player for sixteen "
        "turns, enforce strict retrieval silence after buffer cool-off, then "
        "make it consequential in a six-turn callback."
    ),
    base_goal=_DEEP_SEEDED.base_goal,
    phases=[
        Phase(
            name="explore",
            turn_range=(1, 8),
            instruction=(
                "Respond decisively to the crisis at Saint Orra's Wake. Use "
                "concrete objects and surroundings, speak to people with "
                "specific agendas, and create openings for the DM to invent "
                "new named NPCs, objects, and localized places beyond the "
                "fixed premise. Before this phase ends, learn and repeat the "
                "exact name of at least one such new element and interact with "
                "it; if needed, directly ask a new person to identify themselves. "
                "Avoid initiating combat."
            ),
        ),
        Phase(
            name="washout",
            turn_range=(9, 24),
            instruction=(
                "Commit to a different lead and carry it through a coherent "
                "middle act in new districts. Bargain, investigate, travel, "
                "change an NPC relationship, and acquire or surrender a "
                "concrete item when earned by the fiction. Do not mention or "
                "return to the sealed callback detail; it is fully off-screen. "
                "Avoid combat and generic stalling."
            ),
        ),
        Phase(
            name="callback",
            turn_range=(25, 30),
            instruction=(
                "Return to the emergent {seed_type} {seed_name} and say its exact "
                "name at least once. Pursue its unresolved dramatic potential, "
                "connect it to choices made during the middle act, and let it "
                "change your immediate goal rather than serving as a cameo. "
                "Treat the DM's response as authoritative: if the target is "
                "absent or another person denies being them, do not relabel that "
                "person or keep repeating the false assumption; seek a lead to "
                "the real target and continue naturally."
            ),
        ),
    ],
    total_turns=30,
    seed_pick_after_turn=8,
    seed_pick_prompt=_GLASSWAKE_SEED_PROMPT.replace(
        "more than fifty turns", "more than fifteen turns"
    ),
    world_setting=_DEEP_SEEDED.world_setting,
    opening_situation=_DEEP_SEEDED.opening_situation,
    player_persona=_DEEP_SEEDED.player_persona,
    # Turns 9-14 are a transition/cool-off period while the final explore
    # exchange drains from the five-turn verbatim window. Strict silence starts
    # only once a callback would require compacted/durable memory.
    memory_silence_range=(15, 24),
    creativity_gate=True,
    tool_coverage_gate=True,
    seed_exclusions=_DEEP_SEEDED.seed_exclusions,
    seed_description_exclusions=_DEEP_SEEDED.seed_description_exclusions,
    # Carried items and the current location remain in ambient WorldState and
    # would confound this targeted off-screen relevance test.
    required_seed_type="npc",
)


# ── Player abstractions ────────────────────────────────────────────────


class ScriptedPlayer:
    """Plays a fixed list of actions, one per turn. Useful for
    framework debugging without burning Gemini quota."""

    def __init__(self, actions: list[str]):
        self.actions = actions
        self.turn = 0

    async def next_action(
        self,
        narrator_response: str,
        phase: Phase,
        seed: Optional[Seed],
        forced_action: Optional[str] = None,
    ) -> str:
        if forced_action:
            self.turn += 1
            return forced_action
        if self.turn >= len(self.actions):
            return "I look around and consider my next move."
        action = self.actions[self.turn]
        self.turn += 1
        return action

    async def pick_seed(
        self,
        narration_history: list[str],
        scenario: Scenario,
        seed_candidates: Optional[list[dict[str, str]]] = None,
    ) -> Seed:
        return Seed(type="item", name="lantern", reason="(scripted fallback)", chosen_after_turn=scenario.seed_pick_after_turn)


_PLAYER_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "action": {"type": "string"},
        "continuity": {"type": "string"},
    },
    "required": ["action", "continuity"],
    "additionalProperties": False,
}

_SEED_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "type": {"type": "string", "enum": ["npc", "item", "place"]},
        "name": {"type": "string"},
        "reason": {"type": "string"},
    },
    "required": ["type", "name", "reason"],
    "additionalProperties": False,
}

_GENERIC_STALL_RE = re.compile(
    r"\b(?:look around|take stock|consider my next move|continue exploring|"
    r"continue looking|wait for (?:an?|the) (?:answer|response)|press on|"
    r"stand ready|do nothing)\b",
    re.IGNORECASE,
)


def _main_action_verb(action: str) -> str:
    """Best-effort leading verb for cheap diversity diagnostics."""
    words = re.findall(r"[A-Za-z']+", action.lower())
    if not words or words[0] != "i":
        return ""
    skip = {"am", "will", "would", "want", "decide", "carefully", "quietly", "slowly"}
    for word in words[1:]:
        if word in skip or word.endswith("ly"):
            continue
        return word
    return ""


def _narration_excerpt(text: str, limit: int) -> str:
    """Keep scene setup and the actionable ending of long DM responses."""
    text = (text or "").strip()
    if len(text) <= limit:
        return text
    head = max(1, limit // 3)
    tail = max(1, limit - head - 7)
    return f"{text[:head]}\n[…]\n{text[-tail:]}"


def _redact_seed_text(text: str, seed: Optional[Seed]) -> str:
    """Hide a callback seed from the simulated player during washout."""
    if not text or seed is None:
        return text
    cleaned, token = _clean_seed(seed)
    meaningful_parts = [
        part
        for part in re.findall(r"[A-Za-z0-9']+", cleaned)
        if len(part) >= 4 and part.lower() not in SEED_STOPWORDS
    ]
    # For people, a surname is not a safe alias: distinct relatives or merely
    # unrelated NPCs can share it. Full name + first identity token preserve
    # recall sensitivity without redacting every Venn as if they were Lira.
    component_tokens = (
        {meaningful_parts[0]} if seed.type == "npc" and meaningful_parts
        else set(meaningful_parts)
    )
    if seed.type == "npc":
        token = meaningful_parts[0] if meaningful_parts else ""
    variants = {
        seed.name.strip(),
        cleaned.strip(),
        token.strip(),
        *component_tokens,
    }
    redacted = text
    for variant in sorted(filter(None, variants), key=len, reverse=True):
        parts = re.findall(r"[A-Za-z0-9']+", variant)
        if not parts:
            continue
        # Treat whitespace and punctuation/hyphens as equivalent so
        # "cracked-bell" and "cracked bell" cannot evade the washout.
        pattern = r"\b" + r"[\W_]+".join(map(re.escape, parts)) + r"\b"
        redacted = re.sub(
            pattern,
            "[sealed callback detail]",
            redacted,
            flags=re.IGNORECASE,
        )
    return redacted


def _player_action_problem(action: str) -> Optional[str]:
    """Return a concrete format/completeness problem, if any."""
    if not action:
        return "empty"
    if "\n" in action or "\r" in action:
        return "contains newline"
    if not re.match(r"^I\b", action.strip(), re.IGNORECASE):
        return "not first-person"
    word_count = len(re.findall(r"\b[\w'-]+\b", action))
    if word_count < 8:
        return f"too short ({word_count} words)"
    if word_count > 65:
        return f"too long ({word_count} words)"
    if not re.search(r"[.!?][\"']?$", action.strip()):
        return "missing terminal punctuation (possibly truncated)"
    if action.count('"') % 2 or action.count("'") % 2:
        # Apostrophes inside words are not quotation marks.
        scrubbed = re.sub(r"(?<=\w)'(?=\w)", "", action)
        if scrubbed.count('"') % 2 or scrubbed.count("'") % 2:
            return "unbalanced quotation mark (possibly truncated)"
    return None


def evaluate_player_action_quality(actions: list[str]) -> list[AssertionResult]:
    """Deterministic creativity floor; subjective prose remains rubric work."""
    total = len(actions)
    problems = [
        f"T{i + 1}: {_player_action_problem(action)}"
        for i, action in enumerate(actions)
        if _player_action_problem(action)
    ]

    normalized = [
        re.sub(r"[^a-z0-9 ]+", "", re.sub(r"\s+", " ", a.lower())).strip()
        for a in actions
    ]
    unique = len(set(normalized)) if normalized else 0
    unique_ratio = unique / total if total else 0.0

    verbs = Counter(filter(None, (_main_action_verb(a) for a in actions)))
    distinct_verbs = len(verbs)
    dominant_verb, dominant_count = verbs.most_common(1)[0] if verbs else ("", 0)
    dominant_ratio = dominant_count / total if total else 1.0
    min_verbs = min(12, max(4, total // 6)) if total else 4
    varied = bool(
        total
        and unique_ratio >= 0.80
        and distinct_verbs >= min_verbs
        and dominant_ratio <= 0.35
    )

    generic_turns = [
        i + 1 for i, action in enumerate(actions)
        if len(action.split()) <= 14 and _GENERIC_STALL_RE.search(action)
    ]
    max_generic = max(2, (total + 9) // 10) if total else 0

    return [
        AssertionResult(
            name="player_actions_well_formed",
            passed=not problems and bool(total),
            description="Every simulated-player action is complete, first-person, and actionable.",
            detail="; ".join(problems[:5]) if problems else f"Checked {total} actions",
        ),
        AssertionResult(
            name="player_action_variety",
            passed=varied,
            description="Player avoids repeating the same action wording and leading verb.",
            detail=(
                f"unique={unique}/{total} ({unique_ratio:.0%}), verbs={distinct_verbs}/"
                f"{min_verbs}, dominant={dominant_verb or '-'} ({dominant_ratio:.0%})"
            ),
        ),
        AssertionResult(
            name="player_avoids_generic_stalling",
            passed=len(generic_turns) <= max_generic and bool(total),
            description="Passive filler actions remain below the deterministic stall budget.",
            detail=f"generic turns={generic_turns}; allowed={max_generic}",
        ),
    ]


def evaluate_narrator_prose_quality(
    narrations: list[tuple[int, str]],
) -> list[AssertionResult]:
    """Fail deterministically when private model analysis reaches the story."""
    governance = NarrativeGovernance()
    leaks = []
    for turn, prose in narrations:
        violations = [
            violation
            for violation in governance.validate(prose)
            if violation.rule == "meta_reasoning_leak"
        ]
        if violations:
            leaks.append(f"T{turn}: {violations[0].excerpt!r}")
    return [AssertionResult(
        name="narrator_no_meta_reasoning_leak",
        passed=not leaks and bool(narrations),
        description=(
            "Narrator output contains only player-visible story prose, never "
            "private planning or world-state analysis."
        ),
        detail="; ".join(leaks[:5]) if leaks else f"Checked {len(narrations)} turns",
    )]


def evaluate_tool_coverage(
    effects_by_turn: list[list[dict]],
    proposed_by_turn: Optional[list[list[dict]]] = None,
    rejected_by_turn: Optional[list[list[dict]]] = None,
    diagnostics_by_turn: Optional[list[dict]] = None,
) -> list[AssertionResult]:
    """Broad trajectory checks; the focused tool gauntlet remains stricter."""
    effect_types = Counter(
        str(effect.get("type") or effect.get("effect_type") or "")
        for effects in effects_by_turn
        for effect in effects
        if effect.get("type") or effect.get("effect_type")
    )
    total_turns = len(effects_by_turn)
    active_turns = sum(bool(effects) for effects in effects_by_turn)
    active_ratio = active_turns / total_turns if total_turns else 0.0

    durable_types = {"update_player", "update_entity", "remove_entity", "change_location"}
    durable_counts = {name: effect_types[name] for name in durable_types if effect_types[name]}
    durable_total = sum(durable_counts.values())

    results = [
        AssertionResult(
            name="tool_effect_type_diversity",
            passed=len(effect_types) >= 5,
            description="The long story exercises at least five distinct narrator effect types.",
            detail=f"types={dict(effect_types)}",
        ),
        AssertionResult(
            name="durable_tool_mutation_coverage",
            passed=len(durable_counts) >= 2 and durable_total >= 6,
            description=(
                "At least two durable mutation families fire repeatedly, not just entity references."
            ),
            detail=f"durable={durable_counts}; total={durable_total}",
        ),
        AssertionResult(
            name="tool_effect_turn_coverage",
            passed=bool(total_turns) and active_ratio >= 0.60,
            description="Most turns emit at least one trackable effect.",
            detail=f"effect turns={active_turns}/{total_turns} ({active_ratio:.0%})",
        ),
    ]
    if proposed_by_turn is not None and rejected_by_turn is not None:
        proposed_total = sum(len(effects) for effects in proposed_by_turn)
        executed_total = sum(len(effects) for effects in effects_by_turn)
        rejected_total = sum(len(effects) for effects in rejected_by_turn)
        accounting_balanced = (
            proposed_total == executed_total + rejected_total
        )
        success_ratio = (
            (proposed_total - rejected_total) / proposed_total
            if proposed_total else 0.0
        )
        results.append(AssertionResult(
            name="tool_effect_accounting_balanced",
            passed=bool(proposed_total) and accounting_balanced,
            description=(
                "Every proposed effect has exactly one execution/idempotency "
                "receipt or one rejection receipt."
            ),
            detail=(
                f"proposed={proposed_total}; executed-or-idempotent={executed_total}; "
                f"rejected={rejected_total}"
            ),
        ))
        results.append(AssertionResult(
            name="tool_effect_execution_reliability",
            passed=bool(proposed_total) and success_ratio >= 0.98,
            description="At least 98% of narrator-proposed effects validate and execute.",
            detail=(
                f"executed-or-idempotent={proposed_total - rejected_total}/"
                f"{proposed_total} ({success_ratio:.1%}); rejected={rejected_total}"
            ),
        ))
        if diagnostics_by_turn is not None:
            structural_errors = sum(
                int(diag.get("primary_structural_errors", 0) or 0)
                + int(diag.get("tool_followup_structural_errors", 0) or 0)
                + int(diag.get("tool_repair_structural_errors", 0) or 0)
                + int(
                    diag.get(
                        "effect_obligation_terminal_structural_errors", 0
                    )
                    or 0
                )
                for diag in diagnostics_by_turn
            )
            dropped = sum(
                int(diag.get("tool_invalid_effects_dropped", 0) or 0)
                for diag in diagnostics_by_turn
            )
            policy_suppressed = sum(
                int(diag.get("tool_policy_suppressed_effects", 0) or 0)
                for diag in diagnostics_by_turn
            )
            failed_closed_turns = [
                index + 1
                for index, diag in enumerate(diagnostics_by_turn)
                if diag.get("tool_repair_failed_closed")
            ]
            attempted_effects = proposed_total + dropped
            dropped_ratio = dropped / attempted_effects if attempted_effects else 0.0
            results.append(AssertionResult(
                name="tool_structural_failure_budget",
                passed=dropped_ratio <= 0.05,
                description=(
                    "Malformed narrator tools remain bounded and fail closed "
                    "without dominating the run."
                ),
                detail=(
                    f"structural_errors={structural_errors}; dropped={dropped}/"
                    f"{attempted_effects} ({dropped_ratio:.1%}); "
                    f"failed_closed_turns={failed_closed_turns}; "
                    "semantic misses are enforced by obligation and omission gates"
                ),
            ))
            policy_attempts = proposed_total + policy_suppressed
            policy_suppression_ratio = (
                policy_suppressed / policy_attempts if policy_attempts else 0.0
            )
            results.append(AssertionResult(
                name="tool_policy_suppression_budget",
                passed=bool(policy_attempts) and policy_suppression_ratio <= 0.10,
                description=(
                    "The model rarely attempts high-risk tools that runtime "
                    "policy deliberately withheld."
                ),
                detail=(
                    f"suppressed={policy_suppressed}/{policy_attempts} "
                    f"({policy_suppression_ratio:.1%})"
                ),
            ))
            unmet_obligation_turns = [
                {
                    "turn": index + 1,
                    "missing": list(
                        diag.get("effect_obligation_missing_final") or []
                    ),
                }
                for index, diag in enumerate(diagnostics_by_turn)
                if diag.get("effect_obligation_missing_final")
            ]
            outcome_failed_closed_turns = [
                index + 1
                for index, diag in enumerate(diagnostics_by_turn)
                if diag.get("resolved_outcome_failed_closed")
            ]
            results.append(AssertionResult(
                name="runtime_effect_obligations_met",
                passed=(
                    not unmet_obligation_turns
                    and not outcome_failed_closed_turns
                ),
                description=(
                    "Every high-confidence resolved action proposed each "
                    "required effect family without an unresolved prose conflict."
                ),
                detail=(
                    f"unmet={unmet_obligation_turns}; "
                    f"outcome_failed_closed={outcome_failed_closed_turns}"
                ),
            ))
    return results


_NPC_POSSESSIVE_CUE_RE = re.compile(
    r"\b([A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,2})['’]s\s+"
    r"(?:eyes?|gaze|voice|face|hands?|brow|jaw|shoulders?|expression|"
    r"smile|fingers?|head|lips?|breath|posture)\b"
)
_LEADING_NAME_NOISE = {"And", "As", "But", "Then", "When", "While", "Yet"}


def _normalized_entity_label(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _looks_like_proper_npc_name(value: object) -> bool:
    """Conservative proper-name check for the independent omission audit.

    ``add_npc`` intentionally forbids anonymous labels.  The audit therefore
    ignores extractor entries such as "the cloaked woman" and only treats a
    StateDelta NPC as tool-requiring when its visible label looks like a
    proper name.
    """
    raw = str(value or "").strip()
    normalized = _normalized_entity_label(raw)
    if not normalized or re.match(r"^(?:a|an|the)\s+", normalized):
        return False
    if is_generic_npc_label(raw):
        return False
    return any(character.isupper() for character in raw)


def _entity_labels_overlap(left: object, right: object) -> bool:
    left_tokens = set(_normalized_entity_label(left).split())
    right_tokens = set(_normalized_entity_label(right).split())
    return bool(left_tokens and right_tokens) and (
        left_tokens == right_tokens
        or left_tokens.issubset(right_tokens)
        or right_tokens.issubset(left_tokens)
    )


def _strong_npc_name_cues(text: str) -> set[str]:
    """Extract only grammatical cues that very strongly imply an on-stage NPC."""
    names: set[str] = set()
    for match in _NPC_POSSESSIVE_CUE_RE.finditer(text or ""):
        tokens = match.group(1).split()
        while tokens and tokens[0] in _LEADING_NAME_NOISE:
            tokens.pop(0)
        candidate = " ".join(tokens)
        if _looks_like_proper_npc_name(candidate):
            names.add(candidate)
    return names


def evaluate_canonical_npc_identity(catalog_entities: list[dict]) -> AssertionResult:
    """Reject two durable NPC nodes claiming the same proper canonical name."""
    owners: dict[str, set[str]] = {}
    display_names: dict[str, str] = {}
    for index, entity in enumerate(catalog_entities or []):
        if not isinstance(entity, dict) or entity.get("type") != "npc":
            continue
        name = str(entity.get("name") or "").strip()
        if not name or is_generic_npc_label(name):
            continue
        normalized = _normalized_entity_label(name)
        if not normalized:
            continue
        display_names.setdefault(normalized, name)
        owners.setdefault(normalized, set()).add(
            str(entity.get("id") or f"missing-id-{index}")
        )
    collisions = {
        display_names[name]: sorted(entity_ids)
        for name, entity_ids in owners.items()
        if len(entity_ids) > 1
    }
    return AssertionResult(
        name="canonical_npc_identity_unique",
        passed=not collisions,
        description=(
            "No two durable graph NPC nodes claim the same proper canonical name."
        ),
        detail=f"collisions={collisions}",
    )


def evaluate_tool_omission_signals(
    records: list[tuple[int, dict]],
) -> list[AssertionResult]:
    """Cross-check narrator tools against independently extracted mutations.

    StateDelta is produced by the local brain after narration, so it can catch
    an important failure mode that proposal/execution accounting cannot: the
    narrator wrote a trackable change but never called a tool at all.  Only
    overlaps with an existing narrator tool are audited, keeping this signal
    conservative rather than pretending every fact or quest has a tool.
    """
    expected: list[str] = []
    missing: list[str] = []
    ungrounded_alias_refs: list[str] = []
    misbound_alias_refs: list[str] = []
    prior_catalog_npc_ids: set[str] = set()

    for turn, record in records:
        delta = (record.get("state_delta") or {}).get("delta") or {}
        proposed = (record.get("effects") or {}).get("proposed") or []
        narration = str((record.get("narrator_response") or {}).get("raw") or "")
        normalized_narration = f" {_normalized_entity_label(narration)} "
        # Updates the single-writer store REJECTED (typically "NPC not found"
        # for a background figure that was never tracked) mutated nothing, so
        # they owe no narrator tool.
        rejected_update_labels: set[str] = set()
        for rejection in (record.get("state_delta") or {}).get("rejections") or []:
            match = re.search(
                r"not found for update: id=(\S+) name='([^']*)'", str(rejection)
            )
            if match:
                for value in match.groups():
                    normalized = _normalized_entity_label(value)
                    if normalized and normalized != "none":
                        rejected_update_labels.add(normalized)
        proposed_types = Counter(
            str(effect.get("type") or effect.get("effect_type") or "")
            for effect in proposed
            if isinstance(effect, dict)
        )

        location = str(delta.get("location_change") or "").strip()
        if location:
            before_yaml = str((record.get("world_state") or {}).get("before") or "")
            before_match = re.search(r"(?m)^location:\s*(.+)$", before_yaml)
            before_location = (
                before_match.group(1).strip().strip("'\"")
                if before_match
                else ""
            )
            signal = f"T{turn} change_location({location})"
            normalized_location = _normalized_entity_label(location)
            if (
                normalized_location
                and f" {normalized_location} " in normalized_narration
                and normalized_location != _normalized_entity_label(before_location)
                # A base place and its qualified sub-scene ("Tallow Rows" vs
                # "Tallow Rows alley") are the same location identity, so no
                # change_location proposal is owed.
                and not locations_equivalent(location, before_location)
            ):
                expected.append(signal)
                if proposed_types["change_location"] < 1:
                    missing.append(signal)

        proposed_npc_names = {
            _normalized_entity_label(effect.get("npc_name") or effect.get("name"))
            for effect in proposed
            if isinstance(effect, dict)
            and str(effect.get("type") or effect.get("effect_type") or "") == "add_npc"
        }
        proposed_refs = [
            effect
            for effect in proposed
            if isinstance(effect, dict)
            and str(effect.get("type") or effect.get("effect_type") or "")
            == "ref_entity"
        ]
        for effect in proposed:
            if not isinstance(effect, dict):
                continue
            effect_type = str(effect.get("type") or effect.get("effect_type") or "")
            alias = str(effect.get("ref_alias") or effect.get("alias_used") or "").strip()
            normalized_alias = _normalized_entity_label(alias)
            alias_tokens = normalized_alias.split()
            alias_noise = {
                "a", "an", "the", "this", "that", "it", "he", "her",
                "him", "she", "them", "they", "here", "there", "of",
                "to", "in", "on", "at",
            }
            if (
                effect_type == "ref_entity"
                and normalized_alias
                and (
                    not any(token not in alias_noise for token in alias_tokens)
                    or f" {normalized_alias} " not in normalized_narration
                )
            ):
                ungrounded_alias_refs.append(f"T{turn} ref_entity({alias})")

        catalog_entities = [
            entity
            for entity in (record.get("knowledge_graph") or {}).get("catalog_entities") or []
            if isinstance(entity, dict)
        ]
        catalog_npcs = [
            entity for entity in catalog_entities if entity.get("type") == "npc"
        ]
        catalog_by_id = {
            str(entity.get("id") or "").strip(): entity
            for entity in catalog_entities
            if str(entity.get("id") or "").strip()
        }
        for ref in proposed_refs:
            ref_id = str(ref.get("ref_id") or "").strip()
            alias = str(
                ref.get("ref_alias") or ref.get("alias_used") or ""
            ).strip()
            target = catalog_by_id.get(ref_id)
            if not target or not alias:
                continue
            canonical_name = str(target.get("name") or "").strip()
            if (
                target.get("type") == "npc"
                and is_generic_npc_label(canonical_name)
            ):
                # Explicitly named generic identities are validated against
                # their naming cue in production. The post-turn catalog may
                # still contain the generic label during that transition.
                continue
            if is_generic_npc_label(alias):
                # A role/title alias ("Brother", "the woman") is descriptive
                # address of the entity, not a competing identity claim.
                continue
            canonical_labels = [
                canonical_name,
                *(target.get("aliases") or []),
            ]
            if not any(
                _entity_labels_overlap(alias, label)
                for label in canonical_labels
                if label
            ):
                misbound_alias_refs.append(
                    f"T{turn} ref_entity({canonical_name or ref_id} <- {alias})"
                )
        catalog_npc_ids = {
            str(entity.get("id") or "").strip()
            for entity in catalog_npcs
            if str(entity.get("id") or "").strip()
        }
        known_npc_labels = {
            str(label)
            for entity in catalog_npcs
            for label in [entity.get("name"), *(entity.get("aliases") or [])]
            if label
        }
        player_name = str(record.get("player") or "")
        for candidate in sorted(_strong_npc_name_cues(narration)):
            if player_name and _entity_labels_overlap(candidate, player_name):
                continue
            if any(_entity_labels_overlap(candidate, known) for known in known_npc_labels):
                continue
            signal = f"T{turn} add_npc({candidate}; prose cue)"
            expected.append(signal)
            if not any(
                _entity_labels_overlap(candidate, proposed_name)
                for proposed_name in proposed_npc_names
            ) and not any(
                _entity_labels_overlap(candidate, ref.get("ref_alias"))
                for ref in proposed_refs
            ):
                missing.append(signal)

        for npc in delta.get("new_npcs") or []:
            if not isinstance(npc, dict) or not _looks_like_proper_npc_name(npc.get("name")):
                continue
            name = str(npc.get("name") or "").strip()
            npc_id = str(npc.get("id") or "").strip()
            if npc_id and npc_id in prior_catalog_npc_ids:
                # StateDelta may project a durable off-screen NPC back into the
                # scene. Its existing graph UUID proves this is not a new
                # identity and therefore does not require add_npc.
                continue
            normalized_name = _normalized_entity_label(name)
            if normalized_name and f" {normalized_name} " not in normalized_narration:
                # The extractor can resolve a role from prior context even
                # when the visible prose leaves the person unnamed. add_npc's
                # exact-grounding contract correctly forbids that tool call.
                continue
            # A narrator ref is an explicit identity claim. If StateDelta
            # redundantly labels that same id/alias as new, production merges
            # it onto the existing UUID rather than requiring add_npc.
            if any(
                (npc_id and npc_id == str(ref.get("ref_id") or "").strip())
                or _entity_labels_overlap(name, ref.get("ref_alias"))
                or _entity_labels_overlap(name, ref.get("ref_id"))
                for ref in proposed_refs
            ):
                continue
            signal = f"T{turn} add_npc({name})"
            expected.append(signal)
            if _normalized_entity_label(name) not in proposed_npc_names:
                missing.append(signal)

        prior_catalog_npc_ids.update(catalog_npc_ids)

        mutation_updates = 0
        for update in delta.get("npc_updates") or []:
            if not isinstance(update, dict):
                continue
            update_target = _normalized_entity_label(
                update.get("new_name") or update.get("name") or update.get("id")
            )
            if update_target and update_target in rejected_update_labels:
                continue
            # The extractor may enrich an NPC that add_npc established earlier
            # in this same turn.  add_npc already persisted its initial state;
            # requiring a redundant update_entity call would be a false alarm.
            if update_target and any(
                update_target == proposed_name
                or set(update_target.split()).issubset(set(proposed_name.split()))
                for proposed_name in proposed_npc_names
            ):
                continue
            disposition = str(update.get("disposition") or "").strip().lower()
            explicit_disposition_change = bool(
                disposition
                and re.search(
                    rf"\b(?:becomes?|turns?|now|swears?|joins?|betrays?)\b"
                    rf".{{0,45}}\b(?:{re.escape(disposition)}|ally|enemy|friend)\b",
                    narration,
                    re.IGNORECASE,
                )
            )
            if (
                explicit_disposition_change
                or update.get("alive") is False
                or bool(update.get("add_inventory"))
                or bool(update.get("remove_inventory"))
            ):
                mutation_updates += 1
        for index in range(mutation_updates):
            signal = f"T{turn} update_entity({index + 1}/{mutation_updates})"
            expected.append(signal)
            if proposed_types["update_entity"] <= index:
                missing.append(signal)

    observed = len(expected)
    covered = observed - len(missing)
    return [
        AssertionResult(
            name="tool_omission_observer_exercised",
            passed=observed > 0,
            description=(
                "The independent StateDelta pass observed at least one mutation "
                "that has a matching narrator tool family."
            ),
            detail=f"observed={observed}",
        ),
        AssertionResult(
            name="tool_omission_signal_coverage",
            passed=observed > 0 and not missing,
            description=(
                "Every high-confidence StateDelta mutation had a corresponding "
                "narrator tool proposal; rejected attempts are evaluated separately."
            ),
            detail=(
                f"covered={covered}/{observed}; missing={missing[:10]}"
                + (f" (+{len(missing) - 10} more)" if len(missing) > 10 else "")
            ),
        ),
        AssertionResult(
            name="tool_reference_alias_grounding",
            passed=not ungrounded_alias_refs,
            description=(
                "Every ref_entity alias_used value appears in the narrator prose "
                "that the tool claims referenced it."
            ),
            detail=f"ungrounded={ungrounded_alias_refs[:10]}"
            + (
                f" (+{len(ungrounded_alias_refs) - 10} more)"
                if len(ungrounded_alias_refs) > 10
                else ""
            ),
        ),
        AssertionResult(
            name="tool_reference_identity_grounding",
            passed=not misbound_alias_refs,
            description=(
                "Every ref_entity alias belongs to the canonical graph entity "
                "whose ID the tool claims it referenced."
            ),
            detail=f"misbound={misbound_alias_refs[:10]}"
            + (
                f" (+{len(misbound_alias_refs) - 10} more)"
                if len(misbound_alias_refs) > 10
                else ""
            ),
        ),
    ]


def _supports_structured_effect_accounting(log, turns: list[int]) -> bool:
    """True when a run uses structured effect receipts where applicable.

    Early PGI-blocked turns legitimately end before narration and therefore
    have no ``effects`` key. They count as empty turns; they must not disable
    accounting gates for every other turn in the run. Legacy flat-list logs
    still opt out because they cannot prove proposal/rejection balance.
    """
    payloads = [(log.get(turn) or {}).get("effects") for turn in turns]
    return (
        any(isinstance(payload, dict) for payload in payloads)
        and all(payload is None or isinstance(payload, dict) for payload in payloads)
    )


class GeminiFlashPlayer:
    """Plays a D&D character via Gemini Flash-Lite with persistent memory of
    the goal + history of recent narrator responses. Also handles the
    framework's seed-identification call."""

    def __init__(self, scenario: Scenario, history_window: int = 10):
        from dnd_bot.config import get_profile
        from dnd_bot.llm.client import GeminiClient, OllamaClient
        # The player client is built directly (NOT via _create_client) so it
        # is NOT auto-instrumented by the usage recorder — we record its usage
        # by hand after each chat() call below.
        from dnd_bot.llm import usage_recorder
        self._usage = usage_recorder
        # Flash-Lite is sufficient for a simulated player and does not think by
        # default. Full 2.5 Flash uses dynamic thinking, which can consume a
        # small max-output budget and return a visibly truncated action.
        self.client = GeminiClient(model="gemini-2.5-flash-lite")
        active = get_profile()
        fallback_model = (
            active.brain.model
            if active.brain.provider == "ollama" and active.brain.model
            else "qwen3.5:9b"
        )
        fallback_ctx = (
            active.brain.context_size
            if active.brain.provider == "ollama" and active.brain.context_size
            else 8000
        )
        self.fallback_client = OllamaClient(
            model=fallback_model,
            num_ctx=fallback_ctx,
        )
        self.scenario = scenario
        self.history_window = history_window
        self.history: list[tuple[str, str]] = []  # (action, narrator_response)
        self.last_regenerations = 0
        self.last_provider_fallbacks = 0
        self.character_state = (
            "Kael has not acted yet. Establish a concrete immediate motive, "
            "then update this note with current loyalties, suspicions, promises, "
            "possessions, and emotional pressure."
        )

    def _record_usage(self, response, elapsed_ms: float) -> None:
        """Feed one player/seed-pick LLMResponse into the usage recorder."""
        try:
            model = getattr(response, "model", "") or getattr(
                self.client, "model", "gemini-2.5-flash-lite"
            )
            is_gemini = str(model).casefold().startswith("gemini")
            self._usage.record(
                provider="gemini" if is_gemini else "ollama",
                model=model,
                stage="player" if is_gemini else "player_fallback",
                prompt_tokens=getattr(response, "prompt_tokens", 0) or 0,
                completion_tokens=getattr(response, "completion_tokens", 0) or 0,
                cache_read_tokens=getattr(response, "cache_read_tokens", 0) or 0,
                cache_write_tokens=getattr(response, "cache_write_tokens", 0) or 0,
                elapsed_ms=elapsed_ms,
            )
        except Exception:
            pass  # telemetry must never break a run

    async def _chat_with_fallback(self, **kwargs):
        """Use local Qwen when the external player returns no usable body.

        Gemini can safety-block a benign action after many turns. That is a
        provider artifact, not a product-quality result, so the simulated
        player fails over locally while retaining the same JSON contract.
        """
        started = time.perf_counter()
        response = await self.client.chat(**kwargs)
        self._record_usage(response, (time.perf_counter() - started) * 1000.0)
        if (getattr(response, "content", "") or "").strip():
            return response

        fallback_client = getattr(self, "fallback_client", None)
        if fallback_client is None:
            return response

        self.last_provider_fallbacks = getattr(
            self, "last_provider_fallbacks", 0
        ) + 1
        started = time.perf_counter()
        fallback_response = await fallback_client.chat(**kwargs, think=False)
        self._record_usage(
            fallback_response,
            (time.perf_counter() - started) * 1000.0,
        )
        return fallback_response

    async def next_action(
        self,
        narrator_response: str,
        phase: Phase,
        seed: Optional[Seed],
        forced_action: Optional[str] = None,
    ) -> str:
        self.last_provider_fallbacks = 0

        # Append the previous narrator response to history (for context)
        if self.history:
            self.history[-1] = (self.history[-1][0], narrator_response)

        # A callback test cannot begin without one player-visible, graph-backed
        # emergent entity. The harness may supply a final-turn recovery action
        # rather than pay for another soft-prompt attempt. Preserve it in the
        # player's history exactly like a model-generated action.
        if forced_action:
            self.last_regenerations = 0
            self.history.append((forced_action, ""))
            return forced_action

        absolute_turn = len(self.history) + 1
        callback_phase = next(
            (candidate for candidate in self.scenario.phases if candidate.name == "callback"),
            None,
        )
        # Hide the seed from the simulated player for the *entire* middle act.
        # ``memory_silence_range`` is deliberately narrower: it starts only
        # after production's recent-turn buffer has cooled off and is used for
        # narrator/KG assertions, not for deciding what the player may see.
        in_seed_washout = bool(
            seed
            and callback_phase
            and seed.chosen_after_turn < absolute_turn < callback_phase.turn_range[0]
        )

        # Resolve phase instruction with seed substitutions
        instruction = phase.instruction
        if seed:
            if in_seed_washout:
                instruction = instruction.replace(
                    "{seed_name}", "[sealed callback detail]"
                ).replace("{seed_type}", "detail")
            else:
                instruction = instruction.replace("{seed_name}", seed.name).replace(
                    "{seed_type}", seed.type
                )
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
            first_recent_turn = len(self.history) - len(recent) + 1
            for i, (action, response) in enumerate(recent, start=first_recent_turn):
                visible_action = (
                    _redact_seed_text(action, seed) if in_seed_washout else action
                )
                lines.append(f"  Turn {i}: I said \"{visible_action}\"")
                # The latest DM response is supplied in full head+tail form
                # below; do not duplicate it in the history block.
                if response and i < len(self.history):
                    visible_response = (
                        _redact_seed_text(response, seed)
                        if in_seed_washout
                        else response
                    )
                    lines.append(
                        f"    DM: {_narration_excerpt(visible_response, 280)}"
                    )
            history_block = "Recent turns:\n" + "\n".join(lines) + "\n\n"

        seed_note = ""
        if seed and in_seed_washout:
            seed_note = (
                "\nMEMORY TEST: One opening detail is sealed during this phase. "
                "Do not identify, reconstruct, investigate, or allude to it; "
                "pursue only the visible current story.\n"
            )
        elif seed:
            seed_note = (
                f"\nMEMORY: Earlier you noted a {seed.type} called "
                f"\"{seed.name}\" ({seed.reason}). Use this in the "
                f"appropriate phase per your instruction.\n"
            )

        phase_span = max(1, phase.turn_range[1] - phase.turn_range[0] + 1)
        phase_progress = absolute_turn - phase.turn_range[0] + 1
        recent_verbs = [
            verb for verb in (_main_action_verb(a) for a, _ in recent[-3:]) if verb
        ]

        system = f"""You are playing one protagonist in a serious, imaginative D&D 5e campaign.

CHARACTER VOICE AND FLAWS:
{self.scenario.player_persona}

OVERALL GOAL:
{self.scenario.base_goal}

CURRENT PHASE ({phase.name}, turns {phase.turn_range[0]}-{phase.turn_range[1]}):
{instruction}
This is absolute turn {absolute_turn}, step {phase_progress} of {phase_span} in this phase.
{seed_note}

PRIVATE CONTINUITY NOTE (never address this note aloud):
{_redact_seed_text(self.character_state, seed) if in_seed_washout else self.character_state}

PLAY PRINCIPLES:
- Choose an action because Kael wants something now; do not behave like a QA checklist.
- Prefer actions that do at least two of these: exploit a concrete scene detail, reveal character, risk a cost, change a relationship, or force a meaningful choice.
- Vary your mode among investigation, pointed dialogue, bargaining, confession, deception, travel, using or transferring an item, making a promise, and testing a theory.
- Do not repeat the main approach of the last few actions ({', '.join(recent_verbs) or 'none yet'}) when a different credible approach exists.
- You may include brief, distinctive in-character dialogue, but never narrate an NPC's response or claim an uncertain outcome succeeded.
- Avoid passive filler such as merely looking around, waiting, continuing, or considering your next move.

OUTPUT CONTRACT:
Return exactly one JSON object with two strings:
{{"action": "one concrete first-person action, normally 18-45 words", "continuity": "an updated private 20-60 word note of Kael's motives, loyalties, suspicions, promises, possessions, and pressure"}}
The action must begin with "I", end with punctuation, and contain no markdown or out-of-character commentary."""

        opening = ""
        if not narrator_response and self.scenario.opening_situation:
            opening = f"Player-visible opening situation: {self.scenario.opening_situation}\n\n"
        visible_narrator_response = (
            _redact_seed_text(narrator_response, seed)
            if in_seed_washout
            else narrator_response
        )
        user = (
            history_block
            + opening
            + f"DM's last response: {_narration_excerpt(visible_narrator_response, 1000) if visible_narrator_response else '(scene begins)'}\n\n"
            + "What does your character do next?"
        )

        rejection = ""
        self.last_regenerations = 0
        for attempt in range(1, 4):
            retry_user = user
            if rejection:
                retry_user += (
                    "\n\nYour previous output was rejected: "
                    f"{rejection}. Choose a materially different action and "
                    "obey the output contract."
                )

            response = await self._chat_with_fallback(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": retry_user},
                ],
                temperature=0.9,
                max_tokens=350,
                json_mode=True,
                json_schema=_PLAYER_RESPONSE_SCHEMA,
            )
            self.last_regenerations = attempt - 1
            try:
                payload = json.loads(response.content or "")
                action = str(payload["action"]).strip()
                continuity = str(payload["continuity"]).strip()
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                rejection = f"invalid structured output ({exc})"
                continue

            problem = _player_action_problem(action)
            if problem:
                rejection = f"invalid action: {problem}"
                continue
            if in_seed_washout and seed:
                seed_clean, seed_token = _clean_seed(seed)
                if _seed_matches(action, seed_clean, seed_token, seed.type):
                    rejection = (
                        "the action leaked the sealed callback detail during "
                        "the memory washout"
                    )
                    continue

            if continuity:
                # Bound future prompt growth even if the actor ignores the requested size.
                self.character_state = " ".join(continuity.split()[:80])
            self.history.append((action, ""))
            return action

        raise ValueError(f"Player failed action contract after 3 attempts: {rejection}")

    async def pick_seed(
        self,
        narration_history: list[str],
        scenario: Scenario,
        seed_candidates: Optional[list[dict[str, str]]] = None,
    ) -> Seed:
        """Ask Gemini to identify ONE concrete element from the early
        narration to come back to later.

        Uses ``json_mode=True`` so Gemini's ``response_mime_type`` is set
        to ``application/json``. An unconstrained response truncated the
        emitted JSON mid-token (observed at 23 chars in early runs).
        """
        narr_text = "\n---\n".join(
            f"Turn {i+1}:\n{n}" for i, n in enumerate(narration_history) if n
        )

        # The callback target must be a real durable-memory entity, not merely a
        # phrase rediscovered from prose. When the graph yields exactly one
        # eligible entity there is no ranking decision to outsource to Gemini.
        if seed_candidates and len(seed_candidates) == 1:
            candidate = seed_candidates[0]
            return Seed(
                type=candidate["type"],
                name=candidate["name"],
                reason=candidate.get("description") or "Canonical graph-backed emergent entity.",
                chosen_after_turn=scenario.seed_pick_after_turn,
            )

        candidate_note = ""
        allowed_by_name: dict[str, dict[str, str]] = {}
        if seed_candidates:
            allowed_by_name = {
                candidate["name"].casefold(): candidate
                for candidate in seed_candidates
            }
            candidate_note = (
                "\n\nELIGIBLE CANONICAL GRAPH ENTITIES:\n"
                + json.dumps(seed_candidates, ensure_ascii=False)
                + "\nChoose an exact name and matching type from this list only."
            )

        rejection = ""
        last_problem = "no response"
        for attempt in range(1, 4):
            retry_note = ""
            if rejection:
                retry_note = (
                    "\n\nYOUR PREVIOUS CANDIDATE WAS REJECTED: "
                    f"{rejection}\nChoose a different exact name that first appeared "
                    "in the recent narration and is not in the forbidden premise list."
                )

            response = await self._chat_with_fallback(
                messages=[
                    {"role": "system", "content": (
                        "You select one trustworthy long-term-memory test seed "
                        "from D&D narration. Obey forbidden-name constraints."
                    )},
                    {"role": "user", "content": (
                        scenario.seed_pick_prompt + narr_text + candidate_note + retry_note
                    )},
                ],
                temperature=0.2,
                max_tokens=512,
                json_mode=True,
                json_schema=_SEED_RESPONSE_SCHEMA,
            )
            raw = (response.content or "").strip()

            try:
                seed_data = json.loads(raw)
                picked = Seed(
                    type=str(seed_data["type"]),
                    name=str(seed_data["name"]).strip(),
                    reason=str(seed_data["reason"]).strip(),
                    chosen_after_turn=scenario.seed_pick_after_turn,
                )
            except (json.JSONDecodeError, KeyError, TypeError) as exc:
                last_problem = f"invalid JSON on attempt {attempt}: {exc}; raw={raw[:160]!r}"
                rejection = last_problem
                continue

            if picked.type not in {"npc", "item", "place"}:
                last_problem = f"unsupported type {picked.type!r}"
                rejection = f"{picked.name!r}: {last_problem}"
                continue

            if allowed_by_name:
                canonical = allowed_by_name.get(picked.name.casefold())
                if canonical is None or picked.type != canonical["type"]:
                    last_problem = (
                        f"{picked.name!r} ({picked.type}) is not an exact eligible "
                        "canonical graph candidate"
                    )
                    rejection = last_problem
                    continue
            if not picked.name or picked.name.lower() not in narr_text.lower():
                last_problem = f"name {picked.name!r} does not appear exactly in recent narration"
                rejection = last_problem
                continue

            excluded = [
                name for name in scenario.seed_exclusions
                if _seed_names_overlap(picked.name, name)
            ]
            if excluded:
                last_problem = (
                    f"{picked.name!r} came from the fixed premise: {excluded}"
                )
                rejection = last_problem
                continue
            return picked

        # Gemini kept choosing off-list (observed 2026-07-23: 'metallic
        # residue' three times despite the eligible-candidates constraint).
        # Every candidate is already validated as canonical, emergent, and
        # narration-exact, so ranking is the only thing being outsourced —
        # fall back to the priority-sorted top candidate instead of aborting
        # a run whose graph state is perfectly usable.
        if seed_candidates:
            candidate = seed_candidates[0]
            print(
                f"  {C.YELLOW}seed_pick_fallback: Gemini failed 3 attempts "
                f"({last_problem}); using top canonical candidate "
                f"{candidate['name']!r}{C.RESET}"
            )
            return Seed(
                type=candidate["type"],
                name=candidate["name"],
                reason=candidate.get("description")
                or "Canonical graph-backed emergent entity (ranking fallback).",
                chosen_after_turn=scenario.seed_pick_after_turn,
            )
        raise ValueError(f"Seed selection failed after 3 attempts: {last_problem}")


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


def _seed_is_fallback(seed: Optional[Seed], scenario: Optional[Scenario] = None) -> bool:
    """True when the seed cannot support a trustworthy recall verdict."""
    if seed is None:
        return True
    seed_name = seed.name.strip().lower()
    if seed_name in FALLBACK_SEED_NAMES:
        return True
    if seed.type == "npc" and _is_generic_npc_seed_name(seed.name):
        return True
    if seed.type == "place" and _is_generic_place_seed_name(seed.name):
        return True
    if scenario:
        return any(
            _seed_names_overlap(seed.name, excluded)
            for excluded in scenario.seed_exclusions
        )
    return False


def _is_generic_npc_seed_name(name: str) -> bool:
    """True when an NPC name is only a descriptive role, not an identity."""
    terms = re.findall(r"[a-z0-9']+", name.casefold())
    return is_generic_npc_label(name) or (
        bool(terms) and all(term in GENERIC_NPC_SEED_TERMS for term in terms)
    )


def _is_generic_place_seed_name(name: str) -> bool:
    """True when a place is only a reusable category, not a distinct place."""
    terms = re.findall(r"[a-z0-9']+", name.casefold())
    return bool(terms) and all(
        term in SEED_STOPWORDS or term in GENERIC_PLACE_SEED_TERMS
        for term in terms
    )


def _seed_names_overlap(candidate: str, excluded: str) -> bool:
    """Match premise aliases despite articles, punctuation, or plurality.

    The seed picker previously accepted ``the compass`` even though
    ``living brass compass`` was forbidden. A callback seed must be emergent,
    so sharing a meaningful content term with a fixed premise entity is enough
    to reject it.
    """
    def _terms(value: str) -> set[str]:
        terms = set()
        for token in re.findall(r"[a-z0-9']+", value.lower()):
            if token in SEED_STOPWORDS or len(token) < 4:
                continue
            terms.add(token[:-1] if len(token) > 4 and token.endswith("s") else token)
        return terms

    candidate_terms = _terms(candidate)
    excluded_terms = _terms(excluded)
    return bool(candidate_terms and excluded_terms and candidate_terms & excluded_terms)


def _canonical_seed_candidates(
    ws_session,
    narration_history: list[str],
    scenario: Scenario,
) -> list[dict[str, str]]:
    """Return graph-backed, emergent entities suitable for a callback test.

    Requiring both canonical graph membership and an exact appearance in the
    explore narration prevents the seed picker from inventing a target or
    selecting a fixed-premise alias. NPCs are preferred because they can leave
    the scene cleanly and carry unresolved intent through a washout act.
    """
    graph = getattr(ws_session, "knowledge_graph", None)
    entities = getattr(graph, "_entities", {}) or {}
    narration = "\n".join(narration_history).casefold()
    type_map = {"npc": "npc", "location": "place", "item": "item"}
    candidates: list[dict[str, str]] = []

    for entity in entities.values():
        name = str(getattr(entity, "name", "") or "").strip()
        entity_type = getattr(getattr(entity, "entity_type", None), "value", "")
        seed_type = type_map.get(str(entity_type))
        if not name or not seed_type or name.casefold() not in narration:
            continue
        if scenario.required_seed_type and seed_type != scenario.required_seed_type:
            continue
        if any(_seed_names_overlap(name, excluded) for excluded in scenario.seed_exclusions):
            continue
        properties = getattr(entity, "properties", {}) or {}
        description = str(properties.get("description", "") or "")
        normalized_description = _normalized_entity_label(description)
        if any(
            marker
            and f" {_normalized_entity_label(marker)} "
            in f" {normalized_description} "
            for marker in scenario.seed_description_exclusions
        ):
            continue
        if seed_type == "npc" and (
            properties.get("named") == "false"
            or properties.get("alive") == "false"
            or _is_generic_npc_seed_name(name)
        ):
            continue
        if seed_type == "place" and _is_generic_place_seed_name(name):
            continue
        candidates.append({
            "type": seed_type,
            "name": name,
            "description": description,
        })

    priority = {"npc": 0, "place": 1, "item": 2}
    candidates.sort(key=lambda candidate: (priority[candidate["type"]], candidate["name"].casefold()))
    named_npcs = [candidate for candidate in candidates if candidate["type"] == "npc"]
    return named_npcs or candidates


def _seed_matches(
    text: str,
    cleaned: str,
    token: str,
    seed_type: str = "",
) -> bool:
    """Does ``text`` mention the seed by cleaned name or content token?"""
    low = text.lower()
    if cleaned and cleaned in low:
        return True
    meaningful_components = [
        component
        for component in re.findall(r"[a-z0-9']+", cleaned.casefold())
        if len(component) >= 4 and component not in SEED_STOPWORDS
    ]
    if seed_type == "npc":
        token = meaningful_components[0] if meaningful_components else ""
    if token and re.search(rf"\b{re.escape(token)}\b", low):
        return True
    # Normalize punctuation so hyphenation cannot hide a leak or callback:
    # "cracked-bell" and "cracked bell" are equivalent for test purposes.
    normalized_text = re.sub(r"[^a-z0-9]+", " ", low).strip()
    normalized_cleaned = re.sub(r"[^a-z0-9]+", " ", cleaned.lower()).strip()
    normalized_token = re.sub(r"[^a-z0-9]+", " ", token.lower()).strip()
    if normalized_cleaned and normalized_cleaned in normalized_text:
        return True
    if normalized_token and normalized_token in normalized_text:
        return True
    # During the washout, aliases such as "the ring" must count as a leak of
    # "silver ring". Match every meaningful component, not only the longest.
    if seed_type == "npc":
        return False
    for component in re.findall(r"[a-z0-9']+", normalized_cleaned):
        if len(component) >= 4 and component not in SEED_STOPWORDS:
            if re.search(rf"\b{re.escape(component)}\b", normalized_text):
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
    combat_policy: str = "simulate_victory",
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
    print(
        "  Player      : "
        + (
            "Gemini 2.5 Flash-Lite + local Qwen fallback"
            if use_gemini else "Scripted"
        )
    )
    n_turns = turn_override or scenario.total_turns
    print(f"  Turns       : {n_turns}")
    print(f"  Combat      : {combat_policy}")
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
    session = TestSession(
        combat_policy=combat_policy,
        world_setting=scenario.world_setting,
    )
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
    artifact_errors: list[str] = []

    def _write_manifest(finalized: dict | None = None) -> bool:
        """(Re)write the manifest. Called at seed-pick and in finally."""
        doc = {
            "stem": stem,
            "profile": profile_label,
            "scenario": scenario.name,
            "session_id": session_id,
            "started_at": started_at,
            "n_turns": n_turns,
            "player": "gemini+ollama-fallback" if use_gemini else "scripted",
            "combat_policy": combat_policy,
            "combat_interventions": session.combat_interventions,
            "continuity_interventions": session.continuity_interventions,
            "seed": seed.__dict__ if seed else None,
            "phase_config": [
                {"name": p.name, "turn_range": list(p.turn_range)}
                for p in scenario.phases
            ],
            "seed_pick_after_turn": scenario.seed_pick_after_turn,
            "memory_silence_range": scenario.memory_silence_range,
            "creativity_gate": scenario.creativity_gate,
            "tool_coverage_gate": scenario.tool_coverage_gate,
            "opening_situation": scenario.opening_situation,
            "jsonl": str(jsonl_path),
        }
        if finalized:
            doc.update(finalized)
        try:
            manifest_path.write_text(json.dumps(doc, indent=2, default=str), encoding="utf-8")
            return True
        except Exception as e:
            print(f"  {C.YELLOW}manifest_write_failed: {e}{C.RESET}")
            artifact_errors.append(f"manifest: {e}")
            return False

    jsonl_f = open(jsonl_path, "w", encoding="utf-8")

    def _append_turn(record: dict) -> None:
        turn_records.append(record)
        try:
            jsonl_f.write(json.dumps(record, default=str) + "\n")
            jsonl_f.flush()
        except Exception as e:
            artifact_errors.append(f"turn_jsonl: {e}")

    run_error: Optional[str] = None

    try:
        for turn in range(1, n_turns + 1):
            # Pick the seed AFTER the explore phase but before the next action.
            # A narrator can legally spend the whole explore act on an
            # object-focused, anonymous-cast story (observed 4x consecutively
            # on 2026-07-23 with deepseek-v4-flash: every human stayed "the
            # courier"/"a woman" for 8 turns, so the graph held zero eligible
            # named NPCs). That is a story-lottery outcome, not a memory
            # failure — extend the explore act with forced name-eliciting
            # actions for a few turns before declaring the run untestable.
            seed_pick_retry = False
            if seed is None and turn > scenario.seed_pick_after_turn:
                seed_candidates = _canonical_seed_candidates(
                    ws_session, narration_history, scenario
                )
                retry_turns_used = turn - (scenario.seed_pick_after_turn + 1)
                if not seed_candidates:
                    if retry_turns_used >= SEED_PICK_MAX_RETRY_TURNS:
                        raise RuntimeError(
                            "No trustworthy graph-backed emergent callback seed "
                            f"existed after turn {scenario.seed_pick_after_turn} "
                            f"(+{SEED_PICK_MAX_RETRY_TURNS} nudged retry turns)"
                        )
                    seed_pick_retry = True
            if seed is None and not seed_pick_retry and turn > scenario.seed_pick_after_turn:
                seed = await player.pick_seed(
                    narration_history, scenario, seed_candidates
                )
                # Record when the pick really happened: washout redaction,
                # the silence assertions, and the explore-window assertions
                # all anchor on this, and a nudged retry moves it.
                seed.chosen_after_turn = turn - 1
                print(f"\n{C.MAGENTA}{C.BOLD}  >>> SEED IDENTIFIED <<< {C.RESET}")
                print(f"  {C.MAGENTA}type:   {seed.type}{C.RESET}")
                print(f"  {C.MAGENTA}name:   {seed.name}{C.RESET}")
                print(f"  {C.MAGENTA}reason: {seed.reason}{C.RESET}\n")
                # Write the manifest as soon as the seed exists (crash-survivable).
                _write_manifest()
                manifest_written = True

            phase = _phase_for_turn(scenario, turn)
            player_error: Optional[str] = None
            fallback_action = False
            seed_setup_recovery = False
            washout_transition_recovery = False
            callback_entry_recovery = False
            forced_action = None
            if seed_pick_retry:
                # The graph is still NPC-dry past the pick boundary. Escalate
                # beyond the soft turn-8 nudge: one narration turn of "I walk
                # toward the crowd" produces scenery, not a name exchange, so
                # demand the exchange itself.
                forced_action = (
                    "I plant myself directly in front of the nearest living "
                    "person, look them in the eye, and say, 'Your name. Your "
                    "real, exact name — I need it before we speak of anything "
                    "else.' I refuse titles, evasions, and descriptions, wait "
                    "until they state a proper name, and repeat that name back "
                    "to them twice."
                )
                seed_setup_recovery = True
            elif (
                phase.name == "explore"
                and turn >= scenario.seed_pick_after_turn - 1
                and not _canonical_seed_candidates(
                    ws_session, narration_history, scenario
                )
            ):
                forced_action = (
                    "I leave this empty place for the busiest nearby public square "
                    "or tavern, approach the first living new person I meet, introduce "
                    "myself, and ask, "
                    "'What is your exact name, and what do you want from me?' "
                    "I repeat their name back before discussing anything else."
                )
                seed_setup_recovery = True
            elif (
                seed
                and phase.name == "washout"
                and turn == max(phase.turn_range[0], seed.chosen_after_turn + 1)
            ):
                # Make the callback target genuinely off-screen before the
                # cool-off window begins. The player model may otherwise keep
                # following a turn-8 route toward the newly selected NPC even
                # after the NPC's name is redacted from its prompt.
                forced_action = (
                    "I decide this opening lead can wait. I turn away from the "
                    "people and places involved, cross into a distant district, "
                    "and pursue the newest unrelated disturbance I can find."
                )
                washout_transition_recovery = True
            elif (
                seed
                and phase.name == "callback"
                and turn == phase.turn_range[0]
            ):
                # A soft callback prompt sometimes made the player relabel the
                # NPC currently on screen as the remembered target. Start with
                # an unambiguous retrieval action: seek the real person by
                # exact name, then let the remaining callback turns improvise.
                forced_action = (
                    "I leave this scene and ask at the nearest public crossroads, "
                    f"'Where can I find {seed.name}?' I verify the description "
                    "before following the first credible direction toward them."
                )
                callback_entry_recovery = True
            try:
                action = await player.next_action(
                    last_response_text, phase, seed, forced_action=forced_action
                )
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
                    "player_continuity": getattr(player, "character_state", None),
                    "player_regenerations": getattr(player, "last_regenerations", 0),
                    "player_provider_fallbacks": getattr(
                        player, "last_provider_fallbacks", 0
                    ),
                    "elapsed": elapsed, "narrative_chars": 0,
                    "player_error": player_error, "fallback_action": fallback_action,
                    "seed_setup_recovery": seed_setup_recovery,
                    "washout_transition_recovery": washout_transition_recovery,
                    "callback_entry_recovery": callback_entry_recovery,
                    "orchestrator_error": str(e),
                    "combat_intervention": session.last_combat_intervention,
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
                "player_continuity": getattr(player, "character_state", None),
                "player_regenerations": getattr(player, "last_regenerations", 0),
                "player_provider_fallbacks": getattr(
                    player, "last_provider_fallbacks", 0
                ),
                "elapsed": elapsed, "narrative_chars": len(last_response_text),
                "player_error": player_error, "fallback_action": fallback_action,
                "seed_setup_recovery": seed_setup_recovery,
                "washout_transition_recovery": washout_transition_recovery,
                "callback_entry_recovery": callback_entry_recovery,
                "orchestrator_error": None if response else "no_response",
                "combat_intervention": session.last_combat_intervention,
                **utot, "usage_breakdown": ubreak,
            })
    except Exception as e:
        # Unexpected loop-level failure — still finalize below.
        run_error = f"{type(e).__name__}: {e}"
        print(f"  {C.RED}run_error: {run_error}{C.RESET}")
    finally:
        # FIX-3: always clean up, finalize artifacts, and run assertions over
        # whatever turns landed — even on a partial/crashed run.
        # Cross-store audit must run against the LIVE stores, before teardown.
        consistency_report = None
        try:
            consistency_report = await session.run_consistency_audit()
        except Exception as e:
            artifact_errors.append(f"consistency_audit: {type(e).__name__}: {e}")
        try:
            await session.cleanup()
        except Exception as e:
            cleanup_error = f"cleanup: {type(e).__name__}: {e}"
            print(f"  {C.YELLOW}{cleanup_error}{C.RESET}")
            run_error = run_error or cleanup_error
        try:
            jsonl_f.close()
        except Exception as e:
            artifact_errors.append(f"jsonl_close: {e}")

        events_all = usage_recorder.events()

        # Verdict (FIX-2 / FIX-4).
        callback_phase = next((p for p in scenario.phases if p.name == "callback"), None)
        cb_lo, cb_hi = callback_phase.turn_range if callback_phase else (0, 0)
        callback_fallback = any(
            r.get("fallback_action") and cb_lo <= r.get("turn", -1) <= cb_hi
            for r in turn_records
        )
        fallback_turns = [
            r["turn"] for r in turn_records
            if r.get("fallback_action") and isinstance(r.get("turn"), int)
        ]
        orchestrator_failures = [
            r["turn"] for r in turn_records
            if r.get("orchestrator_error") and isinstance(r.get("turn"), int)
        ]
        run_complete = len(turn_records) == n_turns
        if artifact_errors:
            run_error = run_error or "; ".join(artifact_errors)
        combat_policy_failed = any(
            i.get("outcome") == "failed" or not i.get("teardown_complete")
            for i in session.combat_interventions
        )
        verdict_trusted = not _seed_is_fallback(seed, scenario)

        results: list[AssertionResult] = []
        if seed is not None and session_id:
            try:
                results = run_assertions(scenario, session_id, seed)
                if consistency_report is not None:
                    results.append(AssertionResult(
                        name="cross_store_consistency",
                        passed=bool(consistency_report.get("passed")),
                        description=(
                            "WorldState, knowledge graph, ChromaDB, scene "
                            "registry, and memory tiers agree at end of run."
                        ),
                        detail=(
                            f"violations={consistency_report.get('violations') or []}"[:300]
                            + f"; coverage={consistency_report.get('coverage')}"
                            + f"; counts={consistency_report.get('counts')}"
                        ),
                    ))
                render_results(scenario, seed, results, verdict_trusted)
            except Exception as e:
                print(f"  {C.YELLOW}assertion_error: {e}{C.RESET}")
        else:
            print(f"\n  {C.YELLOW}Assertions skipped "
                  f"(seed={'yes' if seed else 'no'}, session_id={'yes' if session_id else 'no'}){C.RESET}")

        passed = sum(1 for r in results if r.passed)
        total = len(results)
        verdict = _determine_verdict(
            run_error=run_error,
            run_complete=run_complete,
            orchestrator_failures=orchestrator_failures,
            combat_policy_failed=combat_policy_failed,
            fallback_turns=fallback_turns,
            verdict_trusted=verdict_trusted,
            passed=passed,
            total=total,
        )

        report = build_report(events_all, turn_records, session_id, profile_label, scenario, n_turns)
        render_report(report, verdict, seed)

        # Ensure a manifest exists even if we crashed before seed pick.
        if not manifest_written:
            _write_manifest()
        final_manifest_ok = _write_manifest(finalized={
            "finished_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            "run_error": run_error,
            "verdict": verdict,
            "verdict_trusted": verdict_trusted,
            "callback_player_quota": callback_fallback,
            "player_fallback_turns": fallback_turns,
            "orchestrator_failure_turns": orchestrator_failures,
            "run_complete": run_complete,
            "combat_policy_failed": combat_policy_failed,
            "artifact_errors": artifact_errors,
            "assertions": [r.__dict__ for r in results],
            "assertions_passed": passed,
            "assertions_total": total,
            "consistency_audit": consistency_report,
            "report": report,
        })
        if not final_manifest_ok:
            verdict = "FAIL"
            run_error = run_error or "final manifest write failed"
            print(f"  {C.RED}Gate override: final manifest write failed{C.RESET}")
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
        "run_error": run_error,
        "combat_policy": combat_policy,
        "combat_interventions": session.combat_interventions,
        "continuity_interventions": session.continuity_interventions,
        "artifact_errors": artifact_errors,
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
        return _seed_matches(text, seed_clean, seed_token, seed.type)

    callback_phase = next(p for p in scenario.phases if p.name == "callback")
    cb_lo, cb_hi = callback_phase.turn_range
    callback_turns = [t for t in log.turns() if cb_lo <= t <= cb_hi]

    explore_phase = next(p for p in scenario.phases if p.name == "explore")
    ex_lo, ex_hi = explore_phase.turn_range
    # A nudged seed-pick retry extends discovery past the static explore
    # range; the seed's first appearance and persistence live in those extra
    # turns, so the explore window (and therefore the recall gap) follows the
    # actual pick turn.
    ex_hi = max(ex_hi, seed.chosen_after_turn)
    explore_turns = [t for t in log.turns() if ex_lo <= t <= ex_hi]

    # A deep creative campaign can have multiple named middle acts instead of
    # one generic "filler" phase. The recall gap is everything after explore
    # and before callback.
    gap_turns = [t for t in log.turns() if ex_hi < t < cb_lo]

    results: list[AssertionResult] = []
    results.extend(evaluate_narrator_prose_quality([
        (turn, log.narrator_response(turn).text) for turn in log.turns()
    ]))

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

    # 5. The canonical graph catalog retained the seed without requiring it
    # to appear in the narrator's current-scene WorldState prompt projection.
    retained_in_catalog = False
    last_gap = max(gap_turns) if gap_turns else None
    if last_gap:
        catalog = log.kg_context_for(last_gap).catalog_entities
        retained_in_catalog = _matches(json.dumps(catalog))
    results.append(AssertionResult(
        name="canonical_memory_retained_seed_through_gap",
        passed=retained_in_catalog,
        description=(
            f"Canonical graph catalog retained '{seed.name}' without prompt injection."
        ),
        detail=f"Checked turn {last_gap}",
    ))

    final_turn = max(log.turns()) if log.turns() else None
    final_catalog = (
        ((log.get(final_turn) or {}).get("knowledge_graph") or {}).get(
            "catalog_entities"
        )
        if final_turn is not None
        else []
    )
    results.append(evaluate_canonical_npc_identity(final_catalog or []))

    # 6. The seed entered a durable production persistence path in explore.
    # StateDelta is equally valid: it is bridged into the KG and vector store.
    seed_persisted = False
    persistence_sources: list[str] = []
    for t in explore_turns:
        effs = log.effects_at(t)
        for e in effs.effects:
            if _matches(json.dumps(e)):
                seed_persisted = True
                persistence_sources.append(f"turn {t} tool effect")
                break
        record = log.get(t) or {}
        state_delta = (record.get("state_delta") or {}).get("delta") or {}
        kg_state = record.get("knowledge_graph") or {}
        if (
            _matches(json.dumps(state_delta))
            and int(kg_state.get("ops_applied", 0) or 0) > 0
        ):
            seed_persisted = True
            persistence_sources.append(f"turn {t} StateDelta→KG")
        if seed_persisted:
            break
    results.append(AssertionResult(
        name="seed_entered_durable_persistence_path",
        passed=seed_persisted,
        description=f"A tool or StateDelta persisted '{seed.name}' during explore phase.",
        detail=f"sources={persistence_sources}; checked turns {explore_turns}",
    ))

    # A real long-memory result needs a washout interval. If the seed remains
    # in player actions, narrator prose, or injected KG context every few turns,
    # the callback is a recency test wearing a long-run costume.
    if scenario.memory_silence_range:
        silence_lo, silence_hi = scenario.memory_silence_range
        silence_turns = [t for t in log.turns() if silence_lo <= t <= silence_hi]
        player_leaks = [t for t in silence_turns if _matches(log.player_action(t))]
        narrator_leaks = [
            t for t in silence_turns if _matches(log.narrator_response(t).text)
        ]
        kg_leaks = [t for t in silence_turns if (
            log.kg_context_for(t).mentions(seed_clean)
            or (seed_token and log.kg_context_for(t).mentions(seed_token))
        )]
        results.extend([
            AssertionResult(
                name="player_kept_seed_out_of_memory_gap",
                passed=bool(silence_turns) and not player_leaks,
                description=f"Player did not refresh '{seed.name}' during the washout interval.",
                detail=f"turns={silence_lo}-{silence_hi}; leaks={player_leaks}",
            ),
            AssertionResult(
                name="narrator_kept_seed_out_of_memory_gap",
                passed=bool(silence_turns) and not narrator_leaks,
                description=f"Narrator did not refresh '{seed.name}' during the washout interval.",
                detail=f"turns={silence_lo}-{silence_hi}; leaks={narrator_leaks}",
            ),
            AssertionResult(
                name="kg_kept_seed_out_of_irrelevant_context",
                passed=bool(silence_turns) and not kg_leaks,
                description=(
                    f"KG retained '{seed.name}' without continuously injecting it while irrelevant."
                ),
                detail=f"turns={silence_lo}-{silence_hi}; leaks={kg_leaks}",
            ),
        ])

    if scenario.creativity_gate:
        actions = [log.player_action(t) for t in log.turns()]
        results.extend(evaluate_player_action_quality(actions))

    if scenario.tool_coverage_gate:
        turns = log.turns()
        snapshots = [log.effects_at(t) for t in turns]
        structured = _supports_structured_effect_accounting(log, turns)
        results.extend(evaluate_tool_coverage(
            [snapshot.effects for snapshot in snapshots],
            proposed_by_turn=(
                [list(snapshot.proposed or []) for snapshot in snapshots]
                if structured else None
            ),
            rejected_by_turn=(
                [list(snapshot.rejected or []) for snapshot in snapshots]
                if structured else None
            ),
            diagnostics_by_turn=(
                [
                    dict((log.get(turn) or {}).get("narration_diagnostics") or {})
                    for turn in turns
                ]
                if structured else None
            ),
        ))
        if structured:
            results.extend(evaluate_tool_omission_signals([
                (turn, dict(log.get(turn) or {})) for turn in turns
            ]))

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
    verdict_trusted = not _seed_is_fallback(seed, scenario)

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
    return 0 if verdict == "PASS" else 1


# ── Main ───────────────────────────────────────────────────────────────


async def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scenario", default="emergent_callback",
                        choices=list(SCENARIOS.keys()))
    parser.add_argument("--profile", help="Profile to use (overrides ACTIVE_PROFILE)")
    parser.add_argument("--turns", type=int, help="Override total turn count")
    parser.add_argument("--scripted", action="store_true",
                        help="Use scripted actions instead of Gemini Flash-Lite")
    parser.add_argument(
        "--combat-policy",
        choices=("simulate_victory", "fail"),
        default="simulate_victory",
        help=(
            "How a narrative/memory run handles incidental combat: end it as "
            "a simulated player victory (default), or fail the run"
        ),
    )
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
        combat_policy=args.combat_policy,
    )

    if not result.get("session_id"):
        print(f"\n  {C.RED}No session_id captured — assertions could not run{C.RESET}")
    if not result.get("seed"):
        print(f"\n  {C.RED}No seed identified — assertions could not run{C.RESET}")


    return 0 if result.get("verdict") == "PASS" else 1


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
