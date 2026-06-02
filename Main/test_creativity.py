"""
Creativity Benchmark Suite — compare local models on D&D narration quality.

Sends the same open-ended D&D prompts to every local Ollama model, then
evaluates each response with Gemini on creativity-focused dimensions.
Outputs a side-by-side comparison table.

Usage:
    python test_creativity.py                        # All models, all prompts
    python test_creativity.py --models qwen3.5:27b gemma4:26b
    python test_creativity.py --skip-eval            # Generate only, eval later
    python test_creativity.py --eval-only            # Re-evaluate saved responses
    python test_creativity.py --prompts 0 2 4        # Run specific prompts only

Requires: GEMINI_API_KEY env var (for evaluation)
"""

import asyncio
import json
import os
import re
import sys
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Optional

# Force UTF-8 on Windows
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
    print(f"\n{C.BOLD}{C.CYAN}{'=' * 60}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}  {text}{C.RESET}")
    print(f"{C.BOLD}{C.CYAN}{'=' * 60}{C.RESET}")


# =============================================================================
# D&D Narration Prompts
# =============================================================================

# Each prompt is a (name, system_instruction, user_message) tuple.
# The system sets up the narrator role; the user provides the scenario.
# Prompts are intentionally open-ended to measure creativity, not compliance.

NARRATOR_SYSTEM = """You are the Dungeon Master narrating a D&D 5e campaign.
Write vivid, immersive prose. Use sensory details — sounds, smells, textures, light.
Give NPCs distinct voices and mannerisms. Create atmosphere and forward momentum.
Do NOT use bullet points, headers, or meta-commentary. Write pure narrative prose."""

PROMPTS = [
    (
        "tavern_entrance",
        NARRATOR_SYSTEM,
        "The party pushes open the heavy oak door of the Rusty Compass tavern on a "
        "rainy evening. Describe the scene inside — the atmosphere, the patrons, "
        "any NPCs who catch their eye. Make it feel alive.",
    ),
    (
        "npc_dialogue",
        NARRATOR_SYSTEM,
        "The party approaches an old herbalist named Maren who lives alone at the "
        "edge of the Whispering Bog. She's eccentric, paranoid, and brilliant. "
        "She knows something about the missing children but won't share it easily. "
        "Write the scene — her home, her mannerisms, and the conversation as the "
        "party tries to earn her trust.",
    ),
    (
        "combat_narration",
        NARRATOR_SYSTEM,
        "The ranger rolls a natural 20 on their attack against an owlbear that "
        "just burst through the treeline. The arrow strikes true. Meanwhile, the "
        "wizard is preparing a spell and the fighter is flanking. Narrate this "
        "moment of combat — make it cinematic and visceral.",
    ),
    (
        "emotional_moment",
        NARRATOR_SYSTEM,
        "The party discovers the body of an NPC they befriended earlier — a young "
        "halfling scout named Pip who guided them through the Thornwood. He died "
        "defending the village while they were away. The villagers have laid him "
        "out in the chapel. Narrate the scene with emotional weight.",
    ),
    (
        "mystery_tension",
        NARRATOR_SYSTEM,
        "The party descends into a sealed dwarven vault that hasn't been opened in "
        "300 years. Their torchlight reveals the first chamber — but something is "
        "wrong. Build dread and mystery. What do they see, hear, and feel? Don't "
        "reveal the threat yet — just make them deeply uneasy.",
    ),
    (
        "world_description",
        NARRATOR_SYSTEM,
        "The party crests a hill at dawn and sees the Free City of Asterhold for "
        "the first time — a massive trade city built where three rivers meet, "
        "famous for its floating market and its coliseum. Describe the vista. "
        "Make them want to explore it.",
    ),
    (
        "villain_monologue",
        NARRATOR_SYSTEM,
        "The party finally confronts Lady Veshra, the campaign's antagonist — a "
        "fallen paladin who believes she's saving the world by enslaving it. She's "
        "intelligent, charismatic, and genuinely believes she's right. She addresses "
        "the party before the final battle. Write her monologue and the scene.",
    ),
]


# =============================================================================
# Ollama Runner
# =============================================================================

async def generate_response(
    model: str,
    system: str,
    prompt: str,
    temperature: float = 0.8,
    max_tokens: int = 800,
    num_ctx: int = 4096,
    think: bool = False,
) -> tuple[str, float]:
    """Send a prompt to an Ollama model and return (response_text, elapsed_seconds).

    num_ctx: context window size. Default 4096 is plenty for system (~100) +
        user prompt (~200) + response (~800). Smaller context keeps big models
        in VRAM instead of offloading to CPU.
    """
    import ollama

    client = ollama.Client()

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    kwargs = {
        "model": model,
        "messages": messages,
        "options": {
            "temperature": temperature,
            "num_predict": max_tokens,
            "num_ctx": num_ctx,
        },
    }

    # Disable thinking for Qwen models to get clean output
    if not think:
        kwargs["think"] = False

    loop = asyncio.get_event_loop()
    t0 = time.monotonic()

    def _call():
        return client.chat(**kwargs)

    response = await loop.run_in_executor(None, _call)
    elapsed = time.monotonic() - t0

    content = response.get("message", {}).get("content", "")

    # Strip leaked think tags (Qwen 3 issue)
    if "</think>" in content:
        content = content[content.find("</think>") + len("</think>"):].strip()
    elif "<think>" in content:
        idx = content.find("<think>")
        content = content[:idx].strip()

    # Strip CJK leaks
    content = re.sub(r'[\u4e00-\u9fff\u3000-\u303f\uff00-\uffef]+', '', content).strip()

    return content, elapsed


# =============================================================================
# Profile-based runner (for cloud providers via dnd_bot client factory)
# =============================================================================

async def generate_via_profile(
    profile_name: str,
    system: str,
    prompt: str,
    temperature: float = 0.8,
    max_tokens: int = 800,
) -> tuple[str, float]:
    """Generate via a profile's narrator client (uses dnd_bot client factory).

    Routes through whatever provider the profile defines (deepseek, anthropic,
    openrouter, etc.). Each call switches the active profile and resets cached
    clients so the right one is used.
    """
    # Lazy import — keeps Ollama-only runs from booting the whole config stack
    from dnd_bot.config import set_profile, get_profile
    from dnd_bot.llm.client import _create_client

    profile = set_profile(profile_name)
    narrator = profile.narrator

    client = _create_client(narrator.provider, narrator.model, context_size=narrator.context_size)

    messages = [
        {"role": "system", "content": system},
        {"role": "user", "content": prompt},
    ]

    t0 = time.monotonic()
    response = await client.chat(
        messages=messages,
        temperature=temperature,
        max_tokens=max_tokens,
        # think defaults to provider-appropriate behavior; DeepSeekClient
        # disables thinking unless think=True is passed explicitly
    )
    elapsed = time.monotonic() - t0

    content = response.content if hasattr(response, "content") else str(response)
    return content, elapsed


# =============================================================================
# Gemini Evaluator
# =============================================================================

EVAL_SYSTEM = """You evaluate D&D narrator prose quality. You will receive a narration prompt and a model's response.

Score each dimension 1-5:
- 1 = Poor (generic, flat, or broken)
- 2 = Below average (functional but uninspired)
- 3 = Competent (solid, meets expectations)
- 4 = Good (vivid, engaging, some memorable moments)
- 5 = Excellent (exceptional prose that pulls you in)

## DIMENSIONS

**prose_quality**: Sentence variety, word choice, rhythm. Does it read well aloud? Is the language precise or generic? Are there cliches, or fresh descriptions?

**creativity**: Originality of details, unexpected elements, unique imagery. Does it surprise you or feel templated? Are the sensory details specific or vague?

**character_voice**: Do NPCs sound like distinct people? Is dialogue natural? Do characters have mannerisms, speech patterns, personality? (Score 3 if the prompt doesn't involve dialogue.)

**atmosphere**: Does the writing create a mood? Can you feel the space — the temperature, the light, the tension? Does it build immersion or just describe facts?

**dnd_authenticity**: Does it feel like a real D&D session? Does it respect genre conventions (fantasy, adventure) while staying fresh? Does it invite player action? Does it leave room for the players to act rather than railroading?

Respond with ONLY valid JSON:
{
  "prose_quality": {"score": N, "justification": "one sentence"},
  "creativity": {"score": N, "justification": "one sentence"},
  "character_voice": {"score": N, "justification": "one sentence"},
  "atmosphere": {"score": N, "justification": "one sentence"},
  "dnd_authenticity": {"score": N, "justification": "one sentence"},
  "standout_line": "quote the single best sentence from the response",
  "weakest_aspect": "one sentence on what could improve"
}"""


async def evaluate_response(
    gemini_client,
    prompt_name: str,
    prompt_text: str,
    response_text: str,
) -> dict:
    """Evaluate a single response with Gemini. Returns parsed scores dict."""
    eval_prompt = (
        f"## Prompt Given to Model\n"
        f"**Scenario:** {prompt_name}\n"
        f"{prompt_text}\n\n"
        f"## Model's Response\n"
        f"{response_text}\n\n"
        f"Evaluate the response above."
    )

    try:
        raw = await gemini_client.chat(
            messages=[{"role": "user", "content": eval_prompt}],
            system=EVAL_SYSTEM,
            temperature=0.1,
            max_tokens=1024,
            json_mode=True,
        )

        text = raw.strip()
        # Strip markdown fences
        fence = re.match(r"^```(?:json)?\s*\n?(.*?)```\s*$", text, re.DOTALL)
        if fence:
            text = fence.group(1).strip()

        return json.loads(text)

    except Exception as e:
        return {"error": str(e)}


# =============================================================================
# Benchmark Runner
# =============================================================================

@dataclass
class ModelResult:
    model: str
    prompt_name: str
    response: str
    elapsed: float
    evaluation: Optional[dict] = None


def get_default_models() -> list[str]:
    """Get models worth benchmarking (skip sub-2B)."""
    import ollama
    try:
        models = ollama.list().get("models", [])
        names = [m["name"] if isinstance(m, dict) else m.model for m in models]
        # Filter out tiny models and voice variants
        skip = {"qwen3.5:0.8b", "qwen3.5:2b", "qwen3.5:4b-voice"}
        return sorted([n for n in names if n not in skip])
    except Exception:
        return []


async def run_benchmark(
    targets: list[dict],
    prompt_indices: Optional[list[int]] = None,
    skip_eval: bool = False,
    eval_only: bool = False,
    temperature: float = 0.8,
    num_ctx: int = 4096,
) -> list[ModelResult]:
    """Run the full benchmark: generate + evaluate.

    targets: list of {"label": str, "kind": "ollama"|"profile", "value": str}.
    "ollama" runs the local Ollama path, "profile" routes through the dnd_bot
    client factory (DeepSeek / Anthropic / OpenRouter / etc.).
    """

    prompts = PROMPTS
    if prompt_indices:
        prompts = [PROMPTS[i] for i in prompt_indices if i < len(PROMPTS)]

    log_dir = Path("data/creativity_logs")
    log_dir.mkdir(parents=True, exist_ok=True)

    results: list[ModelResult] = []

    # ── Phase 1: Generate responses ──────────────────────────────────────
    if not eval_only:
        header("GENERATION PHASE")
        print(f"  Targets: {len(targets)}")
        print(f"  Prompts: {len(prompts)}")
        print(f"  Total generations: {len(targets) * len(prompts)}")

        for target in targets:
            label = target["label"]
            kind = target["kind"]
            value = target["value"]

            tag = f"[{kind}] {label}"
            print(f"\n  {C.BOLD}{C.MAGENTA}>>> {tag}{C.RESET}")

            for prompt_name, system, user_msg in prompts:
                sys.stdout.write(f"    {prompt_name}... ")
                sys.stdout.flush()

                try:
                    if kind == "ollama":
                        response, elapsed = await generate_response(
                            value, system, user_msg,
                            temperature=temperature, num_ctx=num_ctx,
                        )
                    elif kind == "profile":
                        response, elapsed = await generate_via_profile(
                            value, system, user_msg,
                            temperature=temperature,
                        )
                    else:
                        raise ValueError(f"Unknown target kind: {kind}")
                    word_count = len(response.split())
                    print(f"{C.GREEN}{elapsed:.1f}s, {word_count} words{C.RESET}")

                    results.append(ModelResult(
                        model=label,
                        prompt_name=prompt_name,
                        response=response,
                        elapsed=elapsed,
                    ))

                except Exception as e:
                    print(f"{C.RED}FAILED: {e}{C.RESET}")
                    results.append(ModelResult(
                        model=label,
                        prompt_name=prompt_name,
                        response=f"ERROR: {e}",
                        elapsed=0,
                    ))

        # Save responses before evaluation (in case eval fails/quota)
        _save_results(results, log_dir / "creativity_latest.json")
        print(f"\n  {C.DIM}Responses saved to data/creativity_logs/creativity_latest.json{C.RESET}")

    else:
        # Load previous responses
        latest = log_dir / "creativity_latest.json"
        if not latest.exists():
            print(f"  {C.RED}No saved responses found. Run without --eval-only first.{C.RESET}")
            return []
        results = _load_results(latest)
        print(f"  {C.DIM}Loaded {len(results)} saved responses{C.RESET}")

    if skip_eval:
        print(f"\n  {C.YELLOW}Skipping evaluation (--skip-eval){C.RESET}")
        _print_generation_summary(results)
        return results

    # ── Phase 2: Evaluate with Gemini ────────────────────────────────────
    header("EVALUATION PHASE")

    api_key = os.environ.get("GEMINI_API_KEY", "")
    if not api_key:
        print(f"  {C.RED}GEMINI_API_KEY not set — skipping evaluation{C.RESET}")
        _print_generation_summary(results)
        return results

    # Reuse the GeminiClient from test_eval
    sys.path.insert(0, str(Path(__file__).parent))
    from test_eval import GeminiClient
    gemini = GeminiClient(api_key=api_key, model="gemini-2.5-flash")

    evaluated = 0
    total = len([r for r in results if not r.response.startswith("ERROR")])

    for result in results:
        if result.response.startswith("ERROR"):
            continue
        if result.evaluation and "error" not in result.evaluation:
            evaluated += 1
            continue  # Already evaluated (from loaded file)

        evaluated += 1
        sys.stdout.write(
            f"  [{evaluated}/{total}] {result.model} / {result.prompt_name}... "
        )
        sys.stdout.flush()

        result.evaluation = await evaluate_response(
            gemini,
            result.prompt_name,
            # Find the original prompt text
            next(p[2] for p in PROMPTS if p[0] == result.prompt_name),
            result.response,
        )

        if "error" in (result.evaluation or {}):
            print(f"{C.RED}eval failed{C.RESET}")
        else:
            avg = _avg_score(result.evaluation)
            print(f"{C.GREEN}{avg:.1f}/5{C.RESET}")

        # Rate limit courtesy
        await asyncio.sleep(1)

    # Save with evaluations
    ts = int(time.time())
    _save_results(results, log_dir / "creativity_latest.json")
    _save_results(results, log_dir / f"creativity_{ts}.json")

    # ── Phase 3: Print report ────────────────────────────────────────────
    _print_report(results)

    return results


# =============================================================================
# Scoring Helpers
# =============================================================================

DIMENSIONS = ["prose_quality", "creativity", "character_voice", "atmosphere", "dnd_authenticity"]


def _avg_score(evaluation: Optional[dict]) -> float:
    if not evaluation or "error" in evaluation:
        return 0.0
    scores = []
    for dim in DIMENSIONS:
        dim_data = evaluation.get(dim, {})
        if isinstance(dim_data, dict) and "score" in dim_data:
            scores.append(dim_data["score"])
    return sum(scores) / len(scores) if scores else 0.0


def _dim_score(evaluation: Optional[dict], dim: str) -> float:
    if not evaluation or "error" in evaluation:
        return 0.0
    dim_data = evaluation.get(dim, {})
    return dim_data.get("score", 0) if isinstance(dim_data, dict) else 0.0


# =============================================================================
# Report Printing
# =============================================================================

def _print_generation_summary(results: list[ModelResult]):
    """Print summary when evaluation was skipped."""
    header("GENERATION SUMMARY")
    models = sorted(set(r.model for r in results))
    for model in models:
        model_results = [r for r in results if r.model == model]
        avg_time = sum(r.elapsed for r in model_results) / len(model_results)
        avg_words = sum(len(r.response.split()) for r in model_results) / len(model_results)
        print(f"  {model:<30} avg {avg_time:.1f}s, {avg_words:.0f} words")


def _print_report(results: list[ModelResult]):
    """Print the full comparison report."""
    header("CREATIVITY BENCHMARK RESULTS")

    models = sorted(set(r.model for r in results))

    # ── Per-model averages ───────────────────────────────────────────────
    print(f"\n  {C.BOLD}{'Model':<30} {'Prose':>6} {'Creat':>6} {'Voice':>6} {'Atmos':>6} {'D&D':>6} {'AVG':>6} {'Time':>6}{C.RESET}")
    print(f"  {'─' * 84}")

    model_avgs = {}  # model -> avg score

    for model in models:
        model_results = [r for r in results if r.model == model and r.evaluation and "error" not in r.evaluation]
        if not model_results:
            print(f"  {model:<30} {'(no eval data)':>42}")
            continue

        dim_avgs = {}
        for dim in DIMENSIONS:
            scores = [_dim_score(r.evaluation, dim) for r in model_results]
            dim_avgs[dim] = sum(scores) / len(scores) if scores else 0

        overall = sum(dim_avgs.values()) / len(dim_avgs) if dim_avgs else 0
        model_avgs[model] = overall

        avg_time = sum(r.elapsed for r in model_results) / len(model_results)

        color = C.GREEN if overall >= 4.0 else C.YELLOW if overall >= 3.0 else C.RED
        print(
            f"  {model:<30} "
            f"{dim_avgs.get('prose_quality', 0):>5.1f} "
            f"{dim_avgs.get('creativity', 0):>6.1f} "
            f"{dim_avgs.get('character_voice', 0):>6.1f} "
            f"{dim_avgs.get('atmosphere', 0):>6.1f} "
            f"{dim_avgs.get('dnd_authenticity', 0):>5.1f} "
            f"{color}{overall:>6.1f}{C.RESET} "
            f"{C.DIM}{avg_time:>5.1f}s{C.RESET}"
        )

    # ── Ranking ──────────────────────────────────────────────────────────
    if model_avgs:
        print(f"\n  {C.BOLD}RANKING{C.RESET}")
        ranked = sorted(model_avgs.items(), key=lambda x: x[1], reverse=True)
        for i, (model, score) in enumerate(ranked, 1):
            medal = {1: "\U0001f947", 2: "\U0001f948", 3: "\U0001f949"}.get(i, "  ")
            color = C.GREEN if i == 1 else C.YELLOW if i <= 3 else C.RESET
            print(f"  {medal} {color}#{i} {model:<30} {score:.2f}/5{C.RESET}")

    # ── Per-prompt breakdown ─────────────────────────────────────────────
    prompt_names = sorted(set(r.prompt_name for r in results))
    print(f"\n  {C.BOLD}PER-PROMPT BREAKDOWN{C.RESET}")

    for prompt_name in prompt_names:
        print(f"\n  {C.CYAN}{prompt_name}{C.RESET}")
        prompt_results = [
            r for r in results
            if r.prompt_name == prompt_name and r.evaluation and "error" not in r.evaluation
        ]
        prompt_results.sort(key=lambda r: _avg_score(r.evaluation), reverse=True)

        for r in prompt_results:
            avg = _avg_score(r.evaluation)
            color = C.GREEN if avg >= 4.0 else C.YELLOW if avg >= 3.0 else C.RED
            standout = (r.evaluation or {}).get("standout_line", "")
            if standout and len(standout) > 80:
                standout = standout[:80] + "..."
            print(f"    {r.model:<30} {color}{avg:.1f}{C.RESET}  {C.DIM}{standout}{C.RESET}")

    # ── Standout lines ───────────────────────────────────────────────────
    print(f"\n  {C.BOLD}BEST LINES PER MODEL{C.RESET}")
    for model in models:
        model_results = [
            r for r in results
            if r.model == model and r.evaluation and "error" not in r.evaluation
        ]
        if not model_results:
            continue
        # Pick the highest-scored response's standout line
        best = max(model_results, key=lambda r: _avg_score(r.evaluation))
        line = (best.evaluation or {}).get("standout_line", "(none)")
        print(f"  {C.MAGENTA}{model}{C.RESET}")
        print(f"    {C.DIM}\"{line}\"{C.RESET}")


# =============================================================================
# Persistence
# =============================================================================

def _save_results(results: list[ModelResult], path: Path):
    data = [asdict(r) for r in results]
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)


def _load_results(path: Path) -> list[ModelResult]:
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    return [
        ModelResult(
            model=d["model"],
            prompt_name=d["prompt_name"],
            response=d["response"],
            elapsed=d["elapsed"],
            evaluation=d.get("evaluation"),
        )
        for d in data
    ]


# =============================================================================
# CLI
# =============================================================================

async def main():
    import argparse

    all_models = get_default_models()

    parser = argparse.ArgumentParser(description="D&D creativity benchmark for local + cloud narrators")
    parser.add_argument("--models", nargs="+", default=None,
                        help=f"Ollama models to test. Available: {', '.join(all_models)}")
    parser.add_argument("--profiles", nargs="+", default=None,
                        help="Named profiles from config/profiles.yaml (e.g. deepseek_v4_pro). "
                             "Routes through the dnd_bot client factory — works with any provider "
                             "(deepseek, anthropic, openrouter, etc.).")
    parser.add_argument("--prompts", nargs="+", type=int, default=None,
                        help=f"Prompt indices to run (0-{len(PROMPTS)-1}). Default: all")
    parser.add_argument("--skip-eval", action="store_true",
                        help="Generate responses only, skip Gemini evaluation")
    parser.add_argument("--eval-only", action="store_true",
                        help="Re-evaluate previously saved responses")
    parser.add_argument("--temperature", type=float, default=0.8,
                        help="Sampling temperature (default: 0.8)")
    parser.add_argument("--num-ctx", type=int, default=4096,
                        help="Ollama context window size (default: 4096). Ignored for profile targets.")
    parser.add_argument("--list-prompts", action="store_true",
                        help="List available prompts and exit")
    args = parser.parse_args()

    if args.list_prompts:
        print(f"\n  Available prompts:")
        for i, (name, _, text) in enumerate(PROMPTS):
            preview = text[:80].replace("\n", " ") + "..."
            print(f"  [{i}] {name:<25} {preview}")
        return

    # Build the targets list. If neither --models nor --profiles given, default
    # to all local Ollama models (preserves prior behavior).
    targets: list[dict] = []

    if args.profiles:
        for p in args.profiles:
            targets.append({"label": p, "kind": "profile", "value": p})

    models = args.models if args.models is not None else (all_models if not args.profiles else [])
    if models:
        available = set(get_default_models())
        for m in models:
            if m not in available:
                print(f"  {C.YELLOW}Warning: {m} not found in Ollama. Will attempt anyway.{C.RESET}")
            targets.append({"label": m, "kind": "ollama", "value": m})

    if not targets:
        print(f"  {C.RED}No targets. Pass --models, --profiles, or have Ollama running.{C.RESET}")
        return

    await run_benchmark(
        targets=targets,
        prompt_indices=args.prompts,
        skip_eval=args.skip_eval,
        eval_only=args.eval_only,
        temperature=args.temperature,
        num_ctx=args.num_ctx,
    )


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print(f"\n{C.YELLOW}Interrupted.{C.RESET}")
