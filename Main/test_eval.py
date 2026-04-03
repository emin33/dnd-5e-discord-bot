"""
Automated Narrator Evaluation - Gemini-powered QA for the D&D bot.

Uses Gemini 2.5 Flash as a player agent and Gemini 3 Flash as an evaluator
to test narrator quality: scene consistency, memory continuity, hallucination
detection, and NPC tracking. The actual narration runs on Groq qwen3:32b.

Usage:
    python test_eval.py                         # 12 turns, default persona
    python test_eval.py --turns 20              # More turns
    python test_eval.py --persona aggressive    # Different play style
    python test_eval.py --no-fallback           # Disable Ollama fallback (Groq-only)
    python test_eval.py --seed tavern_explore   # Run scenario first, then eval

Requires: GEMINI_API_KEY env var (or in .env file)
Install:  pip install google-genai
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

# Must be in project root for .env
os.chdir(Path(__file__).parent)

# Load .env before anything else
from dotenv import load_dotenv
load_dotenv()


# =============================================================================
# Colors (borrowed from test_harness)
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
# Gemini Client
# =============================================================================

class GeminiClient:
    """Thin async wrapper over google.genai SDK."""

    def __init__(self, api_key: str, model: str = "gemini-2.5-flash"):
        from google import genai
        self._client = genai.Client(api_key=api_key)
        self.model = model

    async def chat(
        self,
        messages: list[dict],
        temperature: float = 0.7,
        max_tokens: int = 1024,
        json_mode: bool = False,
        system: Optional[str] = None,
    ) -> str:
        """Send messages and return text response.

        Args:
            messages: List of {"role": "user"/"assistant", "content": str}
            temperature: Sampling temperature
            max_tokens: Max output tokens
            json_mode: If True, request JSON output
            system: System instruction (separate from messages)
        """
        from google.genai import types

        # Build contents list for Gemini
        contents = []
        for msg in messages:
            role = "model" if msg["role"] in ("assistant", "system") else "user"
            # Gemini doesn't allow consecutive same-role messages,
            # so merge system messages into user context
            if msg["role"] == "system" and not system:
                system = msg["content"]
                continue
            contents.append(types.Content(
                role=role,
                parts=[types.Part(text=msg["content"])],
            ))

        # Ensure we have at least one user message
        if not contents:
            contents.append(types.Content(
                role="user",
                parts=[types.Part(text="Please respond.")],
            ))

        config = types.GenerateContentConfig(
            system_instruction=system,
            temperature=temperature,
            max_output_tokens=max_tokens,
        )
        if json_mode:
            config.response_mime_type = "application/json"

        # Retry with backoff
        last_err = None
        for attempt in range(3):
            try:
                loop = asyncio.get_event_loop()
                response = await loop.run_in_executor(
                    None,
                    lambda: self._client.models.generate_content(
                        model=self.model,
                        contents=contents,
                        config=config,
                    ),
                )
                return response.text or ""
            except Exception as e:
                last_err = e
                err_str = str(e)
                if "429" in err_str or "quota" in err_str.lower() or "503" in err_str or "UNAVAILABLE" in err_str:
                    wait = 2 ** (attempt + 1)
                    print(f"  {C.DIM}[Gemini {err_str[:30]}..., retrying in {wait}s]{C.RESET}")
                    await asyncio.sleep(wait)
                else:
                    raise
        raise last_err


# =============================================================================
# Player Agent
# =============================================================================

PLAYER_SYSTEM = """You are a D&D player controlling a character named {name}, a {race} {class_name}.

Personality: {personality}
Goals: {goals}

Your job is to generate ONE player action per turn. Output ONLY the action text — no quotes, no commentary, no OOC talk. Write in first person ("I do X").

IMPORTANT RULES:
- Vary your action types: social, exploration, investigation, combat, skill use
- Keep actions concise (1-2 sentences max)
- Stay in character
- React naturally to what happened last turn
"""

PHASE_PLANT = """CURRENT OBJECTIVE: Establish memorable details this turn.
- Ask NPCs their names, note specific landmarks, establish facts
- Create details that can be referenced later to test the DM's memory
- Examples: "I ask the innkeeper what his name is", "I examine the crest on the shield above the fireplace"
"""

PHASE_NORMAL = """CURRENT OBJECTIVE: Play naturally. Mix up your action types.
Try something different from your last few actions. Consider:
- Talking to NPCs, investigating objects, moving to new areas
- Attempting skill checks, examining your surroundings
- Reacting to what the DM described
"""

PHASE_RECALL = """CURRENT OBJECTIVE: Reference earlier details to test the DM's memory.
You previously established these details:
{planted_details}

Reference one of these BY NAME in your action. Examples:
- "I go back to talk to [NPC name from earlier]"
- "I ask about [event/location from earlier]"
- "I look for the [item mentioned earlier]"
"""


@dataclass
class PlayerPersona:
    name: str = "Kael Windrunner"
    race: str = "elf"
    class_name: str = "ranger"
    personality: str = "Cautious but curious. Observant. Asks questions before acting."
    goals: str = "Investigate trouble in the nearby forest. Learn about the local area."


PERSONAS = {
    "default": PlayerPersona(),
    "aggressive": PlayerPersona(
        personality="Impulsive and bold. Charges in, asks questions later. Loves a fight.",
        goals="Find the biggest threat and deal with it. Prove your strength.",
    ),
    "social": PlayerPersona(
        personality="Charming and talkative. Befriends everyone. Collects stories and rumors.",
        goals="Make allies, learn secrets, build a network of contacts.",
    ),
    "cautious": PlayerPersona(
        personality="Paranoid and thorough. Checks everything for traps. Trusts no one.",
        goals="Survive. Gather information carefully. Never walk into an ambush.",
    ),
}


class PlayerAgent:
    """Generates contextual D&D player actions via Gemini."""

    def __init__(self, client: GeminiClient, persona: PlayerPersona):
        self.client = client
        self.persona = persona
        self.history: list[dict] = []  # {"action": str, "narrative_snippet": str}
        self.planted_details: list[str] = []

    def _get_phase(self, turn: int, total: int) -> str:
        if turn <= 4:
            return "plant"
        elif turn > total - 3:
            return "recall"
        return "normal"

    def _build_phase_instruction(self, phase: str) -> str:
        if phase == "plant":
            return PHASE_PLANT
        elif phase == "recall":
            details = "\n".join(f"- {d}" for d in self.planted_details) or "- (none recorded)"
            return PHASE_RECALL.format(planted_details=details)
        return PHASE_NORMAL

    async def generate_action(self, turn: int, total: int) -> str:
        phase = self._get_phase(turn, total)
        system = PLAYER_SYSTEM.format(**{
            "name": self.persona.name,
            "race": self.persona.race,
            "class_name": self.persona.class_name,
            "personality": self.persona.personality,
            "goals": self.persona.goals,
        })

        # Build conversation context
        messages = []
        for entry in self.history[-10:]:  # Last 10 turns for context
            messages.append({"role": "user", "content": f"[Your action]: {entry['action']}"})
            messages.append({"role": "assistant", "content": f"[DM response]: {entry['narrative_snippet']}"})

        # Current turn instruction with dedup
        phase_instruction = self._build_phase_instruction(phase)
        recent_actions = [e["action"] for e in self.history[-3:]]
        dedup = ""
        if recent_actions:
            dedup = "\n\nDO NOT repeat these recent actions:\n" + "\n".join(f"- {a}" for a in recent_actions)

        messages.append({
            "role": "user",
            "content": f"Turn {turn}/{total}. {phase_instruction}{dedup}\nGenerate your next action:",
        })

        action = await self.client.chat(
            messages=messages,
            system=system,
            temperature=0.8,
            max_tokens=150,
        )
        return action.strip().strip('"').strip("'")

    def record_turn(self, action: str, narrative: str, phase: str):
        snippet = narrative[:300] if narrative else "(no response)"
        self.history.append({"action": action, "narrative_snippet": snippet})

        # Auto-extract planted details from Phase 1 narratives
        if phase == "plant" and narrative:
            # Look for NPC names (capitalized words after common introductions)
            for pattern in [
                r'"(?:Name\'s|I\'m|They call me|I am) (\w+)"',
                r'"(\w+),"? (?:he|she|they) (?:say|said|answer|replied)',
                r'introduces (?:himself|herself|themselves) as (\w+)',
            ]:
                matches = re.findall(pattern, narrative, re.IGNORECASE)
                for m in matches:
                    if m and len(m) > 2 and m not in ("The", "You", "Your", "This"):
                        detail = f"NPC named '{m}'"
                        if detail not in self.planted_details:
                            self.planted_details.append(detail)

            # Look for location/landmark names
            for pattern in [
                r'(?:called|named|known as) ["\']?the ([A-Z][\w\s]+?)["\']?[,.\s]',
                r'the ([A-Z][\w]+ (?:Tavern|Inn|Gate|Bridge|Tower|Temple|Market))',
            ]:
                matches = re.findall(pattern, narrative)
                for m in matches:
                    detail = f"Location: '{m.strip()}'"
                    if detail not in self.planted_details:
                        self.planted_details.append(detail)


# =============================================================================
# Narrator Evaluator
# =============================================================================

EVAL_SYSTEM = """You are a D&D narrator quality evaluator. You have the FULL conversation transcript and review each new DM response for factual consistency and quality.

You will receive:
1. The full conversation history (previous turns)
2. The current player action
3. The DM's narrator response to evaluate
4. Any mechanical results (dice rolls, checks)

Score each dimension 1-5:
- 1 = Serious problems
- 2 = Notable issues
- 3 = Acceptable, meets expectations
- 4 = Good, minor or no issues
- 5 = Excellent, exceptional quality

## EVALUATION DIMENSIONS

**scene_consistency**: Is the physical setting consistent? If the party is in a tavern, they shouldn't suddenly be at a crossroads without transition. Are objects, furniture, and environmental details stable across turns? Score 3 if there's no scene to compare against yet.

**contradiction_free**: Does the narrator contradict ESTABLISHED FACTS from earlier in the conversation? Examples of real contradictions:
- An NPC who died earlier reappears alive
- The narrator says it's raining when it was clearly sunny moments ago with no transition
- A locked door that was already opened is described as locked again
- The narrator says the player did something they didn't do in THIS turn's action
NOTE: The narrator SHOULD embellish player actions with physical details (describing them crouching, scanning, drawing a weapon). That is GOOD narration, NOT a contradiction. Only flag things that conflict with established facts.

**npc_continuity**: Do NPCs maintain consistent names, personalities, and status across turns? Is a friendly NPC still friendly (unless something changed)? Are NPCs in the right location? If the bartender was behind the bar, he shouldn't be outside unless he walked there.

**narrative_quality**: Is the prose vivid and engaging? Does it use sensory details? Does it give NPCs distinct voices? Does it create forward momentum that invites the player to act? Does it avoid repetitive phrasing across turns?

IMPORTANT:
- Score 3 is baseline "fine, no issues." Reserve 1-2 for actual problems, 4-5 for good/great.
- On early turns (1-3), score scene_consistency and contradiction_free as 4 unless there's a clear problem — there's not much history to contradict yet.
- Flag SPECIFIC issues with a brief description of the contradiction, NOT just quoted text.
- The narrator adding physical flavor to actions (crouching, scanning, gripping a weapon) is NOT a contradiction.

Respond with ONLY valid JSON matching this exact structure:
{
  "scene_consistency": {"score": N, "justification": "...", "flagged_issues": []},
  "contradiction_free": {"score": N, "justification": "...", "flagged_issues": []},
  "npc_continuity": {"score": N, "justification": "...", "flagged_issues": []},
  "narrative_quality": {"score": N, "justification": "...", "flagged_issues": []},
  "overall_notes": "..."
}"""


@dataclass
class DimensionScore:
    score: int = 0
    justification: str = ""
    flagged_issues: list[str] = field(default_factory=list)


@dataclass
class EvalResult:
    scene_consistency: DimensionScore = field(default_factory=DimensionScore)
    contradiction_free: DimensionScore = field(default_factory=DimensionScore)
    npc_continuity: DimensionScore = field(default_factory=DimensionScore)
    narrative_quality: DimensionScore = field(default_factory=DimensionScore)
    overall_notes: str = ""
    parse_error: str = ""

    @property
    def average_score(self) -> float:
        scores = [
            self.scene_consistency.score,
            self.contradiction_free.score,
            self.npc_continuity.score,
            self.narrative_quality.score,
        ]
        valid = [s for s in scores if s > 0]
        return sum(valid) / len(valid) if valid else 0.0

    @property
    def all_flagged_issues(self) -> list[str]:
        issues = []
        for dim in [self.scene_consistency, self.contradiction_free,
                     self.npc_continuity, self.narrative_quality]:
            issues.extend(dim.flagged_issues)
        return issues


class NarratorEvaluator:
    """Grades narrator responses on a 5-dimension rubric via Gemini."""

    def __init__(self, client: GeminiClient):
        self.client = client

    async def evaluate(
        self,
        turn_history: list[dict],
        current_action: str,
        narrator_response: str,
        mechanical_result: Optional[dict] = None,
    ) -> EvalResult:
        # Build history context
        history_lines = []
        for entry in turn_history:
            history_lines.append(f"Player: {entry['action']}")
            history_lines.append(f"DM: {entry['narrative_snippet']}")
            history_lines.append("")

        history_text = "\n".join(history_lines) if history_lines else "(first turn)"

        mechanics_text = json.dumps(mechanical_result) if mechanical_result else "None"

        prompt = f"""## Conversation History
{history_text}

## Current Turn
**Player action:** {current_action}
**Mechanical result:** {mechanics_text}
**DM narrator response to evaluate:**
{narrator_response}

Evaluate the DM's response above. Return ONLY the JSON scores."""

        try:
            raw = await self.client.chat(
                messages=[{"role": "user", "content": prompt}],
                system=EVAL_SYSTEM,
                temperature=0.1,
                max_tokens=2048,
                json_mode=True,
            )
            return self._parse_eval(raw)
        except Exception as e:
            return EvalResult(parse_error=f"Evaluator call failed: {e}")

    def _parse_eval(self, raw: str) -> EvalResult:
        """Parse JSON evaluation, with fallbacks for common formatting issues."""
        text = raw.strip()

        # Strip markdown fences
        fence_match = re.match(r"^```(?:json)?\s*\n?(.*?)```\s*$", text, re.DOTALL)
        if fence_match:
            text = fence_match.group(1).strip()

        try:
            data = json.loads(text)
        except json.JSONDecodeError as e:
            return EvalResult(parse_error=f"JSON parse error: {e}\nRaw: {text[:200]}")

        result = EvalResult(overall_notes=data.get("overall_notes", ""))

        for dim_name in ["scene_consistency", "contradiction_free",
                         "npc_continuity", "narrative_quality"]:
            dim_data = data.get(dim_name, {})
            if isinstance(dim_data, dict):
                setattr(result, dim_name, DimensionScore(
                    score=dim_data.get("score", 0),
                    justification=dim_data.get("justification", ""),
                    flagged_issues=dim_data.get("flagged_issues", []),
                ))

        return result


# =============================================================================
# Eval Session
# =============================================================================

@dataclass
class TurnRecord:
    turn: int
    phase: str
    action: str
    narrative: str
    mechanics: Optional[dict]
    combat_triggered: bool
    evaluation: Optional[dict]
    elapsed_game: float
    elapsed_eval: float
    harness_issues: list[dict] = field(default_factory=list)
    memory_state: dict = field(default_factory=dict)


class EvalSession:
    """Orchestrates automated evaluation runs."""

    def __init__(
        self,
        total_turns: int = 12,
        persona_name: str = "default",
        gemini_player_model: str = "gemini-2.5-flash",
        gemini_eval_model: str = "gemini-3-flash-preview",
        seed_scenario: Optional[str] = None,
    ):
        self.total_turns = total_turns
        self.persona_name = persona_name
        self.seed_scenario = seed_scenario

        api_key = os.environ.get("GEMINI_API_KEY", "")
        if not api_key:
            raise ValueError("GEMINI_API_KEY environment variable must be set")

        player_client = GeminiClient(api_key=api_key, model=gemini_player_model)
        eval_client = GeminiClient(api_key=api_key, model=gemini_eval_model)

        persona = PERSONAS.get(persona_name, PERSONAS["default"])
        self.player = PlayerAgent(player_client, persona)
        self.evaluator = NarratorEvaluator(eval_client)
        self.turns: list[TurnRecord] = []
        self.started_at = ""
        self.finished_at = ""
        self._gemini_player_model = gemini_player_model
        self._gemini_eval_model = gemini_eval_model

    async def run(self):
        from test_harness import TestSession, SCENARIOS

        header("NARRATOR EVALUATION")
        print(f"  Player: {self._gemini_player_model}")
        print(f"  Evaluator: {self._gemini_eval_model}")
        print(f"  Persona: {self.persona_name}")
        print(f"  Turns: {self.total_turns}")
        if self.seed_scenario:
            print(f"  Seed scenario: {self.seed_scenario}")

        self.started_at = time.strftime("%Y-%m-%d %H:%M:%S")
        session = TestSession()

        try:
            await session.setup()

            # Run seed scenario first if specified
            if self.seed_scenario and self.seed_scenario in SCENARIOS:
                header(f"SEED SCENARIO: {self.seed_scenario}")
                for action in SCENARIOS[self.seed_scenario]:
                    print(f"  {C.DIM}[seed] {action}{C.RESET}")
                    resp = await session.send_action(action)
                    if resp and resp.narrative:
                        self.player.record_turn(action, resp.narrative, "seed")
                    await asyncio.sleep(1)

            # Main evaluation loop
            header("EVALUATION TURNS")
            prev_issue_count = len(session.all_issues)

            for turn in range(1, self.total_turns + 1):
                phase = self.player._get_phase(turn, self.total_turns)
                phase_label = {"plant": "🌱", "normal": "▶", "recall": "🔍"}[phase]

                # 1. Player generates action
                t0 = time.monotonic()
                try:
                    action = await self.player.generate_action(turn, self.total_turns)
                except Exception as e:
                    print(f"  {C.RED}Player agent failed: {e}{C.RESET}")
                    continue

                print(f"\n{C.BOLD}  Turn {turn}/{self.total_turns} {phase_label} {C.CYAN}\"{action}\"{C.RESET}")

                # 2. Game processes the action
                t1 = time.monotonic()
                response = await session.send_action(action)
                t2 = time.monotonic()

                if not response or not response.narrative:
                    print(f"  {C.RED}No response{C.RESET}")
                    continue

                narrative = response.narrative
                mechanics = response.mechanical_result

                # 3. Record for player history
                self.player.record_turn(action, narrative, phase)

                # 4. Evaluator grades the response
                t3 = time.monotonic()
                eval_result = await self.evaluator.evaluate(
                    turn_history=self.player.history[:-1],  # All except current
                    current_action=action,
                    narrator_response=narrative,
                    mechanical_result=mechanics,
                )
                t4 = time.monotonic()

                # 5. Capture diagnostics
                mem_state = session._get_memory_diagnostics()
                new_issues = session.all_issues[prev_issue_count:]
                prev_issue_count = len(session.all_issues)

                # 6. Print evaluation
                if eval_result.parse_error:
                    print(f"  {C.YELLOW}[EVAL ERROR] {eval_result.parse_error[:80]}{C.RESET}")
                else:
                    scores = (
                        f"SC={eval_result.scene_consistency.score} "
                        f"CF={eval_result.contradiction_free.score} "
                        f"NC={eval_result.npc_continuity.score} "
                        f"NQ={eval_result.narrative_quality.score} "
                        f"avg={eval_result.average_score:.1f}"
                    )
                    print(f"  {C.GREEN}[EVAL] {scores}{C.RESET}")

                    for issue in eval_result.all_flagged_issues:
                        print(f"  {C.YELLOW}  ⚠ {issue}{C.RESET}")

                # Memory state
                if mem_state:
                    print(f"  {C.DIM}[MEM] buf={mem_state.get('buffer_size', '?')} "
                          f"overflow={mem_state.get('overflow_size', '?')} "
                          f"summary={'yes' if mem_state.get('has_summary') else 'no'} "
                          f"scratchpad={mem_state.get('scratchpad_entries', '?')}{C.RESET}")

                # Planted details
                if self.player.planted_details:
                    print(f"  {C.DIM}[PLANTED] {', '.join(self.player.planted_details[-3:])}{C.RESET}")

                # 7. Record
                self.turns.append(TurnRecord(
                    turn=turn,
                    phase=phase,
                    action=action,
                    narrative=narrative[:2000],
                    mechanics=mechanics,
                    combat_triggered=response.combat_triggered,
                    evaluation=asdict(eval_result) if not eval_result.parse_error else {"error": eval_result.parse_error},
                    elapsed_game=t2 - t1,
                    elapsed_eval=t4 - t3,
                    harness_issues=[{"severity": i.severity, "category": i.category, "desc": i.description} for i in new_issues],
                    memory_state=mem_state or {},
                ))

                # Save incrementally
                self._save_log()

                # Brief pause between turns
                await asyncio.sleep(1)

        finally:
            self.finished_at = time.strftime("%Y-%m-%d %H:%M:%S")
            await session.cleanup()

        # Print final report
        self._print_report()
        self._save_log(final=True)

    def _save_log(self, final: bool = False):
        log_dir = Path("data/eval_logs")
        log_dir.mkdir(parents=True, exist_ok=True)

        report = {
            "config": {
                "persona": self.persona_name,
                "total_turns": self.total_turns,
                "gemini_player_model": self._gemini_player_model,
                "gemini_eval_model": self._gemini_eval_model,
                "started_at": self.started_at,
                "finished_at": self.finished_at,
            },
            "summary": self._compute_summary(),
            "planted_details": self.player.planted_details,
            "turns": [asdict(t) for t in self.turns],
        }

        # Always save latest
        with open(log_dir / "eval_latest.json", "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, default=str)

        # Save timestamped on final
        if final:
            ts = int(time.time())
            path = log_dir / f"eval_{ts}.json"
            with open(path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2, default=str)
            print(f"\n  Log saved: {path}")

    def _compute_summary(self) -> dict:
        dims = ["scene_consistency", "contradiction_free",
                "npc_continuity", "narrative_quality"]
        dim_scores = {d: [] for d in dims}

        total_flags = 0
        harness_errors = 0

        for t in self.turns:
            ev = t.evaluation
            if not ev or "error" in ev:
                continue
            for d in dims:
                s = ev.get(d, {}).get("score", 0)
                if s > 0:
                    dim_scores[d].append(s)
                flags = ev.get(d, {}).get("flagged_issues", [])
                total_flags += len(flags)
            harness_errors += len(t.harness_issues)

        averages = {}
        for d in dims:
            scores = dim_scores[d]
            averages[d] = round(sum(scores) / len(scores), 2) if scores else 0

        all_scores = [s for sl in dim_scores.values() for s in sl]
        overall = round(sum(all_scores) / len(all_scores), 2) if all_scores else 0

        return {
            "overall_average": overall,
            "dimension_averages": averages,
            "total_flagged_issues": total_flags,
            "harness_errors": harness_errors,
            "turns_evaluated": len([t for t in self.turns if t.evaluation and "error" not in t.evaluation]),
        }

    def _print_report(self):
        summary = self._compute_summary()
        avgs = summary["dimension_averages"]

        header("EVALUATION REPORT")
        elapsed = ""
        if self.started_at and self.finished_at:
            elapsed = f" ({self.started_at} → {self.finished_at})"
        print(f"  {self.total_turns} turns, persona={self.persona_name}{elapsed}")
        print(f"  Narrator: Groq qwen3:32b")
        print(f"  Player: {self._gemini_player_model}")
        print(f"  Evaluator: {self._gemini_eval_model}")

        print()
        labels = {
            "scene_consistency": "Scene Consistency  ",
            "contradiction_free": "Contradiction-Free ",
            "npc_continuity": "NPC Continuity     ",
            "narrative_quality": "Narrative Quality  ",
        }
        for dim, label in labels.items():
            score = avgs.get(dim, 0)
            bar_filled = int(score * 2)
            bar_empty = 10 - bar_filled
            bar = "█" * bar_filled + "░" * bar_empty
            color = C.GREEN if score >= 4 else C.YELLOW if score >= 3 else C.RED
            print(f"  {label}  {color}{bar}  {score:.1f}/5{C.RESET}")

        overall = summary["overall_average"]
        bar_filled = int(overall * 2)
        bar_empty = 10 - bar_filled
        bar = "█" * bar_filled + "░" * bar_empty
        color = C.GREEN if overall >= 4 else C.YELLOW if overall >= 3 else C.RED
        print(f"  {'─' * 40}")
        print(f"  {C.BOLD}OVERALL           {color}{bar}  {overall:.1f}/5{C.RESET}")

        # Flagged issues
        all_flags = []
        for t in self.turns:
            ev = t.evaluation
            if not ev or "error" in ev:
                continue
            for dim in labels:
                for issue in ev.get(dim, {}).get("flagged_issues", []):
                    all_flags.append((t.turn, dim, issue))

        if all_flags:
            print(f"\n  {C.YELLOW}Flagged Issues ({len(all_flags)}):{C.RESET}")
            for turn_num, dim, issue in all_flags[:15]:  # Cap at 15
                print(f"    Turn {turn_num} [{dim}]: {issue}")
            if len(all_flags) > 15:
                print(f"    ... and {len(all_flags) - 15} more (see log)")

        # Planted details
        if self.player.planted_details:
            print(f"\n  {C.CYAN}Planted Details:{C.RESET}")
            for d in self.player.planted_details:
                print(f"    - {d}")


# =============================================================================
# CLI
# =============================================================================

async def main():
    import argparse

    parser = argparse.ArgumentParser(description="Automated narrator evaluation")
    parser.add_argument("--turns", type=int, default=12, help="Number of eval turns (default: 12)")
    parser.add_argument("--persona", type=str, default="default",
                        choices=list(PERSONAS.keys()), help="Player persona")
    parser.add_argument("--provider", type=str, default=None,
                        choices=["ollama", "groq"],
                        help="Override narrator LLM provider (default: use .env setting)")
    parser.add_argument("--no-fallback", action="store_true",
                        help="Disable Ollama fallback (Groq-only narration)")
    parser.add_argument("--seed", type=str, default=None,
                        help="Run a scenario from test_harness before eval turns")
    parser.add_argument("--player-model", type=str, default="gemini-2.5-flash",
                        help="Gemini model for player agent")
    parser.add_argument("--eval-model", type=str, default="gemini-3-flash-preview",
                        help="Gemini model for evaluator")
    args = parser.parse_args()

    # Provider override (must be set before config loads)
    if args.provider:
        os.environ["LLM_PROVIDER"] = args.provider
        print(f"{C.YELLOW}[CONFIG] Narrator provider: {args.provider}{C.RESET}")

    if args.no_fallback:
        os.environ["GROQ_FALLBACK_TO_OLLAMA"] = "false"
        print(f"{C.YELLOW}[CONFIG] Ollama fallback disabled — Groq-only narration{C.RESET}")

    eval_session = EvalSession(
        total_turns=args.turns,
        persona_name=args.persona,
        gemini_player_model=args.player_model,
        gemini_eval_model=args.eval_model,
        seed_scenario=args.seed,
    )

    await eval_session.run()


if __name__ == "__main__":
    asyncio.run(main())
