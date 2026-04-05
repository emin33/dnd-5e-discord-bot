"""DM Orchestrator - Coordinates between Narrator and Rules brains.

Implements Rules-first triage: ALL player messages go through Rules Brain
to determine if mechanics are needed before narration.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING
import json
import uuid

from pydantic import BaseModel
import structlog

from .client import get_llm_client, OllamaClient, _write_debug_log
from .brains.base import BrainContext, BrainResult
from .brains.narrator import NarratorBrain, MechanicalOutcome, get_narrator
from .brains.adjudicator import EffectsAdjudicator, get_adjudicator
from .brains.rules import RulesBrain, get_rules_brain
from .intents import validate_narrator_format, strip_planning_text_fallback
from .narrator_tools import NARRATOR_TOOLS, NARRATOR_TOOLS_CORE, tool_calls_to_effects
from .extractors.entity_extractor import get_entity_extractor, EntityExtractor
from .extractors.state_extractor import get_state_extractor, StateExtractor
from .validators.nli_validator import get_nli_validator, NLIValidator
from .turn_logger import get_turn_logger, TurnLogger, TurnRecord
from .effects import (
    ProposedEffect,
    EffectType,
    EffectValidator,
    EffectExecutor,
    EffectValidationResult,
    EffectExecutionResult,
    build_effect_idempotency_key,
)
from ..game.mechanics.dice import get_roller, DiceRoll
from ..game.scene.registry import SceneEntityRegistry
from ..data.repositories import get_character_repo, get_transaction_repo, generate_transaction_key
from ..data.repositories.inventory_repo import get_inventory_repo
from ..models import Character, CharacterCondition, Condition, InventoryItem
from ..models.npc import SceneEntity, EntityType, Disposition
from ..config import get_settings

if TYPE_CHECKING:
    from ..game.session import GameSession
    from ..game.world_state import WorldState, StateDelta

logger = structlog.get_logger()

# Anti-repetition penalties for narrator calls (research: 0.3-0.8 / 0.2-0.6)
NARRATOR_FREQUENCY_PENALTY = 0.4  # Penalize tokens proportional to frequency
NARRATOR_PRESENCE_PENALTY = 0.3   # Penalize any already-used token

# Style rotation — cycles through tone hints so consecutive scenes feel different
NARRATOR_STYLES = [
    "Write with tense, clipped sentences. Short and punchy.",
    "Write with lush, atmospheric detail. Rich sensory imagery.",
    "Write with dry wit and understated tension. Let subtext do the work.",
    "Write with cinematic pacing. Wide establishing shots, then tight close-ups.",
    "Write with a folkloric tone. The world feels ancient and storied.",
]


def get_style_hint(turn: int) -> str:
    """Get a rotating style hint for narrator variety."""
    return NARRATOR_STYLES[turn % len(NARRATOR_STYLES)]


# Format instructions — XML for Claude, text markers for others
NARRATOR_FORMAT_XML = (
    "Output your response in these two XML blocks:\n"
    "<prose>\n[2-4 paragraphs - your narration]\n</prose>\n\n"
    "<intents>\n[intent commands, or NONE]\n</intents>\n\n"
    "Start with <prose> immediately."
)

NARRATOR_FORMAT_TEXT = (
    "Output ONLY these two blocks:\n"
    "PROSE:\n[2-4 paragraphs]\n\n"
    "INTENTS:\n[intent commands, or NONE]\n\n"
    "Start with PROSE: immediately."
)


def _is_anthropic_client(client) -> bool:
    """Check if the narrator client is Anthropic (Claude)."""
    return type(client).__name__ == "AnthropicClient"


# =============================================================================
# TRIAGE PROMPT - Rules Brain decides if mechanics are needed
# =============================================================================

TRIAGE_SYSTEM_PROMPT = """You are the action classifier for a D&D 5e game. Your job is to identify WHAT the player is trying to do and whether it requires a dice roll.

## CORE PRINCIPLE: THE 4 QUESTIONS

Before deciding needs_roll, ask these questions:

**1. Is the outcome uncertain?**
   - Could a competent adventurer reasonably FAIL at this right now?
   - Looking for hidden things? → Uncertain (might miss them)
   - Climbing a sheer cliff? → Uncertain (might fall)
   - Opening an unlocked door? → NOT uncertain (auto-success)

**2. Is there a meaningful consequence for failure?**
   - If failure just means "try again until you succeed," DON'T roll
   - Searching a room with no time pressure → No roll (they'll eventually find it)
   - Searching while guards approach → Roll (failure = caught)
   - Missing a hidden trap → Roll (failure = trigger trap)

**3. Is there time pressure, danger, or opposition?**
   - Combat, chase, ambush, hostile territory, ticking clock
   - Scanning for danger in unknown territory → Roll (there might BE danger)
   - Looking around a safe tavern → No roll

**4. Does success reveal SPECIALIZED KNOWLEDGE?** ← CRITICAL
   - Would identifying/understanding this require expertise or training?
   - Seeing markings → FREE (anyone can see them)
   - Recognizing markings as elven sigils → ROLL (Investigation/History/Arcana)
   - Understanding what a mechanism does → ROLL (Investigation)
   - Identifying a creature's weaknesses → ROLL (Nature/Arcana)
   - Reading magical runes → ROLL (Arcana)
   - This applies EVEN IN SAFE ENVIRONMENTS with no time pressure!

**If YES to Question 4** → needs_roll = true (knowledge check)
**If YES to Questions 1-3** → needs_roll = true (physical/social check)
**If NO to all** → needs_roll = false, narrate the outcome

## COMMON ROLL TRIGGERS

| Action | Typical Check | When to Roll |
|--------|---------------|--------------|
| Scanning for danger | Perception | Unknown/hostile territory, something might be hidden |
| Searching for items | Investigation | Hidden compartments, time pressure |
| Sneaking past guards | Stealth | Guards present, consequences for detection |
| Climbing a wall | Athletics | Difficult surface, fall damage possible |
| Persuading NPC | Persuasion | NPC is resistant or has reason to refuse |
| Recalling lore | History/Arcana/Religion | Obscure knowledge, not common facts |
| Examining symbols/markings | Investigation/History | Meaning, origin, or intent is non-obvious |
| Identifying magical effects | Arcana | Always (magical knowledge is specialized) |
| Understanding mechanisms | Investigation | How it works, how to disable/activate |
| Recognizing creatures | Nature/Arcana | Identifying type, weaknesses, behaviors |

## WHEN NOT TO ROLL — IMPORTANT

A real DM never asks for a roll unless failure would be INTERESTING. Ask yourself:
- "If they fail, does anything change?" If no → don't roll, just narrate success.
- "Could they just try again with no consequence?" If yes → don't roll, auto-succeed.
- "Is this a routine task for an adventurer?" If yes → don't roll.

Specific NO-ROLL situations:
- Trivial actions anyone could do (open door, walk across room, step closer, pick up item)
- Simple physical movement (walking, stepping forward, approaching something visible)
- Observing OBVIOUS things (seeing a door, noticing a person, looking at something in plain sight)
- Information the character would automatically know (common knowledge for their background)
- Scanning a room for VISIBLE people or objects (they're in plain sight!)
- Looking around when there is nothing hidden or concealed
- Routine tasks with unlimited time and no pressure (setting up camp, lighting a fire normally)
- Actions where the player is just adding narrative flair (no extra roll for a cool description)

**CRITICAL: "I look around" / "I scan the room" / "I check who is here" / "I step closer" / "I approach" in a safe or explored location = NO ROLL. Only roll if something is actively hidden, the environment is dangerous, or specialized knowledge is needed.**

**CRITICAL: Movement is NEVER a skill check.** "I step closer", "I walk to the well", "I approach the figure" = exploration or roleplay, NOT a skill check. Movement only requires a roll if the terrain itself is hazardous (climbing a cliff, crossing a tightrope, swimming rapids).

## SETTING THE DC (Difficulty Class)

When a roll IS needed, set an appropriate DC using this standard 5e table:

| DC | Difficulty | Example |
|----|-----------|---------|
| 5  | Very Easy | Climbing a knotted rope, noticing something large and obvious |
| 8  | Easy | Following fresh tracks in mud, calming a frightened animal |
| 10 | Medium | Picking a simple lock, haggling a fair price, recalling common lore |
| 12 | Moderate | Persuading a skeptical guard, finding a concealed door |
| 15 | Hard | Picking a quality lock, deciphering ancient text, tracking over stone |
| 18 | Very Hard | Sneaking past alert guards, recalling obscure arcane knowledge |
| 20 | Nearly Impossible | Picking a masterwork lock, persuading a hostile king |

**CRITICAL: Match the DC to the specific situation.** A casual Perception check in a quiet tavern is DC 10-12, NOT DC 15. Save DC 15+ for genuinely difficult challenges.

## CHOOSING THE RIGHT SKILL

Pick the skill that best matches HOW the character is doing the action:
- **Perception** = noticing things passively (hearing, seeing, smelling)
- **Investigation** = actively examining, deducing, figuring out HOW something works
- **History/Arcana/Religion/Nature** = recalling knowledge about a topic
- **Stealth** = moving unseen, hiding
- **Athletics** = climbing, jumping, swimming, feats of strength
- **Persuasion/Deception/Intimidation** = social influence
- **Survival** = tracking, foraging, navigating wilderness
- **Insight** = reading someone's intentions or detecting lies

"I look for tracks" = **Survival**, not Perception.
"I examine the mechanism" = **Investigation**, not Perception.
"I recall what I know about elves" = **History**, not Perception.

## ACTION TYPES

**attack** - Hostile action toward a CREATURE (triggers combat)
**cast_spell** - Casting any spell
**skill_check** - Any action with uncertain outcome requiring ability/skill check
**saving_throw** - Forced saves (resisting effects)
**purchase** - Buying items
**sell** - Selling items
**inventory** - Managing items (equip, drop, use)
**movement** - Tactical positioning
**social** - Conversation without persuasion needed
**exploration** - Observing visible things
**roleplay** - Pure character expression

## CLASSIFICATION RULES

1. **Attacks vs Skill Checks**:
   - Hostile action toward a CREATURE → attack (triggers combat)
   - Action against an OBJECT with difficulty → skill_check (DEX for ranged, STR for melee)
   - Action against an OBJECT trivially → roleplay (just narrate)

   Examples:
   - "I attack the goblin" → attack, is_creature_target=true
   - "I shoot a tree 200m away" → skill_check (DEX, DC 20 for extreme range)
   - "I kick a pebble" → roleplay (trivial, no roll)
   - "I smash the locked chest" → skill_check (STR, DC based on material)

2. **Skill Checks**: Outcome is uncertain AND matters
   - Physical feats with difficulty → skill_check (Athletics/Acrobatics/DEX/STR)
   - Finding hidden things → skill_check (Perception/Investigation)
   - Influencing NPCs → skill_check (Persuasion/Deception/Intimidation)

3. **Social**: Only when no persuasion/influence is needed
   - Casual conversation, asking prices, greeting → social

## FOR ATTACK ACTIONS (creature targets only)

- target_name: The creature being attacked
- is_creature_target: MUST be true (otherwise use skill_check)

## FOR SKILL CHECKS

- needs_roll: true
- skill: The relevant skill (or "none" for raw ability check)
- ability: dexterity, strength, wisdom, charisma, etc.
- dc: 5=trivial, 10=easy, 15=moderate, 20=hard, 25=very hard, 30=nearly impossible
- on_success: What SPECIFICALLY the character discovers, achieves, or learns. Be concrete!
- on_failure: What SPECIFICALLY happens on failure — what they miss, what goes wrong, or what consequence occurs. Never leave empty.

on_success/on_failure are CRITICAL — the narrator uses these to know what to describe.
Examples:
- Perception check to scan for danger:
  on_success: ["You spot movement in the treeline — two figures crouching behind brush"]
  on_failure: ["The forest seems quiet and still — you notice nothing unusual"]
- Investigation check on strange runes:
  on_success: ["The runes are a warding spell, old but still active — touching them would be dangerous"]
  on_failure: ["The symbols blur together — you can't make sense of the pattern"]
- Athletics check to climb a cliff:
  on_success: ["You find handholds and pull yourself up safely"]
  on_failure: ["Your grip slips — you slide back down, taking 1d4 bludgeoning damage"]
- Persuasion check on a guard:
  on_success: ["The guard sighs and steps aside, waving you through"]
  on_failure: ["The guard crosses his arms — 'Not happening. Move along.'"]

## RESOURCE CONSUMPTION

Note any resources consumed by the action:
- Firing a bow/crossbow → consumes 1 ammunition
- Throwing a weapon → weapon leaves inventory unless retrieved
- Using a consumable → item consumed

## CURRENCY SPENDING

ALWAYS include currency_spent when the player loses, spends, or gives away money - regardless of action_type.
This applies to inventory, roleplay, social, or any other action type.

Examples:
- "I empty half my coin purse" → currency_spent: {"gold": 50} (if they have 100gp)
- "I tip the barmaid 5 silver" → currency_spent: {"silver": 5}
- "I drop my coins on the ground" → currency_spent based on amount
- "I pay the toll" → currency_spent: {"gold": 2}
- "I give them all my gold" → currency_spent: {"gold": X} (their current gold amount)

Use the denomination the player specifies. Available: copper, silver, electrum, gold, platinum.
Check the character's Currency in the context to calculate amounts like "half" or "all".

## FOR PURCHASES

When action_type is "purchase", also provide:
- item_name: What they're trying to buy
- item_cost: Estimated cost in gold (use SRD prices if known)
- quantity: How many

## FOR ROLEPLAY/SOCIAL/EXPLORATION

Just provide narrative_direction with guidance for the narrator.

## SCENE ENTITIES

When "Scene Entities" context is provided, these NPCs/creatures EXIST and can be interacted with.
- Entities marked [HOSTILE] can be attacked
- Entities marked [THREATENING] may become hostile

## OUTPUT FORMAT

Output valid JSON:

```json
{
    "action_type": "skill_check",  // REQUIRED
    "reasoning": "Brief explanation",

    // For attack (creature targets ONLY):
    "target_name": "goblin",
    "is_creature_target": true,

    // For skill_check (includes difficult ranged/melee vs objects):
    "needs_roll": true,
    "skill": "survival",  // Match skill to the action (see skill guide above)
    "ability": "wisdom",
    "dc": 12,  // Set DC using the difficulty table (5-20 range)
    "on_success": ["You find fresh tracks leading north"],
    "on_failure": ["The ground reveals nothing useful"],

    // For purchase:
    "item_name": "Dagger",
    "item_cost": 2,
    "quantity": 1,

    // For social/roleplay/exploration:
    "narrative_direction": "Guidance for narrator",

    // Resources consumed (if any):
    "resources_consumed": [{"item": "Arrow", "quantity": 1}],

    // Currency spent/lost (if any):
    "currency_spent": {"gold": 50}  // or {"silver": 10, "copper": 5}
}
```

Only include fields relevant to the action_type."""


# =============================================================================
# PYDANTIC SCHEMA FOR STRUCTURED OUTPUT
# =============================================================================

class ActionType(str, Enum):
    """Types of player actions for routing."""
    ATTACK = "attack"           # Combat attack
    CAST_SPELL = "cast_spell"   # Spell casting
    SKILL_CHECK = "skill_check" # Ability/skill check
    SAVING_THROW = "saving_throw"  # Forced save
    PURCHASE = "purchase"       # Buying items
    SELL = "sell"               # Selling items
    INVENTORY = "inventory"     # Use/equip/drop items
    MOVEMENT = "movement"       # Tactical movement
    SOCIAL = "social"           # NPC interaction (no mechanics)
    EXPLORATION = "exploration" # Looking around, investigating
    ROLEPLAY = "roleplay"       # Pure narrative


class TriageSchema(BaseModel):
    """Pydantic schema for structured triage output.

    IMPORTANT: All optional fields use plain defaults (str="", int=0)
    instead of Optional[T]=None to avoid anyOf unions in the JSON schema.
    Groq's JSON validator chokes on anyOf patterns, causing intermittent
    "Failed to generate/validate JSON" 400 errors.
    """
    # Action classification
    action_type: str  # One of ActionType values
    reasoning: str = ""

    # For skill_check/saving_throw actions
    needs_roll: bool = False
    roll_type: str = ""       # "ability_check" or "saving_throw"
    ability: str = ""
    skill: str = ""
    dc: int = 0               # 0 = not set
    advantage: bool = False
    disadvantage: bool = False
    advantage_reason: str = ""
    on_success: list[str] = []
    on_failure: list[str] = []

    # For purchase/sell actions
    item_name: str = ""
    item_cost: int = 0        # In gold pieces, 0 = not set
    quantity: int = 1

    # For attack actions
    target_name: str = ""     # What/who they're attacking
    is_creature_target: bool = False  # True if target is a creature (triggers combat)

    # For social/roleplay (no mechanics needed)
    narrative_direction: str = ""

    # Resources consumed by the action (ammunition, consumables, etc.)
    resources_consumed: list[dict] = []  # [{"item": "Arrow", "quantity": 1}]

    # Currency spent/lost by the action (tipping, paying, dropping, etc.)
    currency_spent: dict = {}  # {"gold": 50} or empty = not set


# Cache the schema so we don't regenerate it every call
_TRIAGE_JSON_SCHEMA: Optional[dict] = None


def get_triage_schema() -> dict:
    """Get the JSON schema for triage structured output."""
    global _TRIAGE_JSON_SCHEMA
    if _TRIAGE_JSON_SCHEMA is None:
        _TRIAGE_JSON_SCHEMA = TriageSchema.model_json_schema()
    return _TRIAGE_JSON_SCHEMA


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TriageResult:
    """Result of triage decision."""
    # Action classification (routes to appropriate handler)
    action_type: str  # One of ActionType values
    reasoning: str

    # For skill_check/saving_throw actions
    needs_roll: bool = False
    roll_type: Optional[str] = None  # ability_check, saving_throw, attack_roll
    ability: Optional[str] = None
    skill: Optional[str] = None
    dc: Optional[int] = None
    advantage: bool = False
    disadvantage: bool = False
    advantage_reason: Optional[str] = None
    on_success: list[str] = field(default_factory=list)
    on_failure: list[str] = field(default_factory=list)

    # For purchase/sell actions
    item_name: Optional[str] = None
    item_cost: Optional[int] = None
    quantity: int = 1

    # For attack actions
    target_name: Optional[str] = None
    is_creature_target: bool = False  # Only trigger combat if True

    # For social/roleplay (no mechanics needed)
    narrative_direction: Optional[str] = None

    # Resources consumed by the action
    resources_consumed: list[dict] = field(default_factory=list)

    # Currency spent/lost by the action
    currency_spent: Optional[dict] = None


@dataclass
class MechanicalResolution:
    """Result of mechanical resolution after rolling."""
    success: bool
    roll_result: Optional[DiceRoll] = None
    total: int = 0
    dc: int = 0
    margin: int = 0  # How much above/below DC
    reveals: list[str] = field(default_factory=list)
    skill: Optional[str] = None
    ability: Optional[str] = None
    roll_type: Optional[str] = None


@dataclass
class DMResponse:
    """Response from the DM orchestrator."""
    narrative: str
    mechanical_result: Optional[dict] = None
    tool_calls_made: list[dict] = field(default_factory=list)
    dice_rolls: list[DiceRoll] = field(default_factory=list)
    combat_triggered: bool = False  # True if hostility threshold triggered combat


@dataclass
class ToolExecutionResult:
    """Result of executing a tool call."""
    tool_name: str
    success: bool
    result: Any
    error: Optional[str] = None


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class DMOrchestrator:
    """
    Coordinates between Narrator, Adjudicator, and Rules brains.

    PROSE + INTENTS architecture:
    1. Narrator (high temp): Outputs PROSE and INTENTS blocks
    2. Adjudicator: Parses INTENTS into ProposedEffects (deterministic, no LLM)

    The narrator explicitly signals intents at creation time.
    Prose is NEVER used as source of truth for mechanics.

    Full flow for player messages:
    1. Rules Brain triage: Classify action, decide if roll needed
    2. If roll needed: Execute roll, build resolution
    3. Narrator: Output PROSE + INTENTS
    4. Adjudicator: Parse INTENTS block (deterministic string parsing)
    5. Validator: Check effects are valid
    6. Executor: Apply effects to game state

    This ensures:
    - Narrator explicitly signals mechanical intents
    - No inference from prose wording
    - Effects are determined at creation, not extracted later
    """

    def __init__(
        self,
        narrator: Optional[NarratorBrain] = None,
        adjudicator: Optional[EffectsAdjudicator] = None,
        rules: Optional[RulesBrain] = None,
        client: Optional[OllamaClient] = None,
    ):
        self.narrator = narrator or get_narrator()
        self.adjudicator = adjudicator or get_adjudicator()
        self.rules = rules or get_rules_brain()
        self.client = client or get_llm_client()
        self.roller = get_roller()
        self._current_session: Optional["GameSession"] = None
        self._scene_registry: Optional[SceneEntityRegistry] = None
        self._entity_extractor: EntityExtractor = get_entity_extractor()
        self._state_extractor: StateExtractor = get_state_extractor()
        self._nli_validator: NLIValidator = get_nli_validator()
        self._turn_logger: TurnLogger = get_turn_logger()

        # Effect processing (replaces mechanics extraction)
        self._effect_validator: Optional[EffectValidator] = None
        self._effect_executor: Optional[EffectExecutor] = None
        self._applied_effects: set[str] = set()  # Idempotency tracking

        # DM Scratchpad: session-scoped state for narrator continuity.
        # Inspired by coordinator scratchpad pattern from agentic orchestration.
        # Stores narrative hints, unresolved tensions, NPC moods, etc. that
        # give the narrator richer context without stuffing the prompt.
        self._scratchpad: list[dict] = []  # [{category, note, turn}]
        self._scratchpad_turn = 0
        self._scratchpad_max_entries = 20  # Rolling window

    def set_session(self, session: Optional["GameSession"]) -> None:
        """Set the current session context for tool execution."""
        self._current_session = session
        self._update_effect_processors()

    def set_scene_registry(self, registry: Optional[SceneEntityRegistry]) -> None:
        """Set the scene entity registry for entity tracking."""
        self._scene_registry = registry
        self._update_effect_processors()

    def _update_effect_processors(self) -> None:
        """Update effect validator and executor with current context."""
        self._effect_validator = EffectValidator(
            scene_registry=self._scene_registry,
            session=self._current_session,
        )
        # Executor needs inventory repo - get it async when needed
        self._effect_executor = None  # Lazy init in _process_proposed_effects

    # ==================== Post-Generation Validation ====================

    def _validate_npc_references(self, narrative: str) -> str:
        """Deterministic check for NPC references in narrator output.

        Two-layer validation:
        1. SceneEntityRegistry: logs NPC mentions for tracking
        2. WorldState: flags NPCs mentioned that aren't at the current location

        Does NOT modify the narrative — logs issues for debugging.
        """
        if not self._scene_registry:
            return narrative

        from ..models.npc import EntityType
        npcs = self._scene_registry.get_by_type(EntityType.NPC)
        if not npcs:
            return narrative

        narrative_lower = narrative.lower()

        # Layer 1: SceneEntityRegistry check
        npc_names = {e.name.lower(): e for e in npcs}
        for name_lower, entity in npc_names.items():
            if len(name_lower) < 3:
                continue
            if name_lower in narrative_lower:
                logger.debug(
                    "narrator_npc_reference",
                    npc=entity.name,
                    disposition=entity.disposition.value if entity.disposition else "unknown",
                )

        # Layer 2: WorldState location validation (referenced_entities check)
        world_state = getattr(self._current_session, 'world_state', None) if self._current_session else None
        if world_state and world_state.current_location:
            for npc_name, npc_state in world_state.npcs.items():
                if len(npc_name) < 3:
                    continue
                if npc_name.lower() in narrative_lower:
                    # NPC mentioned — check if they're at the right location
                    if (
                        npc_state.location
                        and npc_state.location.lower() != world_state.current_location.lower()
                        and npc_state.alive
                    ):
                        logger.warning(
                            "npc_location_mismatch",
                            npc=npc_name,
                            npc_location=npc_state.location,
                            party_location=world_state.current_location,
                        )

        return narrative

    # ==================== DM Scratchpad ====================

    def scratchpad_note(self, category: str, note: str) -> None:
        """Add a narrative hint to the DM scratchpad.

        Categories: tension, npc_mood, foreshadow, unresolved, atmosphere, plot
        """
        self._scratchpad_turn += 1
        self._scratchpad.append({
            "category": category,
            "note": note,
            "turn": self._scratchpad_turn,
        })
        # Trim old entries beyond rolling window
        if len(self._scratchpad) > self._scratchpad_max_entries:
            self._scratchpad = self._scratchpad[-self._scratchpad_max_entries:]

    def scratchpad_context(self) -> str:
        """Build scratchpad context string for narrator injection."""
        if not self._scratchpad:
            return ""
        lines = ["<dm_scratchpad>"]
        for entry in self._scratchpad:
            lines.append(f"[{entry['category']}] {entry['note']}")
        lines.append("</dm_scratchpad>")
        return "\n".join(lines)

    def scratchpad_clear(self) -> None:
        """Clear the scratchpad (e.g., on session end)."""
        self._scratchpad.clear()
        self._scratchpad_turn = 0

    def _update_scratchpad(
        self,
        triage: "TriageResult",
        resolution: Optional["MechanicalResolution"],
        effects: list["ProposedEffect"],
        combat_triggered: bool,
        player_name: str,
    ) -> None:
        """Auto-populate scratchpad from turn results for narrator continuity."""
        # Track failed checks — good narrative hooks
        if resolution and not resolution.success and triage.skill:
            self.scratchpad_note(
                "unresolved",
                f"{player_name} failed {triage.skill} check (DC {triage.dc}). "
                "Something was missed or went wrong.",
            )

        # Track combat triggers
        if combat_triggered:
            self.scratchpad_note("tension", "Combat just broke out. Atmosphere is hostile.")

        # Track NPC interactions from effects
        for effect in effects:
            if effect.effect_type == EffectType.ADD_NPC:
                name = effect.npc_name or "someone"
                disp = effect.npc_disposition or "neutral"
                self.scratchpad_note("npc_mood", f"{name} appeared ({disp} disposition).")

        # Track significant damage
        for effect in effects:
            if effect.effect_type == EffectType.APPLY_DAMAGE:
                amount = effect.amount or 0
                if amount >= 10:
                    target = effect.target or "someone"
                    self.scratchpad_note("tension", f"{target} took {amount} damage — situation is dangerous.")

    async def process_action(
        self,
        action: str,
        player_name: str,
        context: BrainContext,
        on_mechanics_ready: Optional[Callable] = None,
        on_narrative_token: Optional[Callable] = None,
    ) -> DMResponse:
        """
        Process a player action through Rules-first triage.

        Flow:
        1. Triage: Classify action type
        2. Route to appropriate handler based on action_type
        3. Execute mechanics BEFORE narration
        4. Narrator dramatizes the known outcome

        Args:
            on_narrative_token: Async callback(str) for streaming narrator tokens
                to Discord. Enables progressive message edits for better UX.
        """
        context.player_action = action
        context.player_name = player_name

        # ── Turn Logger: start recording ──
        session_id = context.session_id or context.campaign_id or "unknown"
        world_state = getattr(self._current_session, 'world_state', None) if self._current_session else None
        turn_num = world_state.turn if world_state else self._scratchpad_turn
        _turn_record = self._turn_logger.new_turn(session_id, turn_num)
        _turn_record.set("action", action)
        _turn_record.set("player", player_name)
        _turn_record.set("phase", world_state.phase if world_state else "unknown")
        ws_before = world_state.to_yaml() if world_state else ""

        # Store streaming callback for narrator methods to use
        self._on_narrative_token = on_narrative_token

        # Inject scratchpad into narrator context for continuity
        scratchpad_ctx = self.scratchpad_context()
        if scratchpad_ctx:
            context.session_summary = (
                (context.session_summary or "") + "\n\n" + scratchpad_ctx
            ).strip()

        logger.info(
            "processing_action",
            action=action[:50],
            player=player_name,
        )

        # Step 0: Removed pre-triage keyword-based attack check.
        # It used fragile keyword matching + disposition filters that skipped friendly
        # entities (e.g., player says "I attack the sailor" but sailor is friendly,
        # so it grabbed the most hostile entity instead). The LLM triage below
        # correctly classifies attacks with target_name and is_creature_target.

        # Step 1: Triage - classify the action
        _turn_record.start_stage("triage")
        triage = await self._triage_action(action, player_name, context)
        _turn_record.end_stage("triage")
        _turn_record.record_triage(
            action_type=triage.action_type,
            needs_roll=triage.needs_roll,
            skill=triage.skill or "",
            dc=triage.dc or 0,
            parse_warnings=getattr(triage, "_parse_warnings", None),
        )

        logger.info(
            "triage_decision",
            action_type=triage.action_type,
            needs_roll=triage.needs_roll,
            skill=triage.skill,
            dc=triage.dc,
            target_name=triage.target_name,
            is_creature_target=triage.is_creature_target,
            reasoning=triage.reasoning[:100] if triage.reasoning else "",
        )

        # Step 2: Route to appropriate handler based on action_type
        resolution = None
        mechanical_result = None
        dice_rolls = []
        tool_calls = []

        if triage.action_type == "purchase":
            # Execute purchase through Rules Brain
            mechanical_result = await self._handle_purchase(triage, player_name, context)
            tool_calls = mechanical_result.get("tool_calls", []) if mechanical_result else []

        elif triage.action_type == "sell":
            # Execute sale through Rules Brain
            mechanical_result = await self._handle_sell(triage, player_name, context)
            tool_calls = mechanical_result.get("tool_calls", []) if mechanical_result else []

        elif triage.action_type == "skill_check" and triage.needs_roll:
            # Execute skill check with dice roll
            resolution = await self._resolve_mechanics(triage, player_name, context)
            if resolution.roll_result:
                dice_rolls.append(resolution.roll_result)
            mechanical_result = {
                "action_type": "skill_check",
                "success": resolution.success,
                "skill": resolution.skill,
                "ability": resolution.ability,
                "roll": resolution.total,
                "dc": resolution.dc,
                "margin": resolution.margin,
            }

        elif triage.action_type == "inventory":
            # Inventory actions - pick up, drop, use, equip
            mechanical_result = await self._handle_inventory(triage, player_name, context, action)
            tool_calls = mechanical_result.get("tool_calls", []) if mechanical_result else []

        elif triage.action_type == "attack":
            # Check if this is a creature target (triggers combat) or object (just narrate)
            if not triage.is_creature_target:
                # Attacking an object/non-creature - just narrate it
                logger.info(
                    "attack_non_creature",
                    target=triage.target_name,
                    reasoning=triage.reasoning[:50] if triage.reasoning else "",
                )
                # Let narrator handle it with guidance
                if not triage.narrative_direction:
                    triage.narrative_direction = (
                        f"The player attacks {triage.target_name or 'the target'}. "
                        "Describe what happens - this doesn't trigger combat."
                    )
                # Fall through to narrator below
            elif not context.in_combat:
                # Attacking a creature — start combat, but do NOT consume resources.
                # No actual attack happens here; the real attack comes through the
                # combat UI. Resource consumption (arrows, etc.) happens there.
                combat_started = await self._initiate_combat_from_attack(
                    triage.target_name, player_name
                )
                if combat_started:
                    return DMResponse(
                        narrative=f"*You attack the {triage.target_name or 'enemy'}! Combat begins!*",
                        combat_triggered=True,
                    )
                else:
                    # Target not in registry — but the narrator described them previously.
                    # Register them now from context and start combat.
                    logger.info(
                        "attack_target_not_in_scene_creating",
                        target=triage.target_name,
                    )

                    # Create the target entity on the fly with best-guess SRD match
                    target_name = triage.target_name or "enemy"
                    monster_index = self._guess_monster_index(target_name)

                    from ..models.npc import SceneEntity, EntityType, Disposition
                    entity = SceneEntity(
                        name=target_name.title(),
                        entity_type=EntityType.CREATURE,
                        description=f"Hostile creature from the scene",
                        disposition=Disposition.HOSTILE,
                        hostility_score=90,
                        monster_index=monster_index,
                    )

                    if self._scene_registry:
                        self._scene_registry.register_entity(entity)

                    # Now try combat again
                    combat_started = await self._initiate_combat_from_attack(
                        target_name, player_name
                    )
                    if combat_started:
                        return DMResponse(
                            narrative=f"*You attack the {target_name}! Combat begins!*",
                            combat_triggered=True,
                        )

                    # Still failed — absolute fallback, create combat with just this entity
                    hostile_entities = [entity]
                    combat_started = await self._trigger_combat(hostile_entities, player_initiated=True)
                    if combat_started:
                        return DMResponse(
                            narrative=f"*You attack the {target_name}! Combat begins!*",
                            combat_triggered=True,
                        )
            # TODO: Integrate with combat coordinator for in-combat attacks

        elif triage.action_type == "cast_spell":
            # Spell casting — validate slot availability and provide context to narrator.
            # Full spell resolution (attack rolls, saves, AoE) is handled by the
            # combat coordinator when in combat; this path handles exploration/social casting.
            resolved = self._resolve_character_by_name(player_name)
            if resolved:
                char_id, character = resolved
                # Build spell context for the narrator
                slot_info = []
                for level in range(1, 10):
                    current, max_slots = character.spell_slots.get_slots(level)
                    if max_slots > 0:
                        slot_info.append(f"L{level}: {current}/{max_slots}")
                spell_context = ", ".join(slot_info) if slot_info else "no spell slots"

                if not triage.narrative_direction:
                    triage.narrative_direction = (
                        f"The player attempts to cast a spell. "
                        f"Their available spell slots: {spell_context}. "
                        f"Narrate the spell's effect based on the game context. "
                        f"If the spell requires a slot they don't have, describe the failure."
                    )

        # Step 2.5: Send mechanics to Discord immediately (before narration LLM call)
        if on_mechanics_ready and (mechanical_result or dice_rolls):
            try:
                await on_mechanics_ready(mechanical_result, dice_rolls)
            except Exception as e:
                logger.warning("on_mechanics_ready_callback_failed", error=str(e))

        # Step 2.75: Knowledge graph context for narrator
        # Multi-tier entity resolution: scene seeds + substring match + vector fallback
        kg = getattr(self._current_session, 'knowledge_graph', None) if self._current_session else None
        _kg_seed_ids: list[str] = []
        _kg_vector_matches = 0
        _kg_narrative_recalled = 0
        if kg and kg.node_count() > 0:
            try:
                from ..game.knowledge.matcher import EntityNameMatcher
                from ..memory import get_vector_store

                matcher = EntityNameMatcher(kg)
                world_state_pre = getattr(self._current_session, 'world_state', None) if self._current_session else None

                # Tier 1: Substring match on player text
                text_seeds = matcher.match(action)

                # Tier 2: Always include current scene (location + present NPCs)
                scene_seeds = matcher.scene_seeds(world_state_pre) if world_state_pre else []

                # Tier 3: Vector fallback if player text didn't match any entities
                if not text_seeds and kg.node_count() > 0:
                    vs = get_vector_store()
                    text_seeds = matcher.vector_match(action, context.campaign_id, vs)
                    _kg_vector_matches = len(text_seeds)

                # Merge and deduplicate (text matches first for priority)
                seen = set()
                _kg_seed_ids = []
                for sid in text_seeds + scene_seeds:
                    if sid not in seen:
                        _kg_seed_ids.append(sid)
                        seen.add(sid)

                if _kg_seed_ids:
                    # Graph context: structured relationships
                    context.kg_context_yaml = kg.to_context_yaml(_kg_seed_ids)

                    # Narrative recall: past prose about these entities
                    try:
                        vs = get_vector_store()
                        seed_names = [kg.get_entity(s).name for s in _kg_seed_ids if kg.get_entity(s)]
                        if seed_names:
                            ws_turn = world_state_pre.turn if world_state_pre else 0
                            past_chunks = vs.recall_narratives_for_entities(
                                campaign_id=context.campaign_id,
                                entity_ids=_kg_seed_ids,
                                query_text=" ".join(seed_names),
                                max_results=2,
                                current_turn=ws_turn,
                            )
                            if past_chunks:
                                context.narrative_memory = "\n---\n".join(
                                    c["content"][:300] for c in past_chunks
                                )
                                _kg_narrative_recalled = len(past_chunks)
                    except Exception as e:
                        logger.warning("kg_narrative_recall_failed", error=str(e))

                    logger.debug(
                        "kg_context_injected",
                        seed_count=len(_kg_seed_ids),
                        text_matches=len(text_seeds),
                        scene_seeds=len(scene_seeds),
                        vector_matches=_kg_vector_matches,
                        narrative_recalled=_kg_narrative_recalled,
                    )
            except Exception as e:
                logger.warning("kg_context_failed", error=str(e))

        # Step 3: Narrator dramatizes the outcome (with mechanical result as context)
        proposed_effects: list[ProposedEffect] = []

        _turn_record.start_stage("narrate")
        if resolution:
            # Roll happened - narrate the mechanical outcome
            narrative, proposed_effects = await self._narrate_outcome(action, player_name, context, triage, resolution)
        elif mechanical_result:
            # Mechanical action (purchase, etc.) - narrate with the result
            narrative, proposed_effects = await self._narrate_mechanical_result(action, player_name, context, triage, mechanical_result)
        else:
            # No mechanics - just respond to the player's action naturally
            narrative, proposed_effects = await self._narrate_action(action, player_name, context, triage)
        _turn_record.end_stage("narrate")
        _turn_record.record_narrator_response(narrative or "", format_type="xml" if _is_anthropic_client(self.narrator.client) else "text")

        # Step 3.5: Post-generation NPC validation (deterministic, no LLM)
        # Checks if narrator used NPC names in wrong locations
        if narrative and self._scene_registry:
            narrative = self._validate_npc_references(narrative)

        # Step 3.5b: Collect ref_entity IDs from narrator intents.
        # These are entities the narrator explicitly tagged — used to
        # prevent the state extractor from creating duplicates, and as
        # additional KG seeds for narrative chunk tagging.
        _narrator_ref_ids: list[str] = []
        if proposed_effects:
            from .effects import EffectType as _ET
            _narrator_ref_ids = [
                e.ref_entity_id for e in proposed_effects
                if e.effect_type == _ET.REF_ENTITY and e.ref_entity_id
            ]

        # Step 3.6: World state extraction — extract StateDelta and apply
        # Uses cheap brain model to identify what changed in the world
        delta = None
        world_state = getattr(self._current_session, 'world_state', None) if self._current_session else None
        # Capture pre-delta location for knowledge graph connectivity
        pre_delta_location = world_state.current_location if world_state else ""
        if narrative and world_state:
            _turn_record.start_stage("state_extract")
            delta = await self._extract_and_apply_state_delta(
                narrative, world_state, context,
                referenced_entity_ids=_narrator_ref_ids,
            )
            _turn_record.end_stage("state_extract")

            # Record state delta to turn log (including parse warnings)
            if delta:
                _turn_record.record_state_delta(
                    delta_dict=delta.model_dump(exclude_none=True, exclude_defaults=True),
                    rejections=[],
                    parse_warnings=getattr(delta, "_parse_warnings", None),
                )

        # Step 3.6b: Bridge state delta to knowledge graph
        _kg_ops_applied = 0
        _kg_ops_rejected = 0
        graph_ops = []
        if kg and delta and world_state:
            try:
                from ..game.knowledge.bridge import DeltaBridge
                bridge = DeltaBridge(context.campaign_id)
                existing_ids = set(kg._entities.keys()) if kg._entities else set()
                graph_ops = bridge.convert(
                    delta, world_state,
                    existing_node_ids=existing_ids,
                    previous_location=pre_delta_location,
                )
                if graph_ops:
                    rejections = await kg.apply_operations(graph_ops)
                    _kg_ops_applied = len(graph_ops) - len(rejections)
                    _kg_ops_rejected = len(rejections)
                    if rejections:
                        logger.debug("kg_bridge_rejections", rejections=rejections)
            except Exception as e:
                logger.warning("kg_bridge_failed", error=str(e))

        # Step 3.6c: Sync entity descriptions to ChromaDB for vector matching
        if kg and graph_ops:
            try:
                from ..game.knowledge.models import AddNode, UpdateNode
                from ..memory import get_vector_store
                vs = get_vector_store()
                for op in graph_ops:
                    if isinstance(op, AddNode):
                        entity = kg.get_entity(op.entity.node_id)
                    elif isinstance(op, UpdateNode):
                        entity = kg.get_entity(op.node_id)
                    else:
                        continue
                    if entity and entity.properties.get("description"):
                        vs.add_entity_description(
                            campaign_id=context.campaign_id,
                            node_id=entity.node_id,
                            entity_type=entity.entity_type.value,
                            name=entity.name,
                            description=entity.properties["description"],
                            aliases=entity.aliases,
                        )
            except Exception as e:
                logger.warning("kg_entity_sync_failed", error=str(e))

        # Step 3.6d: Store tagged narrative chunk for future recall
        _narrative_chunk_stored = False
        if narrative and kg:
            try:
                from ..game.knowledge.matcher import EntityNameMatcher
                from ..memory import get_vector_store
                matcher = EntityNameMatcher(kg)

                # Tag with entities mentioned in the narrative + scene seeds + narrator refs
                narrator_entity_ids = matcher.match(narrative)
                all_tags = list(set(narrator_entity_ids + _kg_seed_ids + _narrator_ref_ids))

                if all_tags:
                    vs = get_vector_store()
                    vs.add_narrative_chunk(
                        campaign_id=context.campaign_id,
                        chunk_id=str(uuid.uuid4()),
                        narrative_text=narrative,
                        entity_ids=all_tags,
                        turn=world_state.turn if world_state else 0,
                        location=world_state.current_location if world_state else "",
                    )
                    _narrative_chunk_stored = True
            except Exception as e:
                logger.warning("kg_narrative_store_failed", error=str(e))

        # Record knowledge graph state in turn log
        if kg:
            _turn_record.record_knowledge_graph(
                nodes_total=kg.node_count(),
                edges_total=kg.edge_count(),
                seed_entities=_kg_seed_ids,
                context_injected=bool(context.kg_context_yaml),
                ops_applied=_kg_ops_applied,
                ops_rejected=_kg_ops_rejected,
                narrative_chunk_stored=_narrative_chunk_stored,
                vector_matches=_kg_vector_matches,
                narrative_chunks_recalled=_kg_narrative_recalled,
            )

        # Step 3.7: NLI contradiction check — DISABLED
        # The NLI layer detects contradictions but doesn't act on them (no re-narration
        # or correction wired up). It adds 1-1.5s latency per turn for logging only.
        # Re-enable when wired to trigger re-narration on contradiction.
        # See roadmap: "NLI Correction Loop" under needs-work items.
        nli_contradictions = []
        _turn_record.record_nli(
            pairs_checked=0,
            contradictions=[],
            ambiguous_count=0,
            tiebreaker_results=0,
        )

        # Step 4: Process narrator output - entity extraction + proposed effects
        # Skip entity extraction during combat or for simple actions (saves API calls)
        # Also skip if world state handled extraction (avoids duplicate NPC registration)
        combat_triggered = False
        skip_extraction = (
            context.in_combat
            or triage.action_type in ("skill_check", "saving_throw")
            or world_state is not None  # WorldState extraction supersedes entity extraction
        )
        if narrative:
            message_id = str(uuid.uuid4())
            combat_triggered = await self._process_narrator_output(
                narrative, context, proposed_effects, message_id,
                skip_entity_extraction=skip_extraction,
            )

        # Step 5: Consume resources/currency AFTER outcome is determined.
        # This ensures we don't deduct ammunition on a cancelled action or
        # currency on a failed narration. Purchase gold is validated separately
        # in _handle_purchase before reaching here.
        if triage.resources_consumed:
            await self._consume_resources(triage.resources_consumed, player_name)
        if triage.currency_spent:
            await self._consume_currency(triage.currency_spent, player_name)

        # Step 6: Auto-populate DM scratchpad from this turn's results
        self._update_scratchpad(triage, resolution, proposed_effects, combat_triggered, player_name)

        # ── Turn Logger: finalize and flush ──
        ws_after = world_state.to_yaml() if world_state else ""
        _turn_record.record_world_state(ws_before, ws_after)
        _turn_record.set("combat_triggered", combat_triggered)
        _turn_record.set("effects_count", len(proposed_effects))

        # Record effects detail — especially ref_entity for entity grounding observability
        if proposed_effects:
            from .effects import EffectType as _ET
            _turn_record.set("effects", [
                {
                    "type": e.effect_type.value,
                    **({"ref_id": e.ref_entity_id} if e.effect_type == _ET.REF_ENTITY else {}),
                    **({"ref_alias": e.ref_alias_used} if e.ref_alias_used else {}),
                    **({"npc_name": e.npc_name} if e.effect_type == _ET.ADD_NPC else {}),
                    **({"item": e.item_name or e.object_name} if e.item_name or e.object_name else {}),
                }
                for e in proposed_effects
            ])

        self._turn_logger.flush(_turn_record)

        return DMResponse(
            narrative=narrative,
            mechanical_result=mechanical_result,
            tool_calls_made=tool_calls,
            dice_rolls=dice_rolls,
            combat_triggered=combat_triggered,
        )

    async def _triage_action(
        self,
        action: str,
        player_name: str,
        context: BrainContext,
    ) -> TriageResult:
        """
        Rules Brain triage: Decide if this action needs mechanical resolution.

        Returns structured decision with roll requirements and success/failure reveals.
        """
        # Build context for triage
        triage_context = await self._build_triage_context(action, player_name, context)

        messages = [
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
            {"role": "user", "content": triage_context},
        ]

        try:
            response = await self.client.chat(
                messages=messages,
                temperature=0,  # Ollama recommends 0 for structured output
                max_tokens=500,
                json_schema=get_triage_schema(),  # Use proper JSON schema
                think=False,  # Disable Qwen3 thinking mode for structured output
            )

            # Parse JSON response
            triage_data, parse_warnings = self._parse_triage_json(response.content)

            # Log parse failures at warning level so they're visible
            if triage_data.get("_parse_failed"):
                logger.warning(
                    "triage_parse_failure_detected",
                    raw_response=response.content[:300] if response.content else "empty",
                    action=action[:100],
                )
            if parse_warnings:
                logger.warning(
                    "triage_parse_warnings",
                    warnings=parse_warnings,
                    action=action[:80],
                )

            # Log for debugging
            logger.debug("triage_raw_response", content=response.content[:200] if response.content else "empty")

            # Get action type (required)
            action_type = triage_data.get("action_type", "roleplay")

            # Get the appropriate guidance based on action type
            narrative_direction = triage_data.get("narrative_direction")
            needs_roll = triage_data.get("needs_roll", False)

            # For non-mechanical actions, ensure we have narrative direction
            if action_type in ("social", "roleplay", "exploration") and not narrative_direction:
                narrative_direction = f"The action proceeds. {triage_data.get('reasoning', 'Describe the outcome.')}"

            # Convert empty strings to None (schema uses "" defaults to avoid
            # anyOf unions that break Groq, but downstream expects None for "not set")
            def _or_none(val):
                return val if val else None

            result = TriageResult(
                action_type=action_type,
                reasoning=triage_data.get("reasoning", ""),
                needs_roll=needs_roll,
                roll_type=_or_none(triage_data.get("roll_type")),
                ability=_or_none(triage_data.get("ability")),
                skill=_or_none(triage_data.get("skill")),
                dc=triage_data.get("dc") or None,
                advantage=triage_data.get("advantage", False),
                disadvantage=triage_data.get("disadvantage", False),
                advantage_reason=_or_none(triage_data.get("advantage_reason")),
                on_success=triage_data.get("on_success", []),
                on_failure=triage_data.get("on_failure", []),
                # Purchase fields
                item_name=_or_none(triage_data.get("item_name")),
                item_cost=triage_data.get("item_cost") or None,
                quantity=triage_data.get("quantity", 1),
                # Attack fields
                target_name=_or_none(triage_data.get("target_name")),
                is_creature_target=triage_data.get("is_creature_target", False),
                # Narrative guidance
                narrative_direction=_or_none(narrative_direction),
                # Resource consumption
                resources_consumed=triage_data.get("resources_consumed", []),
                # Currency spending
                currency_spent=triage_data.get("currency_spent") or None,
            )
            # Attach parse warnings for turn log observability
            result._parse_warnings = parse_warnings
            return result

        except Exception as e:
            import traceback
            logger.error(
                "triage_failed",
                error=str(e),
                traceback=traceback.format_exc(),
                action=action[:100],
            )
            _write_debug_log("TRIAGE_ERROR", f"Action: {action}\nError: {str(e)}\n{traceback.format_exc()}")
            # Default to roleplay on triage failure
            return TriageResult(
                action_type="roleplay",
                reasoning=f"Triage error: {str(e)}",
                narrative_direction="Describe the character's action and the scene's response naturally.",
            )

    async def _build_triage_context(
        self,
        action: str,
        player_name: str,
        context: BrainContext,
    ) -> str:
        """Build the context string for triage decision."""
        parts = []

        # Scene entities (NPCs, creatures, objects present) - FIRST for priority
        if self._scene_registry and self._scene_registry.has_entities():
            entity_context = self._scene_registry.get_triage_context()
            if entity_context:
                parts.append(entity_context)

        # Current scene
        if context.current_scene:
            parts.append(f"## Current Scene\n{context.current_scene}")

        # Party info
        party = context.party_members or context.party_status
        if party:
            parts.append(f"## Party\n{party}")

        # Character capabilities - critical for determining what's possible
        char_capabilities = await self._get_character_capabilities(player_name)
        if char_capabilities:
            parts.append(f"## {player_name}'s Capabilities\n{char_capabilities}")

        # Combat state
        if context.combat_state:
            parts.append(f"## Combat\n{context.combat_state}")
        elif context.in_combat:
            parts.append(f"## Combat Active\nRound {context.combat_round}")

        # The action to analyze
        parts.append(f"## Player Action\n[{player_name}]: \"{action}\"")
        parts.append("\nAnalyze this action considering the character's capabilities and output your triage decision as JSON.")

        return "\n\n".join(parts)

    async def _get_character_capabilities(self, player_name: str) -> str:
        """Get a summary of character capabilities for triage context."""
        if not self._current_session:
            return ""

        # Find the character
        character = None
        for player in self._current_session.players.values():
            if player.character and player_name.lower() in player.character.name.lower():
                character = player.character
                break

        if not character:
            return ""

        capabilities = []

        # Class and level
        capabilities.append(f"Class: {character.class_index.title()} Level {character.level}")

        # Key proficiencies
        if character.skill_proficiencies:
            skills = [s.value if hasattr(s, 'value') else s for s in character.skill_proficiencies]
            capabilities.append(f"Proficient Skills: {', '.join(skills)}")

        # Spellcasting (if any)
        if hasattr(character, 'spells_known') and character.spells_known:
            spell_names = [s.name if hasattr(s, 'name') else str(s) for s in character.spells_known[:10]]
            capabilities.append(f"Known Spells: {', '.join(spell_names)}")

        # Spell slots available
        if hasattr(character, 'spell_slots') and character.spell_slots:
            slot_parts = []
            for level in range(1, 10):
                try:
                    current, maximum = character.spell_slots.get_slots(level)
                    if maximum > 0:
                        slot_parts.append(f"L{level}: {current}/{maximum}")
                except (ValueError, AttributeError):
                    break
            if slot_parts:
                capabilities.append(f"Spell Slots: {', '.join(slot_parts)}")

        # Inventory and gold - fetch from database
        try:
            inventory_repo = await get_inventory_repo()

            # Get currency first - critical for purchase validation
            currency = await inventory_repo.get_currency(character.id)
            if currency:
                gold_parts = []
                if currency.platinum > 0:
                    gold_parts.append(f"{currency.platinum}pp")
                if currency.gold > 0:
                    gold_parts.append(f"{currency.gold}gp")
                if currency.electrum > 0:
                    gold_parts.append(f"{currency.electrum}ep")
                if currency.silver > 0:
                    gold_parts.append(f"{currency.silver}sp")
                if currency.copper > 0:
                    gold_parts.append(f"{currency.copper}cp")

                if gold_parts:
                    capabilities.append(f"Currency: {', '.join(gold_parts)} (≈{currency.total_in_gold:.1f}gp total)")
                else:
                    capabilities.append("Currency: None (0gp)")

            # Get inventory items
            items = await inventory_repo.get_all_items(character.id)
            if items:
                # Group items by relevance for action capability
                equipped = [i for i in items if i.equipped]
                tools_and_consumables = [
                    i for i in items if not i.equipped and any(
                        keyword in i.item_name.lower()
                        for keyword in [
                            'tinderbox', 'torch', 'lantern', 'rope', 'grappling',
                            'crowbar', 'hammer', 'piton', 'thieves', 'tools',
                            'kit', 'potion', 'scroll', 'wand', 'staff', 'oil',
                            'flask', 'vial', 'holy', 'component', 'focus',
                        ]
                    )
                ]

                if equipped:
                    equipped_names = [f"{i.item_name}" for i in equipped[:5]]
                    capabilities.append(f"Equipped: {', '.join(equipped_names)}")

                if tools_and_consumables:
                    tool_names = [f"{i.item_name} (x{i.quantity})" if i.quantity > 1 else i.item_name
                                  for i in tools_and_consumables[:8]]
                    capabilities.append(f"Tools/Consumables: {', '.join(tool_names)}")

                # Note total inventory size
                other_count = len(items) - len(equipped) - len(tools_and_consumables)
                if other_count > 0:
                    capabilities.append(f"(+{other_count} other items in inventory)")
        except Exception as e:
            logger.debug("inventory_fetch_failed", error=str(e))
            # Continue without inventory if fetch fails

        return "\n".join(capabilities)

    def _parse_triage_json(self, content: str) -> tuple[dict, list[str]]:
        """Parse and validate JSON from triage response.

        Uses Pydantic validation against TriageSchema. Logs warnings on
        parse failures instead of silently returning defaults.

        Returns (data_dict, parse_warnings) so callers can record warnings
        in the turn log for post-mortem observability.
        """
        warnings: list[str] = []
        content = content.strip()

        # Strip markdown code fences if present
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
                warnings.append("stripped_markdown_fence")
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
                warnings.append("stripped_code_fence")

        # Extract JSON object
        if "{" in content:
            json_start = content.find("{")
            json_end = content.rfind("}") + 1
            if json_start > 0 or json_end < len(content):
                # Had to extract JSON from surrounding text
                warnings.append("extracted_json_from_text")
            if json_end > json_start:
                content = content[json_start:json_end]

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.error(
                "triage_json_parse_failed",
                raw_content=content[:500],
                error=str(e),
            )
            warnings.append(f"json_parse_failed: {e}")
            # Return a clearly marked failure so callers can detect it
            return {
                "action_type": "roleplay",
                "needs_roll": False,
                "reasoning": "TRIAGE_PARSE_FAILURE: LLM returned non-JSON response",
                "_parse_failed": True,
            }, warnings

        # Coerce bare strings to single-element lists for list fields.
        # Small models often output "on_success": "description" instead
        # of "on_success": ["description"].
        for field in ("on_success", "on_failure", "resources_consumed"):
            if field in data and isinstance(data[field], str):
                data[field] = [data[field]]
                warnings.append(f"coerced_string_to_list:{field}")

        # Validate against schema — log any coercion issues but don't reject
        try:
            validated = TriageSchema(**data)
            return validated.model_dump(), warnings
        except Exception as e:
            logger.warning(
                "triage_schema_validation_failed",
                raw_data=str(data)[:500],
                error=str(e),
            )
            warnings.append(f"schema_validation_failed: {e}")
            # Data parsed as JSON but failed schema validation — use raw data
            # with defaults for missing fields
            data.setdefault("action_type", "roleplay")
            data.setdefault("needs_roll", False)
            data.setdefault("reasoning", "Schema validation failed, using raw data")
            return data, warnings

    # =========================================================================
    # ACTION HANDLERS - Execute mechanics based on action_type
    # =========================================================================

    async def _handle_purchase(
        self,
        triage: TriageResult,
        player_name: str,
        context: BrainContext,
    ) -> dict:
        """
        Handle purchase action - validate and execute through Rules Brain tools.

        Returns mechanical result dict with success/failure info.
        """
        item_name = triage.item_name or "unknown item"
        item_cost = triage.item_cost or 0
        quantity = triage.quantity or 1

        # Resolve character
        resolved = self._resolve_character_by_name(player_name)
        if not resolved:
            return {
                "action_type": "purchase",
                "success": False,
                "error": f"Character '{player_name}' not found",
                "item": item_name,
                "narrative_hint": f"{player_name} isn't known to the system.",
            }

        character, char_id = resolved

        # Execute the purchase through the existing tool function
        result = await self._execute_purchase_item({
            "buyer_id": char_id,
            "item_index": item_name.lower().replace(" ", "-"),
            "item_name": item_name,
            "cost_gold": item_cost,
            "quantity": quantity,
        })

        # Determine narrative hint based on result
        if result.get("purchased", False):
            narrative_hint = (
                f"{player_name} successfully purchases {quantity}x {item_name} for {item_cost}gp. "
                f"They have {result.get('gold_after', 0)}gp remaining."
            )
        else:
            error = result.get("error", "Unknown error")
            narrative_hint = f"{player_name} cannot complete the purchase: {error}"

        return {
            "action_type": "purchase",
            "success": result.get("purchased", False),
            "item": item_name,
            "quantity": quantity,
            "cost": item_cost,
            "gold_after": result.get("gold_after"),
            "error": result.get("error"),
            "narrative_hint": narrative_hint,
            "tool_calls": [{"name": "purchase_item", "result": result}],
        }

    async def _handle_sell(
        self,
        triage: TriageResult,
        player_name: str,
        context: BrainContext,
    ) -> dict:
        """
        Handle sell action - execute through Rules Brain tools.
        """
        item_name = triage.item_name or "unknown item"
        quantity = triage.quantity or 1

        # For now, return a placeholder - needs inventory lookup to find item_id
        # TODO: Implement full sell logic with inventory lookup
        return {
            "action_type": "sell",
            "success": False,
            "error": "Sell action not yet implemented - use /inventory commands",
            "item": item_name,
            "narrative_hint": f"{player_name} attempts to sell {item_name}, but the merchant isn't interested right now.",
        }

    async def _handle_inventory(
        self,
        triage: TriageResult,
        player_name: str,
        context: BrainContext,
        action: str,
    ) -> dict:
        """
        Handle inventory actions - pick up, drop, use, equip items.

        Parses the action to determine what kind of inventory operation.
        """
        action_lower = action.lower()

        # Resolve character
        resolved = self._resolve_character_by_name(player_name)
        if not resolved:
            return {
                "action_type": "inventory",
                "success": False,
                "error": f"Character '{player_name}' not found",
                "narrative_hint": f"{player_name} isn't known to the system.",
            }

        character, char_id = resolved

        # Determine inventory action type from the action text
        pickup_keywords = ["pick up", "pickup", "take", "grab", "collect", "pocket", "put in", "stash", "keep"]
        drop_keywords = ["drop", "discard", "throw away", "leave", "abandon"]
        equip_keywords = ["equip", "wield", "wear", "don", "put on", "draw"]
        unequip_keywords = ["unequip", "remove", "take off", "sheathe", "stow"]
        use_keywords = ["use", "drink", "eat", "consume", "apply", "read"]

        is_pickup = any(kw in action_lower for kw in pickup_keywords)
        is_drop = any(kw in action_lower for kw in drop_keywords)
        is_equip = any(kw in action_lower for kw in equip_keywords)
        is_unequip = any(kw in action_lower for kw in unequip_keywords)
        is_use = any(kw in action_lower for kw in use_keywords)

        # Extract item name from triage or try to parse from action
        item_name = triage.item_name
        if not item_name:
            # Try to extract from common patterns like "pick up the feather"
            import re
            patterns = [
                r"(?:pick up|take|grab|pocket|keep|put)\s+(?:the\s+)?(.+?)(?:\s+in|\s+into|$)",
                r"(?:drop|discard)\s+(?:the\s+)?(.+?)$",
                r"(?:equip|wield|wear)\s+(?:the\s+)?(.+?)$",
            ]
            for pattern in patterns:
                match = re.search(pattern, action_lower)
                if match:
                    item_name = match.group(1).strip()
                    break

        if not item_name:
            item_name = "item"

        # Clean up item name
        item_name = item_name.rstrip(".")

        quantity = triage.quantity or 1

        if is_pickup:
            # Add item to inventory
            result = await self._execute_add_item({
                "character_id": char_id,
                "item_index": item_name.lower().replace(" ", "-"),
                "item_name": item_name.title(),
                "quantity": quantity,
                "source": "picked up",
            })

            if result.get("added", False):
                narrative_hint = f"{player_name} picks up the {item_name} and adds it to their inventory."
                return {
                    "action_type": "inventory",
                    "operation": "pickup",
                    "success": True,
                    "item": item_name,
                    "quantity": quantity,
                    "narrative_hint": narrative_hint,
                    "tool_calls": [{"name": "add_item", "result": result}],
                }
            else:
                return {
                    "action_type": "inventory",
                    "operation": "pickup",
                    "success": False,
                    "item": item_name,
                    "error": result.get("error", "Failed to add item"),
                    "narrative_hint": f"{player_name} tries to pick up the {item_name}, but something prevents them.",
                }

        elif is_drop:
            # Remove item from inventory
            # TODO: Need to look up item_id first
            return {
                "action_type": "inventory",
                "operation": "drop",
                "success": True,  # Narrative success - they dropped it
                "item": item_name,
                "narrative_hint": f"{player_name} drops the {item_name}.",
            }

        elif is_equip:
            return {
                "action_type": "inventory",
                "operation": "equip",
                "success": True,
                "item": item_name,
                "narrative_hint": f"{player_name} equips the {item_name}.",
            }

        elif is_use:
            return {
                "action_type": "inventory",
                "operation": "use",
                "success": True,
                "item": item_name,
                "narrative_hint": f"{player_name} uses the {item_name}.",
            }

        else:
            # Generic inventory interaction
            return {
                "action_type": "inventory",
                "operation": "interact",
                "success": True,
                "item": item_name,
                "narrative_hint": f"{player_name} interacts with the {item_name}.",
            }

    async def _consume_resources(
        self,
        resources: list[dict],
        player_name: str,
    ) -> None:
        """
        Consume resources used by an action (ammunition, consumables, etc.).

        Args:
            resources: List of {"item": "Arrow", "quantity": 1} dicts
            player_name: Character name to consume from
        """
        resolved = self._resolve_character_by_name(player_name)
        if not resolved:
            logger.warning("cannot_consume_resources_no_character", player=player_name)
            return

        char_id, character = resolved
        inventory_repo = await get_inventory_repo()

        for resource in resources:
            item_name = resource.get("item", "")
            quantity = resource.get("quantity", 1)

            if not item_name:
                continue

            # Find the item in inventory
            items = await inventory_repo.get_all_items(char_id)
            matching = [
                i for i in items
                if item_name.lower() in i.item_name.lower()
            ]

            if matching:
                item = matching[0]
                if item.quantity >= quantity:
                    await inventory_repo.remove_item(item.id, quantity)
                    logger.info(
                        "resource_consumed",
                        character=character.name,
                        item=item.item_name,
                        quantity=quantity,
                        remaining=item.quantity - quantity,
                    )
                else:
                    logger.warning(
                        "insufficient_resource",
                        character=character.name,
                        item=item_name,
                        have=item.quantity,
                        need=quantity,
                    )
            else:
                logger.debug(
                    "resource_not_found",
                    character=character.name,
                    item=item_name,
                )

    async def _consume_currency(
        self,
        currency_spent: dict,
        player_name: str,
    ) -> None:
        """
        Consume currency spent by an action (tipping, paying, dropping, etc.).

        Args:
            currency_spent: Dict like {"gold": 50} or {"silver": 10, "copper": 5}
            player_name: Character name to deduct from
        """
        if not currency_spent:
            return

        resolved = self._resolve_character_by_name(player_name)
        if not resolved:
            logger.warning("cannot_consume_currency_no_character", player=player_name)
            return

        char_id, character = resolved
        inventory_repo = await get_inventory_repo()

        # Convert all to gold equivalent and remove
        # Conversion: 1pp = 10gp, 1gp = 1gp, 1ep = 0.5gp, 1sp = 0.1gp, 1cp = 0.01gp
        total_gold = 0
        total_gold += currency_spent.get("platinum", 0) * 10
        total_gold += currency_spent.get("gold", 0)
        total_gold += currency_spent.get("electrum", 0) * 0.5
        total_gold += currency_spent.get("silver", 0) * 0.1
        total_gold += currency_spent.get("copper", 0) * 0.01

        if total_gold <= 0:
            return

        # Round up to nearest gold piece for removal
        gold_to_remove = max(1, int(total_gold)) if total_gold > 0 else 0

        success, currency = await inventory_repo.remove_gold(char_id, gold_to_remove)

        if success:
            logger.info(
                "currency_spent",
                character=character.name,
                amount=currency_spent,
                gold_equivalent=gold_to_remove,
                gold_remaining=currency.gold,
            )
        else:
            logger.warning(
                "insufficient_currency",
                character=character.name,
                wanted=currency_spent,
                gold_equivalent=gold_to_remove,
                have=currency.gold,
            )

    def _get_narrator_tools(self) -> list[dict]:
        """Select tool set based on narrator provider capability.

        Local models (Ollama) get a reduced 3-tool set — fewer tools
        means better attention in long conversations. Cloud models
        (Anthropic, Groq, OpenRouter) get the full 9-tool set.
        """
        if isinstance(self.narrator.client, OllamaClient):
            return NARRATOR_TOOLS_CORE
        return NARRATOR_TOOLS

    def _inject_tool_example(self, messages: list[dict]) -> list[dict]:
        """Inject a synthetic tool-call example into the conversation history.

        Qwen MoE follows patterns from prior assistant turns. If the history
        has no tool calls, the model writes prose-only. Inserting one example
        of prose + tool_call primes the model to continue the pattern.

        Inserted after the system prompt and before conversation history.
        """
        if not isinstance(self.narrator.client, OllamaClient):
            return messages  # Cloud models don't need priming

        # Find the insertion point: after system messages, before history
        insert_at = 0
        for i, m in enumerate(messages):
            if m["role"] == "system":
                insert_at = i + 1
            else:
                break

        example = [
            {"role": "user", "content": "[Example Player]: I look around the tavern."},
            {"role": "assistant", "content": "The tavern is dim and smoky. A barkeep wipes the counter, eyeing you.",
             "tool_calls": [{"function": {"name": "ref_entity", "arguments": {"entity_id": "barkeep"}}}]},
            {"role": "tool", "content": "ok"},
        ]

        return messages[:insert_at] + example + messages[insert_at:]

    def _extract_prose_and_effects(
        self,
        response,
        action: str = "",
    ) -> tuple[str, list[ProposedEffect]]:
        """Extract prose and effects from a narrator response.

        If the response has tool_calls, those become ProposedEffects and
        the content is pure prose. Otherwise, falls back to text-based
        PROSE/INTENTS parsing via the adjudicator.
        """
        raw_content = response.content.strip() if response.content else ""

        # Tool-based path: content is pure prose, tool_calls are effects
        if response.tool_calls:
            effects = tool_calls_to_effects(response.tool_calls)
            logger.debug(
                "narrator_tools_used",
                tool_count=len(response.tool_calls),
                effects_count=len(effects),
                action=action[:60],
            )
            return raw_content, effects

        # Text fallback: parse PROSE/INTENTS blocks
        if raw_content and validate_narrator_format(raw_content):
            prose, effects, parse_result = self.adjudicator.parse_narrator_response(raw_content)
            return prose, effects

        # No format detected — treat entire content as prose, no effects
        return raw_content, []

    async def _narrate_mechanical_result(
        self,
        action: str,
        player_name: str,
        context: BrainContext,
        triage: TriageResult,
        mechanical_result: dict,
    ) -> tuple[str, list[ProposedEffect]]:
        """
        Narrate a mechanical action result using PROSE + INTENTS architecture.

        Narrator outputs PROSE and INTENTS blocks.
        Adjudicator parses INTENTS (deterministic, no inference from prose).

        The narrator receives the known outcome and describes it dramatically.

        Returns:
            Tuple of (narrative text, proposed effects)
        """
        # Build narrator context with the mechanical result
        narrative_hint = mechanical_result.get("narrative_hint", "")
        action_type = mechanical_result.get("action_type", "action")
        success = mechanical_result.get("success", False)

        # Build enhanced context with the outcome
        enhanced_context = BrainContext(
            current_scene=context.current_scene,
            party_members=context.party_members,
            party_status=context.party_status,
            in_combat=context.in_combat,
            combat_state=context.combat_state,
            combat_round=context.combat_round,
            player_action=action,
            player_name=player_name,
        )

        # Add resolution info to context
        if success:
            resolution_text = f"[RESULT: SUCCESS] {narrative_hint}"
        else:
            resolution_text = f"[RESULT: FAILURE] {narrative_hint}"

        # =====================================================================
        # Narrator: Output prose (tools handle intents)
        # =====================================================================
        prompt = f"""The player {player_name} attempted: "{action}"

{resolution_text}

Narrate this action dramatically. Remember:
- Show the world's REACTION to the action (environment, NPCs, atmosphere)
- Connect to ongoing tension or stakes from the current scene
- End with something that maintains momentum

Write your narration directly.

IMPORTANT: You MUST call the appropriate tool for every NPC, object, or entity you mention. Call ref_entity for existing roster entities. Call add_npc for new NPCs. This is not optional."""

        if context.world_state_yaml:
            enhanced_context.world_state_yaml = context.world_state_yaml
            messages = self.narrator._build_bookend_messages(enhanced_context)
        else:
            messages = self.narrator._build_messages(enhanced_context)
        messages.append({"role": "user", "content": prompt})
        messages = self._inject_tool_example(messages)

        try:
            response = await self.narrator.client.chat(
                messages=messages,
                temperature=self.narrator.temperature,
                max_tokens=1500,
                think=False,
                frequency_penalty=NARRATOR_FREQUENCY_PENALTY,
                presence_penalty=NARRATOR_PRESENCE_PENALTY,
                tools=self._get_narrator_tools(),
                tool_choice="auto",
            )

            prose, proposed_effects = self._extract_prose_and_effects(response, action)

            if not prose:
                prose = narrative_hint

            return prose, proposed_effects

        except Exception as e:
            logger.error("narrate_mechanical_failed", error=str(e))
            return narrative_hint, []  # Fall back to the hint

    async def _resolve_mechanics(
        self,
        triage: TriageResult,
        player_name: str,
        context: BrainContext,
    ) -> MechanicalResolution:
        """
        Execute the mechanical resolution: roll dice and determine outcome.
        """
        # Get character's modifier for this check
        modifier = self._get_character_modifier(player_name, triage.ability, triage.skill)

        # Build dice notation
        notation = f"1d20{modifier:+d}" if modifier != 0 else "1d20"

        # Roll the dice
        roll_result = self.roller.roll(
            notation=notation,
            advantage=triage.advantage,
            disadvantage=triage.disadvantage,
            reason=f"{triage.skill or triage.ability} check" if triage.skill or triage.ability else "check",
        )

        # Determine success
        dc = triage.dc or 10
        success = roll_result.total >= dc
        margin = roll_result.total - dc

        # Get appropriate reveals based on success/failure
        reveals = triage.on_success if success else triage.on_failure

        logger.info(
            "mechanics_resolved",
            roll=roll_result.total,
            dc=dc,
            success=success,
            skill=triage.skill,
            reveals=reveals,
        )

        return MechanicalResolution(
            success=success,
            roll_result=roll_result,
            total=roll_result.total,
            dc=dc,
            margin=margin,
            reveals=reveals,
            skill=triage.skill,
            ability=triage.ability,
            roll_type=triage.roll_type,
        )

    def _get_character_modifier(
        self,
        player_name: str,
        ability: Optional[str],
        skill: Optional[str],
    ) -> int:
        """
        Get the character's modifier for a given ability/skill check.
        """
        if not self._current_session:
            return 0

        # Find the character by player name
        character = None
        for player in self._current_session.players.values():
            if player.character and player_name.lower() in player.character.name.lower():
                character = player.character
                break

        if not character:
            return 0

        # Get ability modifier
        ability_mod = 0
        if ability:
            ability_score = getattr(character.abilities, ability, 10)
            ability_mod = (ability_score - 10) // 2

        # Add proficiency if applicable
        proficiency_mod = 0
        if skill and skill in [p.value if hasattr(p, 'value') else p for p in character.skill_proficiencies]:
            proficiency_mod = character.proficiency_bonus
            # Check for expertise
            if skill in [e.value if hasattr(e, 'value') else e for e in character.skill_expertise]:
                proficiency_mod *= 2

        return ability_mod + proficiency_mod

    async def _narrate_action(
        self,
        action: str,
        player_name: str,
        context: BrainContext,
        triage: TriageResult,
    ) -> tuple[str, list[ProposedEffect]]:
        """
        Narrate a player action using PROSE + INTENTS architecture.

        Narrator outputs PROSE and INTENTS blocks.
        Adjudicator parses INTENTS (deterministic, no inference from prose).

        Uses narrative_direction from triage to guide the narration.

        Returns:
            Tuple of (narrative text, proposed effects)
        """
        # Get narrative direction from triage (tells narrator what to describe)
        direction = triage.narrative_direction or "Describe what happens naturally."

        # Build context for narrator with the action and direction
        enhanced_context = BrainContext(
            campaign_id=context.campaign_id,
            session_id=context.session_id,
            party_members=context.party_members,
            party_status=context.party_status,
            current_scene=context.current_scene,
            active_quests=context.active_quests,
            in_combat=context.in_combat,
            combat_state=context.combat_state,
            combat_round=context.combat_round,
            current_combatant=context.current_combatant,
            initiative_order=context.initiative_order,
            memory_context=context.memory_context,
            recent_messages=context.recent_messages,
            message_history=context.message_history,
            session_summary=context.session_summary,
            player_action=f"{action}\n\n[NARRATIVE DIRECTION: {direction}]",
            player_name=player_name,
        )

        # =====================================================================
        # Narrator: Output PROSE + INTENTS
        # Use bookend layout when world state is available (better grounding)
        # =====================================================================
        if context.world_state_yaml:
            enhanced_context.world_state_yaml = context.world_state_yaml
            messages = self.narrator._build_bookend_messages(enhanced_context)
        else:
            messages = self.narrator._build_messages(enhanced_context)
        # Style rotation + phase-specific tone
        style_hint = get_style_hint(self._scratchpad_turn)
        from ..game.world_state import PHASE_STYLE_HINTS
        world_state = getattr(self._current_session, 'world_state', None) if self._current_session else None
        phase_hint = PHASE_STYLE_HINTS.get(world_state.phase, "") if world_state else ""

        phase_line = f"**Phase tone:** {phase_hint}\n\n" if phase_hint else "\n"

        messages.append({
            "role": "system",
            "content": (
                "###INSTRUCTION###\n"
                "Narrate the player's action according to the NARRATIVE DIRECTION above.\n\n"
                f"**Style for this scene:** {style_hint}\n"
                f"{phase_line}"
                "Remember your storytelling principles:\n"
                "- Show the consequence immediately, don't echo the player's action\n"
                "- The world reacts — NPCs respond, environment shifts\n"
                "- End with a bang — something that demands a response\n\n"
                "Write your narration directly.\n\n"
                "IMPORTANT: You MUST call the appropriate tool for every NPC, object, or "
                "entity you mention. Call ref_entity for existing roster entities. "
                "Call add_npc for new NPCs. This is not optional."
            ),
        })

        # Inject tool-call example to prime local models
        messages = self._inject_tool_example(messages)

        # Use streaming if callback provided and client supports it
        # (streaming doesn't support tools, so fall back to non-streaming with tools)
        if self._on_narrative_token and hasattr(self.narrator.client, "chat_stream"):
            response = await self.narrator.client.chat_stream(
                messages=messages,
                temperature=self.narrator.temperature,
                max_tokens=1500,
                think=False,
                on_token=self._on_narrative_token,
                frequency_penalty=NARRATOR_FREQUENCY_PENALTY,
                presence_penalty=NARRATOR_PRESENCE_PENALTY,
            )
        else:
            response = await self.narrator.client.chat(
                messages=messages,
                temperature=self.narrator.temperature,
                max_tokens=1500,
                think=False,
                frequency_penalty=NARRATOR_FREQUENCY_PENALTY,
                presence_penalty=NARRATOR_PRESENCE_PENALTY,
                tools=self._get_narrator_tools(),
                tool_choice="auto",
            )

        # Extract prose + effects (tools if available, text fallback otherwise)
        prose, proposed_effects = self._extract_prose_and_effects(response, action)

        if not prose:
            logger.warning("narrator_returned_empty_for_action", action=action[:50])
            return f"*{player_name}'s action unfolds...*", []

        # If prose seems truncated (ends mid-sentence), add ellipsis
        if prose and prose[-1] not in '.!?"\'':
            prose += "..."

        return prose, proposed_effects

    async def _narrate_outcome(
        self,
        action: str,
        player_name: str,
        context: BrainContext,
        triage: TriageResult,
        resolution: Optional[MechanicalResolution],
    ) -> tuple[str, list[ProposedEffect]]:
        """
        Narrate a roll outcome using PROSE + INTENTS architecture.

        Narrator outputs PROSE and INTENTS blocks.
        Adjudicator parses INTENTS (deterministic, no inference from prose).

        The Narrator receives ONLY authorized reveals based on success/failure.

        Returns:
            Tuple of (narrative text, proposed effects)
        """
        # Build narrator context with strict constraints
        narrator_context = self._build_narrator_context(
            action, player_name, context, triage, resolution
        )

        # Create enhanced context with resolution constraints
        enhanced_context = BrainContext(
            campaign_id=context.campaign_id,
            session_id=context.session_id,
            party_members=context.party_members,
            party_status=context.party_status,
            current_scene=context.current_scene,
            active_quests=context.active_quests,
            in_combat=context.in_combat,
            combat_state=context.combat_state,
            combat_round=context.combat_round,
            current_combatant=context.current_combatant,
            initiative_order=context.initiative_order,
            memory_context=context.memory_context,
            recent_messages=context.recent_messages,
            message_history=context.message_history,
            session_summary=context.session_summary,
            player_action=f"{action}\n\n[RESOLUTION: {narrator_context}]",
            player_name=player_name,
        )

        # =====================================================================
        # Narrator: Output PROSE + INTENTS
        # Use bookend layout when world state is available
        # =====================================================================
        if context.world_state_yaml:
            enhanced_context.world_state_yaml = context.world_state_yaml
            messages = self.narrator._build_bookend_messages(enhanced_context)
        else:
            messages = self.narrator._build_messages(enhanced_context)
        messages.append({
            "role": "system",
            "content": (
                "###INSTRUCTION###\n"
                f"RESOLUTION: {narrator_context}\n\n"
                "Narrate this outcome. The RESOLUTION above is BINDING — your narration MUST reflect it:\n"
                "- SUCCESS: The character accomplishes what they attempted. Describe what they discover/achieve.\n"
                "- FAILURE: The character FAILS. They miss the clue, botch the climb, fail to persuade. "
                "Show them struggling, missing, or being wrong. Do NOT describe a soft success.\n"
                "- The MARGIN tells degree: critical success = exceptional result; narrow failure = almost but not quite; "
                "critical failure = embarrassing or dangerous consequences.\n\n"
                "Do NOT invent discoveries beyond what is authorized.\n\n"
                "Storytelling principles:\n"
                "- Show the world's REACTION to the success or failure\n"
                "- Connect to ongoing tension or stakes\n"
                "- End with forward momentum\n\n"
                "Write your narration directly.\n\n"
                "IMPORTANT: You MUST call the appropriate tool for every NPC, object, or "
                "entity you mention. Call ref_entity for existing roster entities. "
                "Call add_npc for new NPCs. This is not optional."
            ),
        })

        messages = self._inject_tool_example(messages)

        response = await self.narrator.client.chat(
            messages=messages,
            temperature=self.narrator.temperature,
            max_tokens=1500,
            think=False,
            frequency_penalty=NARRATOR_FREQUENCY_PENALTY,
            presence_penalty=NARRATOR_PRESENCE_PENALTY,
            tools=self._get_narrator_tools(),
            tool_choice="auto",
        )

        prose, proposed_effects = self._extract_prose_and_effects(response, action)

        if not prose:
            logger.warning("narrator_returned_empty", action=action[:50])
            return f"*{player_name} attempts to {action.lower()}...*", []

        # Fix truncated endings
        if prose and prose[-1] not in '.!?"\'':
            prose += "..."

        return prose, proposed_effects

    def _build_narrator_context(
        self,
        action: str,
        player_name: str,
        context: BrainContext,
        triage: TriageResult,
        resolution: Optional[MechanicalResolution],
    ) -> str:
        """Build context string for narrator with resolution constraints."""
        parts = []

        if resolution:
            outcome_word = "SUCCESS" if resolution.success else "FAILURE"
            skill_label = resolution.skill or resolution.ability or triage.action_type
            parts.append(f"{skill_label.title()} Check: {outcome_word}")
            parts.append(f"Rolled {resolution.total} vs DC {resolution.dc} (margin: {resolution.margin:+d})")

            # Degree of success/failure affects narration
            abs_margin = abs(resolution.margin)
            if resolution.success:
                if abs_margin >= 10:
                    parts.append("CRITICAL SUCCESS — reveal everything clearly, with bonus detail")
                elif abs_margin >= 5:
                    parts.append("SOLID SUCCESS — reveal the information clearly")
                else:
                    parts.append("NARROW SUCCESS — reveal the basics, but incompletely")
            else:
                if abs_margin >= 10:
                    parts.append("CRITICAL FAILURE — the character is completely wrong or oblivious, with consequences")
                elif abs_margin >= 5:
                    parts.append("CLEAR FAILURE — the character misses the mark, narrate what they fail to notice or accomplish")
                else:
                    parts.append("NARROW FAILURE — the character almost succeeds but falls short, hint at what they missed")

            if resolution.reveals:
                parts.append(f"AUTHORIZED REVEALS (narrate ONLY these): {', '.join(resolution.reveals)}")

            parts.append("Do NOT invent additional discoveries or outcomes beyond what is listed above.")
        else:
            # No roll needed - free narrative within triage constraints
            if triage.on_success:
                parts.append(f"The action proceeds. Include these details: {', '.join(triage.on_success)}")
            else:
                # Always give narrator something to work with
                parts.append("The action proceeds naturally. Describe what the character experiences.")

        return " | ".join(parts)

    # =========================================================================
    # WORLD STATE EXTRACTION
    # =========================================================================

    async def _extract_and_apply_state_delta(
        self,
        narrative: str,
        world_state: "WorldState",
        context: BrainContext,
        referenced_entity_ids: list[str] | None = None,
    ) -> Optional["StateDelta"]:
        """Extract a StateDelta from narrator prose and apply to WorldState.

        Also syncs NPC changes back to SceneEntityRegistry so the existing
        entity system stays in sync. Returns the delta for downstream use
        (e.g., knowledge graph bridge).

        Args:
            referenced_entity_ids: Entity IDs the narrator explicitly tagged
                via ref_entity intents. Passed to the extractor so it knows
                which entities are already accounted for and avoids creating
                duplicates.
        """
        from ..game.world_state import WorldState, NPCState

        try:
            delta = await self._state_extractor.extract(
                narrative_text=narrative,
                world_state_yaml=context.world_state_yaml,
                current_scene=context.current_scene,
                referenced_entity_ids=referenced_entity_ids,
            )

            # Apply delta (validates internally, returns rejections)
            rejections = world_state.apply_delta(delta)

            if rejections:
                logger.debug(
                    "state_delta_applied_with_rejections",
                    turn=world_state.turn,
                    rejections=rejections,
                )

            # Sync new/updated NPCs to SceneEntityRegistry
            if self._scene_registry:
                self._sync_npcs_to_registry(delta, world_state)

            logger.info(
                "world_state_updated",
                turn=world_state.turn,
                location=world_state.current_location,
                time=world_state.time_of_day,
                npc_count=len(world_state.npcs),
                facts_count=len(world_state.established_facts),
            )

            return delta

        except Exception as e:
            logger.warning("state_delta_extraction_failed", error=str(e))
            return None

    def _sync_effect_to_world_state(self, effect: ProposedEffect) -> None:
        """Sync a successfully executed effect into WorldState.

        This is the critical bridge: INTENTS execute mechanically via the
        effect system, and here we record them into WorldState so the
        narrator sees them in the YAML snapshot next turn.
        """
        world_state = getattr(self._current_session, 'world_state', None) if self._current_session else None
        if not world_state:
            return

        etype = effect.effect_type

        if etype == EffectType.SPAWN_OBJECT:
            obj_id = effect.object_name or "unknown_item"
            desc = effect.object_description or effect.object_name or "an object"
            world_state.spawn_item(obj_id, desc)
            world_state.record_transfer(f"{desc} appeared in the scene")

        elif etype == EffectType.TRANSFER_ITEM:
            item = effect.item_name or "an item"
            src = effect.from_entity or "somewhere"
            dst = effect.to_entity or "someone"
            # If player picked up from scene, remove from scene items
            if src.startswith("scene"):
                world_state.remove_item(effect.object_name or effect.item_name or "")
            world_state.record_transfer(f"{item} moved from {src} to {dst}")

        elif etype == EffectType.GRANT_CURRENCY:
            parts = []
            if effect.gold: parts.append(f"{effect.gold}gp")
            if effect.silver: parts.append(f"{effect.silver}sp")
            if effect.copper: parts.append(f"{effect.copper}cp")
            if effect.platinum: parts.append(f"{effect.platinum}pp")
            if effect.electrum: parts.append(f"{effect.electrum}ep")
            amount = ", ".join(parts) if parts else "currency"
            src = effect.source or "someone"
            dst = effect.target or "player"
            world_state.record_transfer(f"{src} gave {amount} to {dst}")

        elif etype == EffectType.APPLY_DAMAGE:
            target = effect.target or "someone"
            amount = effect.amount or 0
            dtype = effect.damage_type or "damage"
            world_state.record_transfer(f"{target} took {amount} {dtype} damage")

        elif etype == EffectType.APPLY_HEALING:
            target = effect.target or "someone"
            amount = effect.amount or 0
            world_state.record_transfer(f"{target} healed {amount} HP")

        elif etype == EffectType.ADD_CONDITION:
            target = effect.target or "someone"
            condition = effect.condition or "a condition"
            world_state.record_transfer(f"{target} gained condition: {condition}")

        elif etype == EffectType.REMOVE_CONDITION:
            target = effect.target or "someone"
            condition = effect.condition or "a condition"
            world_state.record_transfer(f"{target} lost condition: {condition}")

        elif etype == EffectType.CONSUME_RESOURCE:
            resource = effect.resource_name or effect.item_name or "a resource"
            world_state.record_transfer(f"Consumed {effect.quantity}x {resource}")

    def _sync_npcs_to_registry(
        self,
        delta: "StateDelta",
        world_state: "WorldState",
    ) -> None:
        """Sync NPC changes from WorldState back to SceneEntityRegistry.

        Keeps the existing entity system (hostility, SRD matching, combat triggers)
        working alongside WorldState.
        """
        from ..models.npc import SceneEntity, EntityType, Disposition
        from ..game.world_state import StateDelta

        if not self._scene_registry:
            return

        # Register new NPCs
        for npc_state in delta.new_npcs:
            # Only add to registry if they're at the party's location
            if not npc_state.location or npc_state.location == world_state.current_location:
                disposition_map = {
                    "hostile": Disposition.HOSTILE,
                    "unfriendly": Disposition.UNFRIENDLY,
                    "neutral": Disposition.NEUTRAL,
                    "friendly": Disposition.FRIENDLY,
                    "allied": Disposition.ALLIED,
                }
                entity = SceneEntity(
                    name=npc_state.name,
                    entity_type=EntityType.NPC,
                    description=npc_state.description,
                    disposition=disposition_map.get(npc_state.disposition, Disposition.NEUTRAL),
                )
                self._scene_registry.register_entity(entity)

        # Update dispositions for existing NPCs
        for update in delta.npc_updates:
            if update.disposition:
                existing = self._scene_registry.get_by_name(update.name)
                if existing:
                    disposition_map = {
                        "hostile": Disposition.HOSTILE,
                        "unfriendly": Disposition.UNFRIENDLY,
                        "neutral": Disposition.NEUTRAL,
                        "friendly": Disposition.FRIENDLY,
                        "allied": Disposition.ALLIED,
                    }
                    new_disp = disposition_map.get(update.disposition)
                    if new_disp:
                        existing.disposition = new_disp

        # Remove NPCs who left
        for name in delta.removed_npcs:
            self._scene_registry.remove_by_name(name)

    # =========================================================================
    # ENTITY EXTRACTION AND COMBAT TRIGGERING
    # =========================================================================

    async def _process_narrator_output(
        self,
        narrative: str,
        context: BrainContext,
        proposed_effects: Optional[list[ProposedEffect]] = None,
        message_id: Optional[str] = None,
        skip_entity_extraction: bool = False,
    ) -> bool:
        """
        Process narrator output: extract entities and execute proposed effects.

        Flow:
        1. Entity extraction → Scene registry (NPCs, creatures mentioned in prose)
        2. Proposed effects → Validate and execute (structured, not sniffed from prose)

        Returns True if combat was triggered.
        """
        combat_triggered = False

        # Entity extraction — skip during combat or simple skill checks to save API calls
        if self._scene_registry and not skip_entity_extraction:
            existing = [e.name for e in self._scene_registry.get_all()]
            try:
                entity_result = await self._entity_extractor.extract(
                    narrative_text=narrative,
                    current_scene=context.current_scene or "",
                    existing_entities=existing,
                )

                if entity_result:
                    # Register new entities
                    for extracted in entity_result.entities:
                        entity = self._entity_extractor.convert_to_scene_entity(extracted)
                        self._scene_registry.register_entity(entity)

                    # Update scene description if provided
                    if entity_result.scene_update:
                        self._scene_registry.set_scene_description(entity_result.scene_update)

                    # Combat trigger: LLM determined combat actually started
                    if entity_result.combat_initiated:
                        hostile_entities = [
                            e for e in self._scene_registry.get_all_entities()
                            if e.disposition in (Disposition.HOSTILE, Disposition.UNFRIENDLY)
                            and e.entity_type in (EntityType.NPC, EntityType.CREATURE)
                        ]
                        if hostile_entities:
                            logger.warning(
                                "combat_initiated_by_narrative",
                                hostile_count=len(hostile_entities),
                                hostiles=[e.name for e in hostile_entities],
                            )
                            combat_triggered = await self._trigger_combat(hostile_entities)

                    # Persist entities to DB immediately (not just at session end)
                    try:
                        await self._scene_registry.sync_to_npc_repo()
                    except Exception as sync_err:
                        logger.warning("entity_sync_failed", error=str(sync_err))

            except Exception as e:
                logger.warning("entity_extraction_failed", error=str(e))

        # Process proposed effects (replaces mechanics extraction)
        if proposed_effects:
            effect_combat = await self._process_proposed_effects(
                proposed_effects,
                context,
                message_id,
            )
            combat_triggered = combat_triggered or effect_combat

        return combat_triggered

    async def _process_proposed_effects(
        self,
        effects: list[ProposedEffect],
        context: BrainContext,
        message_id: Optional[str] = None,
    ) -> bool:
        """
        Validate and execute proposed effects from narrator.

        This is the core of the "Narrator proposes, Rules validates" pattern.
        Returns True if any effect triggered combat.
        """
        if not effects:
            return False

        # Lazy init executor with inventory repo
        if not self._effect_executor:
            inventory_repo = await get_inventory_repo()
            self._effect_executor = EffectExecutor(
                scene_registry=self._scene_registry,
                session=self._current_session,
                inventory_repo=inventory_repo,
                applied_effects_store=self._applied_effects,
            )

        # Ensure validator exists
        if not self._effect_validator:
            self._effect_validator = EffectValidator(
                scene_registry=self._scene_registry,
                session=self._current_session,
            )

        combat_triggered = False
        campaign_id = context.campaign_id or "unknown"
        msg_id = message_id or str(uuid.uuid4())

        for i, effect in enumerate(effects):
            # Build idempotency key
            idem_key = build_effect_idempotency_key(campaign_id, msg_id, i)

            # Skip effects requiring confirmation (handled separately via UI)
            if effect.requires_confirmation:
                logger.info(
                    "effect_requires_confirmation",
                    effect_type=effect.effect_type.value,
                    prompt=effect.confirmation_prompt,
                )
                # TODO: Queue for player confirmation via Discord UI
                continue

            # Validate
            validation = self._effect_validator.validate(effect)
            if not validation.valid:
                logger.warning(
                    "effect_validation_failed",
                    effect_type=effect.effect_type.value,
                    reason=validation.rejection_reason,
                )
                continue

            # Execute
            result = await self._effect_executor.execute(effect, idem_key)

            if result.success:
                logger.info(
                    "effect_applied",
                    effect_type=effect.effect_type.value,
                    details=result.details,
                    was_duplicate=result.was_duplicate,
                )

                # Sync effect to WorldState (so narrator sees it next turn)
                self._sync_effect_to_world_state(effect)

                # Check if this effect triggers combat
                if effect.effect_type == EffectType.START_COMBAT:
                    combat_triggered = True
            else:
                logger.warning(
                    "effect_execution_failed",
                    effect_type=effect.effect_type.value,
                    error=result.error,
                )

        return combat_triggered

    async def _check_player_attack_initiation(
        self,
        action: str,
        context: BrainContext,
    ) -> bool:
        """
        Check if the player is initiating combat with an attack.

        If so, triggers combat with enemies marked as surprised.
        Returns True if combat was initiated.
        """
        # Skip if combat already active (from context or existing manager)
        if context.in_combat:
            return False

        # Skip if no scene registry
        if not self._scene_registry:
            return False

        # Also check if combat was just created (context might be stale)
        from ..game.combat.manager import get_combat_for_channel
        if self._current_session:
            existing = get_combat_for_channel(self._current_session.channel_id)
            if existing:
                return False

        # Check for attack-like language
        action_lower = action.lower()
        attack_keywords = [
            "attack", "strike", "hit", "stab", "slash", "shoot",
            "swing at", "punch", "kick", "cast", "fireball", "fire at",
            "throw at", "charge at", "lunge at", "ambush", "kill",
        ]

        is_attack = any(keyword in action_lower for keyword in attack_keywords)
        if not is_attack:
            return False

        # Look for a target entity mentioned in the action
        # NOTE: Do NOT filter by disposition — if a player says "I attack the sailor"
        # and the sailor is friendly, they should still be able to attack them. This is D&D.
        target_entity = None
        best_score = 0
        for entity in self._scene_registry.get_all_entities():
            score = self._entity_name_match_score(entity, action_lower)
            if score > best_score:
                best_score = score
                target_entity = entity

        if not target_entity:
            # No specific target matched by name. If player clearly wants to fight
            # (e.g., "attack the closest enemy", "I fight them"), pick any hostile/unfriendly entity
            generic_attack_phrases = [
                "attack", "fight", "kill", "ambush", "charge",
                "closest enemy", "nearest enemy", "nearest foe", "closest foe",
                "them", "the enemy", "the enemies", "the creature",
            ]
            is_generic_attack = any(phrase in action_lower for phrase in generic_attack_phrases)

            if is_generic_attack:
                # Pick the most hostile non-friendly entity
                candidates = [
                    e for e in self._scene_registry.get_all_entities()
                    if e.disposition not in ("friendly", "allied")
                    and e.entity_type in ("npc", "creature")
                ]
                if candidates:
                    # Sort by hostility score descending, pick most hostile
                    candidates.sort(key=lambda e: e.hostility_score, reverse=True)
                    target_entity = candidates[0]
                    logger.info(
                        "generic_attack_matched_entity",
                        action=action[:50],
                        target=target_entity.name,
                        hostility=target_entity.hostility_score,
                    )

        if not target_entity:
            # No entity found in registry, but player clearly wants to attack.
            # Create the target from the action text and start combat anyway.
            # The narrator already described this entity — we just didn't register it.
            target_words = [w for w in action_lower.split() if w not in (
                "i", "attack", "the", "a", "an", "begin", "start", "want", "to",
                "my", "with", "at", "strike", "hit", "shoot", "kill", "fight",
            ) and len(w) > 2]
            target_name = " ".join(target_words).strip().title() if target_words else "Enemy"

            monster_index = self._guess_monster_index(target_name)
            from ..models.npc import SceneEntity as _SE, EntityType as _ET, Disposition as _D
            target_entity = _SE(
                name=target_name,
                entity_type=_ET.NPC,
                description="Hostile target",
                disposition=_D.HOSTILE,
                monster_index=monster_index,
            )
            if self._scene_registry:
                self._scene_registry.register_entity(target_entity)

        logger.info(
            "player_attack_detected",
            action=action[:50],
            target=target_entity.name,
        )

        # Gather combat participants: the target + other hostile NPCs/creatures
        # Only include actual beings (not objects/environmental), and only if they
        # are clearly hostile (not neutral bystanders)
        hostile_entities = [target_entity]
        for entity in self._scene_registry.get_all_entities():
            if entity.id == target_entity.id:
                continue
            # Must be an NPC or creature (not objects, not environmental effects)
            if entity.entity_type not in (EntityType.NPC, EntityType.CREATURE):
                continue
            # Must be actively hostile (not just unfriendly bystanders)
            if entity.disposition == Disposition.HOSTILE:
                hostile_entities.append(entity)

        # Trigger combat with surprise (player initiated)
        return await self._trigger_combat(hostile_entities, player_initiated=True)

    @staticmethod
    def _entity_name_match_score(entity: "SceneEntity", text_lower: str) -> int:
        """Score how well an entity name matches player text. Higher = better. 0 = no match."""
        name_lower = entity.name.lower()

        # Exact full name match
        if name_lower in text_lower:
            return 100

        # Check aliases
        for alias in (entity.aliases or []):
            if alias.lower() in text_lower:
                return 90

        # Check if any significant word from entity name appears in text
        # Skip common words like "the", "a", "of"
        skip_words = {"the", "a", "an", "of", "at", "in", "on", "to", "and", "or"}
        entity_words = [w for w in name_lower.split() if w not in skip_words and len(w) > 2]
        text_words = set(text_lower.split())

        matching_words = [w for w in entity_words if w in text_words]
        if matching_words:
            # Score by fraction of significant entity words that matched
            return int(60 * len(matching_words) / len(entity_words)) if entity_words else 0

        # Check if any text word is a substring of entity name (e.g., "leader" in "assassin leader")
        for word in text_words:
            if len(word) > 3 and word in name_lower:
                return 40

        return 0

    def _get_max_cr_for_party(self) -> float:
        """Get the maximum CR monster appropriate for the current party."""
        if not self._current_session:
            return 1

        levels = []
        for player in self._current_session.players.values():
            if player.character:
                levels.append(player.character.level)

        if not levels:
            return 1

        avg_level = sum(levels) / len(levels)
        party_size = len(levels)

        # Rough CR budget: avg_level for a full party of 4, scaled down for fewer
        # Solo player: max CR ~= level * 0.5
        # 2 players: max CR ~= level * 0.75
        # 3-4 players: max CR ~= level
        scale = min(1.0, party_size / 4)
        return max(0.5, avg_level * scale)

    def _guess_monster_index(self, entity_name: str) -> Optional[str]:
        """Try to find an SRD monster index from a narrative entity name.

        Uses keyword matching against SRD monster names.
        e.g., 'Hooded Figure' might match 'bandit', 'Ash-clad Intruder' → 'cult-fanatic'
        """
        from ..data.srd.loader import get_srd
        srd = get_srd()

        name_lower = entity_name.lower()

        # Try direct SRD lookup first (e.g., "goblin" → "goblin")
        simple_index = name_lower.replace(" ", "-").replace("'", "")
        monster = srd.get_monster(simple_index)
        if monster:
            return simple_index

        # Try each word from the name as an SRD index
        for word in name_lower.split():
            if len(word) > 3:
                monster = srd.get_monster(word)
                if monster:
                    return word

        # Common narrative-to-SRD fallbacks
        fallbacks = {
            "guard": "guard", "soldier": "guard", "watchman": "guard",
            "thug": "thug", "brute": "thug", "enforcer": "thug",
            "bandit": "bandit", "robber": "bandit", "brigand": "bandit",
            "assassin": "assassin", "killer": "assassin",
            "mage": "mage", "wizard": "mage", "sorcerer": "mage", "spellcaster": "mage",
            "cultist": "cultist", "fanatic": "cult-fanatic", "zealot": "cult-fanatic",
            "knight": "knight", "champion": "knight", "paladin": "knight",
            "priest": "priest", "cleric": "priest", "healer": "priest",
            "spy": "spy", "rogue": "spy", "scout": "scout",
            "wolf": "wolf", "bear": "brown-bear", "rat": "giant-rat",
            "skeleton": "skeleton", "zombie": "zombie", "ghoul": "ghoul",
            "figure": "bandit", "stranger": "bandit", "intruder": "bandit",
        }
        for word in name_lower.split():
            if word in fallbacks:
                return fallbacks[word]

        return None

    @staticmethod
    def _detect_group_count(name: str) -> int:
        """Detect if an entity name represents a group and return the count.

        Returns 1 for singular entities, >1 for groups.
        Examples: "Goblins" → 3, "Three Bandits" → 3, "Goblin" → 1
        """
        import re
        name_lower = name.lower().strip()

        # Check for explicit number words
        number_words = {
            "two": 2, "three": 3, "four": 4, "five": 5, "six": 6,
            "pair": 2, "couple": 2, "trio": 3,
        }
        for word, count in number_words.items():
            if word in name_lower:
                return count

        # Check for digit prefix: "3 goblins"
        digit_match = re.match(r'^(\d+)\s+', name_lower)
        if digit_match:
            return min(int(digit_match.group(1)), 6)  # Cap at 6

        # Check for simple plural (ends in 's' but not 'ss')
        # Common D&D creature names that are plural: goblins, bandits, wolves, skeletons
        if name_lower.endswith('s') and not name_lower.endswith('ss'):
            return 3  # Default group size for unnamed plurals

        return 1

    @staticmethod
    def _singularize_name(name: str) -> str:
        """Convert a plural group name to singular for individual combatants."""
        import re
        # Strip number prefixes: "Three Goblins" → "Goblins", "3 Bandits" → "Bandits"
        name = re.sub(r'^(?:two|three|four|five|six|pair|couple|trio|\d+)\s+', '', name, flags=re.IGNORECASE).strip()

        # Basic singularize: "Goblins" → "Goblin", "Wolves" → "Wolf"
        if name.lower().endswith('ves'):
            return name[:-3] + 'f'  # wolves → wolf
        if name.lower().endswith('ies'):
            return name[:-3] + 'y'  # harpies → harpy
        if name.lower().endswith('s') and not name.lower().endswith('ss'):
            return name[:-1]

        return name

    async def _initiate_combat_from_attack(
        self,
        target_name: Optional[str],
        player_name: str,
    ) -> bool:
        """
        Try to initiate combat when player attacks a creature.

        Looks up the target in the scene registry and triggers combat
        if found. Returns True if combat was started, False if target
        not found (caller should let narrator handle it).

        Args:
            target_name: Name of the creature being attacked
            player_name: Name of the attacking player (for logging)
        """
        if not self._scene_registry or not target_name:
            return False

        # Try to find the target entity - exact first, then fuzzy
        target_entity = self._scene_registry.get_by_name(target_name)

        if not target_entity:
            # Fuzzy match using scoring — require minimum threshold of 30
            # to avoid grabbing the wrong entity on weak partial matches
            target_lower = target_name.lower()
            best_score = 0
            best_candidate = None
            for entity in self._scene_registry.get_all_entities():
                score = self._entity_name_match_score(entity, target_lower)
                if score > best_score:
                    best_score = score
                    best_candidate = entity

            if best_candidate and best_score >= 30:
                target_entity = best_candidate
            elif best_candidate:
                logger.debug(
                    "fuzzy_match_below_threshold",
                    target=target_name,
                    best_match=best_candidate.name,
                    score=best_score,
                )

        if not target_entity:
            logger.debug(
                "attack_target_not_found_in_registry",
                target=target_name,
                player=player_name,
            )
            return False

        # Gather hostile entities: target + other actively hostile NPCs/creatures
        hostile_entities = [target_entity]
        for entity in self._scene_registry.get_all_entities():
            if entity.id == target_entity.id:
                continue
            if entity.entity_type not in (EntityType.NPC, EntityType.CREATURE):
                continue
            if entity.disposition == Disposition.HOSTILE:
                hostile_entities.append(entity)

        logger.info(
            "initiating_combat_from_player_attack",
            target=target_entity.name,
            player=player_name,
            hostile_count=len(hostile_entities),
        )

        # Trigger combat with surprise (player initiated)
        return await self._trigger_combat(hostile_entities, player_initiated=True)

    async def _trigger_combat(
        self,
        hostile_entities: list[SceneEntity],
        player_initiated: bool = False,
    ) -> bool:
        """
        Trigger combat when hostility threshold is crossed or player attacks.

        Creates a combat encounter with hostile entities as combatants.

        Args:
            hostile_entities: Entities to add as enemy combatants
            player_initiated: If True, enemies are surprised (player ambush)
        """
        if not self._current_session:
            logger.warning("cannot_trigger_combat_no_session")
            return False

        from ..game.combat.manager import (
            CombatManager,
            set_combat_for_channel,
            get_combat_for_channel,
        )

        # Check if combat already exists (idempotent - don't create duplicates)
        existing_combat = get_combat_for_channel(self._current_session.channel_id)
        if existing_combat:
            logger.info(
                "combat_already_exists",
                channel_id=self._current_session.channel_id,
                combat_id=existing_combat.combat.id,
            )
            return True  # Combat exists, so "triggered" is true

        # Get entity names for logging
        hostile_names = [e.name for e in hostile_entities]

        logger.warning(
            "auto_triggering_combat",
            hostile_count=len(hostile_entities),
            hostiles=hostile_names,
            player_initiated=player_initiated,
        )

        try:
            # Create combat encounter
            description = f"Combat erupts with {', '.join(hostile_names)}!"
            if player_initiated:
                description = f"Your surprise attack catches them off guard! {description}"

            combat = CombatManager.create_encounter(
                session_id=self._current_session.id,
                channel_id=self._current_session.channel_id,
                name="Combat",
                description=description,
            )

            # Add player combatants
            for player in self._current_session.players.values():
                if player.character:
                    combat.add_player(player.character)

            # Add hostile entities as combatants
            for entity in hostile_entities:
                combatant = None
                monster_index = entity.monster_index

                # If no monster_index, try fuzzy SRD lookup by name
                if not monster_index:
                    monster_index = self._guess_monster_index(entity.name)

                # Detect group entities (plural names like "Goblins", "Three Bandits")
                # and spawn multiple individual combatants
                count = self._detect_group_count(entity.name)
                if count > 1:
                    # Singular name for individual combatants
                    singular = self._singularize_name(entity.name)
                    for i in range(count):
                        individual_name = f"{singular} {i + 1}" if count > 1 else singular
                        ind_combatant = None
                        if monster_index:
                            try:
                                ind_combatant = combat.add_monster(monster_index, name=individual_name)
                            except Exception:
                                pass
                        if not ind_combatant:
                            ind_combatant = combat.add_custom_combatant(
                                name=individual_name,
                                hp=entity.hp_estimate or 20,
                                ac=entity.ac_estimate or 12,
                                is_player=False,
                            )
                        if ind_combatant and player_initiated:
                            ind_combatant.is_surprised = True
                    continue  # Skip the single-combatant path below

                if monster_index:
                    # CR cap: don't spawn monsters too strong for the party
                    max_cr = self._get_max_cr_for_party()
                    try:
                        from ..data.srd.loader import get_srd
                        srd = get_srd()
                        monster_data = srd.get_monster(monster_index)
                        if monster_data:
                            cr = monster_data.get("challenge_rating", 0)
                            if cr > max_cr:
                                # Downgrade to a weaker variant
                                logger.info(
                                    "monster_cr_capped",
                                    monster=monster_index,
                                    cr=cr,
                                    max_cr=max_cr,
                                    entity=entity.name,
                                )
                                # Use the base type (e.g., bandit-captain → bandit)
                                base_index = monster_index.split("-")[0] if "-" in monster_index else None
                                if base_index and srd.get_monster(base_index):
                                    monster_index = base_index
                        combatant = combat.add_monster(monster_index, name=entity.name)
                    except Exception as e:
                        logger.warning(
                            "monster_not_found_using_custom",
                            monster_index=monster_index,
                            error=str(e),
                        )

                # Fallback: create custom combatant with reasonable defaults
                if not combatant:
                    combatant = combat.add_custom_combatant(
                        name=entity.name,
                        hp=entity.hp_estimate or 20,
                        ac=entity.ac_estimate or 12,
                        is_player=False,
                    )

                # SURPRISE: If player initiated, enemies are surprised
                if combatant and player_initiated:
                    combatant.is_surprised = True
                    logger.info("combatant_surprised", combatant=combatant.name)

            # Store combat in session AND register globally for cog access
            self._current_session.combat_manager = combat
            set_combat_for_channel(self._current_session.channel_id, combat)

            # Roll initiative immediately so combat is ready
            combat.roll_all_initiative()
            combat.start_combat()

            # Note: State transition should be handled by session manager
            # We return True to signal combat was triggered

            logger.info(
                "combat_auto_created",
                combatant_count=len(combat.combat.combatants),
                hostile_count=len(hostile_entities),
                enemies_surprised=player_initiated,
            )

            return True

        except Exception as e:
            logger.error("combat_trigger_failed", error=str(e))
            return False

    # =========================================================================
    # TOOL EXECUTION (preserved from original for combat/inventory/etc)
    # =========================================================================

    async def _execute_tool(self, tool_call: dict) -> ToolExecutionResult:
        """Execute a single tool call."""
        tool_name = tool_call.get("name", "")
        arguments = tool_call.get("arguments", {})

        logger.debug("executing_tool", tool=tool_name, args=arguments)

        try:
            if tool_name == "roll_dice":
                result = self._execute_roll_dice(arguments)
            elif tool_name == "update_hp":
                result = await self._execute_update_hp(arguments)
            elif tool_name == "apply_condition":
                result = await self._execute_apply_condition(arguments)
            elif tool_name == "remove_condition":
                result = await self._execute_remove_condition(arguments)
            elif tool_name == "expend_spell_slot":
                result = await self._execute_expend_spell_slot(arguments)
            elif tool_name == "check_rule":
                result = await self._execute_check_rule(arguments)
            elif tool_name == "use_action":
                result = await self._execute_use_action(arguments)
            # Commerce and Inventory tools
            elif tool_name == "purchase_item":
                result = await self._execute_purchase_item(arguments)
            elif tool_name == "sell_item":
                result = await self._execute_sell_item(arguments)
            elif tool_name == "modify_gold":
                result = await self._execute_modify_gold(arguments)
            elif tool_name == "add_item":
                result = await self._execute_add_item(arguments)
            elif tool_name == "remove_item":
                result = await self._execute_remove_item(arguments)
            elif tool_name == "equip_item":
                result = await self._execute_equip_item(arguments)
            elif tool_name == "unequip_item":
                result = await self._execute_unequip_item(arguments)
            elif tool_name == "transfer_item":
                result = await self._execute_transfer_item(arguments)
            else:
                return ToolExecutionResult(
                    tool_name=tool_name,
                    success=False,
                    result=None,
                    error=f"Unknown tool: {tool_name}",
                )

            return ToolExecutionResult(
                tool_name=tool_name,
                success=True,
                result=result,
            )

        except Exception as e:
            logger.error("tool_execution_error", tool=tool_name, error=str(e))
            return ToolExecutionResult(
                tool_name=tool_name,
                success=False,
                result=None,
                error=str(e),
            )

    def _execute_roll_dice(self, args: dict) -> DiceRoll:
        """Execute a dice roll."""
        notation = args.get("notation", "1d20")
        advantage = args.get("advantage", False)
        disadvantage = args.get("disadvantage", False)
        reason = args.get("reason", "")

        return self.roller.roll(
            notation=notation,
            advantage=advantage,
            disadvantage=disadvantage,
            reason=reason,
        )

    def _resolve_character_by_name(self, name: str) -> Optional[tuple[str, "Character"]]:
        """Resolve a character by name within the current session.

        Uses ranked matching: exact match > starts-with > substring.
        Only returns a result if the match is unambiguous.
        """
        if not self._current_session:
            return None

        name_lower = name.lower().strip()
        candidates = []

        for player in self._current_session.players.values():
            if not player.character:
                continue
            char_name = player.character.name.lower()

            if char_name == name_lower:
                # Exact match — return immediately
                return (player.character.id, player.character)

            if char_name.startswith(name_lower):
                candidates.append((0, player.character))  # Priority 0: starts-with
            elif name_lower in char_name:
                candidates.append((1, player.character))  # Priority 1: substring

        if not candidates:
            return None

        # Sort by priority, return best match only if unambiguous at that tier
        candidates.sort(key=lambda x: x[0])
        best_priority = candidates[0][0]
        best_tier = [c for c in candidates if c[0] == best_priority]

        if len(best_tier) == 1:
            char = best_tier[0][1]
            return (char.id, char)

        # Ambiguous match — log and return None
        logger.warning(
            "ambiguous_character_match",
            query=name,
            matches=[c[1].name for c in best_tier],
        )
        return None

    def _resolve_combatant_by_name(self, name: str) -> Optional[tuple[str, Any]]:
        """Resolve a combatant by name in the current combat."""
        if not self._current_session or not self._current_session.combat_manager:
            return None

        combatant = self._current_session.combat_manager.get_combatant_by_name(name)
        if combatant:
            return (combatant.id, combatant)
        return None

    async def _execute_update_hp(self, args: dict) -> dict:
        """Execute HP update."""
        target = args.get("target", args.get("target_id", ""))
        delta = args.get("delta", 0)
        source = args.get("source", "")
        damage_type = args.get("damage_type", "")
        tx_id = args.get("transaction_id")

        resolved = self._resolve_character_by_name(target)
        if resolved:
            char_id, character = resolved

            if tx_id:
                tx_key = generate_transaction_key("hp", char_id, tx_id)
                tx_repo = await get_transaction_repo()
                cached = await tx_repo.get_result(tx_key)
                if cached:
                    return cached

            hp_before = character.hp.current
            new_hp = max(0, min(character.hp.maximum, hp_before + delta))

            repo = await get_character_repo()
            await repo.update_hp(char_id, new_hp, character.hp.temporary)
            character.hp.current = new_hp

            result = {
                "target": character.name,
                "target_id": char_id,
                "delta": delta,
                "hp_before": hp_before,
                "hp_after": new_hp,
                "damage_type": damage_type,
                "source": source,
                "applied": True,
            }

            if tx_id:
                await tx_repo.record(tx_key, "update_hp", char_id, result, args)

            return result

        # Check combat manager for non-player combatants
        combat_resolved = self._resolve_combatant_by_name(target)
        if combat_resolved and self._current_session and self._current_session.combat_manager:
            combatant_id, combatant = combat_resolved
            hp_before = combatant.hp_current

            if delta < 0:
                actual, unconscious, instant_death = self._current_session.combat_manager.apply_damage(
                    combatant_id, abs(delta), damage_type
                )
                delta = -actual
            else:
                actual, revived = self._current_session.combat_manager.apply_healing(combatant_id, delta)
                delta = actual

            return {
                "target": combatant.name,
                "target_id": combatant_id,
                "delta": delta,
                "hp_before": hp_before,
                "hp_after": combatant.hp_current,
                "damage_type": damage_type,
                "source": source,
                "applied": True,
            }

        return {
            "target": target,
            "delta": delta,
            "source": source,
            "applied": False,
            "error": f"Target '{target}' not found",
        }

    async def _execute_apply_condition(self, args: dict) -> dict:
        """Apply a condition to a target."""
        target = args.get("target", args.get("target_id", ""))
        condition_name = args.get("condition", "")
        source = args.get("source", "")
        duration = args.get("duration_rounds")
        tx_id = args.get("transaction_id")

        try:
            condition_enum = Condition(condition_name.lower())
        except ValueError:
            return {
                "target": target,
                "condition": condition_name,
                "applied": False,
                "error": f"Invalid condition: {condition_name}",
            }

        resolved = self._resolve_character_by_name(target)
        if resolved:
            char_id, character = resolved

            if tx_id:
                tx_key = generate_transaction_key("condition", char_id, tx_id)
                tx_repo = await get_transaction_repo()
                cached = await tx_repo.get_result(tx_key)
                if cached:
                    return cached

            char_condition = CharacterCondition(
                id=str(uuid.uuid4()),
                condition=condition_enum,
                source=source,
                expires_round=duration,
            )

            repo = await get_character_repo()
            await repo.add_condition(char_id, char_condition)
            character.conditions.append(char_condition)

            result = {
                "target": character.name,
                "target_id": char_id,
                "condition": condition_name,
                "source": source,
                "duration_rounds": duration,
                "applied": True,
            }

            if tx_id:
                await tx_repo.record(tx_key, "apply_condition", char_id, result, args)

            return result

        return {
            "target": target,
            "condition": condition_name,
            "applied": False,
            "error": f"Target '{target}' not found",
        }

    async def _execute_remove_condition(self, args: dict) -> dict:
        """Remove a condition from a target."""
        target = args.get("target", args.get("target_id", ""))
        condition_name = args.get("condition", "")

        resolved = self._resolve_character_by_name(target)
        if resolved:
            char_id, character = resolved

            repo = await get_character_repo()
            await repo.remove_condition(char_id, condition_name.lower())

            character.conditions = [
                c for c in character.conditions
                if c.condition.value != condition_name.lower()
            ]

            return {
                "target": character.name,
                "target_id": char_id,
                "condition": condition_name,
                "removed": True,
            }

        return {
            "target": target,
            "condition": condition_name,
            "removed": False,
            "error": f"Target '{target}' not found",
        }

    async def _execute_expend_spell_slot(self, args: dict) -> dict:
        """Expend a spell slot."""
        caster = args.get("caster", args.get("caster_id", ""))
        slot_level = args.get("slot_level", 1)
        tx_id = args.get("transaction_id")

        resolved = self._resolve_character_by_name(caster)
        if resolved:
            char_id, character = resolved

            if tx_id:
                tx_key = generate_transaction_key("spell_slot", char_id, tx_id)
                tx_repo = await get_transaction_repo()
                cached = await tx_repo.get_result(tx_key)
                if cached:
                    return cached

            if not character.spell_slots.has_slot(slot_level):
                return {
                    "caster": character.name,
                    "slot_level": slot_level,
                    "expended": False,
                    "error": f"No level {slot_level} spell slots remaining",
                }

            character.spell_slots.expend_slot(slot_level)
            current, _ = character.spell_slots.get_slots(slot_level)

            repo = await get_character_repo()
            await repo.update_spell_slot(char_id, slot_level, current)

            result = {
                "caster": character.name,
                "caster_id": char_id,
                "slot_level": slot_level,
                "slots_remaining": current,
                "expended": True,
            }

            if tx_id:
                await tx_repo.record(tx_key, "expend_spell_slot", char_id, result, args)

            return result

        return {
            "caster": caster,
            "slot_level": slot_level,
            "expended": False,
            "error": f"Caster '{caster}' not found",
        }

    async def _execute_check_rule(self, args: dict) -> str:
        """Look up a rule from the SRD."""
        query = args.get("query", "")

        from ..data.srd import get_srd
        srd = get_srd()

        result_parts = []
        query_lower = query.lower()

        for condition_name in [c.value for c in Condition]:
            if condition_name in query_lower:
                condition_data = srd.get_condition(condition_name)
                if condition_data:
                    desc = condition_data.get("desc", [])
                    result_parts.append(f"**{condition_data.get('name', condition_name)}**: {' '.join(desc)}")

        if "spell" in query_lower or not result_parts:
            for word in query.split():
                spell = srd.get_spell(word.lower().replace(" ", "-"))
                if spell:
                    desc = spell.get("desc", ["No description"])[:1]
                    result_parts.append(f"**{spell.get('name')}**: {desc[0][:200]}...")

        if result_parts:
            return "\n\n".join(result_parts)

        return f"No specific rule found for: {query}. Consider consulting the Player's Handbook."

    async def _execute_use_action(self, args: dict) -> dict:
        """Mark an action as used in combat."""
        combatant = args.get("combatant", args.get("combatant_id", ""))
        action_type = args.get("action_type", "action")
        tx_id = args.get("transaction_id")

        if not self._current_session or not self._current_session.combat_manager:
            return {
                "combatant": combatant,
                "action_type": action_type,
                "used": False,
                "error": "Not in combat",
            }

        combat_manager = self._current_session.combat_manager
        resolved = self._resolve_combatant_by_name(combatant)

        if not resolved:
            char_resolved = self._resolve_character_by_name(combatant)
            if char_resolved:
                resolved = self._resolve_combatant_by_name(char_resolved[1].name)

        if resolved:
            combatant_id, combatant_obj = resolved

            if tx_id:
                tx_key = generate_transaction_key("action", combatant_id, tx_id)
                tx_repo = await get_transaction_repo()
                cached = await tx_repo.get_result(tx_key)
                if cached:
                    return cached

            success = False
            if action_type == "action":
                success = combat_manager.use_action(combatant_id)
            elif action_type == "bonus_action":
                success = combat_manager.use_bonus_action(combatant_id)
            elif action_type == "reaction":
                success = combat_manager.use_reaction(combatant_id)

            result = {
                "combatant": combatant_obj.name,
                "combatant_id": combatant_id,
                "action_type": action_type,
                "used": success,
            }

            if tx_id:
                await tx_repo.record(tx_key, "use_action", combatant_id, result, args)

            return result

        return {
            "combatant": combatant,
            "action_type": action_type,
            "used": False,
            "error": f"Combatant '{combatant}' not found",
        }

    # Commerce tools (preserved)
    async def _execute_purchase_item(self, args: dict) -> dict:
        """Execute item purchase."""
        buyer = args.get("buyer_id", "")
        item_index = args.get("item_index", "")
        item_name = args.get("item_name", "")
        cost_gold = args.get("cost_gold", 0)
        quantity = args.get("quantity", 1)
        tx_id = args.get("transaction_id")

        resolved = self._resolve_character_by_name(buyer)
        if not resolved:
            return {"buyer": buyer, "item": item_name, "purchased": False, "error": f"Character '{buyer}' not found"}

        char_id, character = resolved

        if tx_id:
            tx_key = generate_transaction_key("purchase", char_id, tx_id)
            tx_repo = await get_transaction_repo()
            cached = await tx_repo.get_result(tx_key)
            if cached:
                return cached

        inventory_repo = await get_inventory_repo()
        total_cost = cost_gold * quantity
        success, currency = await inventory_repo.remove_gold(char_id, total_cost)

        if not success:
            return {
                "buyer": character.name, "item": item_name, "cost": total_cost,
                "purchased": False, "error": f"Not enough gold. Have {currency.gold}gp, need {total_cost}gp",
            }

        new_item = InventoryItem(character_id=char_id, item_index=item_index, item_name=item_name, quantity=quantity)
        await inventory_repo.add_item(new_item)

        result = {
            "buyer": character.name, "buyer_id": char_id, "item": item_name,
            "quantity": quantity, "cost": total_cost, "gold_after": currency.gold, "purchased": True,
        }

        if tx_id:
            await tx_repo.record(tx_key, "purchase_item", char_id, result, args)
        return result

    async def _execute_sell_item(self, args: dict) -> dict:
        """Execute item sale."""
        seller = args.get("seller_id", "")
        item_id = args.get("item_id", "")
        sale_price = args.get("sale_price_gold", 0)
        quantity = args.get("quantity", 1)

        resolved = self._resolve_character_by_name(seller)
        if not resolved:
            return {"seller": seller, "sold": False, "error": f"Character '{seller}' not found"}

        char_id, character = resolved
        inventory_repo = await get_inventory_repo()

        item = await inventory_repo.get_item_by_id(item_id)
        if not item or item.character_id != char_id:
            return {"seller": character.name, "sold": False, "error": f"Item not found"}

        await inventory_repo.remove_item(item_id, quantity)
        currency = await inventory_repo.add_gold(char_id, sale_price)

        return {"seller": character.name, "item": item.item_name, "sale_price": sale_price, "gold_after": currency.gold, "sold": True}

    async def _execute_modify_gold(self, args: dict) -> dict:
        """Modify a character's gold."""
        character_name = args.get("character_id", "")
        delta = args.get("delta", 0)
        reason = args.get("reason", "")

        resolved = self._resolve_character_by_name(character_name)
        if not resolved:
            return {"character": character_name, "modified": False, "error": f"Character '{character_name}' not found"}

        char_id, character = resolved
        inventory_repo = await get_inventory_repo()
        currency = await inventory_repo.get_currency(char_id)
        gold_before = currency.gold

        if delta > 0:
            currency = await inventory_repo.add_gold(char_id, delta)
        elif delta < 0:
            success, currency = await inventory_repo.remove_gold(char_id, abs(delta))
            if not success:
                return {"character": character.name, "delta": delta, "modified": False, "error": f"Not enough gold"}

        return {"character": character.name, "delta": delta, "reason": reason, "gold_before": gold_before, "gold_after": currency.gold, "modified": True}

    async def _execute_add_item(self, args: dict) -> dict:
        """Add an item to inventory."""
        character_name = args.get("character_id", "")
        item_index = args.get("item_index", "")
        item_name = args.get("item_name", "")
        quantity = args.get("quantity", 1)
        source = args.get("source", "")

        resolved = self._resolve_character_by_name(character_name)
        if not resolved:
            return {"character": character_name, "item": item_name, "added": False, "error": f"Character '{character_name}' not found"}

        char_id, character = resolved
        inventory_repo = await get_inventory_repo()

        new_item = InventoryItem(character_id=char_id, item_index=item_index, item_name=item_name, quantity=quantity, notes=source)
        added_item = await inventory_repo.add_item(new_item)

        return {"character": character.name, "item": item_name, "item_id": added_item.id, "quantity": quantity, "added": True}

    async def _execute_remove_item(self, args: dict) -> dict:
        """Remove an item from inventory."""
        character_name = args.get("character_id", "")
        item_id = args.get("item_id", "")
        quantity = args.get("quantity", 1)

        resolved = self._resolve_character_by_name(character_name)
        if not resolved:
            return {"character": character_name, "removed": False, "error": f"Character '{character_name}' not found"}

        char_id, character = resolved
        inventory_repo = await get_inventory_repo()

        item = await inventory_repo.get_item_by_id(item_id)
        if not item or item.character_id != char_id:
            return {"character": character.name, "removed": False, "error": "Item not found"}

        await inventory_repo.remove_item(item_id, quantity)
        return {"character": character.name, "item": item.item_name, "quantity": quantity, "removed": True}

    async def _execute_equip_item(self, args: dict) -> dict:
        """Equip an item."""
        character_name = args.get("character_id", "")
        item_id = args.get("item_id", "")

        resolved = self._resolve_character_by_name(character_name)
        if not resolved:
            return {"character": character_name, "equipped": False, "error": f"Character '{character_name}' not found"}

        char_id, character = resolved
        inventory_repo = await get_inventory_repo()

        item = await inventory_repo.get_item_by_id(item_id)
        if not item or item.character_id != char_id:
            return {"character": character.name, "equipped": False, "error": "Item not found"}

        await inventory_repo.equip_item(item_id)
        return {"character": character.name, "item": item.item_name, "equipped": True}

    async def _execute_unequip_item(self, args: dict) -> dict:
        """Unequip an item."""
        character_name = args.get("character_id", "")
        item_id = args.get("item_id", "")

        resolved = self._resolve_character_by_name(character_name)
        if not resolved:
            return {"character": character_name, "unequipped": False, "error": f"Character '{character_name}' not found"}

        char_id, character = resolved
        inventory_repo = await get_inventory_repo()

        item = await inventory_repo.get_item_by_id(item_id)
        if not item or item.character_id != char_id:
            return {"character": character.name, "unequipped": False, "error": "Item not found"}

        await inventory_repo.unequip_item(item_id)
        return {"character": character.name, "item": item.item_name, "unequipped": True}

    async def _execute_transfer_item(self, args: dict) -> dict:
        """Transfer an item between characters."""
        from_name = args.get("from_character_id", "")
        to_name = args.get("to_character_id", "")
        item_id = args.get("item_id", "")
        quantity = args.get("quantity", 1)

        from_resolved = self._resolve_character_by_name(from_name)
        to_resolved = self._resolve_character_by_name(to_name)

        if not from_resolved:
            return {"from": from_name, "to": to_name, "transferred": False, "error": f"Character '{from_name}' not found"}
        if not to_resolved:
            return {"from": from_name, "to": to_name, "transferred": False, "error": f"Character '{to_name}' not found"}

        from_char_id, from_character = from_resolved
        to_char_id, to_character = to_resolved

        inventory_repo = await get_inventory_repo()
        item = await inventory_repo.get_item_by_id(item_id)
        if not item or item.character_id != from_char_id:
            return {"from": from_character.name, "to": to_character.name, "transferred": False, "error": "Item not found"}

        transferred_item = await inventory_repo.transfer_item(item_id, from_char_id, to_char_id, quantity)
        if not transferred_item:
            return {"from": from_character.name, "to": to_character.name, "transferred": False, "error": "Transfer failed"}

        return {"from": from_character.name, "to": to_character.name, "item": item.item_name, "quantity": quantity, "transferred": True}


# Global orchestrator instance
_orchestrator: Optional[DMOrchestrator] = None


def get_orchestrator() -> DMOrchestrator:
    """Get the global DM orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = DMOrchestrator()
    return _orchestrator


def _reset_orchestrator():
    """Clear cached orchestrator so it recreates from the active profile."""
    global _orchestrator
    _orchestrator = None
