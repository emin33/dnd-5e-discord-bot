"""Narrator Brain - Creative storytelling (prose only).

This brain handles pure creative narration. It does NOT output JSON or
propose mechanical effects - that's handled by the EffectsAdjudicator
in a separate, deterministic pass.
"""

from dataclasses import dataclass
from typing import Optional

from ..client import OllamaClient, get_llm_client
from .base import Brain, BrainContext, BrainResult
from ...config import get_settings

import structlog

logger = structlog.get_logger()


NARRATOR_SYSTEM_PROMPT = """You are the creative narrator for a D&D 5e campaign. Your role is to bring the world to life through vivid descriptions, compelling NPC dialogue, and immersive storytelling.

## YOUR ROLE

You are the VOICE of the world. You:
- Describe scenes, NPCs, and events vividly
- Introduce new elements (objects, NPCs, environmental details)
- Roleplay NPCs with distinct personalities
- Narrate outcomes of mechanical resolutions you receive

## STORYTELLING PRINCIPLES

**Never describe an action in isolation.** Every action ripples into the world.

When narrating player actions:
1. **Acknowledge the action briefly** (1-2 sentences)
2. **Show the world's reaction** - How does the environment, NPCs, or atmosphere respond?
3. **Connect to ongoing tension** - Reference established stakes, threats, or mysteries
4. **End with forward momentum** - Something that invites further action or decision

**Avoid "shoe leather"** - Don't waste words on routine movements. Cut to the interesting parts.

**Use sensory details** - Sights, sounds, smells, textures. Make players feel present.

**Reset the scene after major actions** - Briefly remind players where they are and what's relevant now.

Example of BAD narration (isolated):
> "The coins scatter across the floor, glinting in the light."

Example of GOOD narration (connected):
> "The coins scatter through the damp moss, their metallic ring cutting the unnatural silence. The rhythmic thumping from the mill stops. In the sudden quiet, you sense something in the shadows has taken notice."

## OUTPUT FORMAT

You output TWO sections: PROSE and INTENTS.

```
PROSE:
[Your vivid 2-4 paragraph narration here]

INTENTS:
[One intent per line, or NONE if no mechanical effects]
```

PROSE is your creative narration. INTENTS explicitly signals any mechanical effects you're introducing.

## INTENT COMMANDS

When your narration introduces something mechanical, add the matching intent:

**Spawn an object in the scene:**
spawn_object <id> "<name>" "<description>"
Example: spawn_object dagger_1 "Jeweled Dagger" "An ornate dagger with ruby-studded hilt"

**Introduce an NPC:**
add_npc <id> "<name>" <disposition> "<description>"
Dispositions: friendly, neutral, unfriendly, hostile
Example: add_npc merchant_1 "Grizzled Merchant" neutral "A weathered trader with knowing eyes"

**NPC offers something (player must confirm):**
offer_item <from>-><to> "<item>" confirm
Example: offer_item npc:farmer->player "Coin Pouch" confirm

**NPC offers currency:**
grant_currency <target> <amount><denom> confirm
Denominations: cp, sp, ep, gp, pp
Example: grant_currency player 15gp confirm

**Player picks up / acquires item (no confirmation needed):**
transfer_item <from>-><to> "<item>"
Example: transfer_item scene:dagger_1->player "Jeweled Dagger"

**Environmental damage:**
apply_damage <target> <amount> <type> "<reason>"
Example: apply_damage player 5 fire "Touched the brazier"

**DM requests a roll:**
request_roll <target> <type> <ability/skill> dc=<N> "<reason>"
Types: save, check, skill
Example: request_roll player save constitution dc=15 "Resist the poison"
Example: request_roll player skill perception dc=12 "Notice the hidden door"

**Combat starts:**
start_combat "<reason>"
Example: start_combat "The bandits draw their weapons!"

**No mechanical effects:**
NONE

## ENCOUNTER BALANCE

You MUST respect the party's power level when introducing threats:
- **Level 1-2 party (1-2 players):** 1-3 weak enemies (bandits, goblins, wolves). No bosses. Total enemy HP should not exceed 3x the party's total HP.
- **Level 1-2 party (3-4 players):** Up to 4-6 weak enemies or 1 moderate enemy.
- **Level 3-5 party:** Can face tougher foes. A single CR 2-3 creature or groups of weaker ones.
- When in doubt, fewer enemies. Players should feel challenged but not instantly killed.
- Do NOT describe armies, swarms, or overwhelming forces as direct combatants. These can be narrative backdrop but not actual threats that enter combat.
- Magical effects, environmental hazards, and spell effects are FLAVOR — do not describe them as creatures that would enter combat (e.g., "writhing vines" or "shadow tendrils" are atmosphere, not monsters).

## NARRATING MECHANICAL RESULTS (from [RESOLUTION: ...])

When you receive a [RESOLUTION], it is BINDING. Your narration MUST reflect the outcome.

**Degrees of Success (how WELL they succeeded):**
- CRITICAL SUCCESS (beat DC by 10+): Exceptional result. Reveal everything plus bonus detail. The character looks masterful.
- SOLID SUCCESS (beat DC by 5-9): Clear success. Reveal the information/outcome fully.
- NARROW SUCCESS (beat DC by 1-4): Barely made it. Reveal the basics but incompletely — partial information, success with a minor cost or complication.

**Degrees of Failure (how BADLY they failed):**
- NARROW FAILURE (missed DC by 1-4): Almost succeeded. Hint at what they missed — "something feels off but you can't place it." Consider "success at a cost" — they get what they wanted but with a complication.
- CLEAR FAILURE (missed DC by 5-9): They clearly fail. Describe what they miss, botch, or get wrong. The world responds to their failure.
- CRITICAL FAILURE (missed DC by 10+): Embarrassing or dangerous. Consequences beyond just "nothing happens" — they alert enemies, break a tool, draw unwanted attention, misidentify something.

**IMPORTANT: Failure is NOT "nothing happens."** A good DM makes failure interesting:
- Perception failure: You don't just "see nothing" — you feel confident the area is safe (false confidence), or you're distracted by something else
- Athletics failure: You don't just "fail to climb" — your handhold crumbles, you slide back, you pull a muscle
- Persuasion failure: The NPC doesn't just say no — they get annoyed, suspicious, or raise their price
- Investigation failure: You misread the clue, waste time on a red herring, or overlook the obvious

**AUTHORIZED REVEALS** limit what you can describe about the outcome. Do NOT invent discoveries or information beyond what is listed.

## NARRATING COMBAT

When narrating attacks, hits, and misses:
- **Hits**: Describe the attack vividly. Consider how much damage was dealt — a 2-damage hit is a scratch, a 15-damage hit is devastating. Reference the weapon and fighting style.
- **Misses**: Make misses interesting, not boring. WHY did it miss? If the target has a shield and the roll barely missed, the shield blocked it. If it missed by a lot, the attacker overextended or the target dodged easily.
- **Critical hits**: Describe with extra drama — the blow finds a gap in armor, strikes a vital spot, or sends the target reeling.
- **Killing blows**: Build to a dramatic finish. The creature staggers, collapses, or is sent flying.
- **Monster personality**: An ogre swings clumsily, a duelist strikes precisely, an undead is unfazed by wounds.

## CRITICAL RULES

1. **If you describe it, intent it** - If your prose introduces an object, NPC, offer, or hazard, add the matching intent.

2. **OFFER language for gifts** - When an NPC offers something, describe them OFFERING (holding out, presenting). Don't describe the player receiving it.

3. **Describe, don't transfer** - When treasure appears, describe it in scene + spawn_object. Only use transfer_item when player explicitly takes it.

4. **NEVER generate dice results** - You don't roll. Use request_roll to ask for rolls.

5. **ONLY REVEAL AUTHORIZED INFO** - On [RESOLUTION: SUCCESS/FAILURE], narrate only what's authorized.

## EXAMPLES

**Player enters a room with treasure:**
```
PROSE:
The chamber opens before you, dust motes dancing in the faint light filtering through cracks above. In the center, atop a crumbling pedestal, something glints—a jeweled dagger, its blade still keen despite the ages.

INTENTS:
spawn_object dagger_1 "Jeweled Dagger" "An ornate dagger with ruby-studded hilt on a pedestal"
```

**NPC offers a reward:**
```
PROSE:
The grateful farmer's weathered hands shake as he reaches into his coat. 'You saved my daughter. Please—it's all I have.' He holds out a small pouch, coins clinking within.

INTENTS:
grant_currency player 15gp confirm
```

**Player picks up the dagger:**
```
PROSE:
Your fingers close around the hilt. The balance is perfect—whoever crafted this blade knew their art. The ruby in the pommel catches the dim light as you lift it from the pedestal.

INTENTS:
transfer_item scene:dagger_1->player "Jeweled Dagger"
```

**Pure atmosphere (no mechanics):**
```
PROSE:
The wind howls through the mountain pass, carrying with it the scent of pine and distant snow. Your cloak whips around you as you press forward, the path narrowing with each step.

INTENTS:
NONE
```

**Trap triggers a save:**
```
PROSE:
As your foot presses the loose flagstone, you hear a click. A jet of green vapor hisses from the wall beside you, acrid and burning.

INTENTS:
request_roll player save constitution dc=14 "Resist the poison gas"
```"""


@dataclass
class MechanicalOutcome:
    """A mechanical outcome to be narrated."""

    action_type: str  # 'attack', 'spell', 'check', 'save', 'damage', 'healing'
    success: bool
    description: str  # Brief mechanical description
    details: dict  # Additional details (damage amount, condition applied, etc.)


class NarratorBrain(Brain):
    """
    The creative storytelling brain.

    Handles:
    - Scene descriptions and atmosphere
    - NPC dialogue and roleplay
    - Narrating mechanical outcomes dramatically
    - Advancing the story

    Does NOT handle:
    - Dice rolls or random outcomes
    - Determining success/failure
    - Game state changes
    """

    def __init__(
        self,
        client: Optional[OllamaClient] = None,
        temperature: Optional[float] = None,
    ):
        settings = get_settings()
        super().__init__(
            client=client or get_llm_client(),
            temperature=temperature or settings.narrator_temperature,
            system_prompt=NARRATOR_SYSTEM_PROMPT,
        )

    async def process(self, context: BrainContext) -> BrainResult:
        """Generate pure prose narrative content.

        Returns plain text narration - no JSON, no structured effects.
        The EffectsAdjudicator handles effect extraction in a separate pass.
        """
        messages = self._build_messages(context)

        # Add explicit instruction for prose output
        messages.append({
            "role": "system",
            "content": (
                "###YOUR TASK###\n"
                "Respond to the player's action or statement above. "
                "Describe what happens, what they experience, or how the world responds.\n\n"
                "Write 2-4 paragraphs of vivid prose narration. Begin now:"
            ),
        })

        response = await self.client.chat(
            messages=messages,
            temperature=self.temperature,
            max_tokens=1000,
            think=False,
        )

        content = response.content.strip() if response.content else ""

        # Validate non-empty response
        if not content:
            content = "*The scene continues...*"

        return BrainResult(
            content=content,
            raw_response=response.content,
        )

    async def narrate_outcome(
        self,
        context: BrainContext,
        outcome: MechanicalOutcome,
    ) -> BrainResult:
        """
        Narrate a mechanical outcome dramatically.

        The Narrator receives the mechanical result and describes it
        in an engaging, story-appropriate way.
        """
        # Build a specific prompt for narrating the outcome
        outcome_description = self._format_outcome(outcome)

        # Add the outcome to the context
        enhanced_context = BrainContext(
            party_status=context.party_status,
            current_scene=context.current_scene,
            active_quests=context.active_quests,
            in_combat=context.in_combat,
            combat_round=context.combat_round,
            current_combatant=context.current_combatant,
            initiative_order=context.initiative_order,
            recent_messages=context.recent_messages,
            session_summary=context.session_summary,
            player_action=f"{context.player_action}\n\n[MECHANICAL RESULT: {outcome_description}]",
            player_name=context.player_name,
        )

        messages = self._build_messages(enhanced_context)

        # Add instruction for narrating the outcome — explicitly tell the model
        # NOT to use PROSE:/INTENTS: format (it follows the main system prompt otherwise)
        messages.append({
            "role": "system",
            "content": (
                "Narrate the mechanical result above in a dramatic, engaging way. "
                "Do NOT change or add to the mechanical outcome - just describe it vividly. "
                "Output ONLY the narrative prose directly. Do NOT use PROSE:/INTENTS: "
                "format headers or code blocks. Just write the narrative."
            ),
        })

        response = await self.client.chat(
            messages=messages,
            temperature=self.temperature,
            max_tokens=1000,
            think=False,  # Disable thinking - causes truncation issues with Qwen3
        )

        result = self._parse_response(response)

        # Strip PROSE/INTENTS format if the model used it despite instructions
        if result.content:
            import re
            content = result.content.strip()

            # Strip code fence wrappers first (```...```)
            # The model sometimes wraps the entire output in triple backticks
            code_fence = re.match(r'^```\w*\s*\n?(.*?)```\s*$', content, re.DOTALL)
            if code_fence:
                content = code_fence.group(1).strip()

            # Remove PROSE: prefix in various formats:
            # PROSE:, **PROSE:**, ## PROSE:, **PROSE:**
            content = re.sub(
                r'^(?:#{1,3}\s*)?(?:\*{1,2})?PROSE(?:\*{1,2})?\s*:?\s*(?:\*{1,2})?\s*',
                '',
                content,
                count=1,
                flags=re.IGNORECASE,
            ).strip()

            # Remove INTENTS block if present (everything from INTENTS: onward)
            intents_match = re.search(r'\n\s*(?:\*{1,2})?INTENTS(?:\*{1,2})?\s*:', content, re.IGNORECASE)
            if intents_match:
                content = content[:intents_match.start()].strip()

            result.content = content

        # Validate non-empty response
        if not result.content or not result.content.strip():
            result.content = f"*The action unfolds dramatically...*"

        return result

    async def describe_scene(
        self,
        scene_description: str,
        context: BrainContext,
    ) -> BrainResult:
        """Generate a vivid scene description."""
        enhanced_context = BrainContext(
            party_status=context.party_status,
            current_scene=context.current_scene,
            active_quests=context.active_quests,
            recent_messages=context.recent_messages,
            session_summary=context.session_summary,
            player_action=f"[SCENE TRANSITION: {scene_description}]",
            player_name="DM",
        )

        messages = self._build_messages(enhanced_context)
        messages.append({
            "role": "system",
            "content": "Describe this new scene vividly. Set the atmosphere and give the players a sense of place.",
        })

        response = await self.client.chat(
            messages=messages,
            temperature=self.temperature,
            max_tokens=600,
            think=False,  # Disable thinking - causes truncation issues with Qwen3
        )

        result = self._parse_response(response)

        # Validate non-empty response
        if not result.content or not result.content.strip():
            result.content = f"*You take in your new surroundings...*"

        return result

    async def roleplay_npc(
        self,
        npc_name: str,
        npc_description: str,
        player_dialogue: str,
        context: BrainContext,
    ) -> BrainResult:
        """Generate NPC dialogue and actions."""
        npc_context = f"[NPC: {npc_name}]\n{npc_description}"

        enhanced_context = BrainContext(
            party_status=context.party_status,
            current_scene=f"{context.current_scene}\n\n{npc_context}",
            active_quests=context.active_quests,
            recent_messages=context.recent_messages,
            session_summary=context.session_summary,
            player_action=player_dialogue,
            player_name=context.player_name,
        )

        messages = self._build_messages(enhanced_context)
        messages.append({
            "role": "system",
            "content": f"Respond as {npc_name}. Stay in character and respond naturally to what the player said.",
        })

        response = await self.client.chat(
            messages=messages,
            temperature=self.temperature,
            max_tokens=400,
            think=False,  # Disable thinking - causes truncation issues with Qwen3
        )

        result = self._parse_response(response)

        # Validate non-empty response
        if not result.content or not result.content.strip():
            result.content = f'*{npc_name} regards you silently for a moment...*'

        return result

    async def generate_opening(
        self,
        campaign_name: str,
        world_setting: str,
        characters: list,
    ) -> str:
        """Generate an opening narrative to start the adventure."""
        # Build character descriptions
        char_descriptions = []
        for char in characters:
            char_descriptions.append(
                f"- {char.name}, a Level {char.level} {char.race_index.title()} {char.class_index.title()}"
            )

        party_info = "\n".join(char_descriptions) if char_descriptions else "No adventurers yet"

        messages = [
            {
                "role": "system",
                "content": NARRATOR_SYSTEM_PROMPT,
            },
            {
                "role": "user",
                "content": f"""Generate an opening scene for a new D&D adventure.

Campaign: {campaign_name}
World Setting: {world_setting}

The Party:
{party_info}

## PROVEN SCENARIO TEMPLATES (choose one or blend)

**Tavern/Gathering Meet-Up** - Strangers in a public place when a hook arrives (brawl, mysterious stranger, posted notice). Flexible, comfortable start.
> "The Yawning Portal is loud tonight. Somewhere between your second ale and the argument at the bar, you notice the old man watching your table..."

**Festival Gone Wrong** - Lively event (carnival, tournament, wedding) interrupted by chaos (attack, theft, magical mishap). Dynamic and atmospheric.
> "Lanterns bob overhead as the Harvest Festival reaches its peak. Then the screaming starts from the merchant quarter..."

**Road Ambush** - Party already traveling together when danger strikes. Immediate action, natural party formation.
> "Three days out of Neverwinter, the forest road narrows. The fallen tree ahead wasn't there yesterday. Neither were the figures in the shadows..."

**In Medias Res** - Drop straight into action already underway. High energy, forces instant teamwork.
> "The ship lurches. Seawater crashes through the hull breach as something massive scrapes against the wood below..."

**Town in Peril** - Arriving at or living in a place under attack. Cinematic, creates instant heroes.
> "Smoke rises from Greenest. Even from the hill, you can see the shapes wheeling above the chapel - and the dragon's silhouette against the flames..."

**Shared Predicament** - Party starts trapped together (prisoners, shipwrecked, stranded). Powerful team-building through survival.
> "The drow shackles bite your wrists. In the darkness of the slave pen, you count five other prisoners - all strangers, all watching the guards..."

## THE 7 CORE BEATS (your opening MUST hit these)

1. **Immediate Presence** - Start IN a moment, not before it. Sensory grounding: weather, sound, motion, crowd, terrain.

2. **Disruption** - Something is wrong, interrupted, incomplete, or unexpected. Not explained - only felt.

3. **Forced Attention** - The party cannot ignore the situation. Someone addresses them, something blocks their path, or morality demands action.

4. **Unclear Stakes** - Stakes are IMPLIED, not stated. Players sense danger/opportunity/consequence without knowing full context.

5. **Agency Fork** - At least two reasonable responses exist. Neither is "correct." Both move the story forward differently.

6. **World Reactivity Signal** - Show that NPCs/environment WILL respond to player choices. Train players to act, not wait.

7. **Explicit Handoff** - End with a clear prompt: "What do you do?" Address the party directly. Ask broad, permissive question.

## AVOID
- Lore dumps or exposition
- Naming multiple NPCs at once
- Forcing a specific quest immediately
- Narrow questions ("Do you talk to the innkeeper?")
- Pure description with no hook or tension

## FORMAT
Write 2-3 FULL paragraphs (each 3-5 sentences):
- Paragraph 1: Sensory immersion - place the party IN the moment with vivid detail
- Paragraph 2: The disruption unfolds - tension escalates, stakes become felt
- Paragraph 3: The fork crystallizes - present the choice clearly, end with "What do you do?"

Output ONLY the PROSE and INTENTS blocks. No commentary, no planning, no explanations.""",
            },
        ]

        response = await self.client.chat(
            messages=messages,
            temperature=self.temperature,
            max_tokens=8000,  # High limit for thinking models
            think=True,  # Keep thinking in separate block
        )

        content = response.content.strip() if response.content else ""

        # Parse PROSE/INTENTS - for openings we only care about the prose
        from ..intents import extract_intents_block, validate_narrator_format

        # Validate format - should start with PROSE:
        if content and not validate_narrator_format(content):
            logger.warning(
                "opening_format_invalid_reprompting",
                content_preview=content[:100] if content else "(empty)",
            )
            # Reprompt once
            messages.append({
                "role": "system",
                "content": "Your response was malformed. Start with PROSE: immediately, then your narration. No planning or commentary.",
            })
            response = await self.client.chat(
                messages=messages,
                temperature=self.temperature,
                max_tokens=8000,
                think=True,
            )
            content = response.content.strip() if response.content else ""

        prose, _ = extract_intents_block(content or "")

        # Validate non-empty response
        if not prose or not prose.strip():
            prose = (
                f"*The adventure begins in {world_setting}. "
                f"Your party gathers, ready for whatever lies ahead.*\n\n"
                "What do you do?"
            )
        else:
            # Ensure the prose ends with an open-ended handoff question
            # Models often ignore this instruction, so we enforce it
            prose = self._ensure_handoff(prose)

        return prose

    def _ensure_handoff(self, prose: str) -> str:
        """Ensure prose ends with an open-ended question for player agency.

        LLMs often ignore the instruction to end with 'What do you do?',
        so we detect and append it if missing.
        """
        import re

        # Patterns that indicate a proper handoff question
        handoff_patterns = [
            r"what do you do\??$",
            r"what will you do\??$",
            r"how do you respond\??$",
            r"what's your move\??$",
            r"what now\??$",
            r"your move\.?$",
        ]

        prose_lower = prose.strip().lower()

        # Check if prose already ends with a handoff
        for pattern in handoff_patterns:
            if re.search(pattern, prose_lower):
                return prose

        # Check if it ends with any question mark (might be a valid open question)
        if prose.strip().endswith("?"):
            # It's a question, probably fine
            return prose

        # No handoff found - append one
        logger.info("opening_missing_handoff_appending", prose_end=prose[-50:] if len(prose) > 50 else prose)
        return prose.rstrip() + "\n\nWhat do you do?"

    def _format_outcome(self, outcome: MechanicalOutcome) -> str:
        """Format a mechanical outcome for the prompt."""
        parts = [outcome.description]

        if outcome.action_type == "attack":
            if outcome.success:
                damage = outcome.details.get("damage", 0)
                damage_type = outcome.details.get("damage_type", "")
                parts.append(f"Hit for {damage} {damage_type} damage.")
                if outcome.details.get("critical"):
                    parts.append("Critical hit!")
            else:
                parts.append("Miss.")

        elif outcome.action_type == "spell":
            if outcome.success:
                effect = outcome.details.get("effect", "spell takes effect")
                parts.append(effect)
            else:
                parts.append("Spell fails to take effect.")

        elif outcome.action_type in ("check", "save", "ability_check", "saving_throw"):
            dc = outcome.details.get("dc", "?")
            roll = outcome.details.get("roll", "?")
            skill = outcome.details.get("skill", "")
            skill_text = f" ({skill})" if skill else ""
            if outcome.success:
                parts.append(f"Success{skill_text} (rolled {roll} vs DC {dc}).")
            else:
                parts.append(f"Failure{skill_text} (rolled {roll} vs DC {dc}).")

        elif outcome.action_type == "healing":
            amount = outcome.details.get("amount", 0)
            parts.append(f"Healed for {amount} HP.")

        return " ".join(parts)


# Global narrator instance
_narrator: Optional[NarratorBrain] = None


def get_narrator() -> NarratorBrain:
    """Get the global narrator brain."""
    global _narrator
    if _narrator is None:
        _narrator = NarratorBrain()
    return _narrator
