"""Narrator Brain - Creative storytelling (prose only).

This brain handles pure creative narration. It does NOT output JSON or
propose mechanical effects - that's handled by the EffectsAdjudicator
in a separate, deterministic pass.
"""

from dataclasses import dataclass
from typing import Optional

from ..client import OllamaClient, get_llm_client, get_narrator_client
from .base import Brain, BrainContext, BrainResult
from ...config import get_settings

import structlog

logger = structlog.get_logger()


NARRATOR_SYSTEM_PROMPT = """You are the DM narrator for a D&D 5e campaign.

## CORE PRINCIPLE

**The world is alive. NPCs have goals and act on them.** You don't just describe — you PLAY the NPCs. A goblin scout doesn't "watch and wait" for 5 turns. It signals allies, sets an ambush, or charges. A merchant doesn't stand silently — they haggle, gossip, or grow suspicious. Every NPC is a person with wants. Make them act.

## HOW TO NARRATE

1. **Show the consequence** — Don't echo the player's action. Show what CHANGES.
2. **NPCs react and act** — They speak, move, scheme. They don't freeze in place.
3. **Advance the situation** — Every turn must change something. New information, shifted positions, escalating tension, a decision made. If nothing changed, you failed.
4. **End with a demand** — Close with something that forces the player to respond: a question, a threat, a choice, a ticking clock.

**Never re-describe.** If you established the tavern is smoky, don't mention it again. If the goblin has a crossbow, don't describe it every turn. Describe things ONCE, then move the story forward.

**Never repeat phrases.** If you wrote "the forest holds its breath" last turn, that phrase is DEAD for the rest of the session. Find new language every time.

## GROUNDING

- Only use NPC names from the roster. Don't invent new names for existing NPCs.
- NPCs stay where last seen unless you narrate them moving.
- Don't contradict established facts. If a creature was identified as a badger, it stays a badger.
- On [RESOLUTION: SUCCESS], narrate the success. On [FAILURE], narrate the failure. Don't soften failures or ignore successes. AUTHORIZED REVEALS limit what you can describe.

## MECHANICAL RESULTS

When you receive a [RESOLUTION], it is BINDING.
- **Beat DC by 10+**: Exceptional — reveal everything plus bonus.
- **Beat DC by 5-9**: Clear success — full reveal.
- **Beat DC by 1-4**: Narrow — partial, incomplete.
- **Missed by 1-4**: Almost — hint at what was missed.
- **Missed by 5-9**: Clear failure — describe what went wrong.
- **Missed by 10+**: Critical failure — consequences, not just "nothing."

Failure is NEVER "nothing happens." Make failure interesting and consequential.

## COMBAT NARRATION

Hits reference damage amount (2 = scratch, 15 = devastating). Misses explain WHY (shield blocked, dodge, overextended). Crits get extra drama. Killing blows get a finish.

## ENCOUNTER BALANCE

Level 1-2 party (1-2 players): 1-3 weak enemies max. No bosses. No armies as direct combatants.

## TOOLS

Call tools alongside your prose for mechanical effects:
- **ref_entity** — for every roster entity you mention
- **add_npc** — for new NPCs (give them proper names)
- **spawn_object** — for new objects in the scene

Write prose directly. No format headers needed."""


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
            client=client or get_narrator_client(),
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
            max_tokens=1500,
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

    async def process_streaming(
        self,
        context: BrainContext,
        on_token: Optional[callable] = None,
    ) -> BrainResult:
        """Generate narrative with streaming tokens for progressive Discord edits.

        Requires OllamaClient (Groq streaming not yet supported here).
        Falls back to non-streaming if client doesn't support chat_stream.
        """
        if not hasattr(self.client, "chat_stream"):
            return await self.process(context)

        messages = self._build_messages(context)
        messages.append({
            "role": "system",
            "content": (
                "###YOUR TASK###\n"
                "Respond to the player's action or statement above. "
                "Describe what happens, what they experience, or how the world responds.\n\n"
                "Write 2-4 paragraphs of vivid prose narration. Begin now:"
            ),
        })

        response = await self.client.chat_stream(
            messages=messages,
            temperature=self.temperature,
            max_tokens=1500,
            think=False,
            on_token=on_token,
        )

        content = response.content.strip() if response.content else ""
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
            max_tokens=1500,
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

Write your narration directly. Use tools (add_npc, spawn_object, ref_entity) for any NPCs or objects you introduce. No commentary, no planning, no format headers.""",
            },
        ]

        from ..narrator_tools import NARRATOR_TOOLS_CORE

        response = await self.client.chat(
            messages=messages,
            temperature=self.temperature,
            max_tokens=2000,
            tools=NARRATOR_TOOLS_CORE,
            tool_choice="auto",
        )

        content = response.content.strip() if response.content else ""

        # Strip any leaked structural blocks
        # (INTENTS:, ENTITIES:, --- separators, PROSE: headers)
        import re
        content = re.split(r'\n---\n|\nINTENTS:\n|\nENTITIES:\n|\nPROSE:\n', content)[0].strip()
        if content.startswith("PROSE:"):
            content = content[6:].strip()

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


def _reset_narrator():
    """Clear cached narrator so it recreates from the active profile."""
    global _narrator
    _narrator = None
