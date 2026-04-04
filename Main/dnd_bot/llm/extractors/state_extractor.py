"""State Extractor - extracts StateDelta from narrator output.

Runs AFTER the narrator generates prose. Uses the cheap brain model
to emit a structured StateDelta describing what changed in the world.
Python validates the delta before applying it to WorldState.

This is a superset of EntityExtractor: it captures NPC changes plus
time, location, events, and facts.
"""

from typing import Optional
import json
import structlog

from pydantic import ValidationError

from ..client import get_llm_client, OllamaClient
from ...game.world_state import StateDelta, get_state_delta_schema

logger = structlog.get_logger()


EXTRACTION_PROMPT = """You extract world state changes from D&D narrative text.

Given the narrator's prose and the current world state, identify WHAT CHANGED.
Only include fields where something actually changed. Omit unchanged fields.

## What to extract:

**time_change** - Only if time of day shifted (dawn/morning/midday/afternoon/dusk/evening/night/midnight)
**location_change** - Only if the party moved to a new location
**location_description** - Brief description of the new location (only if location changed)
**new_connections** - Newly revealed exits or paths from current location
**new_npcs** - NPCs appearing for the FIRST time. Include:
  - name, location (where they are), disposition (hostile/unfriendly/neutral/friendly/allied)
  - description (brief), important (true if quest-giver, ally, or key story NPC)
**npc_updates** - Changes to EXISTING NPCs (moved, changed disposition, died, etc.)
  - Only include the fields that changed
**removed_npcs** - NPCs who LEFT the scene (not dead, just departed)
**new_events** - Significant narrative events (1 sentence each, max 2)
**new_facts** - Established facts that must not be contradicted later
**flag_changes** - World flags that changed (e.g., "bridge_destroyed": true)
**phase_change** - Only if game phase changed (exploration/combat/dialogue/rest)

## Rules:
- Extract ONLY what the narrative establishes. Do not infer or speculate.
- If nothing changed, return an empty object: {}
- Keep descriptions concise (under 100 characters)
- For NPC disposition: "hostile" = actively aggressive, "unfriendly" = antagonistic but not violent, "neutral" = indifferent, "friendly" = helpful, "allied" = fights alongside party
- Mark NPCs as important=true if they are quest-givers, named allies, rulers, or key to the story
- Do NOT create new_npcs for entities already in the world state

Output valid JSON matching the StateDelta schema."""


class StateExtractor:
    """Extracts StateDelta from narrator output using the brain model."""

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or get_llm_client()

    async def extract(
        self,
        narrative_text: str,
        world_state_yaml: str = "",
        current_scene: str = "",
    ) -> StateDelta:
        """Extract a StateDelta from narrator prose.

        Args:
            narrative_text: The narrator's output prose
            world_state_yaml: Current world state as YAML (for context)
            current_scene: Current scene description

        Returns:
            Validated StateDelta. Empty delta on parse failure.
        """
        if not narrative_text or len(narrative_text) < 20:
            return StateDelta()

        # Build context for extractor
        context_parts = []
        if world_state_yaml:
            context_parts.append(f"Current World State:\n{world_state_yaml}")
        elif current_scene:
            context_parts.append(f"Current Scene: {current_scene}")

        context = "\n\n".join(context_parts) if context_parts else "Session start"

        messages = [
            {"role": "system", "content": EXTRACTION_PROMPT},
            {
                "role": "user",
                "content": f"{context}\n\n---\n\nNarrative to analyze:\n{narrative_text}",
            },
        ]

        try:
            response = await self.client.chat(
                messages=messages,
                temperature=0,
                max_tokens=800,
                json_schema=get_state_delta_schema(),
                think=False,
            )

            delta = self._parse_response(response.content)

            logger.info(
                "state_delta_extracted",
                has_time=delta.time_change is not None,
                has_location=delta.location_change is not None,
                new_npcs=len(delta.new_npcs),
                npc_updates=len(delta.npc_updates),
                new_events=len(delta.new_events),
                new_facts=len(delta.new_facts),
            )

            return delta

        except Exception as e:
            logger.warning("state_extraction_failed", error=str(e))
            return StateDelta()

    def _parse_response(self, content: str) -> StateDelta:
        """Parse LLM response into StateDelta."""
        if not content:
            return StateDelta()

        content = content.strip()

        # Strip markdown code fences
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()
        elif "```" in content:
            start = content.find("```") + 3
            end = content.find("```", start)
            if end > start:
                content = content[start:end].strip()

        # Extract JSON object
        if "{" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            if end > start:
                content = content[start:end]

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(
                "state_delta_json_parse_failed",
                error=str(e),
                content_preview=content[:200],
            )
            return StateDelta()

        try:
            return StateDelta(**data)
        except ValidationError as e:
            logger.warning(
                "state_delta_validation_failed",
                error=str(e),
                data_preview=str(data)[:300],
            )
            return StateDelta()


# Singleton
_state_extractor: Optional[StateExtractor] = None


def get_state_extractor() -> StateExtractor:
    """Get or create the StateExtractor singleton."""
    global _state_extractor
    if _state_extractor is None:
        _state_extractor = StateExtractor()
    return _state_extractor
