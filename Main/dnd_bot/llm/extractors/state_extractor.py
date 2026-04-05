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
**location_change** - Short place NAME only (2-4 words max). Examples: "the tavern", "forest clearing", "north gate". NEVER a description or sentence. If the narrative just describes surroundings without the party moving to a distinct named area, do NOT set this.
**location_description** - Brief description of the new location (only if location changed)
**new_connections** - Newly revealed exits or paths from current location. SHORT place names only (2-4 words, e.g., "north gate", "the river", "dark cave"). NEVER full sentences or descriptions.
**new_npcs** - NPCs appearing for the FIRST time. Include:
  - name: the NPC's name or short label (e.g., "Marta", "the blacksmith")
  - location: MUST use the current location name from the world state (the "location" field). Do NOT invent sub-locations like "behind the stall" or "inside the cottage". If the NPC is where the party is, use the same location string.
  - disposition (hostile/unfriendly/neutral/friendly/allied)
  - description (brief), important (true if quest-giver, ally, or key story NPC)
**npc_updates** - Changes to EXISTING NPCs (moved, changed disposition, died, etc.)
  - Only include the fields that changed
  - location: use short place names, not descriptions
**removed_npcs** - NPCs who LEFT the scene (not dead, just departed)
**new_quests** - Quests or tasks assigned to the party for the FIRST time. Include:
  - name (short quest title), giver (NPC who assigned it), status ("active")
  - objectives (list of goals), location (where to go, if mentioned)
**quest_updates** - Changes to EXISTING quests (completed, failed, new objectives revealed)
  - Only include the fields that changed
**new_events** - Significant narrative events (1 sentence each, max 2)
**new_facts** - Established facts that must not be contradicted later
**flag_changes** - World flags that changed (e.g., "bridge_destroyed": true)
**phase_change** - Only if game phase changed (exploration/combat/dialogue/rest)

## Location naming rules:
- ALL location names (location_change, new_connections, NPC location) must be SHORT: 2-4 words max.
- Use proper place names when available ("Thornfield", "the market"), not descriptions ("small makeshift stall between stable and general store").
- For NPCs at the party's location, copy the exact current location string from the world state.
- Good: "settlement", "the tavern", "north gate", "old mill"
- Bad: "a small clearing near the river bend", "interior of the sealed building", "behind the merchant's stall"

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
        referenced_entity_ids: list[str] | None = None,
    ) -> StateDelta:
        """Extract a StateDelta from narrator prose.

        Args:
            narrative_text: The narrator's output prose
            world_state_yaml: Current world state as YAML (for context)
            current_scene: Current scene description
            referenced_entity_ids: Entity IDs the narrator explicitly tagged
                via ref_entity intents. These entities are ALREADY KNOWN —
                do not create new_npcs entries for them.

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

        # Tell the extractor which NPCs are already accounted for
        if referenced_entity_ids:
            context_parts.append(
                f"These NPCs are already tracked — skip them in new_npcs "
                f"(but still extract location, events, facts, and other changes): "
                f"{', '.join(referenced_entity_ids)}"
            )

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

            delta, parse_warnings = self._parse_response(response.content)

            if parse_warnings:
                logger.warning(
                    "state_delta_parse_warnings",
                    warnings=parse_warnings,
                )

            logger.info(
                "state_delta_extracted",
                has_time=delta.time_change is not None,
                has_location=delta.location_change is not None,
                new_npcs=len(delta.new_npcs),
                npc_updates=len(delta.npc_updates),
                new_events=len(delta.new_events),
                new_facts=len(delta.new_facts),
            )

            # Attach warnings for turn log observability
            delta._parse_warnings = parse_warnings
            return delta

        except Exception as e:
            logger.warning("state_extraction_failed", error=str(e))
            return StateDelta()

    def _parse_response(self, content: str) -> tuple[StateDelta, list[str]]:
        """Parse LLM response into StateDelta.

        Returns (delta, parse_warnings) for turn log observability.
        """
        warnings: list[str] = []
        if not content:
            return StateDelta(), warnings

        content = content.strip()

        # Strip markdown code fences
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
                warnings.append("extracted_json_from_text")
            if json_end > json_start:
                content = content[json_start:json_end]

        try:
            data = json.loads(content)
        except json.JSONDecodeError as e:
            logger.warning(
                "state_delta_json_parse_failed",
                error=str(e),
                content_preview=content[:200],
            )
            warnings.append(f"json_parse_failed: {e}")
            return StateDelta(), warnings

        # Coerce bare strings to single-element lists for list fields.
        # Small models often output "new_connections": "the tavern"
        # instead of "new_connections": ["the tavern"].
        _list_fields = (
            "new_connections", "new_events", "new_facts",
            "removed_npcs", "on_success", "on_failure",
        )
        for field in _list_fields:
            if field in data and isinstance(data[field], str):
                data[field] = [data[field]]
                warnings.append(f"coerced_string_to_list:{field}")

        try:
            return StateDelta(**data), warnings
        except ValidationError as e:
            logger.warning(
                "state_delta_validation_failed",
                error=str(e),
                data_preview=str(data)[:300],
            )
            warnings.append(f"validation_failed: {e}")
            return StateDelta(), warnings


# Singleton
_state_extractor: Optional[StateExtractor] = None


def get_state_extractor() -> StateExtractor:
    """Get or create the StateExtractor singleton."""
    global _state_extractor
    if _state_extractor is None:
        _state_extractor = StateExtractor()
    return _state_extractor
