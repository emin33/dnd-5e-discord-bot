"""State Extractor - extracts StateDelta from narrator output.

Runs AFTER the narrator generates prose. Uses the cheap brain model
to emit a structured StateDelta describing what changed in the world.
Python validates the delta before applying it to WorldState.

This is a superset of EntityExtractor: it captures NPC changes plus
time, location, events, and facts.
"""

from typing import Optional
import structlog

from pydantic import ValidationError

from ..client import get_llm_client, OllamaClient
from ..json_extract import extract_json_object
from ...game.world_state import StateDelta, get_state_delta_schema

logger = structlog.get_logger()


EXTRACTION_PROMPT = """You extract world state changes from D&D narrative prose.

Given the narrator's prose and the current world state, identify WHAT CHANGED.
Only include fields that changed. Omit fields where nothing changed.

## Fields you may emit

- **time_change** — only if time of day shifted (dawn / morning / midday / afternoon / dusk / evening / night / midnight).
- **location_change** — set ONLY if the party moved to a distinct named area.
- **location_description** — brief sentence describing the new location, only if location changed.
- **new_connections** — list of newly revealed exits/paths from current location.
- **new_npcs** — list of NPCs appearing for the FIRST time. Each has: `name`, `location`, `disposition`, `description`, `important` (bool).
- **npc_updates** — list of changes to EXISTING NPCs. Each entry resolves a target by either:
  - `id` — the UUID from the world state's NPC entries (PREFERRED when the registry has the entity).
  - `name` — fallback lookup; matches the entity's current name OR any string in its `aliases` list.

  Then include only changed fields. Special fields beyond `location` / `disposition` / `description` / `alive` / `notes` / `important`:
  - `new_name` — the entity has been **renamed** by the prose ("the hooded figure introduces himself as Fred"). Old name auto-moves into aliases.
  - `add_aliases` — list of paraphrased names observed in prose, but identity unchanged ("Old Bram" when the registry has "Bram"; "the cloaked one" when the same character was "the hooded figure"). Identity stays; the alias is recorded so future paraphrases match.
  - `add_inventory` / `remove_inventory` — items the NPC now holds or no longer holds.
- **removed_npcs** — list of names who LEFT the scene (not dead, just departed).
- **new_quests** — list of NEW quests. Each: `name`, `giver`, `status` ("active"), `objectives` (list), `location` (string, optional).
- **quest_updates** — list of changes to EXISTING quests. Include only changed fields.
- **new_events** — list of significant narrative events (1 sentence each, max 2).
- **new_facts** — list of established facts that must not be contradicted later.
- **flag_changes** — object of world flags that changed (e.g., `{"bridge_destroyed": true}`).
- **phase_change** — only if game phase changed (exploration / combat / dialogue / rest).

## Place names — strict format

Use this format for every location string (location_change, new_connections, NPC `location`, quest `location`):

- 2-4 words, optionally with one article ("the", "a")
- Either a proper name ("Thornfield", "Grimstone Hall") or a short generic ("the tavern", "north gate", "old mill", "ruined clearing")

For NPCs at the party's current location, copy the exact location string from the current world state context — do not invent sub-locations.

Discriminative examples — these formats are accepted vs rejected:

- "the tavern" ✓ vs "behind the bar inside the tavern" ✗
- "ruined clearing" ✓ vs "a small clearing near the river bend" ✗
- "Grimstone vault" ✓ vs "interior of the sealed dwarven vault" ✗

If the narrative describes a new area without naming it, INVENT a 2-word name from its character (a clearing with a shrine → "shrine clearing").

## Field types — strict

- All list fields must be JSON arrays, even if there's only one entry: `"new_facts": ["The bridge is out."]`. Never a bare string.
- All boolean fields must be `true` or `false`, not strings.

## Disposition signals

- **hostile** — actively attacking right now (charging, swinging, casting offensively).
- **unfriendly** — antagonistic but not yet attacking (blocking, refusing, threatening).
- **neutral** — indifferent.
- **friendly** — helpful, accommodating.
- **allied** — fights alongside the party.

`important` = true ONLY when the NPC is a quest-giver, named ally, ruler, or key story figure.

## Extraction rules

- Extract ONLY what the narrative establishes — do not infer or speculate beyond the prose.
- Do NOT create `new_npcs` for entities already in the world state.
- **Prefer `npc_updates` over `new_npcs` whenever the prose refers to an entity already in the registry — even when the prose paraphrases the name** ("Old Bram" while the registry has "Bram"; "the cloaked stranger" while "the hooded figure" already exists with matching description). Match by id, current name, alias, archetype, role, or distinguishing description (clothing, location, profession). Emit `npc_updates: [{id: "<existing-uuid>", add_aliases: ["<paraphrased name>"]}]` to record the alias rather than fragmenting one character into multiple records.
- Use `npc_updates[].new_name` ONLY when prose explicitly establishes a new identity for a previously placeholder-named entity (e.g., "the stranger reveals her name is Marta" → `npc_updates: [{id, new_name: "Marta"}]`). Otherwise use `add_aliases` for paraphrases.
- Keep `description` under 100 characters.
- If nothing changed, return `{}`.

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
            logger.warning("state_extraction_failed", error=str(e), exc_info=True)
            return StateDelta()

    def _parse_response(self, content: str) -> tuple[StateDelta, list[str]]:
        """Parse LLM response into StateDelta.

        Returns (delta, parse_warnings) for turn log observability.
        """
        if not content:
            return StateDelta(), []

        data, warnings = extract_json_object(content)
        if data is None:
            logger.warning(
                "state_delta_json_parse_failed",
                warnings=warnings,
                content_preview=content[:200],
            )
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
                exc_info=True,
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
