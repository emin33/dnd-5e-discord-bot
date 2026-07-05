"""Entity Extractor - extracts entities from narrator output."""

from typing import Optional
import structlog

from pydantic import BaseModel

from ..client import get_llm_client, OllamaClient
from ..json_extract import extract_json_object
from ...models.npc import SceneEntity, EntityType, Disposition

logger = structlog.get_logger()


class ExtractedEntity(BaseModel):
    """An entity extracted from narrative text."""
    name: str
    entity_type: str  # "npc", "creature", "object"
    description: str
    disposition: str = "neutral"  # "hostile", "unfriendly", "neutral", "friendly"
    monster_index: Optional[str] = None  # SRD monster if identifiable


class HostilityChange(BaseModel):
    """A hostility change extracted from narrative."""
    name: str
    delta: int  # Change in hostility
    reason: str


class ExtractionResult(BaseModel):
    """Result of entity extraction."""
    entities: list[ExtractedEntity] = []
    scene_update: Optional[str] = None  # If scene context changed
    hostility_changes: list[HostilityChange] = []
    combat_initiated: bool = False  # True ONLY if combat actually begins in the narrative


EXTRACTION_SYSTEM_PROMPT = """You extract entities from D&D narrative prose.

## What counts as an entity

Three categories — anything else is atmosphere or spell effect, not an entity.

- **NPC** — a being with identity that can be talked to (named or unnamed but notable: "Grimjaw the dwarf", "the hooded stranger", "the innkeeper").
- **Creature** — a being that can be fought (named or by type: "three goblins", "a dire wolf", "the owlbear").
- **Object** — an interactable item or feature: "a locked chest", "a glowing sword", "the ancient altar".

NOT entities: spell effects ("writhing vines", "shadow tendrils"), environmental hazards ("the gust of wind"), narrative flavor ("the smell of roasting meat"), atmospheric description.

## Disposition — judge by signal, not vibe

Look for what the entity is DOING right now:

- **hostile** — actively attacking. Signals: charging, swinging, casting offensively, firing weapons. NOT signals: drawing weapons, looking angry, moving closer, "looking hostile".
- **unfriendly** — antagonistic but not yet attacking. Signals: blocking the path, refusing entry, threatening, accusing.
- **neutral** — indifferent, no stake in the party either way.
- **friendly** — welcoming, helpful, supportive.

When in doubt between hostile and unfriendly, pick unfriendly. Hostile requires an attack happening *right now*.

## monster_index — required for combat-capable entities

If the entity could plausibly fight, provide the closest SRD monster index. Format: lowercase with hyphens. Apply the rule that fits the entity's role/equipment/behavior:

- Soldier or guard role → "guard". With elite gear → "veteran".
- Common enemy with weapon → "thug" or "bandit". Leader → "bandit-captain".
- Stealthy killer → "assassin". Stealthy scout → "spy".
- Spellcaster → "mage". Religious caster → "priest" or "cult-fanatic".
- Heavily armored knight → "knight".
- Civilian (innkeeper, merchant, villager) → "commoner".
- Wolf-like → "wolf". Bear → "brown-bear". Rat → "giant-rat".

If no SRD match fits, use null. Do not invent indices.

## combat_initiated

`true` ONLY when the narrative shows an actual attack landing or being executed against the party right now (a swing connecting, an arrow flying, a spell being cast offensively).

`false` when the narrative shows tension, drawing weapons, threats, hostile posture, or even an entrance ("the goblins burst into the room"). Tension is not combat — combat starts when the first blow is thrown.

A useful test: if the player still has a choice to flee, parley, or fight, combat has not started.

## Extraction rules

- Only emit NEW entities or entities whose disposition or status CHANGED.
- Skip unchanged entities even if mentioned again.
- `scene_update` is optional — set only when the scene's mood or state shifted.
- For purely descriptive narrative with no entity changes, return `{"entities": [], "scene_update": null, "hostility_changes": [], "combat_initiated": false}`.

## Output format

```json
{
    "entities": [
        {
            "name": "Burly Dwarf",
            "entity_type": "npc",
            "description": "A stocky dwarf with a scarred face and smith's apron",
            "disposition": "unfriendly",
            "monster_index": null
        }
    ],
    "scene_update": "The forge grows tense",
    "hostility_changes": [],
    "combat_initiated": false
}
```"""


# Cache the schema
_EXTRACTION_SCHEMA: Optional[dict] = None


def get_extraction_schema() -> dict:
    """Get JSON schema for extraction."""
    global _EXTRACTION_SCHEMA
    if _EXTRACTION_SCHEMA is None:
        _EXTRACTION_SCHEMA = ExtractionResult.model_json_schema()
    return _EXTRACTION_SCHEMA


class EntityExtractor:
    """
    Extracts entities from narrator output using LLM.

    This runs AFTER the narrator generates output, parsing
    the narrative to identify NPCs, creatures, and objects
    that should be tracked in the scene.
    """

    def __init__(self, client: Optional[OllamaClient] = None):
        self.client = client or get_llm_client()

    async def extract(
        self,
        narrative_text: str,
        current_scene: str = "",
        existing_entities: Optional[list[str]] = None,
    ) -> ExtractionResult:
        """
        Extract entities from narrative text.

        Args:
            narrative_text: The narrator's output
            current_scene: Current scene description for context
            existing_entities: Names of entities already in the scene
        """
        if not narrative_text or len(narrative_text) < 20:
            return ExtractionResult()

        # Build context
        context_parts = []
        if current_scene:
            context_parts.append(f"Current Scene: {current_scene}")
        if existing_entities:
            context_parts.append(f"Already Present: {', '.join(existing_entities)}")

        context = "\n".join(context_parts) if context_parts else "New scene"

        messages = [
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Context:\n{context}\n\nNarrative:\n{narrative_text}"},
        ]

        try:
            response = await self.client.chat(
                messages=messages,
                temperature=0,  # Deterministic extraction
                max_tokens=800,
                json_schema=get_extraction_schema(),
                think=False,  # No thinking for extraction
            )

            result = self._parse_response(response.content)

            logger.info(
                "entities_extracted",
                entity_count=len(result.entities),
                hostility_changes=len(result.hostility_changes),
                has_scene_update=result.scene_update is not None,
            )

            return result

        except Exception as e:
            logger.error("entity_extraction_failed", error=str(e))
            return ExtractionResult()

    def _parse_response(self, content: str) -> ExtractionResult:
        """Parse LLM response into ExtractionResult."""
        if not content:
            return ExtractionResult()

        data, warnings = extract_json_object(content)
        if data is None:
            logger.warning(
                "extraction_parse_failed",
                content=content[:100],
                warnings=warnings,
            )
            return ExtractionResult()
        if warnings:
            logger.debug("extraction_json_recovered", warnings=warnings)

        entities = []
        for e in data.get("entities", []):
            try:
                entities.append(ExtractedEntity(**e))
            except Exception:
                continue

        hostility_changes = []
        for h in data.get("hostility_changes", []):
            try:
                hostility_changes.append(HostilityChange(**h))
            except Exception:
                continue

        return ExtractionResult(
            entities=entities,
            scene_update=data.get("scene_update"),
            hostility_changes=hostility_changes,
            combat_initiated=bool(data.get("combat_initiated", False)),
        )

    def convert_to_scene_entity(self, extracted: ExtractedEntity) -> SceneEntity:
        """Convert ExtractedEntity to SceneEntity."""
        # Map string types to enums
        type_map = {
            "npc": EntityType.NPC,
            "creature": EntityType.CREATURE,
            "object": EntityType.OBJECT,
            "environmental": EntityType.ENVIRONMENTAL,
        }
        disposition_map = {
            "hostile": Disposition.HOSTILE,
            "unfriendly": Disposition.UNFRIENDLY,
            "neutral": Disposition.NEUTRAL,
            "friendly": Disposition.FRIENDLY,
            "allied": Disposition.ALLIED,
        }

        entity_type = type_map.get(extracted.entity_type.lower(), EntityType.NPC)
        disposition = disposition_map.get(extracted.disposition.lower(), Disposition.NEUTRAL)

        # Set initial hostility based on disposition
        hostility_map = {
            Disposition.HOSTILE: 80,
            Disposition.UNFRIENDLY: 40,
            Disposition.NEUTRAL: 0,
            Disposition.FRIENDLY: 0,
            Disposition.ALLIED: 0,
        }

        # Auto-generate aliases from name parts
        # "Assassin Leader" → ["leader", "assassin"]
        # "The Hooded Stranger" → ["stranger", "hooded stranger"]
        skip_words = {"the", "a", "an", "of", "at", "in", "on", "to", "and", "or", "with"}
        name_words = [w for w in extracted.name.lower().split() if w not in skip_words and len(w) > 2]
        aliases = []
        if len(name_words) > 1:
            # Add the last significant word (usually the noun: "leader", "stranger", "wolf")
            aliases.append(name_words[-1])
            # Add the full name without articles
            cleaned = " ".join(name_words)
            if cleaned != extracted.name.lower():
                aliases.append(cleaned)

        return SceneEntity(
            name=extracted.name,
            entity_type=entity_type,
            description=extracted.description,
            disposition=disposition,
            hostility_score=hostility_map.get(disposition, 0),
            monster_index=extracted.monster_index,
            aliases=aliases,
        )


# Global extractor instance
_extractor: Optional[EntityExtractor] = None


def get_entity_extractor() -> EntityExtractor:
    """Get the global entity extractor."""
    global _extractor
    if _extractor is None:
        _extractor = EntityExtractor()
    return _extractor
