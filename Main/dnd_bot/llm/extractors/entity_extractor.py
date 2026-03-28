"""Entity Extractor - extracts entities from narrator output."""

from typing import Optional
import json
import structlog

from pydantic import BaseModel

from ..client import get_llm_client, OllamaClient
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


EXTRACTION_SYSTEM_PROMPT = """You extract entities from D&D narrative text.

Given narrative text from a Dungeon Master, identify:

1. **NPCs** - Named characters with distinct identity
   - Example: "Grimjaw the dwarf", "Mayor Helena", "the hooded stranger"
   - Include unnamed but notable NPCs like "the innkeeper", "a burly dwarf"

2. **Creatures** - Monsters or beasts (named or unnamed)
   - Example: "three goblins", "a dire wolf", "the owlbear"

3. **Notable Objects** - Interactable items or features
   - Example: "a locked chest", "a glowing sword", "the ancient altar"

For each entity, determine:
- **disposition**: How they seem toward the party
  - "hostile" - aggressive, attacking, threatening violence
  - "unfriendly" - antagonistic but not violent yet
  - "neutral" - indifferent or unknown
  - "friendly" - welcoming, helpful

- **monster_index**: For ANY entity that could participate in combat, you MUST provide the closest SRD monster match. This is required for combat to work.
  - Use lowercase with hyphens for multi-word names
  - Common mappings:
    - Guards, soldiers, watchmen → "guard"
    - Thugs, enforcers, brutes → "thug"
    - Bandits, robbers, highwaymen, hooded figures → "bandit"
    - Bandit leaders, gang bosses → "bandit-captain"
    - Assassins, shadowy killers → "assassin"
    - Mages, wizards, sorcerers, spellcasters → "mage"
    - Cultists, fanatics, robed figures → "cultist" or "cult-fanatic"
    - Commoners, villagers, merchants, innkeepers → "commoner"
    - Knights, paladins, champions → "knight"
    - Priests, clerics, healers → "priest"
    - Spies, rogues, scouts → "spy"
    - Animals: wolves → "wolf", bears → "brown-bear", rats → "giant-rat"
  - If truly no match exists, use null — but try hard to find one

Also determine if **combat actually started** in this narrative:
- combat_initiated = true ONLY if creatures/NPCs are actively attacking the player RIGHT NOW
- Examples of TRUE: "The goblins charge at you", "The bandit swings his sword", "Arrows fly toward you"
- Examples of FALSE: "The guards look hostile", "Tensions are rising", "They draw weapons threateningly", "The intruders burst in" (threatening but not attacking the player yet)
- If the player has a CHOICE to fight or flee, combat has NOT started yet

Output valid JSON:
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

Rules:
- Only include NEW entities or significantly CHANGED entities
- If an entity was already introduced and unchanged, skip it
- scene_update is optional - only include if the scene atmosphere changed
- combat_initiated should be true ONLY when NPCs/creatures are actively attacking the player
- Do NOT register magical effects, environmental hazards, or narrative flavor as entities (e.g., "writhing vines", "shadow tendrils", "dark energy" are NOT creatures — they are spell effects or atmosphere)
- Only register things that are actual beings that can be talked to, fought, or interacted with as objects
- If purely descriptive with no entities: {"entities": [], "scene_update": null, "hostility_changes": [], "combat_initiated": false}"""


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

        content = content.strip()

        # Handle markdown code blocks
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

        # Find JSON object
        if "{" in content:
            start = content.find("{")
            end = content.rfind("}") + 1
            if end > start:
                content = content[start:end]

        try:
            data = json.loads(content)

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
        except (json.JSONDecodeError, TypeError) as e:
            logger.warning(
                "extraction_parse_failed",
                content=content[:100] if content else "empty",
                error=str(e),
            )
            return ExtractionResult()

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
