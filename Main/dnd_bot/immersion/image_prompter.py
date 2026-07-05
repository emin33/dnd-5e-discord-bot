"""Image prompt reframer: transforms narrator prose into image generation prompts.

Uses a cheap/fast LLM call to convert narrative prose into a concise
visual description optimized for image generation models. Includes
character descriptions for visual consistency.
"""

import structlog

from ..models.character import Character

logger = structlog.get_logger()

_REFRAME_SYSTEM = """You describe D&D 5e scenes for an image generator. You receive narrator prose and character descriptions from the game.

Write a single paragraph describing what a painter would see in this moment. Use the character descriptions provided to accurately depict the people in the scene. This is a medieval fantasy setting -- clothing, architecture, and equipment should reflect that.

Begin with a style tag like "Cinematic dark fantasy illustration" or "D&D concept art". Describe the setting, the characters by their physical appearance, what they are doing, and the lighting/mood. Output only the image prompt, nothing else."""


async def reframe_for_image(
    narrative: str,
    scene_description: str = "",
    entity_descriptions: list[str] = None,
    player_descriptions: list[str] = None,
) -> str:
    """Transform narrative prose into an image generation prompt.

    Args:
        narrative: The narrator's prose output.
        scene_description: Current scene/location description.
        entity_descriptions: NPC descriptions present in the scene.
        player_descriptions: Player character descriptions.

    Returns:
        A concise image generation prompt string.
    """
    entity_descriptions = entity_descriptions or []
    player_descriptions = player_descriptions or []

    # Build context for the reframer
    parts = []
    if scene_description:
        parts.append(f"Setting: {scene_description}")

    if player_descriptions:
        parts.append("Player Characters:")
        for desc in player_descriptions[:4]:  # Limit to avoid token bloat
            parts.append(f"- {desc}")

    if entity_descriptions:
        parts.append("NPCs Present:")
        for desc in entity_descriptions[:6]:
            parts.append(f"- {desc}")

    parts.append(f"Narrative:\n{narrative[:1500]}")  # Cap narrative length

    user_prompt = "\n".join(parts)

    try:
        from ..llm.client import get_llm_client
        client = get_llm_client()

        response = await client.chat(
            messages=[
                {"role": "system", "content": _REFRAME_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_tokens=200,
            temperature=0.7,
            think=False,
        )

        content = response.content if hasattr(response, 'content') else str(response)
        prompt = content.strip()

        logger.debug("image_prompt_generated", prompt_length=len(prompt))
        return prompt

    except Exception as e:
        logger.warning("image_prompt_reframe_failed", error=str(e), exc_info=True)
        # Fallback: extract first two sentences of narrative
        sentences = narrative.split(". ")[:2]
        fallback = ". ".join(sentences).strip()
        return f"{fallback}, high fantasy digital art, dramatic lighting"


def build_character_descriptions(
    characters: list[Character],
    scene_entities: list = None,
) -> tuple[list[str], list[str]]:
    """Extract visual descriptions for characters and NPCs in the scene.

    Returns (player_descriptions, entity_descriptions).
    """
    player_descs = []
    for char in characters:
        desc_parts = []
        if char.description:
            desc_parts.append(char.description)
        else:
            desc_parts.append(f"{char.race_index} {char.class_index}")
        player_descs.append(f"{char.name}: {' '.join(desc_parts)}")

    entity_descs = []
    if scene_entities:
        for entity in scene_entities:
            if hasattr(entity, 'description') and entity.description:
                entity_descs.append(f"{entity.name}: {entity.description[:150]}")

    return player_descs, entity_descs
