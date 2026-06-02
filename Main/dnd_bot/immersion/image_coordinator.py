"""Image generation coordinator: frequency logic and orchestration.

Decides whether to generate an image based on guild settings (every,
scene_change, on_demand) and orchestrates the prompt reframing + generation.
"""

from typing import Optional

import structlog

from ..models.immersion import GuildImmersionSettings, ImageFrequency
from ..models.character import Character
from ..llm.narrative_signals import is_scene_change as _shared_is_scene_change

logger = structlog.get_logger()


async def maybe_generate_image(
    narrative: str,
    proposed_effects: list = None,
    settings: Optional[GuildImmersionSettings] = None,
    scene_registry=None,
    characters: list[Character] = None,
) -> Optional[bytes]:
    """Check frequency setting and generate a scene image if appropriate.

    Args:
        narrative: The narrator's prose output.
        proposed_effects: ProposedEffect list from narrator tool calls.
        settings: Guild immersion settings.
        scene_registry: SceneEntityRegistry for context.
        characters: Player characters in the session.

    Returns:
        PNG image bytes, or None if generation was skipped or failed.
    """
    if not settings or not settings.image_enabled:
        return None

    proposed_effects = proposed_effects or []
    characters = characters or []

    frequency = settings.image_frequency

    # Check frequency rules
    if frequency == ImageFrequency.ON_DEMAND:
        return None  # Only /imagine triggers generation

    if frequency == ImageFrequency.SCENE_CHANGE:
        if not _is_scene_change(proposed_effects):
            return None

    # frequency == EVERY falls through to always generate

    return await generate_scene_image(
        narrative=narrative,
        scene_registry=scene_registry,
        characters=characters,
    )


async def generate_scene_image(
    narrative: str,
    scene_registry=None,
    characters: list[Character] = None,
) -> Optional[bytes]:
    """Generate a scene image from narrative context (used by both auto and /imagine).

    Returns PNG bytes or None.
    """
    characters = characters or []

    try:
        from .image_prompter import reframe_for_image, build_character_descriptions
        from .image_factory import get_image_provider, get_provider_name

        # Gather descriptions
        scene_desc = ""
        if scene_registry:
            scene_desc = getattr(scene_registry, '_scene_description', '') or ''

        scene_entities = scene_registry.get_all() if scene_registry else []
        player_descs, entity_descs = build_character_descriptions(
            characters, scene_entities
        )

        # Reframe narrative into image prompt
        prompt = await reframe_for_image(
            narrative=narrative,
            scene_description=scene_desc,
            entity_descriptions=entity_descs,
            player_descriptions=player_descs,
        )

        if not prompt:
            return None

        # Generate image
        provider = get_image_provider()
        # Local provider uses smaller resolution to fit in VRAM, cloud uses full
        size = "768x768" if get_provider_name() == "local" else "1024x1024"
        image_bytes = await provider.generate(prompt, size=size)

        logger.info("scene_image_generated", prompt_length=len(prompt), image_bytes=len(image_bytes))
        return image_bytes

    except Exception as e:
        logger.warning("scene_image_generation_failed", error=str(e))
        return None


def _is_scene_change(proposed_effects: list) -> bool:
    """Backwards-compatible wrapper over the shared narrative_signals module.

    Existing tests import this symbol directly; keeping the alias avoids
    churn while the real implementation lives in
    ``dnd_bot.llm.narrative_signals``.
    """
    return _shared_is_scene_change(proposed_effects or [])
