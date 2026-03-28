"""Scene management - tracks entities present in the current scene."""

from .registry import (
    SceneEntityRegistry,
    get_scene_registry,
    clear_scene_registry,
    HOSTILITY_CALM,
    HOSTILITY_AGITATED,
    HOSTILITY_THREATENING,
    HOSTILITY_COMBAT,
)

__all__ = [
    "SceneEntityRegistry",
    "get_scene_registry",
    "clear_scene_registry",
    "HOSTILITY_CALM",
    "HOSTILITY_AGITATED",
    "HOSTILITY_THREATENING",
    "HOSTILITY_COMBAT",
]
