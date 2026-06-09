"""Regression tests for the SceneEntity setattr bypass (audit 2026-06-09, P0).

``_execute_update_entity`` used to write undeclared ``status``/``important``
fields onto SceneEntity via setattr; pydantic v2 raised
ValueError('"SceneEntity" object has no field "status"'), the executor's
broad except swallowed it into success=False, and the orchestrator dropped
the narrator's update entirely — with disposition already partially applied.
Relatedly, ``_execute_spawn_object`` passed ``properties=...`` to the
SceneEntity constructor, which pydantic v2 silently discarded.

``status``, ``important``, and ``properties`` are now declared fields on
SceneEntity (models/npc.py) and effects.py uses normal assignment.
"""

import pytest

from dnd_bot.game.scene.registry import SceneEntityRegistry
from dnd_bot.llm.effects import EffectExecutor, EffectType, ProposedEffect
from dnd_bot.models.npc import Disposition, EntityType, SceneEntity


def _registry_with_npc(name: str = "Bram the Guard") -> tuple[SceneEntityRegistry, SceneEntity]:
    """Registry holding one neutral NPC (neutral skips the SRD auto-match path)."""
    registry = SceneEntityRegistry(campaign_id="camp", channel_id=0)
    entity = registry.register_entity(SceneEntity(
        name=name,
        entity_type=EntityType.NPC,
        description="a watchful guard",
        disposition=Disposition.NEUTRAL,
    ))
    return registry, entity


@pytest.mark.asyncio
class TestUpdateEntityStatusImportance:
    """update_entity with update_status/update_importance must succeed."""

    async def test_update_status_and_importance_succeed(self):
        registry, entity = _registry_with_npc()
        executor = EffectExecutor(scene_registry=registry)

        effect = ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="Bram the Guard",
            update_disposition="friendly",
            update_status="Wounded",
            update_importance=True,
        )
        result = await executor.execute(effect)

        # Pre-fix: ValueError after the disposition write -> success=False,
        # update silently dropped with the entity partially mutated.
        assert result.success is True
        assert result.error is None
        assert entity.disposition == "friendly"
        assert entity.status == "wounded"
        assert entity.important is True
        applied = result.details["applied"]
        assert applied["disposition"] == "friendly"
        assert applied["status"] == "wounded"
        assert applied["important"] is True

    async def test_unset_fields_mean_no_change(self):
        registry, entity = _registry_with_npc()
        executor = EffectExecutor(scene_registry=registry)

        effect = ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="Bram",  # by-name resolution, as in production
            update_status="dead",
        )
        result = await executor.execute(effect)

        assert result.success is True
        assert entity.status == "dead"
        # None on the effect means "no change": defaults stay put
        assert entity.important is False
        assert entity.disposition == Disposition.NEUTRAL

    async def test_spawn_object_retains_properties(self):
        registry = SceneEntityRegistry(campaign_id="camp", channel_id=0)
        executor = EffectExecutor(scene_registry=registry)

        effect = ProposedEffect(
            effect_type=EffectType.SPAWN_OBJECT,
            object_name="iron strongbox",
            object_description="a locked iron strongbox",
            object_properties={"locked": True, "value": "50gp"},
        )
        result = await executor.execute(effect)

        assert result.success is True
        spawned = registry.get_by_id(result.details["entity_id"])
        assert spawned is not None
        # Pre-fix: pydantic v2 silently dropped the undeclared constructor kwarg
        assert spawned.properties == {"locked": True, "value": "50gp"}


class TestSceneEntitySerialization:
    """New fields must round-trip, and legacy dumps without them must load."""

    def test_round_trip_preserves_new_fields(self):
        entity = SceneEntity(
            name="Bram the Guard",
            entity_type=EntityType.NPC,
            status="wounded",
            important=True,
            properties={"locked": True},
        )
        data = entity.model_dump()
        assert data["status"] == "wounded"
        assert data["important"] is True
        assert data["properties"] == {"locked": True}

        restored = SceneEntity.model_validate(data)
        assert restored.status == "wounded"
        assert restored.important is True
        assert restored.properties == {"locked": True}

    def test_legacy_dict_without_new_keys_loads_with_defaults(self):
        # Shape of a SceneEntity dump from before the fields existed
        legacy = SceneEntity(name="Old Bram", entity_type=EntityType.NPC).model_dump()
        for key in ("status", "important", "properties"):
            legacy.pop(key)

        restored = SceneEntity.model_validate(legacy)
        assert restored.status is None
        assert restored.important is False
        assert restored.properties == {}
