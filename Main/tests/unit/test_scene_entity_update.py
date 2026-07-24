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

from types import SimpleNamespace

import pytest

from dnd_bot.game.scene.registry import SceneEntityRegistry
from dnd_bot.game.world_state import NPCState, WorldState
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
        assert entity.disposition is Disposition.FRIENDLY
        assert entity.disposition.value == "friendly"
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

    async def test_known_off_scene_npc_update_is_accepted_for_world_store(self):
        world = WorldState(campaign_id="camp")
        npc = NPCState(name="Mara Venn", location="")
        world.npcs[npc.id] = npc
        executor = EffectExecutor(
            scene_registry=SceneEntityRegistry(campaign_id="camp", channel_id=0),
            session=SimpleNamespace(world_state=world),
        )

        result = await executor.execute(ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id=npc.id,
            update_status="fled",
        ))

        assert result.success is True
        assert result.details["found_in_scene"] is False
        assert result.details["found_in_world"] is True

    async def test_graph_only_npc_update_is_accepted_without_scene_materialization(self):
        npc_id = "d04bbdac-c09f-4c1e-855b-5f395546d986"
        graph_npc = SimpleNamespace(
            node_id=npc_id,
            name="Thessa",
            aliases=[],
            entity_type=SimpleNamespace(value="npc"),
        )
        graph = SimpleNamespace(
            get_entity=lambda entity_id: (
                graph_npc if entity_id == npc_id else None
            ),
            resolve_entity_reference=lambda _reference: None,
        )
        world = WorldState(campaign_id="camp", current_location="The Silver Needle")
        executor = EffectExecutor(
            scene_registry=SceneEntityRegistry(campaign_id="camp", channel_id=0),
            session=SimpleNamespace(
                world_state=world,
                knowledge_graph=graph,
            ),
        )

        result = await executor.execute(ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id=npc_id,
            update_description_addition="stands before a small shrine",
        ))

        assert result.success is True
        assert result.details["found_in_scene"] is False
        assert result.details["found_in_world"] is True
        assert world.npcs == {}

    async def test_graph_item_update_is_accepted(self):
        # Soak 20260723_230351 turn 16: 'living-brass-compass' is a real KG
        # item node (alongside its twin 'brass-compass'), so validation
        # passed — but execution only resolved NPCs and rejected it.
        def _item(node_id, name):
            return SimpleNamespace(
                node_id=node_id,
                name=name,
                aliases=[],
                entity_type=SimpleNamespace(value="item"),
            )
        nodes = {
            "living-brass-compass": _item(
                "living-brass-compass", "living brass compass"
            ),
            "brass-compass": _item("brass-compass", "brass compass"),
        }
        graph = SimpleNamespace(
            get_entity=lambda entity_id: nodes.get(entity_id),
            resolve_entity_reference=lambda _reference: None,
        )
        executor = EffectExecutor(
            scene_registry=SceneEntityRegistry(campaign_id="camp", channel_id=0),
            session=SimpleNamespace(
                world_state=WorldState(campaign_id="camp"),
                knowledge_graph=graph,
            ),
        )

        result = await executor.execute(ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="living-brass-compass",
            update_description_addition="its needle now points down",
        ))

        assert result.success is True
        assert result.details["found_in_scene"] is False
        assert result.details["found_in_world"] is True
        assert result.details["world_reference_type"] == "item"

    async def test_graph_item_status_update_is_accepted(self):
        # Soak 20260723_230351 turn 23: status change on the KG item
        # 'orris-vanes-hidden-note' died the same execution-only death.
        note = SimpleNamespace(
            node_id="orris-vanes-hidden-note",
            name="Orris Vane's hidden note",
            aliases=[],
            entity_type=SimpleNamespace(value="item"),
        )
        graph = SimpleNamespace(
            get_entity=lambda entity_id: (
                note if entity_id == "orris-vanes-hidden-note" else None
            ),
            resolve_entity_reference=lambda _reference: None,
        )
        executor = EffectExecutor(
            scene_registry=SceneEntityRegistry(campaign_id="camp", channel_id=0),
            session=SimpleNamespace(
                world_state=WorldState(campaign_id="camp"),
                knowledge_graph=graph,
            ),
        )

        result = await executor.execute(ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="orris-vanes-hidden-note",
            update_status="revealed",
        ))

        assert result.success is True
        assert result.details["applied"]["status"] == "revealed"
        assert result.details["world_reference_type"] == "item"

    async def test_scene_item_slug_update_is_accepted(self):
        # Soak 20260723_230351 turn 69: 'carved-wooden-door' is the slug
        # dialect of the WorldState scene item 'carved wooden door'.
        world = WorldState(
            campaign_id="camp",
            scene_items={"carved wooden door": "A dark oiled wooden door."},
        )
        executor = EffectExecutor(
            scene_registry=SceneEntityRegistry(campaign_id="camp", channel_id=0),
            session=SimpleNamespace(world_state=world),
        )

        result = await executor.execute(ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="carved-wooden-door",
            update_description_addition="a compass-needle symbol glows faintly",
        ))

        assert result.success is True
        assert result.details["found_in_world"] is True
        assert result.details["world_reference_type"] == "item"

    async def test_unknown_off_scene_npc_update_is_rejected(self):
        world = WorldState(campaign_id="camp")
        executor = EffectExecutor(
            scene_registry=SceneEntityRegistry(campaign_id="camp", channel_id=0),
            session=SimpleNamespace(world_state=world),
        )

        result = await executor.execute(ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="invented-id",
            update_status="fled",
        ))

        assert result.success is False
        assert "not a known entity" in result.error


@pytest.mark.asyncio
async def test_unknown_ref_entity_is_rejected():
    executor = EffectExecutor(
        scene_registry=SceneEntityRegistry(campaign_id="camp", channel_id=0),
        session=SimpleNamespace(world_state=WorldState(campaign_id="camp")),
    )

    result = await executor.execute(ProposedEffect(
        effect_type=EffectType.REF_ENTITY,
        ref_entity_id="invented-id",
    ))

    assert result.success is False
    assert "not a known entity" in result.error


@pytest.mark.asyncio
async def test_current_location_ref_entity_is_known_world_reference():
    world = WorldState(campaign_id="camp", current_location="Ash Gate")
    executor = EffectExecutor(
        scene_registry=SceneEntityRegistry(campaign_id="camp", channel_id=0),
        session=SimpleNamespace(world_state=world),
    )

    result = await executor.execute(ProposedEffect(
        effect_type=EffectType.REF_ENTITY,
        ref_entity_id="ash-gate",
    ))

    assert result.success is True
    assert result.details["found_in_scene"] is False
    assert result.details["found_in_world"] is True
    assert result.details["world_reference_type"] == "location"


@pytest.mark.asyncio
async def test_explicit_known_past_location_ref_is_valid_without_ambient_seeding():
    world = WorldState(
        campaign_id="camp",
        current_location="Ash Gate",
        connected_locations=["Copper Finch"],
    )
    executor = EffectExecutor(
        scene_registry=SceneEntityRegistry(campaign_id="camp", channel_id=0),
        session=SimpleNamespace(world_state=world),
    )

    result = await executor.execute(ProposedEffect(
        effect_type=EffectType.REF_ENTITY,
        ref_entity_id="copper-finch",
    ))

    assert result.success is True
    assert result.details["world_reference_type"] == "location"


@pytest.mark.asyncio
async def test_explicit_graph_catalog_location_ref_is_valid():
    graph_location = SimpleNamespace(
        entity_type=SimpleNamespace(value="location"),
        name="Copper Finch",
    )
    graph = SimpleNamespace(
        get_entity=lambda entity_id: (
            graph_location if entity_id == "copper-finch" else None
        )
    )
    executor = EffectExecutor(
        scene_registry=SceneEntityRegistry(campaign_id="camp", channel_id=0),
        session=SimpleNamespace(
            world_state=WorldState(campaign_id="camp", current_location="Ash Gate"),
            knowledge_graph=graph,
        ),
    )

    result = await executor.execute(ProposedEffect(
        effect_type=EffectType.REF_ENTITY,
        ref_entity_id="copper-finch",
    ))

    assert result.success is True
    assert result.details["world_reference_type"] == "location"


@pytest.mark.asyncio
async def test_explicit_graph_catalog_name_ref_resolves_uuid_entity():
    graph_npc = SimpleNamespace(
        node_id="4d4f5bed-eeae-4c77-b096-fd5de5228ec3",
        entity_type=SimpleNamespace(value="npc"),
        name="Tomas Kell",
        aliases=[],
    )
    graph = SimpleNamespace(
        get_entity=lambda entity_id: None,
        resolve_entity_reference=lambda reference: (
            graph_npc if reference in {"tomas-kell", "tomas_kell"} else None
        ),
    )
    executor = EffectExecutor(
        scene_registry=SceneEntityRegistry(campaign_id="camp", channel_id=0),
        session=SimpleNamespace(
            world_state=WorldState(campaign_id="camp", current_location="Ash Gate"),
            knowledge_graph=graph,
        ),
    )

    result = await executor.execute(ProposedEffect(
        effect_type=EffectType.REF_ENTITY,
        ref_entity_id="tomas_kell",
    ))

    assert result.success is True
    assert result.details["world_reference_type"] == "npc"


@pytest.mark.asyncio
async def test_ref_entity_uses_known_alias_when_generated_slug_drifted():
    graph_npc = SimpleNamespace(
        node_id="1ffaed93-893b-4824-a9d2-4fa5f7bf68f1",
        entity_type=SimpleNamespace(value="npc"),
        name="Renn Farrow",
        aliases=[],
    )
    graph = SimpleNamespace(
        get_entity=lambda entity_id: None,
        resolve_entity_reference=lambda reference: (
            graph_npc if reference == "Renn Farrow" else None
        ),
    )
    executor = EffectExecutor(
        scene_registry=SceneEntityRegistry(campaign_id="camp", channel_id=0),
        session=SimpleNamespace(
            world_state=WorldState(campaign_id="camp", current_location="Ash Gate"),
            knowledge_graph=graph,
        ),
    )

    result = await executor.execute(ProposedEffect(
        effect_type=EffectType.REF_ENTITY,
        ref_entity_id="renns-farrow",
        ref_alias_used="Renn Farrow",
    ))

    assert result.success is True
    assert result.details["alias_used"] == "Renn Farrow"
    assert result.details["world_reference_type"] == "npc"


def test_ref_entity_validator_rejects_missing_roster_id():
    from dnd_bot.llm.effects import EffectValidator

    effect = ProposedEffect(effect_type=EffectType.REF_ENTITY, ref_entity_id="")
    result = EffectValidator().validate(effect)

    assert result.valid is False
    assert result.rejection_reason == "ref_entity requires entity_id from the roster"


def test_live_validator_rejects_invented_ref_and_update_targets():
    from types import SimpleNamespace

    from dnd_bot.game.world_state import WorldState
    from dnd_bot.llm.effects import EffectValidator

    validator = EffectValidator(session=SimpleNamespace(world_state=WorldState()))

    ref_result = validator.validate(ProposedEffect(
        effect_type=EffectType.REF_ENTITY,
        ref_entity_id="elara_vex",
    ))
    update_result = validator.validate(ProposedEffect(
        effect_type=EffectType.UPDATE_ENTITY,
        update_entity_id="elara_vex",
        update_importance=True,
    ))

    assert ref_result.valid is False
    assert "not a known entity" in ref_result.rejection_reason
    assert update_result.valid is False
    assert "not a known entity" in update_result.rejection_reason


def test_live_validator_accepts_world_state_entity_by_name_or_id():
    from types import SimpleNamespace

    from dnd_bot.game.world_state import NPCState, WorldState
    from dnd_bot.llm.effects import EffectValidator

    world = WorldState()
    world.npcs["elara-id"] = NPCState(id="elara-id", name="Elara Vex")
    validator = EffectValidator(session=SimpleNamespace(world_state=world))

    by_name = validator.validate(ProposedEffect(
        effect_type=EffectType.REF_ENTITY,
        ref_entity_id="Elara Vex",
    ))
    by_id = validator.validate(ProposedEffect(
        effect_type=EffectType.UPDATE_ENTITY,
        update_entity_id="elara-id",
        update_importance=True,
    ))

    assert by_name.valid is True
    assert by_id.valid is True


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
