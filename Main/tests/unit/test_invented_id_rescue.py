"""Unit tests for narrator-invented id rescue (token-superset resolution).

Live cases from soak 20260723_005611: the narrator embellished real
referents into compound descriptive slugs, which fail-closed rejection then
dropped even though the referent was unambiguous.
"""

from types import SimpleNamespace

from dnd_bot.game.world_state import NPCState, WorldState
from dnd_bot.llm.effects import EffectType, ProposedEffect
from dnd_bot.llm.orchestrator import _resolve_invented_scene_ids


def _world(scene_items=None, npcs=None) -> WorldState:
    return WorldState(
        scene_items=scene_items or {},
        npcs=npcs or {},
    )


def _update(entity_id: str) -> ProposedEffect:
    return ProposedEffect(
        effect_type=EffectType.UPDATE_ENTITY,
        update_entity_id=entity_id,
    )


def _ref(entity_id: str) -> ProposedEffect:
    return ProposedEffect(
        effect_type=EffectType.REF_ENTITY,
        ref_entity_id=entity_id,
    )


class TestInventedIdRescue:
    def test_embellished_item_id_resolves(self):
        world = _world(scene_items={"low wooden door": "A low wooden door."})
        out = _resolve_invented_scene_ids(
            [_update("low-wooden-door-with-token-indentation")], world, None
        )
        assert out[0].update_entity_id == "low wooden door"

    def test_prefixed_item_id_resolves(self):
        world = _world(scene_items={"brass-compass": "A small brass compass."})
        out = _resolve_invented_scene_ids(
            [_ref("living-brass-compass")], world, None
        )
        assert out[0].ref_entity_id == "brass-compass"

    def test_no_contained_referent_keeps_invented_id(self):
        world = _world(scene_items={"brass-compass": "A compass."})
        out = _resolve_invented_scene_ids([_ref("corvins-hallway")], world, None)
        assert out[0].ref_entity_id == "corvins-hallway"

    def test_ambiguous_containment_abstains(self):
        world = _world(scene_items={
            "drilled silver coin": "A coin.",
            "ragpicker token": "A token.",
        })
        out = _resolve_invented_scene_ids(
            [_update("drilled-silver-coin-ragpicker-token")], world, None
        )
        assert out[0].update_entity_id == "drilled-silver-coin-ragpicker-token"

    def test_single_token_candidates_never_rescue(self):
        # A one-word referent ("door") contained in a compound id is far too
        # weak an identity claim.
        world = _world(scene_items={"door": "A door."})
        out = _resolve_invented_scene_ids(
            [_update("heavy-iron-door-of-the-vault")], world, None
        )
        assert out[0].update_entity_id == "heavy-iron-door-of-the-vault"

    def test_known_id_left_untouched(self):
        world = _world(scene_items={"brass-compass": "A compass."})
        out = _resolve_invented_scene_ids([_ref("brass-compass")], world, None)
        assert out[0].ref_entity_id == "brass-compass"

    def test_slug_dialect_of_known_id_left_untouched(self):
        # 'brass compass' vs 'brass-compass' resolve via the normal slug
        # equality path; the rescue must not touch them.
        world = _world(scene_items={"brass compass": "A compass."})
        out = _resolve_invented_scene_ids([_ref("brass-compass")], world, None)
        assert out[0].ref_entity_id == "brass-compass"

    def test_npc_name_containment_resolves(self):
        npc = NPCState(name="Sera Vellik")
        world = _world(npcs={npc.id: npc})
        out = _resolve_invented_scene_ids(
            [_update("sera-vellik-the-market-courier")], world, None
        )
        assert out[0].update_entity_id == npc.id

    def test_graph_node_containment_resolves(self):
        class _EntityType:
            value = "location"

        node = SimpleNamespace(
            node_id="cinder-row",
            name="Cinder Row",
            entity_type=_EntityType(),
        )
        graph = SimpleNamespace(_entities={"cinder-row": node})
        out = _resolve_invented_scene_ids(
            [_ref("cinder-row-rooftops")], _world(), graph
        )
        assert out[0].ref_entity_id == "cinder-row"

    def test_non_id_effects_pass_through(self):
        effect = ProposedEffect(
            effect_type=EffectType.CHANGE_LOCATION,
            location_name="Tallow Rows",
        )
        out = _resolve_invented_scene_ids([effect], _world(), None)
        assert out[0] is effect
