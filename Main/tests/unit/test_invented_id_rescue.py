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

    def test_partial_npc_token_resolves_to_unique_owner(self):
        # Soak 20260723_230351 turn 77: ref_entity 'gideon' for the tracked
        # NPC 'Gideon Hask' (canonical UUID id in the graph).
        class _EntityType:
            value = "npc"

        node = SimpleNamespace(
            node_id="d6434cfa-0ee2-4ebe-bd6c-90f7ad6bdaaf",
            name="Gideon Hask",
            entity_type=_EntityType(),
        )
        graph = SimpleNamespace(_entities={node.node_id: node})
        out = _resolve_invented_scene_ids([_ref("gideon")], _world(), graph)
        assert out[0].ref_entity_id == "d6434cfa-0ee2-4ebe-bd6c-90f7ad6bdaaf"

    def test_partial_npc_token_resolves_via_world_state(self):
        npc = NPCState(name="Gideon Hask")
        world = _world(npcs={npc.id: npc})
        out = _resolve_invented_scene_ids([_ref("gideon")], world, None)
        assert out[0].ref_entity_id == npc.id

    def test_partial_token_shared_by_two_entities_abstains(self):
        # 'sera' when two Seras are tracked must stay rejected.
        first = NPCState(name="Sera Brightwater")
        second = NPCState(name="Sera Duskwalker")
        world = _world(npcs={first.id: first, second.id: second})
        out = _resolve_invented_scene_ids([_ref("sera")], world, None)
        assert out[0].ref_entity_id == "sera"

    def test_partial_token_matching_item_and_npc_abstains(self):
        # A subset shared across entity kinds is just as ambiguous.
        npc = NPCState(name="Brass Tomas")
        world = _world(
            npcs={npc.id: npc},
            scene_items={"brass compass": "A compass."},
        )
        out = _resolve_invented_scene_ids([_update("brass")], world, None)
        assert out[0].update_entity_id == "brass"

    def test_entity_absent_from_every_store_stays_rejected(self):
        # Soak 20260723_230351 turn 1: 'masked_courier' — the courier only
        # ever existed in prose, never as an NPC/item/graph node. Nothing
        # to resolve onto; the fail-closed rejection is correct.
        world = _world(scene_items={
            "living brass compass": "A compass.",
            "brass compass": "A compass.",
        })
        out = _resolve_invented_scene_ids([_update("masked_courier")], world, None)
        assert out[0].update_entity_id == "masked_courier"

    def test_bare_common_noun_never_truncate_resolves(self):
        # Adversarial review: the subset direction shipped with no minimum
        # bar, so 'door' silently rewrote onto the one item containing it.
        # A common object noun is not an identity claim.
        world = _world(scene_items={"carved wooden door": "A door."})
        out = _resolve_invented_scene_ids([_update("door")], world, None)
        assert out[0].update_entity_id == "door"

    def test_bare_generic_role_noun_never_truncate_resolves(self):
        # 'man' inside 'Marcus Guard' — the identity layer refuses to bind
        # generic role nouns, and this path must agree with it.
        npc = NPCState(name="Marcus Guard")
        world = _world(npcs={npc.id: npc})
        out = _resolve_invented_scene_ids([_ref("man")], world, None)
        assert out[0].ref_entity_id == "man"

    def test_short_bare_token_never_truncate_resolves(self):
        npc = NPCState(name="Kae Windrunner")
        world = _world(npcs={npc.id: npc})
        out = _resolve_invented_scene_ids([_ref("kae")], world, None)
        assert out[0].ref_entity_id == "kae"

    def test_bare_token_matching_only_an_item_does_not_resolve(self):
        # The subset direction is for truncated PROPER NAMES; an item whose
        # label merely contains the token is not a proper-named referent.
        world = _world(scene_items={"orris vane hidden note": "A note."})
        out = _resolve_invented_scene_ids([_update("hidden")], world, None)
        assert out[0].update_entity_id == "hidden"

    def test_containment_match_is_not_vetoed_by_a_superset_candidate(self):
        # Pooling both directions into one `matched` set let an unrelated
        # superset count as a second match and defeat a good containment
        # resolution. Containment is authoritative and tried first.
        world = _world(scene_items={
            "low wooden door": "A low wooden door.",
            "low wooden door with token indentation and brass fittings": "x",
        })
        out = _resolve_invented_scene_ids(
            [_update("low-wooden-door-with-token-indentation")], world, None
        )
        assert out[0].update_entity_id == "low wooden door"

    def test_non_id_effects_pass_through(self):
        effect = ProposedEffect(
            effect_type=EffectType.CHANGE_LOCATION,
            location_name="Tallow Rows",
        )
        out = _resolve_invented_scene_ids([effect], _world(), None)
        assert out[0] is effect
