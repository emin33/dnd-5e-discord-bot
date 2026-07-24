"""Post-merge adversarial review of the id-resolution change: four seams.

The world-reference fallback (effects.py) made LOCATION nodes, KG items and
departed NPCs reachable by ``update_entity``. A 4-lens review confirmed four
independent defects in what that reachability then allowed:

1. A LOCATION target accepted NPC-only semantics, so DeltaBridge stamped
   disposition/status (and, for status 'dead', ``alive=false``) onto a place.
2. ``_resolve_invented_scene_ids`` rewrote an id without ever consulting
   ``ref_alias_used`` — the name the narrator actually put in prose — so a
   mis-bound ref permanently grafted a wrong alias onto the resolved entity.
3. A non-NPC update reported ``description_appended``/``items_added`` in its
   ``applied`` receipt with no writer behind any of them, and its dedup read
   ``getattr(None, "description")`` — always '' — so it was non-idempotent.
4. ``campaign_dead_npcs`` was accepted by ``_is_known_entity`` but resolved by
   neither executor helper: validate, then die at execution.
"""

from types import SimpleNamespace

import pytest

from dnd_bot.game.scene.registry import SceneEntityRegistry
from dnd_bot.game.world_state import NPCState, WorldState
from dnd_bot.game.world_store import WorldStateStore
from dnd_bot.llm.effects import (
    EffectExecutor,
    EffectType,
    EffectValidator,
    ProposedEffect,
)
from dnd_bot.llm.orchestrator import _resolve_invented_scene_ids


def _graph(nodes: dict):
    return SimpleNamespace(
        get_entity=lambda entity_id: nodes.get(entity_id),
        resolve_entity_reference=lambda _reference: None,
    )


def _node(node_id: str, name: str, kind: str):
    return SimpleNamespace(
        node_id=node_id,
        name=name,
        aliases=[],
        entity_type=SimpleNamespace(value=kind),
    )


def _old_mill_session():
    """The review's repro: current location also present as a KG LOCATION."""
    return SimpleNamespace(
        world_state=WorldState(campaign_id="camp", current_location="The Old Mill"),
        knowledge_graph=_graph(
            {"the-old-mill": _node("the-old-mill", "The Old Mill", "location")}
        ),
    )


def _registry() -> SceneEntityRegistry:
    return SceneEntityRegistry(campaign_id="camp", channel_id=0)


def _update(entity_id: str, **fields) -> ProposedEffect:
    return ProposedEffect(
        effect_type=EffectType.UPDATE_ENTITY,
        update_entity_id=entity_id,
        **fields,
    )


# ── Seam 1: a place is not a creature ────────────────────────────────────


@pytest.mark.asyncio
class TestLocationTargetRejectsNpcSemantics:
    async def test_status_on_location_is_rejected_by_executor(self):
        session = _old_mill_session()
        executor = EffectExecutor(scene_registry=_registry(), session=session)

        result = await executor.execute(_update("the-old-mill", update_status="dead"))

        # Pre-fix: success=True, and DeltaBridge then wrote status='dead' +
        # alive='false' onto the location node.
        assert result.success is False
        assert "location" in result.error

    async def test_disposition_on_location_is_rejected(self):
        session = _old_mill_session()
        executor = EffectExecutor(scene_registry=_registry(), session=session)

        result = await executor.execute(
            _update("the-old-mill", update_disposition="hostile")
        )

        assert result.success is False
        assert "disposition" in result.error

    async def test_inventory_on_location_is_rejected(self):
        session = _old_mill_session()
        executor = EffectExecutor(scene_registry=_registry(), session=session)

        result = await executor.execute(
            _update("the-old-mill", update_add_items=["a rusted key"])
        )

        assert result.success is False
        assert "add_items" in result.error

    async def test_description_on_location_still_applies(self):
        session = _old_mill_session()
        executor = EffectExecutor(scene_registry=_registry(), session=session)

        result = await executor.execute(
            _update("the-old-mill", update_description_addition="the wheel has stopped")
        )

        # A place legitimately gains description; only creature semantics go.
        assert result.success is True
        assert result.details["world_reference_type"] == "location"


def test_status_on_location_is_rejected_by_validator():
    session = _old_mill_session()
    validator = EffectValidator(scene_registry=_registry(), session=session)

    result = validator.validate(_update("the-old-mill", update_status="dead"))

    # Validator and executor share one resolver, so the rejection lands
    # before execution rather than as a validate-then-die.
    assert result.valid is False
    assert "location" in result.rejection_reason


@pytest.mark.asyncio
class TestNonNpcTargetRejectsPersonSemantics:
    async def test_disposition_on_graph_item_is_rejected(self):
        session = SimpleNamespace(
            world_state=WorldState(campaign_id="camp"),
            knowledge_graph=_graph(
                {"brass-compass": _node("brass-compass", "brass compass", "item")}
            ),
        )
        executor = EffectExecutor(scene_registry=_registry(), session=session)

        result = await executor.execute(
            _update("brass-compass", update_disposition="hostile")
        )

        assert result.success is False
        assert "not an NPC" in result.error

    async def test_status_on_graph_item_is_still_accepted(self):
        # Soak 20260723_230351 turn 23 — the reliability-gate fix this
        # contract must not undo. An item's condition is not person semantics.
        session = SimpleNamespace(
            world_state=WorldState(campaign_id="camp"),
            knowledge_graph=_graph(
                {"orris-note": _node("orris-note", "Orris Vane's note", "item")}
            ),
        )
        executor = EffectExecutor(scene_registry=_registry(), session=session)

        result = await executor.execute(_update("orris-note", update_status="captured"))

        assert result.success is True
        assert result.details["applied"]["status"] == "captured"

    async def test_npc_target_keeps_every_field(self):
        world = WorldState(campaign_id="camp")
        npc = NPCState(name="Mara Venn", location="")
        world.npcs[npc.id] = npc
        executor = EffectExecutor(
            scene_registry=_registry(),
            session=SimpleNamespace(world_state=world),
        )

        result = await executor.execute(_update(
            npc.id,
            update_disposition="hostile",
            update_status="wounded",
            update_add_items=["a bent nail"],
        ))

        assert result.success is True
        assert result.details["applied"]["disposition"] == "hostile"
        assert result.details["applied"]["items_added"] == ["a bent nail"]


# ── Seam 2: a rewrite must agree with the alias used in prose ────────────


class TestInventedIdRespectsAlias:
    def test_rewrite_abstains_when_alias_contradicts_the_winner(self):
        world = WorldState(
            campaign_id="camp",
            npcs={"npc-1": NPCState(id="npc-1", name="Gideon Hask")},
        )
        effect = ProposedEffect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id="gideon",
            ref_alias_used="Mira",
        )

        out = _resolve_invented_scene_ids([effect], world, None)

        # Pre-fix: bound to 'npc-1' and 'Mira' was then appended to Gideon
        # Hask's alias list for the rest of the campaign.
        assert out[0].ref_entity_id == "gideon"

    def test_rewrite_survives_a_consistent_alias(self):
        world = WorldState(
            campaign_id="camp",
            npcs={"npc-1": NPCState(id="npc-1", name="Gideon Hask")},
        )
        effect = ProposedEffect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id="gideon",
            ref_alias_used="Gideon Hask",
        )

        out = _resolve_invented_scene_ids([effect], world, None)

        assert out[0].ref_entity_id == "npc-1"

    def test_embellished_item_rewrite_survives_a_partial_alias(self):
        world = WorldState(
            campaign_id="camp",
            scene_items={"low wooden door": "A low wooden door."},
        )
        effect = ProposedEffect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id="low-wooden-door-with-token-indentation",
            ref_alias_used="the low door",
        )

        out = _resolve_invented_scene_ids([effect], world, None)

        assert out[0].ref_entity_id == "low wooden door"

    def test_absent_alias_leaves_resolution_unchanged(self):
        world = WorldState(
            campaign_id="camp",
            scene_items={"low wooden door": "A low wooden door."},
        )
        effect = ProposedEffect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id="low-wooden-door-with-token-indentation",
        )

        out = _resolve_invented_scene_ids([effect], world, None)

        assert out[0].ref_entity_id == "low wooden door"


# ── Seam 3: `applied` is a receipt, not a wish list ──────────────────────


@pytest.mark.asyncio
class TestNonNpcUpdateReceiptHasAWriter:
    async def test_scene_item_description_is_written_and_is_idempotent(self):
        world = WorldState(
            campaign_id="camp",
            scene_items={"carved wooden door": "A dark oiled wooden door."},
        )
        store = WorldStateStore(world)
        executor = EffectExecutor(
            scene_registry=_registry(),
            session=SimpleNamespace(world_state=world),
        )
        effect = _update(
            "carved-wooden-door",
            update_description_addition="a compass-needle symbol glows faintly",
        )

        first = await executor.execute(effect)
        store.apply_effect(effect)

        assert first.details["applied"]["description_appended"]
        assert "compass-needle symbol" in world.scene_items["carved wooden door"]

        second = await executor.execute(effect)
        store.apply_effect(effect)

        # Pre-fix: `existing` was read off `None`, so the dedup never fired
        # and re-execution re-reported an append nothing had made.
        assert "description_appended" not in second.details["applied"]
        assert world.scene_items["carved wooden door"].count("compass-needle") == 1

    async def test_current_location_description_is_written(self):
        world = WorldState(
            campaign_id="camp",
            current_location="The Old Mill",
            location_description="The wheel turns slowly.",
        )
        store = WorldStateStore(world)
        executor = EffectExecutor(
            scene_registry=_registry(),
            session=SimpleNamespace(
                world_state=world,
                knowledge_graph=_graph(
                    {"the-old-mill": _node("the-old-mill", "The Old Mill", "location")}
                ),
            ),
        )
        effect = _update(
            "the-old-mill",
            update_description_addition="The wheel has stopped.",
        )

        result = await executor.execute(effect)
        store.apply_effect(effect)

        assert result.details["applied"]["description_appended"]
        assert "The wheel has stopped." in world.location_description

    async def test_graph_only_target_claims_nothing_it_cannot_write(self):
        session = SimpleNamespace(
            world_state=WorldState(campaign_id="camp"),
            knowledge_graph=_graph(
                {"brass-compass": _node("brass-compass", "brass compass", "item")}
            ),
        )
        executor = EffectExecutor(scene_registry=_registry(), session=session)

        result = await executor.execute(_update(
            "brass-compass",
            update_description_addition="its needle now points down",
            update_status="wounded",
        ))

        # Identity is still acknowledged and status still reaches DeltaBridge;
        # the description has no writer for a graph-only node, so the receipt
        # no longer claims it.
        assert result.success is True
        assert result.details["applied"]["status"] == "wounded"
        assert "description_appended" not in result.details["applied"]

    async def test_graph_only_npc_inventory_is_not_claimed(self):
        # Same hollow shape on the pre-existing graph-only-NPC path, brought
        # along deliberately: WorldStateStore resolves inventory strictly
        # through WorldState.npcs, which this identity is not in.
        npc_id = "d04bbdac-c09f-4c1e-855b-5f395546d986"
        session = SimpleNamespace(
            world_state=WorldState(campaign_id="camp"),
            knowledge_graph=_graph({npc_id: _node(npc_id, "Thessa", "npc")}),
        )
        executor = EffectExecutor(scene_registry=_registry(), session=session)

        result = await executor.execute(_update(
            npc_id,
            update_add_items=["a folded letter"],
            update_description_addition="stands before a small shrine",
        ))

        assert result.success is True
        assert "items_added" not in result.details["applied"]
        assert "description_appended" not in result.details["applied"]


# ── Seam 4: what the validator accepts, the executor must resolve ────────


def _dead_roster_session():
    dead = NPCState(id="npc-dead", name="Orin Vale", alive=False)
    return SimpleNamespace(
        world_state=WorldState(campaign_id="camp"),
        campaign_dead_npcs={"npc-dead": dead},
    )


def test_validator_accepts_a_dead_roster_npc():
    result = EffectValidator(
        scene_registry=_registry(), session=_dead_roster_session()
    ).validate(_update("npc-dead", update_importance=True))

    assert result.valid is True


def test_reachability_does_not_open_a_resurrection_path():
    result = EffectValidator(
        scene_registry=_registry(), session=_dead_roster_session()
    ).validate(_update("npc-dead", update_status="alive"))

    assert result.valid is False
    assert "resurrection" in result.rejection_reason


@pytest.mark.asyncio
class TestDeadCampaignNpcSymmetry:
    def _session(self):
        return _dead_roster_session()

    async def test_executor_now_resolves_what_the_validator_accepted(self):
        executor = EffectExecutor(
            scene_registry=_registry(), session=self._session()
        )

        result = await executor.execute(_update("npc-dead", update_importance=True))

        # Pre-fix: neither resolver consulted campaign_dead_npcs, so this
        # validated and then died with 'is not a known entity'. It resolves
        # as an NPC identity, so the world-reference probe is skipped and its
        # type tag stays None — same shape as any other off-scene NPC.
        assert result.success is True
        assert result.details["found_in_scene"] is False
        assert result.details["found_in_world"] is True

    async def test_dead_roster_npc_resolves_by_name_too(self):
        executor = EffectExecutor(
            scene_registry=_registry(), session=self._session()
        )

        result = await executor.execute(_update("Orin Vale", update_importance=True))

        assert result.success is True
