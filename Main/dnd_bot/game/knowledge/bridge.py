"""DeltaBridge — converts StateDelta into knowledge graph operations."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import TYPE_CHECKING

import structlog

from .models import (
    AddEdge,
    AddNode,
    DEFAULT_WEIGHTS,
    Entity,
    EntityType,
    GraphOperation,
    Relationship,
    RelationType,
    RemoveEdge,
    UpdateNode,
    slugify,
)

if TYPE_CHECKING:
    from ..world_state import StateDelta, WorldState
    from ...llm.effects import ProposedEffect

logger = structlog.get_logger()


@dataclass(frozen=True)
class NamePromotion:
    """Suggestion to promote an unnamed entity to a proper name."""

    node_id: str
    new_name: str


def _is_proper_name(name: str) -> bool:
    """Check if *name* looks like a proper name rather than a descriptor."""
    lower = name.lower()
    if lower.startswith(("the ", "a ", "an ")):
        return False
    # All lowercase ⇒ descriptive ("old merchant", "cloaked figure")
    if name == lower:
        return False
    return any(word[0].isupper() for word in name.split() if word)


class DeltaBridge:
    """Converts a StateDelta into knowledge graph operations.

    Called after WorldState.apply_delta() succeeds, so the WorldState
    already reflects the new state. The bridge reads both the delta
    (what changed) and the world state (current snapshot) to produce
    graph mutations.
    """

    def __init__(self, campaign_id: str):
        self._campaign_id = campaign_id

    def convert(
        self,
        delta: "StateDelta",
        world_state: "WorldState",
        existing_node_ids: set[str] | None = None,
        previous_location: str = "",
    ) -> list[GraphOperation]:
        """Convert a StateDelta into graph operations.

        Args:
            delta: The state changes extracted from narrator prose.
            world_state: The current world state (already has delta applied).
            existing_node_ids: Set of node_ids already in the graph.
                Used to decide whether to create placeholder nodes.
            previous_location: The location before delta was applied.
                Needed to create CONNECTED_TO edges on location change.
        """
        ops: list[GraphOperation] = []
        known = existing_node_ids or set()
        now = datetime.utcnow()

        # 1. Location change → ensure location node + connected_to from previous
        if delta.location_change:
            ops.extend(self._handle_location_change(delta, world_state, known, now, previous_location))

        # 2. New connections → bidirectional connected_to edges
        if delta.new_connections:
            ops.extend(self._handle_new_connections(delta, world_state, known, now))

        # 3. New NPCs → node + located_at edge
        for npc in delta.new_npcs:
            ops.extend(self._handle_new_npc(npc, world_state, known, now))

        # 4. NPC updates → update node + relocate edges if location changed
        for update in delta.npc_updates:
            ops.extend(self._handle_npc_update(update, world_state, known, now))

        # 5. Removed NPCs → clear location property, drop located_at edge.
        # ``removed_npcs`` carries names (or ids — backwards compat); the
        # handler resolves to a UUID via world_state lookup.
        for npc_identifier in delta.removed_npcs:
            ops.extend(self._handle_removed_npc(npc_identifier, world_state))

        # 6. New quests → quest node + edges to giver and location
        for quest in delta.new_quests:
            ops.extend(self._handle_new_quest(quest, world_state, known, now))

        # 7. Quest updates → update node properties
        for update in delta.quest_updates:
            quest_id = slugify(update.name)
            changed: dict[str, str] = {}
            if update.status is not None:
                changed["status"] = update.status
            if update.location is not None:
                changed["location"] = update.location
            if update.objectives is not None:
                changed["objectives"] = ", ".join(update.objectives)
            if changed:
                ops.append(UpdateNode(node_id=quest_id, properties=changed))

        if ops:
            logger.debug("delta_bridge_converted", op_count=len(ops))

        return ops

    # ------------------------------------------------------------------
    # Conversion handlers
    # ------------------------------------------------------------------

    def _handle_location_change(
        self,
        delta: "StateDelta",
        world_state: "WorldState",
        known: set[str],
        now: datetime,
        previous_location: str = "",
    ) -> list[GraphOperation]:
        ops: list[GraphOperation] = []
        loc_id = slugify(delta.location_change)

        # Create location node if not already known
        if loc_id not in known:
            ops.append(AddNode(entity=Entity(
                node_id=loc_id,
                entity_type=EntityType.LOCATION,
                name=delta.location_change,
                campaign_id=self._campaign_id,
                properties={
                    "description": delta.location_description or "",
                },
                created_at=now,
                updated_at=now,
            )))
            known.add(loc_id)
        elif delta.location_description:
            # Update description if we have a better one
            ops.append(UpdateNode(
                node_id=loc_id,
                properties={"description": delta.location_description},
            ))

        # Create CONNECTED_TO edge from previous location (bidirectional)
        if previous_location:
            prev_id = slugify(previous_location)
            if prev_id and prev_id != loc_id:
                # Ensure previous location node exists
                if prev_id not in known:
                    ops.append(AddNode(entity=Entity(
                        node_id=prev_id,
                        entity_type=EntityType.LOCATION,
                        name=previous_location,
                        campaign_id=self._campaign_id,
                        properties={"placeholder": "true"},
                        created_at=now,
                        updated_at=now,
                    )))
                    known.add(prev_id)

                weight = DEFAULT_WEIGHTS[RelationType.CONNECTED_TO]
                ops.append(AddEdge(relationship=Relationship(
                    source_id=prev_id,
                    target_id=loc_id,
                    relation_type=RelationType.CONNECTED_TO,
                    weight=weight,
                    campaign_id=self._campaign_id,
                    created_at=now,
                )))
                ops.append(AddEdge(relationship=Relationship(
                    source_id=loc_id,
                    target_id=prev_id,
                    relation_type=RelationType.CONNECTED_TO,
                    weight=weight,
                    campaign_id=self._campaign_id,
                    created_at=now,
                )))

        return ops

    def _handle_new_connections(
        self,
        delta: "StateDelta",
        world_state: "WorldState",
        known: set[str],
        now: datetime,
    ) -> list[GraphOperation]:
        ops: list[GraphOperation] = []
        current_loc_id = slugify(world_state.current_location) if world_state.current_location else None

        for conn_name in delta.new_connections:
            conn_id = slugify(conn_name)
            if not conn_id:
                continue

            # Ensure the connected location exists as a node
            if conn_id not in known:
                ops.append(AddNode(entity=Entity(
                    node_id=conn_id,
                    entity_type=EntityType.LOCATION,
                    name=conn_name,
                    campaign_id=self._campaign_id,
                    properties={"placeholder": "true"},
                    created_at=now,
                    updated_at=now,
                )))
                known.add(conn_id)

            # Bidirectional connected_to edges
            if current_loc_id and current_loc_id != conn_id:
                weight = DEFAULT_WEIGHTS[RelationType.CONNECTED_TO]
                ops.append(AddEdge(relationship=Relationship(
                    source_id=current_loc_id,
                    target_id=conn_id,
                    relation_type=RelationType.CONNECTED_TO,
                    weight=weight,
                    campaign_id=self._campaign_id,
                    created_at=now,
                )))
                ops.append(AddEdge(relationship=Relationship(
                    source_id=conn_id,
                    target_id=current_loc_id,
                    relation_type=RelationType.CONNECTED_TO,
                    weight=weight,
                    campaign_id=self._campaign_id,
                    created_at=now,
                )))

        return ops

    def _handle_new_npc(self, npc, world_state: "WorldState", known: set[str], now: datetime) -> list[GraphOperation]:
        """Create a KG node for a new NPC, using the NPCState's stable
        UUID as the node_id. This is the cross-layer identity anchor —
        WorldState NPCState.id == KG entity.node_id, so narrator tool
        calls (which receive the id from the roster) resolve correctly
        in either layer.
        """
        ops: list[GraphOperation] = []
        npc_id = npc.id  # Use the NPCState's stable UUID (was: slugify(npc.name))
        if not npc_id:
            return ops

        # Build properties from NPCState fields
        props: dict[str, str] = {}
        if npc.description:
            props["description"] = npc.description
        if npc.disposition:
            props["disposition"] = npc.disposition
        props["alive"] = str(npc.alive).lower()
        if npc.notes:
            props["notes"] = npc.notes
        if npc.important:
            props["important"] = "true"

        # Detect unnamed NPCs: descriptive phrases used as names
        name_lower = npc.name.lower()
        is_named = not (
            name_lower.startswith(("the ", "a ", "an "))
            or name_lower == npc.name  # all lowercase = likely descriptive
        )
        if not is_named:
            props["named"] = "false"

        # Determine location
        npc_location = npc.location or world_state.current_location
        if npc_location:
            props["location"] = npc_location

        ops.append(AddNode(entity=Entity(
            node_id=npc_id,
            entity_type=EntityType.NPC,
            name=npc.name,
            campaign_id=self._campaign_id,
            properties=props,
            created_at=now,
            updated_at=now,
        )))
        known.add(npc_id)

        # Add located_at edge if we have a location
        if npc_location:
            loc_id = slugify(npc_location)
            # Ensure location node exists
            if loc_id not in known:
                ops.append(AddNode(entity=Entity(
                    node_id=loc_id,
                    entity_type=EntityType.LOCATION,
                    name=npc_location,
                    campaign_id=self._campaign_id,
                    properties={"placeholder": "true"},
                    created_at=now,
                    updated_at=now,
                )))
                known.add(loc_id)

            ops.append(AddEdge(relationship=Relationship(
                source_id=npc_id,
                target_id=loc_id,
                relation_type=RelationType.LOCATED_AT,
                weight=DEFAULT_WEIGHTS[RelationType.LOCATED_AT],
                campaign_id=self._campaign_id,
                created_at=now,
            )))

        return ops

    def _handle_npc_update(self, update, world_state: "WorldState", known: set[str], now: datetime) -> list[GraphOperation]:
        """Convert an NPCUpdate to a KG UpdateNode op. Resolution chain:
        1. ``update.id`` if provided (preferred — narrator passed an id from roster)
        2. WorldState lookup by name/alias (returns NPCState.id)
        3. ``slugify(update.name)`` fallback for legacy callers / older
           KG nodes that were created with slug-based ids
        """
        ops: list[GraphOperation] = []
        npc_id = (update.id or "").strip()
        if not npc_id and update.name and world_state is not None:
            existing = world_state._find_npc(update.name)
            if existing:
                npc_id = existing.id
        if not npc_id and update.name:
            # Legacy fallback: slug-based id for older KG nodes
            npc_id = slugify(update.name)
        if not npc_id:
            return ops

        # Collect changed properties
        changed: dict[str, str] = {}
        if update.disposition is not None:
            changed["disposition"] = update.disposition
        if update.description is not None:
            changed["description"] = update.description
        if update.alive is not None:
            changed["alive"] = str(update.alive).lower()
        if update.notes is not None:
            changed["notes"] = update.notes
        if update.important is not None:
            changed["important"] = str(update.important).lower()
        if update.location is not None:
            changed["location"] = update.location

        if changed:
            ops.append(UpdateNode(node_id=npc_id, properties=changed))

        # If location changed, update the located_at edge
        if update.location is not None:
            # Remove old located_at edge
            ops.append(RemoveEdge(
                source_id=npc_id,
                target_id="",  # target doesn't matter — delete_edges_by_source handles it
                relation_type=RelationType.LOCATED_AT,
            ))

            if update.location:  # Non-empty = moved somewhere
                loc_id = slugify(update.location)
                if loc_id not in known:
                    ops.append(AddNode(entity=Entity(
                        node_id=loc_id,
                        entity_type=EntityType.LOCATION,
                        name=update.location,
                        campaign_id=self._campaign_id,
                        properties={"placeholder": "true"},
                        created_at=now,
                        updated_at=now,
                    )))
                    known.add(loc_id)

                ops.append(AddEdge(relationship=Relationship(
                    source_id=npc_id,
                    target_id=loc_id,
                    relation_type=RelationType.LOCATED_AT,
                    weight=DEFAULT_WEIGHTS[RelationType.LOCATED_AT],
                    campaign_id=self._campaign_id,
                    created_at=now,
                )))

        return ops

    def _handle_removed_npc(self, npc_identifier: str, world_state: "WorldState" = None) -> list[GraphOperation]:
        """NPC left the scene — clear location, keep the node.

        ``npc_identifier`` is preferentially an NPCState.id (UUID); falls
        back to a name lookup against world_state when available, then
        a slugify fallback for legacy callers.
        """
        ops: list[GraphOperation] = []
        npc_id = (npc_identifier or "").strip()
        if not npc_id:
            return ops
        # If it's a name (not a UUID), resolve via world_state to find the id
        if world_state is not None and npc_id not in world_state.npcs:
            existing = world_state._find_npc(npc_id)
            if existing:
                npc_id = existing.id
            else:
                # Final fallback for legacy callers / tests that pass a slug
                npc_id = slugify(npc_identifier)
                if not npc_id:
                    return ops

        ops.append(UpdateNode(node_id=npc_id, properties={"location": ""}))
        ops.append(RemoveEdge(
            source_id=npc_id,
            target_id="",
            relation_type=RelationType.LOCATED_AT,
        ))

        return ops

    def _handle_new_quest(self, quest, world_state: "WorldState", known: set[str], now: datetime) -> list[GraphOperation]:
        """Create quest node + edges to giver NPC and objective location."""
        ops: list[GraphOperation] = []
        quest_id = slugify(quest.name)
        if not quest_id:
            return ops

        props: dict[str, str] = {"status": quest.status}
        if quest.objectives:
            props["objectives"] = ", ".join(quest.objectives)
        if quest.giver:
            props["giver"] = quest.giver
        if quest.location:
            props["location"] = quest.location

        ops.append(AddNode(entity=Entity(
            node_id=quest_id,
            entity_type=EntityType.QUEST,
            name=quest.name,
            campaign_id=self._campaign_id,
            properties=props,
            created_at=now,
            updated_at=now,
        )))
        known.add(quest_id)

        # Edge: giver NPC → quest (QUEST_GIVER, highest priority)
        if quest.giver:
            giver_id = slugify(quest.giver)
            if giver_id and giver_id in known:
                ops.append(AddEdge(relationship=Relationship(
                    source_id=giver_id,
                    target_id=quest_id,
                    relation_type=RelationType.QUEST_GIVER,
                    weight=DEFAULT_WEIGHTS[RelationType.QUEST_GIVER],
                    campaign_id=self._campaign_id,
                    created_at=now,
                )))

        # Edge: quest → objective location (OBJECTIVE_AT)
        if quest.location:
            loc_id = slugify(quest.location)
            if loc_id not in known:
                ops.append(AddNode(entity=Entity(
                    node_id=loc_id,
                    entity_type=EntityType.LOCATION,
                    name=quest.location,
                    campaign_id=self._campaign_id,
                    properties={"placeholder": "true"},
                    created_at=now,
                    updated_at=now,
                )))
                known.add(loc_id)

            ops.append(AddEdge(relationship=Relationship(
                source_id=quest_id,
                target_id=loc_id,
                relation_type=RelationType.OBJECTIVE_AT,
                weight=DEFAULT_WEIGHTS[RelationType.OBJECTIVE_AT],
                campaign_id=self._campaign_id,
                created_at=now,
            )))

        return ops

    # ── Tool-effect entry point ──────────────────────────────────────

    def convert_effects(
        self,
        effects: list["ProposedEffect"],
        world_state: "WorldState",
        existing_node_ids: set[str] | None = None,
    ) -> tuple[list[GraphOperation], list[NamePromotion]]:
        """Convert executed narrator tool effects into KG operations.

        Companion to :meth:`convert` which handles StateDelta.  This handles
        ``ProposedEffect`` objects produced by narrator tool calls
        (add_npc, spawn_object, ref_entity, remove_entity).

        Returns ``(graph_ops, name_promotions)``.
        """
        from ...llm.effects import EffectType

        ops: list[GraphOperation] = []
        promotions: list[NamePromotion] = []
        known = existing_node_ids or set()
        now = datetime.utcnow()

        for effect in effects:
            etype = effect.effect_type
            if etype == EffectType.ADD_NPC:
                ops.extend(self._effect_add_npc(effect, world_state, known, now))
            elif etype == EffectType.SPAWN_OBJECT:
                ops.extend(self._effect_spawn_object(effect, world_state, known, now))
            elif etype == EffectType.REF_ENTITY:
                promo = self._effect_ref_entity(effect, known)
                if promo:
                    promotions.append(promo)
            elif etype == EffectType.REMOVE_ENTITY:
                ops.extend(self._effect_remove_entity(effect, world_state))

        if ops or promotions:
            logger.debug(
                "effect_bridge_converted",
                op_count=len(ops),
                promotion_count=len(promotions),
            )
        return ops, promotions

    # ── Per-effect helpers ───────────────────────────────────────────

    def _effect_add_npc(
        self,
        effect: "ProposedEffect",
        world_state: "WorldState",
        known: set[str],
        now: datetime,
    ) -> list[GraphOperation]:
        """ADD_NPC → AddNode(NPC) + LOCATED_AT edge."""
        ops: list[GraphOperation] = []
        npc_name = effect.npc_name or "Unknown"
        # ROOT-2 (DF-3): anchor on the WorldState NPCState UUID — the same
        # cross-layer identity key the delta path (_handle_new_npc) uses — so
        # the KG node, ChromaDB vector, world_state, and matcher all agree.
        # The effect-sync mints this NPCState (orchestrator:3117) BEFORE the KG
        # bridge runs, so it's available here. Slug only as a last resort.
        existing_npc = world_state._find_npc(npc_name)
        npc_id = (existing_npc.id if existing_npc else None) or slugify(npc_name)
        if not npc_id:
            return ops

        props: dict[str, str] = {}
        if effect.npc_description:
            props["description"] = effect.npc_description
        if effect.npc_disposition:
            props["disposition"] = effect.npc_disposition

        # Detect unnamed NPCs (same heuristic as _handle_new_npc)
        name_lower = npc_name.lower()
        is_named = not (
            name_lower.startswith(("the ", "a ", "an "))
            or name_lower == npc_name  # all lowercase ⇒ descriptive
        )
        if not is_named:
            props["named"] = "false"

        npc_location = world_state.current_location
        if npc_location:
            props["location"] = npc_location

        ops.append(AddNode(entity=Entity(
            node_id=npc_id,
            entity_type=EntityType.NPC,
            name=npc_name,
            campaign_id=self._campaign_id,
            properties=props,
            created_at=now,
            updated_at=now,
        )))
        known.add(npc_id)

        # LOCATED_AT edge
        if npc_location:
            loc_id = slugify(npc_location)
            if loc_id not in known:
                ops.append(AddNode(entity=Entity(
                    node_id=loc_id,
                    entity_type=EntityType.LOCATION,
                    name=npc_location,
                    campaign_id=self._campaign_id,
                    properties={"placeholder": "true"},
                    created_at=now,
                    updated_at=now,
                )))
                known.add(loc_id)

            ops.append(AddEdge(relationship=Relationship(
                source_id=npc_id,
                target_id=loc_id,
                relation_type=RelationType.LOCATED_AT,
                weight=DEFAULT_WEIGHTS[RelationType.LOCATED_AT],
                campaign_id=self._campaign_id,
                created_at=now,
            )))

        return ops

    def _effect_spawn_object(
        self,
        effect: "ProposedEffect",
        world_state: "WorldState",
        known: set[str],
        now: datetime,
    ) -> list[GraphOperation]:
        """SPAWN_OBJECT → AddNode(ITEM) + LOCATED_AT edge."""
        ops: list[GraphOperation] = []
        obj_name = effect.object_name or "unknown_item"
        obj_id = slugify(obj_name)
        if not obj_id:
            return ops

        props: dict[str, str] = {}
        if effect.object_description:
            props["description"] = effect.object_description

        location = world_state.current_location
        if location:
            props["location"] = location

        ops.append(AddNode(entity=Entity(
            node_id=obj_id,
            entity_type=EntityType.ITEM,
            name=obj_name,
            campaign_id=self._campaign_id,
            properties=props,
            created_at=now,
            updated_at=now,
        )))
        known.add(obj_id)

        if location:
            loc_id = slugify(location)
            if loc_id not in known:
                ops.append(AddNode(entity=Entity(
                    node_id=loc_id,
                    entity_type=EntityType.LOCATION,
                    name=location,
                    campaign_id=self._campaign_id,
                    properties={"placeholder": "true"},
                    created_at=now,
                    updated_at=now,
                )))
                known.add(loc_id)

            ops.append(AddEdge(relationship=Relationship(
                source_id=obj_id,
                target_id=loc_id,
                relation_type=RelationType.LOCATED_AT,
                weight=DEFAULT_WEIGHTS[RelationType.LOCATED_AT],
                campaign_id=self._campaign_id,
                created_at=now,
            )))

        return ops

    def _effect_ref_entity(
        self,
        effect: "ProposedEffect",
        known: set[str],
    ) -> NamePromotion | None:
        """REF_ENTITY → name promotion suggestion (if alias is a proper name)."""
        entity_id = effect.ref_entity_id
        alias = effect.ref_alias_used
        if not entity_id or not alias:
            return None

        node_id = slugify(entity_id)
        if not node_id or alias.lower() == entity_id.lower():
            return None

        if _is_proper_name(alias):
            return NamePromotion(node_id=node_id, new_name=alias)
        return None

    def _effect_remove_entity(
        self,
        effect: "ProposedEffect",
        world_state: "WorldState" = None,
    ) -> list[GraphOperation]:
        """REMOVE_ENTITY → clear location, remove LOCATED_AT edge (keep node).

        NPC/creature nodes are anchored on the WorldState NPCState UUID
        (see ``_effect_add_npc``), so the target — a UUID, a name, or the
        roster's ``[id: slug]`` dialect — must resolve UUID-first, mirroring
        the delta path's ``_handle_removed_npc`` (final review): a raw
        ``slugify(target)`` produced an id no NPC node has, so the
        UpdateNode was rejected and LOCATED_AT edges survived removal.
        The slug fallback still serves non-NPC / legacy slug-anchored nodes.
        """
        ops: list[GraphOperation] = []
        target = (effect.target or "").strip()
        if not target:
            return ops

        node_id = target
        if world_state is not None and node_id not in world_state.npcs:
            existing = world_state._find_npc(node_id)
            if existing is None:
                # Roster-slug dialect ('old-bram' for 'Old Bram'):
                # _find_npc matches exact names/aliases only, so bridge
                # via slug equality.
                target_slug = slugify(node_id)
                if target_slug:
                    existing = next(
                        (npc for npc in world_state.npcs.values()
                         if slugify(npc.name or "") == target_slug),
                        None,
                    )
            if existing is not None:
                node_id = existing.id
            else:
                node_id = slugify(node_id)
        elif world_state is None:
            node_id = slugify(node_id)

        if node_id:
            ops.append(UpdateNode(node_id=node_id, properties={"location": ""}))
            ops.append(RemoveEdge(
                source_id=node_id,
                target_id="",
                relation_type=RelationType.LOCATED_AT,
            ))
        return ops
