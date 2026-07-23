"""Provider-neutral sourcebook authoring contract.

This schema is deliberately separate from the compact live ``WorldState``.
An authored sourcebook is immutable canonical input; a future compiler will
validate it and project the relevant pieces into SQLite, the knowledge graph,
and the vector index.  The vector index is never the system of record.
"""

from __future__ import annotations

from collections import Counter
from enum import Enum
from typing import Annotated, Literal

from pydantic import BaseModel, Field, model_validator

StableId = Annotated[str, Field(pattern=r"^[a-z][a-z0-9_.-]{2,79}$")]


def _reject_cycles(edges: dict[str, set[str]], label: str) -> None:
    """Reject directed cycles in containment, dependency, and evidence DAGs."""
    visiting: set[str] = set()
    visited: set[str] = set()

    def visit(node_id: str) -> None:
        if node_id in visiting:
            raise ValueError(f"{label} contains a cycle at {node_id!r}")
        if node_id in visited:
            return
        visiting.add(node_id)
        for target_id in edges.get(node_id, set()):
            visit(target_id)
        visiting.remove(node_id)
        visited.add(node_id)

    for candidate_id in edges:
        visit(candidate_id)


class CanonStatus(str, Enum):
    CANON = "canon"
    PROVISIONAL = "provisional"
    LEGEND = "legend"
    DISPUTED = "disputed"
    FALSE = "false"
    UNKNOWN = "unknown"


class Visibility(str, Enum):
    PUBLIC = "public"
    PLAYER_KNOWN = "player_known"
    DISCOVERABLE = "discoverable"
    DM_ONLY = "dm_only"


class CharacterStatus(str, Enum):
    ALIVE = "alive"
    DEAD = "dead"
    MISSING = "missing"
    UNDEAD = "undead"
    UNKNOWN = "unknown"


class LocationKind(str, Enum):
    WORLD = "world"
    REGION = "region"
    SETTLEMENT = "settlement"
    DISTRICT = "district"
    SITE = "site"
    BUILDING = "building"
    ROOM = "room"
    WILDERNESS = "wilderness"
    PLANE = "plane"


class LoreDomainKind(str, Enum):
    CULTURE = "culture"
    RELIGION = "religion"
    LANGUAGE = "language"
    LAW = "law"
    ECONOMY = "economy"
    MAGIC = "magic"
    HISTORY = "history"
    CUSTOM = "custom"


class RelationshipKind(str, Enum):
    PARENT_OF = "parent_of"
    SIBLING_OF = "sibling_of"
    SPOUSE_OF = "spouse_of"
    LOVES = "loves"
    FRIEND_OF = "friend_of"
    RIVAL_OF = "rival_of"
    HOSTILE_TO = "hostile_to"
    ALLIED_WITH = "allied_with"
    KNOWS = "knows"
    OWES = "owes"
    FEARS = "fears"
    MEMBER_OF = "member_of"
    LEADS = "leads"
    SERVES = "serves"
    CONTROLS = "controls"
    LOCATED_AT = "located_at"
    CONNECTED_TO = "connected_to"
    OWNS = "owns"
    CARRIES = "carries"
    CREATED = "created"
    INVOLVED_IN = "involved_in"
    QUEST_GIVER = "quest_giver"
    KILLED_BY = "killed_by"
    CUSTOM = "custom"


class Provenance(BaseModel):
    source_type: Literal["human", "model", "import", "play"] = "human"
    source_label: str = ""
    authoring_model: str | None = None
    revision: int = Field(default=1, ge=1)
    parent_record_ids: list[StableId] = Field(default_factory=list)


class StatBlock(BaseModel):
    ruleset: str = "dnd5e"
    source_ref: str | None = None
    challenge_rating: str | None = None
    level: int | None = Field(default=None, ge=0, le=30)
    armor_class: int | None = Field(default=None, ge=0)
    hit_points: int | None = Field(default=None, ge=0)
    speed: str | None = None
    abilities: dict[str, int] = Field(default_factory=dict)
    saves: dict[str, int] = Field(default_factory=dict)
    skills: dict[str, int] = Field(default_factory=dict)
    senses: list[str] = Field(default_factory=list)
    languages: list[str] = Field(default_factory=list)
    traits: list[str] = Field(default_factory=list)
    actions: list[str] = Field(default_factory=list)
    reactions: list[str] = Field(default_factory=list)
    legendary_actions: list[str] = Field(default_factory=list)
    overrides: dict[str, str] = Field(default_factory=dict)


class BehaviorProfile(BaseModel):
    voice: str = ""
    mannerisms: list[str] = Field(default_factory=list)
    values: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    fears: list[str] = Field(default_factory=list)
    temptations: list[str] = Field(default_factory=list)
    boundaries: list[str] = Field(default_factory=list)
    decision_rules: list[str] = Field(default_factory=list)
    tells: list[str] = Field(default_factory=list)


class InventoryEntry(BaseModel):
    item_id: StableId
    quantity: int = Field(default=1, ge=1)
    equipped: bool = False
    hidden: bool = False
    notes: str = ""


class NamedEntity(BaseModel):
    id: StableId
    name: str = Field(min_length=1)
    aliases: list[str] = Field(default_factory=list)
    summary: str = ""
    tags: list[str] = Field(default_factory=list)
    provenance: Provenance = Field(default_factory=Provenance)


class NPCSpec(NamedEntity):
    kind: Literal["npc"] = "npc"
    pronouns: str = ""
    ancestry: str = ""
    age: str = ""
    status: CharacterStatus = CharacterStatus.ALIVE
    role: str = ""
    appearance: str = ""
    public_history: list[str] = Field(default_factory=list)
    private_history: list[str] = Field(default_factory=list)
    behavior: BehaviorProfile = Field(default_factory=BehaviorProfile)
    current_location_id: StableId | None = None
    home_location_id: StableId | None = None
    faction_ids: list[StableId] = Field(default_factory=list)
    inventory: list[InventoryEntry] = Field(default_factory=list)
    stat_block: StatBlock | None = None


class CreatureSpec(NamedEntity):
    kind: Literal["creature"] = "creature"
    ecology: str = ""
    behavior: list[str] = Field(default_factory=list)
    common_location_ids: list[StableId] = Field(default_factory=list)
    stat_block: StatBlock


class ItemSpec(NamedEntity):
    kind: Literal["item"] = "item"
    category: str = "other"
    description: str = ""
    history: list[str] = Field(default_factory=list)
    significance: str = ""
    mechanics: list[str] = Field(default_factory=list)
    attunement: str = ""
    charges: int | None = Field(default=None, ge=0)
    unique: bool = True
    default_location_id: StableId | None = None


class LocationSpec(NamedEntity):
    kind: Literal["location"] = "location"
    location_kind: LocationKind
    parent_location_id: StableId | None = None
    description: str = ""
    atmosphere: list[str] = Field(default_factory=list)
    sensory_details: list[str] = Field(default_factory=list)
    notable_features: list[str] = Field(default_factory=list)
    hazards: list[str] = Field(default_factory=list)
    access_rules: list[str] = Field(default_factory=list)
    map_coordinates: tuple[float, float] | None = None


class RouteSpec(BaseModel):
    id: StableId
    from_location_id: StableId
    to_location_id: StableId
    bidirectional: bool = True
    travel_time: str = ""
    distance: str = ""
    access_requirements: list[str] = Field(default_factory=list)
    hazards: list[str] = Field(default_factory=list)
    description: str = ""


class FactionSpec(NamedEntity):
    kind: Literal["faction"] = "faction"
    ideology: list[str] = Field(default_factory=list)
    goals: list[str] = Field(default_factory=list)
    methods: list[str] = Field(default_factory=list)
    resources: list[str] = Field(default_factory=list)
    leader_ids: list[StableId] = Field(default_factory=list)
    notable_member_ids: list[StableId] = Field(default_factory=list)
    headquarters_id: StableId | None = None
    territory_location_ids: list[StableId] = Field(default_factory=list)
    ranks: list[str] = Field(default_factory=list)


class LoreDomainSpec(NamedEntity):
    kind: Literal["lore_domain"] = "lore_domain"
    domain_kind: LoreDomainKind
    tenets: list[str] = Field(default_factory=list)
    practices: list[str] = Field(default_factory=list)
    symbols: list[str] = Field(default_factory=list)
    associated_entity_ids: list[StableId] = Field(default_factory=list)


class RelationshipBeat(BaseModel):
    event_id: StableId | None = None
    description: str
    valence_after: int | None = Field(default=None, ge=-100, le=100)


class RelationshipSpec(BaseModel):
    id: StableId
    source_id: StableId
    target_id: StableId
    kind: RelationshipKind
    custom_kind: str | None = None
    directed: bool = True
    valence: int | None = Field(default=None, ge=-100, le=100)
    public_description: str = ""
    private_description: str = ""
    history: list[RelationshipBeat] = Field(default_factory=list)
    active: bool = True

    @model_validator(mode="after")
    def require_custom_kind(self) -> RelationshipSpec:
        if self.kind == RelationshipKind.CUSTOM and not self.custom_kind:
            raise ValueError("custom relationships require custom_kind")
        return self


class KnowledgeClaim(BaseModel):
    """A truth-bearing, visibility-scoped assertion used for governance."""

    id: StableId
    subject_id: StableId
    text: str = Field(min_length=1)
    canon_status: CanonStatus = CanonStatus.CANON
    visibility: Visibility = Visibility.DM_ONLY
    known_by_ids: list[StableId] = Field(default_factory=list)
    evidence_claim_ids: list[StableId] = Field(default_factory=list)
    valid_from_event_id: StableId | None = None
    invalidated_by_event_id: StableId | None = None
    contradiction_group: str | None = None
    provenance: Provenance = Field(default_factory=Provenance)


class HistoricalEvent(BaseModel):
    id: StableId
    title: str
    date_label: str
    sort_order: int
    summary: str
    participant_ids: list[StableId] = Field(default_factory=list)
    location_ids: list[StableId] = Field(default_factory=list)
    cause_event_ids: list[StableId] = Field(default_factory=list)
    consequence_ids: list[StableId] = Field(default_factory=list)
    visibility: Visibility = Visibility.DM_ONLY


class QuestObjective(BaseModel):
    id: StableId
    description: str
    prerequisite_objective_ids: list[StableId] = Field(default_factory=list)
    completion_conditions: list[str] = Field(default_factory=list)
    failure_conditions: list[str] = Field(default_factory=list)
    location_ids: list[StableId] = Field(default_factory=list)
    involved_entity_ids: list[StableId] = Field(default_factory=list)


class QuestSpec(NamedEntity):
    kind: Literal["quest"] = "quest"
    hook: str = ""
    stakes: list[str] = Field(default_factory=list)
    giver_ids: list[StableId] = Field(default_factory=list)
    objectives: list[QuestObjective] = Field(default_factory=list)
    reveal_claim_ids: list[StableId] = Field(default_factory=list)
    reward_item_ids: list[StableId] = Field(default_factory=list)
    success_consequences: list[str] = Field(default_factory=list)
    failure_consequences: list[str] = Field(default_factory=list)
    expiry_trigger: str | None = None


class StoryBeat(BaseModel):
    id: StableId
    title: str
    purpose: str
    trigger_conditions: list[str] = Field(default_factory=list)
    involved_entity_ids: list[StableId] = Field(default_factory=list)
    location_ids: list[StableId] = Field(default_factory=list)
    reveal_claim_ids: list[StableId] = Field(default_factory=list)
    possible_consequences: list[str] = Field(default_factory=list)
    optional: bool = False


class StoryArcSpec(NamedEntity):
    kind: Literal["story_arc"] = "story_arc"
    premise: str
    central_question: str
    themes: list[str] = Field(default_factory=list)
    involved_entity_ids: list[StableId] = Field(default_factory=list)
    beats: list[StoryBeat] = Field(default_factory=list)
    ending_possibilities: list[str] = Field(default_factory=list)
    escalation_clocks: dict[str, int] = Field(default_factory=dict)


class EncounterSpec(NamedEntity):
    kind: Literal["encounter"] = "encounter"
    location_ids: list[StableId] = Field(default_factory=list)
    participant_ids: list[StableId] = Field(default_factory=list)
    trigger_conditions: list[str] = Field(default_factory=list)
    stakes: list[str] = Field(default_factory=list)
    noncombat_solutions: list[str] = Field(default_factory=list)
    scaling_notes: list[str] = Field(default_factory=list)


class StartingState(BaseModel):
    location_id: StableId
    opening_situation: str
    active_quest_ids: list[StableId] = Field(default_factory=list)
    active_story_arc_ids: list[StableId] = Field(default_factory=list)
    player_known_claim_ids: list[StableId] = Field(default_factory=list)
    party_item_ids: list[StableId] = Field(default_factory=list)
    initial_clocks: dict[str, int] = Field(default_factory=dict)


class SourcebookMetadata(BaseModel):
    schema_version: Literal["1.0"] = "1.0"
    sourcebook_id: StableId
    title: str = Field(min_length=1)
    pitch: str
    ruleset: str = "dnd5e"
    tone: list[str] = Field(default_factory=list)
    themes: list[str] = Field(default_factory=list)
    safety_boundaries: list[str] = Field(default_factory=list)
    authoring_notes: list[str] = Field(default_factory=list)


class CampaignSourcebook(BaseModel):
    metadata: SourcebookMetadata
    locations: list[LocationSpec] = Field(default_factory=list)
    routes: list[RouteSpec] = Field(default_factory=list)
    npcs: list[NPCSpec] = Field(default_factory=list)
    creatures: list[CreatureSpec] = Field(default_factory=list)
    factions: list[FactionSpec] = Field(default_factory=list)
    items: list[ItemSpec] = Field(default_factory=list)
    lore_domains: list[LoreDomainSpec] = Field(default_factory=list)
    relationships: list[RelationshipSpec] = Field(default_factory=list)
    claims: list[KnowledgeClaim] = Field(default_factory=list)
    timeline: list[HistoricalEvent] = Field(default_factory=list)
    quests: list[QuestSpec] = Field(default_factory=list)
    story_arcs: list[StoryArcSpec] = Field(default_factory=list)
    encounters: list[EncounterSpec] = Field(default_factory=list)
    starting_state: StartingState

    @model_validator(mode="after")
    def validate_graph_integrity(self) -> CampaignSourcebook:
        records: list[object] = [
            *self.locations,
            *self.routes,
            *self.npcs,
            *self.creatures,
            *self.factions,
            *self.items,
            *self.lore_domains,
            *self.relationships,
            *self.claims,
            *self.timeline,
            *self.quests,
            *self.story_arcs,
            *self.encounters,
        ]
        records.extend(
            objective for quest in self.quests for objective in quest.objectives
        )
        records.extend(story_beat for arc in self.story_arcs for story_beat in arc.beats)
        record_ids = [str(getattr(record, "id")) for record in records]
        duplicates = sorted(
            record_id for record_id, count in Counter(record_ids).items() if count > 1
        )
        if duplicates:
            raise ValueError(f"duplicate sourcebook ids: {duplicates}")

        known = set(record_ids)
        entity_ids = {
            str(record.id)
            for record in [
                *self.locations,
                *self.npcs,
                *self.creatures,
                *self.factions,
                *self.items,
                *self.lore_domains,
                *self.quests,
                *self.story_arcs,
                *self.encounters,
            ]
        }
        claim_ids = {str(claim.id) for claim in self.claims}
        event_ids = {str(event.id) for event in self.timeline}
        location_ids = {str(location.id) for location in self.locations}
        npc_ids = {str(npc.id) for npc in self.npcs}
        faction_ids = {str(faction.id) for faction in self.factions}
        item_ids = {str(item.id) for item in self.items}
        quest_ids = {str(quest.id) for quest in self.quests}
        story_arc_ids = {str(arc.id) for arc in self.story_arcs}

        def require(ref: str | None, label: str, allowed: set[str] = known) -> None:
            if ref and ref not in allowed:
                raise ValueError(f"{label} references missing id {ref!r}")

        for location in self.locations:
            require(location.parent_location_id, f"location {location.id} parent", location_ids)
        location_parents = {
            str(location.id): {str(location.parent_location_id)}
            for location in self.locations
            if location.parent_location_id
        }
        _reject_cycles(location_parents, "location hierarchy")

        for route in self.routes:
            require(route.from_location_id, f"route {route.id} origin", location_ids)
            require(route.to_location_id, f"route {route.id} destination", location_ids)
            if route.from_location_id == route.to_location_id:
                raise ValueError(f"route {route.id} cannot connect a location to itself")

        unique_items = {str(item.id) for item in self.items if item.unique}
        unique_item_holders: dict[str, str] = {}
        for npc in self.npcs:
            require(npc.current_location_id, f"npc {npc.id} current location", location_ids)
            require(npc.home_location_id, f"npc {npc.id} home", location_ids)
            for faction_id in npc.faction_ids:
                require(faction_id, f"npc {npc.id} faction", faction_ids)
            for entry in npc.inventory:
                require(entry.item_id, f"npc {npc.id} inventory", item_ids)
                item_id = str(entry.item_id)
                if item_id in unique_items:
                    prior = unique_item_holders.get(item_id)
                    if prior:
                        raise ValueError(
                            f"unique item {item_id!r} is held by both {prior!r} and {npc.id!r}"
                        )
                    unique_item_holders[item_id] = str(npc.id)

        for creature in self.creatures:
            for location_id in creature.common_location_ids:
                require(location_id, f"creature {creature.id} location", location_ids)
        for item in self.items:
            require(item.default_location_id, f"item {item.id} default location", location_ids)
            if item.default_location_id and str(item.id) in unique_item_holders:
                raise ValueError(
                    f"unique item {item.id!r} has both default_location_id and NPC holder"
                )
        for faction in self.factions:
            for ref in [*faction.leader_ids, *faction.notable_member_ids]:
                require(ref, f"faction {faction.id} member", npc_ids)
            require(faction.headquarters_id, f"faction {faction.id} headquarters", location_ids)
            for ref in faction.territory_location_ids:
                require(ref, f"faction {faction.id} territory", location_ids)
        for domain in self.lore_domains:
            for ref in domain.associated_entity_ids:
                require(ref, f"lore domain {domain.id} association", entity_ids)

        for relationship in self.relationships:
            require(relationship.source_id, f"relationship {relationship.id} source", entity_ids)
            require(relationship.target_id, f"relationship {relationship.id} target", entity_ids)
            for relationship_beat in relationship.history:
                require(
                    relationship_beat.event_id,
                    f"relationship {relationship.id} history",
                    event_ids,
                )
        for claim in self.claims:
            require(claim.subject_id, f"claim {claim.id} subject", entity_ids)
            for ref in claim.known_by_ids:
                require(ref, f"claim {claim.id} knower", entity_ids)
            for ref in claim.evidence_claim_ids:
                require(ref, f"claim {claim.id} evidence", claim_ids)
            require(claim.valid_from_event_id, f"claim {claim.id} valid_from", event_ids)
            require(claim.invalidated_by_event_id, f"claim {claim.id} invalidated_by", event_ids)
        _reject_cycles(
            {
                str(claim.id): {str(ref) for ref in claim.evidence_claim_ids}
                for claim in self.claims
                if claim.evidence_claim_ids
            },
            "claim evidence graph",
        )
        for event in self.timeline:
            for ref in event.participant_ids:
                require(ref, f"event {event.id} participant", entity_ids)
            for ref in event.location_ids:
                require(ref, f"event {event.id} location", location_ids)
            for ref in event.cause_event_ids:
                require(ref, f"event {event.id} cause", event_ids)
            for ref in event.consequence_ids:
                require(ref, f"event {event.id} consequence", known)
        _reject_cycles(
            {
                str(event.id): {str(ref) for ref in event.cause_event_ids}
                for event in self.timeline
                if event.cause_event_ids
            },
            "timeline cause graph",
        )

        for quest in self.quests:
            for ref in quest.giver_ids:
                require(ref, f"quest {quest.id} giver", entity_ids)
            for ref in quest.reveal_claim_ids:
                require(ref, f"quest {quest.id} reveal", claim_ids)
            for ref in quest.reward_item_ids:
                require(ref, f"quest {quest.id} reward", item_ids)
            objective_ids = {str(objective.id) for objective in quest.objectives}
            for objective in quest.objectives:
                for ref in objective.prerequisite_objective_ids:
                    require(ref, f"quest {quest.id} objective prerequisite", objective_ids)
                for ref in objective.location_ids:
                    require(ref, f"quest {quest.id} objective location", location_ids)
                for ref in objective.involved_entity_ids:
                    require(ref, f"quest {quest.id} objective participant", entity_ids)
            _reject_cycles(
                {
                    str(objective.id): {
                        str(ref) for ref in objective.prerequisite_objective_ids
                    }
                    for objective in quest.objectives
                    if objective.prerequisite_objective_ids
                },
                f"quest {quest.id} objective graph",
            )

        for arc in self.story_arcs:
            for ref in arc.involved_entity_ids:
                require(ref, f"story arc {arc.id} participant", entity_ids)
            for story_beat in arc.beats:
                for ref in story_beat.involved_entity_ids:
                    require(ref, f"story beat {story_beat.id} participant", entity_ids)
                for ref in story_beat.location_ids:
                    require(ref, f"story beat {story_beat.id} location", location_ids)
                for ref in story_beat.reveal_claim_ids:
                    require(ref, f"story beat {story_beat.id} reveal", claim_ids)
        for encounter in self.encounters:
            for ref in encounter.location_ids:
                require(ref, f"encounter {encounter.id} location", location_ids)
            for ref in encounter.participant_ids:
                require(ref, f"encounter {encounter.id} participant", entity_ids)

        require(self.starting_state.location_id, "starting location", location_ids)
        for ref in self.starting_state.active_quest_ids:
            require(ref, "starting quest", quest_ids)
        for ref in self.starting_state.active_story_arc_ids:
            require(ref, "starting story arc", story_arc_ids)
        for ref in self.starting_state.player_known_claim_ids:
            require(ref, "starting player-known claim", claim_ids)
        for ref in self.starting_state.party_item_ids:
            require(ref, "starting party item", item_ids)
            if str(ref) in unique_item_holders:
                raise ValueError(
                    f"unique item {ref!r} is assigned to both the party and "
                    f"NPC {unique_item_holders[str(ref)]!r}"
                )
        return self
