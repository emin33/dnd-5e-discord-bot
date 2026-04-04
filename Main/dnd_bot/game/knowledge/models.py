"""Knowledge graph data models — entities, relationships, and graph operations."""

import re
from datetime import datetime
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def slugify(name: str) -> str:
    """Convert a display name to a stable node ID.

    Lowercase, spaces → hyphens, strip non-alphanumeric except hyphens.
    """
    s = name.lower().strip()
    s = re.sub(r"[^a-z0-9\s-]", "", s)
    s = re.sub(r"[\s]+", "-", s)
    s = re.sub(r"-+", "-", s)
    return s.strip("-")


# ---------------------------------------------------------------------------
# Entity types
# ---------------------------------------------------------------------------

class EntityType(str, Enum):
    NPC = "npc"
    LOCATION = "location"
    ITEM = "item"
    QUEST = "quest"  # Defined for schema completeness; not auto-populated in Phase 1


class Entity(BaseModel):
    """A node in the knowledge graph."""

    node_id: str
    entity_type: EntityType
    name: str
    campaign_id: str
    aliases: list[str] = Field(default_factory=list)
    properties: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Relationship types with default BFS weights
# ---------------------------------------------------------------------------

class RelationType(str, Enum):
    QUEST_GIVER = "quest_giver"      # NPC → Quest (highest priority)
    LOCATED_AT = "located_at"        # NPC/Item → Location
    OBJECTIVE_AT = "objective_at"    # Quest → Location
    CONNECTED_TO = "connected_to"    # Location ↔ Location
    KNOWS = "knows"                  # NPC → NPC
    HOSTILE_TO = "hostile_to"        # NPC → NPC
    ALLIED_WITH = "allied_with"      # NPC → NPC
    OWNS = "owns"                    # NPC → Item
    FOUND_AT = "found_at"            # Item → Location


# Lower weight = traversed sooner by BFS (higher priority)
DEFAULT_WEIGHTS: dict[RelationType, float] = {
    RelationType.QUEST_GIVER: 0.1,
    RelationType.LOCATED_AT: 0.3,
    RelationType.OBJECTIVE_AT: 0.2,
    RelationType.HOSTILE_TO: 0.4,
    RelationType.OWNS: 0.5,
    RelationType.ALLIED_WITH: 0.7,
    RelationType.KNOWS: 0.8,
    RelationType.CONNECTED_TO: 1.0,
    RelationType.FOUND_AT: 1.2,
}


class Relationship(BaseModel):
    """An edge in the knowledge graph."""

    source_id: str
    target_id: str
    relation_type: RelationType
    weight: float = 1.0
    campaign_id: str
    properties: dict[str, str] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# ---------------------------------------------------------------------------
# Graph mutation operations
# ---------------------------------------------------------------------------

class AddNode(BaseModel):
    op: Literal["add_node"] = "add_node"
    entity: Entity


class UpdateNode(BaseModel):
    op: Literal["update_node"] = "update_node"
    node_id: str
    properties: dict[str, str] = Field(default_factory=dict)
    aliases: Optional[list[str]] = None


class RemoveNode(BaseModel):
    op: Literal["remove_node"] = "remove_node"
    node_id: str


class AddEdge(BaseModel):
    op: Literal["add_edge"] = "add_edge"
    relationship: Relationship


class RemoveEdge(BaseModel):
    op: Literal["remove_edge"] = "remove_edge"
    source_id: str
    target_id: str
    relation_type: RelationType


GraphOperation = Union[AddNode, UpdateNode, RemoveNode, AddEdge, RemoveEdge]
