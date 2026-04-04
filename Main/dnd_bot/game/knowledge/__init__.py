"""Knowledge Graph — persistent entity relationship tracking."""

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
    RemoveNode,
    UpdateNode,
    slugify,
)
from .graph import KnowledgeGraph
from .bridge import DeltaBridge
from .matcher import EntityNameMatcher
from .repository import KnowledgeGraphRepository, get_kg_repo

__all__ = [
    "AddEdge",
    "AddNode",
    "DEFAULT_WEIGHTS",
    "DeltaBridge",
    "Entity",
    "EntityNameMatcher",
    "EntityType",
    "GraphOperation",
    "KnowledgeGraph",
    "KnowledgeGraphRepository",
    "Relationship",
    "RelationType",
    "RemoveEdge",
    "RemoveNode",
    "UpdateNode",
    "get_kg_repo",
    "slugify",
]
