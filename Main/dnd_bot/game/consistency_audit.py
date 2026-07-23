"""Cross-store consistency audit — the provable-storage check.

The campaign's truth lives in five places: WorldState (roster, fact
ledger), the knowledge graph (durable identities), ChromaDB (semantic
recall), the scene registry (on-stage entities), and the memory tiers
(pinned facts, summaries). Each write seam is individually tested, but
nothing asserted that the stores actually AGREE at the end of a run —
which is how the pinned-fact resurrection loop hid: memory faithfully
re-synced a fact the world ledger had just retired, every single turn.

This audit is deterministic (no LLM) and cheap enough to run at the end
of every harness run. HARD invariants must hold exactly; COVERAGE
metrics are reported with counts so drift becomes visible before it
becomes a defect.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import structlog

from .identity import identity_keys, is_generic_npc_label

logger = structlog.get_logger()


@dataclass
class ConsistencyReport:
    violations: list[str] = field(default_factory=list)
    coverage: dict[str, Any] = field(default_factory=dict)
    counts: dict[str, int] = field(default_factory=dict)

    @property
    def passed(self) -> bool:
        return not self.violations

    def to_dict(self) -> dict[str, Any]:
        return {
            "passed": self.passed,
            "violations": self.violations,
            "coverage": self.coverage,
            "counts": self.counts,
        }


def run_consistency_audit(
    *,
    world_state,
    knowledge_graph=None,
    scene_registry=None,
    memory=None,
    vector_store=None,
    campaign_id: str = "",
) -> ConsistencyReport:
    """Audit cross-store agreement. Every collaborator except world_state
    is optional — absent stores skip their checks rather than failing."""
    report = ConsistencyReport()

    established = list(getattr(world_state, "established_facts", []) or [])
    superseded_entries = list(getattr(world_state, "superseded_facts", []) or [])
    superseded = {
        str(entry.get("fact") or "") for entry in superseded_entries
    }
    report.counts["established_facts"] = len(established)
    report.counts["superseded_facts"] = len(superseded_entries)

    # H1: a fact is live or retired, never both.
    both = sorted(set(established) & superseded)
    for fact in both[:5]:
        report.violations.append(f"fact_in_both_ledgers: {fact[:120]}")

    # H2: every superseded entry carries provenance.
    for entry in superseded_entries:
        if not str(entry.get("fact") or "") or not str(entry.get("superseded_by") or ""):
            report.violations.append(
                f"superseded_entry_missing_provenance: {str(entry)[:120]}"
            )

    # H3: memory pinned facts must not include retired facts (the
    # resurrection loop's signature).
    if memory is not None:
        pinned = list(getattr(getattr(memory, "buffer", None), "pinned_facts", []) or [])
        report.counts["pinned_facts"] = len(pinned)
        stale = sorted(set(pinned) & superseded)
        for fact in stale[:5]:
            report.violations.append(f"pinned_fact_is_superseded: {fact[:120]}")

    # H4: scene-registry canonical links point at real roster NPCs.
    world_npc_ids = set((getattr(world_state, "npcs", {}) or {}).keys())
    report.counts["world_npcs"] = len(world_npc_ids)
    if scene_registry is not None:
        try:
            scene_entities = scene_registry.get_all()
        except Exception as e:
            scene_entities = []
            report.violations.append(f"scene_registry_unreadable: {e}")
        linked = 0
        for entity in scene_entities:
            npc_id = getattr(entity, "npc_id", None)
            if not npc_id:
                continue
            linked += 1
            if npc_id not in world_npc_ids:
                report.violations.append(
                    "scene_link_dangling: "
                    f"{getattr(entity, 'name', '?')} -> {npc_id}"
                )
        report.counts["scene_entities"] = len(scene_entities)
        report.counts["scene_npc_links"] = linked

    graph_entities = list(
        (getattr(knowledge_graph, "_entities", {}) or {}).values()
    ) if knowledge_graph is not None else []
    graph_npc_nodes = [
        entity for entity in graph_entities
        if getattr(getattr(entity, "entity_type", None), "value", "") == "npc"
    ]
    report.counts["kg_nodes"] = len(graph_entities)
    report.counts["kg_npc_nodes"] = len(graph_npc_nodes)

    # H5: proper-named durable NPC identities are unique in the graph
    # (the canonical_npc_identity_unique gate, enforced live).
    if graph_npc_nodes:
        owners: dict[str, set[str]] = {}
        for node in graph_npc_nodes:
            name = str(getattr(node, "name", "") or "")
            if not name or is_generic_npc_label(name):
                continue
            for key in identity_keys(name):
                owners.setdefault(key, set()).add(
                    str(getattr(node, "node_id", "") or "")
                )
        for key, node_ids in sorted(owners.items()):
            if len(node_ids) > 1:
                report.violations.append(
                    f"kg_npc_name_collision: '{key}' -> {sorted(node_ids)}"
                )

    # S1 (coverage): roster NPCs that own a durable graph node.
    if knowledge_graph is not None and world_npc_ids:
        graph_ids = {
            str(getattr(node, "node_id", "") or "") for node in graph_entities
        }
        in_graph = sum(1 for npc_id in world_npc_ids if npc_id in graph_ids)
        report.coverage["world_npcs_with_kg_node"] = (
            f"{in_graph}/{len(world_npc_ids)}"
        )

    # S2 (coverage): described KG entities that carry a Chroma embedding.
    if vector_store is not None and campaign_id and graph_entities:
        described = [
            str(getattr(node, "node_id", "") or "")
            for node in graph_entities
            if (getattr(node, "properties", {}) or {}).get("description")
        ]
        try:
            indexed = vector_store.indexed_entity_ids(campaign_id, described)
        except Exception as e:
            indexed = set()
            report.violations.append(f"vector_store_unreadable: {e}")
        report.coverage["described_kg_entities_indexed"] = (
            f"{len(indexed)}/{len(described)}"
        )
        missing = sorted(set(described) - indexed)
        if missing:
            report.coverage["unindexed_entity_ids"] = missing[:8]

    logger.info(
        "consistency_audit",
        passed=report.passed,
        violations=len(report.violations),
        counts=report.counts,
        coverage=report.coverage,
    )
    return report
