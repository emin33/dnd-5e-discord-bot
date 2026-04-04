-- Migration 003: Knowledge Graph tables
-- Persistent entity relationships for narrator context enrichment.
-- NetworkX in-memory graph backed by these tables for crash-safe persistence.

CREATE TABLE IF NOT EXISTS kg_node (
    campaign_id TEXT NOT NULL REFERENCES campaign(id) ON DELETE CASCADE,
    node_id     TEXT NOT NULL,
    entity_type TEXT NOT NULL,   -- 'npc', 'location', 'item', 'quest'
    name        TEXT NOT NULL,
    aliases     TEXT DEFAULT '[]',   -- JSON array of alternate names
    properties  TEXT DEFAULT '{}',   -- JSON dict of entity attributes
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (campaign_id, node_id)
);

CREATE TABLE IF NOT EXISTS kg_edge (
    campaign_id   TEXT NOT NULL REFERENCES campaign(id) ON DELETE CASCADE,
    source_id     TEXT NOT NULL,
    target_id     TEXT NOT NULL,
    relation_type TEXT NOT NULL,
    weight        REAL DEFAULT 1.0,
    properties    TEXT DEFAULT '{}',   -- JSON dict of edge metadata
    created_at    TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (campaign_id, source_id, target_id, relation_type),
    FOREIGN KEY (campaign_id, source_id) REFERENCES kg_node(campaign_id, node_id) ON DELETE CASCADE,
    FOREIGN KEY (campaign_id, target_id) REFERENCES kg_node(campaign_id, node_id) ON DELETE CASCADE
);

-- Indexes for common query patterns
CREATE INDEX IF NOT EXISTS idx_kg_node_type   ON kg_node(campaign_id, entity_type);
CREATE INDEX IF NOT EXISTS idx_kg_node_name   ON kg_node(campaign_id, name COLLATE NOCASE);
CREATE INDEX IF NOT EXISTS idx_kg_edge_source ON kg_edge(campaign_id, source_id);
CREATE INDEX IF NOT EXISTS idx_kg_edge_target ON kg_edge(campaign_id, target_id);

INSERT INTO schema_migrations (version) VALUES (3);
