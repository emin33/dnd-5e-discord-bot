-- Migration 007: Canonical tables for authored sourcebooks
--
-- SOURCEBOOK_COMPILER_DESIGN.md: "SQLite/sourcebook data is canonical; the
-- graph and embeddings are rebuildable indexes." Until now a compiled book
-- existed ONLY in the knowledge graph — a rebuildable index with nothing to
-- rebuild from, and a deliberately lossy one (24 RelationshipKinds collapse
-- onto 9 RelationTypes; visibility is enforced by DROPPING content). These
-- tables are the layer the graph is an index OF.
--
-- Column-vs-JSON rule (design doc, "Why canonical tables, and not a document
-- blob"): normalize what the runtime overlay joins, filters or traverses;
-- keep read-whole substructures as JSON columns on their owner row. The
-- queries play actually generates are the ones that decided this shape:
--
--   * which DISCOVERABLE claims has this party not yet earned
--       -> sourcebook_claim LEFT JOIN campaign_claim_state
--   * which claim supersedes which, once play overturns canon
--       -> campaign_claim_state.superseded_by_claim_id
--   * every member of a faction / every NPC hostile to X
--       -> sourcebook_npc_faction / sourcebook_relationship
--   * everything authored in a region the party has not touched
--       -> recursive sourcebook_location + campaign_location_state
--   * what did the party learn, and when
--       -> campaign_claim_state.discovered_at_turn
--
-- Foreign keys are declared on STRUCTURAL parents only (owner rows and join
-- tables, inserted after their parents). They are deliberately NOT declared
-- on the soft cross-references — parent_location_id, current_location_id,
-- default_location_id, headquarters_id, claim.subject_id, relationship
-- endpoints, event ids — because those are polymorphic and/or
-- self-referential, and SQLite enforces FKs immediately, which would force
-- the importer to topologically sort every list. CampaignSourcebook's
-- model validator already rejects dangling references, containment cycles,
-- and duplicate ownership before a single row is written.

-- ============================================
-- VERSION HEADER
-- ============================================

-- sourcebook_key is the sha256 of the book's canonical JSON: a version's
-- identity is its content, so re-importing the same bytes is detectable and
-- an edited book is a different version rather than a silent overwrite.
CREATE TABLE IF NOT EXISTS sourcebook (
    sourcebook_key      TEXT PRIMARY KEY,
    sourcebook_id       TEXT NOT NULL,          -- metadata.sourcebook_id (stable across versions)
    schema_version      TEXT NOT NULL DEFAULT '1.0',
    title               TEXT NOT NULL,
    pitch               TEXT NOT NULL DEFAULT '',
    ruleset             TEXT NOT NULL DEFAULT 'dnd5e',
    -- tone/themes/safety_boundaries/authoring_notes: read whole, never filtered
    metadata_json       TEXT NOT NULL DEFAULT '{}',
    -- StartingState: one row's worth, read whole by the compiler
    starting_state_json TEXT NOT NULL DEFAULT '{}',
    imported_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX IF NOT EXISTS idx_sourcebook_id ON sourcebook(sourcebook_id);

-- ============================================
-- CANONICAL ENTITIES
-- ============================================

-- sort_order on every entity table is the record's POSITION IN THE AUTHORED
-- LIST. It exists so a book survives the round trip byte-for-byte: without
-- it, reading rows back would reorder every list into id order, and the
-- rebuilt projection would stop matching the one compiled from the file.
CREATE TABLE IF NOT EXISTS sourcebook_location (
    sourcebook_key     TEXT NOT NULL REFERENCES sourcebook(sourcebook_key) ON DELETE CASCADE,
    id                 TEXT NOT NULL,
    sort_order         INTEGER NOT NULL DEFAULT 0,
    name               TEXT NOT NULL,
    summary            TEXT NOT NULL DEFAULT '',
    location_kind      TEXT NOT NULL,
    parent_location_id TEXT,
    description        TEXT NOT NULL DEFAULT '',
    aliases_json       TEXT NOT NULL DEFAULT '[]',
    tags_json          TEXT NOT NULL DEFAULT '[]',
    -- atmosphere / sensory_details / notable_features / hazards /
    -- access_rules / map_coordinates: sensory colour, read whole
    detail_json        TEXT NOT NULL DEFAULT '{}',
    provenance_json    TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (sourcebook_key, id)
);

CREATE INDEX IF NOT EXISTS idx_sb_location_parent
    ON sourcebook_location(sourcebook_key, parent_location_id);
CREATE INDEX IF NOT EXISTS idx_sb_location_kind
    ON sourcebook_location(sourcebook_key, location_kind);

CREATE TABLE IF NOT EXISTS sourcebook_route (
    sourcebook_key     TEXT NOT NULL REFERENCES sourcebook(sourcebook_key) ON DELETE CASCADE,
    id                 TEXT NOT NULL,
    sort_order         INTEGER NOT NULL DEFAULT 0,
    from_location_id   TEXT NOT NULL,
    to_location_id     TEXT NOT NULL,
    bidirectional      INTEGER NOT NULL DEFAULT 1,
    travel_time        TEXT NOT NULL DEFAULT '',
    distance           TEXT NOT NULL DEFAULT '',
    description        TEXT NOT NULL DEFAULT '',
    detail_json        TEXT NOT NULL DEFAULT '{}',   -- access_requirements, hazards
    PRIMARY KEY (sourcebook_key, id)
);

CREATE INDEX IF NOT EXISTS idx_sb_route_from
    ON sourcebook_route(sourcebook_key, from_location_id);
CREATE INDEX IF NOT EXISTS idx_sb_route_to
    ON sourcebook_route(sourcebook_key, to_location_id);

CREATE TABLE IF NOT EXISTS sourcebook_faction (
    sourcebook_key  TEXT NOT NULL REFERENCES sourcebook(sourcebook_key) ON DELETE CASCADE,
    id              TEXT NOT NULL,
    sort_order      INTEGER NOT NULL DEFAULT 0,
    name            TEXT NOT NULL,
    summary         TEXT NOT NULL DEFAULT '',
    headquarters_id TEXT,
    aliases_json    TEXT NOT NULL DEFAULT '[]',
    tags_json       TEXT NOT NULL DEFAULT '[]',
    -- ideology / goals / methods / resources / ranks: read whole
    profile_json    TEXT NOT NULL DEFAULT '{}',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (sourcebook_key, id)
);

CREATE TABLE IF NOT EXISTS sourcebook_item (
    sourcebook_key      TEXT NOT NULL REFERENCES sourcebook(sourcebook_key) ON DELETE CASCADE,
    id                  TEXT NOT NULL,
    sort_order          INTEGER NOT NULL DEFAULT 0,
    name                TEXT NOT NULL,
    summary             TEXT NOT NULL DEFAULT '',
    category            TEXT NOT NULL DEFAULT 'other',
    description         TEXT NOT NULL DEFAULT '',
    significance        TEXT NOT NULL DEFAULT '',
    attunement          TEXT NOT NULL DEFAULT '',
    charges             INTEGER,
    is_unique           INTEGER NOT NULL DEFAULT 1,
    default_location_id TEXT,
    aliases_json        TEXT NOT NULL DEFAULT '[]',
    tags_json           TEXT NOT NULL DEFAULT '[]',
    detail_json         TEXT NOT NULL DEFAULT '{}',   -- history, mechanics
    provenance_json     TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (sourcebook_key, id)
);

CREATE INDEX IF NOT EXISTS idx_sb_item_location
    ON sourcebook_item(sourcebook_key, default_location_id);

CREATE TABLE IF NOT EXISTS sourcebook_npc (
    sourcebook_key      TEXT NOT NULL REFERENCES sourcebook(sourcebook_key) ON DELETE CASCADE,
    id                  TEXT NOT NULL,
    sort_order          INTEGER NOT NULL DEFAULT 0,
    name                TEXT NOT NULL,
    summary             TEXT NOT NULL DEFAULT '',
    status              TEXT NOT NULL DEFAULT 'alive',
    role                TEXT NOT NULL DEFAULT '',
    appearance          TEXT NOT NULL DEFAULT '',
    pronouns            TEXT NOT NULL DEFAULT '',
    ancestry            TEXT NOT NULL DEFAULT '',
    age                 TEXT NOT NULL DEFAULT '',
    current_location_id TEXT,
    home_location_id    TEXT,
    aliases_json        TEXT NOT NULL DEFAULT '[]',
    tags_json           TEXT NOT NULL DEFAULT '[]',
    -- BehaviorProfile: the design doc's named example of a read-whole
    -- substructure. Nothing filters on a decision rule.
    behavior_json       TEXT NOT NULL DEFAULT '{}',
    public_history_json TEXT NOT NULL DEFAULT '[]',
    -- private_history is DM-only. It lives here because this layer is the
    -- system of record; the compiler is what refuses to project it.
    private_history_json TEXT NOT NULL DEFAULT '[]',
    stat_block_json     TEXT,
    provenance_json     TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (sourcebook_key, id)
);

CREATE INDEX IF NOT EXISTS idx_sb_npc_current_location
    ON sourcebook_npc(sourcebook_key, current_location_id);
CREATE INDEX IF NOT EXISTS idx_sb_npc_home_location
    ON sourcebook_npc(sourcebook_key, home_location_id);
CREATE INDEX IF NOT EXISTS idx_sb_npc_status
    ON sourcebook_npc(sourcebook_key, status);

-- ============================================
-- MEMBERSHIP / OWNERSHIP / TERRITORY (join tables)
-- ============================================

-- "every member of a faction" is one indexed scan, not a walk over every
-- NPC's faction_ids list. membership_role distinguishes the two directions
-- the book can express it from (NPCSpec.faction_ids vs FactionSpec.leader_ids
-- / notable_member_ids), so the round trip can put each back where it came
-- from instead of inventing membership the author did not write.
CREATE TABLE IF NOT EXISTS sourcebook_npc_faction (
    sourcebook_key  TEXT NOT NULL REFERENCES sourcebook(sourcebook_key) ON DELETE CASCADE,
    npc_id          TEXT NOT NULL,
    faction_id      TEXT NOT NULL,
    membership_role TEXT NOT NULL DEFAULT 'member',  -- 'member' | 'leader' | 'notable'
    -- In the PK, not merely beside it: the authoring contract does not
    -- forbid a list from naming the same id twice, and a PK that collapsed
    -- the duplicate would turn a legal book into an import crash.
    sort_order      INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (sourcebook_key, npc_id, faction_id, membership_role, sort_order),
    FOREIGN KEY (sourcebook_key, npc_id)
        REFERENCES sourcebook_npc(sourcebook_key, id) ON DELETE CASCADE,
    FOREIGN KEY (sourcebook_key, faction_id)
        REFERENCES sourcebook_faction(sourcebook_key, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sb_npc_faction_faction
    ON sourcebook_npc_faction(sourcebook_key, faction_id);

-- hidden is a COLUMN, not a JSON field: it is the visibility boundary for
-- ownership, and the compiler filters on it on every import.
CREATE TABLE IF NOT EXISTS sourcebook_npc_inventory (
    sourcebook_key TEXT NOT NULL REFERENCES sourcebook(sourcebook_key) ON DELETE CASCADE,
    npc_id         TEXT NOT NULL,
    item_id        TEXT NOT NULL,
    quantity       INTEGER NOT NULL DEFAULT 1,
    equipped       INTEGER NOT NULL DEFAULT 0,
    hidden         INTEGER NOT NULL DEFAULT 0,
    notes          TEXT NOT NULL DEFAULT '',
    sort_order     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (sourcebook_key, npc_id, item_id, sort_order),
    FOREIGN KEY (sourcebook_key, npc_id)
        REFERENCES sourcebook_npc(sourcebook_key, id) ON DELETE CASCADE,
    FOREIGN KEY (sourcebook_key, item_id)
        REFERENCES sourcebook_item(sourcebook_key, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sb_inventory_item
    ON sourcebook_npc_inventory(sourcebook_key, item_id);

CREATE TABLE IF NOT EXISTS sourcebook_faction_territory (
    sourcebook_key TEXT NOT NULL REFERENCES sourcebook(sourcebook_key) ON DELETE CASCADE,
    faction_id     TEXT NOT NULL,
    location_id    TEXT NOT NULL,
    sort_order     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (sourcebook_key, faction_id, location_id, sort_order),
    FOREIGN KEY (sourcebook_key, faction_id)
        REFERENCES sourcebook_faction(sourcebook_key, id) ON DELETE CASCADE,
    FOREIGN KEY (sourcebook_key, location_id)
        REFERENCES sourcebook_location(sourcebook_key, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sb_territory_location
    ON sourcebook_faction_territory(sourcebook_key, location_id);

-- ============================================
-- QUESTS
-- ============================================

CREATE TABLE IF NOT EXISTS sourcebook_quest (
    sourcebook_key  TEXT NOT NULL REFERENCES sourcebook(sourcebook_key) ON DELETE CASCADE,
    id              TEXT NOT NULL,
    sort_order      INTEGER NOT NULL DEFAULT 0,
    name            TEXT NOT NULL,
    summary         TEXT NOT NULL DEFAULT '',   -- DM-side: where the author writes the twist
    hook            TEXT NOT NULL DEFAULT '',   -- player-facing
    expiry_trigger  TEXT,
    aliases_json    TEXT NOT NULL DEFAULT '[]',
    tags_json       TEXT NOT NULL DEFAULT '[]',
    -- stakes / success_consequences / failure_consequences: read whole
    detail_json     TEXT NOT NULL DEFAULT '{}',
    giver_ids_json  TEXT NOT NULL DEFAULT '[]',
    reveal_claim_ids_json TEXT NOT NULL DEFAULT '[]',
    reward_item_ids_json  TEXT NOT NULL DEFAULT '[]',
    provenance_json TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (sourcebook_key, id)
);

CREATE TABLE IF NOT EXISTS sourcebook_quest_objective (
    sourcebook_key   TEXT NOT NULL REFERENCES sourcebook(sourcebook_key) ON DELETE CASCADE,
    id               TEXT NOT NULL,
    quest_id         TEXT NOT NULL,
    sort_order       INTEGER NOT NULL DEFAULT 0,
    description      TEXT NOT NULL DEFAULT '',
    -- prerequisite_objective_ids / completion_conditions / failure_conditions
    -- / involved_entity_ids: read whole with the objective
    detail_json      TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (sourcebook_key, id),
    FOREIGN KEY (sourcebook_key, quest_id)
        REFERENCES sourcebook_quest(sourcebook_key, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sb_objective_quest
    ON sourcebook_quest_objective(sourcebook_key, quest_id);

-- Normalized because "everything authored in this region" joins through it,
-- and because it is the source of the graph's OBJECTIVE_AT edges.
CREATE TABLE IF NOT EXISTS sourcebook_quest_objective_location (
    sourcebook_key TEXT NOT NULL REFERENCES sourcebook(sourcebook_key) ON DELETE CASCADE,
    objective_id   TEXT NOT NULL,
    location_id    TEXT NOT NULL,
    sort_order     INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (sourcebook_key, objective_id, location_id, sort_order),
    FOREIGN KEY (sourcebook_key, objective_id)
        REFERENCES sourcebook_quest_objective(sourcebook_key, id) ON DELETE CASCADE,
    FOREIGN KEY (sourcebook_key, location_id)
        REFERENCES sourcebook_location(sourcebook_key, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_sb_objective_location_location
    ON sourcebook_quest_objective_location(sourcebook_key, location_id);

-- ============================================
-- RELATIONSHIPS
-- ============================================

-- The full 24-kind authored vocabulary, NOT the graph's 9 retrieval types.
-- This is the table that makes the graph's collapse survivable: "every NPC
-- hostile to X" answered here distinguishes RIVAL_OF from FEARS from
-- HOSTILE_TO, all three of which the graph flattens onto hostile_to.
CREATE TABLE IF NOT EXISTS sourcebook_relationship (
    sourcebook_key      TEXT NOT NULL REFERENCES sourcebook(sourcebook_key) ON DELETE CASCADE,
    id                  TEXT NOT NULL,
    sort_order          INTEGER NOT NULL DEFAULT 0,
    source_id           TEXT NOT NULL,
    target_id           TEXT NOT NULL,
    kind                TEXT NOT NULL,
    custom_kind         TEXT,
    directed            INTEGER NOT NULL DEFAULT 1,
    valence             INTEGER,
    active              INTEGER NOT NULL DEFAULT 1,
    public_description  TEXT NOT NULL DEFAULT '',
    private_description TEXT NOT NULL DEFAULT '',
    history_json        TEXT NOT NULL DEFAULT '[]',
    PRIMARY KEY (sourcebook_key, id)
);

CREATE INDEX IF NOT EXISTS idx_sb_relationship_source
    ON sourcebook_relationship(sourcebook_key, source_id, kind);
CREATE INDEX IF NOT EXISTS idx_sb_relationship_target
    ON sourcebook_relationship(sourcebook_key, target_id, kind);

-- ============================================
-- TIMELINE
-- ============================================

CREATE TABLE IF NOT EXISTS sourcebook_event (
    sourcebook_key  TEXT NOT NULL REFERENCES sourcebook(sourcebook_key) ON DELETE CASCADE,
    id              TEXT NOT NULL,
    title           TEXT NOT NULL,
    date_label      TEXT NOT NULL DEFAULT '',
    -- The one table where these differ: sort_order is the AUTHORED
    -- chronology (HistoricalEvent.sort_order — what you order a timeline by),
    -- list_order is the record's position in the book's list. A book may
    -- list events out of chronological order, and both facts are data.
    sort_order      INTEGER NOT NULL DEFAULT 0,
    list_order      INTEGER NOT NULL DEFAULT 0,
    summary         TEXT NOT NULL DEFAULT '',
    visibility      TEXT NOT NULL DEFAULT 'dm_only',
    -- participants / locations / causes / consequences: read whole with the
    -- event; claims reach the timeline by id, which is what temporal
    -- validity actually needs.
    detail_json     TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (sourcebook_key, id)
);

-- On list_order, not sort_order: `_load_timeline` reads the book back in
-- AUTHORED order, and indexing the other column left a temp B-tree sort on
-- every load while serving no query that exists.
CREATE INDEX IF NOT EXISTS idx_sb_event_order
    ON sourcebook_event(sourcebook_key, list_order);

-- ============================================
-- CLAIMS — the reason this layer exists
-- ============================================

-- Every column here is filtered on by a query the runtime overlay makes.
--
-- superseded_by_claim_id is NULL on import: the v1 authoring contract has no
-- field for authored supersession (it expresses conflict through
-- contradiction_group plus invalidated_by_event_id). The column exists
-- because the effective-claim resolver coalesces the campaign overlay OVER
-- the book, so canon needs the same shape as the overlay for that COALESCE
-- to be meaningful — and a v1.1 contract that adds `supersedes` then needs no
-- migration.
CREATE TABLE IF NOT EXISTS sourcebook_claim (
    sourcebook_key           TEXT NOT NULL REFERENCES sourcebook(sourcebook_key) ON DELETE CASCADE,
    id                       TEXT NOT NULL,
    sort_order               INTEGER NOT NULL DEFAULT 0,
    subject_id               TEXT NOT NULL,
    claim_text               TEXT NOT NULL,
    canon_status             TEXT NOT NULL DEFAULT 'canon',
    visibility               TEXT NOT NULL DEFAULT 'dm_only',
    superseded_by_claim_id   TEXT,
    contradiction_group      TEXT,
    valid_from_event_id      TEXT,
    invalidated_by_event_id  TEXT,
    known_by_json            TEXT NOT NULL DEFAULT '[]',
    evidence_json            TEXT NOT NULL DEFAULT '[]',
    provenance_json          TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (sourcebook_key, id)
);

CREATE INDEX IF NOT EXISTS idx_sb_claim_subject
    ON sourcebook_claim(sourcebook_key, subject_id);
CREATE INDEX IF NOT EXISTS idx_sb_claim_visibility
    ON sourcebook_claim(sourcebook_key, visibility, canon_status);
CREATE INDEX IF NOT EXISTS idx_sb_claim_contradiction
    ON sourcebook_claim(sourcebook_key, contradiction_group);

-- ============================================
-- RECORDS WITH NO QUERY YET
-- ============================================

-- Creatures, lore domains, story arcs (with their beats) and encounters are
-- not joined, filtered or traversed by anything the overlay does — story
-- beats are the design doc's own example of read-whole JSON. They still get
-- a ROW EACH rather than a corner of one document, so they stay addressable
-- and countable, and so the canonical layer round-trips a book exactly
-- instead of quietly dropping the parts nobody queries yet. Promoting one to
-- its own table later is a migration, not a data recovery.
CREATE TABLE IF NOT EXISTS sourcebook_aux_record (
    sourcebook_key TEXT NOT NULL REFERENCES sourcebook(sourcebook_key) ON DELETE CASCADE,
    record_kind    TEXT NOT NULL,   -- 'creature' | 'lore_domain' | 'story_arc' | 'encounter'
    id             TEXT NOT NULL,
    name           TEXT NOT NULL DEFAULT '',
    sort_order     INTEGER NOT NULL DEFAULT 0,
    payload_json   TEXT NOT NULL DEFAULT '{}',
    PRIMARY KEY (sourcebook_key, record_kind, id)
);

-- No index on (sourcebook_key, record_kind): that is an exact leftmost prefix
-- of this table's PK, so the PK's own index already serves it.

-- ============================================
-- CAMPAIGN OVERLAY
-- ============================================

-- Which book version(s) a campaign is playing. Layer 2 of the design doc's
-- truth layers: the overlay's base pointer.
CREATE TABLE IF NOT EXISTS campaign_sourcebook (
    campaign_id    TEXT NOT NULL REFERENCES campaign(id) ON DELETE CASCADE,
    sourcebook_key TEXT NOT NULL REFERENCES sourcebook(sourcebook_key) ON DELETE CASCADE,
    applied_at     TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (campaign_id, sourcebook_key)
);

-- Leading with sourcebook_key, which the PK does NOT cover. Deleting a book
-- cascades into these overlay tables through their FKs, and SQLite resolves a
-- cascade by looking the child up on the PARENT key columns. Without this the
-- planner does a full SCAN of the whole overlay -- every campaign's rows, not
-- just this book's -- once per deleted parent row: measured 624-883ms against
-- 38ms on a 28,800-row overlay, and it degrades with rows that have nothing to
-- do with the book being dropped. Every sourcebook_* join table already got
-- exactly this treatment; these three were the ones whose PK led with the
-- wrong column.
CREATE INDEX IF NOT EXISTS idx_campaign_sourcebook_book
    ON campaign_sourcebook(sourcebook_key);

-- Sparse by design: a row exists only for a claim this campaign has TOUCHED.
-- "Which discoverable claims has this party not yet earned" is therefore a
-- LEFT JOIN with a NULL/0 test, and an untouched book costs zero rows.
--
-- canon_status is an OVERRIDE (NULL = inherit the book's), so play can
-- demote authored canon to FALSE without editing the immutable book.
CREATE TABLE IF NOT EXISTS campaign_claim_state (
    campaign_id            TEXT NOT NULL REFERENCES campaign(id) ON DELETE CASCADE,
    sourcebook_key         TEXT NOT NULL,
    claim_id               TEXT NOT NULL,
    discovered             INTEGER NOT NULL DEFAULT 0,
    discovered_at_turn     INTEGER,
    discovered_via         TEXT NOT NULL DEFAULT '',
    superseded_by_claim_id TEXT,
    canon_status           TEXT,
    note                   TEXT NOT NULL DEFAULT '',
    updated_at             TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (campaign_id, sourcebook_key, claim_id),
    FOREIGN KEY (sourcebook_key, claim_id)
        REFERENCES sourcebook_claim(sourcebook_key, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_campaign_claim_discovered
    ON campaign_claim_state(campaign_id, discovered, discovered_at_turn);
-- The FK's own lookup key (see the note on idx_campaign_sourcebook_book).
CREATE INDEX IF NOT EXISTS idx_campaign_claim_book
    ON campaign_claim_state(sourcebook_key, claim_id);

-- The other half of "authored in a region the party has not touched".
-- Sparse for the same reason: an unvisited world costs nothing.
CREATE TABLE IF NOT EXISTS campaign_location_state (
    campaign_id        TEXT NOT NULL REFERENCES campaign(id) ON DELETE CASCADE,
    sourcebook_key     TEXT NOT NULL,
    location_id        TEXT NOT NULL,
    visited            INTEGER NOT NULL DEFAULT 0,
    first_visited_turn INTEGER,
    last_visited_turn  INTEGER,
    updated_at         TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (campaign_id, sourcebook_key, location_id),
    FOREIGN KEY (sourcebook_key, location_id)
        REFERENCES sourcebook_location(sourcebook_key, id) ON DELETE CASCADE
);

CREATE INDEX IF NOT EXISTS idx_campaign_location_visited
    ON campaign_location_state(campaign_id, visited);
CREATE INDEX IF NOT EXISTS idx_campaign_location_book
    ON campaign_location_state(sourcebook_key, location_id);

INSERT INTO schema_migrations (version) VALUES (7);
