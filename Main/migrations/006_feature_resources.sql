-- Migration 006: Durable class-feature resources
--
-- (character_id, resource_key, current, maximum, recharge_rule, source) —
-- the durable counter model LONGFORM_READINESS_2026_07.md's rest section
-- calls for. Rest recovery previously probed getattr() attributes the
-- persisted Character never has, so nothing but Warlock pact slots ever
-- actually recovered from a rest. Rows are seeded lazily by the rest flow
-- (no backfill needed for existing characters).

CREATE TABLE IF NOT EXISTS feature_resources (
    character_id TEXT NOT NULL REFERENCES character(id) ON DELETE CASCADE,
    resource_key TEXT NOT NULL,
    name TEXT NOT NULL DEFAULT '',
    current INTEGER NOT NULL DEFAULT 0,
    maximum INTEGER NOT NULL DEFAULT 0,
    recharge_rule TEXT NOT NULL DEFAULT 'long_rest',
    source TEXT NOT NULL DEFAULT '',
    PRIMARY KEY (character_id, resource_key)
);

INSERT INTO schema_migrations (version) VALUES (6);
