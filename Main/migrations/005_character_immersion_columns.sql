-- Migration 005: Add immersion columns that migration 004 forgot
--
-- Migration 004's header claimed to add voice_id to characters and NPCs but the
-- file only contained the new tables (guild_immersion_settings, voice_catalog).
-- The character/NPC columns were added out-of-band on dev DBs; fresh deploys
-- crash on character.create() because the INSERT references missing columns.
--
-- This migration is idempotent: the runner (see Database._run_migrations)
-- swallows "duplicate column name" errors per-statement, so it's safe to run
-- on dev DBs that already have these columns.

ALTER TABLE character ADD COLUMN description TEXT DEFAULT '';
ALTER TABLE character ADD COLUMN portrait_url TEXT;
ALTER TABLE character ADD COLUMN voice_id TEXT;

ALTER TABLE npc ADD COLUMN voice_id TEXT;

INSERT INTO schema_migrations (version) VALUES (5);
