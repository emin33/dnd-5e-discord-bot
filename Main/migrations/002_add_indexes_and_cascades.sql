-- Migration 002: Add missing indexes on foreign keys and fix cascades
-- These indexes improve query performance for common lookups.

-- Character lookups by campaign
CREATE INDEX IF NOT EXISTS idx_character_campaign ON character(campaign_id);

-- Session lookups by campaign
CREATE INDEX IF NOT EXISTS idx_session_campaign ON game_session(campaign_id);

-- Inventory lookups by character
CREATE INDEX IF NOT EXISTS idx_inventory_character ON character_inventory(character_id);

-- Condition lookups by character
CREATE INDEX IF NOT EXISTS idx_condition_character ON character_condition(character_id);

-- Spell lookups by character
CREATE INDEX IF NOT EXISTS idx_spell_character ON character_spell(character_id);

-- Spell slot lookups by character
CREATE INDEX IF NOT EXISTS idx_spell_slots_character ON character_spell_slots(character_id);

-- NPC relationship lookups
CREATE INDEX IF NOT EXISTS idx_npc_rel_npc ON npc_relationship(npc_id);
CREATE INDEX IF NOT EXISTS idx_npc_rel_character ON npc_relationship(character_id);

-- Proficiency lookups by character
CREATE INDEX IF NOT EXISTS idx_proficiency_character ON character_proficiency(character_id);

-- Record this migration
INSERT INTO schema_migrations (version) VALUES (2);
