-- D&D 5e Discord Bot - Initial Database Schema
-- Migration 001

-- ============================================
-- CAMPAIGN & SESSION MANAGEMENT
-- ============================================

CREATE TABLE IF NOT EXISTS campaign (
    id TEXT PRIMARY KEY,
    guild_id INTEGER NOT NULL,
    name TEXT NOT NULL,
    description TEXT,
    world_setting TEXT DEFAULT 'A classic high fantasy world filled with magic, monsters, and adventure.',
    dm_user_id INTEGER NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    last_played_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS game_session (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES campaign(id) ON DELETE CASCADE,
    channel_id INTEGER NOT NULL,
    session_number INTEGER DEFAULT 1,
    state TEXT DEFAULT 'lobby' CHECK(state IN ('lobby', 'exploration', 'combat', 'social', 'resting', 'paused', 'ended')),
    active_combat_id TEXT,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS session_snapshot (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES game_session(id) ON DELETE CASCADE,
    snapshot_type TEXT DEFAULT 'manual',
    game_state TEXT NOT NULL,  -- JSON blob
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- CHARACTER DATA
-- ============================================

CREATE TABLE IF NOT EXISTS character (
    id TEXT PRIMARY KEY,
    discord_user_id INTEGER NOT NULL,
    campaign_id TEXT NOT NULL REFERENCES campaign(id) ON DELETE CASCADE,
    name TEXT NOT NULL,

    -- Core info
    race_index TEXT NOT NULL,
    class_index TEXT NOT NULL,
    subclass_index TEXT,
    level INTEGER DEFAULT 1,
    experience INTEGER DEFAULT 0,
    background_index TEXT,

    -- Ability scores
    strength INTEGER NOT NULL DEFAULT 10,
    dexterity INTEGER NOT NULL DEFAULT 10,
    constitution INTEGER NOT NULL DEFAULT 10,
    intelligence INTEGER NOT NULL DEFAULT 10,
    wisdom INTEGER NOT NULL DEFAULT 10,
    charisma INTEGER NOT NULL DEFAULT 10,

    -- Combat stats
    armor_class INTEGER NOT NULL DEFAULT 10,
    speed INTEGER DEFAULT 30,
    initiative_bonus INTEGER DEFAULT 0,

    -- Hit points
    hp_max INTEGER NOT NULL,
    hp_current INTEGER NOT NULL,
    hp_temp INTEGER DEFAULT 0,
    hit_dice_type INTEGER NOT NULL,  -- 6, 8, 10, or 12
    hit_dice_total INTEGER NOT NULL,
    hit_dice_remaining INTEGER NOT NULL,

    -- Death saves
    death_save_successes INTEGER DEFAULT 0,
    death_save_failures INTEGER DEFAULT 0,

    -- Spellcasting
    spellcasting_ability TEXT,  -- 'str', 'dex', 'con', 'int', 'wis', 'cha'
    concentration_spell_id TEXT,

    -- Status
    is_active INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,

    UNIQUE(discord_user_id, campaign_id)
);

-- Spell slots (separate table for clean updates)
CREATE TABLE IF NOT EXISTS character_spell_slots (
    character_id TEXT NOT NULL REFERENCES character(id) ON DELETE CASCADE,
    slot_level INTEGER NOT NULL CHECK(slot_level BETWEEN 1 AND 9),
    slots_max INTEGER NOT NULL DEFAULT 0,
    slots_current INTEGER NOT NULL DEFAULT 0,
    PRIMARY KEY (character_id, slot_level)
);

-- Known/prepared spells
CREATE TABLE IF NOT EXISTS character_spell (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES character(id) ON DELETE CASCADE,
    spell_index TEXT NOT NULL,
    is_prepared INTEGER DEFAULT 1,
    is_always_prepared INTEGER DEFAULT 0,
    UNIQUE(character_id, spell_index)
);

-- Proficiencies
CREATE TABLE IF NOT EXISTS character_proficiency (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES character(id) ON DELETE CASCADE,
    proficiency_index TEXT NOT NULL,
    proficiency_type TEXT NOT NULL,  -- 'skill', 'save', 'tool', 'weapon', 'armor'
    expertise INTEGER DEFAULT 0,
    UNIQUE(character_id, proficiency_index)
);

-- Active conditions
CREATE TABLE IF NOT EXISTS character_condition (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES character(id) ON DELETE CASCADE,
    condition_name TEXT NOT NULL,
    source TEXT,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_round INTEGER,
    expires_time TIMESTAMP,
    combat_id TEXT,
    stacks INTEGER DEFAULT 1
);

-- Inventory
CREATE TABLE IF NOT EXISTS character_inventory (
    id TEXT PRIMARY KEY,
    character_id TEXT NOT NULL REFERENCES character(id) ON DELETE CASCADE,
    item_index TEXT NOT NULL,
    item_name TEXT NOT NULL,
    quantity INTEGER DEFAULT 1,
    equipped INTEGER DEFAULT 0,
    attunement_required INTEGER DEFAULT 0,
    attuned INTEGER DEFAULT 0,
    notes TEXT,
    added_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Currency
CREATE TABLE IF NOT EXISTS character_currency (
    character_id TEXT PRIMARY KEY REFERENCES character(id) ON DELETE CASCADE,
    copper INTEGER DEFAULT 0,
    silver INTEGER DEFAULT 0,
    electrum INTEGER DEFAULT 0,
    gold INTEGER DEFAULT 0,
    platinum INTEGER DEFAULT 0
);

-- ============================================
-- COMBAT
-- ============================================

CREATE TABLE IF NOT EXISTS combat (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES game_session(id) ON DELETE CASCADE,
    channel_id INTEGER NOT NULL,
    state TEXT DEFAULT 'idle' CHECK(state IN ('idle', 'setup', 'rolling_initiative', 'active', 'awaiting_action', 'resolving_action', 'end_turn', 'combat_end')),
    current_round INTEGER DEFAULT 1,
    current_turn_index INTEGER DEFAULT 0,
    encounter_name TEXT,
    encounter_description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    ended_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS combatant (
    id TEXT PRIMARY KEY,
    combat_id TEXT NOT NULL REFERENCES combat(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    is_player INTEGER NOT NULL,
    character_id TEXT,
    monster_index TEXT,
    initiative_roll INTEGER,
    initiative_bonus INTEGER DEFAULT 0,
    turn_order INTEGER,
    hp_max INTEGER NOT NULL,
    hp_current INTEGER NOT NULL,
    armor_class INTEGER NOT NULL DEFAULT 10,
    speed INTEGER DEFAULT 30,
    is_active INTEGER DEFAULT 1,
    is_surprised INTEGER DEFAULT 0,
    action_used INTEGER DEFAULT 0,
    bonus_action_used INTEGER DEFAULT 0,
    reaction_available INTEGER DEFAULT 1,
    movement_remaining INTEGER
);

CREATE TABLE IF NOT EXISTS readied_action (
    id TEXT PRIMARY KEY,
    combatant_id TEXT NOT NULL REFERENCES combatant(id) ON DELETE CASCADE,
    combat_id TEXT NOT NULL REFERENCES combat(id) ON DELETE CASCADE,
    trigger_condition TEXT NOT NULL,
    action_description TEXT NOT NULL,
    spell_index TEXT,
    created_round INTEGER NOT NULL
);

-- ============================================
-- NPCs & RELATIONSHIPS
-- ============================================

CREATE TABLE IF NOT EXISTS npc (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES campaign(id) ON DELETE CASCADE,
    name TEXT NOT NULL,
    description TEXT,
    location TEXT,
    monster_index TEXT,
    base_disposition INTEGER DEFAULT 0,
    voice_notes TEXT,
    is_alive INTEGER DEFAULT 1,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS npc_relationship (
    npc_id TEXT NOT NULL REFERENCES npc(id) ON DELETE CASCADE,
    character_id TEXT NOT NULL REFERENCES character(id) ON DELETE CASCADE,
    sentiment INTEGER DEFAULT 0,  -- -100 to 100
    interaction_count INTEGER DEFAULT 0,
    notes TEXT,
    last_interaction TIMESTAMP,
    PRIMARY KEY (npc_id, character_id)
);

-- ============================================
-- MEMORY & NARRATIVE
-- ============================================

CREATE TABLE IF NOT EXISTS campaign_memory (
    id TEXT PRIMARY KEY,
    campaign_id TEXT NOT NULL REFERENCES campaign(id) ON DELETE CASCADE,
    memory_type TEXT,  -- 'fact', 'event', 'decision', 'npc_note'
    content TEXT NOT NULL,
    metadata TEXT,  -- JSON
    session_id TEXT,
    embedding_id TEXT,  -- Reference to ChromaDB
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS session_summary (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL REFERENCES game_session(id) ON DELETE CASCADE,
    summary_text TEXT NOT NULL,
    key_events TEXT,  -- JSON array
    message_range_start INTEGER,
    message_range_end INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- IDEMPOTENCY & AUDIT
-- ============================================

CREATE TABLE IF NOT EXISTS transaction_log (
    transaction_key TEXT PRIMARY KEY,
    operation_type TEXT NOT NULL,
    target_id TEXT NOT NULL,
    payload TEXT,  -- JSON
    result TEXT,  -- JSON
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    expires_at TIMESTAMP
);

CREATE TABLE IF NOT EXISTS audit_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT,
    action_type TEXT NOT NULL,
    actor_user_id INTEGER,
    target_type TEXT,
    target_id TEXT,
    old_value TEXT,  -- JSON
    new_value TEXT,  -- JSON
    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- ============================================
-- SCHEMA VERSION TRACKING
-- ============================================

CREATE TABLE IF NOT EXISTS schema_migrations (
    version INTEGER PRIMARY KEY,
    applied_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

INSERT OR IGNORE INTO schema_migrations (version) VALUES (1);

-- ============================================
-- INDEXES
-- ============================================

CREATE INDEX IF NOT EXISTS idx_character_user_campaign ON character(discord_user_id, campaign_id);
CREATE INDEX IF NOT EXISTS idx_character_condition_active ON character_condition(character_id, expires_round);
CREATE INDEX IF NOT EXISTS idx_combatant_combat ON combatant(combat_id, turn_order);
CREATE INDEX IF NOT EXISTS idx_transaction_expires ON transaction_log(expires_at);
CREATE INDEX IF NOT EXISTS idx_audit_session ON audit_log(session_id, timestamp);
CREATE INDEX IF NOT EXISTS idx_campaign_guild ON campaign(guild_id);
CREATE INDEX IF NOT EXISTS idx_session_campaign ON game_session(campaign_id);
CREATE INDEX IF NOT EXISTS idx_npc_campaign ON npc(campaign_id);
