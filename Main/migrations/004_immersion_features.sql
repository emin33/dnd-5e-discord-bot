-- Immersion features: multi-voice TTS + scene image generation
-- Adds voice_id to NPCs and characters, per-guild settings, and voice catalog

-- Per-guild immersion settings
CREATE TABLE IF NOT EXISTS guild_immersion_settings (
    guild_id INTEGER PRIMARY KEY,
    tts_enabled INTEGER DEFAULT 0,
    image_enabled INTEGER DEFAULT 0,
    image_frequency TEXT DEFAULT 'on_demand'
        CHECK(image_frequency IN ('every', 'scene_change', 'on_demand')),
    narrator_tts_provider TEXT DEFAULT 'inworld',
    narrator_tts_voice TEXT DEFAULT 'Diego',
    character_tts_provider TEXT DEFAULT ''
);

-- Voice catalog for NPC auto-assignment
CREATE TABLE IF NOT EXISTS voice_catalog (
    voice_id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    provider TEXT NOT NULL DEFAULT 'elevenlabs',
    gender TEXT CHECK(gender IN ('male', 'female', 'neutral')),
    age TEXT CHECK(age IN ('young', 'mature', 'old')),
    style_tags TEXT DEFAULT '[]'
);
