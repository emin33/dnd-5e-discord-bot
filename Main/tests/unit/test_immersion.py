"""Tests for immersion features: prose parser, voice assigner, image coordinator."""

import pytest
from unittest.mock import MagicMock

from dnd_bot.models.immersion import NarrativeSegment, SegmentType
from dnd_bot.models.npc import SceneEntity, EntityType, Disposition


# ── Prose Parser ────────────────────────────────────────────────────────────


class TestProseParser:
    """Test narrative prose parser (quote detection + speaker attribution)."""

    def _parse(self, text, effects=None, registry=None, characters=None):
        from dnd_bot.immersion.prose_parser import parse_narrative
        return parse_narrative(text, effects or [], registry, characters or [])

    def test_no_dialogue_returns_single_narration(self):
        text = "The forest is dark and foreboding. Twisted branches claw at the sky."
        segments = self._parse(text)
        assert len(segments) == 1
        assert segments[0].segment_type == SegmentType.NARRATION
        assert "forest" in segments[0].text

    def test_single_quote_splits_into_three_segments(self):
        text = 'The merchant leans forward. "Welcome to my shop, adventurer!" He gestures broadly.'
        segments = self._parse(text)
        assert len(segments) == 3
        assert segments[0].segment_type == SegmentType.NARRATION
        assert segments[1].segment_type == SegmentType.DIALOGUE
        assert segments[1].text == "Welcome to my shop, adventurer!"
        assert segments[2].segment_type == SegmentType.NARRATION

    def test_multiple_quotes_interleaved(self):
        text = (
            'Grom pounds his fist on the table. "You dare insult me?" '
            'The crowd falls silent. Elara steps forward. "Peace, Grom. They meant no harm."'
        )
        segments = self._parse(text)
        dialogue_segments = [s for s in segments if s.segment_type == SegmentType.DIALOGUE]
        assert len(dialogue_segments) == 2
        assert "insult" in dialogue_segments[0].text
        assert "Peace" in dialogue_segments[1].text

    def test_speaker_attribution_from_context(self):
        text = 'Captain Thorne says "Stand down, soldier!"'
        segments = self._parse(text)
        dialogue = [s for s in segments if s.segment_type == SegmentType.DIALOGUE][0]
        assert dialogue.speaker_name == "Captain Thorne"

    def test_speaker_attribution_whispers(self):
        text = 'Elara whispers "Follow me through the passage."'
        segments = self._parse(text)
        dialogue = [s for s in segments if s.segment_type == SegmentType.DIALOGUE][0]
        assert dialogue.speaker_name == "Elara"

    def test_speaker_attribution_with_scene_registry(self):
        text = 'Grom growled "Get out of my forge!"'
        registry = MagicMock()
        entity = SceneEntity(
            name="Grom the Blacksmith",
            entity_type=EntityType.NPC,
            description="A burly dwarf",
            npc_id="npc-grom-123",
        )
        registry.get_by_name.return_value = entity
        segments = self._parse(text, registry=registry)
        dialogue = [s for s in segments if s.segment_type == SegmentType.DIALOGUE][0]
        assert dialogue.speaker_entity_id == "npc-grom-123"

    def test_empty_input_returns_empty(self):
        assert self._parse("") == []
        assert self._parse("   ") == []

    def test_trivially_short_quotes_skipped(self):
        text = 'He said "no" and walked away.'
        segments = self._parse(text)
        # "no" is <= 2 chars, should be skipped as trivial
        dialogue = [s for s in segments if s.segment_type == SegmentType.DIALOGUE]
        assert len(dialogue) == 0

    def test_curly_quotes_handled(self):
        text = 'The innkeeper smiles. \u201cWelcome, travelers!\u201d She wipes the counter.'
        segments = self._parse(text)
        dialogue = [s for s in segments if s.segment_type == SegmentType.DIALOGUE]
        assert len(dialogue) == 1
        assert "Welcome" in dialogue[0].text

    def test_player_character_attribution(self):
        from dnd_bot.models import Character, AbilityScores, HitPoints, HitDice
        char = Character(
            discord_user_id=1, campaign_id="test", name="Thorin",
            race_index="dwarf", class_index="fighter",
            abilities=AbilityScores(), hp=HitPoints(maximum=10, current=10),
            hit_dice=HitDice(die_type=10, total=1, remaining=1),
        )
        registry = MagicMock()
        registry.get_by_name.return_value = None
        text = 'Thorin shouts "For the mountain!"'
        segments = self._parse(text, characters=[char], registry=registry)
        dialogue = [s for s in segments if s.segment_type == SegmentType.DIALOGUE][0]
        assert dialogue.speaker_name == "Thorin"
        assert dialogue.speaker_entity_id.startswith("pc:")


# ── Voice Assigner (keyword detection) ──────────────────────────────────────


class TestVoiceAssignerKeywords:
    """Test gender/age keyword detection (no DB or API calls)."""

    def test_detect_female(self):
        from dnd_bot.immersion.voice_assigner import _detect_gender
        assert _detect_gender("A young woman with silver hair") == "female"
        assert _detect_gender("She carries a staff and wears a cloak") == "female"
        assert _detect_gender("The old priestess of the moon") == "female"

    def test_detect_male(self):
        from dnd_bot.immersion.voice_assigner import _detect_gender
        assert _detect_gender("A burly man with a thick beard") == "male"
        assert _detect_gender("He wields a massive battle axe") == "male"
        assert _detect_gender("The old wizard in his tower") == "male"

    def test_detect_neutral_when_ambiguous(self):
        from dnd_bot.immersion.voice_assigner import _detect_gender
        assert _detect_gender("A cloaked figure lurks in the shadows") == "neutral"
        assert _detect_gender("The merchant behind the counter") == "neutral"

    def test_detect_young(self):
        from dnd_bot.immersion.voice_assigner import _detect_age
        assert _detect_age("A young apprentice eager to learn") == "young"
        assert _detect_age("The boy clutches a wooden sword") == "young"

    def test_detect_old(self):
        from dnd_bot.immersion.voice_assigner import _detect_age
        assert _detect_age("An old wizened sage with a long beard") == "old"
        assert _detect_age("The ancient elder speaks slowly") == "old"
        assert _detect_age("A grizzled veteran of many wars") == "old"

    def test_detect_mature_default(self):
        from dnd_bot.immersion.voice_assigner import _detect_age
        assert _detect_age("A soldier standing guard at the gate") == "mature"
        assert _detect_age("The merchant displays exotic wares") == "mature"


# ── Image Coordinator (frequency logic) ─────────────────────────────────────


class TestImageCoordinatorFrequency:
    """Test image generation frequency decision logic (no API calls)."""

    def test_scene_change_detected_on_add_npc(self):
        from dnd_bot.immersion.image_coordinator import _is_scene_change
        from dnd_bot.llm.effects import ProposedEffect, EffectType

        effects = [ProposedEffect(effect_type=EffectType.ADD_NPC, npc_name="Guard")]
        assert _is_scene_change(effects) is True

    def test_scene_change_detected_on_start_combat(self):
        from dnd_bot.immersion.image_coordinator import _is_scene_change
        from dnd_bot.llm.effects import ProposedEffect, EffectType

        effects = [ProposedEffect(effect_type=EffectType.START_COMBAT, reason="Ambush")]
        assert _is_scene_change(effects) is True

    def test_no_scene_change_on_ref_entity(self):
        from dnd_bot.immersion.image_coordinator import _is_scene_change
        from dnd_bot.llm.effects import ProposedEffect, EffectType

        effects = [ProposedEffect(effect_type=EffectType.REF_ENTITY, ref_entity_id="grom")]
        assert _is_scene_change(effects) is False

    def test_no_scene_change_on_empty_effects(self):
        from dnd_bot.immersion.image_coordinator import _is_scene_change
        assert _is_scene_change([]) is False


# ── NarrativeSegment Model ──────────────────────────────────────────────────


class TestNarrativeSegmentModel:
    """Test NarrativeSegment Pydantic model."""

    def test_narration_segment(self):
        seg = NarrativeSegment(text="The forest looms ahead.", segment_type=SegmentType.NARRATION)
        assert seg.voice_id is None
        assert seg.speaker_entity_id is None

    def test_dialogue_segment_with_speaker(self):
        seg = NarrativeSegment(
            text="Welcome to my shop!",
            segment_type=SegmentType.DIALOGUE,
            speaker_name="Grom",
            speaker_entity_id="grom-the-blacksmith",
            voice_id="pNInz6obpgDQGcFmaJgB",
            voice_provider="elevenlabs",
        )
        assert seg.speaker_name == "Grom"
        assert seg.voice_provider == "elevenlabs"


# ── Guild Immersion Settings Model ──────────────────────────────────────────


class TestGuildImmersionSettings:
    """Test GuildImmersionSettings Pydantic model."""

    def test_defaults(self):
        from dnd_bot.models.immersion import GuildImmersionSettings, ImageFrequency
        settings = GuildImmersionSettings(guild_id=12345)
        assert settings.tts_enabled is False
        assert settings.image_enabled is False
        assert settings.image_frequency == ImageFrequency.ON_DEMAND
        assert settings.narrator_tts_provider == "inworld"
        assert settings.narrator_tts_voice == "Sarah"

    def test_custom_settings(self):
        from dnd_bot.models.immersion import GuildImmersionSettings, ImageFrequency
        settings = GuildImmersionSettings(
            guild_id=99999,
            tts_enabled=True,
            image_enabled=True,
            image_frequency=ImageFrequency.EVERY,
            narrator_tts_provider="elevenlabs",
            narrator_tts_voice="pNInz6obpgDQGcFmaJgB",
        )
        assert settings.tts_enabled is True
        assert settings.image_frequency == ImageFrequency.EVERY


# ── Voice Catalog Entry Model ───────────────────────────────────────────────


class TestVoiceCatalogEntry:
    """Test VoiceCatalogEntry Pydantic model."""

    def test_basic_entry(self):
        from dnd_bot.models.immersion import VoiceCatalogEntry
        entry = VoiceCatalogEntry(
            voice_id="abc123",
            name="TestVoice",
            gender="male",
            age="mature",
            style_tags=["gruff", "warrior"],
        )
        assert entry.provider == "elevenlabs"
        assert len(entry.style_tags) == 2

    def test_catalog_json_loadable(self):
        """Verify the voice catalog JSON is valid and has expected structure."""
        import json
        from pathlib import Path
        from dnd_bot.models.immersion import VoiceCatalogEntry

        catalog_path = Path(__file__).parent.parent.parent / "dnd_bot" / "data" / "voice_catalog.json"
        data = json.loads(catalog_path.read_text())

        assert len(data) >= 10, "Catalog should have at least 10 voices"

        # Verify each entry is valid
        for entry_data in data:
            entry = VoiceCatalogEntry(**entry_data)
            assert entry.voice_id
            assert entry.name
            assert entry.gender in ("male", "female", "neutral")
            assert entry.age in ("young", "mature", "old")

        # Verify coverage: at least 1 male and 1 female voice
        genders = {e["gender"] for e in data}
        assert "male" in genders
        assert "female" in genders
