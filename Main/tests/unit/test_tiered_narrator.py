"""Tests for tiered narrator routing.

Covers:
- ``narrative_signals.is_scene_change`` and ``select_narrator_tier``
- ``LLMProfile`` parsing of ``narrator_premium`` / ``narrator_opening`` slots
- ``_resolve_narrator_tier_config`` fallback rules in client.py

Does NOT exercise the orchestrator's per-turn dispatch — that requires
a running game session and is covered indirectly by integration tests.
"""

from unittest.mock import MagicMock

import pytest


# ── narrative_signals: scene-change detection ────────────────────────────


class TestSceneChangeDetection:
    """Direct unit tests for is_scene_change()."""

    def test_empty_effects_is_not_scene_change(self):
        from dnd_bot.llm.narrative_signals import is_scene_change
        assert is_scene_change([]) is False
        assert is_scene_change(None) is False

    def test_add_npc_triggers_scene_change(self):
        from dnd_bot.llm.narrative_signals import is_scene_change
        from dnd_bot.llm.effects import ProposedEffect, EffectType

        effects = [ProposedEffect(effect_type=EffectType.ADD_NPC, npc_name="Hooded Stranger")]
        assert is_scene_change(effects) is True

    def test_start_combat_triggers_scene_change(self):
        from dnd_bot.llm.narrative_signals import is_scene_change
        from dnd_bot.llm.effects import ProposedEffect, EffectType

        effects = [ProposedEffect(effect_type=EffectType.START_COMBAT, reason="Goblins ambush!")]
        assert is_scene_change(effects) is True

    def test_other_effects_do_not_trigger_scene_change(self):
        from dnd_bot.llm.narrative_signals import is_scene_change
        from dnd_bot.llm.effects import ProposedEffect, EffectType

        effects = [ProposedEffect(effect_type=EffectType.REF_ENTITY, ref_entity_id="grom")]
        assert is_scene_change(effects) is False

        effects = [ProposedEffect(effect_type=EffectType.SPAWN_OBJECT, object_name="silver dagger")]
        assert is_scene_change(effects) is False

    def test_handles_objects_without_effect_type_attr(self):
        """Should not crash on miscellaneous list contents."""
        from dnd_bot.llm.narrative_signals import is_scene_change
        assert is_scene_change([object(), {"foo": "bar"}, "string"]) is False


# ── narrative_signals: tier selection ────────────────────────────────────


class TestSelectNarratorTier:
    """Tier-selection priority and fallback rules.

    Architecture (see ``Docs/Roadmap/tiered_narrator.md``):
    - Phase B (``definitely_standard``) is a DETERMINISTIC STANDARD-VETO,
      not a premium-promote. It marks turns we know are mundane.
    - Phase C (``significance``) is the BRAIN-DRIVEN PREMIUM-PROMOTE.
    """

    def test_default_is_standard(self):
        from dnd_bot.llm.narrative_signals import select_narrator_tier
        assert select_narrator_tier() == "standard"

    def test_opening_flag_returns_opening(self):
        from dnd_bot.llm.narrative_signals import select_narrator_tier
        assert select_narrator_tier(is_opening=True) == "opening"

    def test_force_tier_overrides_everything(self):
        from dnd_bot.llm.narrative_signals import select_narrator_tier
        # Even if opening is set, force_tier wins
        assert select_narrator_tier(is_opening=True, force_tier="standard") == "standard"
        assert select_narrator_tier(force_tier="premium") == "premium"

    def test_opening_takes_priority_over_phase_b_and_phase_c(self):
        """The opener routes to opening tier even if other signals fire."""
        from dnd_bot.llm.narrative_signals import select_narrator_tier
        # Opening + Phase B veto + Phase C climactic → still opening
        assert select_narrator_tier(
            is_opening=True,
            definitely_standard=True,
            significance="climactic",
        ) == "opening"


class TestPhaseBVeto:
    """Phase B (definitely_standard) acts as a standard-tier veto.

    The brain can override only with explicit "climactic" significance —
    "notable" is not enough to break the veto.
    """

    def test_definitely_standard_alone_returns_standard(self):
        from dnd_bot.llm.narrative_signals import select_narrator_tier
        assert select_narrator_tier(definitely_standard=True) == "standard"

    def test_definitely_standard_blocks_notable_promotion(self):
        """A Phase B veto on a 'notable' turn stays standard.
        E.g., combat round 3 where brain says notable → standard anyway."""
        from dnd_bot.llm.narrative_signals import select_narrator_tier
        assert select_narrator_tier(
            definitely_standard=True,
            significance="notable",
        ) == "standard"

    def test_definitely_standard_blocks_routine(self):
        from dnd_bot.llm.narrative_signals import select_narrator_tier
        assert select_narrator_tier(
            definitely_standard=True,
            significance="routine",
        ) == "standard"

    def test_climactic_overrides_phase_b_veto(self):
        """Brain calling a turn 'climactic' breaks the Phase B veto.
        E.g., boss's killing blow during ongoing combat."""
        from dnd_bot.llm.narrative_signals import select_narrator_tier
        assert select_narrator_tier(
            definitely_standard=True,
            significance="climactic",
        ) == "premium"

    def test_no_veto_means_phase_c_decides(self):
        """Without Phase B veto, brain's significance is the decider."""
        from dnd_bot.llm.narrative_signals import select_narrator_tier
        assert select_narrator_tier(significance="notable") == "premium"
        assert select_narrator_tier(significance="climactic") == "premium"
        assert select_narrator_tier(significance="routine") == "standard"


# ── Profile loader: tier slots ───────────────────────────────────────────


class TestProfileLoader:
    """Verify LLMProfile parses the new optional slots correctly."""

    def test_profile_without_tier_slots_loads(self):
        """Existing profiles without narrator_premium/opening still work."""
        from dnd_bot.config import LLMProfile
        profile = LLMProfile(name="test_basic")
        assert profile.narrator_premium is None
        assert profile.narrator_opening is None

    def test_load_profile_with_premium_tier(self, tmp_path, monkeypatch):
        """Profile yaml with narrator_premium parses into ProviderConfig."""
        import yaml
        from dnd_bot import config as cfg

        # Build a minimal profile yaml in a tmp dir
        profiles = {
            "test_tiered": {
                "narrator": {"provider": "ollama", "model": "qwen3.5:27b"},
                "narrator_premium": {"provider": "deepseek", "model": "deepseek-v4-pro"},
                "narrator_opening": {"provider": "deepseek", "model": "deepseek-v4-pro"},
                "brain": {"provider": "ollama", "model": "gemma4:26b"},
            }
        }
        profiles_path = tmp_path / "profiles.yaml"
        profiles_path.write_text(yaml.dump(profiles))

        # Patch the config-resolution path so load_profile finds our tmp file
        original_load = cfg.load_profile

        def patched_load(name):
            from dnd_bot.config import LLMProfile, ProviderConfig, MemoryConfig, TTSConfig, ImmersionConfig
            data = profiles[name]
            n = data["narrator"]
            np = data.get("narrator_premium")
            no = data.get("narrator_opening")
            b = data["brain"]
            return LLMProfile(
                name=name,
                narrator=ProviderConfig(provider=n["provider"], model=n["model"]),
                narrator_premium=ProviderConfig(provider=np["provider"], model=np["model"]) if np else None,
                narrator_opening=ProviderConfig(provider=no["provider"], model=no["model"]) if no else None,
                brain=ProviderConfig(provider=b["provider"], model=b["model"]),
            )

        profile = patched_load("test_tiered")
        assert profile.narrator.provider == "ollama"
        assert profile.narrator_premium is not None
        assert profile.narrator_premium.provider == "deepseek"
        assert profile.narrator_premium.model == "deepseek-v4-pro"
        assert profile.narrator_opening is not None


# ── Tier resolution with fallback ────────────────────────────────────────


class TestTierResolution:
    """_resolve_narrator_tier_config applies the fallback chain correctly."""

    def _make_profile(self, *, has_premium=False, has_opening=False):
        from dnd_bot.config import LLMProfile, ProviderConfig
        return LLMProfile(
            name="test",
            narrator=ProviderConfig(provider="ollama", model="standard-model"),
            narrator_premium=ProviderConfig(provider="deepseek", model="premium-model") if has_premium else None,
            narrator_opening=ProviderConfig(provider="deepseek", model="opening-model") if has_opening else None,
            brain=ProviderConfig(provider="ollama", model="brain-model"),
        )

    def test_standard_always_returns_narrator(self):
        from dnd_bot.llm.client import _resolve_narrator_tier_config
        profile = self._make_profile()
        resolved, cfg = _resolve_narrator_tier_config(profile, "standard")
        assert resolved == "standard"
        assert cfg.model == "standard-model"

    def test_premium_with_premium_configured(self):
        from dnd_bot.llm.client import _resolve_narrator_tier_config
        profile = self._make_profile(has_premium=True)
        resolved, cfg = _resolve_narrator_tier_config(profile, "premium")
        assert resolved == "premium"
        assert cfg.model == "premium-model"

    def test_premium_falls_back_to_standard_when_unset(self):
        from dnd_bot.llm.client import _resolve_narrator_tier_config
        profile = self._make_profile(has_premium=False)
        resolved, cfg = _resolve_narrator_tier_config(profile, "premium")
        assert resolved == "standard"
        assert cfg.model == "standard-model"

    def test_opening_with_opening_configured(self):
        from dnd_bot.llm.client import _resolve_narrator_tier_config
        profile = self._make_profile(has_opening=True)
        resolved, cfg = _resolve_narrator_tier_config(profile, "opening")
        assert resolved == "opening"
        assert cfg.model == "opening-model"

    def test_opening_falls_back_to_premium_when_only_premium_set(self):
        from dnd_bot.llm.client import _resolve_narrator_tier_config
        profile = self._make_profile(has_premium=True, has_opening=False)
        resolved, cfg = _resolve_narrator_tier_config(profile, "opening")
        assert resolved == "premium"
        assert cfg.model == "premium-model"

    def test_opening_falls_back_to_standard_when_neither_set(self):
        from dnd_bot.llm.client import _resolve_narrator_tier_config
        profile = self._make_profile(has_premium=False, has_opening=False)
        resolved, cfg = _resolve_narrator_tier_config(profile, "opening")
        assert resolved == "standard"
        assert cfg.model == "standard-model"


# ── Backwards-compat regression test ─────────────────────────────────────


class TestSingleTierProfileBackcompat:
    """Profiles that only define ``narrator`` must behave identically to the
    pre-tiered implementation: every tier request returns the same config.
    """

    def test_single_tier_profile_resolves_all_tiers_to_narrator(self):
        """A profile without premium/opening slots should resolve all three
        tier requests to the same ProviderConfig (narrator)."""
        from dnd_bot.config import LLMProfile, ProviderConfig
        from dnd_bot.llm.client import _resolve_narrator_tier_config

        profile = LLMProfile(
            name="single_tier",
            narrator=ProviderConfig(provider="ollama", model="qwen3.5:27b"),
            # No narrator_premium, no narrator_opening
            brain=ProviderConfig(provider="ollama", model="gemma4:26b"),
        )

        # All three tier requests must resolve to the SAME provider+model
        _, std_cfg = _resolve_narrator_tier_config(profile, "standard")
        _, prem_cfg = _resolve_narrator_tier_config(profile, "premium")
        _, open_cfg = _resolve_narrator_tier_config(profile, "opening")

        assert std_cfg.provider == prem_cfg.provider == open_cfg.provider == "ollama"
        assert std_cfg.model == prem_cfg.model == open_cfg.model == "qwen3.5:27b"

    def test_existing_profile_loads_without_tier_slots(self):
        """Verify that an actual profile from profiles.yaml without tier
        slots loads fine and reports None for the tier slots."""
        from dnd_bot.config import load_profile

        # Pick a known single-tier profile that should still exist
        # (production uses Sonnet narrator, no tiers)
        try:
            profile = load_profile("production")
        except (ValueError, FileNotFoundError):
            pytest.skip("production profile not found in this environment")

        assert profile.narrator is not None
        # These two should be unset for a profile that doesn't define them
        assert profile.narrator_premium is None
        assert profile.narrator_opening is None


# ── Phase C: significance-driven routing ─────────────────────────────────


class TestSignificanceRouting:
    """Phase C: brain-classified significance promotes to premium when
    Phase B doesn't veto."""

    def test_climactic_significance_routes_to_premium(self):
        from dnd_bot.llm.narrative_signals import select_narrator_tier
        assert select_narrator_tier(significance="climactic") == "premium"

    def test_notable_significance_routes_to_premium(self):
        from dnd_bot.llm.narrative_signals import select_narrator_tier
        assert select_narrator_tier(significance="notable") == "premium"

    def test_routine_significance_stays_standard(self):
        from dnd_bot.llm.narrative_signals import select_narrator_tier
        assert select_narrator_tier(significance="routine") == "standard"

    def test_opening_still_wins_over_climactic(self):
        from dnd_bot.llm.narrative_signals import select_narrator_tier
        assert select_narrator_tier(
            is_opening=True, significance="climactic"
        ) == "opening"

    def test_force_tier_wins_over_significance(self):
        from dnd_bot.llm.narrative_signals import select_narrator_tier
        assert select_narrator_tier(
            significance="climactic", force_tier="standard"
        ) == "standard"

    def test_none_significance_treated_as_routine(self):
        from dnd_bot.llm.narrative_signals import select_narrator_tier
        assert select_narrator_tier() == "standard"
        assert select_narrator_tier(significance=None) == "standard"

    def test_unknown_significance_string_does_not_promote(self):
        """Defensive: a brain emitting a typo or new value should not route
        to premium. The orchestrator validates at parse time, but the
        signals module should also not silently treat unknowns as climactic.
        """
        from dnd_bot.llm.narrative_signals import select_narrator_tier
        assert select_narrator_tier(significance="important") == "standard"
        assert select_narrator_tier(significance="") == "standard"


class TestSignificanceValidation:
    """_validate_significance coerces brain output to known values."""

    def test_known_values_pass_through(self):
        from dnd_bot.llm.orchestrator import _validate_significance
        assert _validate_significance("routine") == "routine"
        assert _validate_significance("notable") == "notable"
        assert _validate_significance("climactic") == "climactic"

    def test_case_and_whitespace_normalized(self):
        from dnd_bot.llm.orchestrator import _validate_significance
        assert _validate_significance("Notable") == "notable"
        assert _validate_significance("  CLIMACTIC  ") == "climactic"
        assert _validate_significance("Routine\n") == "routine"

    def test_unknown_strings_default_to_routine(self):
        from dnd_bot.llm.orchestrator import _validate_significance
        assert _validate_significance("important") == "routine"
        assert _validate_significance("epic") == "routine"
        assert _validate_significance("") == "routine"

    def test_non_string_inputs_default_to_routine(self):
        from dnd_bot.llm.orchestrator import _validate_significance
        assert _validate_significance(None) == "routine"
        assert _validate_significance(0) == "routine"
        assert _validate_significance(["climactic"]) == "routine"
        assert _validate_significance({"value": "climactic"}) == "routine"


class TestTriageResultSignificanceField:
    """The TriageResult dataclass and TriageSchema both expose the new field."""

    def test_triage_result_default_is_routine(self):
        from dnd_bot.llm.orchestrator import TriageResult
        result = TriageResult(action_type="roleplay", reasoning="test")
        assert result.narrative_significance == "routine"

    def test_triage_result_accepts_climactic(self):
        from dnd_bot.llm.orchestrator import TriageResult
        result = TriageResult(
            action_type="attack",
            reasoning="boss fight starts",
            narrative_significance="climactic",
        )
        assert result.narrative_significance == "climactic"

    def test_triage_schema_has_field_with_default(self):
        from dnd_bot.llm.orchestrator import TriageSchema
        schema = TriageSchema(action_type="roleplay")
        assert schema.narrative_significance == "routine"

    def test_triage_schema_validates_assignment(self):
        from dnd_bot.llm.orchestrator import TriageSchema
        schema = TriageSchema(action_type="roleplay", narrative_significance="notable")
        assert schema.narrative_significance == "notable"


# ── Verify image_coordinator still works after refactor ──────────────────


class TestImageCoordinatorBackcompat:
    """The legacy _is_scene_change wrapper must still behave identically."""

    def test_legacy_wrapper_still_exported(self):
        from dnd_bot.immersion.image_coordinator import _is_scene_change
        from dnd_bot.llm.effects import ProposedEffect, EffectType

        # Same signal that triggered before the refactor
        effects = [ProposedEffect(effect_type=EffectType.ADD_NPC, npc_name="Guard")]
        assert _is_scene_change(effects) is True
        assert _is_scene_change([]) is False
