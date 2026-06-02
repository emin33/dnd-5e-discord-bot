"""Tests for narrator tool tier system + new tools (change_location, update_entity)."""

import pytest


# ── Tier dispatch ────────────────────────────────────────────────────────


class TestTierDispatch:
    """get_narrator_tools_for_tier returns the right tool set per tier."""

    def test_core_tier_has_three_tools(self):
        from dnd_bot.llm.narrator_tools import get_narrator_tools_for_tier
        tools = get_narrator_tools_for_tier("core")
        names = {t["function"]["name"] for t in tools}
        assert names == {"ref_entity", "add_npc", "spawn_object"}

    def test_core_plus_adds_state_declaration_tools(self):
        from dnd_bot.llm.narrator_tools import get_narrator_tools_for_tier
        tools = get_narrator_tools_for_tier("core_plus")
        names = {t["function"]["name"] for t in tools}
        # Core tools still present
        assert "ref_entity" in names
        assert "add_npc" in names
        assert "spawn_object" in names
        # Plus the two state-declaration tools
        assert "change_location" in names
        assert "start_combat" in names
        assert len(tools) == 5

    def test_full_tier_includes_all_tools(self):
        from dnd_bot.llm.narrator_tools import get_narrator_tools_for_tier
        tools = get_narrator_tools_for_tier("full")
        names = {t["function"]["name"] for t in tools}
        # New tools present in full
        assert "change_location" in names
        assert "update_entity" in names
        assert "start_combat" in names
        # Player-state consolidation: items, currency, HP, conditions
        assert "update_player" in names
        assert "request_roll" in names
        # Tools subsumed by update_player are removed
        assert "update_inventory" not in names
        assert "apply_damage" not in names
        # Tools subsumed earlier (by update_inventory, now update_player)
        assert "offer_item" not in names
        assert "grant_currency" not in names
        assert "transfer_item" not in names
        # Full tier post-consolidation: 8 tools (was 9 before update_player)
        assert len(tools) >= 8

    def test_unknown_tier_falls_back_to_core(self):
        from dnd_bot.llm.narrator_tools import get_narrator_tools_for_tier
        tools = get_narrator_tools_for_tier("super_full_max")
        names = {t["function"]["name"] for t in tools}
        # Should fall back to core, not crash
        assert names == {"ref_entity", "add_npc", "spawn_object"}

    def test_empty_string_tier_falls_back_to_core(self):
        from dnd_bot.llm.narrator_tools import get_narrator_tools_for_tier
        tools = get_narrator_tools_for_tier("")
        names = {t["function"]["name"] for t in tools}
        assert names == {"ref_entity", "add_npc", "spawn_object"}


# ── ProviderConfig field ────────────────────────────────────────────────


class TestProviderConfigToolsField:
    """ProviderConfig has a tools field that defaults to 'core'."""

    def test_default_tier_is_core(self):
        from dnd_bot.config import ProviderConfig
        cfg = ProviderConfig(provider="ollama", model="qwen3.5:27b")
        assert cfg.tools == "core"

    def test_explicit_tier_persists(self):
        from dnd_bot.config import ProviderConfig
        cfg = ProviderConfig(
            provider="deepseek",
            model="deepseek-v4-flash",
            tools="full",
        )
        assert cfg.tools == "full"


# ── change_location tool conversion ─────────────────────────────────────


class TestChangeLocationTool:
    """change_location → ProposedEffect conversion."""

    def test_basic_change_location(self):
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        from dnd_bot.llm.effects import EffectType

        effect = _convert_tool_call("change_location", {
            "location_name": "the rusty compass",
            "description": "A warm tavern with a roaring fire.",
        })
        assert effect is not None
        assert effect.effect_type == EffectType.CHANGE_LOCATION
        assert effect.location_name == "the rusty compass"
        assert "warm tavern" in effect.location_description

    def test_change_location_minimal(self):
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        effect = _convert_tool_call("change_location", {
            "location_name": "north gate",
        })
        assert effect.location_name == "north gate"
        assert effect.location_description is None

    def test_change_location_strips_whitespace(self):
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        effect = _convert_tool_call("change_location", {
            "location_name": "  the tavern  ",
        })
        assert effect.location_name == "the tavern"


# ── update_player tool conversion ────────────────────────────────────────


class TestUpdatePlayerTool:
    """update_player consolidates apply_damage + the player-side of the
    old update_inventory (items + currency). Single tool with optional
    independent fields; all absent fields = no change."""

    def test_player_picks_up_scene_item(self):
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        from dnd_bot.llm.effects import EffectType

        effect = _convert_tool_call("update_player", {
            "item_grant": [{"name": "jade dagger", "source": "scene:jade_dagger_1"}],
        })
        assert effect.effect_type == EffectType.UPDATE_PLAYER
        assert effect.player_item_grant[0]["name"] == "jade dagger"
        assert effect.player_item_grant[0]["source"] == "scene:jade_dagger_1"

    def test_npc_pays_player_currency(self):
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        from dnd_bot.llm.effects import EffectType

        effect = _convert_tool_call("update_player", {
            "currency_delta": {"gp": 50},
        })
        assert effect.effect_type == EffectType.UPDATE_PLAYER
        assert effect.player_currency_delta == {"gp": 50}

    def test_player_spends_currency_negative_delta(self):
        """Player paying NPC → narrator passes negative currency delta directly."""
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        effect = _convert_tool_call("update_player", {
            "currency_delta": {"sp": -12},
            "item_grant": [{"name": "rope", "source": "npc:merchant"}],
        })
        assert effect.player_currency_delta == {"sp": -12}
        assert effect.player_item_grant[0]["name"] == "rope"

    def test_environmental_damage(self):
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        effect = _convert_tool_call("update_player", {
            "hp_delta": -6,
            "damage_type": "poison",
            "hp_reason": "wall trap dart",
            "add_conditions": ["poisoned"],
        })
        assert effect.player_hp_delta == -6
        assert effect.player_damage_type == "poison"
        assert effect.player_hp_reason == "wall trap dart"
        assert effect.player_add_conditions == ["poisoned"]

    def test_healing(self):
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        effect = _convert_tool_call("update_player", {
            "hp_delta": 8,
            "hp_reason": "potion of healing",
            "item_remove": [{"name": "potion of healing"}],
        })
        assert effect.player_hp_delta == 8
        assert effect.player_item_remove[0]["name"] == "potion of healing"

    def test_player_hands_item_to_npc_with_destination(self):
        """User's example: relic given to innkeeper for safekeeping."""
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        effect = _convert_tool_call("update_player", {
            "item_remove": [{"name": "ancient relic", "destination": "npc:innkeeper"}],
        })
        assert effect.player_item_remove[0]["name"] == "ancient relic"
        assert effect.player_item_remove[0]["destination"] == "npc:innkeeper"

    def test_no_op_call_validation(self):
        """Empty update_player should fail validation as a no-op."""
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        from dnd_bot.llm.effects import EffectValidator
        # Converter still returns an effect; validator catches the no-op
        effect = _convert_tool_call("update_player", {})
        assert effect is not None
        result = EffectValidator().validate(effect)
        assert result.valid is False
        assert "no mutation" in result.rejection_reason.lower()

    def test_damage_without_type_rejected(self):
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        from dnd_bot.llm.effects import EffectValidator
        effect = _convert_tool_call("update_player", {
            "hp_delta": -5,
            # no damage_type
        })
        result = EffectValidator().validate(effect)
        assert result.valid is False
        assert "damage_type" in result.rejection_reason

    def test_invalid_currency_denomination_rejected(self):
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        # Invalid denoms get filtered by the converter, leaving an empty
        # currency dict — which then makes the call a no-op.
        effect = _convert_tool_call("update_player", {
            "currency_delta": {"bitcoin": 1},
        })
        from dnd_bot.llm.effects import EffectValidator
        result = EffectValidator().validate(effect)
        assert result.valid is False  # no-op after sanitization

    def test_string_item_normalized_to_dict(self):
        """Some models pass items as bare strings — converter normalizes."""
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        effect = _convert_tool_call("update_player", {
            "item_grant": ["jade dagger"],
        })
        assert effect.player_item_grant == [{"name": "jade dagger"}]


# ── update_entity inventory extension ──────────────────────────────────


class TestUpdateEntityInventory:
    """update_entity now supports add_items / remove_items so NPCs can hold
    things (the innkeeper-with-relic 20-turns-later use case)."""

    def test_npc_gains_item(self):
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        from dnd_bot.llm.effects import EffectType
        effect = _convert_tool_call("update_entity", {
            "entity_id": "innkeeper",
            "add_items": ["ancient relic"],
        })
        assert effect.effect_type == EffectType.UPDATE_ENTITY
        assert effect.update_add_items == ["ancient relic"]

    def test_npc_loses_item(self):
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        effect = _convert_tool_call("update_entity", {
            "entity_id": "bandit",
            "remove_items": ["cutlass"],
            "status": "fled",
        })
        assert effect.update_remove_items == ["cutlass"]
        assert effect.update_status == "fled"

    def test_inventory_only_change_accepted(self):
        """update_entity with ONLY inventory changes is valid (no longer a no-op)."""
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        from dnd_bot.llm.effects import EffectValidator
        effect = _convert_tool_call("update_entity", {
            "entity_id": "marta",
            "add_items": ["herbal potion"],
        })
        result = EffectValidator().validate(effect)
        assert result.valid is True

    def test_items_lowercased(self):
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        effect = _convert_tool_call("update_entity", {
            "entity_id": "marta",
            "add_items": ["Healing Potion"],
        })
        assert effect.update_add_items == ["healing potion"]


# ── update_entity tool conversion ───────────────────────────────────────


class TestUpdateEntityTool:
    """update_entity → ProposedEffect conversion."""

    def test_disposition_only_update(self):
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        from dnd_bot.llm.effects import EffectType

        effect = _convert_tool_call("update_entity", {
            "entity_id": "kael",
            "disposition": "hostile",
        })
        assert effect.effect_type == EffectType.UPDATE_ENTITY
        assert effect.update_entity_id == "kael"
        assert effect.update_disposition == "hostile"
        # Other fields should remain None (= "no change")
        assert effect.update_status is None
        assert effect.update_importance is None
        assert effect.update_description_addition is None

    def test_status_change(self):
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        effect = _convert_tool_call("update_entity", {
            "entity_id": "captain_halloran",
            "status": "dead",
        })
        assert effect.update_status == "dead"
        assert effect.update_disposition is None

    def test_multiple_fields(self):
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        effect = _convert_tool_call("update_entity", {
            "entity_id": "kael",
            "disposition": "hostile",
            "importance": True,
            "description_addition": "cult symbol on robes",
        })
        assert effect.update_disposition == "hostile"
        assert effect.update_importance is True
        assert effect.update_description_addition == "cult symbol on robes"

    def test_disposition_lowercased(self):
        """Brain may emit 'Hostile' — converter normalizes."""
        from dnd_bot.llm.narrator_tools import _convert_tool_call
        effect = _convert_tool_call("update_entity", {
            "entity_id": "x",
            "disposition": "HOSTILE",
        })
        assert effect.update_disposition == "hostile"


# ── update_entity validation: no-op rejection ───────────────────────────


class TestUpdateEntityValidation:
    """The validator rejects no-op update_entity calls so the model gets
    immediate corrective feedback (per prompt-engineering research)."""

    def _validator(self):
        from dnd_bot.llm.effects import EffectValidator
        return EffectValidator()

    def test_no_op_call_rejected(self):
        """update_entity with only entity_id and no change fields = invalid."""
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        effect = ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="kael",
        )
        result = self._validator().validate(effect)
        assert result.valid is False
        assert "no change fields" in result.rejection_reason
        # Rejection message should suggest ref_entity for plain references
        assert "ref_entity" in result.rejection_reason

    def test_missing_entity_id_rejected(self):
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        effect = ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_disposition="hostile",
        )
        result = self._validator().validate(effect)
        assert result.valid is False
        assert "entity_id" in result.rejection_reason

    def test_disposition_change_accepted(self):
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        effect = ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="kael",
            update_disposition="hostile",
        )
        result = self._validator().validate(effect)
        assert result.valid is True

    def test_importance_only_accepted(self):
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        effect = ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="grenn",
            update_importance=True,
        )
        result = self._validator().validate(effect)
        assert result.valid is True

    def test_invalid_disposition_rejected(self):
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        effect = ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="kael",
            update_disposition="grumpy",
        )
        result = self._validator().validate(effect)
        assert result.valid is False
        assert "disposition" in result.rejection_reason

    def test_invalid_status_rejected(self):
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        effect = ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="halloran",
            update_status="zombie",
        )
        result = self._validator().validate(effect)
        assert result.valid is False
        assert "status" in result.rejection_reason


# ── change_location validation ──────────────────────────────────────────


class TestChangeLocationValidation:

    def _validator(self):
        from dnd_bot.llm.effects import EffectValidator
        return EffectValidator()

    def test_short_name_accepted(self):
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        effect = ProposedEffect(
            effect_type=EffectType.CHANGE_LOCATION,
            location_name="the tavern",
        )
        assert self._validator().validate(effect).valid is True

    def test_proper_name_accepted(self):
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        effect = ProposedEffect(
            effect_type=EffectType.CHANGE_LOCATION,
            location_name="Grimstone Hall",
        )
        assert self._validator().validate(effect).valid is True

    def test_missing_name_rejected(self):
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        effect = ProposedEffect(
            effect_type=EffectType.CHANGE_LOCATION,
        )
        result = self._validator().validate(effect)
        assert result.valid is False
        assert "location_name" in result.rejection_reason

    def test_long_sentence_rejected(self):
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        effect = ProposedEffect(
            effect_type=EffectType.CHANGE_LOCATION,
            location_name="a small clearing near the river bend",
        )
        result = self._validator().validate(effect)
        assert result.valid is False
        assert "too long" in result.rejection_reason

    def test_comma_rejected(self):
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        effect = ProposedEffect(
            effect_type=EffectType.CHANGE_LOCATION,
            location_name="behind the bar, inside the tavern",
        )
        result = self._validator().validate(effect)
        assert result.valid is False


# ── Profile loader integration ──────────────────────────────────────────


class TestProfileLoaderToolsField:

    def test_profile_without_tools_field_defaults_to_core(self):
        """Profiles without an explicit tools field get the safe default.

        local_9b is a small dense model where the default 'core' tier is
        the right call — no tools field set in profiles.yaml.
        """
        from dnd_bot.config import load_profile
        try:
            profile = load_profile("local_9b")
        except (ValueError, FileNotFoundError):
            pytest.skip("local_9b profile not in this environment")
        assert profile.narrator.tools == "core"

    def test_deepseek_tiered_profile_uses_full(self):
        """The recommended DeepSeek profile sets tools: full on all narrator slots."""
        from dnd_bot.config import load_profile
        try:
            profile = load_profile("deepseek_tiered")
        except (ValueError, FileNotFoundError):
            pytest.skip("deepseek_tiered profile not in this environment")
        assert profile.narrator.tools == "full"

    def test_tiered_e4b_full_immersion_uses_full(self):
        """The daily-driver tiered profile sets tools: full on its DeepSeek narrators."""
        from dnd_bot.config import load_profile
        try:
            profile = load_profile("tiered_e4b_full_immersion")
        except (ValueError, FileNotFoundError):
            pytest.skip("tiered_e4b_full_immersion profile not in this environment")
        assert profile.narrator.tools == "full"


# ── tool_calls_to_effects rejects invalid no-op silently? ───────────────


class TestToolCallsToEffectsBatch:
    """tool_calls_to_effects converts a batch of tool calls; invalid no-op
    calls become invalid effects that downstream validation will reject."""

    def test_batch_with_update_entity(self):
        from dnd_bot.llm.narrator_tools import tool_calls_to_effects
        from dnd_bot.llm.effects import EffectType

        calls = [
            {"name": "ref_entity", "arguments": {"entity_id": "marta"}},
            {"name": "update_entity", "arguments": {
                "entity_id": "kael",
                "disposition": "hostile",
                "importance": True,
            }},
            {"name": "change_location", "arguments": {
                "location_name": "the ritual chamber",
            }},
        ]
        effects = tool_calls_to_effects(calls)
        assert len(effects) == 3
        types = [e.effect_type for e in effects]
        assert EffectType.REF_ENTITY in types
        assert EffectType.UPDATE_ENTITY in types
        assert EffectType.CHANGE_LOCATION in types


# ── Orchestrator world-state sync (narrator-authoritative) ──────────────


class TestSyncEffectToWorldState:
    """_sync_effect_to_world_state: narrator-declared CHANGE_LOCATION and
    UPDATE_ENTITY mutate WorldState directly, overriding what the state
    extractor produced earlier in the same turn."""

    def _make_orchestrator_with_session(self):
        """Build a minimal orchestrator with just enough plumbing to test
        the sync method in isolation (no LLM, no scene registry, no DB)."""
        from dnd_bot.llm.orchestrator import DMOrchestrator
        from dnd_bot.game.world_state import WorldState
        # Bypass __init__ — we only need the method and a session shim.
        orch = DMOrchestrator.__new__(DMOrchestrator)
        ws = WorldState()

        class _Sess:
            world_state = ws
        orch._current_session = _Sess()
        return orch, ws

    def test_change_location_overrides_world_state(self):
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        orch, ws = self._make_orchestrator_with_session()
        ws.current_location = "the old square"  # extractor's earlier guess

        effect = ProposedEffect(
            effect_type=EffectType.CHANGE_LOCATION,
            location_name="the rusty compass",
            location_description="A warm tavern with a roaring fire.",
        )
        orch._sync_effect_to_world_state(effect)

        assert ws.current_location == "the rusty compass"
        assert "warm tavern" in ws.location_description
        # Previous location becomes a connection (reachable)
        assert "the old square" in ws.connected_locations

    def test_change_location_seeds_when_empty(self):
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        orch, ws = self._make_orchestrator_with_session()
        # No prior location
        effect = ProposedEffect(
            effect_type=EffectType.CHANGE_LOCATION,
            location_name="north gate",
        )
        orch._sync_effect_to_world_state(effect)
        assert ws.current_location == "north gate"
        assert ws.connected_locations == []  # nothing to connect from

    def test_update_entity_overrides_disposition(self):
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        from dnd_bot.game.world_state import NPCState
        orch, ws = self._make_orchestrator_with_session()
        ws.npcs["kael"] = NPCState(name="kael", disposition="friendly")

        effect = ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="kael",
            update_disposition="hostile",
        )
        orch._sync_effect_to_world_state(effect)
        assert ws.npcs["kael"].disposition == "hostile"

    def test_update_entity_status_dead_flips_alive(self):
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        from dnd_bot.game.world_state import NPCState
        orch, ws = self._make_orchestrator_with_session()
        ws.npcs["captain_halloran"] = NPCState(
            name="captain_halloran", alive=True
        )
        effect = ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="captain_halloran",
            update_status="dead",
        )
        orch._sync_effect_to_world_state(effect)
        assert ws.npcs["captain_halloran"].alive is False

    def test_update_entity_importance_promotes(self):
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        from dnd_bot.game.world_state import NPCState
        orch, ws = self._make_orchestrator_with_session()
        ws.npcs["grenn"] = NPCState(name="grenn", important=False)
        effect = ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="grenn",
            update_importance=True,
        )
        orch._sync_effect_to_world_state(effect)
        assert ws.npcs["grenn"].important is True

    def test_update_entity_description_appended(self):
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        from dnd_bot.game.world_state import NPCState
        orch, ws = self._make_orchestrator_with_session()
        ws.npcs["marta"] = NPCState(
            name="marta", description="An old herbalist."
        )
        effect = ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="marta",
            update_description_addition="chipped front teeth",
        )
        orch._sync_effect_to_world_state(effect)
        assert "old herbalist" in ws.npcs["marta"].description
        assert "chipped front teeth" in ws.npcs["marta"].description

    def test_update_entity_unknown_id_no_crash(self):
        """If the narrator references an entity not in world_state, the sync
        is a no-op (it's already executed against scene_registry, and the
        extractor will catch up next turn if it matters)."""
        from dnd_bot.llm.effects import ProposedEffect, EffectType
        orch, ws = self._make_orchestrator_with_session()
        effect = ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="ghost_who_never_existed",
            update_disposition="hostile",
        )
        # Must not raise
        orch._sync_effect_to_world_state(effect)
        assert "ghost_who_never_existed" not in ws.npcs
