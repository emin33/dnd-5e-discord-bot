"""Tests for the narrator tool registry (REFACTOR_PLAN.md Step 1).

The registry is the single authority for tool schemas, tier membership,
and tool→effect converters. Before the strangle, this file pinned the
registry byte-identical to the legacy hardcoded dispatch; the hardcoded
versions are now deleted, so the pins are direct: exact tier composition,
façade-serves-registry, and the executor cross-check that makes a
converter-producible EffectType without an executor row (the audit's
silent remove_entity no-op) impossible to reintroduce.
"""

import pytest

from dnd_bot.llm import tool_registry
from dnd_bot.llm import narrator_tools
from dnd_bot.llm.effects import EffectExecutor, EffectType


# Representative arguments per tool: enough surface to exercise every
# branch of each converter (sanitization, normalization, defaults).
CONVERTER_CASES = {
    "ref_entity": [
        {"entity_id": "marta"},
        {"entity_id": "kael", "alias_used": "the hooded man",
         "dialogue_indices": [1, 3], "dialogue_emotions": ["whispering", "calm"]},
        {},  # missing entity_id → empty-string id
    ],
    "add_npc": [
        {"npc_id": "blacksmith_1", "name": "Korin Ironeye",
         "disposition": "neutral", "gender": "male",
         "description": "A burly dwarven blacksmith"},
        {"npc_id": "g1", "name": "Grom", "disposition": "GRUMPY",  # invalid → neutral
         "description": "A goblin"},
        {"name": "Nameless"},  # no npc_id → source npc:unknown; no gender prefix
    ],
    "spawn_object": [
        {"object_id": "jade_dagger_1", "name": "jade dagger",
         "description": "A small jade dagger"},
        {"name": "iron chest"},  # no id → scene:unknown; no description → None
    ],
    "update_player": [
        {"item_grant": [{"name": "jade dagger", "source": "scene:jade_dagger_1"}]},
        {"item_grant": ["rope"], "currency_delta": {"sp": -12, "bitcoin": 5}},
        {"hp_delta": -6, "damage_type": "poison", "hp_reason": "wall trap dart",
         "add_conditions": ["Poisoned"], "remove_conditions": ["PRONE"]},
        {"hp_delta": 8, "item_remove": [{"name": "potion of healing"}]},
        {"spell_slot_used": 2},
        {},  # no-op call — converter still returns an effect
    ],
    "request_roll": [
        {"target": "player", "roll_type": "save", "ability_or_skill": "constitution",
         "dc": 14, "reason": "resisting the gas"},
        {"target": "player", "roll_type": "check", "ability_or_skill": "strength",
         "dc": 15, "reason": "forcing the door"},
        {"target": "player", "roll_type": "skill", "ability_or_skill": "perception",
         "dc": 12, "reason": "spotting the tripwire"},
        {},  # all defaults
    ],
    "start_combat": [
        {"reason": "The bandit drew his cutlass."},
        {},  # default reason
    ],
    "change_location": [
        {"location_name": "the rusty compass", "description": "A warm tavern."},
        {"location_name": "  north gate  "},
        {},  # missing name → None
    ],
    "update_entity": [
        {"entity_id": "kael", "disposition": "HOSTILE", "importance": True},
        {"entity_id": "captain_halloran", "status": "dead"},
        {"entity_id": "marta", "add_items": [" Herbal Potion ", ""],
         "remove_items": ["Cutlass"]},
        {"entity_id": "  ", "description_addition": ""},  # blanks → None
    ],
}


# Exact tier composition, in canonical schema order. These pins replaced the
# legacy-comparison tests when the hardcoded NARRATOR_TOOLS list was deleted —
# any change here is a deliberate tool-surface change, not drift.
EXPECTED_TIERS = {
    "core": ["ref_entity", "add_npc", "spawn_object"],
    "core_plus": ["ref_entity", "add_npc", "spawn_object",
                  "start_combat", "change_location"],
    "full": ["ref_entity", "add_npc", "spawn_object", "update_player",
             "request_roll", "start_combat", "change_location",
             "update_entity"],
}


class TestTierComposition:
    """Exact per-tier tool lists — the registry is the single truth."""

    @pytest.mark.parametrize("tier", sorted(EXPECTED_TIERS))
    def test_tier_composition_pinned(self, tier):
        tools = tool_registry.tools_for_tier(tier)
        assert [t["function"]["name"] for t in tools] == EXPECTED_TIERS[tier]

    def test_full_tier_covers_every_registered_tool(self):
        full = tool_registry.tools_for_tier("full")
        assert [t["function"]["name"] for t in full] == \
            [spec.name for spec in tool_registry.all_specs()]

    def test_unknown_tier_returns_none(self):
        assert tool_registry.tools_for_tier("super_full_max") is None
        assert tool_registry.tools_for_tier("") is None


class TestFacadeServesRegistry:
    """narrator_tools is a thin read over the registry — same objects."""

    @pytest.mark.parametrize("tier", sorted(EXPECTED_TIERS))
    def test_tier_map_is_registry_derived(self, tier):
        assert narrator_tools.NARRATOR_TOOL_TIERS[tier] == \
            tool_registry.tools_for_tier(tier)
        assert narrator_tools.get_narrator_tools_for_tier(tier) == \
            tool_registry.tools_for_tier(tier)

    def test_convert_delegates_to_registry(self):
        assert narrator_tools._convert_tool_call is tool_registry.convert_tool_call

    def test_unknown_tool_returns_none(self):
        assert tool_registry.convert_tool_call("no_such_tool", {}) is None

    def test_converter_cases_cover_every_registered_tool(self):
        """Meta: if a tool is added to the registry without representative
        cases, fail here instead of silently skipping it."""
        assert set(CONVERTER_CASES) == {s.name for s in tool_registry.all_specs()}


class TestExecutorCrossCheck:
    """Every EffectType a registered converter can emit must have an
    EffectExecutor handler row — the registration drift that made
    remove_entity a silent end-to-end no-op (audit Duplication P0)."""

    def test_every_emittable_type_has_an_executor(self):
        handled = EffectExecutor().handled_effect_types()
        missing = tool_registry.emittable_effect_types() - handled
        assert not missing, (
            f"converter-producible effect types with NO executor row: "
            f"{sorted(t.value for t in missing)}"
        )


class TestSpecDeclarations:
    """Structural invariants on every registered spec."""

    def test_schema_shape(self):
        for spec in tool_registry.all_specs():
            assert spec.schema["type"] == "function"
            fn = spec.schema["function"]
            assert fn["name"] == spec.name
            assert fn["description"]
            assert fn["parameters"]["type"] == "object"

    def test_tiers_known_and_nonempty(self):
        for spec in tool_registry.all_specs():
            assert spec.tiers, f"{spec.name} in no tier"
            assert set(spec.tiers) <= set(tool_registry.KNOWN_TIERS)

    def test_effect_types_declared_and_emitted(self):
        """Each converter's declared effect_types match what it actually
        emits for the representative cases."""
        for spec in tool_registry.all_specs():
            assert spec.effect_types, f"{spec.name} declares no effect types"
            for args in CONVERTER_CASES[spec.name]:
                effect = spec.converter(dict(args))
                assert effect is not None
                assert effect.effect_type in spec.effect_types, (
                    f"{spec.name} emitted undeclared {effect.effect_type}"
                )

    def test_duplicate_registration_rejected(self):
        spec = tool_registry.get_spec("ref_entity")
        with pytest.raises(ValueError, match="already registered"):
            tool_registry.register(spec)

    def test_emittable_effect_types_is_union(self):
        expected = set()
        for spec in tool_registry.all_specs():
            expected.update(spec.effect_types)
        assert tool_registry.emittable_effect_types() == expected
        assert EffectType.REF_ENTITY in expected
