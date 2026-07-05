"""Tests for the narrator tool registry (REFACTOR_PLAN.md Step 1).

Phase 1 (this file's first job): pin the registry byte-identical to the
legacy hardcoded dispatch in narrator_tools.py — schema lists per tier and
converter outputs — BEFORE the hardcoded versions are deleted. When the
strangle lands, the legacy-comparison tests are rewritten into direct pins.
"""

import pytest

from dnd_bot.llm import tool_registry
from dnd_bot.llm import narrator_tools
from dnd_bot.llm.effects import EffectType


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


class TestRegistryMatchesLegacyDispatch:
    """The registry must reproduce the hardcoded narrator_tools dispatch
    exactly — these comparisons guard the strangler window."""

    def test_tier_schema_lists_identical(self):
        for tier in tool_registry.KNOWN_TIERS:
            assert tool_registry.tools_for_tier(tier) == \
                narrator_tools.NARRATOR_TOOL_TIERS[tier], f"tier '{tier}' drifted"

    def test_full_tier_covers_every_registered_tool(self):
        full = tool_registry.tools_for_tier("full")
        assert [t["function"]["name"] for t in full] == \
            [spec.name for spec in tool_registry.all_specs()]

    def test_unknown_tier_returns_none(self):
        assert tool_registry.tools_for_tier("super_full_max") is None
        assert tool_registry.tools_for_tier("") is None

    @pytest.mark.parametrize("tool_name", sorted(CONVERTER_CASES))
    def test_converter_outputs_identical(self, tool_name):
        for args in CONVERTER_CASES[tool_name]:
            legacy = narrator_tools._convert_tool_call(tool_name, dict(args))
            via_registry = tool_registry.convert_tool_call(tool_name, dict(args))
            assert legacy is not None and via_registry is not None
            assert via_registry.model_dump() == legacy.model_dump(), (
                f"{tool_name} converter drifted for args {args!r}"
            )

    def test_unknown_tool_returns_none_both(self):
        assert narrator_tools._convert_tool_call("no_such_tool", {}) is None
        assert tool_registry.convert_tool_call("no_such_tool", {}) is None

    def test_every_legacy_tool_is_registered(self):
        legacy_names = [t["function"]["name"] for t in narrator_tools.NARRATOR_TOOLS]
        registry_names = [spec.name for spec in tool_registry.all_specs()]
        assert registry_names == legacy_names

    def test_converter_cases_cover_every_registered_tool(self):
        """Meta: if a tool is added to the registry without comparison cases,
        fail here instead of silently skipping it."""
        assert set(CONVERTER_CASES) == {s.name for s in tool_registry.all_specs()}


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
