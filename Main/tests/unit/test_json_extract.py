"""Tests for the shared LLM JSON extraction helper (llm/json_extract.py).

This helper is the single implementation of the fence-strip + brace-extract
recovery routine previously copy-pasted across the triage parser, the
state/entity extractors, and the dedup judge.
"""

from dnd_bot.llm.json_extract import extract_json_object


# ── Clean input ──────────────────────────────────────────────────────────


class TestCleanInput:
    def test_bare_json_object(self):
        data, warnings = extract_json_object('{"action_type": "attack", "needs_roll": true}')
        assert data == {"action_type": "attack", "needs_roll": True}
        assert warnings == []

    def test_whitespace_around_object(self):
        data, warnings = extract_json_object('   \n {"a": 1} \n  ')
        assert data == {"a": 1}
        assert warnings == []

    def test_nested_object(self):
        raw = '{"outer": {"inner": {"deep": [1, 2, {"x": null}]}}, "b": "y"}'
        data, warnings = extract_json_object(raw)
        assert data == {"outer": {"inner": {"deep": [1, 2, {"x": None}]}}, "b": "y"}
        assert warnings == []


# ── Fenced input ─────────────────────────────────────────────────────────


class TestFencedInput:
    def test_json_fence(self):
        raw = '```json\n{"action": "accept"}\n```'
        data, warnings = extract_json_object(raw)
        assert data == {"action": "accept"}
        assert "stripped_markdown_fence" in warnings

    def test_bare_fence(self):
        raw = '```\n{"action": "rewrite", "target_id": "npc-1"}\n```'
        data, warnings = extract_json_object(raw)
        assert data == {"action": "rewrite", "target_id": "npc-1"}
        assert "stripped_code_fence" in warnings

    def test_fence_with_surrounding_prose(self):
        raw = 'Here is my answer:\n```json\n{"needs_roll": false}\n```\nHope that helps!'
        data, warnings = extract_json_object(raw)
        assert data == {"needs_roll": False}
        assert "stripped_markdown_fence" in warnings

    def test_nested_object_inside_fence(self):
        raw = '```json\n{"npc_updates": [{"name": "Bram", "changes": {"hp": 4}}]}\n```'
        data, warnings = extract_json_object(raw)
        assert data == {"npc_updates": [{"name": "Bram", "changes": {"hp": 4}}]}
        assert "stripped_markdown_fence" in warnings


# ── Prose-wrapped (unfenced) input ───────────────────────────────────────


class TestProseWrappedInput:
    def test_object_extracted_from_leading_prose(self):
        raw = 'Sure! The triage result is {"action_type": "roleplay"} as requested.'
        data, warnings = extract_json_object(raw)
        assert data == {"action_type": "roleplay"}
        assert "extracted_json_from_text" in warnings

    def test_object_with_trailing_text_only(self):
        raw = '{"a": 1} and some trailing commentary'
        data, warnings = extract_json_object(raw)
        assert data == {"a": 1}
        assert "extracted_json_from_text" in warnings

    def test_nested_braces_use_outermost_span(self):
        raw = 'prefix {"outer": {"inner": 1}} suffix'
        data, warnings = extract_json_object(raw)
        assert data == {"outer": {"inner": 1}}
        assert "extracted_json_from_text" in warnings


# ── Malformed input ──────────────────────────────────────────────────────


class TestMalformedInput:
    def test_empty_string(self):
        data, warnings = extract_json_object("")
        assert data is None
        assert "empty_content" in warnings

    def test_whitespace_only(self):
        data, warnings = extract_json_object("   \n\t ")
        assert data is None
        assert "empty_content" in warnings

    def test_no_json_at_all(self):
        data, warnings = extract_json_object("The goblin sneers and reaches for its blade.")
        assert data is None
        assert "no_json_object_found" in warnings

    def test_truncated_object(self):
        data, warnings = extract_json_object('{"action_type": "attack", "needs_roll"')
        assert data is None
        assert any(w.startswith("json_parse_failed") for w in warnings)

    def test_invalid_json_inside_fence(self):
        raw = "```json\n{not valid json}\n```"
        data, warnings = extract_json_object(raw)
        assert data is None
        assert "stripped_markdown_fence" in warnings
        assert any(w.startswith("json_parse_failed") for w in warnings)

    def test_top_level_array_rejected(self):
        data, warnings = extract_json_object('["not", "an", "object"]')
        assert data is None
        # An array has no '{'... unless items are objects; bare array of
        # strings never reaches json.loads because there is no brace.
        assert "no_json_object_found" in warnings

    def test_none_like_scalar_rejected(self):
        # Braces present but the outermost span parses to a non-dict is not
        # reachable via brace-extraction; guard against dict-typed contract
        # regressions with a scalar wrapped in braces-free content.
        data, warnings = extract_json_object("null")
        assert data is None
        assert "no_json_object_found" in warnings

    def test_warnings_do_not_mask_success(self):
        # A recovered parse still returns data alongside its warnings.
        raw = '```json\n {"ok": true} \n``` trailing'
        data, warnings = extract_json_object(raw)
        assert data == {"ok": True}
        assert warnings  # non-empty: recovery steps were taken
