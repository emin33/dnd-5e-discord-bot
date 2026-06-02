"""Unit tests for `dnd_bot.llm.client`.

Focused on pure-function helpers that don't require running an LLM:
- Hermes block injection gating (audit #95)
- LLMResponse cache_hit_ratio property
"""

from __future__ import annotations

import pytest


def _sample_tools():
    return [
        {
            "type": "function",
            "function": {
                "name": "ref_entity",
                "description": "Reference an existing entity",
                "parameters": {"type": "object"},
            },
        }
    ]


class TestOllamaCompatMessageBuilding:
    """Audit #95: native-tool models (Gemma 4) shouldn't receive the Qwen3 Hermes block.

    Gemma 4 has Ollama RENDERER/PARSER directives that wrap tool calls with
    hard token boundaries (<|tool|>, <|tool_call|>, <|tool_result|>). Injecting
    a Hermes-format XML <tools> block — required for Qwen3 because its template
    auto-injection was broken — either confuses Gemma into emitting Hermes XML
    instead of using the special tokens, or wastes context.
    """

    def test_gemma4_skips_hermes_injection(self):
        from dnd_bot.llm.client import OllamaClient
        messages = [{"role": "user", "content": "Hello"}]
        out, uses_native = OllamaClient._build_compat_messages(
            "gemma4:e2b", messages, _sample_tools(),
        )
        assert uses_native is True
        # Passthrough — no Hermes block prepended.
        assert out == messages
        # Defensive: no system message at all should be injected.
        assert not any(
            m.get("role") == "system" and "<tools>" in m.get("content", "")
            for m in out
        )

    def test_gemma4_26b_also_skips(self):
        """Prefix match covers all gemma4 variants (e2b, e4b, 26b, etc.)."""
        from dnd_bot.llm.client import OllamaClient
        messages = [{"role": "user", "content": "Hello"}]
        out, uses_native = OllamaClient._build_compat_messages(
            "gemma4:26b", messages, _sample_tools(),
        )
        assert uses_native is True
        assert out == messages

    def test_qwen3_still_gets_hermes_injection(self):
        from dnd_bot.llm.client import OllamaClient
        messages = [{"role": "user", "content": "Hello"}]
        out, uses_native = OllamaClient._build_compat_messages(
            "qwen3.6:latest", messages, _sample_tools(),
        )
        assert uses_native is False
        # Hermes block prepended as a system message before the first user msg
        assert out[0]["role"] == "system"
        assert "<tools>" in out[0]["content"]
        assert "<tool_call>" in out[0]["content"]
        assert out[1] == messages[0]

    def test_unknown_model_falls_back_to_hermes(self):
        """Anything not in the allowlist gets the Hermes injection (safe default)."""
        from dnd_bot.llm.client import OllamaClient
        messages = [{"role": "user", "content": "Hello"}]
        out, uses_native = OllamaClient._build_compat_messages(
            "some-future-model:8b", messages, _sample_tools(),
        )
        assert uses_native is False
        assert "<tools>" in out[0]["content"]

    def test_hermes_block_inserted_before_first_user_message(self):
        """When there's a system message before the user, Hermes goes BETWEEN them."""
        from dnd_bot.llm.client import OllamaClient
        messages = [
            {"role": "system", "content": "You are a brain."},
            {"role": "user", "content": "Hello"},
        ]
        out, _ = OllamaClient._build_compat_messages(
            "qwen3.6:latest", messages, _sample_tools(),
        )
        # Pre-existing system message preserved at index 0
        assert out[0] == messages[0]
        # Hermes block injected right before the user message
        assert out[1]["role"] == "system"
        assert "<tools>" in out[1]["content"]
        assert out[2] == messages[1]

    def test_no_user_message_appends_hermes(self):
        """Edge case: no user message — Hermes block lands at the end."""
        from dnd_bot.llm.client import OllamaClient
        messages = [{"role": "system", "content": "Initial"}]
        out, _ = OllamaClient._build_compat_messages(
            "qwen3.6:latest", messages, _sample_tools(),
        )
        assert out[0] == messages[0]
        assert out[-1]["role"] == "system"
        assert "<tools>" in out[-1]["content"]


class TestLLMResponseCacheRatio:
    """Audit #21: LLMResponse exposes cache_hit_ratio for cost-tracking."""

    def test_no_cache_returns_zero(self):
        from dnd_bot.llm.client import LLMResponse
        r = LLMResponse(content="hi", prompt_tokens=100)
        assert r.cache_hit_ratio == 0.0

    def test_full_cache_returns_one(self):
        from dnd_bot.llm.client import LLMResponse
        r = LLMResponse(content="hi", prompt_tokens=0, cache_read_tokens=100)
        assert r.cache_hit_ratio == 1.0

    def test_partial_cache(self):
        from dnd_bot.llm.client import LLMResponse
        r = LLMResponse(content="hi", prompt_tokens=20, cache_read_tokens=80)
        assert r.cache_hit_ratio == 0.8

    def test_empty_response_safe(self):
        from dnd_bot.llm.client import LLMResponse
        r = LLMResponse(content="")
        assert r.cache_hit_ratio == 0.0
