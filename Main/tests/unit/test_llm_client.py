"""Unit tests for `dnd_bot.llm.client`.

Focused on pure-function helpers that don't require running an LLM:
- Hermes block injection gating (audit #95)
- LLMResponse cache_hit_ratio property
- Structured-output enforcement per provider (R4) via injected fakes
- Executor/HTTP-timeout hygiene (AQ-ASYNC-08)

No test here performs network I/O: provider SDK clients are replaced with
recording fakes before ``chat()`` is called.
"""

from __future__ import annotations

import json
from types import SimpleNamespace

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


# ── R4: structured output enforcement ────────────────────────────────────────


_SCHEMA = {
    "type": "object",
    "properties": {"action_type": {"type": "string"}},
    "required": ["action_type"],
}


class _Block:
    """Minimal Anthropic content block: only the attributes it's given."""

    def __init__(self, **kw):
        self.__dict__.update(kw)


class _RecordingAnthropicMessages:
    def __init__(self, response):
        self._response = response
        self.last_kwargs = None

    async def create(self, **kwargs):
        self.last_kwargs = kwargs
        return self._response


def _anthropic_response(blocks):
    return SimpleNamespace(
        content=blocks,
        model="claude-test",
        stop_reason="tool_use",
        usage=SimpleNamespace(input_tokens=10, output_tokens=5),
    )


def _make_anthropic_client(blocks):
    from dnd_bot.llm.client import AnthropicClient

    client = AnthropicClient(model="claude-test", api_key="test-key")
    fake = _RecordingAnthropicMessages(_anthropic_response(blocks))
    client._client = SimpleNamespace(messages=fake)
    return client, fake


class TestAnthropicStructuredOutput:
    """R4: json_schema must be enforced via a forced tool, not a prose hint."""

    async def test_json_schema_forces_tool_choice(self):
        tool_block = _Block(
            type="tool_use",
            name="emit_structured_json",
            input={"action_type": "roleplay"},
        )
        client, fake = _make_anthropic_client([tool_block])

        result = await client.chat(
            messages=[{"role": "user", "content": "hi"}],
            json_schema=_SCHEMA,
        )

        kwargs = fake.last_kwargs
        # The request carries a single forced tool whose input_schema IS the schema
        assert kwargs["tool_choice"] == {
            "type": "tool",
            "name": "emit_structured_json",
        }
        assert len(kwargs["tools"]) == 1
        assert kwargs["tools"][0]["name"] == "emit_structured_json"
        assert kwargs["tools"][0]["input_schema"] is _SCHEMA
        # No prose-hint fallback when the tool is forced
        assert "ONLY a valid JSON" not in str(kwargs.get("system", ""))
        # The tool_use input round-trips as JSON text in content
        assert json.loads(result.content) == {"action_type": "roleplay"}
        # The synthetic tool call is not surfaced to callers
        assert result.tool_calls == []

    async def test_json_schema_with_caller_tools_keeps_hint_fallback(self):
        text_block = _Block(type="text", text='{"action_type": "attack"}')
        client, fake = _make_anthropic_client([text_block])

        result = await client.chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=_sample_tools(),
            json_schema=_SCHEMA,
            tool_choice="auto",
        )

        kwargs = fake.last_kwargs
        # Caller tools win — no synthetic tool injected alongside them
        assert [t["name"] for t in kwargs["tools"]] == ["ref_entity"]
        assert kwargs["tool_choice"] == {"type": "auto"}
        # Falls back to the system hint
        assert "ONLY a valid JSON" in str(kwargs["system"])
        assert result.content == '{"action_type": "attack"}'

    async def test_json_mode_only_uses_hint(self):
        text_block = _Block(type="text", text="{}")
        client, fake = _make_anthropic_client([text_block])

        await client.chat(
            messages=[{"role": "user", "content": "hi"}],
            json_mode=True,
        )

        kwargs = fake.last_kwargs
        assert "tools" not in kwargs
        assert "ONLY a valid JSON" in str(kwargs["system"])


def _fake_genai_client(captured: dict, text: str = "{}", usage_metadata=None):
    """Fake google.genai Client exposing aio.models.generate_content."""
    if usage_metadata is None:
        usage_metadata = SimpleNamespace(
            prompt_token_count=1, candidates_token_count=1,
        )

    async def _generate_content(**call_kwargs):
        captured.update(call_kwargs)
        part = SimpleNamespace(text=text, function_call=None)
        candidate = SimpleNamespace(
            content=SimpleNamespace(parts=[part]),
            finish_reason=SimpleNamespace(name="STOP"),
        )
        return SimpleNamespace(
            candidates=[candidate], usage_metadata=usage_metadata,
        )

    return SimpleNamespace(
        aio=SimpleNamespace(
            models=SimpleNamespace(generate_content=_generate_content)
        )
    )


class TestGeminiStructuredOutput:
    """R4: json_schema must be wired through as response_schema."""

    async def test_json_schema_sets_response_schema(self):
        from dnd_bot.llm.client import GeminiClient

        client = GeminiClient(model="gemini-test", api_key="test-key")
        captured: dict = {}
        client._client = _fake_genai_client(
            captured, text='{"action_type": "roleplay"}'
        )

        result = await client.chat(
            messages=[{"role": "user", "content": "hi"}],
            json_schema=_SCHEMA,
        )

        config = captured["config"]
        assert config.response_mime_type == "application/json"
        # The caller schema is normalized and passed through — reverting the
        # fix leaves response_schema unset and this fails.
        schema = config.response_schema
        assert schema is not None
        assert schema["type"] == "object"
        assert "action_type" in schema["properties"]
        assert list(schema["required"]) == ["action_type"]
        assert json.loads(result.content) == {"action_type": "roleplay"}

    async def test_json_mode_only_omits_response_schema(self):
        from dnd_bot.llm.client import GeminiClient

        client = GeminiClient(model="gemini-test", api_key="test-key")
        captured: dict = {}
        client._client = _fake_genai_client(captured)

        await client.chat(
            messages=[{"role": "user", "content": "hi"}],
            json_mode=True,
        )

        config = captured["config"]
        assert config.response_mime_type == "application/json"
        assert config.response_schema is None

    async def test_think_false_zeroes_thinking_budget(self):
        """2.5 models think by default and thinking spends max_output_tokens;
        think=False must disable it or small JSON calls come back truncated."""
        from dnd_bot.llm.client import GeminiClient

        client = GeminiClient(model="gemini-test", api_key="test-key")
        captured: dict = {}
        client._client = _fake_genai_client(captured)

        await client.chat(
            messages=[{"role": "user", "content": "hi"}],
            think=False,
        )
        assert captured["config"].thinking_config.thinking_budget == 0

        captured.clear()
        await client.chat(messages=[{"role": "user", "content": "hi"}])
        assert captured["config"].thinking_config is None

    async def test_tools_use_json_schema_declarations(self):
        """OpenAI-format tools must survive as parameters_json_schema."""
        from dnd_bot.llm.client import GeminiClient

        client = GeminiClient(model="gemini-test", api_key="test-key")
        captured: dict = {}
        client._client = _fake_genai_client(captured)

        await client.chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=[{
                "type": "function",
                "function": {
                    "name": "ref_entity",
                    "description": "Reference a roster entity",
                    "parameters": {
                        "type": "object",
                        "properties": {"entity_id": {"type": "string"}},
                        "required": ["entity_id"],
                    },
                },
            }],
            tool_choice="required",
        )

        config = captured["config"]
        declaration = config.tools[0].function_declarations[0]
        assert declaration.name == "ref_entity"
        assert declaration.parameters_json_schema["required"] == ["entity_id"]
        assert config.tool_config.function_calling_config.mode == "ANY"
        # Declarations are dicts — automatic calling must be disabled so
        # function_call parts come back to the orchestrator.
        assert config.automatic_function_calling.disable is True


class TestGeminiSchemaNormalization:
    """The response_schema wiring must survive real Pydantic schemas.

    Pydantic's ``model_json_schema()`` emits ``$ref``/``$defs`` (nested
    models) and ``anyOf: [X, null]`` (Optional fields). The naive
    ``_convert_schema`` collapsed every such node to a bare OBJECT proto
    with empty properties — which the Gemini API rejects with 400
    INVALID_ARGUMENT — so the R4 enforcement was broken for every schema
    the codebase actually passes. These pin the normalization pass:
    refs inlined, Optional folded into nullable, and inexpressible
    schemas (free-form dicts) falling back to mime-type-only.
    """

    @staticmethod
    def _assert_no_empty_objects(schema: dict):
        """Walk a normalized schema: every object node must carry properties."""
        if schema.get("type") == "object":
            assert schema.get("properties"), (
                "object node with empty properties survived normalization"
            )
            for sub in schema["properties"].values():
                TestGeminiSchemaNormalization._assert_no_empty_objects(sub)
        elif schema.get("type") == "array":
            TestGeminiSchemaNormalization._assert_no_empty_objects(
                schema["items"]
            )

    def test_extraction_schema_round_trips_without_empty_objects(self):
        from dnd_bot.llm.client import GeminiClient
        from dnd_bot.llm.extractors.entity_extractor import get_extraction_schema

        client = GeminiClient(model="gemini-test", api_key="test-key")
        normalized = client._normalize_response_schema(get_extraction_schema())
        # $defs/anyOf all resolved — the schema is fully expressible.
        assert normalized is not None
        self._assert_no_empty_objects(normalized)
        # Nested-model refs were inlined, not destroyed: the entities
        # items carry ExtractedEntity's real properties.
        entity_items = normalized["properties"]["entities"]["items"]
        assert entity_items["type"] == "object"
        assert "name" in entity_items["properties"]
        # Optional[str] became a nullable string, not an empty object.
        monster = entity_items["properties"]["monster_index"]
        assert monster["type"] == "string"
        assert monster["nullable"] is True

    def test_summary_schema_round_trips(self):
        from dnd_bot.llm.client import GeminiClient

        client = GeminiClient(model="gemini-test", api_key="test-key")
        # The memory-summary schema (memory/manager.py) is hand-written
        # and plain — full enforcement applies.
        summary_schema = {
            "type": "object",
            "properties": {
                "summary": {"type": "string"},
                "key_events": {"type": "array", "items": {"type": "string"}},
            },
            "required": ["summary"],
        }
        normalized = client._normalize_response_schema(summary_schema)
        assert normalized is not None
        self._assert_no_empty_objects(normalized)

    def test_triage_schema_falls_back(self):
        """TriageSchema carries free-form dicts (``currency_spent: dict``,
        ``resources_consumed: list[dict]``) — inexpressible, so it must
        take the mime-type-only fallback, not an empty OBJECT proto."""
        from dnd_bot.llm.client import GeminiClient
        from dnd_bot.llm.orchestrator import get_triage_schema

        client = GeminiClient(model="gemini-test", api_key="test-key")
        assert client._normalize_response_schema(get_triage_schema()) is None

    async def test_inexpressible_schema_falls_back_to_mime_type_only(self):
        """StateDelta carries ``flag_changes: dict[str, bool]`` — a
        free-form object the schema normalization cannot express. The chat
        path must drop response_schema (mime-type-only behavior), not send
        an empty object node the API rejects."""
        from dnd_bot.game.world_state import get_state_delta_schema
        from dnd_bot.llm.client import GeminiClient

        client = GeminiClient(model="gemini-test", api_key="test-key")
        assert client._normalize_response_schema(get_state_delta_schema()) is None

        captured: dict = {}
        client._client = _fake_genai_client(captured)

        await client.chat(
            messages=[{"role": "user", "content": "hi"}],
            json_schema=get_state_delta_schema(),
        )

        config = captured["config"]
        assert config.response_mime_type == "application/json"
        assert config.response_schema is None


class TestOllamaCompatStructuredOutput:
    """R4: tool-bearing compat path must forward json_schema/json_mode/think."""

    @staticmethod
    def _make_client_with_fake_compat():
        from dnd_bot.llm.client import OllamaClient

        client = OllamaClient(model="qwen-test")
        captured: dict = {}

        class _FakeCompletions:
            async def create(self, **kwargs):
                captured.update(kwargs)
                message = SimpleNamespace(content="{}", tool_calls=None)
                choice = SimpleNamespace(message=message, finish_reason="stop")
                return SimpleNamespace(
                    choices=[choice],
                    model="qwen-test",
                    usage=SimpleNamespace(prompt_tokens=1, completion_tokens=1),
                )

        fake_compat = SimpleNamespace(
            chat=SimpleNamespace(completions=_FakeCompletions())
        )
        client._openai_compat_client = fake_compat
        return client, captured

    async def test_json_schema_and_think_forwarded(self):
        client, captured = self._make_client_with_fake_compat()

        await client.chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=_sample_tools(),
            json_schema=_SCHEMA,
            think=False,
        )

        assert captured["response_format"] == {
            "type": "json_schema",
            "json_schema": {"name": "response", "schema": _SCHEMA},
        }
        assert captured["extra_body"]["think"] is False

    async def test_json_mode_forwarded(self):
        client, captured = self._make_client_with_fake_compat()

        await client.chat(
            messages=[{"role": "user", "content": "hi"}],
            tools=_sample_tools(),
            json_mode=True,
        )

        assert captured["response_format"] == {"type": "json_object"}
        # think unset -> explicitly DISABLED, not omitted. Omitting it left
        # thinking on, and a thinking model (qwen3.5) returns an empty
        # visible response — on this, the tool-bearing path, that means no
        # tool calls at all. Opt in with think=True.
        assert captured["extra_body"]["think"] is False


# ── AQ-ASYNC-08: executor hygiene ────────────────────────────────────────────


class _CloseCounter:
    def __init__(self):
        self.closed = 0

    def close(self):
        self.closed += 1


class TestExecutorHygiene:
    def test_ollama_http_client_has_timeout_above_wait_for_deadline(self):
        """The httpx client must time out slightly after asyncio.wait_for does,
        so an abandoned executor thread unblocks instead of wedging the pool."""
        from dnd_bot.llm.client import OllamaClient

        client = OllamaClient(model="qwen-test")
        httpx_timeout = client._client._client.timeout
        assert httpx_timeout.read == client.timeout + 10
        assert httpx_timeout.connect == client.timeout + 10

    def test_reset_clients_closes_cached_clients_once(self, monkeypatch):
        """_reset_clients must close() old clients (executor leak per /profile
        switch), deduping shared instances and tolerating close-less clients."""
        from dnd_bot.llm import client as client_mod

        shared = _CloseCounter()
        brain_only = _CloseCounter()
        no_close = object()  # e.g. a client without a close() method

        monkeypatch.setattr(client_mod, "_client", brain_only)
        monkeypatch.setattr(client_mod, "_narrator_client", shared)
        monkeypatch.setattr(
            client_mod, "_narrator_clients_by_tier", {"standard": shared},
        )
        monkeypatch.setattr(
            client_mod,
            "_clients_by_provider_model",
            {("ollama", "a"): shared, ("x", "y"): no_close},
        )

        client_mod._reset_clients()

        assert brain_only.closed == 1
        assert shared.closed == 1  # closed once despite three cache slots
        assert client_mod._client is None
        assert client_mod._narrator_client is None
        assert client_mod._narrator_clients_by_tier == {}
        assert client_mod._clients_by_provider_model == {}

    def test_groq_close_closes_ollama_fallback(self):
        from dnd_bot.llm.client import GroqClient

        client = GroqClient.__new__(GroqClient)  # skip __init__ (needs API key)
        fallback = _CloseCounter()
        client._ollama_fallback = fallback
        client.close()
        assert fallback.closed == 1

    def test_groq_close_without_fallback_is_noop(self):
        from dnd_bot.llm.client import GroqClient

        client = GroqClient.__new__(GroqClient)
        client.close()  # must not raise

    async def test_chat_after_close_recreates_executor(self):
        """A /profile switch closes cached clients while a turn may still
        be in flight — switch_profile's contract says in-progress turns
        finish with their existing clients. chat() after close() must
        recreate the pool instead of dying with 'cannot schedule new
        futures after shutdown'."""
        from dnd_bot.llm.client import OllamaClient

        client = OllamaClient(model="qwen-test")
        client._client = SimpleNamespace(
            chat=lambda **kwargs: {"message": {"content": "still alive"}},
        )
        client.close()
        old_executor = client._executor

        result = await client.chat(messages=[{"role": "user", "content": "hi"}])

        assert "still alive" in (result.content or "")
        assert client._executor is not old_executor
        client.close()  # the replacement pool is closeable too


# ── Cache-read telemetry fills (usage-ledger sweep) ──────────────────────────


class TestProviderCacheTokens:
    """Groq/OpenRouter surface OpenAI-shape ``prompt_tokens_details.cached_tokens``;
    Gemini surfaces ``cached_content_token_count``. For all three,
    ``prompt_tokens`` INCLUDES the cached slice (see LLMResponse comment).
    Absent/None fields must default to 0."""

    @staticmethod
    def _openai_response(usage):
        message = SimpleNamespace(content="hi", tool_calls=None)
        choice = SimpleNamespace(message=message, finish_reason="stop")
        return SimpleNamespace(choices=[choice], model="test-model", usage=usage)

    @staticmethod
    def _fake_openai_client(response):
        class _FakeCompletions:
            async def create(self, **kwargs):
                return response

        return SimpleNamespace(chat=SimpleNamespace(completions=_FakeCompletions()))

    async def test_groq_cached_tokens_surfaced(self):
        from dnd_bot.llm.client import GroqClient

        usage = SimpleNamespace(
            prompt_tokens=100,
            completion_tokens=5,
            prompt_tokens_details=SimpleNamespace(cached_tokens=64),
        )
        client = GroqClient(model="groq-test", api_key="test-key")
        client._client = self._fake_openai_client(self._openai_response(usage))

        resp = await client.chat(messages=[{"role": "user", "content": "hi"}])
        assert resp.prompt_tokens == 100  # includes the cached slice
        assert resp.cache_read_tokens == 64

    async def test_groq_missing_details_defaults_to_zero(self):
        from dnd_bot.llm.client import GroqClient

        usage = SimpleNamespace(prompt_tokens=10, completion_tokens=2)
        client = GroqClient(model="groq-test", api_key="test-key")
        client._client = self._fake_openai_client(self._openai_response(usage))

        resp = await client.chat(messages=[{"role": "user", "content": "hi"}])
        assert resp.cache_read_tokens == 0

    async def test_openrouter_cached_tokens_surfaced(self):
        from dnd_bot.llm.client import OpenRouterClient

        usage = SimpleNamespace(
            prompt_tokens=50,
            completion_tokens=5,
            prompt_tokens_details=SimpleNamespace(cached_tokens=32),
        )
        client = OpenRouterClient(model="or-test", api_key="test-key")
        client._client = self._fake_openai_client(self._openai_response(usage))

        resp = await client.chat(messages=[{"role": "user", "content": "hi"}])
        assert resp.cache_read_tokens == 32

    async def test_openrouter_none_cached_tokens_defaults_to_zero(self):
        from dnd_bot.llm.client import OpenRouterClient

        usage = SimpleNamespace(
            prompt_tokens=50,
            completion_tokens=5,
            prompt_tokens_details=SimpleNamespace(cached_tokens=None),
        )
        client = OpenRouterClient(model="or-test", api_key="test-key")
        client._client = self._fake_openai_client(self._openai_response(usage))

        resp = await client.chat(messages=[{"role": "user", "content": "hi"}])
        assert resp.cache_read_tokens == 0

    async def test_gemini_cached_content_tokens_surfaced(self):
        from dnd_bot.llm.client import GeminiClient

        client = GeminiClient(model="gemini-test", api_key="test-key")
        client._client = _fake_genai_client(
            {},
            text="hi",
            usage_metadata=SimpleNamespace(
                prompt_token_count=100,
                candidates_token_count=5,
                cached_content_token_count=37,
            ),
        )

        resp = await client.chat(messages=[{"role": "user", "content": "hi"}])
        assert resp.prompt_tokens == 100  # includes the cached slice
        assert resp.cache_read_tokens == 37

    async def test_gemini_none_cached_count_defaults_to_zero(self):
        """The SDK reports None (not 0) when no cached content applies."""
        from dnd_bot.llm.client import GeminiClient

        client = GeminiClient(model="gemini-test", api_key="test-key")
        client._client = _fake_genai_client(
            {},
            text="hi",
            usage_metadata=SimpleNamespace(
                prompt_token_count=10,
                candidates_token_count=2,
                cached_content_token_count=None,
            ),
        )

        resp = await client.chat(messages=[{"role": "user", "content": "hi"}])
        assert resp.cache_read_tokens == 0
