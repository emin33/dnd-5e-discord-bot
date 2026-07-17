"""Unit tests for `dnd_bot.llm.usage_recorder` and the client instrumentation seam.

The recorder is module-level opt-in state; the seam is
``client._instrument`` applied to every instance ``_create_client`` returns.
No test here performs network I/O.
"""

from __future__ import annotations

import asyncio

import pytest

from dnd_bot.llm import usage_recorder


@pytest.fixture(autouse=True)
def _clean_recorder():
    """Every test starts and ends disabled with no events."""
    usage_recorder.disable()
    usage_recorder.reset()
    yield
    usage_recorder.disable()
    usage_recorder.reset()


class TestRecorderRoundTrip:
    def test_disabled_by_default_and_record_is_noop(self):
        assert usage_recorder.is_enabled() is False
        usage_recorder.record(
            provider="groq", model="m", prompt_tokens=10, completion_tokens=5,
        )
        assert usage_recorder.events() == []

    def test_enable_record_events_reset(self):
        usage_recorder.enable()
        assert usage_recorder.is_enabled() is True

        usage_recorder.record(
            provider="anthropic",
            model="claude-test",
            prompt_tokens=100,
            completion_tokens=20,
            cache_read_tokens=64,
            cache_write_tokens=8,
            elapsed_ms=12.5,
        )

        events = usage_recorder.events()
        assert len(events) == 1
        ev = events[0]
        assert ev.provider == "anthropic"
        assert ev.model == "claude-test"
        assert ev.prompt_tokens == 100
        assert ev.completion_tokens == 20
        assert ev.cache_read_tokens == 64
        assert ev.cache_write_tokens == 8
        assert ev.elapsed_ms == 12.5
        assert ev.ts > 0
        assert ev.stage == ""  # no ambient stage active

        # events() returns a copy — mutating it doesn't touch recorder state
        events.clear()
        assert len(usage_recorder.events()) == 1

        usage_recorder.reset()
        assert usage_recorder.events() == []
        # reset() clears events but not the enabled flag
        assert usage_recorder.is_enabled() is True

    def test_stage_contextmanager_sets_and_restores(self):
        usage_recorder.enable()

        with usage_recorder.stage("triage"):
            usage_recorder.record(
                provider="ollama", model="m", prompt_tokens=1, completion_tokens=1,
            )
            with usage_recorder.stage("narration"):
                usage_recorder.record(
                    provider="ollama", model="m", prompt_tokens=1, completion_tokens=1,
                )
            # inner stage restored on exit
            usage_recorder.record(
                provider="ollama", model="m", prompt_tokens=1, completion_tokens=1,
            )
        # outer stage restored to default
        usage_recorder.record(
            provider="ollama", model="m", prompt_tokens=1, completion_tokens=1,
        )

        stages = [ev.stage for ev in usage_recorder.events()]
        assert stages == ["triage", "narration", "triage", ""]

    def test_explicit_stage_overrides_ambient(self):
        usage_recorder.enable()
        with usage_recorder.stage("ambient"):
            usage_recorder.record(
                provider="ollama", model="m", prompt_tokens=1,
                completion_tokens=1, stage="explicit",
            )
        assert usage_recorder.events()[0].stage == "explicit"


class _StubClient:
    """Duck-typed provider client: only an async chat() and a model attr."""

    def __init__(self, response):
        self.model = "stub-model"
        self._response = response
        self.calls = 0

    async def chat(self, *args, **kwargs):
        self.calls += 1
        await asyncio.sleep(0.001)  # make elapsed_ms measurably > 0
        return self._response


class TestInstrument:
    async def test_wrapped_chat_records_one_event(self):
        from dnd_bot.llm.client import LLMResponse, _instrument

        resp = LLMResponse(
            content="hi",
            prompt_tokens=100,
            completion_tokens=20,
            cache_read_tokens=64,
            cache_write_tokens=8,
        )
        client = _instrument(_StubClient(resp), "groq")

        usage_recorder.enable()
        result = await client.chat([{"role": "user", "content": "hi"}])

        # Response passes through unchanged
        assert result is resp
        assert client.calls == 1

        events = usage_recorder.events()
        assert len(events) == 1
        ev = events[0]
        assert ev.provider == "groq"
        assert ev.model == "stub-model"
        assert ev.prompt_tokens == 100
        assert ev.completion_tokens == 20
        assert ev.cache_read_tokens == 64
        assert ev.cache_write_tokens == 8
        assert ev.elapsed_ms > 0

    async def test_disabled_recorder_records_nothing(self):
        from dnd_bot.llm.client import LLMResponse, _instrument

        client = _instrument(_StubClient(LLMResponse(content="hi")), "groq")
        result = await client.chat([{"role": "user", "content": "hi"}])
        assert result.content == "hi"
        assert usage_recorder.events() == []

    async def test_errors_propagate_unchanged(self):
        from dnd_bot.llm.client import _instrument

        class _Boom:
            model = "boom"

            async def chat(self, *args, **kwargs):
                raise TimeoutError("provider timed out")

        client = _instrument(_Boom(), "deepseek")
        usage_recorder.enable()
        with pytest.raises(TimeoutError, match="provider timed out"):
            await client.chat([])
        # Failed calls record nothing
        assert usage_recorder.events() == []

    def test_create_client_returns_instrumented_instance(self):
        """The single seam: every _create_client branch wraps chat()."""
        from dnd_bot.llm.client import OllamaClient, _create_client

        client = _create_client("ollama", "qwen-test")
        try:
            assert isinstance(client, OllamaClient)
            # functools.wraps leaves the original reachable via __wrapped__
            assert hasattr(client.chat, "__wrapped__")
            assert client.chat.__wrapped__.__func__ is OllamaClient.chat
        finally:
            client.close()
