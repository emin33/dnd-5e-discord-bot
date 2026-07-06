"""Deterministic fake LLM clients for the test net.

Each satisfies ``dnd_bot.llm.client.LLMClient`` — an async ``chat()`` returning
an ``LLMResponse``. They let ``process_action`` tests assert on the tool-call
sequence and the resulting state diff with zero real provider I/O. Modeled on
Pydantic-AI's ``TestModel`` / ``FunctionModel`` and LangChain's
``FakeListChatModel`` (testing-llm-pipelines.md §1).

Pair with ``client.set_model_requests_allowed(False)``: any LLM seam a test
forgets to inject then raises loudly instead of hitting the network.
"""

from __future__ import annotations

import json
from typing import Any, Callable

from dnd_bot.llm.client import LLMResponse


class ScriptedBrain:
    """Cycles a fixed list of ``LLMResponse``s in order (wrapping at the end).

    Like LangChain's ``FakeListChatModel``. Use when the call order is known.
    Records every call on ``.calls`` for spying — each entry is
    ``{"messages": [...], "kwargs": {...}, "method": "chat"|"chat_stream"}``
    (the ``method`` tag lets the narration pins assert which client entry
    point a path drove, since streaming carries different kwargs).
    """

    def __init__(self, responses: list[LLMResponse]):
        if not responses:
            raise ValueError("ScriptedBrain needs at least one response")
        self._responses = list(responses)
        self._i = 0
        self.calls: list[dict] = []  # each: {"messages", "kwargs", "method"}

    async def chat(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        self.calls.append({"messages": messages, "kwargs": kwargs, "method": "chat"})
        resp = self._responses[self._i % len(self._responses)]
        self._i += 1
        return resp

    async def chat_stream(
        self, messages: list[dict], on_token: Any = None, **kwargs: Any
    ) -> LLMResponse:
        resp = await self.chat(messages, **kwargs)
        self.calls[-1]["method"] = "chat_stream"
        if on_token and resp.content:
            await on_token(resp.content)
        return resp


class FunctionBrain:
    """Computes each response from the call via ``fn(messages, **kwargs)``.

    Like Pydantic-AI's ``FunctionModel``. Use when the response must branch on
    the call — e.g. triage and the state/entity extractors share one
    ``get_llm_client()`` seam, so one instance must answer both.
    """

    def __init__(self, fn: Callable[..., LLMResponse]):
        self._fn = fn
        self.calls: list[dict] = []

    async def chat(self, messages: list[dict], **kwargs: Any) -> LLMResponse:
        self.calls.append({"messages": messages, "kwargs": kwargs, "method": "chat"})
        return self._fn(messages, **kwargs)

    async def chat_stream(
        self, messages: list[dict], on_token: Any = None, **kwargs: Any
    ) -> LLMResponse:
        resp = await self.chat(messages, **kwargs)
        self.calls[-1]["method"] = "chat_stream"
        if on_token and resp.content:
            await on_token(resp.content)
        return resp


# ── Response builders ─────────────────────────────────────────────────────────

def triage_response(action_type: str, reasoning: str = "test", **fields: Any) -> LLMResponse:
    """A triage ``LLMResponse`` (JSON content parsed by ``_parse_triage_json``).

    Pass any ``TriageResult`` field as a kwarg, e.g. ``needs_roll=False``,
    ``target_name="goblin"``, ``is_creature_target=True``.
    """
    data = {"action_type": action_type, "reasoning": reasoning, **fields}
    return LLMResponse(content=json.dumps(data))


def narration_response(prose: str, tool_calls: list[dict] | None = None) -> LLMResponse:
    """A narrator ``LLMResponse``: prose + structured tool calls → effects.

    ``tool_calls`` are ``{"name": <tool>, "arguments": {...}}`` dicts, matching
    what ``tool_calls_to_effects`` consumes.
    """
    return LLMResponse(content=prose, tool_calls=tool_calls or [])


def brain_router(triage: LLMResponse) -> Callable[..., LLMResponse]:
    """A ``FunctionBrain`` fn for the shared ``get_llm_client()`` seam.

    Returns ``triage`` for the triage call (identified by its 'action
    classifier' system prompt) and a no-op ``{}`` for every other call — the
    state/entity extractors and dedup judge — so they extract nothing and the
    test's narrator tool calls are the only source of state change.
    """

    def _fn(messages: list[dict], **kwargs: Any) -> LLMResponse:
        system = messages[0].get("content", "") if messages else ""
        if "action classifier" in system:
            return triage
        return LLMResponse(content="{}")

    return _fn
