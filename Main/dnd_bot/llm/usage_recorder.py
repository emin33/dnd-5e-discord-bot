"""In-process LLM usage recorder (opt-in, disabled by default).

Collects one :class:`UsageEvent` per provider ``chat()`` call when enabled —
the instrumentation seam lives in ``client._instrument`` (wired in
``client._create_client``) so brain, all narrator tiers, and the KG/summarizer
helpers are covered without touching any provider body.

This module deliberately imports NOTHING from ``client.py`` (client imports
us), so there is no import cycle. State is module-level and guarded by a lock
because Gemini/Ollama calls can complete on executor threads. ``record()`` is
an early-return no-op while disabled — no lock taken on the hot path.
"""

from __future__ import annotations

import threading
import time
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass
from typing import Iterator


@dataclass(frozen=True)
class UsageEvent:
    """One LLM call's usage telemetry."""

    ts: float
    provider: str
    model: str
    stage: str
    prompt_tokens: int
    completion_tokens: int
    cache_read_tokens: int
    cache_write_tokens: int
    elapsed_ms: float


_events: list[UsageEvent] = []
_enabled: bool = False
_lock = threading.Lock()
_stage: ContextVar[str] = ContextVar("llm_usage_stage", default="")


def enable() -> None:
    """Turn recording on (events accumulate until reset())."""
    global _enabled
    _enabled = True


def disable() -> None:
    """Turn recording off (existing events are kept until reset())."""
    global _enabled
    _enabled = False


def is_enabled() -> bool:
    return _enabled


def reset() -> None:
    """Drop all recorded events."""
    with _lock:
        _events.clear()


def events() -> list[UsageEvent]:
    """Snapshot copy of recorded events (oldest first)."""
    with _lock:
        return list(_events)


def record(
    provider: str,
    model: str,
    prompt_tokens: int,
    completion_tokens: int,
    cache_read_tokens: int = 0,
    cache_write_tokens: int = 0,
    elapsed_ms: float = 0.0,
    stage: str | None = None,
) -> None:
    """Append one usage event. No-op while disabled.

    ``stage`` defaults to the ambient stage() context (empty string when
    no stage is active).
    """
    if not _enabled:
        return
    event = UsageEvent(
        ts=time.time(),
        provider=provider,
        model=model,
        stage=stage if stage is not None else _stage.get(),
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cache_read_tokens=cache_read_tokens,
        cache_write_tokens=cache_write_tokens,
        elapsed_ms=elapsed_ms,
    )
    with _lock:
        _events.append(event)


@contextmanager
def stage(name: str) -> Iterator[None]:
    """Tag events recorded inside this context with ``name``.

    ContextVar-based so concurrent asyncio tasks each see their own stage.
    """
    token = _stage.set(name)
    try:
        yield
    finally:
        _stage.reset(token)
