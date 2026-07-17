"""Tests for the Discord text frontend's streamed-narrative preview (C23).

The streamed preview is a PLAIN Discord message (2000-char cap), not an
embed. These pin that the preview stays under the cap, shows the trailing
window (latest prose, leading ellipsis), and that a failed send/edit is
swallowed (best-effort) rather than crashing the turn.
"""

from __future__ import annotations

import discord
import pytest

if not hasattr(discord, "ApplicationContext"):
    # The bot package needs py-cord (discord.ApplicationContext); the system
    # python only has discord.py. The venv (Main/venv) runs these for real.
    pytest.skip(
        "discord_text frontend requires py-cord", allow_module_level=True
    )

# Unguarded on purpose: with py-cord present, a missing symbol (e.g. a
# reverted _STREAM_PREVIEW_LIMIT) must FAIL collection, not skip.
from dnd_bot.bot.frontends.discord_text import (
    _STREAM_PREVIEW_LIMIT,
    DiscordTextFrontend,
)

from dnd_bot.game.frontend import GameEvent

DISCORD_MESSAGE_CAP = 2000


class FakeMessage:
    def __init__(self) -> None:
        self.edits: list[str] = []

    async def edit(self, *, content: str) -> None:
        self.edits.append(content)


class FakeChannel:
    def __init__(self) -> None:
        self.sent: list[str] = []
        self.message = FakeMessage()

    async def send(self, content: str) -> FakeMessage:
        self.sent.append(content)
        return self.message


class _FakeResponse:
    status = 400
    reason = "Bad Request"


class HTTPErrorChannel:
    """Channel whose send always fails like Discord's 400 on >2000 chars."""

    def __init__(self) -> None:
        self.attempts = 0

    async def send(self, content: str) -> FakeMessage:
        self.attempts += 1
        raise discord.HTTPException(_FakeResponse(), "Invalid Form Body")


async def test_stream_preview_stays_under_discord_message_cap() -> None:
    channel = FakeChannel()
    frontend = DiscordTextFrontend(channel)  # type: ignore[arg-type]

    await frontend._handle_narrative_token(
        GameEvent.narrative_token("x" * 3000)
    )

    assert len(channel.sent) == 1
    assert len(channel.sent[0]) <= DISCORD_MESSAGE_CAP

    # Edits (the path that used to die at ~2000 chars) must obey the cap too.
    frontend._last_edit = 0.0  # bypass the rate-limit throttle
    await frontend._handle_narrative_token(
        GameEvent.narrative_token("y" * 500)
    )

    assert len(channel.message.edits) == 1
    assert len(channel.message.edits[0]) <= DISCORD_MESSAGE_CAP


async def test_stream_preview_shows_trailing_window() -> None:
    channel = FakeChannel()
    frontend = DiscordTextFrontend(channel)  # type: ignore[arg-type]

    text = "a" * (_STREAM_PREVIEW_LIMIT + 600) + " THE LATEST PROSE"
    await frontend._handle_narrative_token(GameEvent.narrative_token(text))

    content = channel.sent[0]
    # Leading ellipsis marks the trimmed head; the newest prose is visible.
    assert content.startswith("*...")
    assert "THE LATEST PROSE" in content


async def test_short_stream_preview_is_untruncated() -> None:
    channel = FakeChannel()
    frontend = DiscordTextFrontend(channel)  # type: ignore[arg-type]

    await frontend._handle_narrative_token(GameEvent.narrative_token("Hello"))

    assert channel.sent == ["*Hello*"]


async def test_stream_http_error_is_swallowed_not_raised() -> None:
    channel = HTTPErrorChannel()
    frontend = DiscordTextFrontend(channel)  # type: ignore[arg-type]

    # Must not raise: streaming is best-effort and never crashes the turn.
    await frontend._handle_narrative_token(GameEvent.narrative_token("x" * 100))

    assert channel.attempts == 1
    assert frontend._stream_msg is None
