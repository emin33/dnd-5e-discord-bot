"""Fakes for the session-layer collaborator seams (ROOT-3 net).

Shared by the start_session pins (tests/unit/test_session_lifecycle.py)
and the recovery tests (tests/unit/test_session_recovery.py) — the whole
point of the ROOT-3 slice is that both paths run the SAME init, so both
test files must drive the same fakes.

Repo fakes mirror only the surface GameSessionManager actually calls;
the real repos are covered by the integration round-trip tests.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any, Optional


class FakeSessionRepo:
    """In-memory stand-in for SessionRepository."""

    def __init__(self, session_number: int = 7):
        self.session_number = session_number
        self.saved: list[dict] = []
        self.ended: list[str] = []
        self.active_rows: list[dict] = []
        self.snapshots: dict[str, str] = {}
        self.snapshot_saves: list[tuple[str, str]] = []
        self.save_snapshot_raises: Optional[Exception] = None

    async def get_session_number(self, campaign_id: str) -> int:
        return self.session_number

    async def save_session(self, **kwargs: Any) -> None:
        self.saved.append(kwargs)

    async def end_session(self, session_id: str) -> None:
        self.ended.append(session_id)

    async def load_active_sessions(self) -> list[dict]:
        return list(self.active_rows)

    async def save_world_snapshot(self, session_id: str, game_state: str) -> None:
        if self.save_snapshot_raises:
            raise self.save_snapshot_raises
        self.snapshot_saves.append((session_id, game_state))
        self.snapshots[session_id] = game_state

    async def get_latest_snapshot(self, session_id: str) -> Optional[str]:
        return self.snapshots.get(session_id)


class FakeNpcRepo:
    """Serves campaign NPCs; can be armed to raise."""

    def __init__(self, npcs=None, raises: Optional[Exception] = None):
        self.npcs = npcs or []
        self.raises = raises

    async def get_alive_by_campaign(self, campaign_id: str):
        if self.raises:
            raise self.raises
        return self.npcs


class FakeCharacterRepo:
    """Serves Character objects by id (the recovery re-fetch seam)."""

    def __init__(self, characters=None):
        self.characters = {c.id: c for c in (characters or [])}

    async def get_by_id(self, character_id: str):
        return self.characters.get(character_id)


class FakeCampaignRepo:
    """Serves Campaign rows by id (the DF-7 guild_id source)."""

    def __init__(self, campaigns=None):
        self.campaigns = {c.id: c for c in (campaigns or [])}

    async def get_by_id(self, campaign_id: str):
        return self.campaigns.get(campaign_id)


class FakeKnowledgeGraph:
    """Stands in for KnowledgeGraph(campaign_id, kg_repo)."""

    def __init__(self, campaign_id, kg_repo, entities=None):
        self.campaign_id = campaign_id
        self.kg_repo = kg_repo
        self.loaded = False
        self._entities = entities or []

    async def load(self) -> None:
        self.loaded = True

    def node_count(self) -> int:
        return len(self._entities)

    def edge_count(self) -> int:
        return 0

    def get_entities_for_indexing(self):
        return self._entities


def kg_entity(node_id: str, name: str, description: str = "", aliases=None):
    """A KG node shaped like get_entities_for_indexing() output."""
    from dnd_bot.models.npc import EntityType

    return SimpleNamespace(
        node_id=node_id,
        entity_type=EntityType.NPC,
        name=name,
        properties={"description": description},
        aliases=aliases or [],
    )


class FakeMemoryManager:
    """The slice of MemoryManager that process_message touches."""

    def __init__(self):
        self.buffer = SimpleNamespace(pinned_facts=[])
        self.core = SimpleNamespace(get_block=lambda name: None)

    def set_combat_state(self, in_combat: bool) -> None:
        pass

    async def add_player_message(self, content: str, author_name: str) -> None:
        pass

    async def add_dm_response(self, content: str, is_narration: bool = True) -> None:
        pass

    def update_scene(self, summary: str) -> None:
        pass

    def build_context(self, current_input: str) -> str:
        return "memory-context"

    def get_message_history(self, limit: int = 30) -> str:
        return "message-history"
