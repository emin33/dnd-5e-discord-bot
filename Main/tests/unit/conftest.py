"""Fixtures for the session-layer collaborator seams (ROOT-3 net).

Shared by test_session_lifecycle.py (start_session pins) and
test_session_recovery.py (per-turn persist + recover_sessions). Seam map:

- ``dnd_bot.game.session.get_session_repo`` / ``get_npc_repo`` /
  ``get_character_repo`` / ``get_campaign_repo`` / ``get_memory_manager``
  / ``get_orchestrator`` — top-level imports into the session module,
  patched there.
- ``dnd_bot.game.knowledge.KnowledgeGraph`` / ``get_kg_repo`` and
  ``dnd_bot.memory.get_vector_store`` — lazy imports resolved at call
  time, patched at their source modules.

The scene registry is REAL (module-global keyed by session_key); tests
use run-unique channel ids and clear their keys via ``registry_cleanup``.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dnd_bot.game.scene.registry import clear_scene_registry
from dnd_bot.game.session import GameSessionManager
from tests.session_fakes import (
    FakeCampaignRepo,
    FakeCharacterRepo,
    FakeKnowledgeGraph,
    FakeNpcRepo,
    FakeSessionRepo,
)


@pytest.fixture
def fake_session_repo(monkeypatch) -> FakeSessionRepo:
    repo = FakeSessionRepo()

    async def _get_repo():
        return repo

    monkeypatch.setattr("dnd_bot.game.session.get_session_repo", _get_repo)
    return repo


@pytest.fixture
def fake_npc_repo(monkeypatch) -> FakeNpcRepo:
    repo = FakeNpcRepo()

    async def _get_repo():
        return repo

    monkeypatch.setattr("dnd_bot.game.session.get_npc_repo", _get_repo)
    return repo


@pytest.fixture
def fake_character_repo(monkeypatch) -> FakeCharacterRepo:
    repo = FakeCharacterRepo()

    async def _get_repo():
        return repo

    monkeypatch.setattr("dnd_bot.game.session.get_character_repo", _get_repo)
    return repo


@pytest.fixture
def fake_campaign_repo(monkeypatch) -> FakeCampaignRepo:
    repo = FakeCampaignRepo()

    async def _get_repo():
        return repo

    monkeypatch.setattr("dnd_bot.game.session.get_campaign_repo", _get_repo)
    return repo


@pytest.fixture
def memory_warm_calls(monkeypatch) -> list[str]:
    """Record get_memory_manager warms; serves an inert mock manager."""
    calls: list[str] = []

    async def _get_memory(campaign_id: str):
        calls.append(campaign_id)
        return MagicMock()

    monkeypatch.setattr("dnd_bot.game.session.get_memory_manager", _get_memory)
    return calls


@pytest.fixture
def fake_kg(monkeypatch):
    """Patch the lazy KG imports; returns a mutable holder for entities."""
    holder = SimpleNamespace(entities=[], built=[], repo_raises=None)

    async def _get_kg_repo():
        if holder.repo_raises:
            raise holder.repo_raises
        return MagicMock()

    def _build(campaign_id, kg_repo):
        kg = FakeKnowledgeGraph(campaign_id, kg_repo, entities=holder.entities)
        holder.built.append(kg)
        return kg

    monkeypatch.setattr("dnd_bot.game.knowledge.get_kg_repo", _get_kg_repo)
    monkeypatch.setattr("dnd_bot.game.knowledge.KnowledgeGraph", _build)
    return holder


@pytest.fixture
def chroma_calls(monkeypatch) -> list[dict]:
    calls: list[dict] = []

    class FakeVectorStore:
        def add_entity_description(self, **kwargs):
            calls.append(kwargs)

    monkeypatch.setattr("dnd_bot.memory.get_vector_store", lambda: FakeVectorStore())
    return calls


@pytest.fixture
def manager() -> GameSessionManager:
    return GameSessionManager()


@pytest.fixture
def registry_cleanup():
    """Clear the module-global scene registry for keys a test dirtied."""
    keys: list[str] = []
    yield keys
    for key in keys:
        clear_scene_registry(key)
