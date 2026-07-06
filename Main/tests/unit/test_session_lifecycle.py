"""Pins for GameSessionManager.start_session's init sequence (ROOT-3 net).

start_session had ZERO direct coverage before the ROOT-3 slice. These pins
freeze the init blocks that recovery must gain parity with (audit
DF-16/N10: "factor start_session's post-load init into shared helpers
invoked by both paths") BEFORE any extraction, so the helper factoring is
provably behavior-neutral for the start path.

The current restart story these pins coexist with: NO recovery exists —
``recover_sessions`` was deleted as dead code (commit ffc1b1b, audit P0-10)
and ``SessionRepository.end_stale_sessions`` sweeps non-terminal rows at
startup (pinned in tests/integration/test_session_repo_stale.py). ROOT-3
replaces that sweep with real recovery; the pins here must stay green
through it.

Collaborator seams (all module-level names, monkeypatched):
- ``dnd_bot.game.session.get_session_repo`` / ``get_npc_repo`` /
  ``get_memory_manager`` — top-level imports into the session module.
- ``dnd_bot.game.knowledge.KnowledgeGraph`` / ``get_kg_repo`` and
  ``dnd_bot.memory.get_vector_store`` — lazy imports resolved at call
  time, patched at their source modules.
The scene registry is REAL (module-global keyed by session_key); tests use
run-unique channel ids and clear their key afterwards.
"""

from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest

from dnd_bot.game.scene.registry import clear_scene_registry, get_scene_registry
from dnd_bot.game.session import GameSessionManager, SessionState
from dnd_bot.models.npc import NPC, Disposition, EntityType

CAMPAIGN_ID = "lifecycle-campaign"


class FakeSessionRepo:
    """Records save_session/end_session; serves a fixed session number."""

    def __init__(self, session_number: int = 7):
        self.session_number = session_number
        self.saved: list[dict] = []
        self.ended: list[str] = []

    async def get_session_number(self, campaign_id: str) -> int:
        return self.session_number

    async def save_session(self, **kwargs) -> None:
        self.saved.append(kwargs)

    async def end_session(self, session_id: str) -> None:
        self.ended.append(session_id)


class FakeNpcRepo:
    """Serves campaign NPCs; can be armed to raise."""

    def __init__(self, npcs=None, raises: Exception | None = None):
        self.npcs = npcs or []
        self.raises = raises

    async def get_alive_by_campaign(self, campaign_id: str):
        if self.raises:
            raise self.raises
        return self.npcs


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


def _kg_entity(node_id: str, name: str, description: str = "", aliases=None):
    return SimpleNamespace(
        node_id=node_id,
        entity_type=EntityType.NPC,
        name=name,
        properties={"description": description},
        aliases=aliases or [],
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
def memory_warm_calls(monkeypatch) -> list[str]:
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


async def test_start_session_registers_and_initializes(
    manager,
    fake_session_repo,
    fake_npc_repo,
    memory_warm_calls,
    fake_kg,
    chroma_calls,
    unique_channel_id,
    registry_cleanup,
):
    session = await manager.start_session(
        channel_id=unique_channel_id,
        guild_id=42,
        campaign_id=CAMPAIGN_ID,
        dm_user_id=111,
    )
    registry_cleanup.append(session.session_key)

    # Registered under the discord key, ACTIVE, ids threaded through.
    assert manager.get_session(unique_channel_id) is session
    assert session.session_key == f"discord:{unique_channel_id}"
    assert session.state == SessionState.ACTIVE
    assert session.guild_id == 42
    assert session.campaign_id == CAMPAIGN_ID
    assert session.dm_user_id == 111

    # Fresh authoritative world state.
    assert session.world_state is not None
    assert session.world_state.turn == 0
    assert session.world_state.npcs == {}
    assert session.world_state.players == {}

    # Persisted exactly once with the DB-dialect state string.
    assert fake_session_repo.saved == [
        {
            "session_id": session.id,
            "campaign_id": CAMPAIGN_ID,
            "channel_id": unique_channel_id,
            "session_number": 7,
            "state": "exploration",
        }
    ]

    # Memory tiers warmed for the campaign.
    assert memory_warm_calls == [CAMPAIGN_ID]

    # KG built for the campaign and loaded.
    assert session.knowledge_graph is fake_kg.built[0]
    assert session.knowledge_graph.loaded is True


async def test_start_session_syncs_kg_entities_to_chroma(
    manager,
    fake_session_repo,
    fake_npc_repo,
    memory_warm_calls,
    fake_kg,
    chroma_calls,
    unique_channel_id,
    registry_cleanup,
):
    """N10's start-side half: every indexable KG entity reaches ChromaDB."""
    fake_kg.entities = [
        _kg_entity("uuid-1", "Fred", "the innkeeper", aliases=["the fat man"]),
        _kg_entity("uuid-2", "Vex", "a hooded figure"),
    ]

    session = await manager.start_session(
        channel_id=unique_channel_id, guild_id=1, campaign_id=CAMPAIGN_ID
    )
    registry_cleanup.append(session.session_key)

    assert chroma_calls == [
        {
            "campaign_id": CAMPAIGN_ID,
            "node_id": "uuid-1",
            "entity_type": "npc",
            "name": "Fred",
            "description": "the innkeeper",
            "aliases": ["the fat man"],
        },
        {
            "campaign_id": CAMPAIGN_ID,
            "node_id": "uuid-2",
            "entity_type": "npc",
            "name": "Vex",
            "description": "a hooded figure",
            "aliases": [],
        },
    ]


async def test_start_session_skips_chroma_sync_for_empty_kg(
    manager,
    fake_session_repo,
    fake_npc_repo,
    memory_warm_calls,
    fake_kg,
    chroma_calls,
    unique_channel_id,
    registry_cleanup,
):
    session = await manager.start_session(
        channel_id=unique_channel_id, guild_id=1, campaign_id=CAMPAIGN_ID
    )
    registry_cleanup.append(session.session_key)

    assert chroma_calls == []


async def test_start_session_preloads_alive_npcs_into_scene_registry(
    manager,
    fake_session_repo,
    fake_npc_repo,
    memory_warm_calls,
    fake_kg,
    chroma_calls,
    unique_channel_id,
    registry_cleanup,
):
    """DF-16's start-side half: alive campaign NPCs seed the registry."""
    fake_npc_repo.npcs = [
        NPC(
            id="npc-1",
            campaign_id=CAMPAIGN_ID,
            name="Grokk",
            description="an orc bouncer",
            monster_index="orc",
            base_disposition=Disposition.UNFRIENDLY,
            voice_id="voice-9",
        ),
        NPC(id="npc-2", campaign_id=CAMPAIGN_ID, name="Mira"),
    ]

    session = await manager.start_session(
        channel_id=unique_channel_id, guild_id=1, campaign_id=CAMPAIGN_ID
    )
    registry_cleanup.append(session.session_key)

    registry = get_scene_registry(CAMPAIGN_ID, session.session_key)
    by_name = {e.name: e for e in registry.get_all()}
    assert set(by_name) == {"Grokk", "Mira"}

    grokk = by_name["Grokk"]
    assert grokk.npc_id == "npc-1"
    assert grokk.entity_type == EntityType.NPC
    assert grokk.description == "an orc bouncer"
    assert grokk.monster_index == "orc"
    assert grokk.disposition == Disposition.UNFRIENDLY
    assert grokk.voice_id == "voice-9"

    assert by_name["Mira"].npc_id == "npc-2"
    assert by_name["Mira"].disposition == Disposition.NEUTRAL


async def test_start_session_kg_failure_leaves_none_and_still_starts(
    manager,
    fake_session_repo,
    fake_npc_repo,
    memory_warm_calls,
    fake_kg,
    chroma_calls,
    unique_channel_id,
    registry_cleanup,
):
    fake_kg.repo_raises = RuntimeError("kg db unavailable")

    session = await manager.start_session(
        channel_id=unique_channel_id, guild_id=1, campaign_id=CAMPAIGN_ID
    )
    registry_cleanup.append(session.session_key)

    assert session.knowledge_graph is None
    assert manager.get_session(unique_channel_id) is session
    assert session.state == SessionState.ACTIVE


async def test_start_session_npc_preload_failure_still_starts(
    manager,
    fake_session_repo,
    fake_npc_repo,
    memory_warm_calls,
    fake_kg,
    chroma_calls,
    unique_channel_id,
    registry_cleanup,
):
    fake_npc_repo.raises = RuntimeError("npc table locked")

    session = await manager.start_session(
        channel_id=unique_channel_id, guild_id=1, campaign_id=CAMPAIGN_ID
    )
    registry_cleanup.append(session.session_key)

    assert manager.get_session(unique_channel_id) is session
    registry = get_scene_registry(CAMPAIGN_ID, session.session_key)
    assert registry.get_all() == []
