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

Collaborator seams: see tests/unit/conftest.py (shared with the recovery
tests — both paths must drive the SAME fakes, that's the parity point).
"""

from dnd_bot.game.scene.registry import get_scene_registry
from dnd_bot.game.session import SessionState
from dnd_bot.models.npc import NPC, Disposition, EntityType
from tests.session_fakes import kg_entity as _kg_entity

CAMPAIGN_ID = "lifecycle-campaign"


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
