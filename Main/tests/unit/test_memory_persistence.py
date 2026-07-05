"""Tests for campaign-memory persistence (audit P0-5).

The memory tiers the compaction machinery exists to protect — pinned facts,
running summary, condensed summaries — must round-trip through
``MemoryManager.to_dict``/``from_dict`` and the ``campaign_memory`` table, and
must be saved when a manager is evicted from the LRU cache. Legacy or corrupt
persisted state must load as a fresh manager, never crash.
"""

import asyncio
import json
from pathlib import Path

import pytest
from structlog.testing import capture_logs

from dnd_bot.data import database as database_module
from dnd_bot.data.database import Database
from dnd_bot.memory import manager as manager_module
from dnd_bot.memory.blocks import Message, MessageBuffer, SessionSummary
from dnd_bot.memory.manager import MemoryManager


def _populate(mgr: MemoryManager) -> None:
    """Fill every restorable tier with distinctive content."""
    mgr.buffer.add_user_message("I enter the tavern", "Kael", message_id="111")
    mgr.buffer.add_assistant_message("The tavern is dimly lit.")
    mgr.buffer.condense(["Kael entered the tavern and met Garrick."])
    mgr.buffer.compact(
        "The party formed in Eldermoor.",
        ["NPC: Garrick - friendly bartender", "LOCATION: Eldermoor - small village"],
    )
    # In-flight transition buffers (mid-condensation / mid-compaction state)
    mgr.buffer._condensation_buffer.append(
        Message(role="user", content="aged-out exchange", author_name="Kael")
    )
    mgr.buffer._overflow_buffer.append(
        Message(role="system", content="old condensed line")
    )
    mgr.core.update_world("A dark world named Ravenloft.")
    mgr._session_summaries.append(
        SessionSummary(
            session_id="s1",
            campaign_id=mgr.campaign_id,
            summary="They met Garrick.",
            key_events=["met Garrick"],
            npcs_encountered=["Garrick"],
            locations_visited=["Eldermoor"],
        )
    )
    mgr._message_count = 7
    mgr._last_summary_at = 5


def _assert_tiers_match(restored: MemoryManager, original: MemoryManager) -> None:
    """The restorable tiers of ``restored`` equal those of ``original``."""
    assert restored.buffer.pinned_facts == original.buffer.pinned_facts
    assert restored.buffer.running_summary == original.buffer.running_summary
    assert restored.buffer.condensed_summaries == original.buffer.condensed_summaries
    assert [m.content for m in restored.buffer._messages] == [
        m.content for m in original.buffer._messages
    ]
    assert [m.content for m in restored.buffer._condensation_buffer] == [
        m.content for m in original.buffer._condensation_buffer
    ]
    assert [m.content for m in restored.buffer._overflow_buffer] == [
        m.content for m in original.buffer._overflow_buffer
    ]
    assert restored._message_count == original._message_count
    assert restored._last_summary_at == original._last_summary_at


# ==================== MessageBuffer serialization ====================


class TestBufferSerialization:
    """MessageBuffer.to_dict / load_dict round trip."""

    def test_round_trip_preserves_all_tiers(self):
        buf = MessageBuffer(verbatim_size=4, condensed_size=6)
        buf.add_user_message("I look around", "Kael", message_id="42")
        buf.add_assistant_message("The tavern is dimly lit.")
        buf.condense(["Kael scoped out the tavern."])
        buf.compact("The party arrived.", ["NPC: Garrick - bartender"])
        buf._condensation_buffer.append(Message(role="user", content="pending"))
        buf._overflow_buffer.append(Message(role="system", content="overflowed"))

        fresh = MessageBuffer(verbatim_size=4, condensed_size=6)
        fresh.load_dict(buf.to_dict())

        assert fresh.pinned_facts == ["NPC: Garrick - bartender"]
        assert fresh.running_summary == "The party arrived."
        assert fresh.condensed_summaries == ["Kael scoped out the tavern."]
        assert len(fresh._messages) == 2
        assert fresh._messages[0].content == "I look around"
        assert fresh._messages[0].author_name == "Kael"
        assert fresh._messages[0].message_id == "42"
        assert fresh._messages[1].role == "assistant"
        assert fresh._condensation_buffer[0].content == "pending"
        assert fresh._overflow_buffer[0].content == "overflowed"

    def test_to_dict_is_json_serializable(self):
        buf = MessageBuffer(verbatim_size=4)
        buf.add_user_message("hello", "Kael")
        buf.compact("Summary.", ["NPC: Garrick"])

        # Must survive the json.dumps in save_memory_state
        payload = json.loads(json.dumps(buf.to_dict()))
        fresh = MessageBuffer(verbatim_size=4)
        fresh.load_dict(payload)
        assert fresh.pinned_facts == ["NPC: Garrick"]
        assert fresh._messages[0].content == "hello"

    def test_load_dict_tolerates_missing_keys(self):
        """Legacy/partial payloads load with empty defaults, no crash."""
        buf = MessageBuffer(verbatim_size=4)
        buf.load_dict({})

        assert buf._messages == []
        assert buf.condensed_summaries == []
        assert buf.running_summary == ""
        assert buf.pinned_facts == []


# ==================== MemoryManager serialization ====================


class TestManagerSerialization:
    """MemoryManager.to_dict / from_dict round trip (no DB)."""

    def test_to_dict_includes_buffer_tiers(self):
        mgr = MemoryManager("test-campaign")
        _populate(mgr)

        data = mgr.to_dict()
        assert "buffer" in data
        assert data["buffer"]["pinned_facts"] == mgr.buffer.pinned_facts
        assert data["buffer"]["running_summary"] == mgr.buffer.running_summary
        assert data["buffer"]["condensed"] == mgr.buffer.condensed_summaries

    def test_from_dict_restores_tiers(self):
        mgr = MemoryManager("test-campaign")
        _populate(mgr)

        restored = MemoryManager.from_dict(mgr.to_dict())

        _assert_tiers_match(restored, mgr)
        assert restored._session_summaries[0].summary == "They met Garrick."
        world = restored.core.get_block("world")
        assert world is not None and "Ravenloft" in world.content

    def test_from_dict_copies_summary_lists(self):
        """Restored summary lists must not alias the payload's lists."""
        mgr = MemoryManager("test-campaign")
        _populate(mgr)
        payload = json.loads(json.dumps(mgr.to_dict()))

        restored = MemoryManager.from_dict(payload)
        restored._session_summaries[0].key_events.append("mutation")

        assert payload["session_summaries"][0]["key_events"] == ["met Garrick"]

    def test_from_dict_legacy_payload_without_buffer(self):
        """Pre-P0-5 payloads (no 'buffer' key) load with a fresh buffer."""
        mgr = MemoryManager("test-campaign")
        _populate(mgr)
        legacy = mgr.to_dict()
        legacy.pop("buffer")

        restored = MemoryManager.from_dict(legacy)

        assert restored.buffer.pinned_facts == []
        assert restored.buffer.running_summary == ""
        assert restored._message_count == 7  # counters still restored
        world = restored.core.get_block("world")
        assert world is not None and "Ravenloft" in world.content


# ==================== DB round trip & eviction ====================


@pytest.fixture
async def mem_db(tmp_path: Path, monkeypatch):
    """Tmp DB wired into the get_database() global + isolated manager cache."""
    db = Database(db_path=tmp_path / "test.db")
    await db.connect()
    for cid in ("camp-a", "camp-b"):
        await db.execute(
            "INSERT INTO campaign (id, guild_id, name, dm_user_id) VALUES (?, ?, ?, ?)",
            (cid, 1, cid, 1),
        )
    await db.commit()

    monkeypatch.setattr(database_module, "_db", db)
    monkeypatch.setattr(manager_module, "_managers", {})
    monkeypatch.setattr(manager_module, "_pending_saves", set())

    try:
        yield db
    finally:
        # Close aiosqlite or its background thread keeps pytest from exiting.
        await db.disconnect()


class TestDbRoundTrip:
    """save_memory_state / get_memory_manager against the campaign_memory table."""

    async def test_save_then_load_round_trips_tiers(self, mem_db):
        mgr = manager_module.get_memory_manager_sync("camp-a")
        _populate(mgr)
        await manager_module.save_memory_state("camp-a")

        # Simulate restart: cache cleared, async getter loads from the DB
        manager_module._managers.clear()
        restored = await manager_module.get_memory_manager("camp-a")

        assert restored is not mgr
        _assert_tiers_match(restored, mgr)

    async def test_save_for_uncached_campaign_warns_and_noops(self, mem_db):
        """No cached instance and none handed in: nothing to persist, but the
        silent no-op hid the eviction split-brain — it must warn now."""
        with capture_logs() as logs:
            await manager_module.save_memory_state("never-seen")

        assert any(
            e["event"] == "memory_save_skipped_uncached" for e in logs
        ), f"expected memory_save_skipped_uncached warning, got: {logs}"
        row = await mem_db.fetch_one(
            "SELECT content FROM campaign_memory WHERE id = ?", ("never-seen-state",)
        )
        assert row is None

    async def test_save_with_explicit_manager_persists_uncached_instance(self, mem_db):
        """Session end holds the manager instance; after LRU eviction the
        campaign_id lookup misses, so save must persist the handed instance
        (and never clobber the snapshot with a fresh re-creation)."""
        mgr = MemoryManager("camp-a")
        _populate(mgr)
        assert "camp-a" not in manager_module._managers  # simulates eviction

        await manager_module.save_memory_state("camp-a", manager=mgr)

        row = await mem_db.fetch_one(
            "SELECT content FROM campaign_memory WHERE id = ?", ("camp-a-state",)
        )
        assert row is not None
        data = json.loads(row[0])
        assert data["buffer"]["pinned_facts"] == mgr.buffer.pinned_facts
        assert data["buffer"]["running_summary"] == mgr.buffer.running_summary

    async def test_load_absent_state_starts_fresh(self, mem_db):
        restored = await manager_module.get_memory_manager("camp-b")
        assert restored.buffer.pinned_facts == []
        assert restored.buffer.running_summary == ""
        assert restored._message_count == 0

    async def test_load_corrupt_state_starts_fresh(self, mem_db):
        await mem_db.execute(
            """
            INSERT OR REPLACE INTO campaign_memory (id, campaign_id, memory_type, content, metadata)
            VALUES (?, ?, 'manager_state', ?, '{}')
            """,
            ("camp-a-state", "camp-a", "{not valid json"),
        )
        await mem_db.commit()

        restored = await manager_module.get_memory_manager("camp-a")
        assert restored.buffer.pinned_facts == []
        assert restored._message_count == 0


class TestEvictionSavesState:
    """LRU eviction persists the evicted manager instead of discarding it."""

    async def test_async_eviction_saves_state(self, mem_db, monkeypatch):
        monkeypatch.setattr(manager_module, "_MAX_CACHED_MANAGERS", 1)
        mgr_a = manager_module.get_memory_manager_sync("camp-a")
        _populate(mgr_a)

        await manager_module.get_memory_manager("camp-b")  # evicts camp-a

        assert "camp-a" not in manager_module._managers
        row = await mem_db.fetch_one(
            "SELECT content FROM campaign_memory WHERE id = ?", ("camp-a-state",)
        )
        assert row is not None
        data = json.loads(row[0])
        assert data["buffer"]["pinned_facts"] == mgr_a.buffer.pinned_facts

    async def test_sync_eviction_schedules_save(self, mem_db, monkeypatch):
        monkeypatch.setattr(manager_module, "_MAX_CACHED_MANAGERS", 1)
        mgr_a = manager_module.get_memory_manager_sync("camp-a")
        _populate(mgr_a)

        manager_module.get_memory_manager_sync("camp-b")  # evicts camp-a

        pending = list(manager_module._pending_saves)
        assert pending, "sync eviction should schedule a best-effort save"
        await asyncio.gather(*pending)

        row = await mem_db.fetch_one(
            "SELECT content FROM campaign_memory WHERE id = ?", ("camp-a-state",)
        )
        assert row is not None
        data = json.loads(row[0])
        assert data["buffer"]["running_summary"] == mgr_a.buffer.running_summary
