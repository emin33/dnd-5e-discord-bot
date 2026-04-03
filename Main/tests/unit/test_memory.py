"""Unit tests for the memory system.

Tests three subsystems added from agentic orchestration patterns:
1. MessageBuffer overflow + compaction mechanics
2. Three-gate consolidation logic
3. DM Scratchpad state management
"""

import time
import pytest

from dnd_bot.memory.blocks import MessageBuffer, Message
from dnd_bot.memory.manager import MemoryManager


# ==================== MessageBuffer: Overflow & Compaction ====================


class TestMessageBufferOverflow:
    """Test that messages overflow into compaction buffer correctly."""

    def test_no_overflow_under_limit(self):
        """Messages within max_messages stay in the main buffer."""
        buf = MessageBuffer(max_messages=5)
        for i in range(5):
            buf.add_user_message(f"msg {i}", "Player")

        assert len(buf._messages) == 5
        assert len(buf._overflow_buffer) == 0
        assert not buf.has_pending_compaction

    def test_overflow_moves_oldest_messages(self):
        """Adding beyond max_messages pushes oldest into overflow."""
        buf = MessageBuffer(max_messages=5)
        for i in range(8):
            buf.add_user_message(f"msg {i}", "Player")

        # 5 in main, 3 in overflow
        assert len(buf._messages) == 5
        assert len(buf._overflow_buffer) == 3
        # Oldest messages went to overflow
        assert buf._overflow_buffer[0].content == "msg 0"
        assert buf._overflow_buffer[1].content == "msg 1"
        assert buf._overflow_buffer[2].content == "msg 2"
        # Newest stay in main
        assert buf._messages[0].content == "msg 3"

    def test_has_pending_compaction_threshold(self):
        """has_pending_compaction triggers at 6+ overflow messages."""
        buf = MessageBuffer(max_messages=5)
        # Add 10 messages → 5 overflow
        for i in range(10):
            buf.add_user_message(f"msg {i}", "Player")
        assert not buf.has_pending_compaction  # 5 < 6

        # One more → 6 overflow
        buf.add_user_message("msg 10", "Player")
        assert buf.has_pending_compaction  # 6 >= 6

    def test_overflow_text_formatting(self):
        """get_overflow_text formats player vs DM messages correctly."""
        buf = MessageBuffer(max_messages=2)
        buf.add_user_message("I look around", "Kael")
        buf.add_assistant_message("The tavern is dimly lit.")
        buf.add_user_message("I talk to the bartender", "Kael")
        buf.add_assistant_message("The bartender nods.")
        # First 2 messages overflowed
        text = buf.get_overflow_text()
        assert "Kael: I look around" in text
        assert "DM: The tavern is dimly lit." in text

    def test_compact_merges_summaries(self):
        """compact() merges new summary with existing running summary."""
        buf = MessageBuffer(max_messages=5)

        # First compaction
        buf._overflow_buffer = [Message(role="user", content="old stuff")]
        buf.compact("The party explored a cave.")
        assert buf.running_summary == "The party explored a cave."
        assert len(buf._overflow_buffer) == 0

        # Second compaction merges
        buf._overflow_buffer = [Message(role="user", content="more stuff")]
        buf.compact("They fought goblins and found treasure.")
        assert "The party explored a cave." in buf.running_summary
        assert "They fought goblins and found treasure." in buf.running_summary

    def test_compact_clears_overflow(self):
        """compact() clears the overflow buffer."""
        buf = MessageBuffer(max_messages=3)
        for i in range(10):
            buf.add_user_message(f"msg {i}", "Player")

        assert len(buf._overflow_buffer) > 0
        buf.compact("Summary of old messages.")
        assert len(buf._overflow_buffer) == 0

    def test_running_summary_starts_empty(self):
        """Fresh buffer has no running summary."""
        buf = MessageBuffer(max_messages=10)
        assert buf.running_summary == ""

    def test_get_messages_returns_current(self):
        """get_messages only returns current buffer, not overflow."""
        buf = MessageBuffer(max_messages=3)
        for i in range(5):
            buf.add_user_message(f"msg {i}", "Player")

        messages = buf.get_messages()
        assert len(messages) == 3
        assert messages[0].content == "msg 2"
        assert messages[2].content == "msg 4"

    def test_system_messages_overflow_too(self):
        """System messages participate in overflow like any other."""
        buf = MessageBuffer(max_messages=3)
        buf.add_system_message("Combat started")
        buf.add_user_message("I attack", "Player")
        buf.add_assistant_message("You swing your sword.")
        buf.add_system_message("Goblin takes 5 damage")  # This pushes "Combat started" out

        assert len(buf._overflow_buffer) == 1
        assert buf._overflow_buffer[0].content == "Combat started"


# ==================== Three-Gate Consolidation ====================


class TestGatedConsolidation:
    """Test the three-gate memory consolidation logic."""

    def _make_manager(self, **overrides) -> MemoryManager:
        """Create a MemoryManager with controllable gate thresholds."""
        mgr = MemoryManager("test-campaign")
        mgr._gate_min_messages = overrides.get("min_messages", 10)
        mgr._gate_min_elapsed_sec = overrides.get("min_elapsed", 60)
        mgr._gate_not_in_combat = overrides.get("not_in_combat", True)
        return mgr

    def test_all_gates_closed_initially(self):
        """Fresh manager: not enough messages → should NOT summarize."""
        mgr = self._make_manager()
        assert not mgr._should_summarize()

    def test_gate1_message_count(self):
        """Gate 1: blocks until enough messages accumulated."""
        mgr = self._make_manager(min_messages=5, min_elapsed=0)
        mgr._message_count = 4
        assert not mgr._should_summarize()

        mgr._message_count = 5
        assert mgr._should_summarize()

    def test_gate2_time_elapsed(self):
        """Gate 2: blocks until enough time has passed."""
        mgr = self._make_manager(min_messages=1, min_elapsed=10)
        mgr._message_count = 20  # Plenty of messages

        # Pretend last summary was just now
        mgr._last_summary_time = time.monotonic()
        assert not mgr._should_summarize()

        # Pretend last summary was 15 seconds ago
        mgr._last_summary_time = time.monotonic() - 15
        assert mgr._should_summarize()

    def test_gate3_combat_blocks(self):
        """Gate 3: blocks during active combat."""
        mgr = self._make_manager(min_messages=1, min_elapsed=0)
        mgr._message_count = 20
        mgr._last_summary_time = time.monotonic() - 999  # Long ago

        # Not in combat → should summarize
        mgr._is_in_combat = False
        assert mgr._should_summarize()

        # In combat → should NOT summarize
        mgr._is_in_combat = True
        assert not mgr._should_summarize()

    def test_gate3_combat_disable(self):
        """Gate 3 can be disabled to allow combat summarization."""
        mgr = self._make_manager(min_messages=1, min_elapsed=0, not_in_combat=False)
        mgr._message_count = 20
        mgr._last_summary_time = time.monotonic() - 999
        mgr._is_in_combat = True

        # Combat gate disabled → should summarize even in combat
        assert mgr._should_summarize()

    def test_gate4_lock_prevents_concurrent(self):
        """Gate 4 (lock): blocks if consolidation already in progress."""
        mgr = self._make_manager(min_messages=1, min_elapsed=0)
        mgr._message_count = 20
        mgr._last_summary_time = time.monotonic() - 999

        assert mgr._should_summarize()

        # Simulate consolidation in progress
        mgr._is_consolidating = True
        assert not mgr._should_summarize()

    def test_set_combat_state(self):
        """set_combat_state correctly toggles the combat flag."""
        mgr = self._make_manager()
        assert not mgr._is_in_combat

        mgr.set_combat_state(True)
        assert mgr._is_in_combat

        mgr.set_combat_state(False)
        assert not mgr._is_in_combat

    def test_all_gates_pass(self):
        """When all conditions met, summarization is allowed."""
        mgr = self._make_manager(min_messages=5, min_elapsed=30)
        mgr._message_count = 10
        mgr._last_summary_at = 0
        mgr._last_summary_time = time.monotonic() - 60  # 60s ago
        mgr._is_in_combat = False
        mgr._is_consolidating = False

        assert mgr._should_summarize()

    def test_summary_resets_counters(self):
        """After _last_summary_at is updated, gate 1 re-closes."""
        mgr = self._make_manager(min_messages=5, min_elapsed=0)
        mgr._message_count = 10
        mgr._last_summary_time = time.monotonic() - 999

        assert mgr._should_summarize()

        # Simulate summary happened
        mgr._last_summary_at = mgr._message_count
        mgr._last_summary_time = time.monotonic()

        assert not mgr._should_summarize()


# ==================== Context Building ====================


class TestContextBuilding:
    """Test that build_context includes running summary."""

    def test_context_includes_story_so_far(self):
        """build_context should include the running summary when present."""
        mgr = MemoryManager("test-campaign")
        mgr.buffer._running_summary = "The party explored the cave and met a dragon."

        context = mgr.build_context()
        assert "<story_so_far>" in context
        assert "The party explored the cave and met a dragon." in context
        assert "</story_so_far>" in context

    def test_context_excludes_empty_summary(self):
        """build_context should not include story_so_far if empty."""
        mgr = MemoryManager("test-campaign")
        context = mgr.build_context()
        assert "<story_so_far>" not in context

    def test_context_includes_core_memory(self):
        """build_context always includes core memory."""
        mgr = MemoryManager("test-campaign")
        mgr.core.update_world("A dark fantasy world called Ravenloft.")

        context = mgr.build_context()
        assert "Ravenloft" in context


# ==================== DM Scratchpad ====================


class TestDMScratchpad:
    """Test the DM scratchpad from the orchestrator."""

    def _make_orchestrator(self):
        """Create a minimal orchestrator for scratchpad testing."""
        # Import here to avoid pulling in the full LLM stack at module level
        from dnd_bot.llm.orchestrator import DMOrchestrator
        orch = DMOrchestrator.__new__(DMOrchestrator)
        orch._scratchpad = []
        orch._scratchpad_turn = 0
        orch._scratchpad_max_entries = 20
        return orch

    def test_scratchpad_starts_empty(self):
        orch = self._make_orchestrator()
        assert orch.scratchpad_context() == ""

    def test_scratchpad_note_adds_entry(self):
        orch = self._make_orchestrator()
        orch.scratchpad_note("tension", "The bridge is creaking.")

        assert len(orch._scratchpad) == 1
        assert orch._scratchpad[0]["category"] == "tension"
        assert orch._scratchpad[0]["note"] == "The bridge is creaking."
        assert orch._scratchpad[0]["turn"] == 1

    def test_scratchpad_context_format(self):
        orch = self._make_orchestrator()
        orch.scratchpad_note("npc_mood", "Bartender seems nervous.")
        orch.scratchpad_note("unresolved", "Player failed perception check.")

        ctx = orch.scratchpad_context()
        assert "<dm_scratchpad>" in ctx
        assert "[npc_mood] Bartender seems nervous." in ctx
        assert "[unresolved] Player failed perception check." in ctx
        assert "</dm_scratchpad>" in ctx

    def test_scratchpad_rolling_window(self):
        """Scratchpad trims to max_entries."""
        orch = self._make_orchestrator()
        orch._scratchpad_max_entries = 5

        for i in range(10):
            orch.scratchpad_note("plot", f"Event {i}")

        assert len(orch._scratchpad) == 5
        # Should keep the 5 most recent
        assert orch._scratchpad[0]["note"] == "Event 5"
        assert orch._scratchpad[4]["note"] == "Event 9"

    def test_scratchpad_clear(self):
        orch = self._make_orchestrator()
        orch.scratchpad_note("tension", "Something is wrong.")
        orch.scratchpad_note("plot", "A clue was found.")
        orch.scratchpad_clear()

        assert len(orch._scratchpad) == 0
        assert orch._scratchpad_turn == 0
        assert orch.scratchpad_context() == ""

    def test_scratchpad_turn_increments(self):
        orch = self._make_orchestrator()
        orch.scratchpad_note("a", "first")
        orch.scratchpad_note("b", "second")
        orch.scratchpad_note("c", "third")

        assert orch._scratchpad[0]["turn"] == 1
        assert orch._scratchpad[1]["turn"] == 2
        assert orch._scratchpad[2]["turn"] == 3

    def test_scratchpad_multiple_categories(self):
        """Different categories coexist."""
        orch = self._make_orchestrator()
        orch.scratchpad_note("tension", "The walls are closing in.")
        orch.scratchpad_note("npc_mood", "Guard is suspicious.")
        orch.scratchpad_note("foreshadow", "A distant howl echoes.")

        categories = [e["category"] for e in orch._scratchpad]
        assert categories == ["tension", "npc_mood", "foreshadow"]
