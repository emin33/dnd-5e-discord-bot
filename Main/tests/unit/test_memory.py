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
        buf = MessageBuffer(max_messages=5, verbatim_size=5)
        for i in range(5):
            buf.add_user_message(f"msg {i}", "Player")

        assert len(buf._messages) == 5
        assert len(buf._overflow_buffer) == 0
        assert not buf.has_pending_compaction

    def test_overflow_moves_oldest_messages(self):
        """Adding beyond verbatim_size pushes oldest into condensation buffer."""
        buf = MessageBuffer(max_messages=5, verbatim_size=5)
        for i in range(8):
            buf.add_user_message(f"msg {i}", "Player")

        # 5 in verbatim, 3 in condensation buffer
        assert len(buf._messages) == 5
        assert len(buf._condensation_buffer) == 3
        # Oldest messages went to condensation
        assert buf._condensation_buffer[0].content == "msg 0"
        assert buf._condensation_buffer[1].content == "msg 1"
        assert buf._condensation_buffer[2].content == "msg 2"
        # Newest stay in verbatim
        assert buf._messages[0].content == "msg 3"

    def test_has_pending_condensation_threshold(self):
        """has_pending_condensation triggers at 4+ condensation buffer messages."""
        buf = MessageBuffer(max_messages=5, verbatim_size=5)
        # Add 8 messages → 3 in condensation buffer
        for i in range(8):
            buf.add_user_message(f"msg {i}", "Player")
        assert not buf.has_pending_condensation  # 3 < 4

        # One more → 4 in condensation buffer
        buf.add_user_message("msg 8", "Player")
        assert buf.has_pending_condensation  # 4 >= 4

    def test_condensation_text_formatting(self):
        """get_condensation_text formats player vs DM messages correctly."""
        buf = MessageBuffer(max_messages=2, verbatim_size=2)
        buf.add_user_message("I look around", "Kael")
        buf.add_assistant_message("The tavern is dimly lit.")
        buf.add_user_message("I talk to the bartender", "Kael")
        buf.add_assistant_message("The bartender nods.")
        # First 2 messages went to condensation buffer
        text = buf.get_condensation_text()
        assert "Kael: I look around" in text
        assert "DM: The tavern is dimly lit." in text

    def test_compact_merges_summaries(self):
        """compact() merges new summary with existing running summary."""
        buf = MessageBuffer(max_messages=5, verbatim_size=5)

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
        buf = MessageBuffer(max_messages=3, verbatim_size=3)
        for i in range(10):
            buf.add_user_message(f"msg {i}", "Player")

        # Messages overflow to condensation buffer first
        assert len(buf._condensation_buffer) > 0
        # Simulate condensation moving some to overflow
        buf._overflow_buffer = list(buf._condensation_buffer)
        buf._condensation_buffer.clear()
        buf.compact("Summary of old messages.")
        assert len(buf._overflow_buffer) == 0

    def test_running_summary_starts_empty(self):
        """Fresh buffer has no running summary."""
        buf = MessageBuffer(max_messages=10, verbatim_size=10)
        assert buf.running_summary == ""

    def test_get_messages_returns_verbatim_only(self):
        """get_messages only returns verbatim tier, not condensation/overflow."""
        buf = MessageBuffer(max_messages=3, verbatim_size=3)
        for i in range(5):
            buf.add_user_message(f"msg {i}", "Player")

        messages = buf.get_messages()
        assert len(messages) == 3
        assert messages[0].content == "msg 2"
        assert messages[2].content == "msg 4"

    def test_compact_with_pinned_facts(self):
        """compact() stores typed facts separately from narrative summary."""
        buf = MessageBuffer(max_messages=5, verbatim_size=5)
        buf.compact(
            "The party entered the tavern and met the innkeeper.",
            ["NPC: Garrick - friendly bartender", "LOCATION: The Silver Tankard - tavern"],
        )

        assert "Garrick" in buf.pinned_facts[0]
        assert "Silver Tankard" in buf.pinned_facts[1]
        assert "tavern" in buf.running_summary

    def test_pinned_facts_deduplicated(self):
        """Duplicate facts aren't added twice."""
        buf = MessageBuffer(max_messages=5, verbatim_size=5)
        buf.compact("First summary.", ["NPC: Garrick - bartender"])
        buf.compact("Second summary.", ["NPC: Garrick - bartender", "NPC: Mira - merchant"])

        assert len(buf.pinned_facts) == 2  # Garrick + Mira, not Garrick twice

    def test_pinned_facts_survive_multiple_compactions(self):
        """Facts from early compaction survive later ones."""
        buf = MessageBuffer(max_messages=5, verbatim_size=5)
        buf.compact("Part 1.", ["NPC: Aldric - guard captain"])
        buf.compact("Part 2.", ["LOCATION: Eldermoor - small village"])
        buf.compact("Part 3.", ["EVENT: Wolves attacked the caravan"])

        assert len(buf.pinned_facts) == 3
        assert any("Aldric" in f for f in buf.pinned_facts)
        assert any("Eldermoor" in f for f in buf.pinned_facts)
        assert any("Wolves" in f for f in buf.pinned_facts)

    def test_pinned_facts_start_empty(self):
        """Fresh buffer has no pinned facts."""
        buf = MessageBuffer(max_messages=10, verbatim_size=10)
        assert buf.pinned_facts == []

    def test_system_messages_overflow_too(self):
        """System messages participate in overflow like any other."""
        buf = MessageBuffer(max_messages=3, verbatim_size=3)
        buf.add_system_message("Combat started")
        buf.add_user_message("I attack", "Player")
        buf.add_assistant_message("You swing your sword.")
        buf.add_system_message("Goblin takes 5 damage")  # This pushes "Combat started" out

        assert len(buf._condensation_buffer) == 1
        assert buf._condensation_buffer[0].content == "Combat started"


# ==================== Tiered Compression ====================


class TestTieredCompression:
    """Test the three-tier message compression system."""

    def test_tier1_to_condensation_buffer(self):
        """Messages overflow from verbatim to condensation buffer at verbatim_size."""
        buf = MessageBuffer(verbatim_size=4, condensed_size=6)
        for i in range(6):
            buf.add_user_message(f"msg {i}", "Player")

        assert len(buf._messages) == 4
        assert len(buf._condensation_buffer) == 2

    def test_condense_moves_to_tier2(self):
        """condense() adds summaries to the condensed tier."""
        buf = MessageBuffer(verbatim_size=4, condensed_size=6)
        buf.condense(["Party entered the tavern.", "Kael talked to the barkeep."])

        assert len(buf.condensed_summaries) == 2
        assert "tavern" in buf.condensed_summaries[0]

    def test_condensed_overflow_to_tier3(self):
        """When condensed tier exceeds condensed_size, oldest flow to overflow."""
        buf = MessageBuffer(verbatim_size=4, condensed_size=3)
        buf.condense(["Event 1", "Event 2", "Event 3", "Event 4", "Event 5"])

        # 3 kept in condensed, 2 overflow
        assert len(buf.condensed_summaries) == 3
        assert len(buf._overflow_buffer) == 2
        # Oldest went to overflow
        assert buf._overflow_buffer[0].content == "Event 1"

    def test_condensation_buffer_cleared_after_condense(self):
        """condense() clears the condensation buffer."""
        buf = MessageBuffer(verbatim_size=4, condensed_size=6)
        for i in range(8):
            buf.add_user_message(f"msg {i}", "Player")
        assert len(buf._condensation_buffer) == 4

        buf.condense(["Summary 1", "Summary 2"])
        assert len(buf._condensation_buffer) == 0

    def test_full_three_tier_flow(self):
        """Messages flow through all three tiers correctly."""
        buf = MessageBuffer(verbatim_size=4, condensed_size=3)
        buf._compaction_threshold = 2

        # Fill verbatim
        for i in range(4):
            buf.add_user_message(f"msg {i}", "Player")
        assert len(buf._messages) == 4
        assert len(buf._condensation_buffer) == 0

        # Overflow to condensation
        for i in range(4, 8):
            buf.add_user_message(f"msg {i}", "Player")
        assert len(buf._messages) == 4
        assert len(buf._condensation_buffer) == 4

        # Condense → fills Tier 2
        buf.condense(["Sum A", "Sum B", "Sum C", "Sum D"])
        assert len(buf.condensed_summaries) == 3  # condensed_size=3
        assert len(buf._overflow_buffer) == 1     # 1 overflowed to Tier 3

    def test_condensed_summaries_property(self):
        """condensed_summaries returns a copy, not the internal list."""
        buf = MessageBuffer(verbatim_size=4, condensed_size=6)
        buf.condense(["One", "Two"])
        summaries = buf.condensed_summaries
        summaries.append("Mutated")
        assert len(buf.condensed_summaries) == 2  # internal unaffected


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

    def test_context_includes_pinned_facts(self):
        """build_context includes pinned facts in established_facts block."""
        mgr = MemoryManager("test-campaign")
        mgr.buffer._pinned_facts = [
            "NPC: Garrick - friendly bartender",
            "LOCATION: Eldermoor - small village",
        ]

        context = mgr.build_context()
        assert "<established_facts>" in context
        assert "Garrick" in context
        assert "Eldermoor" in context
        assert "</established_facts>" in context

    def test_context_pinned_facts_before_story(self):
        """Pinned facts appear before story_so_far in context."""
        mgr = MemoryManager("test-campaign")
        mgr.buffer._pinned_facts = ["NPC: Garrick"]
        mgr.buffer._running_summary = "The party explored."

        context = mgr.build_context()
        facts_pos = context.index("<established_facts>")
        story_pos = context.index("<story_so_far>")
        assert facts_pos < story_pos

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
