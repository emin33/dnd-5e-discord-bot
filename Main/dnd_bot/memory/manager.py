"""Memory manager - coordinates all memory subsystems.

Uses a three-gate consolidation pattern (inspired by agentic orchestration
systems) to avoid wasting LLM calls on rapid-fire exchanges like combat.
"""

import time
from typing import Optional
from datetime import datetime
import json
import structlog

from .blocks import CoreMemory, MessageBuffer, SessionSummary
from .vector_store import VectorStore, get_vector_store
from ..llm.client import get_ollama_client

logger = structlog.get_logger()


# Session summary generation prompt
SUMMARIZE_PROMPT = """You are summarizing a D&D game session.
Given the conversation below, provide:
1. A brief narrative summary (2-3 sentences)
2. Key events that occurred (bullet points)
3. NPCs encountered or mentioned (names only)
4. Locations visited (names only)

Be concise and focus on plot-relevant details.

Conversation:
{conversation}

Respond in JSON format:
{
    "summary": "narrative summary here",
    "key_events": ["event 1", "event 2"],
    "npcs": ["npc name 1", "npc name 2"],
    "locations": ["location 1", "location 2"]
}"""


class MemoryManager:
    """
    Manages the tiered memory system for a campaign.

    Tiers:
    1. Core Memory - Always in context (~1500 tokens)
    2. Message Buffer - Recent messages (sliding window)
    3. Session Summaries - Compressed past sessions
    4. Vector Store - RAG for distant memories
    """

    def __init__(self, campaign_id: str):
        self.campaign_id = campaign_id

        # Initialize memory tiers
        self.core = CoreMemory(campaign_id)
        self.buffer = MessageBuffer(max_messages=20)
        self.vector_store = get_vector_store()

        # Tracking
        self._message_count = 0
        self._last_summary_at = 0
        self._session_summaries: list[SessionSummary] = []

        # Three-gate consolidation (inspired by agentic orchestration patterns)
        self._gate_min_messages = 10       # Gate 1: minimum messages since last summary
        self._gate_min_elapsed_sec = 60    # Gate 2: minimum seconds since last summary
        self._gate_not_in_combat = True    # Gate 3: skip during active combat
        self._last_summary_time = time.monotonic()
        self._is_in_combat = False
        self._is_consolidating = False     # Lock gate: prevent concurrent consolidation

    async def add_player_message(
        self,
        content: str,
        author_name: str,
        message_id: Optional[str] = None,
    ) -> None:
        """Add a player message to the buffer."""
        self.buffer.add_user_message(
            content=content,
            author_name=author_name,
            message_id=message_id,
        )
        self._message_count += 1

        # Compact overflow messages into running summary when buffer overflows
        if self.buffer.has_pending_compaction:
            await self._compact_overflow()

        # Check if we should generate a session-level summary
        if self._should_summarize():
            await self._generate_incremental_summary()

    async def add_dm_response(
        self,
        content: str,
        is_narration: bool = True,
    ) -> None:
        """Add a DM response to the buffer."""
        self.buffer.add_assistant_message(
            content=content,
            is_dm_narration=is_narration,
        )
        self._message_count += 1

    async def _compact_overflow(self) -> None:
        """Compact overflow messages into a running narrative summary.

        Inspired by the continuation strategy from agentic orchestration:
        summarize old messages, merge with existing summary, preserve recent verbatim.
        """
        overflow_text = self.buffer.get_overflow_text()
        if not overflow_text:
            return

        try:
            client = get_ollama_client()

            compact_prompt = (
                "Summarize this D&D game conversation into a brief narrative paragraph "
                "(3-5 sentences). Focus on what happened, key decisions, and story progression. "
                "Write in past tense, third person.\n\n"
                f"Conversation:\n{overflow_text}"
            )

            response = await client.chat(
                messages=[
                    {"role": "system", "content": "You are a concise D&D session summarizer."},
                    {"role": "user", "content": compact_prompt},
                ],
                temperature=0.3,
                max_tokens=200,
                think=False,
            )

            summary = response.content.strip() if response.content else ""
            if summary:
                self.buffer.compact(summary)
                logger.info(
                    "overflow_compacted",
                    campaign_id=self.campaign_id,
                    overflow_messages=len(self.buffer._overflow_buffer),
                    summary_length=len(summary),
                )
            else:
                # Failed to summarize — just clear overflow to prevent infinite growth
                self.buffer._overflow_buffer.clear()

        except Exception as e:
            logger.warning("overflow_compaction_failed", error=str(e))
            # Clear overflow to prevent re-trying the same batch
            self.buffer._overflow_buffer.clear()

    def add_system_event(self, content: str) -> None:
        """Add a system event (combat result, roll, etc.)."""
        self.buffer.add_system_message(content)

    def set_combat_state(self, in_combat: bool) -> None:
        """Update combat state for gated consolidation."""
        self._is_in_combat = in_combat

    def _should_summarize(self) -> bool:
        """Three-gate check for memory consolidation.

        All gates must pass before summarization runs:
        1. Message gate: enough new messages since last summary
        2. Time gate: enough wall-clock time elapsed (prevents rapid-fire spam)
        3. Combat gate: not in active combat (combat is too fast-paced)
        4. Lock gate: no concurrent consolidation in progress
        """
        # Gate 1: enough messages
        messages_since = self._message_count - self._last_summary_at
        if messages_since < self._gate_min_messages:
            return False

        # Gate 2: enough time elapsed
        elapsed = time.monotonic() - self._last_summary_time
        if elapsed < self._gate_min_elapsed_sec:
            return False

        # Gate 3: not in combat (optional, can be disabled)
        if self._gate_not_in_combat and self._is_in_combat:
            return False

        # Gate 4: not already consolidating
        if self._is_consolidating:
            return False

        return True

    async def _generate_incremental_summary(self) -> None:
        """Generate a summary of recent messages (with lock gate)."""
        self._is_consolidating = True
        try:
            client = get_ollama_client()

            # Get conversation text
            conversation = self.buffer.get_summary_text()
            if not conversation:
                return

            # Generate summary using LLM
            prompt = SUMMARIZE_PROMPT.format(conversation=conversation)

            response = await client.chat(
                messages=[
                    {"role": "system", "content": "You are a helpful assistant that summarizes D&D sessions."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.3,
                max_tokens=300,
                think=False,
            )
            response = response.content or ""

            # Parse response
            try:
                # Try to extract JSON from response
                response_text = response.strip()
                if "```json" in response_text:
                    response_text = response_text.split("```json")[1].split("```")[0]
                elif "```" in response_text:
                    response_text = response_text.split("```")[1].split("```")[0]

                data = json.loads(response_text)

                summary = SessionSummary(
                    session_id=f"incremental_{self._message_count}",
                    campaign_id=self.campaign_id,
                    summary=data.get("summary", ""),
                    key_events=data.get("key_events", []),
                    npcs_encountered=data.get("npcs", []),
                    locations_visited=data.get("locations", []),
                    message_count=self._gate_min_messages,
                )

                self._session_summaries.append(summary)

                # Store in vector DB for RAG
                self.vector_store.add_session_summary(
                    campaign_id=self.campaign_id,
                    session_id=summary.session_id,
                    summary=summary.summary,
                    key_events=summary.key_events,
                )

                # Update NPCs in core memory if new ones
                for npc in summary.npcs_encountered:
                    current_npcs = self.core.get_block("npcs")
                    if current_npcs and npc.lower() not in current_npcs.content.lower():
                        self.core.add_npc(npc)

                self._last_summary_at = self._message_count
                self._last_summary_time = time.monotonic()

                logger.info(
                    "incremental_summary_generated",
                    campaign_id=self.campaign_id,
                    message_count=self._message_count,
                )

            except (json.JSONDecodeError, KeyError, ValueError) as parse_err:
                logger.warning(
                    "summary_json_parse_failed_using_raw",
                    error=str(parse_err)[:80],
                    response_preview=response[:100],
                )
                # Fallback: use the raw response as a plain text summary
                raw_summary = response.strip()
                if raw_summary and len(raw_summary) > 20:
                    summary = SessionSummary(
                        session_id=f"incremental_{self._message_count}",
                        campaign_id=self.campaign_id,
                        summary=raw_summary[:500],  # Cap length
                        key_events=[],
                        npcs_encountered=[],
                        locations_visited=[],
                        message_count=self._gate_min_messages,
                    )
                    self._session_summaries.append(summary)
                    self._last_summary_at = self._message_count
                    self._last_summary_time = time.monotonic()

        except Exception as e:
            logger.error(
                "summary_generation_failed",
                campaign_id=self.campaign_id,
                error=str(e),
            )
        finally:
            self._is_consolidating = False

    def build_context(self, current_input: str = "") -> str:
        """
        Build the full context for the LLM.

        Includes:
        1. Core memory blocks
        2. Running compacted narrative (story so far)
        3. Recalled memories from RAG (if relevant)
        4. Recent session summaries
        """
        parts = []

        # Core memory
        parts.append(self.core.to_context_string())
        parts.append("")

        # Running compacted narrative from message overflow
        if self.buffer.running_summary:
            parts.append("<story_so_far>")
            parts.append(self.buffer.running_summary)
            parts.append("</story_so_far>")
            parts.append("")

        # RAG recall for relevant past context
        if current_input:
            recalled = self.vector_store.recall_for_context(
                campaign_id=self.campaign_id,
                current_situation=current_input,
                max_results=3,
            )
            if recalled:
                parts.append(recalled)
                parts.append("")

        # Recent session summaries (last 2)
        if self._session_summaries:
            recent = self._session_summaries[-2:]
            parts.append("<recent_session_context>")
            for summary in recent:
                parts.append(f"Previous: {summary.summary}")
            parts.append("</recent_session_context>")
            parts.append("")

        return "\n".join(parts)

    def get_message_history(self, limit: Optional[int] = None) -> list[dict]:
        """Get message history for LLM API."""
        return self.buffer.get_for_llm(limit)

    def update_scene(self, description: str) -> None:
        """Update the current scene context."""
        self.core.update_scene(description)

    def update_party_status(self, status: str) -> None:
        """Update the party status."""
        self.core.update_party(status)

    def set_world_setting(self, setting: str) -> None:
        """Set the world setting."""
        self.core.update_world(setting)

    def add_quest(self, quest: str) -> None:
        """Add a quest to active quests."""
        self.core.add_quest(quest)

    def add_npc(self, npc: str, description: str = "") -> None:
        """Add an NPC to memory."""
        self.core.add_npc(npc)

        # Also add to vector store for RAG
        if description:
            import uuid
            self.vector_store.add_npc(
                campaign_id=self.campaign_id,
                npc_id=str(uuid.uuid4()),
                name=npc,
                description=description,
            )

    def add_location(self, name: str, description: str) -> None:
        """Add a location to memory."""
        import uuid
        self.vector_store.add_location(
            campaign_id=self.campaign_id,
            location_id=str(uuid.uuid4()),
            name=name,
            description=description,
        )

    def add_event(self, description: str) -> None:
        """Record a significant event."""
        import uuid
        self.vector_store.add_event(
            campaign_id=self.campaign_id,
            event_id=str(uuid.uuid4()),
            description=description,
        )

    async def end_session(self) -> Optional[SessionSummary]:
        """End the session and generate a final summary."""
        if len(self.buffer) < 5:
            # Not enough content to summarize
            return None

        await self._generate_incremental_summary()

        # Clear buffer for next session
        self.buffer.clear()
        self._message_count = 0
        self._last_summary_at = 0

        return self._session_summaries[-1] if self._session_summaries else None

    def to_dict(self) -> dict:
        """Serialize manager state for persistence."""
        return {
            "campaign_id": self.campaign_id,
            "core": self.core.to_dict(),
            "message_count": self._message_count,
            "last_summary_at": self._last_summary_at,
            "session_summaries": [
                {
                    "session_id": s.session_id,
                    "campaign_id": s.campaign_id,
                    "summary": s.summary,
                    "key_events": s.key_events,
                    "npcs_encountered": s.npcs_encountered,
                    "locations_visited": s.locations_visited,
                    "created_at": s.created_at.isoformat(),
                    "message_count": s.message_count,
                }
                for s in self._session_summaries
            ],
        }

    @classmethod
    def from_dict(cls, data: dict) -> "MemoryManager":
        """Deserialize manager state."""
        manager = cls(data["campaign_id"])

        # Restore core memory
        if "core" in data:
            manager.core = CoreMemory.from_dict(data["core"])

        # Restore counters
        manager._message_count = data.get("message_count", 0)
        manager._last_summary_at = data.get("last_summary_at", 0)

        # Restore session summaries
        for s in data.get("session_summaries", []):
            manager._session_summaries.append(SessionSummary(
                session_id=s["session_id"],
                campaign_id=s["campaign_id"],
                summary=s["summary"],
                key_events=s.get("key_events", []),
                npcs_encountered=s.get("npcs_encountered", []),
                locations_visited=s.get("locations_visited", []),
                created_at=datetime.fromisoformat(s.get("created_at", datetime.utcnow().isoformat())),
                message_count=s.get("message_count", 0),
            ))

        return manager


# Active memory managers by campaign (bounded to prevent memory leaks)
_managers: dict[str, MemoryManager] = {}
_MAX_CACHED_MANAGERS = 50


async def get_memory_manager(campaign_id: str) -> MemoryManager:
    """Get or create a memory manager for a campaign.

    Loads persisted state from the database on first access.
    Evicts oldest cached managers if cache exceeds _MAX_CACHED_MANAGERS.
    """
    if campaign_id not in _managers:
        # Evict oldest entries if at capacity
        if len(_managers) >= _MAX_CACHED_MANAGERS:
            oldest_key = next(iter(_managers))
            logger.info("memory_manager_evicted", campaign_id=oldest_key)
            _managers.pop(oldest_key)

        manager = await _load_memory_state(campaign_id)
        if manager is None:
            manager = MemoryManager(campaign_id)
        _managers[campaign_id] = manager
    return _managers[campaign_id]


def get_memory_manager_sync(campaign_id: str) -> MemoryManager:
    """Sync fallback — get cached manager or create new (no DB load)."""
    if campaign_id not in _managers:
        if len(_managers) >= _MAX_CACHED_MANAGERS:
            oldest_key = next(iter(_managers))
            _managers.pop(oldest_key)
        _managers[campaign_id] = MemoryManager(campaign_id)
    return _managers[campaign_id]


async def save_memory_state(campaign_id: str) -> None:
    """Persist memory manager state to the database."""
    if campaign_id not in _managers:
        return

    try:
        from ..data.database import get_database
        db = await get_database()
        manager = _managers[campaign_id]
        state_json = json.dumps(manager.to_dict())

        await db.execute(
            """
            INSERT OR REPLACE INTO campaign_memory (id, campaign_id, memory_type, content, metadata)
            VALUES (?, ?, 'manager_state', ?, '{}')
            """,
            (f"{campaign_id}-state", campaign_id, state_json),
        )
        await db.commit()
        logger.debug("memory_state_saved", campaign_id=campaign_id)
    except Exception as e:
        logger.warning("memory_state_save_failed", campaign_id=campaign_id, error=str(e))


async def _load_memory_state(campaign_id: str) -> Optional[MemoryManager]:
    """Load persisted memory manager state from the database."""
    try:
        from ..data.database import get_database
        db = await get_database()
        row = await db.fetch_one(
            "SELECT content FROM campaign_memory WHERE id = ? AND memory_type = 'manager_state'",
            (f"{campaign_id}-state",),
        )
        if row and row[0]:
            data = json.loads(row[0])
            manager = MemoryManager.from_dict(data)
            logger.info("memory_state_loaded", campaign_id=campaign_id)
            return manager
    except Exception as e:
        logger.warning("memory_state_load_failed", campaign_id=campaign_id, error=str(e))
    return None


def clear_memory_manager(campaign_id: str) -> None:
    """Clear a memory manager."""
    _managers.pop(campaign_id, None)
