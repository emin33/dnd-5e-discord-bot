"""Core memory blocks for the MemGPT-inspired memory system."""

from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from typing import Optional
from collections import deque


class MemoryBlockType(str, Enum):
    """Types of core memory blocks."""
    WORLD = "world"  # World setting, geography, lore
    PARTY = "party"  # Party status, composition, dynamics
    SCENE = "scene"  # Current scene context
    QUESTS = "quests"  # Active quests and objectives
    NPCS = "npcs"  # Key NPCs the party has met
    CUSTOM = "custom"  # DM-defined blocks


@dataclass
class MemoryBlock:
    """A block of core memory that's always in context."""

    block_type: MemoryBlockType
    name: str
    content: str
    max_tokens: int = 500  # Approximate token limit
    priority: int = 1  # Higher = more important
    last_updated: datetime = field(default_factory=datetime.utcnow)
    campaign_id: Optional[str] = None

    def update(self, new_content: str) -> None:
        """Update the block's content."""
        self.content = new_content
        self.last_updated = datetime.utcnow()

    def append(self, additional_content: str, separator: str = "\n") -> None:
        """Append to the block's content."""
        if self.content:
            self.content = f"{self.content}{separator}{additional_content}"
        else:
            self.content = additional_content
        self.last_updated = datetime.utcnow()

    def to_context_string(self) -> str:
        """Format the block for inclusion in LLM context."""
        return f"<{self.name}>\n{self.content}\n</{self.name}>"

    def estimate_tokens(self) -> int:
        """Rough token estimate (4 chars per token)."""
        return len(self.content) // 4


@dataclass
class Message:
    """A message in the conversation history."""

    role: str  # "user", "assistant", "system"
    content: str
    author_name: Optional[str] = None  # Display name for user messages
    timestamp: datetime = field(default_factory=datetime.utcnow)
    message_id: Optional[str] = None  # Discord message ID
    is_dm_narration: bool = False  # True if this was DM narration

    def to_dict(self) -> dict:
        """Convert to dict for LLM API."""
        return {
            "role": self.role,
            "content": self.content,
        }

    def to_context_string(self) -> str:
        """Format for context display."""
        if self.author_name:
            return f"**{self.author_name}:** {self.content}"
        return self.content


class MessageBuffer:
    """
    Three-tier sliding window buffer with gradual context compaction.

    Tier 1 (Verbatim): Last N messages in full prose — the narrator
        sees exactly what was said.
    Tier 2 (Condensed): Per-exchange summaries (1-2 sentences each) —
        the narrator knows what happened but not every word.
    Tier 3 (Compressed): Batch narrative summary + pinned facts —
        broad strokes of older events.

    Messages flow: Verbatim → Condensation buffer → Condensed →
                   Overflow buffer → Running summary + Pinned facts
    """

    def __init__(
        self,
        max_messages: int = 20,
        verbatim_size: int = 8,
        condensed_size: int = 12,
        preserve_recent: int = 4,
    ):
        self.max_messages = max_messages       # Kept for compat
        self._verbatim_size = verbatim_size    # Tier 1 cap
        self._condensed_size = condensed_size  # Tier 2 cap
        self.preserve_recent = preserve_recent

        # Tier 1: Full prose messages
        self._messages: list[Message] = []

        # Tier 1→2 transition buffer
        self._condensation_buffer: list[Message] = []

        # Tier 2: Per-exchange summaries
        self._condensed: list[str] = []

        # Tier 2→3 transition buffer (existing)
        self._overflow_buffer: list[Message] = []

        # Tier 3: Batch narrative + facts (existing)
        self._running_summary: str = ""
        self._pinned_facts: list[str] = []

    def add(self, message: Message) -> None:
        """Add a message. Overflow cascades through tiers."""
        self._messages.append(message)
        # Tier 1 overflow → condensation buffer
        while len(self._messages) > self._verbatim_size:
            aged = self._messages.pop(0)
            self._condensation_buffer.append(aged)

    # ------------------------------------------------------------------
    # Tier 1→2: Condensation
    # ------------------------------------------------------------------

    @property
    def has_pending_condensation(self) -> bool:
        """True when enough messages are waiting for condensation (~2 exchanges)."""
        return len(self._condensation_buffer) >= 4

    def get_condensation_text(self) -> str:
        """Get condensation buffer as text for LLM summarization."""
        return self._format_messages(self._condensation_buffer)

    def condense(self, summaries: list[str]) -> None:
        """Store per-exchange summaries from condensation, clear the buffer.

        If Tier 2 overflows, oldest summaries cascade to the Tier 3 overflow buffer.
        """
        self._condensed.extend(summaries)
        self._condensation_buffer.clear()

        # Tier 2 overflow → Tier 3 overflow buffer
        while len(self._condensed) > self._condensed_size:
            old_summary = self._condensed.pop(0)
            self._overflow_buffer.append(Message(
                role="system",
                content=old_summary,
                is_dm_narration=False,
            ))

    @property
    def condensed_summaries(self) -> list[str]:
        """Tier 2 per-exchange summaries for context building."""
        return list(self._condensed)

    # ------------------------------------------------------------------
    # Tier 2→3: Compaction (existing, unchanged)
    # ------------------------------------------------------------------

    @property
    def has_pending_compaction(self) -> bool:
        """True if overflow buffer has messages waiting for compaction."""
        threshold = getattr(self, '_compaction_threshold', 6)
        return len(self._overflow_buffer) >= threshold

    def get_overflow_text(self) -> str:
        """Get overflow messages as text for summarization."""
        return self._format_messages(self._overflow_buffer)

    def compact(self, summary: str, extracted_facts: Optional[list[str]] = None) -> None:
        """Merge a new summary with existing running summary, clear overflow."""
        if self._running_summary:
            self._running_summary = f"{self._running_summary}\n\n{summary}"
        else:
            self._running_summary = summary

        if extracted_facts:
            for fact in extracted_facts:
                fact = fact.strip()
                if fact and fact not in self._pinned_facts:
                    self._pinned_facts.append(fact)

        self._overflow_buffer.clear()

    @property
    def running_summary(self) -> str:
        """The running compacted narrative of older exchanges."""
        return self._running_summary

    @property
    def pinned_facts(self) -> list[str]:
        """Typed facts extracted during compaction that must survive indefinitely."""
        return self._pinned_facts

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _format_messages(self, messages: list[Message]) -> str:
        """Format a list of messages as text for LLM summarization."""
        lines = []
        for msg in messages:
            if msg.author_name:
                lines.append(f"{msg.author_name}: {msg.content}")
            elif msg.role == "assistant":
                lines.append(f"DM: {msg.content}")
            else:
                lines.append(msg.content)
        return "\n\n".join(lines)

    def add_user_message(
        self,
        content: str,
        author_name: str,
        message_id: Optional[str] = None,
    ) -> Message:
        """Add a user message."""
        msg = Message(
            role="user",
            content=content,
            author_name=author_name,
            message_id=message_id,
        )
        self.add(msg)
        return msg

    def add_assistant_message(
        self,
        content: str,
        is_dm_narration: bool = True,
    ) -> Message:
        """Add an assistant (DM) message."""
        msg = Message(
            role="assistant",
            content=content,
            is_dm_narration=is_dm_narration,
        )
        self.add(msg)
        return msg

    def add_system_message(self, content: str) -> Message:
        """Add a system message (e.g., combat results)."""
        msg = Message(
            role="system",
            content=content,
        )
        self.add(msg)
        return msg

    def get_messages(self, limit: Optional[int] = None) -> list[Message]:
        """Get messages, optionally limited to most recent N."""
        messages = list(self._messages)
        if limit:
            messages = messages[-limit:]
        return messages

    def get_for_llm(self, limit: Optional[int] = None) -> list[dict]:
        """Get messages formatted for LLM API."""
        return [m.to_dict() for m in self.get_messages(limit)]

    def get_summary_text(self) -> str:
        """Get all messages as text for summarization."""
        lines = []
        for msg in self._messages:
            if msg.author_name:
                lines.append(f"{msg.author_name}: {msg.content}")
            elif msg.role == "assistant":
                lines.append(f"DM: {msg.content}")
            else:
                lines.append(msg.content)
        return "\n\n".join(lines)

    def clear(self) -> None:
        """Clear the buffer."""
        self._messages.clear()

    def __len__(self) -> int:
        return len(self._messages)


@dataclass
class SessionSummary:
    """Summary of a past game session."""

    session_id: str
    campaign_id: str
    summary: str
    key_events: list[str]
    npcs_encountered: list[str]
    locations_visited: list[str]
    created_at: datetime = field(default_factory=datetime.utcnow)
    message_count: int = 0


class CoreMemory:
    """
    The core memory system with persistent blocks.

    These blocks are always included in the LLM context,
    providing consistent world knowledge and party status.
    """

    def __init__(self, campaign_id: str):
        self.campaign_id = campaign_id
        self._blocks: dict[str, MemoryBlock] = {}
        self._initialize_default_blocks()

    def _initialize_default_blocks(self) -> None:
        """Create default core memory blocks."""
        self._blocks["world"] = MemoryBlock(
            block_type=MemoryBlockType.WORLD,
            name="World Setting",
            content="A standard D&D fantasy world.",
            max_tokens=500,
            priority=1,
            campaign_id=self.campaign_id,
        )

        self._blocks["party"] = MemoryBlock(
            block_type=MemoryBlockType.PARTY,
            name="Party Status",
            content="The party has just formed.",
            max_tokens=400,
            priority=2,
            campaign_id=self.campaign_id,
        )

        self._blocks["scene"] = MemoryBlock(
            block_type=MemoryBlockType.SCENE,
            name="Current Scene",
            content="The adventure is about to begin.",
            max_tokens=300,
            priority=3,
            campaign_id=self.campaign_id,
        )

        self._blocks["quests"] = MemoryBlock(
            block_type=MemoryBlockType.QUESTS,
            name="Active Quests",
            content="No active quests yet.",
            max_tokens=400,
            priority=2,
            campaign_id=self.campaign_id,
        )

        self._blocks["npcs"] = MemoryBlock(
            block_type=MemoryBlockType.NPCS,
            name="Key NPCs",
            content="No notable NPCs encountered yet.",
            max_tokens=400,
            priority=2,
            campaign_id=self.campaign_id,
        )

    def get_block(self, name: str) -> Optional[MemoryBlock]:
        """Get a memory block by name."""
        return self._blocks.get(name)

    def set_block(self, name: str, content: str) -> MemoryBlock:
        """Set or update a memory block."""
        if name in self._blocks:
            self._blocks[name].update(content)
        else:
            # Determine block type from name
            block_type = MemoryBlockType.CUSTOM
            for bt in MemoryBlockType:
                if bt.value == name.lower():
                    block_type = bt
                    break

            self._blocks[name] = MemoryBlock(
                block_type=block_type,
                name=name,
                content=content,
                campaign_id=self.campaign_id,
            )
        return self._blocks[name]

    def update_world(self, content: str) -> None:
        """Update the world setting block."""
        self._blocks["world"].update(content)

    def update_party(self, content: str) -> None:
        """Update the party status block."""
        self._blocks["party"].update(content)

    def update_scene(self, content: str) -> None:
        """Update the current scene block."""
        self._blocks["scene"].update(content)

    def add_quest(self, quest: str) -> None:
        """Add a quest to the active quests."""
        if self._blocks["quests"].content == "No active quests yet.":
            self._blocks["quests"].update(f"- {quest}")
        else:
            self._blocks["quests"].append(f"- {quest}")

    def add_npc(self, npc: str) -> None:
        """Add an NPC to the key NPCs."""
        if self._blocks["npcs"].content == "No notable NPCs encountered yet.":
            self._blocks["npcs"].update(f"- {npc}")
        else:
            self._blocks["npcs"].append(f"- {npc}")

    def get_all_blocks(self) -> list[MemoryBlock]:
        """Get all blocks sorted by priority."""
        return sorted(
            self._blocks.values(),
            key=lambda b: b.priority,
            reverse=True,
        )

    def to_context_string(self) -> str:
        """Generate the full core memory context string."""
        blocks = self.get_all_blocks()
        parts = [
            "<core_memory>",
            "These are persistent facts about the campaign that you must maintain:",
            "",
        ]

        for block in blocks:
            parts.append(block.to_context_string())
            parts.append("")

        parts.append("</core_memory>")
        return "\n".join(parts)

    def estimate_tokens(self) -> int:
        """Estimate total tokens for all blocks."""
        return sum(block.estimate_tokens() for block in self._blocks.values())

    def to_dict(self) -> dict:
        """Serialize to dict for persistence."""
        return {
            "campaign_id": self.campaign_id,
            "blocks": {
                name: {
                    "type": block.block_type.value,
                    "name": block.name,
                    "content": block.content,
                    "max_tokens": block.max_tokens,
                    "priority": block.priority,
                    "last_updated": block.last_updated.isoformat(),
                }
                for name, block in self._blocks.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "CoreMemory":
        """Deserialize from dict."""
        memory = cls(data["campaign_id"])

        for name, block_data in data.get("blocks", {}).items():
            memory._blocks[name] = MemoryBlock(
                block_type=MemoryBlockType(block_data["type"]),
                name=block_data["name"],
                content=block_data["content"],
                max_tokens=block_data.get("max_tokens", 500),
                priority=block_data.get("priority", 1),
                last_updated=datetime.fromisoformat(block_data.get("last_updated", datetime.utcnow().isoformat())),
                campaign_id=data["campaign_id"],
            )

        return memory
