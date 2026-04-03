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
    Sliding window buffer with context compaction.

    Keeps the last N messages for immediate context. When messages
    overflow, they're captured in a compaction buffer for summarization
    into a running "story so far" block (inspired by agentic orchestration
    continuation strategy — summarize old, preserve recent verbatim).
    """

    def __init__(self, max_messages: int = 20, preserve_recent: int = 4):
        self.max_messages = max_messages
        self.preserve_recent = preserve_recent  # Messages to keep verbatim
        self._messages: list[Message] = []
        self._overflow_buffer: list[Message] = []  # Messages waiting for compaction
        self._running_summary: str = ""  # Compacted "story so far"
        self._pinned_facts: list[str] = []  # Typed facts that survive compaction

    def add(self, message: Message) -> None:
        """Add a message to the buffer. Overflow goes to compaction buffer."""
        self._messages.append(message)
        # When we exceed max, move oldest to overflow buffer
        while len(self._messages) > self.max_messages:
            overflow_msg = self._messages.pop(0)
            self._overflow_buffer.append(overflow_msg)

    @property
    def has_pending_compaction(self) -> bool:
        """True if overflow buffer has messages waiting for compaction."""
        return len(self._overflow_buffer) >= 6  # Compact in batches

    def get_overflow_text(self) -> str:
        """Get overflow messages as text for summarization."""
        lines = []
        for msg in self._overflow_buffer:
            if msg.author_name:
                lines.append(f"{msg.author_name}: {msg.content}")
            elif msg.role == "assistant":
                lines.append(f"DM: {msg.content}")
            else:
                lines.append(msg.content)
        return "\n\n".join(lines)

    def compact(self, summary: str, extracted_facts: Optional[list[str]] = None) -> None:
        """Merge a new summary with existing running summary, clear overflow.

        Args:
            summary: Narrative prose summary of the compacted messages.
            extracted_facts: Typed facts (NPC names, locations, events) that must
                survive future compaction intact. These are pinned separately from
                the narrative and never re-summarized.
        """
        if self._running_summary:
            self._running_summary = f"{self._running_summary}\n\n{summary}"
        else:
            self._running_summary = summary

        # Pin extracted facts — deduplicated, never re-summarized
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
