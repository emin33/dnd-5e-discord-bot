"""Base class for LLM brains."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Optional

from ..client import LLMResponse, OllamaClient


@dataclass
class BrainContext:
    """Context passed to brain for decision making."""

    # Session identifiers
    campaign_id: str = ""
    session_id: str = ""

    # Current game state
    party_members: str = ""  # List of party members and their status
    party_status: str = ""  # Alias for compatibility
    current_scene: str = ""
    active_quests: str = ""

    # Combat state (if in combat)
    in_combat: bool = False
    combat_state: str = ""  # Full combat context
    combat_round: int = 0
    current_combatant: str = ""
    initiative_order: str = ""

    # Memory context (from MemGPT-style system)
    memory_context: str = ""  # Core memory blocks + RAG results

    # Recent history
    recent_messages: list[dict] = field(default_factory=list)
    message_history: list[dict] = field(default_factory=list)  # Alias
    session_summary: str = ""

    # The current action being processed
    player_action: str = ""
    player_name: str = ""

    # Mechanical context (for Rules Brain)
    character_stats: Optional[dict] = None
    available_actions: list[str] = field(default_factory=list)


@dataclass
class BrainResult:
    """Result from a brain's processing."""

    content: str
    tool_calls: list[dict] = field(default_factory=list)
    metadata: dict = field(default_factory=dict)
    raw_response: str = ""  # Original response for debugging

    # Proposed effects from narrator (validated/executed by orchestrator)
    proposed_effects: list = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return len(self.tool_calls) > 0

    @property
    def has_proposed_effects(self) -> bool:
        return len(self.proposed_effects) > 0


class Brain(ABC):
    """Abstract base class for LLM brains."""

    def __init__(
        self,
        client: OllamaClient,
        temperature: float = 0.7,
        system_prompt: str = "",
    ):
        self.client = client
        self.temperature = temperature
        self.system_prompt = system_prompt

    @abstractmethod
    async def process(self, context: BrainContext) -> BrainResult:
        """Process the context and return a result."""
        pass

    def _build_messages(self, context: BrainContext) -> list[dict]:
        """Build the messages array for the LLM."""
        messages = []

        # System prompt with full context
        if self.system_prompt:
            system_content = self.system_prompt

            # Add memory context (includes core blocks + RAG results)
            if context.memory_context:
                system_content += f"\n\n{context.memory_context}"

            # Add party info
            party = context.party_members or context.party_status
            if party:
                system_content += f"\n\n## Party\n{party}"

            # Add current scene
            if context.current_scene:
                system_content += f"\n\n## Current Scene\n{context.current_scene}"

            # Add quests
            if context.active_quests:
                system_content += f"\n\n## Active Quests\n{context.active_quests}"

            # Add acting character details
            if context.character_stats:
                char_stats = context.character_stats if isinstance(context.character_stats, str) else str(context.character_stats)
                system_content += f"\n\n## Acting Character\n{char_stats}"

            # Add combat state
            if context.combat_state:
                system_content += f"\n\n## Combat\n{context.combat_state}"
            elif context.in_combat:
                system_content += f"\n\n## Combat State\nRound: {context.combat_round}"
                if context.initiative_order:
                    system_content += f"\nInitiative Order:\n{context.initiative_order}"
                if context.current_combatant:
                    system_content += f"\nCurrent Turn: {context.current_combatant}"

            messages.append({"role": "system", "content": system_content})

        # Session summary if available
        if context.session_summary:
            messages.append({
                "role": "system",
                "content": f"## Session Summary\n{context.session_summary}",
            })

        # Recent messages (use message_history if available, fall back to recent_messages)
        history = context.message_history or context.recent_messages
        messages.extend(history)

        # Current player action
        if context.player_action:
            messages.append({
                "role": "user",
                "content": f"[{context.player_name}]: {context.player_action}",
            })

        return messages

    def _parse_response(self, response: LLMResponse) -> BrainResult:
        """Parse an LLM response into a BrainResult."""
        return BrainResult(
            content=response.content,
            tool_calls=response.tool_calls,
            metadata={
                "model": response.model,
                "finish_reason": response.finish_reason,
                "tokens": response.prompt_tokens + response.completion_tokens,
            },
        )
