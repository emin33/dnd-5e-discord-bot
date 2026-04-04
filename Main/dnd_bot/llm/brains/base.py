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

    # Authoritative world state (YAML-serialized for narrator bookend injection)
    world_state_yaml: str = ""

    # Previous turn mechanical trace (injected in bottom reminder for grounding)
    last_turn_trace: str = ""


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

    def _build_bookend_messages(self, context: BrainContext) -> list[dict]:
        """Build messages using bookend layout for narrator calls.

        Based on "Lost in the Middle" research: LLMs attend strongly to the
        beginning and end of context, with a 30%+ accuracy drop for info in
        the middle. This layout exploits both primacy and recency bias:

        [SYSTEM]  — persona + stable rules (cacheable)
        [USER 1]  — HIGH ATTENTION: world state YAML + party + scene entities
        [ASST 1]  — acknowledgment anchor
        [USER 2]  — MIDDLE: session summary + compressed history (lower attention OK)
        [ASST 2]  — acknowledgment anchor
        ...recent_messages (last 5-8 verbatim)...
        [USER N]  — HIGH ATTENTION: player action + grounding reminders
        """
        messages = []

        # ── SYSTEM: Stable persona + behavioral rules (cacheable) ──
        if self.system_prompt:
            system_content = self.system_prompt

            # Combat context goes in system (it's structural, not dynamic facts)
            if context.combat_state:
                system_content += f"\n\n## Combat\n{context.combat_state}"
            elif context.in_combat:
                system_content += f"\n\n## Combat State\nRound: {context.combat_round}"
                if context.initiative_order:
                    system_content += f"\nInitiative Order:\n{context.initiative_order}"
                if context.current_combatant:
                    system_content += f"\nCurrent Turn: {context.current_combatant}"

            messages.append({"role": "system", "content": system_content})

        # ── USER 1 (HIGH ATTENTION): Authoritative world state ──
        top_parts = []

        if context.world_state_yaml:
            top_parts.append(f"<world_state>\n{context.world_state_yaml}</world_state>")

        # Party status (critical for narrator to know HP, conditions)
        party = context.party_members or context.party_status
        if party:
            top_parts.append(f"<party>\n{party}\n</party>")

        # Scene entities and current scene
        if context.current_scene:
            top_parts.append(f"<current_scene>\n{context.current_scene}\n</current_scene>")

        # Active quests
        if context.active_quests:
            top_parts.append(f"<active_quests>\n{context.active_quests}\n</active_quests>")

        # Acting character details
        if context.character_stats:
            char_stats = context.character_stats if isinstance(context.character_stats, str) else str(context.character_stats)
            top_parts.append(f"<acting_character>\n{char_stats}\n</acting_character>")

        if top_parts:
            messages.append({"role": "user", "content": "\n\n".join(top_parts)})
            messages.append({
                "role": "assistant",
                "content": "I have the current world state. I will use it as authoritative ground truth for my narration.",
            })

        # ── MIDDLE (LOWER ATTENTION): Compressed history ──
        middle_parts = []

        # Memory context (core blocks + RAG recall + pinned facts)
        if context.memory_context:
            middle_parts.append(f"<memory>\n{context.memory_context}\n</memory>")

        # Session summary
        if context.session_summary:
            middle_parts.append(f"<session_history>\n{context.session_summary}\n</session_history>")

        if middle_parts:
            messages.append({"role": "user", "content": "\n\n".join(middle_parts)})
            messages.append({
                "role": "assistant",
                "content": "I have the session history and memory context.",
            })

        # ── Recent messages (verbatim, last 5-8) ──
        history = context.message_history or context.recent_messages
        # Limit to last 8 messages to keep context lean
        recent = history[-8:] if len(history) > 8 else history
        messages.extend(recent)

        # ── FINAL USER (HIGH ATTENTION): Player action + reminders ──
        if context.player_action:
            bottom_parts = [f"<player_action>[{context.player_name}]: {context.player_action}</player_action>"]

            # Grounding reminders at the very end (high attention zone)
            reminders = []
            if context.last_turn_trace:
                reminders.append(f"Last turn: {context.last_turn_trace}")
            reminders.append("Use ONLY the world state provided above as ground truth.")
            reminders.append("Every NPC you mention must appear in the world state at their listed location.")
            if context.in_combat:
                reminders.append(f"Combat is active. Current round: {context.combat_round}.")

            bottom_parts.append(f"<reminder>\n" + "\n".join(reminders) + "\n</reminder>")

            messages.append({"role": "user", "content": "\n\n".join(bottom_parts)})

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
