"""Narrator tool surface — thin reads over the tool registry.

The narrator emits tool calls alongside its prose content; each tool call
maps 1:1 to a ProposedEffect so the downstream pipeline (validation,
execution, world-state sync) is unchanged.

Since REFACTOR_PLAN.md Step 1, the single authority for tool schemas, tier
membership, and tool→effect converters is ``dnd_bot.llm.tool_registry`` —
one declarative entry per tool. Adding a narrator tool means adding ONE
registration there; this module only preserves the historical import
surface (``NARRATOR_TOOLS*``, ``get_narrator_tools_for_tier``,
``tool_calls_to_effects``) and the unknown-tier fallback policy.

Tool tiers (see ``tool_registry.KNOWN_TIERS``):
- "core" (3 tools): minimum set for the smallest local narrators that
  struggle to juggle tools. The state and entity extractors cover
  everything else as fallback.
- "core_plus" (5 tools): adds change_location and start_combat. Suitable
  for capable local models and any cloud narrator.
- "full": every registered tool, for cloud narrators (DeepSeek V4
  Pro/Flash, Claude Sonnet/Haiku) that handle large tool surfaces reliably.
"""

from . import tool_registry
from .effects import ProposedEffect

import structlog

logger = structlog.get_logger()


# ── Tier lists (registry-derived; kept for the historical import surface) ──

NARRATOR_TOOLS: list[dict] = tool_registry.tools_for_tier("full")
NARRATOR_TOOLS_CORE: list[dict] = tool_registry.tools_for_tier("core")
NARRATOR_TOOLS_CORE_PLUS: list[dict] = tool_registry.tools_for_tier("core_plus")

# Map tier name → tool list. Used by the orchestrator to look up tools
# from a profile's narrator config.
NARRATOR_TOOL_TIERS: dict[str, list[dict]] = {
    "core": NARRATOR_TOOLS_CORE,
    "core_plus": NARRATOR_TOOLS_CORE_PLUS,
    "full": NARRATOR_TOOLS,
}


def get_narrator_tools_for_tier(tier: str) -> list[dict]:
    """Get the narrator tool list for a tier name.

    Falls back to ``"core"`` for unknown tier names so a typo in a
    profile doesn't break narration. Logs a warning when falling back.
    """
    tools = NARRATOR_TOOL_TIERS.get(tier)
    if tools is None:
        logger.warning(
            "narrator_tool_tier_unknown_falling_back_to_core",
            requested_tier=tier,
        )
        return NARRATOR_TOOLS_CORE
    return tools


# ======================================================================
# Tool Call → ProposedEffect Converter
# ======================================================================


def tool_calls_to_effects(tool_calls: list[dict]) -> list[ProposedEffect]:
    """Convert narrator tool calls to ProposedEffect objects.

    Each tool call maps 1:1 to a ProposedEffect via the registered
    converter. Unknown tool names are logged and skipped.
    """
    effects = []

    for tc in tool_calls:
        name = tc.get("name", "")
        args = tc.get("arguments", {})

        try:
            effect = tool_registry.convert_tool_call(name, args)
            if effect:
                effects.append(effect)
        except Exception as e:
            logger.warning("tool_call_conversion_failed", tool=name, error=str(e), exc_info=True)

    return effects


# Kept as a named seam for unit tests; the registry owns the dispatch.
_convert_tool_call = tool_registry.convert_tool_call
