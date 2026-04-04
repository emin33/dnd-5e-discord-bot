"""Effects Adjudicator - Parses INTENTS block into ProposedEffects.

This module does NOT use LLM inference. It's deterministic string parsing.

The narrator outputs:
```
PROSE:
[creative narration]

INTENTS:
[explicit intent commands]
```

The adjudicator parses INTENTS into ProposedEffect objects.
Prose is NEVER used as the source of truth for mechanics.
"""

from typing import Optional

from ..effects import ProposedEffect
from ..intents import extract_intents_block, parse_intents, IntentParseResult

import structlog

logger = structlog.get_logger()


class EffectsAdjudicator:
    """
    Deterministic intent parser.

    Parses the INTENTS block from narrator output into ProposedEffect objects.
    Does NOT use LLM inference - this is pure string parsing.
    """

    def parse_narrator_response(
        self,
        response: str,
    ) -> tuple[str, list[ProposedEffect], IntentParseResult]:
        """
        Parse narrator response into prose and effects.

        Args:
            response: Full narrator response with PROSE and INTENTS blocks

        Returns:
            Tuple of (prose, effects_list, parse_result)
        """
        # Extract PROSE and INTENTS blocks
        prose, intents_block = extract_intents_block(response)

        # Parse INTENTS into ProposedEffects
        parse_result = parse_intents(intents_block)

        if parse_result.errors:
            logger.warning(
                "intent_parse_errors",
                error_count=len(parse_result.errors),
                errors=parse_result.errors[:3],  # Log first 3
            )

        logger.debug(
            "parsed_narrator_response",
            prose_length=len(prose),
            intents_count=len(parse_result.effects),
            had_none=parse_result.had_none,
        )

        return prose, parse_result.effects, parse_result

    def extract_effects(
        self,
        intents_block: str,
    ) -> list[ProposedEffect]:
        """
        Parse an INTENTS block directly into effects.

        Args:
            intents_block: Just the INTENTS section (not full response)

        Returns:
            List of ProposedEffect objects
        """
        result = parse_intents(intents_block)
        return result.effects


# Global adjudicator instance
_adjudicator: Optional[EffectsAdjudicator] = None


def get_adjudicator() -> EffectsAdjudicator:
    """Get the global effects adjudicator."""
    global _adjudicator
    if _adjudicator is None:
        _adjudicator = EffectsAdjudicator()
    return _adjudicator


def _reset_adjudicator():
    """Clear cached adjudicator so it recreates from the active profile."""
    global _adjudicator
    _adjudicator = None
