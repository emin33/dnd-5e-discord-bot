"""Dynamic SRD Rule Injection.

Instead of bloating every prompt with the full SRD, this module scores
rule sections against the current action context and injects only the
top-3 most relevant rules into the narrator prompt. Keeps rules fresh
in context exactly when they're needed.

Uses lightweight keyword scoring — no vector search needed for 33 sections.
"""

import re
from typing import Optional

import structlog

logger = structlog.get_logger()

# Keyword mappings: action keywords → relevant SRD rule section names.
# When a keyword appears in the player action or triage context, the
# associated rule sections get a relevance boost.
KEYWORD_RULES = {
    # Combat
    "attack": ["Making an Attack", "Actions in Combat", "The Order of Combat"],
    "hit": ["Making an Attack", "Damage and Healing"],
    "damage": ["Damage and Healing", "Making an Attack"],
    "heal": ["Damage and Healing", "Resting"],
    "death": ["Damage and Healing"],
    "initiative": ["The Order of Combat"],
    "opportunity": ["Actions in Combat", "Movement and Position"],
    "grapple": ["Making an Attack", "Actions in Combat"],
    "shove": ["Making an Attack", "Actions in Combat"],
    "dodge": ["Actions in Combat"],
    "disengage": ["Actions in Combat", "Movement and Position"],
    "dash": ["Actions in Combat", "Movement"],
    "cover": ["Cover"],
    "mount": ["Mounted Combat"],
    "underwater": ["Underwater Combat"],

    # Ability checks
    "check": ["Ability Checks", "Using Each Ability"],
    "strength": ["Using Each Ability", "Ability Checks"],
    "dexterity": ["Using Each Ability", "Ability Checks"],
    "constitution": ["Using Each Ability", "Ability Checks"],
    "intelligence": ["Using Each Ability", "Ability Checks"],
    "wisdom": ["Using Each Ability", "Ability Checks"],
    "charisma": ["Using Each Ability", "Ability Checks"],
    "perception": ["Using Each Ability", "Ability Checks"],
    "stealth": ["Using Each Ability", "Ability Checks"],
    "investigation": ["Using Each Ability", "Ability Checks"],
    "persuasion": ["Using Each Ability", "Ability Checks"],
    "intimidation": ["Using Each Ability", "Ability Checks"],
    "deception": ["Using Each Ability", "Ability Checks"],
    "athletics": ["Using Each Ability", "Ability Checks"],
    "acrobatics": ["Using Each Ability", "Ability Checks"],
    "survival": ["Using Each Ability", "Ability Checks"],
    "advantage": ["Advantage and Disadvantage"],
    "disadvantage": ["Advantage and Disadvantage"],
    "proficiency": ["Proficiency Bonus"],
    "save": ["Saving Throws"],
    "saving throw": ["Saving Throws"],

    # Movement & exploration
    "climb": ["Movement", "Movement and Position"],
    "swim": ["Movement", "Underwater Combat"],
    "jump": ["Movement"],
    "fly": ["Movement"],
    "difficult terrain": ["Movement", "Movement and Position"],
    "travel": ["Movement", "Time"],
    "rest": ["Resting"],
    "short rest": ["Resting"],
    "long rest": ["Resting"],
    "sleep": ["Resting"],
    "camp": ["Resting"],
    "dark": ["The Environment"],
    "light": ["The Environment"],
    "vision": ["The Environment"],
    "trap": ["Traps"],

    # Magic
    "spell": ["Casting a Spell", "What Is a Spell?"],
    "cast": ["Casting a Spell"],
    "concentrate": ["Casting a Spell"],
    "concentration": ["Casting a Spell"],
    "ritual": ["Casting a Spell"],
    "cantrip": ["Casting a Spell", "What Is a Spell?"],
    "slot": ["Casting a Spell"],
    "magic item": ["Attunement", "Activating an Item", "Wearing and Wielding Items"],
    "attune": ["Attunement"],
    "potion": ["Activating an Item"],
    "scroll": ["Activating an Item"],

    # Commerce & items
    "buy": ["Standard Exchange Rates"],
    "sell": ["Standard Exchange Rates"],
    "gold": ["Standard Exchange Rates"],
    "coin": ["Standard Exchange Rates"],
    "shop": ["Standard Exchange Rates"],
    "poison": ["Poisons"],
    "disease": ["Diseases"],
}

# Maximum characters to inject per rule section (truncate long ones)
MAX_RULE_CHARS = 800
MAX_RULES_TO_INJECT = 3


class RuleInjector:
    """Scores and retrieves relevant SRD rules for narrator context."""

    def __init__(self):
        self._rules_cache: Optional[dict[str, str]] = None

    def _load_rules(self) -> dict[str, str]:
        """Load rule sections from SRD on first use."""
        if self._rules_cache is not None:
            return self._rules_cache

        try:
            from ...data.srd import get_srd
            srd = get_srd()
            sections = srd.get_all("rule_sections")
            self._rules_cache = {}
            for key, data in sections.items():
                name = data.get("name", key)
                desc = data.get("desc", "")
                if isinstance(desc, list):
                    desc = "\n".join(str(d) for d in desc)
                self._rules_cache[name] = str(desc)
            logger.info("srd_rules_loaded", count=len(self._rules_cache))
        except Exception as e:
            logger.warning("srd_rules_load_failed", error=str(e))
            self._rules_cache = {}

        return self._rules_cache

    def get_relevant_rules(
        self,
        action: str,
        triage_context: str = "",
        phase: str = "exploration",
        max_rules: int = MAX_RULES_TO_INJECT,
    ) -> str:
        """Score rule sections and return top-N as formatted text.

        Args:
            action: The player's action text
            triage_context: Additional context from triage (skill, ability, etc.)
            phase: Current game phase
            max_rules: Maximum rules to return

        Returns:
            Formatted rule text for narrator injection, or empty string.
        """
        rules = self._load_rules()
        if not rules:
            return ""

        # Score each rule section by keyword matches
        combined = f"{action} {triage_context} {phase}".lower()
        scores: dict[str, int] = {}

        for keyword, section_names in KEYWORD_RULES.items():
            if keyword in combined:
                for name in section_names:
                    scores[name] = scores.get(name, 0) + 1

        if not scores:
            return ""

        # Sort by score descending, take top N
        top_rules = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:max_rules]

        # Build formatted output
        parts = []
        for name, score in top_rules:
            content = rules.get(name, "")
            if not content:
                continue
            # Truncate long rules
            if len(content) > MAX_RULE_CHARS:
                content = content[:MAX_RULE_CHARS] + "..."
            parts.append(f"### {name}\n{content}")

        if not parts:
            return ""

        return "\n\n".join(parts)


# Singleton
_rule_injector: Optional[RuleInjector] = None


def get_rule_injector() -> RuleInjector:
    """Get or create the rule injector singleton."""
    global _rule_injector
    if _rule_injector is None:
        _rule_injector = RuleInjector()
    return _rule_injector
