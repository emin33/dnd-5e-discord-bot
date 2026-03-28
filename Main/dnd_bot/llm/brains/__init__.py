"""LLM brains for the two-brain pattern.

Architecture:
- NarratorBrain (high temp): Creative prose narration
- EffectsAdjudicator (temp 0): Deterministic effect extraction
- RulesBrain (temp 0): Mechanical resolution via tools
"""

from .base import Brain, BrainContext, BrainResult
from .narrator import NarratorBrain, MechanicalOutcome, get_narrator
from .adjudicator import EffectsAdjudicator, get_adjudicator
from .rules import RulesBrain, MechanicalResult, get_rules_brain

__all__ = [
    # Base
    "Brain",
    "BrainContext",
    "BrainResult",
    # Narrator (creative prose)
    "NarratorBrain",
    "MechanicalOutcome",
    "get_narrator",
    # Adjudicator (effect extraction)
    "EffectsAdjudicator",
    "get_adjudicator",
    # Rules (mechanical resolution)
    "RulesBrain",
    "MechanicalResult",
    "get_rules_brain",
]
