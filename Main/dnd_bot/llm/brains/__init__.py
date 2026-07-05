"""LLM brains for the two-brain pattern.

Architecture:
- NarratorBrain (high temp): Creative prose narration
- EffectsAdjudicator (temp 0): Deterministic effect extraction
"""

from .base import Brain, BrainContext, BrainResult
from .narrator import NarratorBrain, MechanicalOutcome, get_narrator
from .adjudicator import EffectsAdjudicator, get_adjudicator

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
]
