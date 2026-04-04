"""NLI Cross-Encoder Validation Layer.

Post-generation contradiction detection using DeBERTa-v3-xsmall (22M params).
Extracts key claims from narrator output, pairs each with world state facts,
and batch-predicts contradiction scores. ~150-300ms on CPU.

Integration: runs after narrator generates prose, before sending to Discord.
On contradiction: logs the issue and optionally flags for the orchestrator.
"""

import os
import re
import time
from typing import Optional

import structlog

logger = structlog.get_logger()

# Labels from the NLI model
CONTRADICTION = 0
ENTAILMENT = 1
NEUTRAL = 2

# Threshold for flagging a contradiction (logit score)
CONTRADICTION_THRESHOLD = 2.0

# Ambiguous zone — scores between these trigger the LLM tiebreaker
AMBIGUOUS_LOW = 0.5
AMBIGUOUS_HIGH = 2.0


class NLIContradiction:
    """A detected contradiction between narrator output and world state."""
    def __init__(self, claim: str, fact: str, score: float):
        self.claim = claim
        self.fact = fact
        self.score = score

    def __repr__(self):
        return f"Contradiction(score={self.score:.2f}, claim='{self.claim[:60]}', fact='{self.fact[:60]}')"


class NLIValidator:
    """Validates narrator output against world state facts using NLI.

    Uses cross-encoder/nli-deberta-v3-xsmall for fast CPU inference.
    The model is loaded lazily on first use to avoid slowing bot startup.
    """

    def __init__(self, model_name: str = "cross-encoder/nli-deberta-v3-xsmall"):
        self._model_name = model_name
        self._model = None
        self._load_failed = False

    def _ensure_model(self) -> bool:
        """Lazy-load the model on first use. Returns True if ready."""
        if self._model is not None:
            return True
        if self._load_failed:
            return False

        try:
            # Force CPU — RTX 5090 (Blackwell/sm_120) needs newer PyTorch for CUDA
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")
            from sentence_transformers import CrossEncoder
            self._model = CrossEncoder(self._model_name, device="cpu")
            logger.info("nli_model_loaded", model=self._model_name)
            return True
        except Exception as e:
            self._load_failed = True
            logger.warning("nli_model_load_failed", error=str(e))
            return False

    def extract_claims(self, narrative: str, max_claims: int = 5) -> list[str]:
        """Extract key factual claims from narrator prose.

        Focuses on statements about: who is present, where things are,
        time/weather, character states, and actions taken.
        """
        # Split into sentences
        sentences = re.split(r'(?<=[.!?])\s+', narrative.strip())

        claims = []
        for sentence in sentences:
            sentence = sentence.strip()
            if len(sentence) < 15 or len(sentence) > 200:
                continue

            # Skip pure dialogue (quotes)
            if sentence.startswith(("'", '"', "\u2018", "\u201c")):
                continue

            # Prioritize sentences with factual indicators
            sentence_lower = sentence.lower()
            is_factual = any(indicator in sentence_lower for indicator in [
                # Static state
                " is ", " are ", " was ", " were ", " stands ", " sits ",
                " lies ", " holds ", " wears ", " carries ", " contains ",
                # Time/environment
                "the sun ", "the moon ", "dawn ", "dusk ", "night ",
                "morning ", "afternoon ", "evening ",
                # Existence/presence
                " dead ", " alive ", " gone ", " left ", " arrived ",
                " empty ", " locked ", " open ", " closed ",
                " appears ", " disappears ", " vanishes ",
                # State-changing actions (items, currency, inventory)
                " gives ", " gave ", " hands ", " handed ",
                " takes ", " took ", " picks up ", " picked up ",
                " drops ", " dropped ", " slides ", " slid ",
                " returns ", " returned ", " offers ", " offered ",
                " pays ", " paid ", " pockets ", " pocketed ",
                " draws ", " drew ", " sheathes ", " sheathed ",
                " drinks ", " drank ", " eats ", " ate ",
                " opens ", " opened ", " closes ", " closed ",
                " lights ", " lit ", " extinguishes ",
                # Currency/commerce
                " coin", " gold ", " silver ", " copper ",
                " platinum ", " electrum ",
            ])

            if is_factual:
                claims.append(sentence)

            if len(claims) >= max_claims:
                break

        # If we didn't find enough factual sentences, take the first few
        if len(claims) < 2:
            for sentence in sentences[:max_claims]:
                sentence = sentence.strip()
                if len(sentence) >= 15 and sentence not in claims:
                    claims.append(sentence)
                if len(claims) >= max_claims:
                    break

        return claims

    def build_categorized_facts(self, world_state_yaml: str) -> dict[str, list[str]]:
        """Extract checkable facts from world state YAML, grouped by category.

        Returns facts organized by type so claims can be paired only against
        relevant categories instead of brute-force all-pairs.

        Categories:
        - environment: time, location, weather
        - npcs: NPC presence, disposition, descriptions
        - items: scene items, transfers, currency
        - events: recent events and actions
        """
        import yaml

        cats: dict[str, list[str]] = {
            "environment": [],
            "npcs": [],
            "items": [],
            "events": [],
        }

        try:
            data = yaml.safe_load(world_state_yaml)
            if not data or not isinstance(data, dict):
                return cats
        except Exception:
            return cats

        # Environment
        if "time_of_day" in data:
            cats["environment"].append(f"It is {data['time_of_day']}.")
        if "location" in data:
            cats["environment"].append(f"The party is at {data['location']}.")

        # NPCs
        if "npcs_here" in data:
            for npc in data["npcs_here"]:
                name = npc.get("name", "someone")
                disp = npc.get("disposition", "neutral")
                cats["npcs"].append(f"{name} is present and is {disp}.")
                if npc.get("desc"):
                    cats["npcs"].append(f"{name} is {npc['desc'][:80]}.")
        if "key_npcs_elsewhere" in data:
            for line in data["key_npcs_elsewhere"]:
                cats["npcs"].append(f"{line.split(':')[0].strip()} is NOT at the current location.")

        # Items
        if "scene_items" in data:
            for item_line in data["scene_items"]:
                cats["items"].append(f"In the scene: {item_line}.")
        if "recent_transfers" in data:
            for transfer in data["recent_transfers"]:
                cats["items"].append(transfer)

        # Events (recent only — these are the most likely to contradict)
        if "recent_events" in data:
            for event in data["recent_events"][-3:]:
                cats["events"].append(event)
        if "facts" in data:
            # Only check last 5 established facts (older ones are less likely to contradict)
            for fact in data["facts"][-5:]:
                cats["events"].append(fact)

        return cats

    def _classify_claim(self, claim: str) -> list[str]:
        """Determine which fact categories a claim should be checked against.

        Returns list of category names. A claim may match multiple categories.
        """
        claim_lower = claim.lower()
        categories = []

        # NPC indicators
        npc_words = [" he ", " she ", " they ", " him ", " her ", " them ",
                     " man ", " woman ", " figure ", " traveler ", " stranger ",
                     " guard ", " merchant ", " keeper ", " npc ", " person ",
                     " says ", " said ", " speaks ", " spoke ", " whisper",
                     " greet", " nod", " gesture", " expression"]
        if any(w in claim_lower for w in npc_words):
            categories.append("npcs")

        # Item/transfer indicators
        item_words = [" gives ", " gave ", " takes ", " took ", " picks ",
                      " drops ", " slides ", " hands ", " holds ", " carries ",
                      " coin", " gold ", " silver ", " sword ", " dagger ",
                      " potion ", " item ", " weapon ", " shield ", " bow ",
                      " arrow"]
        if any(w in claim_lower for w in item_words):
            categories.append("items")

        # Environment indicators
        env_words = ["the sun ", "the moon ", "dawn", "dusk", "night",
                     "morning", "afternoon", "evening", "midnight",
                     " dark ", " light ", " bright ", " dim ",
                     " rain ", " snow ", " wind ", " storm "]
        if any(w in claim_lower for w in env_words):
            categories.append("environment")

        # Events — always check recent events (they're the most specific)
        categories.append("events")

        return categories if categories else ["events"]

    def validate(
        self,
        narrative: str,
        world_state_yaml: str,
        max_claims: int = 5,
    ) -> list[NLIContradiction]:
        """Validate narrator output against world state.

        Uses structured pairing: claims are classified by category (NPC,
        item, environment, event) and only checked against facts from
        matching categories. This prevents quadratic scaling and eliminates
        garbage pairings like "his footing slips" vs "signpost description."

        Returns list of detected contradictions (empty = all clear).
        Typical latency: 100-400ms on CPU for 5-15 targeted pairs.
        """
        if not self._ensure_model():
            return []  # Model unavailable, skip validation

        claims = self.extract_claims(narrative, max_claims=max_claims)
        categorized_facts = self.build_categorized_facts(world_state_yaml)

        # Count total facts across categories
        total_facts = sum(len(v) for v in categorized_facts.values())
        if not claims or total_facts == 0:
            return []

        # Build targeted pairs: each claim only paired with relevant categories
        pairs = []
        for claim in claims:
            relevant_categories = self._classify_claim(claim)
            for cat in relevant_categories:
                for fact in categorized_facts.get(cat, []):
                    pairs.append((fact, claim))

        if not pairs:
            return []

        # Cap pairs to prevent runaway latency (max 20 pairs)
        if len(pairs) > 20:
            pairs = pairs[:20]

        t0 = time.monotonic()

        try:
            scores = self._model.predict(pairs)
        except Exception as e:
            logger.warning("nli_prediction_failed", error=str(e))
            return []

        elapsed_ms = (time.monotonic() - t0) * 1000

        # Find clear contradictions and ambiguous cases
        contradictions = []
        ambiguous_pairs = []
        for (fact, claim), score in zip(pairs, scores):
            contradiction_score = float(score[CONTRADICTION])
            if contradiction_score > CONTRADICTION_THRESHOLD:
                contradictions.append(NLIContradiction(
                    claim=claim,
                    fact=fact,
                    score=contradiction_score,
                ))
            elif contradiction_score > AMBIGUOUS_LOW:
                ambiguous_pairs.append((fact, claim, contradiction_score))

        logger.info(
            "nli_validation_complete",
            pairs_checked=len(pairs),
            claims=len(claims),
            facts=total_facts,
            contradictions_found=len(contradictions),
            ambiguous_count=len(ambiguous_pairs),
            elapsed_ms=round(elapsed_ms),
        )

        # Store ambiguous pairs for async tiebreaker (called separately)
        self._pending_ambiguous = ambiguous_pairs

        return contradictions

    async def resolve_ambiguous(self) -> list[NLIContradiction]:
        """Run LLM tiebreaker on ambiguous NLI results.

        Uses the cheap brain model for a single-shot PASS/FAIL judgment
        on pairs where the NLI score was ambiguous (0.5-2.0).
        Only fires ~10-20% of turns. ~300-800ms per call.

        Call this after validate() if _pending_ambiguous is non-empty.
        """
        if not hasattr(self, '_pending_ambiguous') or not self._pending_ambiguous:
            return []

        ambiguous = self._pending_ambiguous
        self._pending_ambiguous = []

        # Limit to top 3 most suspicious pairs
        ambiguous.sort(key=lambda x: x[2], reverse=True)
        ambiguous = ambiguous[:3]

        try:
            from ..client import get_llm_client
            client = get_llm_client()

            # Build a single verification prompt for all ambiguous pairs
            pair_text = "\n".join(
                f"Fact: {fact}\nClaim: {claim}"
                for fact, claim, _ in ambiguous
            )

            response = await client.chat(
                messages=[
                    {
                        "role": "system",
                        "content": (
                            "You are a fact-checker for a D&D game. For each fact-claim pair, "
                            "determine if the claim CONTRADICTS the established fact. "
                            "Respond with one line per pair: PASS (no contradiction) or "
                            "FAIL: <brief reason>."
                        ),
                    },
                    {"role": "user", "content": pair_text},
                ],
                temperature=0,
                max_tokens=200,
                think=False,
            )

            content = response.content.strip() if response.content else ""
            lines = content.split("\n")

            confirmed = []
            for i, (fact, claim, score) in enumerate(ambiguous):
                if i < len(lines) and "FAIL" in lines[i].upper():
                    reason = lines[i].split(":", 1)[1].strip() if ":" in lines[i] else "LLM tiebreaker"
                    confirmed.append(NLIContradiction(
                        claim=f"{claim} [{reason}]",
                        fact=fact,
                        score=score,
                    ))

            if confirmed:
                logger.info(
                    "nli_tiebreaker_resolved",
                    ambiguous_checked=len(ambiguous),
                    contradictions_confirmed=len(confirmed),
                )

            return confirmed

        except Exception as e:
            logger.warning("nli_tiebreaker_failed", error=str(e))
            return []


# Singleton — lazy loaded
_nli_validator: Optional[NLIValidator] = None


def get_nli_validator() -> NLIValidator:
    """Get or create the NLI validator singleton."""
    global _nli_validator
    if _nli_validator is None:
        _nli_validator = NLIValidator()
    return _nli_validator
