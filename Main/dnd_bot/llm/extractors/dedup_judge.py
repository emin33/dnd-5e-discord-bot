"""EntityDedupJudge — server-side rewrite of duplicate add_npc to ref_entity.

Solves the paraphrase-fragmentation bug where the narrator fires fresh
``add_npc`` calls for the same character after its prose paraphrases the
name (e.g. "Hooded Figure" → "The Cloaked Figure" → "The Shadow Figure"
all became 4 separate NPCState records over 22 turns).

Pattern: Graphiti-style. The narrator emits ``add_npc(name, description)``.
The orchestrator passes the proposed entity + a recency-windowed registry
slice to this judge. The judge (running on the configured brain client,
same path as triage and state extraction) decides:

- ``{"action": "rewrite", "target_id": "<existing_id>", "alias": "<new_name>"}``
  — high confidence the proposed entity is an existing one. The orchestrator
  rewrites the ADD_NPC effect to a REF_ENTITY pointing at ``target_id``,
  optionally accumulating ``alias`` on the existing record.

- ``{"action": "accept"}`` — genuinely new entity, or insufficient evidence
  to merge. The orchestrator processes ``add_npc`` as-is.

Default safe: prefer ``accept`` on borderline cases. False negatives
(missed dedup) are recoverable as the entity gets ref_entity'd in later
turns when the narrator picks a name that DOES match the registry.
False positives (wrongly merging two distinct characters) are not.

Per project policy, the judge runs on the configured brain
(``profile.brain.model``) — no new model parameter. It's another task
in the existing two-brain architecture.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Optional

import structlog

from ..client import get_llm_client
from ..json_extract import extract_json_object

logger = structlog.get_logger()


JUDGE_PROMPT = """You are an entity-resolution judge for a D&D narrator's tool calls.

The narrator just emitted an ``add_npc`` tool call (proposing a NEW NPC).
Your job: decide if this is genuinely new, or if the narrator is paraphrasing
an entity that already exists in the registry.

You will be given:
- The narrator's prose for this turn (what was actually written)
- The proposed new NPC: name + description from the add_npc call
- A slice of the current registry (recently-active entities only)

Decide ONE of:
- **rewrite**: HIGH CONFIDENCE this is an existing entity. The narrator just
  used a paraphrased name. Output the existing entity's id and the
  paraphrased name to record as an alias.
- **accept**: genuinely new, OR ambiguous, OR you can't tell with high
  confidence. Default to ``accept`` whenever uncertain.

How to decide:
- Same archetype + same recent appearance + matching distinguishing details
  ("the cloaked figure" right after a "hooded figure" was at the same scene
  with grey wool clothing) → rewrite.
- Adjective-shifted name on the same character ("Old Bram" when the
  registry has "Bram" who was last seen 1 turn ago) → rewrite.
- Two distinct characters that happen to share a label ("the merchant"
  could be different merchants in different scenes) → accept.
- New scene, no overlap with registry, no recent connection → accept.

Bias toward **accept** on any uncertainty. Wrongly merging two distinct
characters (false positive) is much worse than missing a dedup (false
negative — the entity gets correctly ref'd in later turns).

Output EXACTLY ONE JSON object on one line:
- ``{"action": "rewrite", "target_id": "<existing_id>", "alias": "<paraphrased name>"}``
- ``{"action": "accept"}``

Do not include any prose, reasoning, or markdown — just the JSON."""


@dataclass
class DedupDecision:
    """Result of a dedup-judge call."""
    action: str            # "rewrite" | "accept"
    target_id: Optional[str] = None
    alias: Optional[str] = None
    raw_response: str = ""

    @property
    def is_rewrite(self) -> bool:
        return self.action == "rewrite" and bool(self.target_id)


def _format_registry_slice(npcs: list, current_turn: int, recency_window: int = 15) -> str:
    """Format the recency-bounded registry slice for the judge prompt.

    ``recency_window`` is in turns — entities last seen within this many
    turns of ``current_turn`` are included. Entities outside the window
    are dropped from the slice so the judge isn't biased to merge against
    long-dormant characters.
    """
    if not npcs:
        return "(registry empty)"

    candidates = []
    for npc in npcs:
        if not getattr(npc, "alive", True):
            continue  # Dead NPCs aren't dedup candidates
        last_seen = getattr(npc, "last_seen_turn", 0) or 0
        # Always include entities marked as important regardless of recency,
        # since they're likely to be referenced again across long gaps.
        if last_seen and (current_turn - last_seen) > recency_window and not getattr(npc, "important", False):
            continue
        entry = {
            "id": npc.id,
            "name": npc.name,
            "description": (npc.description or "")[:200],
            "disposition": npc.disposition,
            "location": npc.location,
            "last_seen_turn": last_seen,
        }
        if npc.aliases:
            entry["aliases"] = list(npc.aliases)
        if npc.inventory:
            entry["inventory"] = list(npc.inventory)
        candidates.append(entry)

    if not candidates:
        return "(no candidates within recency window)"

    return json.dumps(candidates, indent=2)


class EntityDedupJudge:
    """Brain task that decides whether a proposed add_npc is a duplicate."""

    def __init__(self, client=None):
        # Same brain client as triage / state extractor — defined per profile.
        self.client = client or get_llm_client()

    async def judge_add_npc(
        self,
        proposed_name: str,
        proposed_description: str,
        narrator_prose: str,
        existing_npcs: list,
        current_turn: int = 0,
        recency_window: int = 15,
    ) -> DedupDecision:
        """Decide whether ``add_npc`` should be rewritten to ``ref_entity``.

        Args:
            proposed_name: name from the narrator's add_npc call
            proposed_description: description from the call
            narrator_prose: the prose this turn (judge needs this for context)
            existing_npcs: list of NPCState objects (orchestrator passes
                ``world_state.npcs.values()`` or a pre-filtered slice)
            current_turn: WorldState.turn — used for recency filtering
            recency_window: how many turns back to include in registry slice
        """
        if not existing_npcs:
            # Nothing to dedup against — fast path
            return DedupDecision(action="accept", raw_response="(empty-registry fast-path)")

        registry_block = _format_registry_slice(
            list(existing_npcs), current_turn, recency_window
        )

        user_prompt = f"""## Narrator's prose this turn
{narrator_prose[:1500]}

## Proposed new NPC (from add_npc call)
- name: {proposed_name}
- description: {proposed_description[:300]}

## Current registry (recency-windowed slice)
{registry_block}

Decide. Output one JSON object."""

        try:
            response = await self.client.chat(
                messages=[
                    {"role": "system", "content": JUDGE_PROMPT},
                    {"role": "user", "content": user_prompt},
                ],
                temperature=0.1,
                max_tokens=200,
                json_mode=True,
            )
        except Exception as e:
            logger.warning("dedup_judge_call_failed", error=str(e))
            return DedupDecision(action="accept", raw_response=f"error: {e}")

        raw = (response.content or "").strip()
        if not raw:
            return DedupDecision(action="accept", raw_response="(empty response)")

        # Extract the first JSON object even if model added stray text
        data, warnings = extract_json_object(raw)
        if data is None:
            logger.warning("dedup_judge_parse_failed", warnings=warnings, raw_preview=raw[:200])
            return DedupDecision(action="accept", raw_response=raw)

        action = (data.get("action") or "").lower().strip()
        if action == "rewrite":
            target_id = (data.get("target_id") or "").strip()
            alias = (data.get("alias") or "").strip() or None
            if not target_id:
                # Malformed rewrite (no target) — default safe
                logger.warning("dedup_judge_rewrite_missing_target", raw=raw[:200])
                return DedupDecision(action="accept", raw_response=raw)
            return DedupDecision(
                action="rewrite",
                target_id=target_id,
                alias=alias,
                raw_response=raw,
            )

        # Anything else — accept (safe default)
        return DedupDecision(action="accept", raw_response=raw)


# Singleton accessor — mirrors get_state_extractor / get_entity_extractor patterns
_JUDGE: Optional[EntityDedupJudge] = None


def get_dedup_judge() -> EntityDedupJudge:
    global _JUDGE
    if _JUDGE is None:
        _JUDGE = EntityDedupJudge()
    return _JUDGE
