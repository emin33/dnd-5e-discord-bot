"""Fact supersession — retire established facts a new fact makes untrue.

``established_facts`` was append-only: "The wax seal is soft and warm"
and "The wax smudge is dry and brittle" coexisted after the state
changed, and the scene-relevant projection then fed the narrator both
sides of a contradiction (narrative-grader severe contradiction, run
20260723_003823 turn 22). Facts are free prose, so retirement follows
the extractor dedup-judge pattern: a cheap deterministic candidate gate
(shared anchor words) followed by a brain-judge decision, defaulting to
KEEP BOTH on any uncertainty. Retired facts move to an archive with
provenance — the campaign history keeps them; prompts stop seeing them.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger()

# Words too common in campaign facts to indicate a shared subject.
_ANCHOR_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "been", "but", "by", "for",
    "from", "had", "has", "have", "her", "his", "in", "into", "is", "it",
    "its", "npc", "item", "location", "no", "not", "now", "of", "on",
    "one", "or", "that", "the", "their", "them", "they", "this", "to",
    "was", "were", "which", "while", "who", "will", "with",
}

MAX_CANDIDATES = 8

JUDGE_PROMPT = """You maintain a D&D campaign's fact ledger. A new fact \
was just established. Decide whether it SUPERSEDES any of the numbered \
older facts — meaning the old fact is NO LONGER TRUE given the new one \
(a location changed, a physical state changed, possession transferred, \
a status ended).

Do NOT supersede when the facts coexist: elaboration, additional detail, \
history ("X was born in Y" survives "X now lives in Z"), or facts about \
different subjects that merely share a word.

New fact: {new_fact}

Older facts:
{candidate_block}

Output EXACTLY one JSON object, no prose:
{{"supersedes": [<numbers of older facts now untrue>]}}
Empty list when unsure."""


def fact_anchor_words(fact: str) -> frozenset[str]:
    """Content words that can indicate a shared subject between facts."""
    words = re.findall(r"[a-z0-9']+", (fact or "").casefold())
    return frozenset(
        word for word in words
        if len(word) >= 3 and word not in _ANCHOR_STOPWORDS
    )


def candidate_indices(new_fact: str, established: list[str]) -> list[int]:
    """Indices of established facts sharing an anchor word, recent first."""
    anchors = fact_anchor_words(new_fact)
    if not anchors:
        return []
    hits = [
        index
        for index in range(len(established) - 1, -1, -1)
        if anchors & fact_anchor_words(established[index])
    ]
    return hits[:MAX_CANDIDATES]


@dataclass
class SupersessionResult:
    retired: list[str] = field(default_factory=list)


class FactSupersessionJudge:
    """Brain task deciding whether a new fact retires older ones.

    Same brain client as triage/extraction/dedup — no new model knob.
    Default safe: any error, empty response, or out-of-range index keeps
    every fact. Wrongly retiring a live fact loses state; keeping a stale
    one is recoverable noise.
    """

    def __init__(self, client=None):
        if client is None:
            from ..llm.client import get_llm_client
            client = get_llm_client()
        self.client = client

    async def judge(
        self,
        new_fact: str,
        candidates: list[str],
    ) -> list[str]:
        """Return the subset of *candidates* the new fact supersedes."""
        if not new_fact or not candidates:
            return []
        candidate_block = "\n".join(
            f"{index + 1}. {fact}" for index, fact in enumerate(candidates)
        )
        try:
            response = await self.client.chat(
                messages=[{
                    "role": "user",
                    "content": JUDGE_PROMPT.format(
                        new_fact=new_fact, candidate_block=candidate_block
                    ),
                }],
                temperature=0.0,
                max_tokens=120,
                json_mode=True,
                think=False,
            )
        except Exception as e:
            logger.warning("fact_supersession_judge_failed", error=str(e), exc_info=True)
            return []
        raw = (getattr(response, "content", "") or "").strip()
        if not raw:
            return []
        try:
            from ..llm.json_extract import extract_json_object
            data, _warnings = extract_json_object(raw)
        except Exception:
            data = None
        if not isinstance(data, dict):
            return []
        retired = []
        for value in data.get("supersedes") or []:
            if isinstance(value, int) and 1 <= value <= len(candidates):
                retired.append(candidates[value - 1])
        return retired
