"""Deterministic identity resolution for campaign entities.

Names and slugs are aliases, never authorities. This module intentionally
handles only high-precision matches; semantic ambiguity is left unresolved
for a later adjudication layer rather than guessed into canonical state.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from typing import Any, Optional, TypeVar


_T = TypeVar("_T")

# Leading descriptors and honorifics commonly drift in narrator/extractor
# output. Stripping only the leading run keeps "Old Bram" -> "bram" while
# avoiding unsafe last-name-only matching for ordinary full names.
_LEADING_TITLES = {
    "the",
    "old",
    "young",
    "elder",
    "captain",
    "commander",
    "warden",
    "guard",
    "sir",
    "lady",
    "lord",
    "master",
    "mistress",
    "doctor",
    "dr",
    "mister",
    "mr",
    "mrs",
    "ms",
}

_LOCATION_FILLER = {"the", "a", "an", "of", "at", "in", "on"}

# Labels made entirely from these words describe an archetype, role, or
# unnamed placeholder rather than a durable personal identity.  Keeping this
# list beside the canonical resolver lets the production reconciliation path
# and the long-form audit make the same conservative decision.
_GENERIC_NPC_TERMS = {
    "a",
    "acolyte",
    "an",
    "apothecary",
    "archer",
    "bartender",
    "beggar",
    "boy",
    "captain",
    "child",
    "clerk",
    "cloaked",
    "courier",
    "cultist",
    "customer",
    "dockworker",
    "distiller",
    "dwarf",
    "elder",
    "elf",
    "figure",
    "girl",
    "guard",
    "hooded",
    "innkeeper",
    "keeper",
    "laborer",
    "labourer",
    "man",
    "masked",
    "merchant",
    "messenger",
    "observer",
    "officer",
    "old",
    "older",
    "patron",
    "priest",
    "priestess",
    "proprietor",
    "ragpicker",
    "refugee",
    "scavenger",
    "scribe",
    "soldier",
    "stranger",
    "the",
    "traveler",
    "traveller",
    "unidentified",
    "unknown",
    "unseen",
    "vendor",
    "voice",
    "warden",
    "watch",
    "woman",
    "worker",
    "young",
    "younger",
}


def _normalized_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (value or "").casefold())


def identity_keys(value: str) -> frozenset[str]:
    """Return conservative exact keys for one display name or alias."""
    words = _normalized_words(value)
    if not words:
        return frozenset()
    keys = {" ".join(words)}
    index = 0
    while index < len(words) - 1 and words[index] in _LEADING_TITLES:
        index += 1
    if index:
        keys.add(" ".join(words[index:]))
    return frozenset(keys)


def is_generic_npc_label(value: str) -> bool:
    """Return whether *value* is only an unnamed NPC role/description.

    This deliberately abstains when any token looks identity-bearing.  For
    example, ``"the hooded figure"`` and ``"Ragpicker"`` are generic, while
    ``"Mira"`` and ``"Warden Elara"`` are not.
    """
    words = _normalized_words(value)
    return bool(words) and all(word in _GENERIC_NPC_TERMS for word in words)


def explicit_npc_naming_link(
    prose: str,
    generic_name: str,
    proper_name: str,
) -> bool:
    """Return whether prose explicitly names one visible generic NPC.

    This is intentionally grammatical rather than semantic. It supports the
    high-confidence identity transition used by both narrator tools and the
    independent StateDelta path (``the woman`` -> ``Orra``), while refusing
    to infer a link merely because a role and a proper name share a turn.
    """
    if (
        not is_generic_npc_label(generic_name)
        or is_generic_npc_label(proper_name)
    ):
        return False
    normalized_prose = " ".join(_normalized_words(prose))
    normalized_generic = " ".join(_normalized_words(generic_name))
    normalized_name = " ".join(_normalized_words(proper_name))
    if not normalized_prose or not normalized_generic or not normalized_name:
        return False
    if f" {normalized_generic} " not in f" {normalized_prose} ":
        return False

    name_pattern = re.escape(normalized_name).replace(r"\ ", r"\s+")
    direct_cue = re.compile(
        rf"\b(?:i\s+m(?:\s+called)?|i\s+am(?:\s+called)?|"
        rf"my\s+name\s+is|name\s+s|call\s+me|this\s+is)\s+"
        rf"{name_pattern}\b",
        re.IGNORECASE,
    )
    if direct_cue.search(normalized_prose):
        return True

    # Narrative confirmation after a standalone spoken name:
    # ``Mira. She said her name.`` / ``Orra — that is my name.``
    confirmation_cue = re.compile(
        rf"\b{name_pattern}\b(?:\s+\w+){{0,5}}\s+"
        rf"(?:she\s+said\s+her\s+name|he\s+said\s+his\s+name|"
        rf"they\s+said\s+their\s+name|that\s+is\s+my\s+name)\b",
        re.IGNORECASE,
    )
    return bool(confirmation_cue.search(normalized_prose))


def name_is_fragment_of(candidate: str, existing: str) -> bool:
    """Return whether *candidate* adds no identity words beyond *existing*.

    'Choir' is a fragment of 'a Choir acolyte'; 'Orina' is not a fragment
    of 'the woman'. A narrator alias that merely excerpts the current
    descriptive label is a reference to the entity, not a newly revealed
    name — renaming onto it hijacks whatever the excerpted word denotes
    (a faction, a role) and orphans the rest of the label.
    """
    candidate_words = set(_normalized_words(candidate))
    existing_words = set(_normalized_words(existing))
    return bool(candidate_words) and candidate_words <= existing_words


def entity_identity_keys(entity: Any) -> frozenset[str]:
    """Return exact identity keys from an object's name and aliases."""
    keys: set[str] = set()
    for value in (
        getattr(entity, "name", ""),
        *(getattr(entity, "aliases", None) or []),
    ):
        keys.update(identity_keys(str(value or "")))
    return frozenset(keys)


def resolve_unique_identity(query: str, candidates: Iterable[_T]) -> Optional[_T]:
    """Resolve only a unique ID/name/alias match; otherwise abstain."""
    candidate_list = list(candidates)
    exact_id = [
        candidate
        for candidate in candidate_list
        if query and str(
            getattr(candidate, "id", "")
            or getattr(candidate, "node_id", "")
        ) == query
    ]
    if len(exact_id) == 1:
        return exact_id[0]

    query_keys = identity_keys(query)
    if not query_keys:
        return None
    matches = [
        candidate
        for candidate in candidate_list
        if query_keys.intersection(entity_identity_keys(candidate))
    ]
    return matches[0] if len(matches) == 1 else None


def location_identity_words(value: str) -> frozenset[str]:
    """Return stable lexical identity words for a free-form place label."""
    return frozenset(
        word for word in _normalized_words(value) if word not in _LOCATION_FILLER
    )


def locations_equivalent(left: str, right: str) -> bool:
    """Collapse a base place and one conservative qualified scene label.

    ``Ash Gate`` equals ``the Ash Gate clearing``; ``tavern`` does not equal
    ``back room of the tavern``. The two-word minimum keeps broad one-word
    locations from swallowing distinct sub-scenes.
    """
    left_words = location_identity_words(left)
    right_words = location_identity_words(right)
    if not left_words or not right_words:
        return False
    if left_words == right_words:
        return True
    smaller, larger = sorted((left_words, right_words), key=len)
    return (
        len(smaller) >= 2
        and smaller.issubset(larger)
        and len(larger) - len(smaller) <= 1
    )
