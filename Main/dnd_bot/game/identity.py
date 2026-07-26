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
    "barkeep",
    "bartender",
    "beggar",
    "boy",
    "brother",
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
    "serving",
    "shopkeeper",
    "single",
    "sister",
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

# Deliberately ABSENT, though they are role nouns: "smith", "cook", "baker",
# "fletcher", "mason". They are also ordinary surnames, and this list is used
# to REFUSE an anchor — a false entry here silently makes a real character
# unreachable by name, which no test would notice.


def _is_generic_word(word: str) -> bool:
    """One placeholder token, singular or plural.

    The plural arm matters: the extractor writes "the guards" as readily as
    "the guard", and matching only the singular let every pluralised
    placeholder through as though it were a name.
    """
    if word.isdigit() or word in _GENERIC_NPC_TERMS:
        return True
    for suffix in ("es", "s"):
        if word.endswith(suffix) and word[: -len(suffix)] in _GENERIC_NPC_TERMS:
            return True
    return False


def _normalized_words(value: str) -> list[str]:
    return re.findall(r"[a-z0-9]+", (value or "").casefold())


def normalized_identity_text(value: str) -> str:
    """Lowercase alphanumeric tokens, single-spaced. The matching surface."""
    return " ".join(_normalized_words(value))


def padded_identity_text(value: str) -> str:
    """:func:`normalized_identity_text` fenced by sentinel spaces.

    Lets ``f" {anchor} " in padded`` stand in for a word-boundary match.
    Normalize once, test many anchors against the result.
    """
    return f" {normalized_identity_text(value)} "


def entity_named_in_text(text: str, names: Iterable[str]) -> list[str]:
    """An entity's identity-bearing names, IF *text* names it outright.

    One rule for "did the player raise this subject, and under which names
    can prose about it be recognized?" — so callers that resolve entities
    differently still agree on the answer.

    Two conditions, both load-bearing. The name must land on token
    boundaries: substring matching turns "brambles" into Bram and "I pry
    the grate" into Ron. And the name that lands must be identity-bearing:
    a placeholder promoted to an alias ("the innkeeper") recurs in most
    tavern turns, so anchoring on it would make one off-screen NPC salient
    forever. Aliases still come back on a distinctive hit — the text says
    "the black arch", the ledger says "Ash Gate".
    """
    return names_addressed_in_text(text, [names])[0]


def _distinctive(names: Iterable[str]) -> list[str]:
    """An entity's identity-bearing names.

    A generic label neither addresses the entity nor anchors prose about it:
    naming "the dwarf" must not raise Grimjaw, because the next sentence
    about any dwarf would then read as being about him. Which labels count
    as generic is `is_generic_npc_label`'s judgement, and it is deliberately
    a narrow one — see the note on `_GENERIC_NPC_TERMS` about what is left
    out on purpose.
    """
    return [
        name for name in names
        if name and normalized_identity_text(name)
        and not is_generic_npc_label(name)
    ]


def _token_spans(haystack: list[str], needle: list[str]) -> list[tuple[int, int]]:
    """Every [start, end) run where *needle*'s tokens appear in *haystack*."""
    width = len(needle)
    if not width or width > len(haystack):
        return []
    return [
        (start, start + width)
        for start in range(len(haystack) - width + 1)
        if haystack[start:start + width] == needle
    ]


def names_addressed_in_text(
    text: str, entities: Iterable[Iterable[str]]
) -> list[list[str]]:
    """Which of several entities the text names, longest name winning its span.

    Resolving entities TOGETHER is what makes this safe. Token boundaries are
    not entity boundaries: a one-word name is always a token of a longer name
    that contains it, so asking each entity in isolation let "I ask Mara Venn
    what she saw" anchor an unrelated NPC called Mara — and put HER canon in
    the narrator's prompt. This codebase makes that common rather than exotic,
    because naming-promotion routinely leaves bare first names in aliases.

    So the longest name claims its tokens first, and a shorter one survives
    only where it occurs OUTSIDE every longer claim. "I ask Mara Venn about
    Mara" still names both; "I ask Mara Venn what she saw" names only her.

    Returns one list per entity, in the order given: all of that entity's
    identity-bearing names when it was named, empty when it was not.
    """
    per_entity = [_distinctive(names) for names in entities]
    tokens = _normalized_words(text)
    if not tokens:
        return [[] for _ in per_entity]

    candidates: list[tuple[int, int, str, list[tuple[int, int]]]] = []
    for index, names in enumerate(per_entity):
        for name in names:
            needle = _normalized_words(name)
            spans = _token_spans(tokens, needle)
            if spans:
                candidates.append((len(needle), index, name, spans))
    # Longest first; ties broken on the name so the result never depends on
    # dict or set ordering.
    candidates.sort(key=lambda item: (-item[0], item[2]))

    claimed = [False] * len(tokens)
    named: set[int] = set()
    for _width, index, _name, spans in candidates:
        if not any(not any(claimed[start:end]) for start, end in spans):
            continue
        named.add(index)
        # EVERY occurrence, not just the one that carried the match. Claiming
        # only the first left the second "Mara Venn" in "Mara Venn, tell Mara
        # Venn to wait" unclaimed, so a shorter name sitting inside it — an
        # unrelated NPC called Mara — took that span and brought her canon
        # along. A name owns all the ground it covers.
        for start, end in spans:
            for position in range(start, end):
                claimed[position] = True
    return [
        names if index in named else []
        for index, names in enumerate(per_entity)
    ]


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


# Prepositions that attach a wardrobe/position descriptor to a generic role
# noun ("man in apron", "woman with the lantern", "figure by the door").
_DESCRIPTOR_PREPOSITIONS = {"in", "with", "by", "at", "near", "behind"}


def is_generic_npc_label(value: str) -> bool:
    """Return whether *value* is only an unnamed NPC role/description.

    This deliberately abstains when any token looks identity-bearing.  For
    example, ``"the hooded figure"`` and ``"Ragpicker"`` are generic, while
    ``"Mira"`` and ``"Warden Elara"`` are not.
    """
    words = _normalized_words(value)
    if not words:
        return False
    # Bare numerals are spawn-numbering artifacts ("acolyte 1", "guard 2"),
    # not identity-bearing tokens.
    if all(_is_generic_word(word) for word in words):
        return True
    # A generic role noun plus a prepositional descriptor ("man in apron")
    # is still an unnamed placeholder: the descriptor names clothing or
    # position, not a person. Any capitalized token beyond the label's
    # leading character abstains — "man in Orin's shop" carries an identity.
    if any(character.isupper() for character in value.strip()[1:]):
        return False
    for index, word in enumerate(words):
        if index and word in _DESCRIPTOR_PREPOSITIONS:
            head_is_generic = all(
                head in _GENERIC_NPC_TERMS or head.isdigit()
                for head in words[:index]
            )
            return head_is_generic and bool(words[index + 1:])
    return False


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
