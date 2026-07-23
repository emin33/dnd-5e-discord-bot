"""Extractor-coordinated targeted tool recovery signals.

StateDelta extraction runs after narration and its delta is applied by the
single-writer store, so world state converges even when the narrator omits
a tool call. What the omission loses is narrator-authored grounding: the
canonical short location name, and the explicit identity binding that keeps
the knowledge graph from splitting one person across ids. The generic
mutation followup in the narration layer cannot close this gap because it
never learns *what* was missed — reprompted blindly, weaker narrators return
more ``ref_entity`` calls and stop.

This module compares the applied delta against the turn's proposed effects
and produces targeted recovery signals, each naming the exact missing call.
The narration layer then issues one tool-only followup restricted to those
families.

Detection is deliberately conservative and mirrors the long-form audit's
tool-omission observer: a signal fires only when the extractor's claim is
literally grounded in the narration prose and not already covered by a
proposed tool. Ambiguity abstains — a wrong forced call is worse than a
missing one, because the delta already kept the state correct.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterable

from ..game.identity import (
    identity_keys,
    is_generic_npc_label,
    locations_equivalent,
)
from .effects import EffectType, ProposedEffect

if TYPE_CHECKING:
    from ..game.world_state import NPCUpdate, StateDelta, WorldState

# One followup request stays small and reviewable; anything past this cap is
# a sign the extractor went wide, not that the narrator went silent.
MAX_SIGNALS_PER_TURN = 4

_DISPOSITION_SHIFT_RE = (
    r"\b(?:becomes?|turns?|now|swears?|joins?|betrays?)\b"
    r".{{0,45}}\b(?:{disposition}|ally|allied|enemy|friend|hostile)\b"
)


@dataclass(frozen=True)
class StateFollowupSignal:
    """One missing tool call, with the exact instruction to request it."""

    kind: str  # "location" | "new_npc" | "npc_update" | "unnamed_identity"
    tool_names: tuple[str, ...]
    instruction: str


# Grammatical cues that very strongly imply a proper-named person is on
# stage ("Elara Venn's eyes narrow"). Mirrors the long-form audit observer.
_NPC_POSSESSIVE_CUE_RE = re.compile(
    r"\b([A-Z][A-Za-z'-]*(?:\s+[A-Z][A-Za-z'-]*){0,2})['’]s\s+"
    r"(?:eyes?|gaze|voice|face|hands?|brow|jaw|shoulders?|expression|"
    r"smile|fingers?|head|lips?|breath|posture)\b"
)
_LEADING_NAME_NOISE = {"And", "As", "But", "Then", "When", "While", "Yet"}


def strong_npc_name_cues(text: str) -> set[str]:
    """Extract only cues that very strongly imply a proper-named NPC."""
    names: set[str] = set()
    for match in _NPC_POSSESSIVE_CUE_RE.finditer(text or ""):
        tokens = match.group(1).split()
        while tokens and tokens[0] in _LEADING_NAME_NOISE:
            tokens.pop(0)
        candidate = " ".join(tokens)
        if _looks_like_proper_npc_name(candidate):
            names.add(candidate)
    return names


def _normalized_label(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", str(value or "").lower()).strip()


def _labels_overlap(left: object, right: object) -> bool:
    left_tokens = set(_normalized_label(left).split())
    right_tokens = set(_normalized_label(right).split())
    return bool(left_tokens and right_tokens) and (
        left_tokens == right_tokens
        or left_tokens.issubset(right_tokens)
        or right_tokens.issubset(left_tokens)
    )


def _looks_like_proper_npc_name(value: object) -> bool:
    raw = str(value or "").strip()
    normalized = _normalized_label(raw)
    if not normalized or re.match(r"^(?:a|an|the)\s+", normalized):
        return False
    if is_generic_npc_label(raw):
        return False
    return any(character.isupper() for character in raw)


def _effect_type(effect: ProposedEffect) -> EffectType:
    return effect.effect_type


def _resolve_roster_id(
    world_state: "WorldState",
    npc_id: str,
    name: str,
) -> str:
    """Return the applied roster id for an extractor NPC, or empty string.

    The delta may have been merged or reanchored onto a canonical id before
    application, so the extractor's own id is a hint, not an authority. A
    signal without a resolvable roster id abstains — ``ref_entity`` requires
    a real id and a guessed one would be rejected anyway.
    """
    if npc_id and npc_id in world_state.npcs:
        return npc_id
    name_keys = identity_keys(name)
    if not name_keys:
        return ""
    matches = [
        candidate_id
        for candidate_id, npc in world_state.npcs.items()
        if name_keys.intersection(identity_keys(npc.name))
        or any(
            name_keys.intersection(identity_keys(alias))
            for alias in npc.aliases
        )
    ]
    return matches[0] if len(matches) == 1 else ""


def _mutation_grade(update: "NPCUpdate", narrative: str) -> str:
    """Return a short description of a durable mutation, or empty string."""
    if update.alive is False:
        return "died or was killed"
    changes = []
    if update.add_inventory:
        changes.append(f"gained {', '.join(update.add_inventory)}")
    if update.remove_inventory:
        changes.append(f"lost {', '.join(update.remove_inventory)}")
    disposition = str(update.disposition or "").strip().lower()
    if disposition and re.search(
        _DISPOSITION_SHIFT_RE.format(disposition=re.escape(disposition)),
        narrative,
        re.IGNORECASE,
    ):
        changes.append(f"disposition became {disposition}")
    return "; ".join(changes)


def uncovered_state_signals(
    delta: "StateDelta",
    *,
    before_location: str,
    narrative: str,
    proposed_effects: Iterable[ProposedEffect],
    world_state: "WorldState",
    player_name: str = "",
    known_entity_labels: Iterable[str] = (),
) -> list[StateFollowupSignal]:
    """Detect applied delta mutations with no matching narrator tool.

    ``known_entity_labels`` extends the roster with durable labels the
    caller already tracks elsewhere (knowledge-graph NPC names/aliases), so
    a prose-cue name is only flagged when NO store knows the person.
    """
    effects = [e for e in proposed_effects if isinstance(e, ProposedEffect)]
    normalized_narration = f" {_normalized_label(narrative)} "
    signals: list[StateFollowupSignal] = []

    location = str(delta.location_change or "").strip()
    if location:
        normalized_location = _normalized_label(location)
        proposed_location = any(
            _effect_type(e) == EffectType.CHANGE_LOCATION for e in effects
        )
        if (
            normalized_location
            # Mirror the change_location validator's short-name contract; a
            # label it would reject is not worth a recovery round-trip.
            and len(location.split()) <= 5
            and "," not in location
            and f" {normalized_location} " in normalized_narration
            and normalized_location != _normalized_label(before_location)
            # A base place and its qualified sub-scene ("Tallow Rows" vs
            # "Tallow Rows alley") share one location identity; no call owed.
            and not locations_equivalent(location, before_location)
            and not proposed_location
        ):
            signals.append(StateFollowupSignal(
                kind="location",
                tool_names=("change_location",),
                instruction=(
                    f'The party moved to "{location}"'
                    + (f' (from "{before_location}")' if before_location else "")
                    + f'. Call change_location(location_name="{location}").'
                ),
            ))

    proposed_npc_names = {
        _normalized_label(e.npc_name)
        for e in effects
        if _effect_type(e) == EffectType.ADD_NPC and e.npc_name
    }
    proposed_refs = [
        e for e in effects if _effect_type(e) == EffectType.REF_ENTITY
    ]

    for npc in delta.new_npcs:
        name = str(npc.name or "").strip()
        if not _looks_like_proper_npc_name(name):
            continue
        if player_name and _labels_overlap(name, player_name):
            continue
        normalized_name = _normalized_label(name)
        if f" {normalized_name} " not in normalized_narration:
            # The extractor resolved a role from prior context; the exact
            # grounding contract for narrator tools forbids that call.
            continue
        if normalized_name in proposed_npc_names or any(
            _labels_overlap(name, ref.ref_alias_used)
            or _labels_overlap(name, ref.ref_entity_id)
            or (npc.id and npc.id == (ref.ref_entity_id or ""))
            for ref in proposed_refs
        ):
            continue
        roster_id = _resolve_roster_id(world_state, npc.id, name)
        if not roster_id:
            continue
        signals.append(StateFollowupSignal(
            kind="new_npc",
            tool_names=("ref_entity",),
            instruction=(
                f'Your narration introduced "{name}" '
                f'(roster id "{roster_id}"). Call '
                f'ref_entity(entity_id="{roster_id}", alias_used="{name}").'
            ),
        ))

    proposed_update_ids = {
        _normalized_label(e.update_entity_id)
        for e in effects
        if _effect_type(e) == EffectType.UPDATE_ENTITY and e.update_entity_id
    }
    for update in delta.npc_updates:
        summary = _mutation_grade(update, narrative)
        if not summary:
            continue
        label = str(update.new_name or update.name or update.id or "").strip()
        # add_npc earlier this turn already persisted initial state; a
        # redundant update_entity would be a false alarm.
        if label and any(
            _labels_overlap(label, proposed_name)
            for proposed_name in proposed_npc_names
        ):
            continue
        roster_id = _resolve_roster_id(
            world_state, str(update.id or ""), label
        )
        if not roster_id:
            continue
        if _normalized_label(roster_id) in proposed_update_ids or any(
            _labels_overlap(label, e.update_entity_id)
            for e in [
                e for e in effects
                if _effect_type(e) == EffectType.UPDATE_ENTITY
            ]
        ):
            continue
        display = label or roster_id
        signals.append(StateFollowupSignal(
            kind="npc_update",
            tool_names=("update_entity",),
            instruction=(
                f'"{display}" (roster id "{roster_id}") {summary} this turn. '
                f'Call update_entity(entity_id="{roster_id}") with the '
                f"matching arguments."
            ),
        ))

    # A strongly-cued proper name no store knows ("Elara Venn's eyes
    # narrow") is an identity event: either a new person on stage, or an
    # existing generically-labeled entity being named for the first time.
    # The narrator sees the roster and decides which; every resulting call
    # still passes the normal validators and the add_npc dedup judge.
    known_labels: list[str] = []
    for npc in world_state.npcs.values():
        known_labels.append(npc.name)
        known_labels.extend(npc.aliases)
    known_labels.extend(str(label or "") for label in known_entity_labels)
    if player_name:
        known_labels.append(player_name)
    for candidate in sorted(strong_npc_name_cues(narrative)):
        # Token-subset overlap, matching the audit observer: the cue "Kael"
        # is already covered by roster/player label "Kael Windrunner".
        if any(_labels_overlap(candidate, label) for label in known_labels):
            continue
        if any(
            _labels_overlap(candidate, proposed_name)
            for proposed_name in proposed_npc_names
        ) or any(
            _labels_overlap(candidate, ref.ref_alias_used)
            for ref in proposed_refs
        ):
            continue
        signals.append(StateFollowupSignal(
            kind="unnamed_identity",
            tool_names=("add_npc", "update_entity", "ref_entity"),
            instruction=(
                f'Your narration names "{candidate}", who is not tracked '
                f"under any roster name. If this is a NEW person on stage, "
                f'call add_npc(name="{candidate}"). If this names an '
                f"existing roster entity currently under a generic label, "
                f"call update_entity with that entity_id and "
                f'new_name="{candidate}" instead. Skip the call if '
                f'"{candidate}" is not an on-stage person.'
            ),
        ))

    return signals[:MAX_SIGNALS_PER_TURN]
