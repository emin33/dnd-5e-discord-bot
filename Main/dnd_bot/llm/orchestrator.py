"""DM Orchestrator - drives the per-turn pipeline from player action to narration.

Triage is a structured-output call on the brain client
(client.chat(json_schema=TriageSchema)); mechanics resolve before narration;
the narrator then emits prose plus tool calls, which are converted into
ProposedEffects (tool_calls_to_effects), validated, executed, and synced to
world state and the knowledge graph. The legacy PROSE+INTENTS text format
survives only as a fallback when the narrator returns no tool calls.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Callable, Optional, TYPE_CHECKING
import asyncio
import json
import re
import uuid

from pydantic import BaseModel
import structlog

from .client import get_llm_client, get_narrator_client_for, OllamaClient, AnthropicClient, _write_debug_log
from .narrative_signals import select_narrator_tier
from .narration import NarrationSpec, NarrationStrategy
from .continuity import NarrativeGovernance
from .json_extract import extract_json_object
from .brains.base import BrainContext
from .brains.narrator import NarratorBrain, get_narrator
from ..game.identity import (
    explicit_npc_naming_link,
    identity_keys,
    is_generic_npc_label,
    locations_equivalent,
    resolve_unique_identity,
)
from .brains.adjudicator import EffectsAdjudicator, get_adjudicator
from .intents import validate_narrator_format
from .narrator_tools import (
    get_narrator_tools_for_tier,
    tool_calls_to_effects,
)
from .extractors.entity_extractor import get_entity_extractor, EntityExtractor
from .extractors.state_extractor import get_state_extractor, StateExtractor
from .turn_logger import get_turn_logger, TurnLogger
from .effects import (
    ProposedEffect,
    EffectType,
    EffectValidator,
    EffectExecutor,
    build_effect_idempotency_key,
)
from ..game.combat.encounter import (
    gather_scene_hostiles,
    guess_monster_index,
    start_encounter,
)
from ..game.world_store import WorldStateStore
from ..game.mechanics.dice import get_roller, DiceRoll
from ..game.mechanics.validation import validate_action, ValidationResult
from ..game.scene.registry import SceneEntityRegistry
from ..data.repositories import get_character_repo
from ..data.repositories.inventory_repo import get_inventory_repo
from ..models import Character
from ..models.npc import SceneEntity, EntityType, Disposition

if TYPE_CHECKING:
    from ..game.session import GameSession
    from ..game.world_state import WorldState, StateDelta

logger = structlog.get_logger()

# Style rotation — cycles through tone hints so consecutive scenes feel different
NARRATOR_STYLES = [
    "Write with tense, clipped sentences. Short and punchy.",
    "Write with lush, atmospheric detail. Rich sensory imagery.",
    "Write with dry wit and understated tension. Let subtext do the work.",
    "Write with cinematic pacing. Wide establishing shots, then tight close-ups.",
    "Write with a folkloric tone. The world feels ancient and storied.",
]


def get_style_hint(turn: int) -> str:
    """Get a rotating style hint for narrator variety."""
    return NARRATOR_STYLES[turn % len(NARRATOR_STYLES)]


def _get_client_provider(client) -> str:
    """Get the provider name for a client instance (for logging)."""
    if isinstance(client, AnthropicClient):
        return "anthropic"
    if isinstance(client, OllamaClient):
        return "ollama"
    return type(client).__name__.replace("Client", "").lower()


# Cap for the KG <entity_relationships> YAML injected into narrator context.
# to_context_yaml caps entity COUNT (max_entities=15) but not per-entity size,
# so a densely connected end-game subgraph can balloon the block and eat the
# local narrator's context budget.
KG_CONTEXT_MAX_CHARS = 4000


def _explicit_narrative_tags(
    mentioned_entity_ids: list[str],
    narrator_ref_ids: list[str],
) -> list[str]:
    """Return stable, explicit provenance tags for a narrative episode."""
    return list(dict.fromkeys(mentioned_entity_ids + narrator_ref_ids))


def _effect_telemetry_record(effect) -> dict:
    """Compact, stable effect shape for proposed/executed/rejected logs."""
    effect_type = getattr(getattr(effect, "effect_type", None), "value", "")
    record = {"type": effect_type}
    if effect_type == "ref_entity":
        record["ref_id"] = getattr(effect, "ref_entity_id", None)
    alias = getattr(effect, "ref_alias_used", None)
    if alias:
        record["ref_alias"] = alias
    if effect_type == "add_npc":
        record["npc_name"] = getattr(effect, "npc_name", None)
        canonical_id = getattr(effect, "npc_canonical_id", None)
        if canonical_id:
            record["canonical_id"] = canonical_id
    if effect_type == "update_entity":
        record["entity_id"] = getattr(effect, "update_entity_id", None)
        record["changes"] = [
            name
            for name, value in (
                ("importance", getattr(effect, "update_importance", None)),
                ("disposition", getattr(effect, "update_disposition", None)),
                ("status", getattr(effect, "update_status", None)),
                (
                    "description",
                    getattr(effect, "update_description_addition", None),
                ),
                ("add_items", getattr(effect, "update_add_items", None)),
                ("remove_items", getattr(effect, "update_remove_items", None)),
            )
            if value not in (None, [], "")
        ]
    item = getattr(effect, "item_name", None) or getattr(effect, "object_name", None)
    if item:
        record["item"] = item
    return record


# Words that never name a concrete item on their own. The live player-state
# sweep (run 20260724_231350) caught the old "item" fallback substring-matching
# a junk 'unknown item' row and transferring THAT instead of the named compass.
_NON_ITEM_TOKENS = frozenset({
    "it", "them", "this", "that", "these", "those", "item", "items",
    "one", "ones", "thing", "things", "something", "object", "objects",
})

_ITEM_STOPWORDS = frozenset({
    "a", "an", "the", "of", "my", "your", "her", "his", "their", "its",
    "from", "for", "some", "own",
})


def _format_coin(copper: int) -> str:
    """Render a copper amount in the largest denomination that divides it."""
    copper = int(copper)
    if copper and copper % 100 == 0:
        return f"{copper // 100}gp"
    if copper and copper % 10 == 0:
        return f"{copper // 10}sp"
    return f"{copper}cp"


def _item_content_tokens(text: str) -> set[str]:
    """Lowercased tokens of an item phrase, minus articles/possessives.

    Trailing plurals fold to singular so "2 daggers" and "Dagger" compare
    equal; both sides of every comparison go through here, so the folding
    stays consistent.
    """
    tokens = set()
    for token in re.findall(r"[a-z0-9']+", (text or "").casefold()):
        if token in _ITEM_STOPWORDS:
            continue
        if len(token) > 3 and token.endswith("s") and not token.endswith("ss"):
            token = token[:-1]
        tokens.add(token)
    return tokens


def _is_generic_item_phrase(item_name: Optional[str]) -> bool:
    """True when the phrase cannot name a specific item ("it", "item", "")."""
    tokens = _item_content_tokens(item_name or "")
    return not tokens or tokens <= _NON_ITEM_TOKENS


def _srd_goods_match(item_name: str) -> bool:
    """True when the SRD equipment/magic-item catalog plausibly sells this.

    Exact index first, then equality of content-token SETS, so word order
    and articles don't matter ("Healing Potion" finds "Potion of Healing")
    while a partial phrase stays out. Subset matching was too loose in
    exactly the direction that hurts: a bare "silver" on a receive-money
    turn would have matched "Silver Dragon Scale Mail" and sold it.

    Anything unmatched is not goods, so the deterministic commerce path
    stays out of the turn — the narrator's receipted update_player owns
    whatever actually changes hands. A false negative costs nothing; a
    false positive charges the player for something they never bought.
    """
    try:
        from ..data.srd import get_srd
        srd = get_srd()
    except Exception:
        return False  # catalog unavailable → nothing verifiable to sell

    slug = "-".join(re.findall(r"[a-z0-9]+", item_name.casefold()))
    if slug and (srd.get_equipment(slug) or srd.get_magic_item(slug)):
        return True

    tokens = _item_content_tokens(item_name)
    if not tokens:
        return False
    for category in ("equipment", "magic_items"):
        for entry in srd.get_all(category).values():
            if tokens == _item_content_tokens(str(entry.get("name") or "")):
                return True
    return False


def _resolve_transfer_item(
    action_lower: str,
    candidate_names: list[str],
    item_name: Optional[str],
) -> Optional[str]:
    """Resolve which held item a transfer names, from real holdings only.

    When triage named an item, that name decides WHICH holding is in play;
    appearing in the sentence only breaks ties among holdings that already
    match it. Letting any incidentally-mentioned holding win outright
    transferred the wrong item — "I hand the compass to Mara, then draw an
    arrow" moved the Arrow — and even "succeeded" for an item the player
    does not hold. If the named item matches no holding, resolution fails
    closed rather than substituting something else.

    Only when triage named nothing usable does the longest holding mentioned
    in the text win. Generic phrases never match anything — the old fallback
    matched 'unknown item' on the literal "item" (run 20260724_231350 T9).
    """
    held = [name for name in candidate_names if name]
    mentioned = [name for name in held if name.casefold() in action_lower]

    if item_name and not _is_generic_item_phrase(item_name):
        item_lower = item_name.casefold()
        matching = [
            name for name in held
            if item_lower in name.casefold() or name.casefold() in item_lower
        ]
        if not matching:
            return None  # named something the giver does not hold
        preferred = [name for name in matching if name in mentioned]
        pool = preferred or matching
        return max(pool, key=len)

    if mentioned:
        return max(mentioned, key=len)
    return None


# update_player mutation families, as (field, empty value) pairs. A
# deterministic handler that claims a family owns it for the whole turn.
_UPDATE_PLAYER_FIELDS = (
    ("player_currency_delta", {}),
    ("player_item_grant", []),
    ("player_item_remove", []),
    ("player_hp_delta", None),
    ("player_add_conditions", []),
    ("player_remove_conditions", []),
    ("player_spell_slot_used", None),
)


def _scrub_claimed_mutations(
    narrator_effects: list,
    authoritative_effects: list,
) -> list:
    """Drop narrator mutations a deterministic handler already claims.

    Dedup is per claimed field (update_player) or per targeted entity
    (update_entity) rather than per effect type, so one narrated event still
    has exactly one writer while an unrelated mutation on the same turn — a
    narrator HP change during a purchase, an update_entity for a bystander —
    survives. A narrator effect left with nothing to change is dropped
    entirely: update_player validation rejects no-op calls.
    """
    claimed_player_fields = {
        field
        for effect in authoritative_effects
        if effect.effect_type == EffectType.UPDATE_PLAYER
        for field, empty in _UPDATE_PLAYER_FIELDS
        if getattr(effect, field, empty) not in (empty, None)
    }
    claimed_entity_ids = {
        (effect.update_entity_id or "").strip()
        for effect in authoritative_effects
        if effect.effect_type == EffectType.UPDATE_ENTITY
    }

    kept: list = []
    for effect in narrator_effects:
        if effect.effect_type == EffectType.UPDATE_ENTITY:
            if (effect.update_entity_id or "").strip() in claimed_entity_ids:
                continue
            kept.append(effect)
            continue
        if effect.effect_type != EffectType.UPDATE_PLAYER or not claimed_player_fields:
            kept.append(effect)
            continue
        remaining = effect.model_copy(deep=True)
        mutates = False
        for field, empty in _UPDATE_PLAYER_FIELDS:
            if field in claimed_player_fields:
                # Fresh container per effect — never alias the module-level
                # empty value into a live ProposedEffect.
                setattr(remaining, field, type(empty)() if empty is not None else None)
            elif getattr(remaining, field, empty) not in (empty, None):
                mutates = True
        if mutates:
            kept.append(remaining)
    return kept


def _reanchor_returning_npc_ids(
    delta,
    knowledge_graph,
    current_npc_ids: set[str],
    candidate_entity_ids: list[str],
) -> list[tuple[str, str, str]]:
    """Reuse canonical KG ids when an off-scene NPC returns.

    Scene rescoping removes departed non-important NPCs from
    ``WorldState.npcs`` while the KG retains them. If one later returns in
    narration, the state extractor sees no current-scene roster entry and may
    emit a fresh UUID. Proper names resolve against the whole durable graph;
    generic roles remain restricted to entities activated by this turn.
    """
    if not delta or not knowledge_graph:
        return []

    graph_candidates = [
        entity
        for entity in (getattr(knowledge_graph, "_entities", {}) or {}).values()
        if getattr(getattr(entity, "entity_type", None), "value", "") == "npc"
        and str(getattr(entity, "name", "") or "").strip()
    ]
    activated_candidates: list[Any] = []
    for entity_id in dict.fromkeys(candidate_entity_ids):
        entity = knowledge_graph.get_entity(entity_id)
        entity_type = getattr(getattr(entity, "entity_type", None), "value", "")
        name = str(getattr(entity, "name", "") or "").strip()
        if not entity or entity_type != "npc" or not name:
            continue
        # The graph uses node_id; the shared resolver accepts either id or
        # node_id, so expose the canonical graph id without mutating Entity.
        activated_candidates.append(entity)
        if all(
            getattr(candidate, "node_id", None) != entity.node_id
            for candidate in graph_candidates
        ):
            graph_candidates.append(entity)

    reanchored: list[tuple[str, str, str]] = []
    proposed_npcs = list(getattr(delta, "new_npcs", []) or [])
    for npc in proposed_npcs:
        name = str(getattr(npc, "name", "") or "").strip()
        generic = is_generic_npc_label(name)
        match = resolve_unique_identity(
            name,
            activated_candidates if generic else graph_candidates,
        )
        if (
            match is None
            and generic
            and len(proposed_npcs) == 1
            and len(activated_candidates) == 1
        ):
            # A sole role label ("Ragpicker", "the hooded figure") is not a
            # new identity.  The narrator's sole activated/ref'd NPC is the
            # only safe canonical anchor; ambiguous turns still abstain.
            match = activated_candidates[0]
        if match is None:
            continue
        canonical_id = match.node_id
        if canonical_id in current_npc_ids or npc.id == canonical_id:
            continue
        previous_id = npc.id
        npc.id = canonical_id
        reanchored.append((previous_id, canonical_id, npc.name))
    return reanchored


def _canonicalize_npc_effect_ids(
    effects: list[ProposedEffect],
    world_state,
    knowledge_graph,
) -> list[ProposedEffect]:
    """Rewrite narrator NPC slugs onto durable UUIDs before extraction.

    Models sometimes echo ``cinder-vex`` even though the authoritative graph
    key is a UUID. Both dialects resolve at execution time, but retaining the
    slug prevents StateDelta from reconciling an off-screen return. Only exact,
    unique NPC identities are rewritten.
    """
    if not effects:
        return effects
    graph_npcs = [
        entity
        for entity in (getattr(knowledge_graph, "_entities", {}) or {}).values()
        if getattr(getattr(entity, "entity_type", None), "value", "") == "npc"
    ]
    normalized: list[ProposedEffect] = []
    for effect in effects:
        field = None
        value = ""
        alias = None
        if effect.effect_type == EffectType.REF_ENTITY and effect.ref_entity_id:
            field = "ref_entity_id"
            value = effect.ref_entity_id
            alias = effect.ref_alias_used
        elif effect.effect_type == EffectType.UPDATE_ENTITY and effect.update_entity_id:
            field = "update_entity_id"
            value = effect.update_entity_id
        if field is None:
            normalized.append(effect)
            continue

        existing = (
            world_state._resolve_npc(value, alias or None)
            if world_state is not None
            else None
        )
        canonical_id = getattr(existing, "id", None)
        if not canonical_id:
            match = resolve_unique_identity(value, graph_npcs)
            canonical_id = getattr(match, "node_id", None)
        normalized.append(
            effect.model_copy(update={field: canonical_id})
            if canonical_id and canonical_id != value
            else effect
        )
    return normalized


_ID_FILLER_TOKENS = {"the", "a", "an", "of", "with", "and", "in", "on", "at", "to"}

# Everyday object/material nouns that identify a THING only in context. A bare
# one of these must never truncate-resolve onto a tracked entity the way a
# proper name token ('gideon') legitimately does.
_ID_COMMON_NOUNS = {
    "blade", "book", "boot", "bottle", "box", "brass", "candle", "chain",
    "cloak", "coat", "coin", "compass", "cup", "door", "gate", "gear",
    "glass", "hatch", "iron", "key", "knife", "lamp", "lantern", "ledger",
    "letter", "lock", "mask", "note", "page", "paper", "pouch", "ring",
    "rope", "sack", "seal", "shard", "silver", "staff", "stair", "stone",
    "sword", "table", "token", "vial", "wood", "wooden",
}


def _id_tokens(value: object) -> frozenset:
    words = re.findall(r"[a-z0-9]+", str(value or "").lower())
    return frozenset(word for word in words if word not in _ID_FILLER_TOKENS)


def _resolve_invented_scene_ids(
    effects: list[ProposedEffect],
    world_state,
    knowledge_graph,
) -> list[ProposedEffect]:
    """Resolve narrator-embellished or partial ids onto their unique entity.

    Models compose descriptive slugs around a real referent's name —
    ``low-wooden-door-with-token-indentation`` for the scene item
    ``low wooden door`` (soak 20260723_005611, turns 22-32) — and also
    truncate names to a bare identifying token — ``gideon`` for the
    tracked NPC ``Gideon Hask`` (soak 20260723_230351, turn 77).

    The two directions are tried in order and never pooled, because they
    carry different evidence and different risk:

    1. CONTAINMENT (embellishment): exactly one known entity's full token
       set (>=2 tokens) sits inside the proposed id. This is the original
       soak-#1 rule and stays authoritative — an inverse candidate must
       never be able to out-vote it (pooling the two into one ``matched``
       set let an unrelated superset veto a good containment match).
    2. SUBSET (truncation), only when containment found nothing: the
       proposed id's tokens are a strict subset of one candidate's. A
       single bare token is the dangerous shape here — ``door`` sits
       inside ``carved wooden door`` and ``man`` inside half a roster — so
       one-token ids resolve ONLY when the token is identity-bearing
       (not a generic role/common noun, >=4 chars) and the candidate is a
       proper-named NPC. ``gideon`` -> ``Gideon Hask`` passes; ``door``,
       ``man``, ``brass`` do not.

    Both directions require a unique winner and a type-compatible one: an
    item-shaped id never rewrites onto an NPC. Anything else keeps the
    fail-closed rejection — ``corvins-hallway`` contains no known entity
    and still rejects; a ``sera`` with two tracked Seras abstains.

    A winner must also agree with ``ref_alias_used`` when the narrator
    supplied one. The alias is the name that actually appeared in prose, so
    it is the strongest available evidence about who was meant; a rewrite
    that contradicts it mis-binds the effect AND permanently grafts the
    wrong alias onto the resolved entity downstream (post-merge review,
    seam 2). Disagreement abstains rather than guessing.
    """
    if not effects:
        return effects

    known_ids: set[str] = set()
    # (id, tokens, entity_type, is_proper_named) — the type keeps an
    # item-shaped id from rewriting onto an NPC, and the proper-name flag
    # gates the one-token subset direction.
    candidates: list[tuple[str, frozenset, str, bool]] = []
    # Registered aliases participate in alias-agreement ONLY, never in
    # matching: a recorded epithet ('the innkeeper') is evidence about who a
    # prose alias meant, but must not widen the id-matching surface.
    alias_labels: dict[str, list[frozenset]] = {}
    if world_state is not None:
        for item_id in world_state.scene_items:
            known_ids.add(item_id)
            tokens = _id_tokens(item_id)
            if tokens:
                candidates.append((item_id, tokens, "item", False))
        for npc_id, npc in world_state.npcs.items():
            known_ids.add(npc_id)
            tokens = _id_tokens(npc.name)
            if tokens:
                candidates.append((
                    npc_id, tokens, "npc", not is_generic_npc_label(npc.name),
                ))
            for alias in npc.aliases:
                alias_label = _id_tokens(alias)
                if alias_label:
                    alias_labels.setdefault(npc_id, []).append(alias_label)
    for entity in (getattr(knowledge_graph, "_entities", {}) or {}).values():
        node_id = str(getattr(entity, "node_id", "") or "")
        if not node_id:
            continue
        known_ids.add(node_id)
        entity_type = str(
            getattr(getattr(entity, "entity_type", None), "value", "") or ""
        )
        entity_name = str(getattr(entity, "name", "") or "")
        proper = entity_type == "npc" and not is_generic_npc_label(entity_name)
        for label in (node_id, entity_name):
            tokens = _id_tokens(label)
            if tokens:
                candidates.append((node_id, tokens, entity_type, proper))
        for alias in getattr(entity, "aliases", None) or []:
            alias_label = _id_tokens(alias)
            if alias_label:
                alias_labels.setdefault(node_id, []).append(alias_label)
    known_token_ids = {_id_tokens(kid): kid for kid in known_ids}

    normalized: list[ProposedEffect] = []
    for effect in effects:
        field = None
        value = ""
        if effect.effect_type == EffectType.REF_ENTITY and effect.ref_entity_id:
            field = "ref_entity_id"
            value = effect.ref_entity_id
        elif effect.effect_type == EffectType.UPDATE_ENTITY and effect.update_entity_id:
            field = "update_entity_id"
            value = effect.update_entity_id
        if field is None:
            normalized.append(effect)
            continue
        value_tokens = _id_tokens(value)
        if (
            value in known_ids
            or value_tokens in known_token_ids
            or not value_tokens
        ):
            # Resolvable as-is (exact id or slug dialect of one) — leave it
            # to the normal validation path.
            normalized.append(effect)
            continue
        # Direction 1 — containment (narrator embellished a real name).
        # Authoritative: never pooled with the subset direction, so an
        # unrelated superset candidate cannot veto a good match here.
        matched = {
            candidate_id
            for candidate_id, candidate_tokens, _type, _proper in candidates
            if len(candidate_tokens) >= 2 and candidate_tokens < value_tokens
        }
        direction = "containment"
        if not matched:
            # Direction 2 — subset (narrator truncated a real name). A bare
            # one-token id is only identity-bearing when it is a distinctive
            # proper-name token: 'gideon' resolves, 'door'/'man' must not.
            single = len(value_tokens) == 1
            token = next(iter(value_tokens)) if single else ""
            single_ok = (
                len(token) >= 4
                and not is_generic_npc_label(token)
                and token not in _ID_COMMON_NOUNS
            )
            matched = {
                candidate_id
                for candidate_id, candidate_tokens, _type, proper in candidates
                if value_tokens < candidate_tokens
                and (not single or (single_ok and proper))
            }
            direction = "subset"
        if len(matched) == 1:
            resolved_id = next(iter(matched))
            # Alias agreement: the prose name the narrator actually used must
            # be comparable to one of the winner's labels — 'low wooden door'
            # inside 'low wooden door', 'Gideon Hask' against 'Gideon Hask',
            # and a REGISTERED alias counts: naming promotion stores old
            # epithets ('the innkeeper') on the entity precisely so later
            # paraphrases resolve. A disjoint alias ('Mira' resolving onto
            # 'Gideon Hask') is evidence the rewrite is wrong, so it abstains.
            alias_tokens = _id_tokens(getattr(effect, "ref_alias_used", "") or "")
            winner_labels = [
                label_tokens
                for candidate_id, label_tokens, _type, _proper in candidates
                if candidate_id == resolved_id
            ] + alias_labels.get(resolved_id, [])
            if alias_tokens and not any(
                alias_tokens <= label_tokens or label_tokens <= alias_tokens
                for label_tokens in winner_labels
            ):
                logger.debug(
                    "invented_id_alias_mismatch_abstained",
                    proposed_id=value,
                    resolved_id=resolved_id,
                    alias=effect.ref_alias_used,
                )
                normalized.append(effect)
                continue
            # Type compatibility: an item-shaped id must not rewrite onto an
            # NPC (and vice versa) just because their tokens overlap.
            resolved_types = {
                candidate_type
                for candidate_id, _tokens, candidate_type, _proper in candidates
                if candidate_id == resolved_id and candidate_type
            }
            proposed_types = {
                candidate_type
                for _id, candidate_tokens, candidate_type, _proper in candidates
                if candidate_type and candidate_tokens & value_tokens
            }
            if (
                len(proposed_types) > 1
                or not resolved_types
                or not proposed_types
                or resolved_types & proposed_types
            ):
                logger.info(
                    "invented_id_resolved",
                    proposed_id=value,
                    resolved_id=resolved_id,
                    direction=direction,
                    effect_type=effect.effect_type.value,
                )
                normalized.append(
                    effect.model_copy(update={field: resolved_id})
                )
            else:
                logger.debug(
                    "invented_id_type_mismatch_abstained",
                    proposed_id=value,
                    resolved_id=resolved_id,
                )
                normalized.append(effect)
        else:
            normalized.append(effect)
    return normalized


def _reanchor_same_turn_generic_npc_effects(
    effects: list[ProposedEffect],
    delta,
    world_state,
) -> list[ProposedEffect]:
    """Anchor an early role-slug update to a sole NPC created this turn.

    Narrator effects are proposed before StateDelta materializes anonymous
    people.  A model may therefore target ``archive-courier`` while the same
    prose creates ``the courier``.  After StateDelta applies, reconcile only a
    unique generic-role token overlap; ambiguous scenes deliberately abstain.
    """
    if not effects or not delta or world_state is None:
        return effects

    generic_new_npcs = [
        npc
        for npc in (getattr(delta, "new_npcs", []) or [])
        if is_generic_npc_label(str(getattr(npc, "name", "") or ""))
        and getattr(npc, "id", None) in world_state.npcs
    ]
    if not generic_new_npcs:
        return effects

    noise = {"a", "an", "the", "this", "that"}
    anchored: list[ProposedEffect] = []
    for effect in effects:
        target = str(effect.update_entity_id or "").strip()
        if effect.effect_type != EffectType.UPDATE_ENTITY or not target:
            anchored.append(effect)
            continue
        resolved_target = world_state._resolve_npc(target, target)
        if resolved_target is not None:
            if (
                resolved_target.id != target
                and any(
                    npc.id == resolved_target.id for npc in generic_new_npcs
                )
            ):
                anchored.append(effect.model_copy(update={
                    "update_entity_id": resolved_target.id,
                }))
            else:
                anchored.append(effect)
            continue

        target_words = set(re.findall(r"[a-z0-9]+", target.casefold())) - noise
        matches = []
        for npc in generic_new_npcs:
            role_words = (
                set(re.findall(r"[a-z0-9]+", npc.name.casefold())) - noise
            )
            if target_words and role_words and target_words.intersection(role_words):
                matches.append(npc)
        unique_matches = {npc.id: npc for npc in matches}
        if len(unique_matches) != 1:
            anchored.append(effect)
            continue

        npc = next(iter(unique_matches.values()))
        anchored.append(effect.model_copy(update={"update_entity_id": npc.id}))
        logger.info(
            "same_turn_generic_npc_effect_reanchored",
            previous_id=target,
            canonical_id=npc.id,
            npc_name=npc.name,
        )
    return anchored


def _merge_delta_npcs_with_narrator_refs(
    delta,
    world_state,
    narrator_refs: list[tuple[str, str]],
    narrator_prose: str = "",
) -> list[tuple[str, str, str]]:
    """Turn extractor ``new_npcs`` into updates when tools proved identity.

    ``ref_entity`` is the narrator's explicit identity claim.  The cheaper
    StateDelta extractor can nevertheless see a newly revealed proper name
    (``the cloaked woman`` -> ``Mira``) and emit a fresh UUID.  When that name
    exactly matches the alias attached to a reference, merge the descriptive
    delta onto the referenced WorldState NPC instead of fragmenting the graph.

    A generic role is also safe to merge when the turn contains exactly one
    referenced current NPC and one proposed NPC; roles are aliases, never new
    durable identities.  All ambiguous cases abstain.
    """
    if not delta or not world_state or not narrator_refs:
        return []

    resolved_refs: list[tuple[Any, str]] = []
    seen_ids: set[str] = set()
    for entity_id, alias in narrator_refs:
        existing = world_state._resolve_npc(entity_id, alias or None)
        if existing is None or existing.id in seen_ids:
            continue
        seen_ids.add(existing.id)
        resolved_refs.append((existing, alias))
    if not resolved_refs:
        return []

    proposed_npcs = list(getattr(delta, "new_npcs", []) or [])
    surviving = []
    merged: list[tuple[str, str, str]] = []
    from ..game.world_state import NPCUpdate

    for proposed in proposed_npcs:
        matches = []
        for existing, alias in resolved_refs:
            if resolve_unique_identity(proposed.name, [existing]) is existing:
                matches.append((existing, alias))
                continue
            if alias and identity_keys(proposed.name) & identity_keys(alias):
                matches.append((existing, alias))
                continue
            if explicit_npc_naming_link(
                narrator_prose,
                str(getattr(existing, "name", "") or ""),
                str(getattr(proposed, "name", "") or ""),
            ):
                matches.append((existing, alias))

        if (
            not matches
            and len(proposed_npcs) == 1
            and len(resolved_refs) == 1
            and is_generic_npc_label(proposed.name)
        ):
            matches = resolved_refs

        unique_matches = {match[0].id: match for match in matches}
        if len(unique_matches) != 1:
            surviving.append(proposed)
            continue

        existing, _alias = next(iter(unique_matches.values()))
        explicit_name = (
            proposed.name
            if not is_generic_npc_label(proposed.name)
            and not resolve_unique_identity(proposed.name, [existing])
            else None
        )
        aliases = None
        if (
            is_generic_npc_label(proposed.name)
            and proposed.name.casefold() != existing.name.casefold()
        ):
            aliases = [proposed.name]
        disposition = (
            proposed.disposition
            if proposed.disposition and proposed.disposition != existing.disposition
            else None
        )
        delta.npc_updates.append(NPCUpdate(
            id=existing.id,
            location=(
                proposed.location
                if proposed.location
                and proposed.location == world_state.current_location
                and proposed.location != existing.location
                else None
            ),
            disposition=disposition,
            description=proposed.description or None,
            important=True if proposed.important and not existing.important else None,
            new_name=explicit_name,
            add_aliases=aliases,
            add_inventory=list(proposed.inventory) or None,
        ))
        merged.append((proposed.id, existing.id, proposed.name))

    delta.new_npcs = surviving
    return merged


def _anchor_returning_add_npc_effects(
    effects: list[ProposedEffect],
    knowledge_graph,
) -> list[ProposedEffect]:
    """Attach a durable graph UUID when add_npc brings back a known NPC.

    Scene-scoped WorldState legitimately forgets off-screen NPC projections,
    but the campaign graph does not.  An exact unique name/alias match means
    the visible arrival is a materialization of that durable identity, not a
    license to mint a second UUID.
    """
    if not effects or knowledge_graph is None:
        return effects
    candidates = [
        entity
        for entity in (getattr(knowledge_graph, "_entities", {}) or {}).values()
        if getattr(getattr(entity, "entity_type", None), "value", "") == "npc"
    ]
    anchored: list[ProposedEffect] = []
    for effect in effects:
        if (
            effect.effect_type != EffectType.ADD_NPC
            or not effect.npc_name
            or is_generic_npc_label(effect.npc_name)
        ):
            anchored.append(effect)
            continue
        match = resolve_unique_identity(effect.npc_name, candidates)
        if match is None:
            anchored.append(effect)
            continue
        anchored_effect = effect.model_copy(update={"npc_canonical_id": match.node_id})
        anchored.append(anchored_effect)
        logger.info(
            "returning_add_npc_anchored",
            npc_name=effect.npc_name,
            canonical_id=match.node_id,
        )
    return anchored


def _drop_dead_npc_reintroductions(delta, dead_npcs: list[Any]) -> list[str]:
    """Remove extractor-created NPCs that resolve to immutable death facts."""
    if not delta or not dead_npcs:
        return []
    surviving = []
    rejections = []
    for proposed in getattr(delta, "new_npcs", []) or []:
        dead = resolve_unique_identity(str(getattr(proposed, "name", "")), dead_npcs)
        if dead is None:
            surviving.append(proposed)
            continue
        rejections.append(
            f"Dead NPC cannot be reintroduced as living: {dead.name}"
        )
    delta.new_npcs = surviving
    return rejections


def _cap_kg_context_yaml(yaml_text: str, max_chars: int = KG_CONTEXT_MAX_CHARS) -> str:
    """Cap KG context YAML, cutting at an entity-entry boundary.

    Entries in ``KnowledgeGraph.to_context_yaml`` output start with
    ``- name:`` at column 0 (items of the ``known_entities`` list), so
    cutting just before one keeps the remaining YAML well-formed. Falls
    back to a line boundary if even the first entry overflows. A one-line
    ``# truncated`` marker is appended so the model (and turn-log readers)
    can tell the block was cut.
    """
    if len(yaml_text) <= max_chars:
        return yaml_text
    first_entry = yaml_text.find("\n- name:")
    cut = yaml_text.rfind("\n- name:", 0, max_chars)
    if cut <= first_entry:
        # The first entry alone overflows (or no entry marker found) —
        # fall back to a whole-line boundary rather than keeping nothing.
        cut = yaml_text.rfind("\n", 0, max_chars)
        if cut <= 0:
            cut = max_chars
    return yaml_text[:cut].rstrip() + "\n# truncated\n"


# =============================================================================
# TRIAGE PROMPT - Rules Brain decides if mechanics are needed
# =============================================================================

TRIAGE_SYSTEM_PROMPT = """You are the action classifier for a D&D 5e game. You make TWO independent judgments per turn:

1. **Narrative significance** — how weighty is the SCENE (about scene context, not action mechanic).
2. **Mechanical classification** — what is the player doing and does it need a roll (about action mechanic).

These judgments are independent. A mundane action in a heavy scene is mechanically routine but narratively weighty. A dramatic action in an empty scene is mechanically routine and narratively routine. Decide narrative significance FIRST, before mechanical classification, because mechanical reasoning will bias scene judgment if you do it the other way.

## NARRATIVE SIGNIFICANCE (decide first)

Tag each turn's narrative weight as `routine`, `notable`, or `climactic`. This decides whether the prose goes to a fast/cheap narrator or a premium narrator that takes its time.

The tag is about the SCENE'S weight, not the action's mechanical complexity. A mundane action ("look up at the sky") in a heavy scene (a bruised omen visible above the village) is weighty. A dramatic action ("draw blade theatrically") in an empty scene (haggling for fun in a market) is mundane.

Judge the scene by signals, not vibes. Check each signal:

- **Stakes:** is failure or success here irreversible, or recoverable?
- **Setup payoff:** does this turn close out something the campaign has been building toward, or is it self-contained?
- **Introduction:** is the turn introducing a person, place, or revelation that will matter to the players going forward?
- **Emotional register:** is the scene's tone quiet/transactional, or weighted (grief, dread, awe, betrayal, awe-of-scale)?

Tier rules based on signals:

- **routine** — zero or one signal present, and the signal is mild.
- **notable** — one or two signals present at meaningful strength, OR a tense single-scene beat (high stakes within one scene, even with no campaign-arc resonance).
- **climactic** — the moment carries CAMPAIGN-ARC weight. Reserved for beats that would matter to a player retelling the campaign months later: a death of a known character, a betrayal of a long-running ally, the moment a campaign-central question is answered, the boss falling, a defining choice with permanent consequences, a long-pursued goal paying off.

Critical: scene-level tension is NOT climactic. A skill check with TPK risk is a tense notable, not a climactic moment, because the campaign arc isn't turning on it. Climactic is reserved for beats the players will REMEMBER.

Boundary discrimination — these patterns LOOK weighty but aren't:

- An ongoing combat round inside a dramatic encounter is still routine. The encounter being important doesn't make every swing important. Only the decisive moment (the boss falling, a death) is climactic.
- A tense skill check with high consequences is notable. Stakes raise it above routine but don't reach climactic — the campaign arc isn't pivoting on a trap disarm.
- A first meeting with an NPC is only notable if that NPC will matter going forward. A vendor encountered for one transaction is routine even on the first visit.
- Dramatic player verbiage with no real stakes is routine. Pose ≠ weight.
- A death is only climactic if the dead character was a known, named figure the players had a relationship with. A reported offscreen death of an unnamed NPC is routine.

When two tiers feel equally valid, pick the LOWER one. The premium tier must be earned — overuse dulls its impact.

**Important: significance is independent of whether the turn needs a roll.** Many notable scenes have no roll (an omen observed, a body discovered, a friend's confession). Don't conflate "no triage signals" with "routine significance."

## DOES THIS NEED A ROLL?

A roll is needed when ALL of these are true:
1. **Uncertainty** — a competent adventurer could plausibly fail at this attempt.
2. **Stakes** — failure changes the situation (something is lost, missed, or made worse). Not just "try again."
3. **Pressure or specialization** — there's time pressure, danger, opposition, OR success requires expertise the character may not have.

If any one is missing → no roll. Narrate the outcome directly.

Boundary discrimination:

- "I open the unlocked door" → no roll. No uncertainty.
- "I pick the locked door" → roll. Uncertainty + stakes.
- "I search the room over an hour" (safe, no pressure) → no roll. They find what's findable.
- "I search the room while guards approach" → roll. Pressure + stakes.
- "I look at the markings on the wall" → no roll. Anyone can see them.
- "I recognize the markings as elven sigils" → roll. Specialized knowledge required.

Routine signals that mean NO ROLL even with adventurer context:
- Movement to a visible destination ("I step closer", "I walk to the bar"). Movement is never a skill check unless the terrain itself is hazardous (cliff, tightrope, rapids).
- Observation of obvious things ("I look around the tavern", "I check who is here") in safe or explored locations.
- Common knowledge a character would know automatically.
- Routine actions with unlimited time and no consequences (setting up camp, lighting a normal fire).
- Player flavor — describing HOW the action looks doesn't add a roll.

## SETTING THE DC

DC scales with difficulty: 5=trivial, 10=easy, 12=moderate, 15=hard, 18=very hard, 20=nearly impossible.

Anchor by situation, not by drama. A casual Perception check in a quiet tavern is DC 10-12, not DC 15. Reserve DC 15+ for genuine challenges where most adventurers would meaningfully struggle.

## CHOOSING THE SKILL

Match the skill to HOW the character is doing the action, not what they're targeting:

- **Perception** — passive noticing (hearing, seeing, smelling).
- **Investigation** — actively examining, deducing, figuring out HOW something works.
- **History / Arcana / Religion / Nature** — recalling knowledge about a topic.
- **Stealth** — moving unseen, hiding.
- **Athletics** — climbing, jumping, swimming, feats of strength.
- **Acrobatics** — balance, dexterity, evasion.
- **Persuasion / Deception / Intimidation** — social influence.
- **Survival** — tracking, foraging, wilderness navigation.
- **Insight** — reading intentions, detecting lies.

Discriminative cases people get wrong:
- "I look for tracks" → Survival, not Perception.
- "I examine the mechanism" → Investigation, not Perception.
- "I recall what I know about elves" → History, not Perception.

## ACTION TYPES

- **attack** — melee/ranged WEAPON attack toward a CREATURE (triggers combat). Spells are not attacks.
- **cast_spell** — casting any spell, including attack cantrips. If the action names a spell, this overrides attack.
- **skill_check** — any action with uncertain outcome requiring ability/skill check (including difficult attacks against objects).
- **saving_throw** — forced saves (resisting effects).
- **purchase** — buying GOODS: the player pays AND receives a tangible item. Paying for help, information, services, or settling a debt is social (record the money in currency_spent). A player RECEIVING money is NEVER a purchase.
- **sell** — selling items.
- **inventory** — managing or MOVING a physical item: equip, drop, use, pick up, hand an item to someone, or receive one. Money is not an item — coins changing hands are social, with the amount in currency_spent.
- **movement** — tactical positioning.
- **social** — conversation that doesn't require persuasion (asking prices, greeting, small talk).
- **exploration** — observing visible things, no roll required.
- **roleplay** — pure character expression with no mechanical effect.

Boundary discrimination:

- "I attack the goblin" → attack, is_creature_target=true.
- "I cast fire bolt at the goblin" → cast_spell. Spell name overrides attack classification.
- "I shoot an arrow at a tree 200m away" → skill_check (DEX, DC 20). Object target with difficulty.
- "I kick a pebble" → roleplay. Trivial object, no stakes.
- "I smash the locked chest" → skill_check (STR, DC by material).
- "I greet the innkeeper warmly" → social. No persuasion needed.
- "I convince the guard to let us pass" → skill_check (Persuasion).
- "I buy a dagger from the smith" → purchase (goods received for money).
- "I pay the guide two gold for her help" → social, currency_spent {"gold": 2}. Money for a service — no goods, not a purchase.
- "The merchant counts five silver into my palm" → social. The player RECEIVES money: not a purchase, and never currency_spent.
- "I hand my brass compass to Mara, she accepts it" → inventory. An ITEM changes hands, so it is an inventory transfer even though the moment is conversational. The money examples above are social ONLY because coins are not items.
- "She takes the vial from her coat and hands it to me" → inventory. Receiving an item is a transfer too.

## FOR ATTACK ACTIONS (creature targets only)

- target_name: The creature being attacked
- is_creature_target: MUST be true (otherwise use skill_check)

## FOR SKILL CHECKS

- needs_roll: true
- skill: The relevant skill (or "none" for raw ability check)
- ability: dexterity, strength, wisdom, charisma, etc.
- dc: 5=trivial, 10=easy, 15=moderate, 20=hard, 25=very hard, 30=nearly impossible
- on_success: What SPECIFICALLY the character discovers, achieves, or learns. Be concrete!
- on_failure: What SPECIFICALLY happens on failure — what they miss, what goes wrong, or what consequence occurs. Never leave empty.

on_success/on_failure are CRITICAL — the narrator uses these to know what to describe.
Examples:
- Perception check to scan for danger:
  on_success: ["You spot movement in the treeline — two figures crouching behind brush"]
  on_failure: ["The forest seems quiet and still — you notice nothing unusual"]
- Investigation check on strange runes:
  on_success: ["The runes are a warding spell, old but still active — touching them would be dangerous"]
  on_failure: ["The symbols blur together — you can't make sense of the pattern"]
- Athletics check to climb a cliff:
  on_success: ["You find handholds and pull yourself up safely"]
  on_failure: ["Your grip slips — you slide back down, taking 1d4 bludgeoning damage"]
- Persuasion check on a guard:
  on_success: ["The guard sighs and steps aside, waving you through"]
  on_failure: ["The guard crosses his arms — 'Not happening. Move along.'"]

## RESOURCE CONSUMPTION

Note any resources consumed by the action:
- Firing a bow/crossbow → consumes 1 ammunition
- Throwing a weapon → weapon leaves inventory unless retrieved
- Using a consumable → item consumed

## CURRENCY SPENDING

Always include `currency_spent` when the player spends, loses, or gives away money — regardless of action_type. Use the denomination the player specifies (copper, silver, electrum, gold, platinum). For amounts like "half" or "all," compute from the character's current Currency in the context. NEVER set currency_spent for money the player RECEIVES — that is a gain, not spending.

## FOR PURCHASES

When action_type is "purchase", also provide:
- item_name: The tangible item being bought (a real object the player will carry — never a service, favor, or piece of information)
- item_cost: Estimated cost in gold (use SRD prices if known)
- quantity: How many

## FOR ROLEPLAY/SOCIAL/EXPLORATION

Just provide narrative_direction with guidance for the narrator.

## SCENE ENTITIES

When "Scene Entities" context is provided, these NPCs/creatures EXIST and can be interacted with.
- Entities marked [HOSTILE] can be attacked
- Entities marked [THREATENING] may become hostile

## OUTPUT FORMAT

Output valid JSON. Field order matters — emit them in this sequence so your
scene-weight judgment is fixed before mechanical details narrow your view:

```json
{
    // STEP 1: think through the scene first
    "reasoning": "Brief explanation. Note which significance signals fired and why.",

    // STEP 2: commit to scene weight before letting mechanics bias you
    "narrative_significance": "routine",  // routine | notable | climactic — REQUIRED

    // STEP 3: classify the mechanical action
    "action_type": "skill_check",  // REQUIRED

    // For attack (creature targets ONLY):
    "target_name": "goblin",
    "is_creature_target": true,

    // For skill_check (includes difficult ranged/melee vs objects):
    "needs_roll": true,
    "skill": "survival",  // Match skill to the action (see skill guide above)
    "ability": "wisdom",
    "dc": 12,  // Set DC using the difficulty table (5-20 range)
    "on_success": ["You find fresh tracks leading north"],
    "on_failure": ["The ground reveals nothing useful"],

    // For purchase:
    "item_name": "Dagger",
    "item_cost": 2,
    "quantity": 1,

    // For social/roleplay/exploration:
    "narrative_direction": "Guidance for narrator",

    // Resources consumed (if any):
    "resources_consumed": [{"item": "Arrow", "quantity": 1}],

    // Currency spent/lost (if any):
    "currency_spent": {"gold": 50}  // or {"silver": 10, "copper": 5}
}
```

Only include fields relevant to the action_type, but ALWAYS include
`reasoning`, `narrative_significance`, and `action_type` (in that order)."""


# =============================================================================
# PYDANTIC SCHEMA FOR STRUCTURED OUTPUT
# =============================================================================

class ActionType(str, Enum):
    """Types of player actions for routing."""
    ATTACK = "attack"           # Combat attack
    CAST_SPELL = "cast_spell"   # Spell casting
    SKILL_CHECK = "skill_check" # Ability/skill check
    SAVING_THROW = "saving_throw"  # Forced save
    PURCHASE = "purchase"       # Buying items
    SELL = "sell"               # Selling items
    INVENTORY = "inventory"     # Use/equip/drop items
    MOVEMENT = "movement"       # Tactical movement
    SOCIAL = "social"           # NPC interaction (no mechanics)
    EXPLORATION = "exploration" # Looking around, investigating
    ROLEPLAY = "roleplay"       # Pure narrative


class TriageSchema(BaseModel):
    """Pydantic schema for structured triage output.

    IMPORTANT: All optional fields use plain defaults (str="", int=0)
    instead of Optional[T]=None to avoid anyOf unions in the JSON schema.
    Groq's JSON validator chokes on anyOf patterns, causing intermittent
    "Failed to generate/validate JSON" 400 errors.
    """
    # Action classification
    action_type: str  # One of ActionType values
    reasoning: str = ""

    # For skill_check/saving_throw actions
    needs_roll: bool = False
    roll_type: str = ""       # "ability_check" or "saving_throw"
    ability: str = ""
    skill: str = ""
    dc: int = 0               # 0 = not set
    advantage: bool = False
    disadvantage: bool = False
    advantage_reason: str = ""
    on_success: list[str] = []
    on_failure: list[str] = []

    # For purchase/sell actions
    item_name: str = ""
    item_cost: int = 0        # In gold pieces, 0 = not set
    quantity: int = 1

    # For attack actions
    target_name: str = ""     # What/who they're attacking
    is_creature_target: bool = False  # True if target is a creature (triggers combat)

    # For social/roleplay (no mechanics needed)
    narrative_direction: str = ""

    # Phase C: tier-routing hint (see Docs/Roadmap/tiered_narrator.md)
    # Brain classifies the narrative weight so the orchestrator can pick a
    # premium narrator client for moments worth dwelling on.
    # - "routine"   : ordinary turns; trash-mob combat, casual exploration, small talk
    # - "notable"   : first major NPC, scene change, surprising reveal
    # - "climactic" : boss fight, character death, major confession, betrayal
    # Default is "routine" so older brains/legacy responses get safe routing.
    narrative_significance: str = "routine"

    # Resources consumed by the action (ammunition, consumables, etc.)
    resources_consumed: list[dict] = []  # [{"item": "Arrow", "quantity": 1}]

    # Currency spent/lost by the action (tipping, paying, dropping, etc.)
    currency_spent: dict = {}  # {"gold": 50} or empty = not set


# Cache the schema so we don't regenerate it every call
_TRIAGE_JSON_SCHEMA: Optional[dict] = None


def get_triage_schema() -> dict:
    """Get the JSON schema for triage structured output."""
    global _TRIAGE_JSON_SCHEMA
    if _TRIAGE_JSON_SCHEMA is None:
        _TRIAGE_JSON_SCHEMA = TriageSchema.model_json_schema()
    return _TRIAGE_JSON_SCHEMA


# =============================================================================
# DATA CLASSES
# =============================================================================

@dataclass
class TriageResult:
    """Result of triage decision."""
    # Action classification (routes to appropriate handler)
    action_type: str  # One of ActionType values
    reasoning: str

    # For skill_check/saving_throw actions
    needs_roll: bool = False
    roll_type: Optional[str] = None  # ability_check, saving_throw, attack_roll
    ability: Optional[str] = None
    skill: Optional[str] = None
    dc: Optional[int] = None
    advantage: bool = False
    disadvantage: bool = False
    advantage_reason: Optional[str] = None
    on_success: list[str] = field(default_factory=list)
    on_failure: list[str] = field(default_factory=list)

    # For purchase/sell actions
    item_name: Optional[str] = None
    item_cost: Optional[int] = None
    quantity: int = 1

    # For attack actions
    target_name: Optional[str] = None
    is_creature_target: bool = False  # Only trigger combat if True

    # For social/roleplay (no mechanics needed)
    narrative_direction: Optional[str] = None

    # Phase C: narrator-tier hint from the brain (routine/notable/climactic).
    # Defaults to "routine" so legacy callers and parse failures route safely.
    narrative_significance: str = "routine"

    # Resources consumed by the action
    resources_consumed: list[dict] = field(default_factory=list)

    # Currency spent/lost by the action
    currency_spent: Optional[dict] = None


_VALID_SIGNIFICANCES = frozenset({"routine", "notable", "climactic"})


def _validate_significance(value) -> str:
    """Coerce a brain-output significance value to one of the known tiers.

    Falls back to "routine" for missing, empty, or unknown values so a
    misbehaving brain can't poison the routing decision. Lowercase + strip
    so common formatting variants ("Notable ", "CLIMACTIC") all work.
    """
    if not isinstance(value, str):
        return "routine"
    cleaned = value.strip().lower()
    if cleaned in _VALID_SIGNIFICANCES:
        return cleaned
    return "routine"


@dataclass
class MechanicalResolution:
    """Result of mechanical resolution after rolling."""
    success: bool
    roll_result: Optional[DiceRoll] = None
    total: int = 0
    dc: int = 0
    margin: int = 0  # How much above/below DC
    reveals: list[str] = field(default_factory=list)
    skill: Optional[str] = None
    ability: Optional[str] = None
    roll_type: Optional[str] = None


@dataclass
class DMResponse:
    """Response from the DM orchestrator."""
    narrative: str
    mechanical_result: Optional[dict] = None
    tool_calls_made: list[dict] = field(default_factory=list)
    dice_rolls: list[DiceRoll] = field(default_factory=list)
    combat_triggered: bool = False  # True if hostility threshold triggered combat
    proposed_effects: list = field(default_factory=list)  # ProposedEffect list for immersion


class _BoundedKeySet:
    """Bounded FIFO key set used by `_applied_effects` for idempotency.

    Behaves like a `set` for the two operations we need (`__contains__`,
    `add`) but evicts the oldest entry when full so memory growth stays
    bounded. Audit #5: prior implementation was a raw `set` that grew
    forever because message IDs were unique-per-turn UUIDs.
    """

    def __init__(self, maxlen: int = 1000):
        import collections as _collections
        self._set: set[str] = set()
        self._queue: "_collections.deque[str]" = _collections.deque(maxlen=maxlen)

    def __contains__(self, key: str) -> bool:
        return key in self._set

    def add(self, key: str) -> None:
        if key in self._set:
            return
        if self._queue.maxlen is not None and len(self._queue) == self._queue.maxlen:
            evicted = self._queue[0]
            self._set.discard(evicted)
        self._queue.append(key)
        self._set.add(key)

    def __len__(self) -> int:
        return len(self._set)


# =============================================================================
# ORCHESTRATOR
# =============================================================================

class DMOrchestrator:
    """
    Coordinates the per-turn pipeline between the LLM clients and game state.

    Live flow for a player action (see process_action):
    1. Triage: the brain client classifies the action via a JSON-schema
       structured-output call (TriageSchema).
    2. Mechanics BEFORE narration: skill checks roll dice, purchases/sales/
       inventory run their handlers, attacks on creatures initiate combat.
    3. PGI validation: deterministic rule checks (validate_action) can
       hard-block the action or soften the narration directive.
    4. Narration: a tier-routed narrator client (per narrative significance)
       dramatizes the known outcome with narrator tools attached; tool calls
       become ProposedEffects via tool_calls_to_effects. The INTENTS text
       parser is only the fallback when no tool calls come back.
    5. State extraction: the brain client emits a StateDelta from the prose,
       applied to WorldState and bridged to the knowledge graph + ChromaDB.
    6. Effects: EffectValidator/EffectExecutor apply the ProposedEffects
       (idempotency-keyed), then world-state sync and the KG/vector bridges
       run for executed effects.
    7. Resources/currency are consumed only after the outcome is known.

    Prose is never the source of truth for mechanics — effects come from
    explicit tool calls (or the INTENTS fallback). Every stage is recorded
    in the TurnLogger for post-mortem observability.
    """

    def __init__(
        self,
        narrator: Optional[NarratorBrain] = None,
        adjudicator: Optional[EffectsAdjudicator] = None,
        client: Optional[OllamaClient] = None,
        narrator_client_factory: Optional[Callable[[str], Any]] = None,
    ):
        self.narrator = narrator or get_narrator()
        self.adjudicator = adjudicator or get_adjudicator()
        self.client = client or get_llm_client()
        # Tier→client resolver for narration. Injectable so a test can pin a
        # fake narrator client that survives the per-turn tier swap; the
        # production default routes by narrative significance.
        self._narrator_client_factory: Callable[[str], Any] = (
            narrator_client_factory or get_narrator_client_for
        )
        self.roller = get_roller()
        self._current_session: Optional["GameSession"] = None
        self._scene_registry: Optional[SceneEntityRegistry] = None
        self._entity_extractor: EntityExtractor = get_entity_extractor()
        self._state_extractor: StateExtractor = get_state_extractor()
        self._turn_logger: TurnLogger = get_turn_logger()

        # Effect processing (replaces mechanics extraction)
        self._effect_validator: Optional[EffectValidator] = None
        self._effect_executor: Optional[EffectExecutor] = None
        # Audit #5: was an unbounded set that grew forever because message_id
        # was uuid-per-turn (so retries never deduped and old keys never aged
        # out). Now a bounded FIFO that holds the last N keys. message_id is
        # also derived from `session_key:turn-N` so retries within the same
        # turn collapse onto the same key.
        self._applied_effects: _BoundedKeySet = _BoundedKeySet(maxlen=1000)
        self._last_executed_effects: list[ProposedEffect] = []  # For KG bridging
        self._last_effect_executions: list[dict] = []
        self._last_effect_rejections: list[dict] = []
        # Step 5's receipted writes (deterministic consumption) — counted as
        # proposed too, so the turn's effect accounting stays balanced.
        self._last_deterministic_proposed: list[dict] = []
        # Set when this turn is an acquisition whose row cannot exist yet, so
        # PGI must not validate the item against current inventory.
        self._pgi_acquisition_turn: bool = False

        # DM Scratchpad: session-scoped state for narrator continuity.
        # Inspired by coordinator scratchpad pattern from agentic orchestration.
        # Stores narrative hints, unresolved tensions, NPC moods, etc. that
        # give the narrator richer context without stuffing the prompt.
        self._scratchpad: list[dict] = []  # [{category, note, turn}]
        self._scratchpad_turn = 0
        self._scratchpad_max_entries = 20  # Rolling window
        # DF-22: which session the scratchpad belongs to. The orchestrator
        # is a process-wide singleton, so without this a new session would
        # inherit the previous session's tensions/moods.
        self._scratchpad_session_key: Optional[str] = None

        # Step 2 (REFACTOR_PLAN): the single narration path. The three
        # _narrate_* methods only build NarrationSpecs now; this strategy
        # owns prompt assembly, invocation, and the tool-followup leg.
        # Collaborators are bound as callables so the Step-0 seams
        # (_narrator_client_factory via _select_narrator_client_for_turn)
        # keep working and the per-turn narrator.client swap is respected.
        self._narration_strategy = NarrationStrategy(
            get_narrator=lambda: self.narrator,
            select_client=self._select_narrator_client_for_turn,
            get_tools=self._get_narrator_tools,
            append_tool_reminder=self._append_tool_reminder,
            extract_prose_and_effects=self._extract_prose_and_effects,
            get_on_token=lambda: getattr(self, "_on_narrative_token", None),
            get_governance=self._get_narrative_governance,
        )

    def set_session(self, session: Optional["GameSession"]) -> None:
        """Set the current session context for tool execution."""
        # DF-22: clear the narrator scratchpad when a DIFFERENT session
        # takes over — a same-session re-set keeps narrative continuity.
        if session is not None:
            new_key = session.session_key
            if (
                self._scratchpad_session_key is not None
                and new_key != self._scratchpad_session_key
            ):
                logger.info(
                    "scratchpad_cleared_on_session_change",
                    previous_session=self._scratchpad_session_key,
                    new_session=new_key,
                )
                self.scratchpad_clear()
            self._scratchpad_session_key = new_key
        self._current_session = session
        self._update_effect_processors()

    def set_scene_registry(self, registry: Optional[SceneEntityRegistry]) -> None:
        """Set the scene entity registry for entity tracking."""
        self._scene_registry = registry
        self._update_effect_processors()

    def _update_effect_processors(self) -> None:
        """Update effect validator and executor with current context."""
        self._effect_validator = EffectValidator(
            scene_registry=self._scene_registry,
            session=self._current_session,
        )
        # Executor needs inventory repo - get it async when needed
        self._effect_executor = None  # Lazy init in _process_proposed_effects

    # ==================== Post-Generation Validation ====================

    def _graph_dead_npcs(self) -> list[Any]:
        """Death facts the knowledge graph holds, or [] if it cannot answer."""
        graph = getattr(self._current_session, "knowledge_graph", None)
        reader = getattr(graph, "dead_npcs", None)
        if not callable(reader):
            return []
        try:
            return list(reader())
        except Exception as exc:  # a degraded graph must not un-guard prose
            logger.warning("graph_dead_npc_read_failed", error=str(exc))
            return []

    def _known_dead_npcs(self) -> list[Any]:
        """Return de-duplicated session, campaign and authored death facts."""
        known: list[Any] = []
        seen: set[str] = set()
        world_ids: set[str] = set()
        if self._current_session:
            world_state = getattr(self._current_session, "world_state", None)
            if world_state is not None:
                world_ids = set(world_state.npcs)
                for npc in world_state.npcs.values():
                    if not npc.alive and npc.id not in seen:
                        known.append(npc)
                        seen.add(npc.id)
            for npc in getattr(
                self._current_session, "campaign_dead_npcs", {}
            ).values():
                # A code-authorized revival in WorldState overrides the older
                # durable death fact until persistence catches up.
                if npc.id not in seen and npc.id not in world_ids:
                    known.append(npc)
                    seen.add(npc.id)
            # An AUTHORED death reaches neither roster above: a sourcebook NPC
            # gets no DB row for `_load_campaign_npcs` to find, and hydration
            # deliberately skips the dead, so `status: DEAD` in the book
            # produced an EMPTY dead-list here — governance had no fact to
            # match, `requires_buffering` stayed False, and the extractor was
            # free to mint the corpse a fresh living UUID. The graph is where
            # that fact durably lives (for authored AND played-in deaths
            # alike), and the roster layer already trusts it.
            graph_dead = self._graph_dead_npcs()
            if graph_dead:
                from ..game.world_state import NPCState
            for entity in graph_dead:
                node_id = str(getattr(entity, "node_id", "") or "")
                if not node_id or node_id in seen or node_id in world_ids:
                    continue
                known.append(NPCState(
                    id=node_id,
                    name=str(getattr(entity, "name", "") or ""),
                    aliases=list(getattr(entity, "aliases", []) or []),
                    description=str(
                        (getattr(entity, "properties", {}) or {}).get(
                            "description", ""
                        )
                    ),
                    alive=False,
                ))
                seen.add(node_id)
        return known

    def _get_narrative_governance(self) -> NarrativeGovernance:
        """Build rules from authoritative state for this exact turn."""
        return NarrativeGovernance(self._known_dead_npcs())

    def _validate_npc_references(self, narrative: str) -> str:
        """Deterministic check for NPC references in narrator output.

        Two-layer validation:
        1. SceneEntityRegistry: logs NPC mentions for tracking
        2. WorldState: flags NPCs mentioned that aren't at the current location

        Does NOT modify the narrative — logs issues for debugging.
        """
        if not self._scene_registry:
            return narrative

        from ..models.npc import EntityType
        npcs = self._scene_registry.get_by_type(EntityType.NPC)
        if not npcs:
            return narrative

        narrative_lower = narrative.lower()

        # Layer 1: SceneEntityRegistry check
        npc_names = {e.name.lower(): e for e in npcs}
        for name_lower, entity in npc_names.items():
            if len(name_lower) < 3:
                continue
            if name_lower in narrative_lower:
                logger.debug(
                    "narrator_npc_reference",
                    npc=entity.name,
                    disposition=entity.disposition.value if entity.disposition else "unknown",
                )

        # Layer 2: WorldState location validation (referenced_entities check).
        # Iterate values (the dict is keyed by id now); check the canonical
        # name AND any aliases so paraphrases get caught too.
        world_state = getattr(self._current_session, 'world_state', None) if self._current_session else None
        if world_state and world_state.current_location:
            for npc_state in world_state.npcs.values():
                # Build the set of names to check (canonical + aliases)
                candidates = [npc_state.name] + list(npc_state.aliases)
                matched_via = next(
                    (c for c in candidates if c and len(c) >= 3 and c.lower() in narrative_lower),
                    None,
                )
                if matched_via:
                    # NPC mentioned — check if they're at the right location
                    if (
                        npc_state.location
                        and npc_state.location.lower() != world_state.current_location.lower()
                        and npc_state.alive
                    ):
                        logger.warning(
                            "npc_location_mismatch",
                            npc=npc_state.name,
                            npc_location=npc_state.location,
                            party_location=world_state.current_location,
                        )

        return narrative

    # ==================== DM Scratchpad ====================

    def scratchpad_note(self, category: str, note: str) -> None:
        """Add a narrative hint to the DM scratchpad.

        Categories: tension, npc_mood, foreshadow, unresolved, atmosphere, plot
        """
        self._scratchpad_turn += 1
        self._scratchpad.append({
            "category": category,
            "note": note,
            "turn": self._scratchpad_turn,
        })
        # Trim old entries beyond rolling window
        if len(self._scratchpad) > self._scratchpad_max_entries:
            self._scratchpad = self._scratchpad[-self._scratchpad_max_entries:]

    def scratchpad_context(self) -> str:
        """Build scratchpad context string for narrator injection."""
        if not self._scratchpad:
            return ""
        lines = ["<dm_scratchpad>"]
        for entry in self._scratchpad:
            lines.append(f"[{entry['category']}] {entry['note']}")
        lines.append("</dm_scratchpad>")
        return "\n".join(lines)

    def scratchpad_clear(self) -> None:
        """Clear the scratchpad (e.g., on session end)."""
        self._scratchpad.clear()
        self._scratchpad_turn = 0

    def _update_scratchpad(
        self,
        triage: "TriageResult",
        resolution: Optional["MechanicalResolution"],
        effects: list["ProposedEffect"],
        combat_triggered: bool,
        player_name: str,
    ) -> None:
        """Auto-populate scratchpad from turn results for narrator continuity."""
        # Track failed checks — good narrative hooks
        if resolution and not resolution.success and triage.skill:
            self.scratchpad_note(
                "unresolved",
                f"{player_name} failed {triage.skill} check (DC {triage.dc}). "
                "Something was missed or went wrong.",
            )

        # Track combat triggers
        if combat_triggered:
            self.scratchpad_note("tension", "Combat just broke out. Atmosphere is hostile.")

        # Track NPC interactions from effects
        for effect in effects:
            if effect.effect_type == EffectType.ADD_NPC:
                name = effect.npc_name or "someone"
                disp = effect.npc_disposition or "neutral"
                self.scratchpad_note("npc_mood", f"{name} appeared ({disp} disposition).")

        # Track significant damage
        for effect in effects:
            if effect.effect_type == EffectType.APPLY_DAMAGE:
                amount = effect.amount or 0
                if amount >= 10:
                    target = effect.target or "someone"
                    self.scratchpad_note("tension", f"{target} took {amount} damage — situation is dangerous.")

    async def process_action(
        self,
        action: str,
        player_name: str,
        context: BrainContext,
        on_mechanics_ready: Optional[Callable] = None,
        on_narrative_token: Optional[Callable] = None,
    ) -> DMResponse:
        """
        Process a player action through the turn pipeline.

        Flow:
        1. Triage: classify the action (structured-output call on the brain client)
        2. Route by action_type and execute mechanics BEFORE narration
           (dice rolls, purchase/sell/inventory, combat initiation), gated
           by deterministic PGI validation
        3. Narrator dramatizes the known outcome; its tool calls are
           converted to ProposedEffects
        4. Extract + apply StateDelta, validate/execute effects, sync world
           state and the knowledge graph, then consume resources

        Args:
            on_mechanics_ready: Async callback(mechanical_result, dice_rolls)
                fired before the narration LLM call so mechanics reach
                Discord immediately.
            on_narrative_token: Async callback(str) for streaming narrator tokens
                to Discord. Enables progressive message edits for better UX.
        """
        context.player_action = action
        context.player_name = player_name

        # Effect receipts are turn-local even when this turn returns early or
        # continuity fails closed with no proposals. Stale receipts overstated
        # execution telemetry and could be bridged into the graph again.
        self._last_executed_effects = []
        self._last_effect_executions = []
        self._last_effect_rejections = []
        self._last_deterministic_proposed = []
        self._pgi_acquisition_turn = False

        # ── Turn Logger: start recording ──
        session_id = context.session_id or context.campaign_id or "unknown"
        world_state = getattr(self._current_session, 'world_state', None) if self._current_session else None
        turn_num = world_state.turn if world_state else self._scratchpad_turn
        _turn_record = self._turn_logger.new_turn(session_id, turn_num)
        _turn_record.set("action", action)
        _turn_record.set("player", player_name)
        _turn_record.set("phase", world_state.phase if world_state else "unknown")
        # The turn log's world-state bookends are read as evidence about the
        # PROMPT, so they resolve action entities through the same seam
        # session context assembly uses. On the one-action-at-a-time path
        # that makes ``before`` identical to what the narrator was handed:
        # same action string, same graph, and nothing mutates world state in
        # between. It is a reconstruction, not a capture — two players
        # narrating concurrently would each have built context before the
        # other's turn landed, and the reconstruction would show the later
        # state. ``after`` deliberately reuses the PRE-turn entity set so the
        # bookends differ only by what the turn changed.
        from ..game.knowledge.matcher import action_entity_names

        _ws_action_entities = action_entity_names(
            getattr(self._current_session, 'knowledge_graph', None)
            if self._current_session else None,
            action,
        )
        ws_before = world_state.to_yaml(
            action_text=action, action_entities=_ws_action_entities,
        ) if world_state else ""
        # Where the party stood before this turn — scene hydration only runs
        # on an actual arrival, not on every turn spent in the same room.
        ws_location_before = world_state.current_location if world_state else ""

        # Store streaming callback for narrator methods to use
        self._on_narrative_token = on_narrative_token

        # Inject scratchpad into narrator context for continuity
        scratchpad_ctx = self.scratchpad_context()
        if scratchpad_ctx:
            context.session_summary = (
                (context.session_summary or "") + "\n\n" + scratchpad_ctx
            ).strip()

        logger.info(
            "processing_action",
            action=action[:50],
            player=player_name,
        )

        # Step 0: Removed pre-triage keyword-based attack check.
        # It used fragile keyword matching + disposition filters that skipped friendly
        # entities (e.g., player says "I attack the sailor" but sailor is friendly,
        # so it grabbed the most hostile entity instead). The LLM triage below
        # correctly classifies attacks with target_name and is_creature_target.

        # Step 1: Triage - classify the action
        _turn_record.start_stage("triage")
        triage = await self._triage_action(action, player_name, context)
        _turn_record.end_stage("triage")
        _turn_record.record_triage(
            action_type=triage.action_type,
            needs_roll=triage.needs_roll,
            skill=triage.skill or "",
            dc=triage.dc or 0,
            parse_warnings=getattr(triage, "_parse_warnings", None),
            currency_spent=triage.currency_spent,
            resources_consumed=triage.resources_consumed,
        )

        logger.info(
            "triage_decision",
            action_type=triage.action_type,
            needs_roll=triage.needs_roll,
            skill=triage.skill,
            dc=triage.dc,
            target_name=triage.target_name,
            is_creature_target=triage.is_creature_target,
            reasoning=triage.reasoning[:100] if triage.reasoning else "",
        )

        # Step 2: Route to appropriate handler based on action_type
        resolution = None
        mechanical_result = None
        dice_rolls = []
        tool_calls = []

        if triage.action_type == "purchase":
            # Execute purchase through Rules Brain
            mechanical_result = await self._handle_purchase(triage, player_name, context)
            tool_calls = mechanical_result.get("tool_calls", []) if mechanical_result else []

        elif triage.action_type == "sell":
            # Execute sale through Rules Brain
            mechanical_result = await self._handle_sell(triage, player_name, context)
            tool_calls = mechanical_result.get("tool_calls", []) if mechanical_result else []

        elif triage.action_type == "skill_check" and triage.needs_roll:
            # Execute skill check with dice roll
            resolution = await self._resolve_mechanics(triage, player_name, context)
            if resolution.roll_result:
                dice_rolls.append(resolution.roll_result)
            mechanical_result = {
                "action_type": "skill_check",
                "success": resolution.success,
                "skill": resolution.skill,
                "ability": resolution.ability,
                "roll": resolution.total,
                "dc": resolution.dc,
                "margin": resolution.margin,
            }

        elif triage.action_type == "inventory":
            # Inventory actions - pick up, drop, use, equip
            mechanical_result = await self._handle_inventory(triage, player_name, context, action)
            tool_calls = mechanical_result.get("tool_calls", []) if mechanical_result else []

        elif triage.action_type == "attack":
            # Check if this is a creature target (triggers combat) or object (just narrate)
            if not triage.is_creature_target:
                # Attacking an object/non-creature - just narrate it
                logger.info(
                    "attack_non_creature",
                    target=triage.target_name,
                    reasoning=triage.reasoning[:50] if triage.reasoning else "",
                )
                # Let narrator handle it with guidance
                if not triage.narrative_direction:
                    triage.narrative_direction = (
                        f"The player attacks {triage.target_name or 'the target'}. "
                        "Describe what happens - this doesn't trigger combat."
                    )
                # Fall through to narrator below
            elif not context.in_combat:
                # Attacking a creature — start combat, but do NOT consume resources.
                # No actual attack happens here; the real attack comes through the
                # combat UI. Resource consumption (arrows, etc.) happens there.
                combat_started = await self._initiate_combat_from_attack(
                    triage.target_name, player_name
                )
                if combat_started:
                    return DMResponse(
                        narrative=f"*You attack the {triage.target_name or 'enemy'}! Combat begins!*",
                        combat_triggered=True,
                    )
                else:
                    # Target not in registry — but the narrator described them previously.
                    # Register them now from context and start combat.
                    logger.info(
                        "attack_target_not_in_scene_creating",
                        target=triage.target_name,
                    )

                    # Create the target entity on the fly with best-guess SRD match
                    target_name = triage.target_name or "enemy"
                    monster_index = guess_monster_index(target_name)

                    from ..models.npc import SceneEntity, EntityType, Disposition
                    entity = SceneEntity(
                        name=target_name.title(),
                        entity_type=EntityType.CREATURE,
                        description=f"Hostile creature from the scene",
                        disposition=Disposition.HOSTILE,
                        hostility_score=90,
                        monster_index=monster_index,
                    )

                    if self._scene_registry:
                        self._scene_registry.register_entity(entity)

                    # Now try combat again
                    combat_started = await self._initiate_combat_from_attack(
                        target_name, player_name
                    )
                    if combat_started:
                        return DMResponse(
                            narrative=f"*You attack the {target_name}! Combat begins!*",
                            combat_triggered=True,
                        )

                    # Still failed — absolute fallback, create combat with just this entity
                    hostile_entities = [entity]
                    combat_started = start_encounter(
                        self._current_session, hostile_entities, player_initiated=True
                    )
                    if combat_started:
                        return DMResponse(
                            narrative=f"*You attack the {target_name}! Combat begins!*",
                            combat_triggered=True,
                        )
            # TODO: Integrate with combat coordinator for in-combat attacks

        elif triage.action_type == "cast_spell":
            # Spell casting — slot validation handled by PGI layer below.
            # Full spell resolution (attack rolls, saves, AoE) is handled by the
            # combat coordinator when in combat; this path handles exploration/social casting.
            resolved = self._resolve_character_by_name(player_name)
            if resolved:
                char_id, character = resolved
                # Build spell context for the narrator (enrichment only, not rule enforcement)
                slot_info = []
                for level in range(1, 10):
                    current, max_slots = character.spell_slots.get_slots(level)
                    if max_slots > 0:
                        slot_info.append(f"L{level}: {current}/{max_slots}")
                spell_context = ", ".join(slot_info) if slot_info else "no spell slots"

                if not triage.narrative_direction:
                    triage.narrative_direction = (
                        f"The player casts a spell. "
                        f"Available spell slots: {spell_context}. "
                        f"Narrate the spell's dramatic effect."
                    )

        # ── PGI Validation: deterministic rule checks before narrator fires ──
        pgi_result: Optional[ValidationResult] = None
        pgi_resolved = self._resolve_character_by_name(player_name)
        if pgi_resolved:
            _pgi_char_id, pgi_character = pgi_resolved

            # Pre-fetch inventory/currency only when the action type needs them
            pgi_items = None
            pgi_currency = None
            needs_inventory = (
                triage.action_type in ("inventory",)
                or triage.resources_consumed
            )
            needs_currency = (
                triage.action_type == "purchase"
                and triage.item_cost is not None
            )

            if needs_inventory or needs_currency:
                try:
                    inv_repo = await get_inventory_repo()
                    if needs_inventory:
                        pgi_items = await inv_repo.get_all_items(_pgi_char_id)
                    if needs_currency:
                        pgi_currency = await inv_repo.get_currency(_pgi_char_id)
                except Exception as e:
                    logger.warning("pgi_prefetch_failed", error=str(e), exc_info=True)

            pgi_result = await validate_action(
                action_type=triage.action_type,
                character=pgi_character,
                action_text=action,
                items=pgi_items,
                currency=pgi_currency,
                resources_consumed=triage.resources_consumed or None,
                # Acquisitions can't require prior possession: transfers FROM
                # an NPC and pickups both grant via authoritative effects that
                # execute after this gate, so the row legitimately doesn't
                # exist yet. A DEFERRED acquisition has no mechanical_result
                # to read the operation from, hence the flag.
                item_name=(
                    None
                    if self._pgi_acquisition_turn
                    or (
                        mechanical_result
                        and mechanical_result.get("operation")
                        in ("transfer_from_npc", "pickup")
                    )
                    else triage.item_name
                ),
                cost_gold=float(triage.item_cost) if triage.item_cost else 0,
                scene_item_names=(
                    [entity.name for entity in self._scene_registry.get_all()]
                    if self._scene_registry
                    else []
                ) + (
                    list(self._current_session.world_state.scene_items)
                    if self._current_session
                    and self._current_session.world_state
                    else []
                ),
            )

            # Record PGI results for observability
            _turn_record.set("pgi", {
                "passed": pgi_result.passed,
                "failures": [
                    {"code": f.code, "severity": f.severity.value, "priority": f.priority}
                    for f in pgi_result.failures
                ],
            })

            if pgi_result.has_hard_fail:
                # Hard fail: return immediately — no narrator, no resources consumed
                logger.info(
                    "pgi_hard_fail_intercept",
                    player=player_name,
                    action_type=triage.action_type,
                    codes=[f.code for f in pgi_result.hard_failures],
                )
                _turn_record.set("pgi_blocked", True)
                self._turn_logger.flush(_turn_record)
                return DMResponse(
                    narrative=pgi_result.player_feedback(),
                    mechanical_result={"pgi_blocked": True, "failures": [
                        {"code": f.code, "message": f.message} for f in pgi_result.hard_failures
                    ]},
                )

            if pgi_result.has_soft_fail:
                # Soft fail: narrator gets modified payload with game-state-unchanged directive
                soft_context = " ".join(f.message for f in pgi_result.soft_failures)
                triage.narrative_direction = (
                    f"[GAME STATE UNCHANGED] {soft_context}\n"
                    f"{triage.narrative_direction or ''}"
                ).strip()

        # Step 2.5: Send mechanics to Discord immediately (before narration LLM call)
        if on_mechanics_ready and (mechanical_result or dice_rolls):
            try:
                await on_mechanics_ready(mechanical_result, dice_rolls)
            except Exception as e:
                logger.warning("on_mechanics_ready_callback_failed", error=str(e), exc_info=True)

        # Step 2.75: Knowledge graph context for narrator
        # Multi-tier entity resolution: scene seeds + substring match + vector fallback
        kg = getattr(self._current_session, 'knowledge_graph', None) if self._current_session else None
        _kg_seed_ids: list[str] = []
        _kg_vector_matches = 0
        _kg_narrative_recalled = 0
        # Telemetry buckets — filled out below for the assertion API
        _kg_text_match_seeds: list[str] = []
        _kg_scene_seeds: list[str] = []
        _kg_vector_match_seeds: list[str] = []
        _kg_recalled_chunks: list[dict] = []
        if kg and kg.node_count() > 0:
            try:
                from ..game.knowledge.matcher import EntityNameMatcher
                from ..memory import get_vector_store

                matcher = EntityNameMatcher(kg)
                world_state_pre = getattr(self._current_session, 'world_state', None) if self._current_session else None

                # Tier 1: Substring match on player text
                text_seeds = matcher.match(action)
                _kg_text_match_seeds = list(text_seeds)

                # Tier 2: Always include current scene (location + present NPCs)
                scene_seeds = matcher.scene_seeds(world_state_pre) if world_state_pre else []
                _kg_scene_seeds = list(scene_seeds)

                # Tier 3: Vector fallback if player text didn't match any entities
                if not text_seeds and kg.node_count() > 0:
                    vs = get_vector_store()
                    # Chroma embeds synchronously; off the event loop so a
                    # long graph history cannot stall the Discord heartbeat.
                    text_seeds = await asyncio.to_thread(
                        matcher.vector_match, action, context.campaign_id, vs
                    )
                    _kg_vector_matches = len(text_seeds)
                    _kg_vector_match_seeds = list(text_seeds)

                # Merge and deduplicate (text matches first for priority)
                seen = set()
                _kg_seed_ids = []
                for sid in text_seeds + scene_seeds:
                    if sid not in seen:
                        _kg_seed_ids.append(sid)
                        seen.add(sid)

                if _kg_seed_ids:
                    # Graph context: structured relationships, capped —
                    # to_context_yaml limits entity count, not rendered size.
                    # A 2.0-radius neighborhood crossed location-to-location
                    # edges and then pulled in NPCs at the adjacent location.
                    # That leaked off-screen people into otherwise irrelevant
                    # prompts. A 1.0 weighted radius still includes directly
                    # related co-located entities (LOCATED_AT = 0.3) while
                    # requiring explicit text/vector retrieval for NPCs beyond
                    # the immediate graph neighborhood. Vector-fallback seeds
                    # are speculative: they appear themselves but do not
                    # expand a neighborhood (a similar-looking item otherwise
                    # drags its owner NPC into an unrelated scene).
                    context.kg_context_yaml = _cap_kg_context_yaml(
                        kg.to_context_yaml(
                            _kg_seed_ids,
                            radius=1.0,
                            no_expand_ids=set(_kg_vector_match_seeds),
                        )
                    )

                    # Narrative recall is triggered only by entities the
                    # player's action named EXACTLY. Ambient scene membership
                    # and speculative vector matches supply current graph
                    # context, but must not reactivate old episodes — a
                    # topically similar chunk can carry an unrelated NPC's
                    # name into the prompt (validation 20260722_232549,
                    # turn 21).
                    try:
                        vs = get_vector_store()
                        recall_seed_ids = list(_kg_text_match_seeds)
                        seed_names = [
                            kg.get_entity(s).name
                            for s in recall_seed_ids
                            if kg.get_entity(s)
                        ]
                        if seed_names:
                            ws_turn = world_state_pre.turn if world_state_pre else 0
                            past_chunks = await asyncio.to_thread(
                                vs.recall_narratives_for_entities,
                                campaign_id=context.campaign_id,
                                entity_ids=recall_seed_ids,
                                query_text=" ".join(seed_names),
                                max_results=2,
                                current_turn=ws_turn,
                            )
                            if past_chunks:
                                context.narrative_memory = "\n---\n".join(
                                    c["content"][:300] for c in past_chunks
                                )
                                _kg_narrative_recalled = len(past_chunks)
                                _kg_recalled_chunks = list(past_chunks)
                    except Exception as e:
                        logger.warning("kg_narrative_recall_failed", error=str(e), exc_info=True)

                    logger.debug(
                        "kg_context_injected",
                        seed_count=len(_kg_seed_ids),
                        text_matches=len(text_seeds),
                        scene_seeds=len(scene_seeds),
                        vector_matches=_kg_vector_matches,
                        narrative_recalled=_kg_narrative_recalled,
                    )
            except Exception as e:
                logger.warning("kg_context_failed", error=str(e), exc_info=True)

        # Step 3: Narrator dramatizes the outcome (with mechanical result as context)
        proposed_effects: list[ProposedEffect] = []

        _turn_record.start_stage("narrate")
        if resolution:
            # Roll happened - narrate the mechanical outcome
            narrative, proposed_effects = await self._narrate_outcome(action, player_name, context, triage, resolution)
        elif mechanical_result:
            # Mechanical action (purchase, etc.) - narrate with the result
            narrative, proposed_effects = await self._narrate_mechanical_result(action, player_name, context, triage, mechanical_result)
        else:
            # No mechanics - just respond to the player's action naturally
            narrative, proposed_effects = await self._narrate_action(action, player_name, context, triage)

        # Some mechanical handlers own deterministic state transitions that
        # narration must dramatize but must not be trusted to reconstruct.
        # Keep narrator references/scene additions, and dedupe by claimed
        # field/entity so the authoritative write is the only writer for what
        # it claims — without discarding an unrelated narrator mutation that
        # happens to share an effect type on the same turn.
        authoritative_effects = list(
            (mechanical_result or {}).get("authoritative_effects") or []
        )
        # A turn that failed closed narrates "no world-state change is
        # committed" and drops the narrator's effects. The deterministic
        # claim has to honor that same sentence, or the prose denies a write
        # the receipts record. The player's action simply didn't land.
        #
        # Deliberately NOT the continuity door, narration.py's other
        # effect-clearing path: its fallback prose ("The consequences of your
        # action settle across the scene", or a dead NPC staying dead) never
        # denies the player's own action. Voiding a validated purchase there
        # would fail the turn for an unrelated hallucination.
        if authoritative_effects and (
            getattr(self._narration_strategy, "last_diagnostics", {}) or {}
        ).get("resolved_outcome_failed_closed"):
            logger.info(
                "authoritative_effects_dropped_failed_closed",
                count=len(authoritative_effects),
                action_type=(mechanical_result or {}).get("action_type"),
            )
            authoritative_effects = []
        if authoritative_effects:
            proposed_effects = _scrub_claimed_mutations(
                proposed_effects, authoritative_effects
            ) + authoritative_effects
        # Cache the prose for the dedup judge in _process_proposed_effects.
        # The judge reads this to decide whether an add_npc looks like an
        # entity already in the registry.
        self._last_narrator_prose = narrative or ""
        _turn_record.end_stage("narrate")
        _narration_diagnostics = dict(
            getattr(self._narration_strategy, "last_diagnostics", {})
        )
        _turn_record.record_narrator_response(
            narrative or "",
            format_type=_get_client_provider(self.narrator.client),
            reprompted=bool(
                _narration_diagnostics.get("continuity_repair_attempted")
                or _narration_diagnostics.get("tool_followup_attempted")
            ),
        )
        _turn_record.set("narration_diagnostics", _narration_diagnostics)

        # Phase A telemetry: record which narrator tier handled this turn.
        # Source-of-truth is _last_narrator_routing populated by
        # _get_narrator_client_for_turn during this exact turn.
        _routing = getattr(self, "_last_narrator_routing", None)
        if _routing:
            _turn_record.record_narrator_routing(**_routing)

        # Step 3.5: Post-generation NPC validation (deterministic, no LLM)
        # Checks if narrator used NPC names in wrong locations
        if narrative and self._scene_registry:
            narrative = self._validate_npc_references(narrative)

        world_state = (
            getattr(self._current_session, 'world_state', None)
            if self._current_session else None
        )
        proposed_effects = _anchor_returning_add_npc_effects(
            proposed_effects,
            kg,
        )
        proposed_effects = _canonicalize_npc_effect_ids(
            proposed_effects,
            world_state,
            kg,
        )
        proposed_effects = _resolve_invented_scene_ids(
            proposed_effects,
            world_state,
            kg,
        )

        # Step 3.5b: Collect ref_entity IDs from narrator intents.
        # These are entities the narrator explicitly tagged — used to
        # prevent the state extractor from creating duplicates, and as
        # additional KG seeds for narrative chunk tagging.
        _narrator_ref_ids: list[str] = []
        _narrator_refs: list[tuple[str, str]] = []
        _extractor_accounted_entities: list[str] = []
        if proposed_effects:
            for e in proposed_effects:
                if e.effect_type == EffectType.REF_ENTITY and e.ref_entity_id:
                    _narrator_ref_ids.append(e.ref_entity_id)
                    alias = str(e.ref_alias_used or "").strip()
                    _narrator_refs.append((e.ref_entity_id, alias))
                    _extractor_accounted_entities.append(
                        f"{e.ref_entity_id} (prose alias: {alias})"
                        if alias else e.ref_entity_id
                    )
                elif e.effect_type == EffectType.ADD_NPC and e.npc_name:
                    # Also tell extractor about narrator-declared new NPCs
                    # so it doesn't create duplicates via new_npcs
                    _narrator_ref_ids.append(e.npc_name)
                    _extractor_accounted_entities.append(e.npc_name)
                    if e.npc_canonical_id:
                        _narrator_ref_ids.append(e.npc_canonical_id)
                        _narrator_refs.append(
                            (e.npc_canonical_id, e.npc_name)
                        )

        # Step 3.6: World state extraction — extract StateDelta and apply
        # Uses cheap brain model to identify what changed in the world
        delta = None
        self._last_state_rejections = []
        # Capture pre-delta location for knowledge graph connectivity
        pre_delta_location = world_state.current_location if world_state else ""
        if narrative and world_state:
            _turn_record.start_stage("state_extract")
            delta = await self._extract_and_apply_state_delta(
                narrative, world_state, context,
                referenced_entity_ids=_extractor_accounted_entities,
                canonical_entity_ids=list(dict.fromkeys(
                    _kg_seed_ids + _narrator_ref_ids
                )),
                narrator_refs=_narrator_refs,
            )
            _turn_record.end_stage("state_extract")

            # Step 3.6a: Extractor-coordinated tool recovery. When the applied
            # delta contains a high-confidence mutation with no matching
            # narrator tool, request exactly the missing calls so canonical
            # grounding (location names, identity bindings) still flows
            # through Step 4. The delta already kept world state correct;
            # this leg only restores narrator-authored grounding.
            if delta:
                try:
                    from .state_followup import uncovered_state_signals
                    _kg_npc_labels: list[str] = []
                    if kg is not None:
                        for _node in (kg._entities or {}).values():
                            if getattr(
                                getattr(_node, "entity_type", None), "value", ""
                            ) != "npc":
                                continue
                            _kg_npc_labels.append(str(_node.name or ""))
                            _kg_npc_labels.extend(
                                str(a) for a in (_node.aliases or [])
                            )
                    _followup_signals = uncovered_state_signals(
                        delta,
                        before_location=pre_delta_location,
                        narrative=narrative,
                        proposed_effects=proposed_effects,
                        world_state=world_state,
                        player_name=context.player_name,
                        known_entity_labels=_kg_npc_labels,
                    )
                    if _followup_signals:
                        _turn_record.start_stage("state_followup")
                        _recovered = await (
                            self._narration_strategy.targeted_state_followup(
                                narrative, _followup_signals
                            )
                        )
                        _turn_record.end_stage("state_followup")
                        proposed_effects.extend(_recovered)
                        _turn_record.record_state_followup(
                            signals=[
                                s.instruction for s in _followup_signals
                            ],
                            recovered=len(_recovered),
                            structural_errors=list(
                                self._narration_strategy.last_diagnostics.get(
                                    "state_followup_structural_errors", []
                                )
                            ),
                        )
                        logger.info(
                            "state_followup_recovery",
                            signals=len(_followup_signals),
                            recovered=len(_recovered),
                        )
                except Exception as e:
                    logger.warning(
                        "state_followup_failed", error=str(e), exc_info=True
                    )

            proposed_effects = _reanchor_same_turn_generic_npc_effects(
                proposed_effects,
                delta,
                world_state,
            )

            # Record state delta to turn log (including parse warnings)
            if delta:
                _turn_record.record_state_delta(
                    delta_dict=delta.model_dump(exclude_none=True, exclude_defaults=True),
                    rejections=list(self._last_state_rejections),
                    parse_warnings=getattr(delta, "_parse_warnings", None),
                )

        # Step 3.6b: Bridge state delta to knowledge graph
        _kg_ops_applied = 0
        _kg_ops_rejected = 0
        graph_ops = []
        if kg and delta and world_state:
            try:
                from ..game.knowledge.bridge import DeltaBridge
                # The resolver collapses a narrator paraphrase ("the Copper
                # Finch") onto the place canon already knows, so residency
                # edges land on the node hydration will read.
                bridge = DeltaBridge(
                    context.campaign_id,
                    location_resolver=kg.resolve_location_node,
                )
                existing_ids = set(kg._entities.keys()) if kg._entities else set()
                graph_ops = bridge.convert(
                    delta, world_state,
                    existing_node_ids=existing_ids,
                    previous_location=pre_delta_location,
                )
                if graph_ops:
                    rejections = await kg.apply_operations(graph_ops)
                    _kg_ops_applied = len(graph_ops) - len(rejections)
                    _kg_ops_rejected = len(rejections)
                    if rejections:
                        logger.debug("kg_bridge_rejections", rejections=rejections)
            except Exception as e:
                logger.warning("kg_bridge_failed", error=str(e), exc_info=True)

        # Step 3.6c: Sync entity descriptions to ChromaDB for vector matching
        if kg and graph_ops:
            try:
                from ..game.knowledge.models import AddNode, UpdateNode
                from ..memory import get_vector_store
                vs = get_vector_store()
                for op in graph_ops:
                    if isinstance(op, AddNode):
                        entity = kg.get_entity(op.entity.node_id)
                    elif isinstance(op, UpdateNode):
                        entity = kg.get_entity(op.node_id)
                    else:
                        continue
                    if entity and entity.properties.get("description"):
                        await asyncio.to_thread(
                            vs.add_entity_description,
                            campaign_id=context.campaign_id,
                            node_id=entity.node_id,
                            entity_type=entity.entity_type.value,
                            name=entity.name,
                            description=entity.properties["description"],
                            aliases=entity.aliases,
                        )
            except Exception as e:
                logger.warning("kg_entity_sync_failed", error=str(e), exc_info=True)

        # Step 3.6d: Store tagged narrative chunk for future recall
        _narrative_chunk_stored = False
        if narrative and kg:
            try:
                from ..game.knowledge.matcher import EntityNameMatcher
                from ..memory import get_vector_store
                matcher = EntityNameMatcher(kg)

                # Narrative memory provenance must reflect this output, not
                # the context that happened to be retrieved before generation.
                narrator_entity_ids = matcher.match(narrative)
                all_tags = _explicit_narrative_tags(
                    narrator_entity_ids,
                    _narrator_ref_ids,
                )

                if all_tags:
                    vs = get_vector_store()
                    await asyncio.to_thread(
                        vs.add_narrative_chunk,
                        campaign_id=context.campaign_id,
                        chunk_id=str(uuid.uuid4()),
                        narrative_text=narrative,
                        entity_ids=all_tags,
                        turn=world_state.turn if world_state else 0,
                        location=world_state.current_location if world_state else "",
                    )
                    _narrative_chunk_stored = True
            except Exception as e:
                logger.warning("kg_narrative_store_failed", error=str(e), exc_info=True)

        # Record knowledge graph state in turn log
        if kg:
            _turn_record.record_knowledge_graph(
                nodes_total=kg.node_count(),
                edges_total=kg.edge_count(),
                seed_entities=_kg_seed_ids,
                context_injected=bool(context.kg_context_yaml),
                ops_applied=_kg_ops_applied,
                ops_rejected=_kg_ops_rejected,
                narrative_chunk_stored=_narrative_chunk_stored,
                vector_matches=_kg_vector_matches,
                narrative_chunks_recalled=_kg_narrative_recalled,
                # Phase A telemetry: actual content surfaced
                context_yaml=context.kg_context_yaml,
                narrative_chunks=_kg_recalled_chunks,
                text_match_seeds=_kg_text_match_seeds,
                scene_seeds=_kg_scene_seeds,
                vector_match_seeds=_kg_vector_match_seeds,
                catalog_entities=[
                    {
                        "id": entity.node_id,
                        "name": entity.name,
                        "type": entity.entity_type.value,
                        "aliases": list(entity.aliases),
                    }
                    for entity in kg._entities.values()
                ],
            )

        # Step 4: Process narrator output - entity extraction + proposed effects
        # Skip entity extraction during combat or for simple actions (saves API calls)
        # Also skip if world state handled extraction (avoids duplicate NPC registration)
        combat_triggered = False
        skip_extraction = (
            context.in_combat
            or triage.action_type in ("skill_check", "saving_throw")
            or world_state is not None  # WorldState extraction supersedes entity extraction
        )
        if narrative:
            # Audit #5: derive message_id from session+turn so retries within
            # the same turn collapse onto the same idempotency key. Two
            # different sessions in the same campaign won't collide because
            # session_key is unique per session. Falls back to a uuid if
            # world_state isn't available (rare path).
            if world_state and self._current_session:
                message_id = f"{self._current_session.session_key}:turn-{world_state.turn}"
            else:
                message_id = str(uuid.uuid4())
            combat_triggered = await self._process_narrator_output(
                narrative, context, proposed_effects, message_id,
                skip_entity_extraction=skip_extraction,
            )

        # Step 4b: Bridge executed tool effects into KG + ChromaDB
        # Tool effects (add_npc, spawn_object, ref_entity) already landed in
        # WorldState via _sync_effect_to_world_state, but the KG and vector
        # store only received data from the StateDelta path (Step 3.6b/c).
        # This step closes the gap so every tool-created entity gets a KG node,
        # LOCATED_AT edges, and a ChromaDB vector entry.
        _effect_kg_ops = 0
        if kg and world_state and self._last_executed_effects:
            try:
                from ..game.knowledge.bridge import DeltaBridge
                from ..game.knowledge.models import AddNode as _AddNode, UpdateNode as _UpdateNode
                from ..memory import get_vector_store

                bridge = DeltaBridge(
                    context.campaign_id,
                    location_resolver=kg.resolve_location_node,
                )
                existing_ids = set(kg._entities.keys()) if kg._entities else set()
                effect_ops, promotions = bridge.convert_effects(
                    self._last_executed_effects, world_state,
                    existing_node_ids=existing_ids,
                )

                # Apply graph operations
                if effect_ops:
                    rejections = await kg.apply_operations(effect_ops)
                    _effect_kg_ops = len(effect_ops) - len(rejections)
                    if rejections:
                        logger.debug("effect_kg_rejections", rejections=rejections)

                # Name promotions from ref_entity aliases
                for promo in promotions:
                    entity = kg.get_entity(promo.node_id)
                    if entity and entity.properties.get("named") == "false":
                        promoted = await kg.promote_entity_name(
                            promo.node_id, promo.new_name,
                        )
                        if promoted:
                            logger.info(
                                "entity_name_promoted",
                                node_id=promo.node_id,
                                new_name=promo.new_name,
                            )

                # Sync new/updated entities to ChromaDB
                if effect_ops:
                    vs = get_vector_store()
                    for op in effect_ops:
                        if isinstance(op, _AddNode):
                            entity = kg.get_entity(op.entity.node_id)
                        elif isinstance(op, _UpdateNode):
                            entity = kg.get_entity(op.node_id)
                        else:
                            continue
                        if entity and entity.properties.get("description"):
                            await asyncio.to_thread(
                                vs.add_entity_description,
                                campaign_id=context.campaign_id,
                                node_id=entity.node_id,
                                entity_type=entity.entity_type.value,
                                name=entity.name,
                                description=entity.properties["description"],
                                aliases=entity.aliases,
                            )

                # Also re-index promoted entities (name changed)
                for promo in promotions:
                    entity = kg.get_entity(promo.node_id)
                    if entity and entity.properties.get("description"):
                        vs = get_vector_store()
                        await asyncio.to_thread(
                            vs.add_entity_description,
                            campaign_id=context.campaign_id,
                            node_id=entity.node_id,
                            entity_type=entity.entity_type.value,
                            name=entity.name,
                            description=entity.properties["description"],
                            aliases=entity.aliases,
                        )
            except Exception as e:
                logger.warning("effect_kg_bridge_failed", error=str(e), exc_info=True)

        # Step 4.9: arriving somewhere repopulates it from durable knowledge.
        # rescope_scene only DROPS — leaving a tavern evicts its non-important
        # regulars from the scene roster and returning never brought them
        # back, so the narrator met an empty room and either ignored the
        # barkeep or invented a new one (which the returning-NPC re-anchoring
        # then had to reconcile after the fact).
        if world_state is not None and kg is not None:
            try:
                self._hydrate_scene_from_knowledge(world_state, kg, ws_location_before)
            except Exception as e:
                logger.warning("scene_hydration_failed", error=str(e), exc_info=True)

        # Step 5: Consume resources/currency AFTER outcome is determined.
        # This ensures we don't deduct ammunition on a cancelled action or
        # currency on a failed narration. Both consumers dedupe against the
        # turn's executed update_player receipts (a purchase's authoritative
        # currency claim included) and route residual writes through the
        # effect pipeline, so every player-state write this turn is receipted
        # exactly once.
        if triage.resources_consumed:
            await self._consume_resources(triage.resources_consumed, player_name)
        if triage.currency_spent:
            await self._consume_currency(triage.currency_spent, player_name)

        # Step 6: Auto-populate DM scratchpad from this turn's results
        self._update_scratchpad(triage, resolution, proposed_effects, combat_triggered, player_name)

        # ── Turn Logger: finalize and flush ──
        ws_after = world_state.to_yaml(
            action_text=action, action_entities=_ws_action_entities,
        ) if world_state else ""
        _turn_record.record_world_state(ws_before, ws_after)
        _turn_record.set("combat_triggered", combat_triggered)
        _turn_record.set("effects_count", len(self._last_effect_executions))
        _turn_record.set(
            "effects_proposed_count",
            len(proposed_effects) + len(self._last_deterministic_proposed),
        )
        _turn_record.record_effects(
            proposed=[_effect_telemetry_record(effect) for effect in proposed_effects]
            + list(self._last_deterministic_proposed),
            executed=list(self._last_effect_executions),
            rejected=list(self._last_effect_rejections),
        )

        # Token usage stats
        narr_prompt = getattr(self, '_last_narrator_prompt_tokens', 0)
        narr_completion = getattr(self, '_last_narrator_completion_tokens', 0)
        narr_cache_read = getattr(self, '_last_narrator_cache_read_tokens', 0)
        narr_cache_write = getattr(self, '_last_narrator_cache_write_tokens', 0)
        if narr_prompt or narr_completion:
            narr_time_s = _turn_record.data.get("timings", {}).get("narrate", 0) / 1000
            tps = narr_completion / narr_time_s if narr_time_s > 0 else 0
            token_stats = {
                "narrator_prompt": narr_prompt,
                "narrator_completion": narr_completion,
                "narrator_tps": round(tps, 1),
            }
            # Surface cache stats only when the provider reports them — keeps
            # turn logs clean for Ollama/Groq runs.
            if narr_cache_read or narr_cache_write:
                token_stats["narrator_cache_read"] = narr_cache_read
                token_stats["narrator_cache_write"] = narr_cache_write
                total_input = narr_prompt + narr_cache_read
                token_stats["narrator_cache_hit_ratio"] = (
                    round(narr_cache_read / total_input, 3) if total_input else 0.0
                )
            _turn_record.set("tokens", token_stats)

        self._turn_logger.flush(_turn_record)

        return DMResponse(
            narrative=narrative,
            mechanical_result=mechanical_result,
            tool_calls_made=tool_calls,
            dice_rolls=dice_rolls,
            combat_triggered=combat_triggered,
            proposed_effects=proposed_effects,
        )

    async def _triage_action(
        self,
        action: str,
        player_name: str,
        context: BrainContext,
    ) -> TriageResult:
        """
        Triage: decide if this action needs mechanical resolution.

        Runs a JSON-schema structured-output call (TriageSchema) on the
        brain client. Returns a structured decision with roll requirements
        and success/failure reveals.
        """
        # Build context for triage
        triage_context = await self._build_triage_context(action, player_name, context)

        messages = [
            {"role": "system", "content": TRIAGE_SYSTEM_PROMPT},
            {"role": "user", "content": triage_context},
        ]

        try:
            response = await self.client.chat(
                messages=messages,
                temperature=0,  # Ollama recommends 0 for structured output
                max_tokens=500,
                json_schema=get_triage_schema(),  # Use proper JSON schema
                think=False,  # Disable Qwen3 thinking mode for structured output
            )

            # Parse JSON response
            triage_data, parse_warnings = self._parse_triage_json(response.content)

            # Log parse failures at warning level so they're visible
            if triage_data.get("_parse_failed"):
                logger.warning(
                    "triage_parse_failure_detected",
                    raw_response=response.content[:300] if response.content else "empty",
                    action=action[:100],
                )
            if parse_warnings:
                logger.warning(
                    "triage_parse_warnings",
                    warnings=parse_warnings,
                    action=action[:80],
                )

            # Log for debugging
            logger.debug("triage_raw_response", content=response.content[:200] if response.content else "empty")

            # Get action type (required)
            action_type = triage_data.get("action_type", "roleplay")

            # Get the appropriate guidance based on action type
            narrative_direction = triage_data.get("narrative_direction")
            needs_roll = triage_data.get("needs_roll", False)

            # For non-mechanical actions, ensure we have narrative direction
            if action_type in ("social", "roleplay", "exploration") and not narrative_direction:
                narrative_direction = f"The action proceeds. {triage_data.get('reasoning', 'Describe the outcome.')}"

            # Convert empty strings to None (schema uses "" defaults to avoid
            # anyOf unions that break Groq, but downstream expects None for "not set")
            def _or_none(val):
                return val if val else None

            result = TriageResult(
                action_type=action_type,
                reasoning=triage_data.get("reasoning", ""),
                needs_roll=needs_roll,
                roll_type=_or_none(triage_data.get("roll_type")),
                ability=_or_none(triage_data.get("ability")),
                skill=_or_none(triage_data.get("skill")),
                dc=triage_data.get("dc") or None,
                advantage=triage_data.get("advantage", False),
                disadvantage=triage_data.get("disadvantage", False),
                advantage_reason=_or_none(triage_data.get("advantage_reason")),
                on_success=triage_data.get("on_success", []),
                on_failure=triage_data.get("on_failure", []),
                # Purchase fields
                item_name=_or_none(triage_data.get("item_name")),
                item_cost=triage_data.get("item_cost") or None,
                quantity=triage_data.get("quantity", 1),
                # Attack fields
                target_name=_or_none(triage_data.get("target_name")),
                is_creature_target=triage_data.get("is_creature_target", False),
                # Narrative guidance
                narrative_direction=_or_none(narrative_direction),
                # Tier-routing hint (Phase C). Validate to known values; any
                # other string defaults to "routine" so brain hallucinations
                # don't bleed into routing decisions.
                narrative_significance=_validate_significance(
                    triage_data.get("narrative_significance")
                ),
                # Resource consumption
                resources_consumed=triage_data.get("resources_consumed", []),
                # Currency spending
                currency_spent=triage_data.get("currency_spent") or None,
            )
            # Attach parse warnings for turn log observability
            result._parse_warnings = parse_warnings
            return result

        except Exception as e:
            import traceback
            logger.error(
                "triage_failed",
                error=str(e),
                traceback=traceback.format_exc(),
                action=action[:100],
            )
            _write_debug_log("TRIAGE_ERROR", f"Action: {action}\nError: {str(e)}\n{traceback.format_exc()}")
            # Default to roleplay on triage failure
            return TriageResult(
                action_type="roleplay",
                reasoning=f"Triage error: {str(e)}",
                narrative_direction="Describe the character's action and the scene's response naturally.",
            )

    async def _build_triage_context(
        self,
        action: str,
        player_name: str,
        context: BrainContext,
    ) -> str:
        """Build the context string for triage decision."""
        parts = []

        # Scene entities (NPCs, creatures, objects present) - FIRST for priority
        if self._scene_registry and self._scene_registry.has_entities():
            entity_context = self._scene_registry.get_triage_context()
            if entity_context:
                parts.append(entity_context)

        # Current scene
        if context.current_scene:
            parts.append(f"## Current Scene\n{context.current_scene}")

        # Party info
        party = context.party_members or context.party_status
        if party:
            parts.append(f"## Party\n{party}")

        # Character capabilities - critical for determining what's possible
        char_capabilities = await self._get_character_capabilities(player_name)
        if char_capabilities:
            parts.append(f"## {player_name}'s Capabilities\n{char_capabilities}")

        # Combat state
        if context.combat_state:
            parts.append(f"## Combat\n{context.combat_state}")
        elif context.in_combat:
            parts.append(f"## Combat Active\nRound {context.combat_round}")

        # The action to analyze
        parts.append(f"## Player Action\n[{player_name}]: \"{action}\"")
        parts.append("\nAnalyze this action considering the character's capabilities and output your triage decision as JSON.")

        return "\n\n".join(parts)

    async def _get_character_capabilities(self, player_name: str) -> str:
        """Get a summary of character capabilities for triage context."""
        if not self._current_session:
            return ""

        # Find the character
        character = None
        for player in self._current_session.players.values():
            if player.character and player_name.lower() in player.character.name.lower():
                character = player.character
                break

        if not character:
            return ""

        capabilities = []

        # Class and level
        capabilities.append(f"Class: {character.class_index.title()} Level {character.level}")

        # Key proficiencies
        if character.skill_proficiencies:
            skills = [s.value if hasattr(s, 'value') else s for s in character.skill_proficiencies]
            capabilities.append(f"Proficient Skills: {', '.join(skills)}")

        # Spellcasting (if any)
        if hasattr(character, 'spells_known') and character.spells_known:
            spell_names = [s.name if hasattr(s, 'name') else str(s) for s in character.spells_known[:10]]
            capabilities.append(f"Known Spells: {', '.join(spell_names)}")

        # Spell slots available
        if hasattr(character, 'spell_slots') and character.spell_slots:
            slot_parts = []
            for level in range(1, 10):
                try:
                    current, maximum = character.spell_slots.get_slots(level)
                    if maximum > 0:
                        slot_parts.append(f"L{level}: {current}/{maximum}")
                except (ValueError, AttributeError):
                    break
            if slot_parts:
                capabilities.append(f"Spell Slots: {', '.join(slot_parts)}")

        # Inventory and gold - fetch from database
        try:
            inventory_repo = await get_inventory_repo()

            # Get currency first - critical for purchase validation
            currency = await inventory_repo.get_currency(character.id)
            if currency:
                gold_parts = []
                if currency.platinum > 0:
                    gold_parts.append(f"{currency.platinum}pp")
                if currency.gold > 0:
                    gold_parts.append(f"{currency.gold}gp")
                if currency.electrum > 0:
                    gold_parts.append(f"{currency.electrum}ep")
                if currency.silver > 0:
                    gold_parts.append(f"{currency.silver}sp")
                if currency.copper > 0:
                    gold_parts.append(f"{currency.copper}cp")

                if gold_parts:
                    capabilities.append(f"Currency: {', '.join(gold_parts)} (≈{currency.total_in_gold:.1f}gp total)")
                else:
                    capabilities.append("Currency: None (0gp)")

            # Get inventory items
            items = await inventory_repo.get_all_items(character.id)
            if items:
                # Group items by relevance for action capability
                equipped = [i for i in items if i.equipped]
                tools_and_consumables = [
                    i for i in items if not i.equipped and any(
                        keyword in i.item_name.lower()
                        for keyword in [
                            'tinderbox', 'torch', 'lantern', 'rope', 'grappling',
                            'crowbar', 'hammer', 'piton', 'thieves', 'tools',
                            'kit', 'potion', 'scroll', 'wand', 'staff', 'oil',
                            'flask', 'vial', 'holy', 'component', 'focus',
                        ]
                    )
                ]

                if equipped:
                    equipped_names = [f"{i.item_name}" for i in equipped[:5]]
                    capabilities.append(f"Equipped: {', '.join(equipped_names)}")

                if tools_and_consumables:
                    tool_names = [f"{i.item_name} (x{i.quantity})" if i.quantity > 1 else i.item_name
                                  for i in tools_and_consumables[:8]]
                    capabilities.append(f"Tools/Consumables: {', '.join(tool_names)}")

                # Note total inventory size
                other_count = len(items) - len(equipped) - len(tools_and_consumables)
                if other_count > 0:
                    capabilities.append(f"(+{other_count} other items in inventory)")
        except Exception as e:
            logger.debug("inventory_fetch_failed", error=str(e), exc_info=True)
            # Continue without inventory if fetch fails

        return "\n".join(capabilities)

    def _parse_triage_json(self, content: str) -> tuple[dict, list[str]]:
        """Parse and validate JSON from triage response.

        Uses Pydantic validation against TriageSchema. Logs warnings on
        parse failures instead of silently returning defaults.

        Returns (data_dict, parse_warnings) so callers can record warnings
        in the turn log for post-mortem observability.
        """
        data, warnings = extract_json_object(content)
        if data is None:
            logger.error(
                "triage_json_parse_failed",
                raw_content=content[:500],
                warnings=warnings,
            )
            # Return a clearly marked failure so callers can detect it
            return {
                "action_type": "roleplay",
                "needs_roll": False,
                "reasoning": "TRIAGE_PARSE_FAILURE: LLM returned non-JSON response",
                "_parse_failed": True,
            }, warnings

        # Coerce bare strings to single-element lists for list fields.
        # Small models often output "on_success": "description" instead
        # of "on_success": ["description"].
        for field in ("on_success", "on_failure", "resources_consumed"):
            if field in data and isinstance(data[field], str):
                data[field] = [data[field]]
                warnings.append(f"coerced_string_to_list:{field}")

        # Validate against schema — log any coercion issues but don't reject
        try:
            validated = TriageSchema(**data)
            return validated.model_dump(), warnings
        except Exception as e:
            logger.warning(
                "triage_schema_validation_failed",
                raw_data=str(data)[:500],
                error=str(e),
                exc_info=True,
            )
            warnings.append(f"schema_validation_failed: {e}")
            # Data parsed as JSON but failed schema validation — use raw data
            # with defaults for missing fields
            data.setdefault("action_type", "roleplay")
            data.setdefault("needs_roll", False)
            data.setdefault("reasoning", "Schema validation failed, using raw data")
            return data, warnings

    # =========================================================================
    # ACTION HANDLERS - Execute mechanics based on action_type
    # =========================================================================

    def _is_known_goods(self, item_name: str) -> bool:
        """Goods the deterministic commerce path may transact: SRD catalog
        matches, or objects already registered in the scene."""
        if _srd_goods_match(item_name):
            return True
        if self._scene_registry:
            entity = self._scene_registry.get_by_name(item_name)
            if entity is not None and entity.entity_type == EntityType.OBJECT:
                return True
        return False

    async def _handle_purchase(
        self,
        triage: TriageResult,
        player_name: str,
        context: BrainContext,
    ) -> Optional[dict]:
        """Handle a purchase — validate goods/funds, then claim the turn's
        player-state write via authoritative effects.

        One narrated event, one writer: the returned authoritative_effects
        execute through the effect pipeline (receipted update_player) and
        REPLACE the narrator's own update_player for the turn. The old shape
        charged gold directly (no receipt) while the narrator's currency_delta
        also executed — the player paid twice for one narrated payment (live
        run 20260724_231350 T2) — and mistriaged social payments minted
        service phrases and the literal "unknown item" fallback as inventory
        rows (T2/T8).

        Returns None when there is nothing deterministic to transact (no
        concrete goods named): the turn falls through to the plain narrator
        path, whose receipted update_player owns any currency/item movement.
        """
        item_name = (triage.item_name or "").strip()
        if _is_generic_item_phrase(item_name):
            logger.info(
                "purchase_deferred_no_item",
                player=player_name,
                item_name=item_name,
            )
            return None
        if not self._is_known_goods(item_name):
            logger.info(
                "purchase_deferred_not_goods",
                player=player_name,
                item_name=item_name,
            )
            return None

        # Costs arrive as whatever the brain emitted: TriageSchema declares
        # int, but _parse_triage_json falls back to the RAW dict when schema
        # validation fails, so a float (an SRD 4cp ale, captured live) reaches
        # here. A float currency delta is rejected by effect validation —
        # which used to drop the purchase silently AFTER the narrator's own
        # write had been scrubbed. Work in copper, the smallest real unit.
        try:
            cost_copper_each = max(0, round(float(triage.item_cost or 0) * 100))
        except (TypeError, ValueError):
            cost_copper_each = 0
        item_cost = cost_copper_each / 100
        quantity = max(1, int(triage.quantity or 1))

        # Resolve character
        resolved = self._resolve_character_by_name(player_name)
        if not resolved:
            return {
                "action_type": "purchase",
                "success": False,
                "error": f"Character '{player_name}' not found",
                "item": item_name,
                "narrative_hint": f"{player_name} isn't known to the system.",
            }

        _, character = resolved

        # Funds check is read-only; the write happens once, in the effect
        # pipeline, which breaks larger coins. Gate on TOTAL wealth, matching
        # PGI's validate_currency — gating on the gold column alone both
        # refused affordable purchases from a platinum purse and disagreed
        # with PGI about the same turn.
        total_copper = cost_copper_each * quantity
        total_cost = total_copper / 100
        inventory_repo = await get_inventory_repo()
        currency = await inventory_repo.get_currency(character.id)
        if currency.total_in_copper < total_copper:
            error = (
                f"Not enough coin. Have {_format_coin(currency.total_in_copper)}, "
                f"need {_format_coin(total_copper)}"
            )
            return {
                "action_type": "purchase",
                "success": False,
                "item": item_name,
                "quantity": quantity,
                "cost": item_cost,
                "gold_after": currency.gold,
                "error": error,
                "narrative_hint": (
                    f"{player_name} cannot complete the purchase: {error}"
                ),
            }

        copper_after = currency.total_in_copper - total_copper
        gold_after = round(copper_after / 100, 2)
        effect = ProposedEffect(
            effect_type=EffectType.UPDATE_PLAYER,
            player_currency_delta=(
                {"cp": -total_copper} if total_copper else {}
            ),
            player_item_grant=[{"name": item_name, "quantity": quantity}],
        )
        return {
            "action_type": "purchase",
            "success": True,
            "item": item_name,
            "quantity": quantity,
            "cost": item_cost,
            "gold_after": gold_after,
            "error": None,
            "narrative_hint": (
                f"{player_name} successfully purchases {quantity}x {item_name} "
                f"for {_format_coin(total_copper)}. They have "
                f"{_format_coin(copper_after)} remaining."
            ),
            "authoritative_effects": [effect],
        }

    async def _handle_sell(
        self,
        triage: TriageResult,
        player_name: str,
        context: BrainContext,
    ) -> dict:
        """
        Handle sell action - execute through Rules Brain tools.
        """
        item_name = triage.item_name or "unknown item"
        quantity = triage.quantity or 1

        # For now, return a placeholder - needs inventory lookup to find item_id
        # TODO: Implement full sell logic with inventory lookup
        return {
            "action_type": "sell",
            "success": False,
            "error": "Sell action not yet implemented - use /inventory commands",
            "item": item_name,
            "narrative_hint": f"{player_name} attempts to sell {item_name}, but the merchant isn't interested right now.",
        }

    async def _handle_inventory(
        self,
        triage: TriageResult,
        player_name: str,
        context: BrainContext,
        action: str,
    ) -> Optional[dict]:
        """
        Handle inventory actions - pick up, drop, use, equip items.

        Parses the action to determine what kind of inventory operation.
        Mutations go through authoritative effects (receipted update_player,
        replacing the narrator's own) — never direct repo writes. Returns
        None to defer the whole turn to the narrator when there is nothing
        deterministic to do (consumption/interaction, or no resolvable item):
        the narrator's receipted tools own the state change, and Step 5's
        consumption net covers a forgotten removal.
        """
        action_lower = action.lower()

        # Resolve character
        resolved = self._resolve_character_by_name(player_name)
        if not resolved:
            return {
                "action_type": "inventory",
                "success": False,
                "error": f"Character '{player_name}' not found",
                "narrative_hint": f"{player_name} isn't known to the system.",
            }

        _, character = resolved

        # Determine inventory action type from the action text. NPC transfers
        # are their own operations: treating "hand my compass to Mara" as an
        # unknown interaction fed [RESULT: FAILURE] to the narrator even when
        # the player owned the item and Mara had already agreed.
        transfer_from_npc = bool(re.search(
            r"\b(?:hands?|gives?|passes?|returns?)\b.+\b(?:back\s+)?to\s+me\b",
            action_lower,
        ))
        transfer_to_npc = bool(re.search(
            r"\b(?:i\s+)?(?:hand|give|pass|offer)\b.+\bto\b",
            action_lower,
        )) and not transfer_from_npc

        pickup_keywords = ["pick up", "pickup", "take", "grab", "collect", "pocket", "put in", "stash", "keep"]
        drop_keywords = ["drop", "discard", "throw away", "leave", "abandon"]
        equip_keywords = ["equip", "wield", "wear", "don", "put on", "draw"]
        unequip_keywords = ["unequip", "remove", "take off", "sheathe", "stow"]
        use_keywords = ["use", "drink", "eat", "consume", "apply", "read"]

        # Resolve overlapping phrases before branching.  In particular,
        # ``unequip`` contains ``equip`` and ``take off`` contains ``take``;
        # the old independent booleans therefore routed both to the wrong
        # mutation and reported a success that never reached the DB.
        if transfer_from_npc:
            operation = "transfer_from_npc"
        elif transfer_to_npc:
            operation = "transfer_to_npc"
        elif any(kw in action_lower for kw in unequip_keywords):
            operation = "unequip"
        elif any(kw in action_lower for kw in equip_keywords):
            operation = "equip"
        elif any(kw in action_lower for kw in drop_keywords):
            operation = "drop"
        elif any(kw in action_lower for kw in pickup_keywords):
            operation = "pickup"
        elif any(kw in action_lower for kw in use_keywords):
            operation = "use"
        else:
            operation = "interact"

        # Extract item name from triage or try to parse from action
        item_name = triage.item_name
        if not item_name:
            # Try to extract from common patterns like "pick up the feather"
            patterns = [
                r"(?:hands?|gives?|passes?|offers?|returns?)\s+(?:my\s+|the\s+|her\s+|his\s+|their\s+)?(.+?)\s+(?:back\s+)?to\b",
                r"(?:pick up|take|grab|pocket|keep|put)\s+(?:the\s+)?(.+?)(?:\s+in|\s+into|$)",
                r"(?:drop|discard)\s+(?:the\s+)?(.+?)$",
                r"(?:unequip|remove|take off|sheathe|stow)\s+(?:the\s+)?(.+?)$",
                r"(?:equip|wield|wear|don|put on|draw)\s+(?:the\s+)?(.+?)$",
                r"(?:use|drink|eat|consume|apply|read)\s+(?:the\s+)?(.+?)$",
            ]
            for pattern in patterns:
                match = re.search(pattern, action_lower)
                if match:
                    item_name = match.group(1).strip()
                    break

        # No "item" fallback: a generic phrase must never reach row matching.
        # The live sweep's junk 'unknown item' row was transferred instead of
        # the named compass exactly because "item" substring-matched it.
        if item_name:
            item_name = item_name.rstrip(".")
        if _is_generic_item_phrase(item_name):
            item_name = None

        quantity = triage.quantity or 1

        if operation in {"transfer_to_npc", "transfer_from_npc"}:
            world = (
                self._current_session.world_state
                if self._current_session else None
            )
            local_npcs = world.get_npcs_at_location() if world else []
            target_npc = next(
                (
                    npc for npc in local_npcs
                    if npc.name and npc.name.casefold() in action_lower
                ),
                None,
            )
            if target_npc is None:
                return {
                    "action_type": "inventory",
                    "operation": operation,
                    "success": False,
                    "item": item_name,
                    "error": "Transfer target is not a named NPC in the scene",
                    "narrative_hint": (
                        f"{player_name} cannot complete the transfer because "
                        "the recipient is not present."
                    ),
                }

            inventory_repo = await get_inventory_repo()
            player_items = await inventory_repo.get_all_items(character.id)
            holdings = (
                [item.item_name for item in player_items]
                if operation == "transfer_to_npc"
                else list(target_npc.inventory)
            )
            canonical_item = _resolve_transfer_item(
                action_lower, holdings, item_name
            )
            if canonical_item is None:
                if item_name:
                    # A concrete item was named but the giver doesn't hold it
                    # — fail closed rather than let narration move phantoms.
                    holder = (
                        player_name if operation == "transfer_to_npc"
                        else target_npc.name
                    )
                    error = (
                        f"'{item_name}' not found in inventory"
                        if operation == "transfer_to_npc"
                        else f"{target_npc.name} does not hold '{item_name}'"
                    )
                    return {
                        "action_type": "inventory",
                        "operation": operation,
                        "success": False,
                        "item": item_name,
                        "error": error,
                        "narrative_hint": (
                            f"{holder} does not have the {item_name} to give."
                        ),
                    }
                # Nothing resolvable ("hands it to me" with no named item):
                # defer the exchange to the narrator's receipted tools. The
                # operation tag keeps PGI's acquisition exemption intact.
                logger.info(
                    "inventory_transfer_deferred_unresolved_item",
                    operation=operation,
                    npc=target_npc.name,
                )
                return {
                    "action_type": "inventory",
                    "operation": operation,
                    "success": True,
                    "item": "",
                    "narrative_hint": (
                        f"The exchange between {player_name} and "
                        f"{target_npc.name} proceeds as described; the "
                        "narration resolves which item changes hands."
                    ),
                }

            if operation == "transfer_to_npc":
                authoritative_effects = [
                    ProposedEffect(
                        effect_type=EffectType.UPDATE_PLAYER,
                        player_item_remove=[{
                            "name": canonical_item,
                            "quantity": quantity,
                            "destination": f"npc:{target_npc.id}",
                        }],
                    ),
                    ProposedEffect(
                        effect_type=EffectType.UPDATE_ENTITY,
                        update_entity_id=target_npc.id,
                        update_add_items=[canonical_item],
                    ),
                ]
                hint = (
                    f"{player_name} hands the {canonical_item} to "
                    f"{target_npc.name}, who accepts it and now holds it."
                )
            else:
                authoritative_effects = [
                    ProposedEffect(
                        effect_type=EffectType.UPDATE_PLAYER,
                        player_item_grant=[{
                            "name": canonical_item,
                            "quantity": quantity,
                            "source": f"npc:{target_npc.id}",
                        }],
                    ),
                    ProposedEffect(
                        effect_type=EffectType.UPDATE_ENTITY,
                        update_entity_id=target_npc.id,
                        update_remove_items=[canonical_item],
                    ),
                ]
                hint = (
                    f"{target_npc.name} hands the {canonical_item} back to "
                    f"{player_name}, who puts it in their pack."
                )

            return {
                "action_type": "inventory",
                "operation": operation,
                "success": True,
                "item": canonical_item,
                "quantity": quantity,
                "target_npc_id": target_npc.id,
                "narrative_hint": hint,
                "authoritative_effects": authoritative_effects,
            }

        if operation == "pickup":
            if item_name is None:
                # "…a distinct object I can pick up" carried no resolvable
                # item name; the old "item" fallback minted a literal 'Item'
                # inventory row here (live run 20260724_231350 T10). Defer:
                # the narrator's receipted grant owns any acquisition.
                # Keep PGI's acquisition exemption: a pickup's row does not
                # exist yet by definition, so validating triage's item_name
                # against inventory would hard-fail the turn with no
                # narration at all — the worst failure this gate has.
                self._pgi_acquisition_turn = True
                logger.info(
                    "inventory_pickup_deferred_no_item", player=player_name
                )
                return None
            return {
                "action_type": "inventory",
                "operation": "pickup",
                "success": True,
                "item": item_name,
                "quantity": quantity,
                "narrative_hint": (
                    f"{player_name} picks up the {item_name} and adds it to "
                    "their inventory."
                ),
                "authoritative_effects": [ProposedEffect(
                    effect_type=EffectType.UPDATE_PLAYER,
                    player_item_grant=[{
                        "name": item_name.title(),
                        "quantity": quantity,
                    }],
                )],
            }

        elif operation == "drop":
            # Resolve the item against the character's real inventory before
            # claiming a drop; the removal itself rides an authoritative
            # effect so it lands exactly once, with a receipt, replacing any
            # narrator mirror of the same removal.
            if item_name is None:
                logger.info(
                    "inventory_drop_deferred_no_item", player=player_name
                )
                return None
            inventory_repo = await get_inventory_repo()
            items = await inventory_repo.get_all_items(character.id)
            item_lower = item_name.lower()
            matching = [i for i in items if item_lower in i.item_name.lower()]

            if not matching:
                return {
                    "action_type": "inventory",
                    "operation": "drop",
                    "success": False,
                    "item": item_name,
                    "error": f"'{item_name}' not found in inventory",
                    "narrative_hint": (
                        f"{player_name} reaches for the {item_name}, "
                        "but doesn't have it."
                    ),
                }

            item = matching[0]
            drop_qty = min(quantity, item.quantity)
            return {
                "action_type": "inventory",
                "operation": "drop",
                "success": True,
                "item": item.item_name,
                "quantity": drop_qty,
                "narrative_hint": f"{player_name} drops the {item.item_name}.",
                "authoritative_effects": [ProposedEffect(
                    effect_type=EffectType.UPDATE_PLAYER,
                    player_item_remove=[{
                        "name": item.item_name,
                        # Carry the resolved ROW: a name-only removal is
                        # re-resolved downstream and missed equipped rows and
                        # SRD-indexed rows entirely (silent no-op drops).
                        "item_index": item.item_index,
                        "quantity": drop_qty,
                    }],
                )],
            }

        elif operation in {"equip", "unequip"}:
            if item_name is None:
                # No update_player field mutates equip state, so there is
                # nothing to defer to — an unspecified target fails honestly.
                return {
                    "action_type": "inventory",
                    "operation": operation,
                    "success": False,
                    "item": "",
                    "error": f"No item named to {operation}",
                    "narrative_hint": (
                        f"{player_name} isn't holding anything specific to "
                        f"{operation}."
                    ),
                }
            inventory_repo = await get_inventory_repo()
            items = await inventory_repo.get_all_items(character.id)
            item_lower = item_name.lower()
            should_be_equipped = operation == "equip"
            matching = [
                item for item in items
                if item_lower in item.item_name.lower()
                and item.equipped is not should_be_equipped
            ]

            if not matching:
                desired_state = "unequipped" if should_be_equipped else "equipped"
                return {
                    "action_type": "inventory",
                    "operation": operation,
                    "success": False,
                    "item": item_name,
                    "error": f"'{item_name}' not found {desired_state} in inventory",
                    "narrative_hint": (
                        f"{player_name} tries to {operation} the {item_name}, "
                        f"but doesn't have it {desired_state}."
                    ),
                }

            item = matching[0]
            if should_be_equipped:
                changed = await inventory_repo.equip_item(item.id)
            else:
                changed = await inventory_repo.unequip_item(item.id)

            if not changed:
                return {
                    "action_type": "inventory",
                    "operation": operation,
                    "success": False,
                    "item": item.item_name,
                    "error": f"Failed to {operation} '{item.item_name}'",
                    "narrative_hint": (
                        f"{player_name} tries to {operation} the {item.item_name}, "
                        "but its state does not change."
                    ),
                }

            verb = "equips" if should_be_equipped else "unequips"
            return {
                "action_type": "inventory",
                "operation": operation,
                "success": True,
                "item": item.item_name,
                "narrative_hint": f"{player_name} {verb} the {item.item_name}.",
                "tool_calls": [{
                    "name": f"{operation}_item",
                    "result": {
                        "character": character.name,
                        "item": item.item_name,
                        "item_id": item.id,
                        "equipped": should_be_equipped,
                    },
                }],
            }

        elif operation == "use":
            # Consumption is narrator-owned: its receipted update_player
            # removes the item and applies the effect. Feeding the narrator a
            # [RESULT: FAILURE] here contradicted established premises like
            # "I drink the antidote" (live run 20260724_231350 T6), while
            # Step 5's receipted consumption net covers a forgotten removal.
            # PGI has already validated the player actually holds the item.
            logger.info(
                "inventory_use_deferred_to_narrator",
                player=player_name,
                item=item_name,
            )
            return None

        else:
            # Unclassifiable inventory phrasing: nothing deterministic to
            # claim, so the plain narrator path owns the turn (and any state
            # change rides its receipted tools).
            logger.info(
                "inventory_interact_deferred_to_narrator",
                player=player_name,
                item=item_name,
            )
            return None

    def _hydrate_scene_from_knowledge(
        self,
        world_state: "WorldState",
        knowledge_graph,
        location_before: str,
    ) -> list[str]:
        """Restore a location's known inhabitants when the party arrives.

        Only runs on an actual arrival. Reads the graph's durable residency
        record and hands it to the store, which owns the WorldState write
        (Step-4 single-writer rule) and the dead/cap policy.
        """
        destination = (world_state.current_location or "").strip()
        if not destination or locations_equivalent(location_before, destination):
            return []

        store = (
            getattr(self._current_session, "world_store", None)
            if self._current_session else None
        )
        if store is None:
            return []

        # Resolve by equivalence, not raw slug: the narrator's "the Copper
        # Finch" and the graph's "Copper Finch" are the same room.
        location_node = knowledge_graph.resolve_location_node(destination)
        if not location_node:
            return []
        # Anchor the scene to canon's name for this place before restoring
        # anyone: residents are recorded under it, and a paraphrased
        # current_location ("the tavern") makes every later comparison miss.
        canonical = getattr(
            knowledge_graph.get_entity(location_node), "name", ""
        )
        if canonical:
            store.canonicalize_location(canonical)
        residents = knowledge_graph.residents_of(location_node)
        if not residents:
            return []

        return store.hydrate_residents(
            residents,
            dead_facts=list(
                (getattr(self._current_session, "campaign_dead_npcs", {}) or {}).values()
            ),
        )

    def _turn_update_player_receipts(self) -> list[dict]:
        """This turn's executed update_player receipts ("applied" payloads)."""
        receipts: list[dict] = []
        for record in self._last_effect_executions:
            if record.get("type") != "update_player" or record.get("was_duplicate"):
                continue
            applied = (record.get("details") or {}).get("applied") or {}
            if applied:
                receipts.append(applied)
        return receipts

    async def _execute_deterministic_player_effect(
        self,
        effect: ProposedEffect,
        character_id: str,
    ) -> bool:
        """Run a deterministic player-state write through the effect pipeline.

        Step 5's consumption used to call the inventory repo directly — a
        write with no receipt, invisible to the receipt-vs-state agreement
        gate and stacked on top of the narrator's update_player for the same
        narrated event (double-charge observed live, run 20260724_234241 T2).
        Routing through the executor gives the write the same receipt,
        idempotency, and WorldState sync as every other player-state writer,
        and the proposed/executed records keep the turn's effect accounting
        balanced.
        """
        if not self._effect_executor:
            inventory_repo = await get_inventory_repo()
            self._effect_executor = EffectExecutor(
                scene_registry=self._scene_registry,
                session=self._current_session,
                inventory_repo=inventory_repo,
                applied_effects_store=self._applied_effects,
            )
        self._effect_executor.acting_character_id = character_id

        # Same derivation the narrator path uses (session+turn, not a fresh
        # id), so a retried turn collapses onto the same keys instead of
        # consuming twice; the index is this turn's deterministic-write
        # sequence, mirroring the tool path's per-effect position.
        campaign_id = (
            (self._current_session.campaign_id if self._current_session else None)
            or "unknown"
        )
        world_state = (
            getattr(self._current_session, "world_state", None)
            if self._current_session else None
        )
        session_key = getattr(self._current_session, "session_key", None)
        # The acting character is part of the key: two players' turns can
        # observe the SAME world turn number (begin_turn increments before
        # the processing lock), and without it the second player's charge
        # collapsed onto the first's key as an idempotency "duplicate" —
        # logged as spent, never written.
        message_id = (
            f"{session_key}:{character_id}:turn-{world_state.turn}:step5"
            if session_key and world_state is not None
            else f"step5:{uuid.uuid4()}"
        )
        idem_key = build_effect_idempotency_key(
            campaign_id, message_id, len(self._last_deterministic_proposed)
        )
        result = await self._effect_executor.execute(effect, idem_key)
        if not result.success:
            logger.warning(
                "deterministic_player_effect_failed",
                effect_type=effect.effect_type.value,
                details=result.details,
            )
            return False

        # success=True is not proof a write landed: the executor reports
        # success even when a removal resolved to no row. Callers log
        # "consumed"/"spent" off this return, so it must mean "state moved".
        applied = (result.details or {}).get("applied") or {}
        landed = any(
            applied.get(key)
            for key in (
                "currency_delta", "items_granted", "items_removed",
                "conditions_added", "conditions_removed", "hp_delta",
                "spell_slot_used",
            )
        )
        if not landed and not result.was_duplicate:
            logger.warning(
                "deterministic_player_effect_no_op",
                effect_type=effect.effect_type.value,
                details=result.details,
            )
            return False

        record = {
            **_effect_telemetry_record(effect),
            "was_duplicate": bool(result.was_duplicate),
            "details": dict(result.details or {}),
        }
        self._last_deterministic_proposed.append(_effect_telemetry_record(effect))
        self._last_effect_executions.append(record)

        if not result.was_duplicate:
            world_store = (
                getattr(self._current_session, "world_store", None)
                if self._current_session else None
            )
            if world_store is not None:
                world_store.apply_effect(effect)
        return True

    async def _consume_resources(
        self,
        resources: list[dict],
        player_name: str,
    ) -> None:
        """
        Consume resources used by an action (ammunition, consumables, etc.).

        One writer per narrated event: a resource the narrator's executed
        update_player already removed this turn is skipped (its receipt owns
        the write); the residual consumption executes through the effect
        pipeline so it carries a receipt of its own.

        Args:
            resources: List of {"item": "Arrow", "quantity": 1} dicts
            player_name: Character name to consume from
        """
        resolved = self._resolve_character_by_name(player_name)
        if not resolved:
            logger.warning("cannot_consume_resources_no_character", player=player_name)
            return

        char_id, character = resolved
        inventory_repo = await get_inventory_repo()

        def _slug(name: str) -> str:
            return (name or "").strip().lower().replace(" ", "-")

        def _receipted_removals() -> set[str]:
            # Recomputed per resource: a residual write earlier in this loop
            # adds its own receipt, and a repeated entry must dedupe against
            # it just as it would against the narrator's.
            return {
                _slug(str(entry.get("name") or ""))
                for applied in self._turn_update_player_receipts()
                for entry in (applied.get("items_removed") or [])
            }

        for resource in resources:
            item_name = resource.get("item", "")
            quantity = resource.get("quantity", 1)

            if not item_name:
                continue

            # Find the item in inventory
            items = await inventory_repo.get_all_items(char_id)
            matching = [
                i for i in items
                if item_name.lower() in i.item_name.lower()
            ]

            if not matching:
                # Often the narrator's receipted removal already deleted the
                # row this turn — the ledger is complete without us.
                logger.debug(
                    "resource_not_found",
                    character=character.name,
                    item=item_name,
                )
                continue

            item = matching[0]
            receipted_removals = _receipted_removals()
            if (
                item.item_index in receipted_removals
                or _slug(item.item_name) in receipted_removals
            ):
                logger.info(
                    "resource_consumption_deduped",
                    character=character.name,
                    item=item.item_name,
                    reason="update_player receipt already removed it this turn",
                )
                continue

            if item.quantity < quantity:
                logger.warning(
                    "insufficient_resource",
                    character=character.name,
                    item=item_name,
                    have=item.quantity,
                    need=quantity,
                )
                continue

            consumed = await self._execute_deterministic_player_effect(
                ProposedEffect(
                    effect_type=EffectType.UPDATE_PLAYER,
                    player_item_remove=[{
                        "name": item.item_name,
                        "item_index": item.item_index,
                        "quantity": quantity,
                    }],
                ),
                char_id,
            )
            if consumed:
                logger.info(
                    "resource_consumed",
                    character=character.name,
                    item=item.item_name,
                    quantity=quantity,
                    remaining=item.quantity - quantity,
                )

    async def _consume_currency(
        self,
        currency_spent: dict,
        player_name: str,
    ) -> None:
        """
        Consume currency spent by an action (tipping, paying, dropping, etc.).

        Args:
            currency_spent: Dict like {"gold": 50} or {"silver": 10, "copper": 5}
            player_name: Character name to deduct from
        """
        if not currency_spent:
            return

        # One writer per narrated payment: any executed update_player
        # currency_delta this turn already owns the money movement — charging
        # again here is the live double-charge (run 20260724_234241 T2:
        # narrator receipted -2gp AND this path silently removed 2 more).
        if any(
            applied.get("currency_delta")
            for applied in self._turn_update_player_receipts()
        ):
            logger.info(
                "currency_consumption_deduped",
                requested=currency_spent,
                reason="update_player receipt already moved currency this turn",
            )
            return

        resolved = self._resolve_character_by_name(player_name)
        if not resolved:
            logger.warning("cannot_consume_currency_no_character", player=player_name)
            return

        char_id, character = resolved
        inventory_repo = await get_inventory_repo()

        # Convert all to gold equivalent and remove
        # Conversion: 1pp = 10gp, 1gp = 1gp, 1ep = 0.5gp, 1sp = 0.1gp, 1cp = 0.01gp
        total_gold = 0
        total_gold += currency_spent.get("platinum", 0) * 10
        total_gold += currency_spent.get("gold", 0)
        total_gold += currency_spent.get("electrum", 0) * 0.5
        total_gold += currency_spent.get("silver", 0) * 0.1
        total_gold += currency_spent.get("copper", 0) * 0.01

        if total_gold <= 0:
            return

        # Charge in copper so sub-gold amounts aren't rounded up to 1gp, and
        # gate on TOTAL wealth — the executor breaks larger coins, so a purse
        # holding platinum but no gold can still pay. Gating on the gold
        # column alone silently gave the service away for free.
        copper_to_remove = round(total_gold * 100)
        if copper_to_remove <= 0:
            return

        currency = await inventory_repo.get_currency(char_id)
        if currency.total_in_copper < copper_to_remove:
            logger.warning(
                "insufficient_currency",
                character=character.name,
                wanted=currency_spent,
                copper_equivalent=copper_to_remove,
                have_copper=currency.total_in_copper,
            )
            return

        spent = await self._execute_deterministic_player_effect(
            ProposedEffect(
                effect_type=EffectType.UPDATE_PLAYER,
                player_currency_delta={"cp": -copper_to_remove},
            ),
            char_id,
        )
        if spent:
            logger.info(
                "currency_spent",
                character=character.name,
                amount=currency_spent,
                copper_equivalent=copper_to_remove,
                copper_remaining=currency.total_in_copper - copper_to_remove,
            )

    # Action types Phase B vetoes as definitely-mundane. The brain can still
    # override with an explicit "climactic" classification (e.g. dropping the
    # entire party fortune as a final dramatic gesture flips the currency
    # transaction to climactic), but otherwise these route to standard.
    _PHASE_B_MUNDANE_ACTION_TYPES = frozenset({
        "inventory",   # equip / drop / use item
    })

    def _select_narrator_client_for_turn(
        self,
        action: str,
        triage: Optional["TriageResult"],
        context: Optional[BrainContext],
    ):
        """Pick the narrator client to use for this turn.

        Uses pre-narration signals only. See ``Docs/Roadmap/tiered_narrator.md``
        for the full design.

        Signals (in priority order):
        - ``FORCE_NARRATOR_TIER`` env var (QA/testing override)
        - Action prefixed with ``[OPENING SCENE]`` → opening tier
        - **Phase B veto** (deterministic) — known mundane paths force
          standard tier UNLESS the brain explicitly says ``"climactic"``:
          - In ongoing combat (combat narration round-by-round)
          - Action type is inventory
        - **Phase C** (brain-driven) — brain says ``"notable"`` or
          ``"climactic"`` → premium tier
        - Otherwise → standard tier
        """
        import os

        force_tier = os.environ.get("FORCE_NARRATOR_TIER") or None

        is_opening = bool(action and action.lstrip().startswith("[OPENING SCENE]"))

        # Phase C: brain-classified narrative weight (routine/notable/climactic)
        significance = getattr(triage, "narrative_significance", "routine") if triage else "routine"

        # Phase B veto: deterministic mundane-path detection. These paths
        # don't need premium prose — combat rounds, inventory shuffling,
        # routine purchases. The brain can still override with "climactic"
        # for true peak moments (boss death blow, PC death in combat, etc.).
        action_type = getattr(triage, "action_type", None) if triage else None
        in_combat = bool(context and getattr(context, "in_combat", False))

        definitely_standard = bool(
            in_combat
            or (action_type and action_type in self._PHASE_B_MUNDANE_ACTION_TYPES)
        )

        tier = select_narrator_tier(
            is_opening=is_opening,
            definitely_standard=definitely_standard,
            significance=significance,
            force_tier=force_tier,
        )

        client = self._narrator_client_factory(tier)

        # Keep self.narrator.client in sync so brain-side helpers that read
        # off the brain (e.g. format builders) see the current tier client.
        # This is an intentional mutation — narrator is a thin holder.
        self.narrator.client = client

        # Cache routing decision for the turn record (read by
        # process_narrator_turn → record_narrator_routing).
        self._last_narrator_routing = {
            "tier": tier,
            "provider": _get_client_provider(client),
            "model": getattr(client, "model", "") or "",
            "significance": significance,
            "phase_b_veto": bool(definitely_standard),
        }

        # Telemetry: log the tier choice with the signals that drove it. Lets
        # us tune the Phase B veto and brain classification thresholds over time.
        logger.info(
            "narrator_tier_selected",
            tier=tier,
            is_opening=is_opening,
            significance=significance,
            definitely_standard=definitely_standard,
            in_combat=in_combat,
            action_type=action_type,
            forced=bool(force_tier),
        )
        return client

    def _get_narrator_tools(self) -> list[dict]:
        """Select narrator tool set based on the active profile's tier.

        Tiers (configured in profiles.yaml under narrator.tools):
        - "core" (default): 3 tools — ref_entity, add_npc, spawn_object.
          Safest for sub-10B local narrators that drop tools as count grows.
        - "core_plus": adds change_location and start_combat. Suitable for
          capable local narrators (Qwen 3.6, Qwen 3.5:27b dense) and any
          cloud narrator.
        - "full": all registered tools for the tier. Cloud narrators
          (DeepSeek V4 Pro/Flash, Claude Sonnet/Haiku) handle the larger
          surface reliably.

        Whatever the narrator doesn't declare is filled in by the state
        and entity extractors as fallback.
        """
        from ..config import get_profile
        try:
            tier = get_profile().narrator.tools or "core"
        except Exception:
            # Profile load failed — fall back to safest tier
            tier = "core"
        return get_narrator_tools_for_tier(tier)

    def _append_tool_reminder(self, messages: list[dict]) -> None:
        """Append entity constraints, situational directives, and tool reminder.

        Placed last so it's the freshest instruction in the model's
        attention window. Dynamically injects rules based on current
        game state rather than loading all rules upfront.
        """
        parts = []

        # Inject entity constraints from world state — these are HARD FACTS.
        # Iterate values (dict is id-keyed); show npc.name (canonical), not the UUID.
        world_state = getattr(self._current_session, 'world_state', None) if self._current_session else None
        if world_state and world_state.npcs:
            constraints = []
            hostile_npcs = []
            # The reminder is part of the narrator prompt, so it must obey
            # the same scene boundary as WorldState.to_yaml and KG seeding.
            # Important off-screen NPCs remain durable in ``world_state.npcs``
            # but must not become ambient cast members on every turn.
            for npc in world_state.get_npcs_at_location():
                if npc.alive:
                    desc = npc.description[:80] if npc.description else ""
                    constraints.append(
                        f"- **{npc.name}** [id: {npc.id}]: "
                        f"{npc.disposition}, {desc}"
                    )
                    if npc.disposition in ("hostile", "unfriendly"):
                        hostile_npcs.append(npc.name)
            if constraints:
                parts.append(
                    "ENTITY FACTS (your prose MUST NOT contradict these):\n"
                    + "\n".join(constraints)
                )

            # Dynamic escalation: hostile NPCs should act, not wait
            if hostile_npcs:
                parts.append(
                    "HOSTILE NPC DIRECTIVE: "
                    + ", ".join(hostile_npcs)
                    + " are hostile. They do NOT wait passively. They advance, "
                    "threaten, attack, flee, or take tactical action EVERY turn. "
                    "Describe what THEY do, not just what they look like."
                )

        # Keep the final tool instruction self-contained. Some callers pass
        # a prebuilt BrainContext that omits the scene block; without this
        # roster copy the model can echo a plausible ID that neither the
        # followup guard nor StateDelta can safely classify as existing.
        scene_registry = getattr(self, "_scene_registry", None)
        if scene_registry:
            live_roster = scene_registry.get_narrator_roster()
            if live_roster:
                parts.append(live_roster)

        # The durable death catalog may grow large in a seeded campaign.
        # Surface only facts relevant to the assembled context; the
        # deterministic post-validator still checks every known dead NPC.
        grounding_text = "\n".join(
            str(message.get("content") or "") for message in messages
        ).casefold()
        dead_npcs = []
        for npc in self._known_dead_npcs():
            candidates = [npc.name, *npc.aliases]
            if not any(
                candidate and candidate.casefold() in grounding_text
                for candidate in candidates
            ):
                continue
            aliases = (
                f" (aliases: {', '.join(npc.aliases)})" if npc.aliases else ""
            )
            dead_npcs.append(f"- {npc.name}{aliases}: DEAD")
        if dead_npcs:
            parts.append(
                "IMMUTABLE DEAD-NPC FACTS (never contradict these):\n"
                + "\n".join(dead_npcs)
                + "\nA dead NPC cannot speak, move, react, arrive, or appear "
                "alive. Mention one only as an explicitly framed corpse, "
                "memory, dream, spirit, or illusion. Never invent a resurrection."
            )

        parts.append(
            "AFTER writing your prose, you MUST call tools:\n"
            "- ref_entity for each roster entity you mentioned\n"
            "- add_npc ONLY for a properly named new NPC physically present in this scene; the exact proper name must already appear in your visible prose; if you leave someone anonymous as 'the woman', 'a guard', or a role, do not call add_npc and do not invent a tool-only name\n"
            "- spawn_object for any new object you described\n"
            "Do NOT skip tool calls. Every entity in your prose must be tagged.\n\n"
            "Every tool call MUST include all required arguments; never call a tool with {}.\n\n"
            "NEVER write [id: ...] tags in your prose text."
        )

        messages.append({
            "role": "system",
            "content": "\n\n".join(parts),
        })

    def _extract_prose_and_effects(
        self,
        response,
        action: str = "",
    ) -> tuple[str, list[ProposedEffect]]:
        """Extract prose and effects from a narrator response.

        If the response has tool_calls, those become ProposedEffects and
        the content is pure prose. Otherwise, falls back to text-based
        PROSE/INTENTS parsing via the adjudicator.
        """
        # Capture token stats for turn logging
        self._last_narrator_prompt_tokens = getattr(response, 'prompt_tokens', 0)
        self._last_narrator_completion_tokens = getattr(response, 'completion_tokens', 0)
        # Prompt-cache telemetry (Anthropic, DeepSeek, ...). Zero on providers
        # without caching support — observability only, no behavior gating.
        self._last_narrator_cache_read_tokens = getattr(response, 'cache_read_tokens', 0)
        self._last_narrator_cache_write_tokens = getattr(response, 'cache_write_tokens', 0)

        raw_content = response.content.strip() if response.content else ""

        # Strip leaked [id: ...] tags from prose — the model sometimes
        # writes roster IDs inline instead of using tool calls
        if "[id:" in raw_content:
            import re
            raw_content = re.sub(r'\s*\[id:\s*[^\]]+\]', '', raw_content)

        # Tool-based path: content is pure prose, tool_calls are effects
        if response.tool_calls:
            effects = tool_calls_to_effects(response.tool_calls)
            logger.debug(
                "narrator_tools_used",
                tool_count=len(response.tool_calls),
                effects_count=len(effects),
                action=action[:60],
            )
            return raw_content, effects

        # Text fallback: parse PROSE/INTENTS blocks
        if raw_content and validate_narrator_format(raw_content):
            prose, effects, parse_result = self.adjudicator.parse_narrator_response(raw_content)
            return prose, effects

        # No format detected — treat entire content as prose, no effects
        return raw_content, []

    async def _narrate_mechanical_result(
        self,
        action: str,
        player_name: str,
        context: BrainContext,
        triage: TriageResult,
        mechanical_result: dict,
    ) -> tuple[str, list[ProposedEffect]]:
        """Narrate a mechanical action result (purchase/sell/inventory).

        Builds the NarrationSpec for a known mechanical outcome: the outcome
        rides in a USER prompt after the built messages, and the
        player_action reaches the narrator undecorated (the outcome is
        already in the prompt). On empty prose the mech narrative_hint is
        substituted and the turn continues (tool followup still runs); on
        any narration failure the hint is the whole answer — the mechanics
        already executed, so a narration hiccup must not hide the outcome.

        Returns:
            Tuple of (narrative text, proposed effects)
        """
        narrative_hint = mechanical_result.get("narrative_hint", "")
        success = mechanical_result.get("success", False)

        if success:
            resolution_text = f"[RESULT: SUCCESS] {narrative_hint}"
        else:
            resolution_text = f"[RESULT: FAILURE] {narrative_hint}"

        prompt = f"""The player {player_name} attempted: "{action}"

{resolution_text}

Narrate this action dramatically. Remember:
- Show the world's REACTION to the action (environment, NPCs, atmosphere)
- Connect to ongoing tension or stakes from the current scene
- End with something that maintains momentum

Write your narration directly."""

        spec = NarrationSpec(
            action=action,
            player_name=player_name,
            player_action=action,
            prompt=prompt,
            prompt_role="user",
            empty_prose_fallback=narrative_hint,
            continue_on_empty_prose=True,
            # The handler already owns these mutations; the obligation net
            # must not demand a duplicate narrator tool for them (a probe on
            # a handler-driven transfer showed it entering a repair leg for
            # exactly that, whose failure branch discards the claim).
            claimed_effects=list(
                mechanical_result.get("authoritative_effects") or []
            ),
        )

        try:
            return await self._narration_strategy.run(spec, context, triage)
        except Exception as e:
            logger.error("narrate_mechanical_failed", error=str(e), exc_info=True)
            # Clear the cached routing: run() may have died before (or during)
            # tier selection, and the turn record must not report the PREVIOUS
            # turn's routing for this degraded turn.
            self._last_narrator_routing = None
            return narrative_hint, []  # Fall back to the hint

    async def _resolve_mechanics(
        self,
        triage: TriageResult,
        player_name: str,
        context: BrainContext,
    ) -> MechanicalResolution:
        """
        Execute the mechanical resolution: roll dice and determine outcome.
        """
        # Get character's modifier for this check
        modifier = self._get_character_modifier(player_name, triage.ability, triage.skill)

        # Build dice notation
        notation = f"1d20{modifier:+d}" if modifier != 0 else "1d20"

        # Roll the dice
        roll_result = self.roller.roll(
            notation=notation,
            advantage=triage.advantage,
            disadvantage=triage.disadvantage,
            reason=f"{triage.skill or triage.ability} check" if triage.skill or triage.ability else "check",
        )

        # Determine success
        dc = triage.dc or 10
        success = roll_result.total >= dc
        margin = roll_result.total - dc

        # Get appropriate reveals based on success/failure
        reveals = triage.on_success if success else triage.on_failure

        logger.info(
            "mechanics_resolved",
            roll=roll_result.total,
            dc=dc,
            success=success,
            skill=triage.skill,
            reveals=reveals,
        )

        return MechanicalResolution(
            success=success,
            roll_result=roll_result,
            total=roll_result.total,
            dc=dc,
            margin=margin,
            reveals=reveals,
            skill=triage.skill,
            ability=triage.ability,
            roll_type=triage.roll_type,
        )

    def _get_character_modifier(
        self,
        player_name: str,
        ability: Optional[str],
        skill: Optional[str],
    ) -> int:
        """
        Get the character's modifier for a given ability/skill check.
        """
        if not self._current_session:
            return 0

        # Find the character by player name
        character = None
        for player in self._current_session.players.values():
            if player.character and player_name.lower() in player.character.name.lower():
                character = player.character
                break

        if not character:
            return 0

        # Get ability modifier
        ability_mod = 0
        if ability:
            ability_score = getattr(character.abilities, ability, 10)
            ability_mod = (ability_score - 10) // 2

        # Add proficiency if applicable
        proficiency_mod = 0
        if skill and skill in [p.value if hasattr(p, 'value') else p for p in character.skill_proficiencies]:
            proficiency_mod = character.proficiency_bonus
            # Check for expertise
            if skill in [e.value if hasattr(e, 'value') else e for e in character.skill_expertise]:
                proficiency_mod *= 2

        return ability_mod + proficiency_mod

    async def _narrate_action(
        self,
        action: str,
        player_name: str,
        context: BrainContext,
        triage: TriageResult,
    ) -> tuple[str, list[ProposedEffect]]:
        """Narrate a no-mechanics player action (social/exploration/…).

        Builds the NarrationSpec: triage's narrative_direction decorates the
        player_action; the ###INSTRUCTION### system prompt carries per-intent
        guidance plus the rotating style and phase-tone hints. This is the
        only spec with allow_streaming — streamed prose reaches the token
        callback while tool recovery rides the followup leg.

        Returns:
            Tuple of (narrative text, proposed effects)
        """
        # Narrative direction from triage (tells narrator what to describe)
        direction = triage.narrative_direction or "Describe what happens naturally."

        # Style rotation + phase-specific tone
        style_hint = get_style_hint(self._scratchpad_turn)
        from ..game.world_state import PHASE_STYLE_HINTS
        world_state = getattr(self._current_session, 'world_state', None) if self._current_session else None
        phase_hint = PHASE_STYLE_HINTS.get(world_state.phase, "") if world_state else ""

        phase_line = f"**Phase tone:** {phase_hint}\n\n" if phase_hint else "\n"

        # Intent-based dynamic instruction — different guidance per action type
        action_type = triage.action_type
        if action_type == "social":
            intent_guidance = (
                "This is a SOCIAL interaction. NPCs should RESPOND with dialogue, "
                "personality, and their own agenda. They don't just listen — they "
                "react, counter, question, or reveal something."
            )
        elif action_type == "exploration":
            intent_guidance = (
                "This is EXPLORATION. Show what the player discovers. Reveal new "
                "details, paths, or clues. Advance their understanding of the area. "
                "Don't redescribe what's already established."
            )
        elif action_type == "movement":
            intent_guidance = (
                "The player is MOVING. Don't stop at the threshold — narrate them "
                "arriving and what they find in the new location. Describe the "
                "destination, not the journey."
            )
        else:
            intent_guidance = (
                "Show the consequence of the action and how the world responds."
            )

        spec = NarrationSpec(
            action=action,
            player_name=player_name,
            player_action=f"{action}\n\n[NARRATIVE DIRECTION: {direction}]",
            prompt=(
                "###INSTRUCTION###\n"
                "Narrate the player's action according to the NARRATIVE DIRECTION above.\n\n"
                f"{intent_guidance}\n\n"
                f"**Style:** {style_hint}\n"
                f"{phase_line}"
                "Write your narration directly."
            ),
            prompt_role="system",
            allow_streaming=True,
            empty_prose_fallback=f"*{player_name}'s action unfolds...*",
            empty_prose_warn_event="narrator_returned_empty_for_action",
        )

        return await self._narration_strategy.run(spec, context, triage)

    async def _narrate_outcome(
        self,
        action: str,
        player_name: str,
        context: BrainContext,
        triage: TriageResult,
        resolution: Optional[MechanicalResolution],
    ) -> tuple[str, list[ProposedEffect]]:
        """Narrate a dice-roll outcome under authorized-reveal constraints.

        Builds the NarrationSpec: the [RESOLUTION: …] decoration carries the
        roll/DC/margin plus authorized reveals, and the ###INSTRUCTION###
        system prompt restates the resolution with an extremely explicit
        success/failure directive (inspired by the old bot's "CRITICAL DM
        RULE" pattern). The narrator receives ONLY authorized reveals.

        Returns:
            Tuple of (narrative text, proposed effects)
        """
        # Build narrator context with strict constraints
        narrator_context = self._build_narrator_context(
            action, player_name, context, triage, resolution
        )

        is_success = resolution.success if resolution else True
        skill_label = resolution.skill or resolution.ability or "check" if resolution else "action"

        if resolution and not is_success:
            outcome_instruction = (
                f"CRITICAL: This {skill_label} roll FAILED (rolled {resolution.total} vs DC {resolution.dc}).\n\n"
                f"{player_name} DOES NOT achieve their goal. Period.\n"
                "You MUST narrate a failure with real consequences:\n"
                "- They find NOTHING useful, miss the clue, botch the attempt\n"
                "- NPCs react negatively — suspicion, hostility, refusal\n"
                "- Physical consequences — they slip, break something, alert enemies\n"
                "Do NOT describe a soft success or 'almost' result disguised as progress.\n"
                "Show what goes WRONG, then show how the world responds."
            )
        elif resolution and is_success:
            outcome_instruction = (
                f"This {skill_label} roll SUCCEEDED (rolled {resolution.total} vs DC {resolution.dc}).\n\n"
                f"{player_name} achieves their goal. Narrate the positive outcome.\n"
                "Describe what they discover or accomplish — then show how it advances the story."
            )
        else:
            outcome_instruction = "The action proceeds naturally."

        spec = NarrationSpec(
            action=action,
            player_name=player_name,
            player_action=f"{action}\n\n[RESOLUTION: {narrator_context}]",
            prompt=(
                "###INSTRUCTION###\n"
                f"RESOLUTION: {narrator_context}\n\n"
                f"{outcome_instruction}\n\n"
                "AUTHORIZED REVEALS limit what you can describe. Do NOT invent discoveries beyond what is listed.\n\n"
                "Write your narration directly."
            ),
            prompt_role="system",
            empty_prose_fallback=f"*{player_name} attempts to {action.lower()}...*",
            empty_prose_warn_event="narrator_returned_empty",
        )

        return await self._narration_strategy.run(spec, context, triage)

    def _build_narrator_context(
        self,
        action: str,
        player_name: str,
        context: BrainContext,
        triage: TriageResult,
        resolution: Optional[MechanicalResolution],
    ) -> str:
        """Build context string for narrator with resolution constraints."""
        parts = []

        if resolution:
            outcome_word = "SUCCESS" if resolution.success else "FAILURE"
            skill_label = resolution.skill or resolution.ability or triage.action_type
            parts.append(f"{skill_label.title()} Check: {outcome_word}")
            parts.append(f"Rolled {resolution.total} vs DC {resolution.dc} (margin: {resolution.margin:+d})")

            # Degree of success/failure affects narration
            abs_margin = abs(resolution.margin)
            if resolution.success:
                if abs_margin >= 10:
                    parts.append("CRITICAL SUCCESS — reveal everything clearly, with bonus detail")
                elif abs_margin >= 5:
                    parts.append("SOLID SUCCESS — reveal the information clearly")
                else:
                    parts.append("NARROW SUCCESS — reveal the basics, but incompletely")
            else:
                if abs_margin >= 10:
                    parts.append("CRITICAL FAILURE — the character is completely wrong or oblivious, with consequences")
                elif abs_margin >= 5:
                    parts.append("CLEAR FAILURE — the character misses the mark, narrate what they fail to notice or accomplish")
                else:
                    parts.append("NARROW FAILURE — the character almost succeeds but falls short, hint at what they missed")

            if resolution.reveals:
                parts.append(f"AUTHORIZED REVEALS (narrate ONLY these): {', '.join(resolution.reveals)}")

            parts.append("Do NOT invent additional discoveries or outcomes beyond what is listed above.")
        else:
            # No roll needed - free narrative within triage constraints
            if triage.on_success:
                parts.append(f"The action proceeds. Include these details: {', '.join(triage.on_success)}")
            else:
                # Always give narrator something to work with
                parts.append("The action proceeds naturally. Describe what the character experiences.")

        return " | ".join(parts)

    # =========================================================================
    # WORLD STATE EXTRACTION
    # =========================================================================

    async def _extract_and_apply_state_delta(
        self,
        narrative: str,
        world_state: "WorldState",
        context: BrainContext,
        referenced_entity_ids: list[str] | None = None,
        canonical_entity_ids: list[str] | None = None,
        narrator_refs: list[tuple[str, str]] | None = None,
    ) -> Optional["StateDelta"]:
        """Extract a StateDelta from narrator prose and apply to WorldState.

        Also syncs NPC changes back to SceneEntityRegistry so the existing
        entity system stays in sync. Returns the delta for downstream use
        (e.g., knowledge graph bridge).

        Args:
            referenced_entity_ids: Entity IDs the narrator explicitly tagged
                via ref_entity intents. Passed to the extractor so it knows
                which entities are already accounted for and avoids creating
                duplicates.
            canonical_entity_ids: KG entities activated by the player's action
                this turn. Exact-name returning NPCs may reuse these durable
                IDs after scene rescoping removed them from WorldState.
            narrator_refs: Explicit ``(entity_id, prose_alias)`` pairs from
                narrator tools. These are authoritative identity evidence for
                merging extractor-created duplicate NPCs.
        """

        try:
            delta = await self._state_extractor.extract(
                narrative_text=narrative,
                world_state_yaml=context.world_state_yaml,
                current_scene=context.current_scene,
                referenced_entity_ids=referenced_entity_ids,
            )

            merged_refs = _merge_delta_npcs_with_narrator_refs(
                delta,
                world_state,
                narrator_refs or [],
                narrative,
            )
            for previous_id, canonical_id, npc_name in merged_refs:
                logger.info(
                    "state_delta_npc_merged_into_narrator_reference",
                    npc_name=npc_name,
                    previous_id=previous_id,
                    canonical_id=canonical_id,
                )

            kg = getattr(self._current_session, "knowledge_graph", None)
            reanchored = _reanchor_returning_npc_ids(
                delta,
                kg,
                set(world_state.npcs),
                canonical_entity_ids or [],
            )
            for previous_id, canonical_id, npc_name in reanchored:
                logger.info(
                    "returning_npc_reanchored",
                    npc_name=npc_name,
                    previous_id=previous_id,
                    canonical_id=canonical_id,
                )

            governance_rejections = _drop_dead_npc_reintroductions(
                delta,
                self._known_dead_npcs(),
            )
            for rejection in governance_rejections:
                logger.warning(
                    "state_delta_governance_rejection",
                    reason=rejection,
                )

            # Apply through the single-writer store: dedup (the brain judge
            # catching extractor paraphrases like "Old Bram" for a
            # registered "Bram") → validate → write, all inside apply_delta
            # (Step 5).
            rejections = governance_rejections + await WorldStateStore(
                world_state
            ).apply_delta(
                delta,
                narrator_prose=narrative,
                scene_registry=self._scene_registry,
            )
            self._last_state_rejections = list(rejections)

            if rejections:
                logger.debug(
                    "state_delta_applied_with_rejections",
                    turn=world_state.turn,
                    rejections=rejections,
                )

            # Sync new/updated NPCs to SceneEntityRegistry
            if self._scene_registry:
                self._sync_npcs_to_registry(delta, world_state)

            logger.info(
                "world_state_updated",
                turn=world_state.turn,
                location=world_state.current_location,
                time=world_state.time_of_day,
                npc_count=len(world_state.npcs),
                facts_count=len(world_state.established_facts),
            )

            return delta

        except Exception as e:
            logger.warning("state_delta_extraction_failed", error=str(e), exc_info=True)
            return None

    def _sync_npcs_to_registry(
        self,
        delta: "StateDelta",
        world_state: "WorldState",
    ) -> None:
        """Sync NPC changes from WorldState back to SceneEntityRegistry.

        Keeps the existing entity system (hostility, SRD matching, combat triggers)
        working alongside WorldState.
        """
        from ..models.npc import SceneEntity, EntityType, Disposition

        if not self._scene_registry:
            return

        disposition_map = {
            "hostile": Disposition.HOSTILE,
            "unfriendly": Disposition.UNFRIENDLY,
            "neutral": Disposition.NEUTRAL,
            "friendly": Disposition.FRIENDLY,
            "allied": Disposition.ALLIED,
        }

        # Register new NPCs
        for npc_state in delta.new_npcs:
            # Only add to registry if they're at the party's location
            if not npc_state.location or npc_state.location == world_state.current_location:
                entity = SceneEntity(
                    name=npc_state.name,
                    entity_type=EntityType.NPC,
                    description=npc_state.description,
                    disposition=disposition_map.get(npc_state.disposition, Disposition.NEUTRAL),
                    npc_id=npc_state.id,  # Stage C (DF-29): stamp the canonical id
                )
                self._scene_registry.register_entity(entity)

        # Update existing NPCs from the extractor's npc_updates.
        for update in delta.npc_updates:
            key = update.id or update.name
            if not key:
                continue
            existing = self._scene_registry.get_by_name(key)
            if existing is None:
                continue
            if update.disposition:
                new_disp = disposition_map.get(update.disposition)
                if new_disp:
                    existing.disposition = new_disp
            # Extractor-channel death (Stage-C DF-4): the extractor reports a
            # kill as NPCUpdate(alive=False), which reached WorldState + the
            # KG but never the SceneEntity — so sync_to_npc_repo left the DB
            # row alive and the NPC resurrected on the next fresh session.
            # Mark the entity dead here, mirroring the tool path's
            # _execute_update_entity status write, so death reaches the DB.
            if update.alive is False:
                existing.status = "dead"

        # Remove NPCs who left
        for name in delta.removed_npcs:
            self._scene_registry.remove_by_name(name)

    # =========================================================================
    # ENTITY EXTRACTION AND COMBAT TRIGGERING
    # =========================================================================

    async def _process_narrator_output(
        self,
        narrative: str,
        context: BrainContext,
        proposed_effects: Optional[list[ProposedEffect]] = None,
        message_id: Optional[str] = None,
        skip_entity_extraction: bool = False,
    ) -> bool:
        """
        Process narrator output: extract entities and execute proposed effects.

        Flow:
        1. Entity extraction → Scene registry (NPCs, creatures mentioned in prose)
        2. Proposed effects → Validate and execute (structured, not sniffed from prose)

        Returns True if combat was triggered.
        """
        combat_triggered = False

        # Entity extraction — skip during combat or simple skill checks to save API calls
        if self._scene_registry and not skip_entity_extraction:
            existing = [e.name for e in self._scene_registry.get_all()]
            try:
                entity_result = await self._entity_extractor.extract(
                    narrative_text=narrative,
                    current_scene=context.current_scene or "",
                    existing_entities=existing,
                )

                if entity_result:
                    # Register new entities
                    for extracted in entity_result.entities:
                        entity = self._entity_extractor.convert_to_scene_entity(extracted)
                        self._scene_registry.register_entity(entity)

                    # Update scene description if provided
                    if entity_result.scene_update:
                        self._scene_registry.set_scene_description(entity_result.scene_update)

                    # Combat trigger: LLM determined combat actually started
                    if entity_result.combat_initiated:
                        hostile_entities = gather_scene_hostiles(self._scene_registry)
                        if hostile_entities:
                            logger.warning(
                                "combat_initiated_by_narrative",
                                hostile_count=len(hostile_entities),
                                hostiles=[e.name for e in hostile_entities],
                            )
                            combat_triggered = start_encounter(
                                self._current_session, hostile_entities
                            )

                    # Persist entities to DB immediately (not just at session end)
                    try:
                        await self._scene_registry.sync_to_npc_repo()
                    except Exception as sync_err:
                        logger.warning("entity_sync_failed", error=str(sync_err), exc_info=True)

            except Exception as e:
                logger.warning("entity_extraction_failed", error=str(e), exc_info=True)

        # Process proposed effects (replaces mechanics extraction)
        if proposed_effects:
            effect_combat = await self._process_proposed_effects(
                proposed_effects,
                context,
                message_id,
            )
            combat_triggered = combat_triggered or effect_combat

        return combat_triggered

    async def _process_proposed_effects(
        self,
        effects: list[ProposedEffect],
        context: BrainContext,
        message_id: Optional[str] = None,
    ) -> bool:
        """
        Validate and execute proposed effects from narrator.

        This is the core of the "Narrator proposes, Rules validates" pattern.
        Returns True if any effect triggered combat.
        """
        self._last_executed_effects = []
        self._last_effect_executions = []
        self._last_effect_rejections = []

        if not effects:
            return False

        # Lazy init executor with inventory repo
        if not self._effect_executor:
            inventory_repo = await get_inventory_repo()
            self._effect_executor = EffectExecutor(
                scene_registry=self._scene_registry,
                session=self._current_session,
                inventory_repo=inventory_repo,
                applied_effects_store=self._applied_effects,
            )

        # Thread the acting player so update_player effects (audit #1) target the
        # right PC instead of guessing. Set every turn — the executor is reused.
        acting = self._resolve_character_by_name(context.player_name) if context.player_name else None
        self._effect_executor.acting_character_id = acting[0] if acting else None

        # Ensure validator exists
        if not self._effect_validator:
            self._effect_validator = EffectValidator(
                scene_registry=self._scene_registry,
                session=self._current_session,
            )

        combat_triggered = False
        campaign_id = context.campaign_id or "unknown"
        msg_id = message_id or str(uuid.uuid4())

        # Single-writer seam (Step 4): successfully executed effects sync
        # into WorldState only through the session's store.
        world_store = (
            getattr(self._current_session, "world_store", None)
            if self._current_session else None
        )

        narrator_prose = getattr(self, "_last_narrator_prose", "") or ""

        for i, effect in enumerate(effects):
            # Build idempotency key BEFORE any rewrite, so retries hit the
            # same key whether the original effect was add_npc or its
            # rewritten ref_entity.
            idem_key = build_effect_idempotency_key(campaign_id, msg_id, i)

            # No confirmation UI exists (the flag is an INTENTS-only knob;
            # the tool path never sets it). Skipping here silently discarded
            # the effect while the narration claimed it happened — so parity
            # with the tool path: execute like any other effect, warning that
            # confirmation was requested but auto-approved.
            if effect.requires_confirmation:
                logger.warning(
                    "effect_confirmation_auto_approved",
                    effect_type=effect.effect_type.value,
                    prompt=effect.confirmation_prompt,
                )

            # Dedup judge — a store-owned write-pipeline step (Step 5):
            # ADD_NPC effects that look like a paraphrase of a roster
            # entity are rewritten to REF_ENTITY BEFORE validation and
            # execution, so the rewritten effect is what executes. The
            # judge defaults to ACCEPT on any uncertainty (false negatives
            # are recoverable; false positives merge distinct characters
            # and are not). Idempotency keys are indexed by tool-call
            # position, so retry safety survives the rewrite.
            if world_store is not None:
                effect = await world_store.dedup_effect(
                    effect, narrator_prose, scene_registry=self._scene_registry,
                )

            # Validate
            validation = self._effect_validator.validate(effect)
            if not validation.valid:
                self._last_effect_rejections.append({
                    **_effect_telemetry_record(effect),
                    "stage": "validation",
                    "reason": validation.rejection_reason,
                })
                logger.warning(
                    "effect_validation_failed",
                    effect_type=effect.effect_type.value,
                    reason=validation.rejection_reason,
                )
                continue

            # Execute
            result = await self._effect_executor.execute(effect, idem_key)

            if result.success:
                self._last_effect_executions.append({
                    **_effect_telemetry_record(effect),
                    "was_duplicate": bool(result.was_duplicate),
                    "details": dict(result.details or {}),
                })
                logger.info(
                    "effect_applied",
                    effect_type=effect.effect_type.value,
                    details=result.details,
                    was_duplicate=result.was_duplicate,
                )

                # DF-12: an idempotency hit means this exact effect already
                # applied once (retry within the same turn). Re-running the
                # side-effects would double-apply to WorldState/KG.
                if not result.was_duplicate:
                    # Sync effect to WorldState (so narrator sees it next turn)
                    if world_store is not None:
                        world_store.apply_effect(effect)
                    self._last_executed_effects.append(effect)

                    # Combat-entry signal: route through the single decision
                    # point (Step 3). Previously this only set the flag — no
                    # encounter, no participants — so the session flipped to
                    # COMBAT with no CombatManager (audit: "three live
                    # combat-entry deciders"). Now the narrator's signal drafts
                    # the scene hostiles like the extractor path does, and an
                    # empty scene refuses to trigger at all.
                    if effect.effect_type == EffectType.START_COMBAT:
                        hostiles = (
                            gather_scene_hostiles(self._scene_registry)
                            if self._scene_registry else []
                        )
                        if hostiles:
                            combat_triggered = (
                                start_encounter(self._current_session, hostiles)
                                or combat_triggered
                            )
                        else:
                            logger.warning(
                                "start_combat_signal_without_hostiles",
                                reason=effect.reason,
                            )
            else:
                self._last_effect_rejections.append({
                    **_effect_telemetry_record(effect),
                    "stage": "execution",
                    "reason": result.error,
                })
                logger.warning(
                    "effect_execution_failed",
                    effect_type=effect.effect_type.value,
                    error=result.error,
                )

        return combat_triggered

    @staticmethod
    def _entity_name_match_score(entity: "SceneEntity", text_lower: str) -> int:
        """Score how well an entity name matches player text. Higher = better. 0 = no match."""
        name_lower = entity.name.lower()

        # Exact full name match
        if name_lower in text_lower:
            return 100

        # Check aliases
        for alias in (entity.aliases or []):
            if alias.lower() in text_lower:
                return 90

        # Check if any significant word from entity name appears in text
        # Skip common words like "the", "a", "of"
        skip_words = {"the", "a", "an", "of", "at", "in", "on", "to", "and", "or"}
        entity_words = [w for w in name_lower.split() if w not in skip_words and len(w) > 2]
        text_words = set(text_lower.split())

        matching_words = [w for w in entity_words if w in text_words]
        if matching_words:
            # Score by fraction of significant entity words that matched
            return int(60 * len(matching_words) / len(entity_words)) if entity_words else 0

        # Check if any text word is a substring of entity name (e.g., "leader" in "assassin leader")
        for word in text_words:
            if len(word) > 3 and word in name_lower:
                return 40

        return 0

    async def _initiate_combat_from_attack(
        self,
        target_name: Optional[str],
        player_name: str,
    ) -> bool:
        """
        Try to initiate combat when player attacks a creature.

        Looks up the target in the scene registry and triggers combat
        if found. Returns True if combat was started, False if target
        not found (caller should let narrator handle it).

        Args:
            target_name: Name of the creature being attacked
            player_name: Name of the attacking player (for logging)
        """
        if not self._scene_registry or not target_name:
            return False

        # Try to find the target entity - exact first, then fuzzy
        target_entity = self._scene_registry.get_by_name(target_name)

        if not target_entity:
            # Fuzzy match using scoring — require minimum threshold of 30
            # to avoid grabbing the wrong entity on weak partial matches
            target_lower = target_name.lower()
            best_score = 0
            best_candidate = None
            for entity in self._scene_registry.get_all_entities():
                score = self._entity_name_match_score(entity, target_lower)
                if score > best_score:
                    best_score = score
                    best_candidate = entity

            if best_candidate and best_score >= 30:
                target_entity = best_candidate
            elif best_candidate:
                logger.debug(
                    "fuzzy_match_below_threshold",
                    target=target_name,
                    best_match=best_candidate.name,
                    score=best_score,
                )

        if not target_entity:
            logger.debug(
                "attack_target_not_found_in_registry",
                target=target_name,
                player=player_name,
            )
            return False

        # Gather hostile entities: target + other actively hostile NPCs/creatures
        hostile_entities = [target_entity]
        for entity in self._scene_registry.get_all_entities():
            if entity.id == target_entity.id:
                continue
            if entity.entity_type not in (EntityType.NPC, EntityType.CREATURE):
                continue
            if entity.disposition == Disposition.HOSTILE:
                hostile_entities.append(entity)

        logger.info(
            "initiating_combat_from_player_attack",
            target=target_entity.name,
            player=player_name,
            hostile_count=len(hostile_entities),
        )

        # Trigger combat with surprise (player initiated); the encounter
        # builder pushes combat mode itself (Step 3 single entry point).
        return start_encounter(
            self._current_session, hostile_entities, player_initiated=True
        )

    def _resolve_character_by_name(self, name: str) -> Optional[tuple[str, "Character"]]:
        """Resolve a character by name within the current session.

        Uses ranked matching: exact match > starts-with > substring.
        Only returns a result if the match is unambiguous.
        """
        if not self._current_session:
            return None

        name_lower = name.lower().strip()
        candidates = []

        for player in self._current_session.players.values():
            if not player.character:
                continue
            char_name = player.character.name.lower()

            if char_name == name_lower:
                # Exact match — return immediately
                return (player.character.id, player.character)

            if char_name.startswith(name_lower):
                candidates.append((0, player.character))  # Priority 0: starts-with
            elif name_lower in char_name:
                candidates.append((1, player.character))  # Priority 1: substring

        if not candidates:
            return None

        # Sort by priority, return best match only if unambiguous at that tier
        candidates.sort(key=lambda x: x[0])
        best_priority = candidates[0][0]
        best_tier = [c for c in candidates if c[0] == best_priority]

        if len(best_tier) == 1:
            char = best_tier[0][1]
            return (char.id, char)

        # Ambiguous match — log and return None
        logger.warning(
            "ambiguous_character_match",
            query=name,
            matches=[c[1].name for c in best_tier],
        )
        return None


    # The direct-write commerce executors (_execute_purchase_item /
    # _execute_add_item) are gone: purchase and pickup now claim their
    # player-state writes as authoritative effects, so the effect pipeline
    # is the one receipted writer for those turns.


# Global orchestrator instance
_orchestrator: Optional[DMOrchestrator] = None


def get_orchestrator() -> DMOrchestrator:
    """Get the global DM orchestrator."""
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = DMOrchestrator()
    return _orchestrator


def _reset_orchestrator() -> None:
    """Clear cached orchestrator so it recreates from the active profile."""
    global _orchestrator
    _orchestrator = None
