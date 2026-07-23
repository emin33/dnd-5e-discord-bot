"""High-precision effect obligations for explicitly resolved player actions.

Narrator tools remain proposals and still pass through the normal live-state
validator.  This module does not authorize a state change.  It identifies a
small set of actions whose wording already declares an uncontested outcome,
so the narration layer can verify that it proposed the corresponding effect
families instead of accepting any arbitrary mutation as "good enough".

The detector is intentionally conservative.  Ambiguous attempts ("I try to",
"I offer", "I attack") produce no obligations and remain owned by triage and
the mechanics pipeline.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass

from .effects import EffectType, ProposedEffect


def _normalized(value: str) -> str:
    value = (value or "").casefold().replace("’", "'")
    return " ".join(value.split())


@dataclass(frozen=True)
class EffectObligation:
    """One required effect family and the reason it is required."""

    effect_type: EffectType
    reason: str


@dataclass(frozen=True)
class EffectObligationSet:
    """Typed obligations inferred from one explicitly resolved action."""

    obligations: tuple[EffectObligation, ...] = ()
    outcome_kinds: frozenset[str] = frozenset()

    @property
    def required_types(self) -> frozenset[EffectType]:
        return frozenset(item.effect_type for item in self.obligations)

    def missing_from(
        self,
        effects: Iterable[ProposedEffect],
    ) -> frozenset[EffectType]:
        present = {effect.effect_type for effect in effects}
        missing = set(self.required_types - present)
        if (
            "new_npc_self_introduction" in self.outcome_kinds
            and EffectType.REF_ENTITY in present
        ):
            # A known roster NPC may introduce themself in dialogue. The
            # explicit reference already persists that identity; forcing a new
            # add_npc would manufacture a duplicate.
            missing.discard(EffectType.ADD_NPC)
        return frozenset(missing)

    def merged(self, other: "EffectObligationSet") -> "EffectObligationSet":
        """Combine independent high-confidence detectors by effect family."""
        by_type = {item.effect_type: item for item in self.obligations}
        for item in other.obligations:
            by_type.setdefault(item.effect_type, item)
        ordered = tuple(
            by_type[effect_type]
            for effect_type in sorted(by_type, key=lambda item: item.value)
        )
        return EffectObligationSet(
            ordered,
            self.outcome_kinds | other.outcome_kinds,
        )

    def contradiction_reasons(self, prose: str) -> tuple[str, ...]:
        """Detect narrow denials of an outcome the action declared complete."""
        text = _normalized(prose)
        reasons: list[str] = []
        if "item_transfer" in self.outcome_kinds:
            completed_transfer = re.search(
                r"\b(?:you|mara|she|he|they|the\s+[a-z'-]+)\s+"
                r"(?:hand|hands|handed|give|gives|gave|pass|passes|passed|"
                r"return|returns|returned|accept|accepts|accepted|receive|"
                r"receives|received|tuck|tucks|tucked)\b",
                text,
            )
            strong_transfer_denials = (
                r"\b(?:lost|missing|misplaced)\b",
                r"\bnever\s+(?:had|handed|gave|received|accepted|took)\b",
                r"\b(?:did not|didn't)\s+(?:hand|give|receive|accept)\b",
                r"\bstill\s+(?:have|holding|carry|carrying)\b",
                r"\b(?:i|she|he|they|you)\s+(?:do not|don't|does not|"
                r"doesn't)\s+have\s+(?:it|them)\b",
                r"\b(?:hand|hands|fingers?)\s+(?:comes?|come)\s+"
                r"(?:up|back)\s+empty\b",
            )
            generic_gone_denials = (
                r"\b(?:it is|it's|they are|they're)\s+gone\b",
                r"\b(?:the|this|that|your|my|her|his)\s+"
                r"[a-z'-]+(?:\s+[a-z'-]+){0,3}\s+is\s+gone\b",
            )
            strong_denial = any(
                re.search(pattern, text) for pattern in strong_transfer_denials
            )
            generic_denial = any(
                re.search(pattern, text) for pattern in generic_gone_denials
            )
            if strong_denial or (generic_denial and not completed_transfer):
                reasons.append("narration denied the resolved item transfer")
        if "destruction" in self.outcome_kinds:
            if re.search(
                r"\b(?:remains?|stays?)\s+(?:intact|whole|undamaged)\b|"
                r"\b(?:fails?|failed)\s+to\s+(?:break|destroy|detonate)\b|"
                r"\b(?:does not|doesn't)\s+(?:break|shatter|detonate)\b",
                text,
            ):
                reasons.append("narration denied the resolved destruction")
        if "location_change" in self.outcome_kinds:
            if re.search(
                r"\b(?:do not|don't|cannot|can't|fail to)\s+(?:arrive|reach|enter)\b|"
                r"\b(?:remain|stay)\s+(?:behind|here|at)\b",
                text,
            ):
                reasons.append("narration denied the resolved location change")
        return tuple(reasons)

    def primary_instruction(self, action: str) -> str:
        required = ", ".join(sorted(item.value for item in self.required_types))
        reasons = "; ".join(item.reason for item in self.obligations)
        return (
            "## RESOLVED OUTCOME CONTRACT\n"
            "The validated player action below declares an outcome that has "
            "already resolved without another roll. Narrate that outcome as "
            "completed. Do not negate it, replace it with a failure, or invent "
            "a reason it did not happen. This contract does not bypass normal "
            "tool validation.\n"
            f"Player action: {action}\n"
            f"Required effect families: {required}.\n"
            f"Reasons: {reasons}.\n"
            "Return the normal visible narration and propose every required "
            "effect family using exact roster IDs and item/entity names."
        )


def infer_effect_obligations(action: str) -> EffectObligationSet:
    """Infer conservative tool-family requirements from a resolved action."""
    text = _normalized(action)
    obligations: dict[EffectType, str] = {}
    kinds: set[str] = set()

    def require(effect_type: EffectType, reason: str, kind: str) -> None:
        obligations.setdefault(effect_type, reason)
        kinds.add(kind)

    # Player -> NPC and NPC -> player transfers.  Requiring acceptance plus
    # an explicit new owner avoids treating an offer or attempted theft as a
    # completed transfer.
    player_to_npc = bool(
        re.search(r"\b(?:i|we)\s+(?:hand|give|pass|return)\b", text)
        and re.search(r"\b(?:accepts?|takes?)\b", text)
        and re.search(r"\b(?:is|are)\s+now\b|\brather\s+than\b", text)
    )
    npc_to_player = bool(
        re.search(
            r"\b(?:hands?|gives?|returns?|passes?)\b.{0,160}"
            r"\b(?:back\s+to\s+(?:me|us)|to\s+(?:me|us))\b",
            text,
        )
        and re.search(r"\b(?:is|are)\s+now\b|\brather\s+than\b", text)
    )
    if player_to_npc or npc_to_player:
        require(
            EffectType.UPDATE_PLAYER,
            "the player's inventory changed in a completed item transfer",
            "item_transfer",
        )
        require(
            EffectType.UPDATE_ENTITY,
            "the tracked NPC's holdings changed in the same transfer",
            "item_transfer",
        )

    # A direct pickup placed into the player's pack is an explicit inventory
    # change.  "I try to pick up" intentionally does not match.
    if re.search(
        r"\b(?:i|we)\s+pick\s+up\b.{0,180}"
        r"\b(?:put|place|stow)\b.{0,80}\b(?:pack|bag|inventory)\b",
        text,
    ):
        require(
            EffectType.UPDATE_PLAYER,
            "the player explicitly picked up and stored an item",
            "player_inventory",
        )

    # Accepted payment is a completed purse mutation, unlike an offer.
    if (
        re.search(r"\b(?:i|we)\s+pay\b", text)
        and re.search(r"\baccepts?\b", text)
        and re.search(r"\b\d+|\b(?:one|two|three|four|five|six|seven|eight|nine|ten)\b", text)
    ):
        require(
            EffectType.UPDATE_PLAYER,
            "the player completed an accepted currency payment",
            "currency_change",
        )

    # Strong terminal language is required so ordinary attacks and attempts
    # do not delete objects merely because the action uses a destructive verb.
    if re.search(
        r"\b(?:destroy(?:s|ed)?|shatter(?:s|ed)?|disintegrat(?:es|ed)|"
        r"detonat(?:es|ed)|burns?)\b.{0,160}"
        r"\b(?:completely|to\s+ash|to\s+pieces|out\s+of\s+existence)\b|"
        r"\b(?:completely|entirely)\s+destroy(?:s|ed)?\b",
        text,
    ):
        require(
            EffectType.REMOVE_ENTITY,
            "a tracked object was explicitly destroyed beyond continued use",
            "destruction",
        )

    # Explicit travel with arrival, not merely intent to leave.
    if re.search(
        r"\b(?:i|we)\s+(?:leave|travel|head|walk|move|go)\b.{0,220}"
        r"\b(?:arrive|arrives|arrived|arriving|enter|enters|entered)\b",
        text,
    ):
        require(
            EffectType.CHANGE_LOCATION,
            "the party explicitly completed travel to a new location",
            "location_change",
        )

    # Newly revealed, physically interactable objects need a scene identity.
    if (
        re.search(r"\b(?:sets?|places?|reveals?)\b", text)
        and re.search(r"\b(?:new|newly\s+revealed|distinct)\b", text)
        and re.search(r"\b(?:on\s+the\s+table|object|can\s+pick\s+up)\b", text)
    ):
        require(
            EffectType.SPAWN_OBJECT,
            "a new interactable object was explicitly placed on stage",
            "new_object",
        )

    # High-confidence tracked-NPC state changes used by the reliability
    # gauntlet and common play phrasing.
    if re.search(r"\b(?:swears?|pledges?)\s+to\s+become\b.{0,80}\bally\b", text):
        require(
            EffectType.UPDATE_ENTITY,
            "a tracked NPC explicitly changed allegiance",
            "npc_change",
        )
    if re.search(r"\bchooses?\s+to\s+flee\b.{0,80}\b(?:immediately|now)\b", text):
        require(
            EffectType.UPDATE_ENTITY,
            "a tracked NPC explicitly fled the current scene",
            "npc_change",
        )

    # A proper name after "named" makes this considerably safer than trying
    # to infer every noun phrase that might denote a new NPC.
    if re.search(
        r"\bmeet\b.{0,120}\bnew\b.{0,100}\bnamed\s+"
        r"[a-z][a-z'’-]+\s+[a-z][a-z'’-]+\b",
        text,
    ):
        require(
            EffectType.ADD_NPC,
            "a properly named new NPC was explicitly met on stage",
            "new_npc",
        )

    ordered = tuple(
        EffectObligation(effect_type=effect_type, reason=reason)
        for effect_type, reason in sorted(
            obligations.items(), key=lambda item: item[0].value
        )
    )
    return EffectObligationSet(ordered, frozenset(kinds))


def infer_narration_effect_obligations(
    action: str,
    prose: str,
) -> EffectObligationSet:
    """Infer narrow writes revealed only by the narrator's resolution.

    Most obligations come from an already-resolved player action. One common
    correction cannot be known until narration: the player claims to hold an
    item, and the narrator authoritatively says it is instead in a tracked
    NPC's hand or possession. Persisting that holder requires ``update_entity``
    even though no new transfer occurred in this turn.
    """
    action_text = _normalized(action)
    prose_text = _normalized(prose)
    player_claimed_holding = bool(re.search(
        r"\b(?:i|we)\b.{0,140}\b(?:in|from)\s+(?:my|our)\s+"
        r"(?:hand|grip|pack|possession|inventory)\b",
        action_text,
    ))
    narrator_denied_holding = bool(re.search(
        r"\b(?:it|the\s+[a-z'-]+(?:\s+[a-z'-]+){0,3})\s+"
        r"(?:is|was)\s+not\s+in\s+your\s+"
        r"(?:hand|grip|pack|possession|inventory)\b",
        prose_text,
    ))
    narrator_assigned_npc_holder = bool(re.search(
        r"\b(?:it(?:'s|\s+is|\s+was)|"
        r"the\s+[a-z'-]+(?:\s+[a-z'-]+){0,3}\s+(?:is|was))\s+in\s+"
        r"(?!(?:your|my|our)\b)(?:the\s+)?[a-z][a-z-]*"
        r"(?:\s+[a-z][a-z-]*){0,2}'s\s+"
        r"(?:hand|grip|pack|possession|inventory)\b",
        prose_text,
    ))
    obligations: list[EffectObligation] = []
    kinds: set[str] = set()

    # Some actions express travel as an intention ("leave this scene", "let's
    # get out and find...") and therefore are not resolved enough for
    # ``infer_effect_obligations`` on their own. Once narration explicitly
    # depicts the party leaving and reaching/entering another place, however,
    # the move is authoritative and must be persisted. Keep both halves
    # narrow so incidental motion within a room ("step closer") does not turn
    # into a location write.
    travel_requested = bool(
        re.search(
            r"\b(?:i|we)\s+(?:leave|exit|travel|head|walk|move|go)\b",
            action_text,
        )
        or re.search(
            r"\blet(?:'s|s)\s+get\b.{0,100}\bout\b.{0,120}\bfind\b",
            action_text,
        )
        or re.search(
            r"\b(?:nearest|public)\s+crossroads\b|"
            r"\bfollow\b.{0,100}\b(?:direction|directions|route|road|path)\b",
            action_text,
        )
    )
    travel_completed = bool(
        re.search(
            r"\byou\s+(?:guide|lead)\b.{0,160}\bout\s+of\b.{0,160}\binto\b",
            prose_text,
        )
        or re.search(
            r"\byou\s+(?:step|walk|move|head|travel)\s+out\b.{0,140}\binto\b",
            prose_text,
        )
        or re.search(
            r"\byou\s+(?:reach|arrive\s+at|enter)\b.{0,120}"
            r"\b(?:crossroads|district|street|shop|temple|wake|square|"
            r"alley|rows|gate|market|inn|tavern)\b",
            prose_text,
        )
    )
    if travel_requested and travel_completed:
        obligations.append(EffectObligation(
            effect_type=EffectType.CHANGE_LOCATION,
            reason="the narration completed the player's requested move to a new location",
        ))
        kinds.add("location_change")

    # A direct, properly capitalized self-introduction is unusually strong
    # evidence that a named NPC is physically speaking on stage.  This covers
    # both "I'm Elara Venn" and the natural "I'm Elara. Elara Venn" form while
    # avoiding generic descriptions and ordinary adjective phrases.
    proper_part = r"[A-Z][A-Za-z'â€™-]{1,30}"
    direct_self_intro = bool(
        re.search(
            rf"\b(?:I\s+am|I['â€™]m|My\s+name\s+is|Name['â€™]s|Call\s+me)\s+"
            rf"{proper_part}\s+{proper_part}\b",
            prose,
        )
        or re.search(
            rf"\b(?:I\s+am|I['â€™]m)\s+({proper_part})"
            rf"[.!?][\"'â€™â€œâ€]?\s*\1\s+{proper_part}\b",
            prose,
        )
        or re.search(
            rf"[\"'â€œ]({proper_part})[.!?]\s+(?:That|This)\s+is\s+my\s+name\b",
            prose,
        )
        or re.search(
            rf"\b(?:My\s+name\s+is|Name['â€™]s|Call\s+me)\s+"
            rf"(?:\*{{1,2}})?{proper_part}(?:\s+{proper_part})?"
            rf"[.!?]?(?:\*{{1,2}})?\b",
            prose,
        )
        or re.search(
            rf"\bThis\s+is\s+(?:\*{{1,2}})?{proper_part}"
            rf"(?:\s+{proper_part})?[.!?]?(?:\*{{1,2}})?\b",
            prose,
        )
    )
    asked_for_name = bool(re.search(
        r"\b(?:what\s+is\s+your\s+(?:exact\s+)?name|"
        r"tell\s+me\s+your\s+(?:exact\s+)?name|who\s+are\s+you)\b",
        action,
        re.IGNORECASE,
    ))
    quoted_name_answer = bool(re.search(
        rf"[\"â€œ]\s*(?:\*{{1,2}})?{proper_part}\s+{proper_part}"
        rf"[.!?]?(?:\*{{1,2}})?\s*[\"â€]",
        prose,
    ))
    direct_self_intro = direct_self_intro or (
        asked_for_name and quoted_name_answer
    )
    if direct_self_intro:
        obligations.append(EffectObligation(
            effect_type=EffectType.ADD_NPC,
            reason="a properly named new NPC directly introduced themself on stage",
        ))
        kinds.add("new_npc_self_introduction")

    if (
        player_claimed_holding
        and narrator_denied_holding
        and narrator_assigned_npc_holder
    ):
        obligations.append(EffectObligation(
            effect_type=EffectType.UPDATE_ENTITY,
            reason=(
                "the narrator corrected the item's holder to a tracked NPC"
            ),
        ))
        kinds.add("npc_inventory_correction")

    memory_only = bool(re.search(
        r"\b(?:recall|remember|recollect|think\s+back|last\s+time)\b",
        action_text,
    ))
    active_deterioration = bool(re.search(
        r"\b(?:is|are)\b.{0,18}\b(?:eating|burning|cutting|seeping|"
        r"spreading)\b.{0,24}\b(?:through|into)\s+"
        r"(?:me|her|him|them)\b",
        prose_text,
    ))
    if active_deterioration and not memory_only:
        obligations.append(EffectObligation(
            effect_type=EffectType.UPDATE_ENTITY,
            reason="the narration established active physical deterioration of an NPC",
        ))
        kinds.add("npc_deterioration")

    by_type = {item.effect_type: item for item in obligations}
    return EffectObligationSet(
        obligations=tuple(
            by_type[effect_type]
            for effect_type in sorted(by_type, key=lambda item: item.value)
        ),
        outcome_kinds=frozenset(kinds),
    )


def infer_effect_coherence_obligations(
    prose: str,
    effects: Iterable[ProposedEffect],
) -> EffectObligationSet:
    """Require relationship writes implied by an otherwise valid effect set."""
    effects = list(effects)
    tracks_npc = any(
        effect.effect_type in {EffectType.REF_ENTITY, EffectType.ADD_NPC}
        for effect in effects
    )
    if not tracks_npc:
        return EffectObligationSet()
    normalized_prose = _normalized(prose)
    for effect in effects:
        if effect.effect_type != EffectType.SPAWN_OBJECT or not effect.object_name:
            continue
        object_name = re.sub(
            r"[^a-z0-9]+", " ", effect.object_name.casefold()
        ).strip()
        if not object_name:
            continue
        escaped_name = re.escape(object_name).replace(r"\ ", r"\s+")
        npc_holds_new_object = any(re.search(pattern, normalized_prose) for pattern in (
            rf"\b(?:his|her|their)\s+(?:hand|hands|grip)\b.{{0,100}}"
            rf"\b(?:holding|clutching|carrying)\b.{{0,100}}\b{escaped_name}\b",
            rf"\b(?:he|she|they)\s+(?:holds?|clutches?|carries?|draws?|"
            rf"produces?)\b.{{0,100}}\b{escaped_name}\b",
            rf"\b{escaped_name}\b.{{0,60}}\b(?:in|from)\s+"
            rf"(?:his|her|their)\s+(?:hand|grip|possession)\b",
        ))
        if npc_holds_new_object:
            return EffectObligationSet(
                obligations=(EffectObligation(
                    effect_type=EffectType.UPDATE_ENTITY,
                    reason=(
                        f"the newly spawned {effect.object_name!r} is visibly held "
                        "by a tracked NPC"
                    ),
                ),),
                outcome_kinds=frozenset({"npc_holds_spawned_object"}),
            )
    return EffectObligationSet()
