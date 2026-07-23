"""Deterministic governance for immutable narrative facts.

LLMs may propose prose, but they cannot silently rewrite authoritative world
state. This provider-independent layer finds contradictions that must be
repaired before prose reaches the player.
"""

from __future__ import annotations

import re
from collections.abc import Iterable
from dataclasses import dataclass
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from ..game.world_state import NPCState, WorldState


@dataclass(frozen=True)
class ContinuityViolation:
    """A machine-detected contradiction with authoritative campaign state."""

    rule: str
    entity_id: str
    entity_name: str
    reason: str
    excerpt: str

    def to_prompt_line(self) -> str:
        return f"{self.entity_name}: {self.reason} Draft excerpt: {self.excerpt!r}"


@dataclass(frozen=True)
class _DeadEntity:
    entity_id: str
    name: str
    aliases: tuple[str, ...]


# Concrete non-living frames. A generic word such as "once" would
# incorrectly excuse "Bram walks through the door once more."
_NONLIVING_FRAME = re.compile(
    r"\b(?:"
    r"corpse|body|remains|dead|death|died|deceased|lifeless|slain|grave|tomb|"
    r"memory|memories|remember(?:s|ed|ing)?|recall(?:s|ed|ing)?|flashback|"
    r"dream|vision|illusion|hallucination|portrait|statue|recording|"
    r"ghost|spirit|spect(?:er|re)|apparition|phantom|shade|echo|undead|"
    r"last words|used to|had (?:said|told|warned|asked|promised)|the late"
    r")\b",
    re.IGNORECASE,
)

_ACTING_VERB = (
    r"(?:"
    r"says?|said|asks?|asked|answers?|answered|repl(?:y|ies|ied)|"
    r"whispers?|whispered|shouts?|shouted|calls?|called|speaks?|spoke|"
    r"warns?|warned|tells?|told|laughs?|laughed|smiles?|smiled|"
    r"nods?|nodded|breathes?|breathed|looks?|looked|watches?|watched|"
    r"walks?|walked|runs?|ran|steps?|stepped|enters?|entered|arrives?|arrived|"
    r"approaches?|approached|follows?|followed|turns?|turned|moves?|moved|"
    r"stands?|stood|sits?|sat|rises?|rose|reaches?|reached|grabs?|grabbed|"
    r"takes?|took|gives?|gave|opens?|opened|closes?|closed|attacks?|attacked|"
    r"strikes?|struck|draws?|drew|points?|pointed|waits?|waited|"
    r"appears?|appeared|emerges?|emerged|gestures?|gestured"
    r")"
)

_INTRODUCES_LIVING_SUBJECT = (
    r"(?:see|sees|saw|spot|spots|spotted|find|finds|found|meet|meets|met|"
    r"encounter|encounters|encountered|reveal|reveals|revealed)"
)

_BODY_PART = (
    r"(?:eyes?|hands?|fingers?|arms?|head|face|mouth|lips?|chest|feet|voice)"
)

# Provider reasoning must never become player-visible narration.  These are
# deliberately specific first-person/system-analysis phrases observed in live
# runs, not broad expressions such as "I think" that could occur in dialogue.
_META_REASONING = re.compile(
    r"(?:"
    r"\bi need to (?:process|analy[sz]e|check|reason through) this carefully\b|"
    r"\bthe player(?:'s|â€™s) action says\b|"
    r"\blet me (?:check|inspect|review) the world state\b|"
    r"\b(?:there is|there's|thereâ€™s) a narrative continuity issue\b|"
    r"\blet me (?:write|narrate|compose) (?:the )?narration\b|"
    r"\bauthorized reveals?\b|"
    r"\bnarration context\b|"
    r"\bactually,? looking more carefully\b"
    r")",
    re.IGNORECASE,
)


def _chunks(text: str) -> list[str]:
    """Return sentence-like units so an exemption cannot leak paragraphs."""
    return [
        chunk.strip()
        for chunk in re.split(
            r"(?<=[.!?])\s+|\n+|;\s+|,?\s+(?:and|but|then)\s+",
            text,
            flags=re.IGNORECASE,
        )
        if chunk.strip()
    ]


def _name_pattern(name: str) -> str:
    return rf"(?<!\w){re.escape(name)}(?!\w)"


class NarrativeGovernance:
    """Validate narrator prose against immutable, code-owned facts."""

    def __init__(self, dead_entities: Iterable[NPCState] = ()) -> None:
        facts: list[_DeadEntity] = []
        seen_ids: set[str] = set()
        for npc in dead_entities:
            if npc.id in seen_ids:
                continue
            candidates: list[str] = []
            for candidate in (npc.name, *npc.aliases):
                cleaned = (candidate or "").strip()
                if len(cleaned) >= 3 and cleaned.casefold() not in {
                    value.casefold() for value in candidates
                }:
                    candidates.append(cleaned)
            if candidates:
                facts.append(
                    _DeadEntity(
                        entity_id=npc.id,
                        name=npc.name,
                        aliases=tuple(candidates),
                    )
                )
                seen_ids.add(npc.id)
        self._dead_entities = tuple(facts)

    @classmethod
    def from_world_state(cls, world_state: WorldState | None) -> NarrativeGovernance:
        dead = (
            (npc for npc in world_state.npcs.values() if not npc.alive)
            if world_state is not None
            else ()
        )
        return cls(dead)

    @property
    def requires_buffering(self) -> bool:
        """Bad prose cannot be recalled after it has already been streamed."""
        return bool(self._dead_entities)

    @property
    def dead_names(self) -> tuple[str, ...]:
        return tuple(fact.name for fact in self._dead_entities)

    def validate(self, prose: str) -> list[ContinuityViolation]:
        violations: list[ContinuityViolation] = []
        if not prose:
            return violations

        meta_match = _META_REASONING.search(prose)
        if meta_match:
            excerpt_start = max(0, meta_match.start() - 40)
            violations.append(ContinuityViolation(
                rule="meta_reasoning_leak",
                entity_id="narrator",
                entity_name="Narrator output",
                reason=(
                    "the draft exposes private model reasoning or system-state "
                    "analysis instead of player-visible narration"
                ),
                excerpt=prose[excerpt_start:meta_match.end() + 120][:240],
            ))

        for chunk in _chunks(prose):
            if _NONLIVING_FRAME.search(chunk):
                continue

            for entity in self._dead_entities:
                matched_alias = next(
                    (
                        alias
                        for alias in sorted(entity.aliases, key=len, reverse=True)
                        if re.search(_name_pattern(alias), chunk, re.IGNORECASE)
                    ),
                    None,
                )
                if not matched_alias:
                    continue

                name = _name_pattern(matched_alias)
                direct_action = re.search(
                    rf"{name}(?:\s*,[^,.!?]{{0,48}},)?\s+"
                    rf"(?:(?:suddenly|slowly|quietly|softly|quickly|now)\s+){{0,2}}"
                    rf"{_ACTING_VERB}\b",
                    chunk,
                    re.IGNORECASE,
                )
                body_action = re.search(
                    rf"{name}[\u2019']s\s+{_BODY_PART}\s+{_ACTING_VERB}\b",
                    chunk,
                    re.IGNORECASE,
                )
                living_introduction = re.search(
                    rf"{_INTRODUCES_LIVING_SUBJECT}\b[^.!?\n]{{0,64}}{name}",
                    chunk,
                    re.IGNORECASE,
                )
                dialogue_label = re.search(
                    rf"{name}\s*:\s*[\"\u201c\u2018']",
                    chunk,
                    re.IGNORECASE,
                )
                dialogue_attribution = re.search(
                    rf"[\"\u201d\u2019']\s*,?\s*"
                    rf"(?:says?|asks?|replies|whispers?|shouts?)\s+{name}",
                    chunk,
                    re.IGNORECASE,
                )
                placed_as_living = re.search(
                    rf"\b(?:there|here|door|gate|threshold)\b[^.!?\n]{{0,32}}"
                    rf"(?:stands?|waits?|appears?)\s+{name}",
                    chunk,
                    re.IGNORECASE,
                )
                explicitly_alive = re.search(
                    rf"{name}\s+(?:is|was)\s+(?:alive|here|waiting|standing|sitting)\b",
                    chunk,
                    re.IGNORECASE,
                )

                if any(
                    (direct_action, body_action, living_introduction,
                     dialogue_label, dialogue_attribution, placed_as_living,
                     explicitly_alive)
                ):
                    violations.append(
                        ContinuityViolation(
                            rule="dead_npc_cannot_act",
                            entity_id=entity.entity_id,
                            entity_name=entity.name,
                            reason=(
                                "authoritative world state marks this NPC dead, "
                                "but the draft presents them as a living actor"
                            ),
                            excerpt=chunk[:240],
                        )
                    )
                    break

        return violations

    def repair_instruction(self, violations: list[ContinuityViolation]) -> str:
        details = "\n".join(f"- {v.to_prompt_line()}" for v in violations)
        rules = {violation.rule for violation in violations}
        constraints: list[str] = []
        if "dead_npc_cannot_act" in rules:
            constraints.append(
                "Authoritative world state is immutable: a dead NPC cannot "
                "speak, move, react, arrive, or otherwise behave as living. "
                "You may omit that NPC or mention them only with an explicit "
                "corpse, memory, dream, spirit, or illusion frame. Do not "
                "invent a resurrection."
            )
        if "meta_reasoning_leak" in rules:
            constraints.append(
                "Return only immersive, player-visible story prose. Remove all "
                "planning, chain-of-thought, prompt discussion, world-state "
                "inspection, continuity analysis, and statements about writing "
                "the narration."
            )
        return (
            "CONTINUITY REPAIR REQUIRED. Rewrite the entire draft while preserving "
            "the player's action and every non-conflicting outcome. "
            + " ".join(constraints)
            + " Return replacement prose and a complete "
            "replacement set of tool calls for only the rewritten prose.\n"
            f"Detected violations:\n{details}"
        )

    def safe_fallback(self, violations: list[ContinuityViolation]) -> str:
        if not any(v.rule == "dead_npc_cannot_act" for v in violations):
            return (
                "The consequences of your action settle across the scene. "
                "The way forward remains yours to choose."
            )
        names = list(dict.fromkeys(
            v.entity_name
            for v in violations
            if v.rule == "dead_npc_cannot_act"
        ))
        subject = ", ".join(names) if names else "The dead"
        verb = "remains" if len(names) == 1 else "remain"
        return (
            "The moment resolves without overturning established reality. "
            f"{subject} {verb} dead and cannot answer, move, or re-enter the scene."
        )
