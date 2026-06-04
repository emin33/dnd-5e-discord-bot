"""TurnLogReader — assertion-friendly accessor over per-session JSONL turn logs.

The orchestrator writes one JSON record per turn to
``data/turn_logs/{session_id}.jsonl`` via ``TurnRecord``. This reader
loads those records and exposes a fluent query API for tests to assert
state at any historical turn.

Designed for the long-horizon "seed-and-callback" test: plant something
at turn 5, run 25 turns of unrelated activity, assert the state still
holds at turn 30.

Usage::

    log = TurnLogReader.load(session_id)
    assert log.world_state_after(turn=5).npc("innkeeper").holds("ancient relic")
    assert log.kg_context_for(turn=25).mentions("ancient relic")
    assert log.narrator_response(turn=25).text != ""
    history = log.entity_history("ancient relic")  # → list of (turn, holder)
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional

import yaml

TURN_LOG_DIR = Path("data/turn_logs")


# ── Snapshot wrappers ──────────────────────────────────────────────────


@dataclass
class NPCSnapshot:
    """A single NPC's state from a WorldState YAML snapshot."""
    name: str
    disposition: str = "neutral"
    description: str = ""
    inventory: list[str] = None
    notes: str = ""
    location: str = ""

    def __post_init__(self) -> None:
        if self.inventory is None:
            self.inventory = []

    def holds(self, item_name: str) -> bool:
        """Case-insensitive substring match on the NPC's inventory."""
        target = item_name.strip().lower()
        return any(target in i.lower() for i in self.inventory)

    def __bool__(self) -> bool:  # truthy if real (i.e. found)
        return bool(self.name)


_MISSING_NPC = NPCSnapshot(name="")


@dataclass
class WorldStateSnapshot:
    """Parsed WorldState YAML at one point in time."""
    raw_yaml: str
    parsed: dict

    @property
    def turn(self) -> int:
        return int(self.parsed.get("turn", 0))

    @property
    def location(self) -> str:
        return self.parsed.get("location", "")

    @property
    def time_of_day(self) -> str:
        return self.parsed.get("time_of_day", "")

    @property
    def phase(self) -> str:
        return self.parsed.get("phase", "")

    def npc(self, name_or_id: str) -> NPCSnapshot:
        """Find an NPC by id (preferred), name, or alias substring match.

        Searches both ``npcs_here`` (full detail) and ``key_npcs_elsewhere``
        (also full detail under the new schema). Falls back to the legacy
        ``key_npcs_elsewhere`` one-line string format for older logs.
        """
        target = name_or_id.strip().lower()

        def _matches(entry: dict) -> bool:
            if not isinstance(entry, dict):
                return False
            # Exact id match
            if entry.get("id") and entry["id"].lower() == target:
                return True
            entry_name = (entry.get("name") or "").lower()
            if entry_name and (target in entry_name or entry_name in target):
                return True
            # Alias match
            for alias in entry.get("aliases", []) or []:
                if alias and target in alias.lower():
                    return True
            return False

        def _to_snapshot(entry: dict, default_loc: str) -> NPCSnapshot:
            # Schema renamed `holds` → `inventory`; read both for back-compat
            inv = entry.get("inventory")
            if inv is None:
                inv = entry.get("holds", [])
            return NPCSnapshot(
                name=entry.get("name", "") or "",
                disposition=entry.get("disposition", "neutral"),
                description=entry.get("desc", "") or entry.get("description", ""),
                inventory=list(inv or []),
                notes=entry.get("notes", ""),
                location=entry.get("location", default_loc),
            )

        # Detailed entries: list of dicts (npcs_here)
        for entry in self.parsed.get("npcs_here", []) or []:
            if _matches(entry):
                return _to_snapshot(entry, self.location)

        # key_npcs_elsewhere: now dicts (post-refactor) or strings (legacy)
        for entry in self.parsed.get("key_npcs_elsewhere", []) or []:
            if isinstance(entry, dict):
                if _matches(entry):
                    return _to_snapshot(entry, entry.get("location", ""))
            elif isinstance(entry, str) and target in entry.lower():
                snap = self._parse_distant_npc_line(entry)
                if snap:
                    return snap
        return _MISSING_NPC

    def all_npcs(self) -> list[NPCSnapshot]:
        result: list[NPCSnapshot] = []

        def _to_snapshot(entry: dict, default_loc: str) -> NPCSnapshot:
            inv = entry.get("inventory")
            if inv is None:
                inv = entry.get("holds", [])
            return NPCSnapshot(
                name=entry.get("name", "") or "",
                disposition=entry.get("disposition", "neutral"),
                description=entry.get("desc", "") or entry.get("description", ""),
                inventory=list(inv or []),
                notes=entry.get("notes", ""),
                location=entry.get("location", default_loc),
            )

        for entry in self.parsed.get("npcs_here", []) or []:
            if isinstance(entry, dict):
                result.append(_to_snapshot(entry, self.location))
        for entry in self.parsed.get("key_npcs_elsewhere", []) or []:
            if isinstance(entry, dict):
                result.append(_to_snapshot(entry, entry.get("location", "")))
            elif isinstance(entry, str):
                snap = self._parse_distant_npc_line(entry)
                if snap:
                    result.append(snap)
        return result

    def has_quest(self, name_substring: str) -> bool:
        target = name_substring.strip().lower()
        for q in self.parsed.get("active_quests", []) or []:
            qname = (q.get("name", "") if isinstance(q, dict) else str(q)).lower()
            if target in qname:
                return True
        return False

    def has_event_mentioning(self, substring: str) -> bool:
        target = substring.strip().lower()
        for ev in self.parsed.get("recent_events", []) or []:
            if target in str(ev).lower():
                return True
        return False

    def has_fact_mentioning(self, substring: str) -> bool:
        target = substring.strip().lower()
        for fact in self.parsed.get("established_facts", []) or []:
            if target in str(fact).lower():
                return True
        return False

    @staticmethod
    def _parse_distant_npc_line(line: str) -> Optional[NPCSnapshot]:
        """Parse a one-liner like 'Bron: at inn, friendly - holds: relic, key - notes...'"""
        if ":" not in line:
            return None
        name, _, rest = line.partition(":")
        rest = rest.strip()
        # Extract holds list (between 'holds:' and next '-' or end)
        inventory: list[str] = []
        m = re.search(r"holds:\s*([^-]+)", rest)
        if m:
            items_str = m.group(1).strip()
            inventory = [i.strip() for i in items_str.split(",") if i.strip()]
        # Disposition (the third comma-separated field of the prefix)
        disposition = "neutral"
        for tok in ("hostile", "unfriendly", "neutral", "friendly", "allied"):
            if tok in rest:
                disposition = tok
                break
        # Location
        loc_m = re.search(r"at\s+([^,]+)", rest)
        location = loc_m.group(1).strip() if loc_m else ""
        return NPCSnapshot(
            name=name.strip(),
            disposition=disposition,
            description="",
            inventory=inventory,
            notes="",
            location=location,
        )


@dataclass
class KGContextSnapshot:
    """The KG context YAML injected for a single turn."""
    yaml_text: str
    parsed: dict
    seed_entities: list[str]
    text_match_seeds: list[str]
    scene_seeds: list[str]
    vector_match_seeds: list[str]
    narrative_chunks: list[dict]

    def mentions(self, substring: str) -> bool:
        """Does the injected context (yaml + chunks) reference this string?"""
        target = substring.strip().lower()
        if target in self.yaml_text.lower():
            return True
        for chunk in self.narrative_chunks:
            if target in (chunk.get("preview") or "").lower():
                return True
        return False

    def chunks_mentioning(self, substring: str) -> list[dict]:
        target = substring.strip().lower()
        return [c for c in self.narrative_chunks if target in (c.get("preview") or "").lower()]


@dataclass
class NarratorResponse:
    """Narrator output for a single turn."""
    text: str
    format: str
    reprompted: bool

    def references(self, substring: str) -> bool:
        return substring.strip().lower() in self.text.lower()

    @property
    def char_length(self) -> int:
        return len(self.text)


@dataclass
class EffectsSnapshot:
    """All effects emitted on a single turn.

    The orchestrator currently writes effects as a flat list of dicts
    via ``_turn_record.set("effects", [...])``. Each entry has a "type"
    key plus ad-hoc shape fields (item, ref_id, npc_name, etc.). The
    reader normalizes both that list and the legacy
    {proposed, executed, rejected} shape into a single flat ``effects``
    list and continues to expose ``executed_of_type`` etc. as if all
    fired effects landed.
    """
    effects: list[dict]
    proposed: list[dict] = None
    executed: list[dict] = None
    rejected: list[dict] = None

    def __post_init__(self):
        # Backwards-compat: if the legacy buckets exist, fold executed
        # into the flat list.
        if self.executed and not self.effects:
            self.effects = list(self.executed)

    def of_type(self, effect_type: str) -> list[dict]:
        """All effects of a given type — supports both 'type' (new)
        and 'effect_type' (legacy) keys."""
        return [
            e for e in self.effects
            if e.get("type") == effect_type or e.get("effect_type") == effect_type
        ]

    def has(self, effect_type: str) -> bool:
        return bool(self.of_type(effect_type))

    # Back-compat aliases for tests written against the old API
    def executed_of_type(self, effect_type: str) -> list[dict]:
        return self.of_type(effect_type)

    def has_executed(self, effect_type: str) -> bool:
        return self.has(effect_type)


@dataclass
class NarratorRouting:
    """Which narrator tier ran this turn."""
    tier: str
    provider: str
    model: str
    significance: Optional[str]
    phase_b_veto: bool


# ── Reader ─────────────────────────────────────────────────────────────


class TurnLogReader:
    """Loads and queries per-session turn logs.

    Construction is cheap (one file read + JSONL parse). All accessors
    are stateless lookups against the parsed records list.
    """

    def __init__(self, session_id: str, records: list[dict]):
        self.session_id = session_id
        self.records = records
        # Index turns for O(1) lookup. Some sessions may have non-monotonic
        # turns if the game allows replays; we prefer the LAST record per
        # turn number.
        self._by_turn: dict[int, dict] = {}
        for rec in records:
            t = rec.get("turn")
            if isinstance(t, int):
                self._by_turn[t] = rec

    @classmethod
    def load(cls, session_id: str, log_dir: Path = TURN_LOG_DIR) -> "TurnLogReader":
        """Load all turn records for a session from JSONL.

        Raises FileNotFoundError if no log exists for the session.
        """
        path = log_dir / f"{session_id}.jsonl"
        if not path.exists():
            raise FileNotFoundError(f"No turn log at {path}")
        records: list[dict] = []
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    records.append(json.loads(line))
                except json.JSONDecodeError:
                    # Skip malformed lines but don't fail the whole load
                    continue
        return cls(session_id, records)

    # ── Basic access ──────────────────────────────────────────────────

    def turns(self) -> list[int]:
        """All turn numbers present, sorted ascending."""
        return sorted(self._by_turn.keys())

    def get(self, turn: int) -> Optional[dict]:
        return self._by_turn.get(turn)

    def __iter__(self) -> Iterator[dict]:
        return iter(self.records)

    def __len__(self) -> int:
        return len(self.records)

    # ── Snapshot queries ──────────────────────────────────────────────

    def world_state_after(self, turn: int) -> WorldStateSnapshot:
        """The WorldState at the end of the given turn."""
        rec = self._by_turn.get(turn) or {}
        ws = rec.get("world_state") or {}
        raw = ws.get("after", "") or ""
        return WorldStateSnapshot(raw_yaml=raw, parsed=_safe_parse_yaml(raw))

    def world_state_before(self, turn: int) -> WorldStateSnapshot:
        rec = self._by_turn.get(turn) or {}
        ws = rec.get("world_state") or {}
        raw = ws.get("before", "") or ""
        return WorldStateSnapshot(raw_yaml=raw, parsed=_safe_parse_yaml(raw))

    def kg_context_for(self, turn: int) -> KGContextSnapshot:
        """The KG context injected into the narrator for this turn."""
        rec = self._by_turn.get(turn) or {}
        kg = rec.get("knowledge_graph") or {}
        raw = kg.get("context_yaml", "") or ""
        return KGContextSnapshot(
            yaml_text=raw,
            parsed=_safe_parse_yaml(raw),
            seed_entities=list(kg.get("seed_entities", []) or []),
            text_match_seeds=list(kg.get("text_match_seeds", []) or []),
            scene_seeds=list(kg.get("scene_seeds", []) or []),
            vector_match_seeds=list(kg.get("vector_match_seeds", []) or []),
            narrative_chunks=list(kg.get("narrative_chunks", []) or []),
        )

    def narrator_response(self, turn: int) -> NarratorResponse:
        rec = self._by_turn.get(turn) or {}
        nr = rec.get("narrator_response") or {}
        return NarratorResponse(
            text=nr.get("raw") or nr.get("preview") or "",
            format=nr.get("format", ""),
            reprompted=bool(nr.get("reprompted", False)),
        )

    def effects_at(self, turn: int) -> EffectsSnapshot:
        rec = self._by_turn.get(turn) or {}
        eff = rec.get("effects")
        # Two formats: flat list (current orchestrator) or
        # {proposed, executed, rejected} dict (planned but unused)
        if isinstance(eff, list):
            return EffectsSnapshot(effects=list(eff))
        if isinstance(eff, dict):
            return EffectsSnapshot(
                effects=list(eff.get("executed", []) or []),
                proposed=list(eff.get("proposed", []) or []),
                executed=list(eff.get("executed", []) or []),
                rejected=list(eff.get("rejected", []) or []),
            )
        return EffectsSnapshot(effects=[])

    def narrator_routing(self, turn: int) -> Optional[NarratorRouting]:
        rec = self._by_turn.get(turn) or {}
        nr = rec.get("narrator_routing")
        if not nr:
            return None
        return NarratorRouting(
            tier=nr.get("tier", ""),
            provider=nr.get("provider", ""),
            model=nr.get("model", ""),
            significance=nr.get("significance"),
            phase_b_veto=bool(nr.get("phase_b_veto", False)),
        )

    def player_action(self, turn: int) -> str:
        """The player's input action text on this turn.

        Reads the orchestrator's top-level ``action`` field first; falls
        back to the user message in ``prompt.messages`` for compatibility
        with synthetic test fixtures.
        """
        rec = self._by_turn.get(turn) or {}
        action = rec.get("action")
        if isinstance(action, str) and action:
            return action
        prompt = rec.get("prompt", {}) or {}
        for msg in reversed(prompt.get("messages", []) or []):
            if msg.get("role") == "user":
                return msg.get("content") or msg.get("preview", "")
        return ""

    # ── Trajectory queries ─────────────────────────────────────────────

    def entity_history(self, name: str) -> list[tuple[int, NPCSnapshot]]:
        """Trace an NPC's state across all turns where they appear.

        Returns ``[(turn, snapshot), ...]`` sorted by turn. Useful for
        questions like "did the innkeeper hold the relic continuously
        from turn 5 to turn 30?".
        """
        target = name.strip().lower()
        history: list[tuple[int, NPCSnapshot]] = []
        for turn in self.turns():
            ws = self.world_state_after(turn)
            for npc in ws.all_npcs():
                if target in npc.name.lower() or npc.name.lower() in target:
                    history.append((turn, npc))
                    break
        return history

    def turns_mentioning(self, substring: str) -> list[int]:
        """All turn numbers where the substring appears in the prose."""
        target = substring.strip().lower()
        result: list[int] = []
        for turn in self.turns():
            response = self.narrator_response(turn)
            if target in response.text.lower():
                result.append(turn)
        return result

    def turns_with_kg_mention(self, substring: str) -> list[int]:
        """All turn numbers where the KG context (or recalled chunks)
        mentions the substring — i.e. when the narrator HAD knowledge of it.
        """
        target = substring.strip().lower()
        result: list[int] = []
        for turn in self.turns():
            ctx = self.kg_context_for(turn)
            if ctx.mentions(target):
                result.append(turn)
        return result


# ── Internals ──────────────────────────────────────────────────────────


def _safe_parse_yaml(text: str) -> dict:
    """Parse YAML defensively — return empty dict on any failure."""
    if not text:
        return {}
    try:
        result = yaml.safe_load(text)
        return result if isinstance(result, dict) else {}
    except Exception:
        return {}
