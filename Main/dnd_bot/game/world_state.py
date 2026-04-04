"""Authoritative World State - the single source of truth for narrator context.

The LLM narrator reads this state but never writes it directly. After each
narration, a cheap extraction model emits a StateDelta that Python validates
before applying. This eliminates entity teleportation, NPC contradictions,
and fact drift at their architectural root.

Design: WorldState layers ON TOP of SceneEntityRegistry. The registry keeps
hostility math, SRD matching, and combat triggers. WorldState provides the
narrator's view of reality as a compact YAML snapshot.
"""

from typing import Literal, Optional
import structlog
import yaml

from pydantic import BaseModel, Field

logger = structlog.get_logger()

# Valid time-of-day values (ordered for advancement)
TIME_PROGRESSION = [
    "dawn", "morning", "midday", "afternoon",
    "dusk", "evening", "night", "midnight",
]

# Valid game phases
VALID_PHASES = ["exploration", "combat", "dialogue", "rest", "shopping"]

# Phase State Machine — valid transitions between game phases.
# Each phase maps to the set of phases it can transition TO.
# Prevents nonsensical transitions (e.g., shopping during combat).
PHASE_TRANSITIONS = {
    "exploration": {"combat", "dialogue", "rest", "shopping", "exploration"},
    "combat":      {"exploration", "dialogue", "combat"},  # Can't shop/rest mid-combat
    "dialogue":    {"exploration", "combat", "shopping", "dialogue"},
    "rest":        {"exploration", "combat", "rest"},  # Can be interrupted by combat
    "shopping":    {"exploration", "dialogue", "combat", "shopping"},
}

# Phase-specific narrator style hints — injected into the narrator instruction
# to shape tone and pacing per phase.
PHASE_STYLE_HINTS = {
    "exploration": "Describe the environment with rich detail. Build atmosphere and mystery. Invite the player to investigate.",
    "combat": "Write with urgency and kinetic energy. Short, punchy sentences. Focus on action and consequences.",
    "dialogue": "Give NPCs distinct voices and mannerisms. Let subtext and personality drive the scene.",
    "rest": "Slow the pace. Reflective, quiet moments. Campfire conversations and recovery.",
    "shopping": "Be practical but characterful. Merchants have personalities. Describe wares vividly.",
}


def is_valid_phase_transition(current: str, target: str) -> bool:
    """Check if a phase transition is valid according to the FSM."""
    valid_targets = PHASE_TRANSITIONS.get(current, set())
    return target in valid_targets


class NPCState(BaseModel):
    """State of an NPC in the world."""
    name: str
    location: str = ""
    disposition: str = "neutral"  # hostile/unfriendly/neutral/friendly/allied
    description: str = ""
    alive: bool = True
    notes: str = ""
    important: bool = False  # Quest-givers, allies, key story NPCs stay visible


class NPCUpdate(BaseModel):
    """A proposed change to an existing NPC."""
    name: str
    location: Optional[str] = None
    disposition: Optional[str] = None
    description: Optional[str] = None
    alive: Optional[bool] = None
    notes: Optional[str] = None
    important: Optional[bool] = None


class PlayerSnapshot(BaseModel):
    """Snapshot of player state for narrator context."""
    name: str
    hp: int = 0
    max_hp: int = 0
    conditions: list[str] = Field(default_factory=list)
    concentration: str = ""


class StateDelta(BaseModel):
    """Proposed changes to world state, emitted by extraction model.

    Only non-None fields represent changes. Python validates each change
    before applying it to WorldState.
    """
    time_change: Optional[str] = None
    location_change: Optional[str] = None
    location_description: Optional[str] = None
    new_connections: list[str] = Field(default_factory=list)
    npc_updates: list[NPCUpdate] = Field(default_factory=list)
    new_npcs: list[NPCState] = Field(default_factory=list)
    removed_npcs: list[str] = Field(default_factory=list)
    new_events: list[str] = Field(default_factory=list)
    new_facts: list[str] = Field(default_factory=list)
    flag_changes: dict[str, bool] = Field(default_factory=dict)
    phase_change: Optional[str] = None


# Cache the schema so we don't regenerate it every call
_STATE_DELTA_SCHEMA: Optional[dict] = None


def get_state_delta_schema() -> dict:
    """Get the JSON schema for StateDelta structured output."""
    global _STATE_DELTA_SCHEMA
    if _STATE_DELTA_SCHEMA is None:
        _STATE_DELTA_SCHEMA = StateDelta.model_json_schema()
    return _STATE_DELTA_SCHEMA


class WorldState(BaseModel):
    """Authoritative world state managed by Python code.

    The narrator receives a YAML snapshot of this state and renders prose
    from it. It never writes to this state directly -- changes come through
    validated StateDeltas.
    """
    turn: int = 0
    phase: str = "exploration"
    time_of_day: str = "morning"
    current_location: str = ""
    location_description: str = ""
    connected_locations: list[str] = Field(default_factory=list)

    npcs: dict[str, NPCState] = Field(default_factory=dict)
    players: dict[str, PlayerSnapshot] = Field(default_factory=dict)

    # Scene items — objects present in the current scene (spawned, dropped, visible)
    scene_items: dict[str, str] = Field(default_factory=dict)  # id -> description

    # Transaction ledger — recent item/currency transfers (ring buffer, narrator sees these)
    recent_transfers: list[str] = Field(default_factory=list)

    active_effects: list[str] = Field(default_factory=list)
    recent_events: list[str] = Field(default_factory=list)
    established_facts: list[str] = Field(default_factory=list)
    global_flags: dict[str, bool] = Field(default_factory=dict)

    # Max recent events/transfers to keep (ring buffer)
    _max_recent_events: int = 5
    _max_recent_transfers: int = 8

    def increment_turn(self) -> None:
        """Advance the turn counter."""
        self.turn += 1

    # ── Item & Currency Tracking ──

    def spawn_item(self, item_id: str, description: str) -> None:
        """Record an item appearing in the scene."""
        self.scene_items[item_id] = description

    def remove_item(self, item_id: str) -> None:
        """Remove an item from the scene (picked up, destroyed, etc.)."""
        self.scene_items.pop(item_id, None)

    def record_transfer(self, description: str) -> None:
        """Record an item or currency transfer for narrator context.

        Examples:
            "Player picked up Jeweled Dagger from the pedestal"
            "Farmer gave 15gp to player"
            "NPC returned silver coins to player"
        """
        self.recent_transfers.append(description)
        if len(self.recent_transfers) > self._max_recent_transfers:
            self.recent_transfers = self.recent_transfers[-self._max_recent_transfers:]

    def sync_player(self, name: str, hp: int, max_hp: int,
                    conditions: list[str], concentration: str = "") -> None:
        """Sync a player snapshot from the Character model."""
        self.players[name] = PlayerSnapshot(
            name=name,
            hp=hp,
            max_hp=max_hp,
            conditions=conditions,
            concentration=concentration,
        )

    def apply_delta(self, delta: StateDelta) -> list[str]:
        """Validate and apply a StateDelta. Returns list of rejected changes."""
        rejections = []

        # Phase change (validated against FSM)
        if delta.phase_change:
            if delta.phase_change not in VALID_PHASES:
                rejections.append(f"Invalid phase: {delta.phase_change}")
            elif not is_valid_phase_transition(self.phase, delta.phase_change):
                rejections.append(
                    f"Invalid phase transition: {self.phase} -> {delta.phase_change}"
                )
            else:
                self.phase = delta.phase_change

        # Time change
        if delta.time_change:
            if delta.time_change in TIME_PROGRESSION:
                self.time_of_day = delta.time_change
            else:
                rejections.append(f"Invalid time: {delta.time_change}")

        # Location change
        if delta.location_change:
            self.current_location = delta.location_change
            if delta.location_description:
                self.location_description = delta.location_description

        # New connections
        for conn in delta.new_connections:
            if conn and conn not in self.connected_locations:
                self.connected_locations.append(conn)

        # New NPCs
        for npc in delta.new_npcs:
            if npc.name in self.npcs:
                rejections.append(f"NPC already exists: {npc.name}")
                continue
            # Default location to current party location if not specified
            if not npc.location:
                npc.location = self.current_location
            self.npcs[npc.name] = npc

        # NPC updates
        for update in delta.npc_updates:
            existing = self.npcs.get(update.name)
            if not existing:
                # Try case-insensitive match
                existing = self._find_npc(update.name)
                if not existing:
                    rejections.append(f"NPC not found for update: {update.name}")
                    continue

            # Validate: dead NPCs can't act
            if not existing.alive and update.alive is not True:
                if update.disposition or update.location or update.notes:
                    rejections.append(f"Dead NPC cannot act: {update.name}")
                    continue

            # Apply non-None fields
            if update.location is not None:
                existing.location = update.location
            if update.disposition is not None:
                existing.disposition = update.disposition
            if update.description is not None:
                existing.description = update.description
            if update.alive is not None:
                existing.alive = update.alive
            if update.notes is not None:
                existing.notes = update.notes
            if update.important is not None:
                existing.important = update.important

        # Removed NPCs (left the scene, not dead)
        for name in delta.removed_npcs:
            npc = self._find_npc(name)
            if npc:
                # Don't delete -- just clear their location (they left)
                npc.location = ""

        # New events (ring buffer)
        for event in delta.new_events:
            if event:
                self.recent_events.append(event)
        # Trim to max
        if len(self.recent_events) > self._max_recent_events:
            self.recent_events = self.recent_events[-self._max_recent_events:]

        # New facts (deduplicated)
        for fact in delta.new_facts:
            fact = fact.strip()
            if fact and fact not in self.established_facts:
                self.established_facts.append(fact)

        # Flag changes
        for key, value in delta.flag_changes.items():
            self.global_flags[key] = value

        if rejections:
            logger.info(
                "state_delta_rejections",
                turn=self.turn,
                rejections=rejections,
            )

        return rejections

    def _find_npc(self, name: str) -> Optional[NPCState]:
        """Find NPC by exact or case-insensitive name."""
        if name in self.npcs:
            return self.npcs[name]
        name_lower = name.lower()
        for key, npc in self.npcs.items():
            if key.lower() == name_lower or npc.name.lower() == name_lower:
                return npc
        return None

    def get_npcs_at_location(self, location: str = "") -> list[NPCState]:
        """Get NPCs at a specific location (default: current party location)."""
        loc = location or self.current_location
        if not loc:
            return list(self.npcs.values())
        return [
            npc for npc in self.npcs.values()
            if npc.alive and npc.location and npc.location.lower() == loc.lower()
        ]

    def get_important_npcs_elsewhere(self) -> list[NPCState]:
        """Get important NPCs NOT at the current location."""
        loc = self.current_location.lower() if self.current_location else ""
        return [
            npc for npc in self.npcs.values()
            if npc.alive and npc.important
            and (not npc.location or npc.location.lower() != loc)
        ]

    def to_yaml(self) -> str:
        """Serialize to compact YAML for narrator injection.

        Tiered NPC detail:
        - Full detail for NPCs at current location
        - One-line summary for important NPCs elsewhere
        - Minor NPCs at other locations omitted
        """
        data: dict = {
            "turn": self.turn,
            "phase": self.phase,
            "time_of_day": self.time_of_day,
        }

        # Location
        if self.current_location:
            data["location"] = self.current_location
            if self.location_description:
                data["location_desc"] = self.location_description
            if self.connected_locations:
                data["exits"] = self.connected_locations

        # Players
        if self.players:
            player_list = []
            for p in self.players.values():
                entry = f"{p.name}: HP {p.hp}/{p.max_hp}"
                if p.conditions:
                    entry += f" [{', '.join(p.conditions)}]"
                if p.concentration:
                    entry += f" (concentrating: {p.concentration})"
                player_list.append(entry)
            data["party"] = player_list

        # NPCs at current location (full detail)
        local_npcs = self.get_npcs_at_location()
        if local_npcs:
            npc_entries = []
            for npc in local_npcs:
                entry: dict = {
                    "name": npc.name,
                    "disposition": npc.disposition,
                }
                if npc.description:
                    entry["desc"] = npc.description[:100]
                if npc.notes:
                    entry["notes"] = npc.notes[:80]
                npc_entries.append(entry)
            data["npcs_here"] = npc_entries

        # Important NPCs elsewhere (one-line summaries)
        distant_important = self.get_important_npcs_elsewhere()
        if distant_important:
            data["key_npcs_elsewhere"] = [
                f"{npc.name}: at {npc.location or 'unknown'}, {npc.disposition}"
                + (f" - {npc.notes[:60]}" if npc.notes else "")
                for npc in distant_important
            ]

        # Scene items (objects present in the current location)
        if self.scene_items:
            data["scene_items"] = [
                f"{item_id}: {desc}" for item_id, desc in self.scene_items.items()
            ]

        # Recent transfers (item/currency changes the narrator must not contradict)
        if self.recent_transfers:
            data["recent_transfers"] = self.recent_transfers

        # Active effects
        if self.active_effects:
            data["active_effects"] = self.active_effects

        # Recent events
        if self.recent_events:
            data["recent_events"] = self.recent_events

        # Established facts
        if self.established_facts:
            data["facts"] = self.established_facts

        # Global flags (only true ones, for brevity)
        active_flags = {k: v for k, v in self.global_flags.items() if v}
        if active_flags:
            data["flags"] = active_flags

        return yaml.dump(data, default_flow_style=False, sort_keys=False, width=120)

    @classmethod
    def from_session_start(cls, player_names: list[str]) -> "WorldState":
        """Create initial WorldState with just player names.

        Location, time, and NPCs will be seeded from the first narrator
        response via StateDelta extraction.
        """
        ws = cls()
        for name in player_names:
            ws.players[name] = PlayerSnapshot(name=name)
        return ws
