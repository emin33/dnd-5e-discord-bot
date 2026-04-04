"""Tests for authoritative WorldState system."""

import pytest
import yaml

from dnd_bot.game.world_state import (
    WorldState,
    StateDelta,
    NPCState,
    NPCUpdate,
    PlayerSnapshot,
    get_state_delta_schema,
)


class TestWorldState:
    """Test WorldState creation and basic operations."""

    def test_empty_creation(self):
        ws = WorldState()
        assert ws.turn == 0
        assert ws.phase == "exploration"
        assert ws.time_of_day == "morning"
        assert ws.current_location == ""
        assert ws.npcs == {}
        assert ws.players == {}

    def test_from_session_start(self):
        ws = WorldState.from_session_start(["Thorin", "Elara"])
        assert "Thorin" in ws.players
        assert "Elara" in ws.players
        assert ws.players["Thorin"].name == "Thorin"

    def test_increment_turn(self):
        ws = WorldState()
        ws.increment_turn()
        assert ws.turn == 1
        ws.increment_turn()
        assert ws.turn == 2

    def test_sync_player(self):
        ws = WorldState()
        ws.sync_player("Thorin", hp=15, max_hp=20, conditions=["poisoned"], concentration="Shield")
        p = ws.players["Thorin"]
        assert p.hp == 15
        assert p.max_hp == 20
        assert "poisoned" in p.conditions
        assert p.concentration == "Shield"

    def test_get_npcs_at_location(self):
        ws = WorldState(current_location="tavern")
        ws.npcs["Barkeep"] = NPCState(name="Barkeep", location="tavern", disposition="friendly")
        ws.npcs["King"] = NPCState(name="King", location="castle", disposition="neutral", important=True)

        local = ws.get_npcs_at_location()
        assert len(local) == 1
        assert local[0].name == "Barkeep"

    def test_get_important_npcs_elsewhere(self):
        ws = WorldState(current_location="tavern")
        ws.npcs["Barkeep"] = NPCState(name="Barkeep", location="tavern", disposition="friendly")
        ws.npcs["King"] = NPCState(name="King", location="castle", disposition="neutral", important=True)
        ws.npcs["Random Guard"] = NPCState(name="Random Guard", location="gate", disposition="neutral")

        important = ws.get_important_npcs_elsewhere()
        assert len(important) == 1
        assert important[0].name == "King"


class TestStateDelta:
    """Test StateDelta application and validation."""

    def test_apply_location_change(self):
        ws = WorldState(current_location="tavern")
        delta = StateDelta(
            location_change="forest",
            location_description="A dark pine forest",
        )
        rejections = ws.apply_delta(delta)
        assert rejections == []
        assert ws.current_location == "forest"
        assert ws.location_description == "A dark pine forest"

    def test_apply_time_change(self):
        ws = WorldState(time_of_day="morning")
        delta = StateDelta(time_change="dusk")
        rejections = ws.apply_delta(delta)
        assert rejections == []
        assert ws.time_of_day == "dusk"

    def test_reject_invalid_time(self):
        ws = WorldState()
        delta = StateDelta(time_change="invalid_time")
        rejections = ws.apply_delta(delta)
        assert len(rejections) == 1
        assert "Invalid time" in rejections[0]

    def test_apply_new_npc(self):
        ws = WorldState(current_location="tavern")
        delta = StateDelta(
            new_npcs=[NPCState(name="Grimjaw", disposition="unfriendly", description="A scarred dwarf")]
        )
        rejections = ws.apply_delta(delta)
        assert rejections == []
        assert "Grimjaw" in ws.npcs
        assert ws.npcs["Grimjaw"].location == "tavern"  # Defaults to current location
        assert ws.npcs["Grimjaw"].disposition == "unfriendly"

    def test_reject_duplicate_npc(self):
        ws = WorldState()
        ws.npcs["Grimjaw"] = NPCState(name="Grimjaw", disposition="neutral")
        delta = StateDelta(
            new_npcs=[NPCState(name="Grimjaw", disposition="hostile")]
        )
        rejections = ws.apply_delta(delta)
        assert len(rejections) == 1
        assert "already exists" in rejections[0]

    def test_apply_npc_update(self):
        ws = WorldState()
        ws.npcs["Grimjaw"] = NPCState(name="Grimjaw", location="tavern", disposition="neutral")
        delta = StateDelta(
            npc_updates=[NPCUpdate(name="Grimjaw", disposition="friendly", notes="Helped the party")]
        )
        rejections = ws.apply_delta(delta)
        assert rejections == []
        assert ws.npcs["Grimjaw"].disposition == "friendly"
        assert ws.npcs["Grimjaw"].notes == "Helped the party"
        assert ws.npcs["Grimjaw"].location == "tavern"  # Unchanged

    def test_reject_update_nonexistent_npc(self):
        ws = WorldState()
        delta = StateDelta(
            npc_updates=[NPCUpdate(name="Nobody", disposition="hostile")]
        )
        rejections = ws.apply_delta(delta)
        assert len(rejections) == 1
        assert "not found" in rejections[0]

    def test_reject_dead_npc_action(self):
        ws = WorldState()
        ws.npcs["Grimjaw"] = NPCState(name="Grimjaw", alive=False)
        delta = StateDelta(
            npc_updates=[NPCUpdate(name="Grimjaw", disposition="hostile")]
        )
        rejections = ws.apply_delta(delta)
        assert len(rejections) == 1
        assert "Dead NPC" in rejections[0]

    def test_revive_dead_npc(self):
        ws = WorldState()
        ws.npcs["Grimjaw"] = NPCState(name="Grimjaw", alive=False)
        delta = StateDelta(
            npc_updates=[NPCUpdate(name="Grimjaw", alive=True, disposition="neutral")]
        )
        rejections = ws.apply_delta(delta)
        assert rejections == []
        assert ws.npcs["Grimjaw"].alive is True

    def test_remove_npc_clears_location(self):
        ws = WorldState()
        ws.npcs["Grimjaw"] = NPCState(name="Grimjaw", location="tavern")
        delta = StateDelta(removed_npcs=["Grimjaw"])
        rejections = ws.apply_delta(delta)
        assert rejections == []
        assert ws.npcs["Grimjaw"].location == ""  # Left, not deleted

    def test_apply_events_ring_buffer(self):
        ws = WorldState()
        for i in range(10):
            delta = StateDelta(new_events=[f"Event {i}"])
            ws.apply_delta(delta)
        assert len(ws.recent_events) == 5  # Max 5
        assert ws.recent_events[0] == "Event 5"
        assert ws.recent_events[-1] == "Event 9"

    def test_apply_facts_deduplicated(self):
        ws = WorldState()
        delta1 = StateDelta(new_facts=["The bridge is destroyed", "The king is alive"])
        delta2 = StateDelta(new_facts=["The bridge is destroyed", "New fact"])
        ws.apply_delta(delta1)
        ws.apply_delta(delta2)
        assert len(ws.established_facts) == 3
        assert "The bridge is destroyed" in ws.established_facts
        assert "New fact" in ws.established_facts

    def test_apply_flags(self):
        ws = WorldState()
        delta = StateDelta(flag_changes={"bridge_destroyed": True, "king_dead": False})
        ws.apply_delta(delta)
        assert ws.global_flags["bridge_destroyed"] is True
        assert ws.global_flags["king_dead"] is False

    def test_apply_new_connections(self):
        ws = WorldState(connected_locations=["forest"])
        delta = StateDelta(new_connections=["cave", "forest"])  # "forest" already exists
        ws.apply_delta(delta)
        assert "cave" in ws.connected_locations
        assert ws.connected_locations.count("forest") == 1  # No duplicate

    def test_apply_phase_change(self):
        ws = WorldState(phase="exploration")
        delta = StateDelta(phase_change="dialogue")
        ws.apply_delta(delta)
        assert ws.phase == "dialogue"

    def test_reject_invalid_phase(self):
        ws = WorldState()
        delta = StateDelta(phase_change="flying")
        rejections = ws.apply_delta(delta)
        assert len(rejections) == 1
        assert "Invalid phase" in rejections[0]

    def test_case_insensitive_npc_lookup(self):
        ws = WorldState()
        ws.npcs["Grimjaw"] = NPCState(name="Grimjaw", location="tavern")
        delta = StateDelta(
            npc_updates=[NPCUpdate(name="grimjaw", disposition="hostile")]
        )
        rejections = ws.apply_delta(delta)
        assert rejections == []
        assert ws.npcs["Grimjaw"].disposition == "hostile"


class TestWorldStateYAML:
    """Test YAML serialization for narrator injection."""

    def test_basic_yaml(self):
        ws = WorldState(
            turn=5,
            phase="exploration",
            time_of_day="dusk",
            current_location="tavern",
            location_description="A cozy tavern with a roaring fire",
        )
        ws.sync_player("Thorin", hp=15, max_hp=20, conditions=[], concentration="")

        yml = ws.to_yaml()
        data = yaml.safe_load(yml)

        assert data["turn"] == 5
        assert data["phase"] == "exploration"
        assert data["time_of_day"] == "dusk"
        assert data["location"] == "tavern"
        assert "party" in data

    def test_tiered_npc_detail(self):
        ws = WorldState(current_location="tavern")
        ws.npcs["Barkeep"] = NPCState(
            name="Barkeep", location="tavern", disposition="friendly",
            description="A jolly halfling behind the bar",
        )
        ws.npcs["King Aldric"] = NPCState(
            name="King Aldric", location="castle", disposition="neutral",
            important=True, notes="Sent party on quest",
        )
        ws.npcs["Random Guard"] = NPCState(
            name="Random Guard", location="gate", disposition="neutral",
        )

        yml = ws.to_yaml()
        data = yaml.safe_load(yml)

        # Barkeep should be in npcs_here (at current location)
        assert "npcs_here" in data
        assert any(n["name"] == "Barkeep" for n in data["npcs_here"])

        # King should be in key_npcs_elsewhere (important, not at current location)
        assert "key_npcs_elsewhere" in data
        assert any("King Aldric" in line for line in data["key_npcs_elsewhere"])

        # Random Guard should NOT appear in YAML at all
        yml_str = yml
        assert "Random Guard" not in yml_str

    def test_empty_state_yaml(self):
        ws = WorldState()
        yml = ws.to_yaml()
        data = yaml.safe_load(yml)
        assert data["turn"] == 0
        assert "npcs_here" not in data
        assert "party" not in data

    def test_facts_and_events_in_yaml(self):
        ws = WorldState()
        ws.established_facts = ["The bridge is destroyed"]
        ws.recent_events = ["Party arrived at the village"]
        yml = ws.to_yaml()
        data = yaml.safe_load(yml)
        assert "facts" in data
        assert "recent_events" in data

    def test_flags_only_true_in_yaml(self):
        ws = WorldState()
        ws.global_flags = {"bridge_destroyed": True, "king_alive": True, "door_locked": False}
        yml = ws.to_yaml()
        data = yaml.safe_load(yml)
        assert "flags" in data
        assert data["flags"]["bridge_destroyed"] is True
        assert "door_locked" not in data["flags"]  # False flags omitted


class TestStateDeltaSchema:
    """Test JSON schema generation for structured output."""

    def test_schema_generation(self):
        schema = get_state_delta_schema()
        assert isinstance(schema, dict)
        assert "properties" in schema

    def test_schema_cached(self):
        schema1 = get_state_delta_schema()
        schema2 = get_state_delta_schema()
        assert schema1 is schema2  # Same object (cached)

    def test_empty_delta_from_json(self):
        delta = StateDelta(**{})
        assert delta.time_change is None
        assert delta.new_npcs == []
        assert delta.new_events == []
