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
    is_valid_phase_transition,
    PHASE_TRANSITIONS,
    PHASE_STYLE_HINTS,
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
        """NPCs are stored keyed by stable UUID (NPCState.id), not by name.
        ``find_npc(name)`` resolves through the name → alias → id chain."""
        ws = WorldState(current_location="tavern")
        npc = NPCState(name="Grimjaw", disposition="unfriendly", description="A scarred dwarf")
        delta = StateDelta(new_npcs=[npc])
        rejections = ws.apply_delta(delta)
        assert rejections == []
        # Stored under the auto-generated UUID
        assert npc.id in ws.npcs
        # Resolvable by name
        found = ws._find_npc("Grimjaw")
        assert found is not None
        assert found.location == "tavern"  # defaults to current location
        assert found.disposition == "unfriendly"

    def test_apply_two_new_npcs_with_same_name_NOT_rejected(self):
        """Name uniqueness is no longer enforced at the data-model layer.
        Dedup is now the brain judge's job (orchestrator-side), so the
        underlying store accepts both. Each gets a distinct UUID. This
        test pins the new behavior so a regression that re-adds name
        uniqueness here is caught."""
        ws = WorldState()
        first = NPCState(name="Grimjaw", disposition="neutral")
        ws.npcs[first.id] = first
        delta = StateDelta(
            new_npcs=[NPCState(name="Grimjaw", disposition="hostile")]
        )
        rejections = ws.apply_delta(delta)
        assert rejections == []  # data layer doesn't reject same-name NPCs
        assert len(ws.npcs) == 2

    def test_apply_npc_update(self):
        """Update resolves the target by name → id chain."""
        ws = WorldState()
        npc = NPCState(name="Grimjaw", location="tavern", disposition="neutral")
        ws.npcs[npc.id] = npc
        delta = StateDelta(
            npc_updates=[NPCUpdate(name="Grimjaw", disposition="friendly", notes="Helped the party")]
        )
        rejections = ws.apply_delta(delta)
        assert rejections == []
        found = ws._find_npc("Grimjaw")
        assert found is not None
        assert found.disposition == "friendly"
        assert found.notes == "Helped the party"
        assert found.location == "tavern"

    def test_update_resolves_name_placed_in_id_field(self):
        """Extractor sloppiness class (run 20260722_154704 T29-style): the
        NAME lands in the id field. The NPCUpdate contract promises
        id -> name -> alias resolution; the id value must get the same
        name/alias/slug fallback instead of a hard miss."""
        ws = WorldState()
        npc = NPCState(name="Vex Harlow", disposition="neutral")
        ws.npcs[npc.id] = npc
        delta = StateDelta(
            npc_updates=[NPCUpdate(id="Vex Harlow", disposition="friendly")]
        )
        rejections = ws.apply_delta(delta)
        assert rejections == []
        assert ws.npcs[npc.id].disposition == "friendly"

    def test_update_resolves_slug_placed_in_id_field(self):
        ws = WorldState()
        npc = NPCState(name="Vex Harlow")
        ws.npcs[npc.id] = npc
        delta = StateDelta(
            npc_updates=[NPCUpdate(id="vex-harlow", notes="slug dialect")]
        )
        rejections = ws.apply_delta(delta)
        assert rejections == []
        assert ws.npcs[npc.id].notes == "slug dialect"

    def test_update_with_item_slug_in_id_field_still_rejected(self):
        """A non-NPC referent ('small-pouch-of-grey-ash') keeps failing
        closed — the fallback is exact NPC resolution, not a rescue."""
        ws = WorldState()
        npc = NPCState(name="Vex Harlow")
        ws.npcs[npc.id] = npc
        delta = StateDelta(
            npc_updates=[NPCUpdate(id="small-pouch-of-grey-ash", notes="x")]
        )
        rejections = ws.apply_delta(delta)
        assert len(rejections) == 1
        assert "not found" in rejections[0]

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
        npc = NPCState(name="Grimjaw", alive=False)
        ws.npcs[npc.id] = npc
        delta = StateDelta(
            npc_updates=[NPCUpdate(name="Grimjaw", disposition="hostile")]
        )
        rejections = ws.apply_delta(delta)
        assert len(rejections) == 1
        assert "Dead NPC" in rejections[0]

    def test_generic_delta_cannot_revive_dead_npc(self):
        ws = WorldState()
        npc = NPCState(name="Grimjaw", alive=False)
        ws.npcs[npc.id] = npc
        delta = StateDelta(
            npc_updates=[NPCUpdate(name="Grimjaw", alive=True, disposition="neutral")]
        )
        rejections = ws.apply_delta(delta)
        assert len(rejections) == 1
        assert "authoritative transition" in rejections[0]
        assert ws._find_npc("Grimjaw").alive is False

    def test_authoritative_transition_can_revive_dead_npc(self):
        ws = WorldState()
        npc = NPCState(name="Grimjaw", alive=False)
        ws.npcs[npc.id] = npc

        revived = ws.revive_npc(
            npc.id,
            authoritative_reason="raise dead spell resolved",
        )

        assert revived is True
        assert npc.alive is True
        assert "raise dead spell resolved" in npc.notes
        assert ws.recent_events == [
            "Grimjaw was revived: raise dead spell resolved"
        ]

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

    def test_reject_invalid_phase_transition(self):
        ws = WorldState(phase="combat")
        delta = StateDelta(phase_change="shopping")  # Can't shop mid-combat
        rejections = ws.apply_delta(delta)
        assert len(rejections) == 1
        assert "Invalid phase transition" in rejections[0]
        assert ws.phase == "combat"  # Unchanged

    def test_valid_phase_transition(self):
        ws = WorldState(phase="combat")
        delta = StateDelta(phase_change="exploration")  # Valid: combat -> exploration
        rejections = ws.apply_delta(delta)
        assert rejections == []
        assert ws.phase == "exploration"

    def test_case_insensitive_npc_lookup(self):
        ws = WorldState()
        ws.npcs["Grimjaw"] = NPCState(name="Grimjaw", location="tavern")
        delta = StateDelta(
            npc_updates=[NPCUpdate(name="grimjaw", disposition="hostile")]
        )
        rejections = ws.apply_delta(delta)
        assert rejections == []
        assert ws.npcs["Grimjaw"].disposition == "hostile"


class TestSceneRescopeOnLocationChange:
    """DF-18: a real location change drops the old scene's transients.

    ``scene_items`` always clear; non-important NPCs not recorded at the
    new location leave the scene-scoped roster (never killed — DB/KG
    untouched); important NPCs stay for ``key_npcs_elsewhere``. The
    rescope is deferred to the END of the delta so same-delta updates and
    additions resolve against the pre-move roster first.
    """

    def _world_with_scene(self) -> tuple:
        ws = WorldState(current_location="tavern")
        barkeep = NPCState(name="Barkeep", location="tavern")
        guard = NPCState(name="Cellar Guard", location="cellar")
        king = NPCState(name="King", location="castle", important=True)
        for npc in (barkeep, guard, king):
            ws.npcs[npc.id] = npc
        ws.spawn_item("Rusty Key", "an old iron key")
        return ws, barkeep, guard, king

    def test_location_change_clears_scene_items(self):
        ws, *_ = self._world_with_scene()
        rejections = ws.apply_delta(StateDelta(location_change="cellar"))
        assert rejections == []
        assert ws.scene_items == {}

    def test_location_change_rescopes_roster(self):
        ws, barkeep, guard, king = self._world_with_scene()
        ws.apply_delta(StateDelta(location_change="cellar"))
        assert barkeep.id not in ws.npcs   # old room -> left scene scope
        assert guard.id in ws.npcs         # recorded at the new location
        assert king.id in ws.npcs          # important stays (key_npcs_elsewhere)
        assert barkeep.alive is True       # scene scope only, never killed

    def test_departed_npc_not_name_resolvable_after_move(self):
        ws, barkeep, _, _ = self._world_with_scene()
        ws.apply_delta(StateDelta(removed_npcs=["Barkeep"]))  # departs: location=""
        assert ws._find_npc("Barkeep") is not None  # pre-move: unchanged behavior
        ws.apply_delta(StateDelta(location_change="cellar"))
        assert ws._find_npc("Barkeep") is None

    def test_same_delta_npc_update_moves_npc_with_party(self):
        ws, barkeep, _, _ = self._world_with_scene()
        delta = StateDelta(
            location_change="cellar",
            npc_updates=[NPCUpdate(name="Barkeep", location="cellar")],
        )
        rejections = ws.apply_delta(delta)
        assert rejections == []
        assert barkeep.id in ws.npcs  # followed the party, survived the rescope

    def test_same_delta_new_npcs_land_in_new_scene(self):
        ws, *_ = self._world_with_scene()
        new_npc = NPCState(name="Rat Catcher")  # location defaults to the new one
        ws.apply_delta(StateDelta(location_change="cellar", new_npcs=[new_npc]))
        assert new_npc.id in ws.npcs
        assert ws.npcs[new_npc.id].location == "cellar"

    def test_restated_location_does_not_rescope(self):
        ws, barkeep, _, _ = self._world_with_scene()
        ws.apply_delta(StateDelta(location_change="Tavern"))  # case-only restatement
        assert ws.scene_items == {"Rusty Key": "an old iron key"}
        assert barkeep.id in ws.npcs

    def test_qualified_same_location_does_not_drop_new_npc(self):
        ws = WorldState(current_location="Ash Gate")
        courier = NPCState(name="Sable Quill", location="Ash Gate")
        ws.npcs[courier.id] = courier
        ws.spawn_item("Letter", "a sealed message")

        ws.apply_delta(StateDelta(location_change="the Ash Gate clearing"))

        assert courier.id in ws.npcs
        assert ws.scene_items == {"Letter": "a sealed message"}
        assert ws.get_npcs_at_location() == [courier]

    def test_tool_location_qualifier_does_not_drop_same_scene_npc(self):
        from dnd_bot.game.world_store import WorldStateStore
        from dnd_bot.llm.effects import EffectType, ProposedEffect

        ws = WorldState(current_location="Ash Gate")
        courier = NPCState(name="Sable Quill", location="Ash Gate")
        ws.npcs[courier.id] = courier

        WorldStateStore(ws).apply_effect(ProposedEffect(
            effect_type=EffectType.CHANGE_LOCATION,
            location_name="Ash Gate clearing",
        ))

        assert courier.id in ws.npcs
        assert ws.current_location == "Ash Gate clearing"

    def test_first_location_set_does_not_rescope(self):
        ws = WorldState()  # scene establishment, not a move
        npc = NPCState(name="Barkeep")  # minted before any location was known
        ws.npcs[npc.id] = npc
        ws.spawn_item("Rusty Key", "an old iron key")
        ws.apply_delta(StateDelta(location_change="tavern"))
        assert npc.id in ws.npcs
        assert "Rusty Key" in ws.scene_items

    def test_extractor_update_for_out_of_scene_npc_is_safe_noop(self):
        ws, barkeep, _, _ = self._world_with_scene()
        ws.apply_delta(StateDelta(location_change="cellar"))
        rejections = ws.apply_delta(StateDelta(
            npc_updates=[NPCUpdate(name="Barkeep", notes="waves goodbye")]
        ))
        assert len(rejections) == 1
        assert "not found" in rejections[0]
        assert ws._find_npc("Barkeep") is None  # not resurrected into the cellar

    def test_yaml_after_move_has_no_stale_scene(self):
        ws, *_ = self._world_with_scene()
        ws.apply_delta(StateDelta(location_change="cellar"))
        data = yaml.safe_load(ws.to_yaml())
        assert "scene_items" not in data
        assert [n["name"] for n in data.get("npcs_here", [])] == ["Cellar Guard"]


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

    def test_yaml_contains_current_scene_npcs_only(self):
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

        # Important entities remain durable in ``ws.npcs`` but do not become
        # ambient narrator context while they are off scene.
        assert "King Aldric" in {npc.name for npc in ws.npcs.values()}
        assert "key_npcs_elsewhere" not in data

        # Random Guard should NOT appear in YAML at all
        yml_str = yml
        assert "King Aldric" not in yml_str
        assert "Random Guard" not in yml_str

    def test_yaml_omits_historical_connections_from_current_exits(self):
        ws = WorldState(
            current_location="tavern",
            connected_locations=["old square", "distant harbor"],
        )

        data = yaml.safe_load(ws.to_yaml())

        assert "exits" not in data

    def test_empty_state_yaml(self):
        ws = WorldState()
        yml = ws.to_yaml()
        data = yaml.safe_load(yml)
        assert data["turn"] == 0
        assert "npcs_here" not in data
        assert "party" not in data

    def test_durable_facts_are_not_ambient_but_recent_events_are(self):
        ws = WorldState()
        ws.established_facts = ["The bridge is destroyed"]
        ws.recent_events = ["Party arrived at the village"]
        yml = ws.to_yaml()
        data = yaml.safe_load(yml)
        assert "facts" not in data
        assert ws.established_facts == ["The bridge is destroyed"]
        assert "recent_events" in data

    def test_yaml_includes_only_facts_anchored_to_current_scene(self):
        ws = WorldState(current_location="Copper Finch")
        mara = NPCState(
            name="Mara Venn",
            location="Copper Finch",
            important=True,
        )
        sera = NPCState(
            name="Sera Vellian",
            location="Tallow Market",
            important=True,
        )
        ws.npcs[mara.id] = mara
        ws.npcs[sera.id] = sera
        anonymous = NPCState(name="the woman", location="Copper Finch")
        ws.npcs[anonymous.id] = anonymous
        ws.spawn_item("sealed-reliquary", "An iron box with a bronze pin")
        ws.established_facts = [
            "Mara Venn promised to accept the brass compass.",
            "The sealed reliquary detonates when Kael says now.",
            "Sera Vellian waits at her distant market cart.",
            "The woman called Sera Vellian still has the old letter.",
        ]

        data = yaml.safe_load(ws.to_yaml())

        assert data["facts"] == ws.established_facts[:2]
        assert "Sera Vellian" not in ws.to_yaml()

    def test_flags_only_true_in_yaml(self):
        ws = WorldState()
        ws.global_flags = {"bridge_destroyed": True, "king_alive": True, "door_locked": False}
        yml = ws.to_yaml()
        data = yaml.safe_load(yml)
        assert "flags" in data
        assert data["flags"]["bridge_destroyed"] is True
        assert "door_locked" not in data["flags"]  # False flags omitted


class TestActionAnchoredFacts:
    """Facts about what the player just RAISED, not only where they stand.

    Scene anchoring alone made canon unreachable by asking for it: a fact
    whose subject was not in the room could not enter the prompt however
    directly the player named its subject, while knowledge-graph entity
    retrieval was already seeded from the action text. These pin the two
    retrieval paths to the same notion of relevance — and pin the limit
    that keeps it from becoming "everything, forever".
    """

    @staticmethod
    def _tavern() -> WorldState:
        """Copper Finch, Mara on stage, the Ash Gate a place elsewhere."""
        ws = WorldState(
            current_location="Copper Finch",
            connected_locations=["Ash Gate"],
        )
        mara = NPCState(name="Mara Venn", location="Copper Finch")
        ws.npcs[mara.id] = mara
        ws.established_facts = [
            "Mara Venn keeps the Copper Finch's ledger.",
            "The Ash Gate has been shut since Old Bram the ferryman died; "
            "his boat the Grey Hind still rots at the landing.",
        ]
        return ws

    def test_action_naming_offstage_subject_reaches_its_facts(self):
        ws = self._tavern()

        facts = ws.get_scene_relevant_facts(
            action_text=(
                "I ask Mara Venn what she knows about the Ash Gate "
                "and why it is closed."
            ),
        )

        assert "Grey Hind" in " ".join(facts)
        # The scene anchor still holds alongside it.
        assert ws.established_facts[0] in facts

    def test_unraised_offstage_facts_stay_out_of_the_prompt(self):
        ws = self._tavern()

        facts = ws.get_scene_relevant_facts(
            action_text="I order a bowl of stew and warm my hands.",
        )

        assert facts == [ws.established_facts[0]]

    def test_caller_supplied_entity_names_anchor_an_aliased_reference(self):
        """A graph match on an alias anchors the entity's CANONICAL name.

        The player says "the black arch"; the knowledge graph resolves that
        to Ash Gate and hands the name back. Fact text uses canonical names,
        so anchoring on the resolution — not on the player's wording — is
        what makes an aliased question reach canon.
        """
        ws = self._tavern()

        facts = ws.get_scene_relevant_facts(
            action_text="I ask about the black arch on the road north.",
            action_entities=["Ash Gate"],
        )

        assert "Grey Hind" in " ".join(facts)

    def test_offstage_npc_named_in_the_action_anchors_without_a_graph(self):
        ws = self._tavern()
        sera = NPCState(
            name="Sera Vellian",
            aliases=["the quartermaster"],
            location="Tallow Market",
        )
        ws.npcs[sera.id] = sera
        ws.established_facts.append("Sera Vellian still holds the old letter.")

        by_name = ws.get_scene_relevant_facts(
            action_text="I ask around after Sera Vellian.",
        )
        by_alias = ws.get_scene_relevant_facts(
            action_text="I ask around after the quartermaster.",
        )

        assert "Sera Vellian still holds the old letter." in by_name
        assert "Sera Vellian still holds the old letter." in by_alias

    def test_anchors_match_on_token_boundaries_not_substrings(self):
        """The WorldState-local half of the rule.

        The graph-resolved half is pinned against a real graph in
        `test_knowledge_graph.py` and end to end in `test_sourcebook_apply`.
        Asserting it here with `action_text` alone would vouch for a
        guarantee the shipped path does not get from this code.
        """
        ws = self._tavern()

        facts = ws.get_scene_relevant_facts(
            action_text="I study the Ash Gateway Ledger, a bound volume.",
        )

        assert "Grey Hind" not in " ".join(facts)

    def test_a_placeholder_alias_never_anchors_its_entity(self):
        """The "not globally salient forever" clause, defended.

        Name promotion moves a descriptive placeholder into `aliases`, and
        "innkeeper" recurs in most tavern turns. Anchoring on it would make
        one off-screen NPC's whole file permanent furniture in the prompt.
        """
        ws = WorldState(current_location="Gilded Hart")
        bettan = NPCState(
            name="Bettan Roor",
            aliases=["the innkeeper"],
            location="Copper Finch",
        )
        ws.npcs[bettan.id] = bettan
        ws.established_facts = ["Bettan Roor owes the brewer eleven gold."]

        by_placeholder = ws.get_scene_relevant_facts(
            action_text="I ask the innkeeper for a room.",
        )
        by_name = ws.get_scene_relevant_facts(
            action_text="I ask Bettan Roor for a room.",
        )

        assert by_placeholder == []
        assert by_name == ws.established_facts

    def test_a_word_that_merely_contains_a_name_anchors_nothing(self):
        """The graph seeds on bare substring; anchoring must not."""
        ws = WorldState(current_location="Copper Finch")
        bram = NPCState(name="Bram", location="Tallow Market")
        ws.npcs[bram.id] = bram
        ws.established_facts = ["Bram owes the tax collector nine gold."]

        facts = ws.get_scene_relevant_facts(
            action_text="I push through the brambles toward the river.",
        )

        assert facts == []

    def test_generic_entity_names_never_become_action_anchors(self):
        """The guard on CALLER-supplied entities, not the pre-existing blocklist.

        This used to pass with `"the woman"`, which is a literal member of
        `_GENERIC_FACT_ANCHORS` — so `_normalized_anchor_set` dropped it either
        way and the assertion could not fail. These labels are dropped only by
        `is_generic_npc_label`, and they are shapes this project's own turn
        logs actually produce.
        """
        for label, fact in (
            ("the innkeeper", "The innkeeper of the Gilded Hart hid the body."),
            ("the barkeep", "The barkeep waters the ale."),
            ("serving girl", "The serving girl reports to the Guild."),
            ("the guards", "The guards were paid to look away."),
        ):
            ws = self._tavern()
            ws.established_facts.append(fact)

            facts = ws.get_scene_relevant_facts(
                action_text=f"I speak with {label}.", action_entities=[label],
            )

            assert fact not in facts, f"{label!r} anchored as if it were a name"

    def test_a_real_name_supplied_by_a_caller_still_anchors(self):
        """Positive control for the guard above: it refuses placeholders, not
        every caller-supplied entity."""
        ws = self._tavern()
        ws.established_facts.append("Sera Vellian keeps the second ledger.")

        facts = ws.get_scene_relevant_facts(
            action_text="I ask after Sera Vellian.",
            action_entities=["Sera Vellian"],
        )

        assert "Sera Vellian keeps the second ledger." in facts

    def test_action_facts_carry_their_own_budget(self):
        """Scene facts are the standing prompt cost; recall must not evict it.

        One budget for both would let a well-documented subject push the
        room the party is standing in out of the narrator's view.
        """
        ws = WorldState(current_location="Copper Finch")
        ws.established_facts = [
            f"The Ash Gate detail number {n}." for n in range(20)
        ] + ["The Copper Finch keeps a copper lamp burning."]

        facts = ws.get_scene_relevant_facts(
            action_text="I ask what is known about the Ash Gate.",
            action_entities=["Ash Gate"],
            max_action_facts=3,
        )

        assert "The Copper Finch keeps a copper lamp burning." in facts
        assert len([f for f in facts if "Ash Gate detail" in f]) == 3
        # Most recent wins, matching the scene projection's recency rule.
        assert "The Ash Gate detail number 19." in facts

    def test_scene_budget_is_not_spent_on_action_facts(self):
        ws = WorldState(current_location="Copper Finch")
        ws.established_facts = [
            f"The Copper Finch detail number {n}." for n in range(4)
        ] + ["The Ash Gate is shut."]

        facts = ws.get_scene_relevant_facts(
            max_facts=4,
            action_text="I ask what is known about the Ash Gate.",
            action_entities=["Ash Gate"],
        )

        assert len(facts) == 5
        assert "The Ash Gate is shut." in facts

    def test_naming_someone_in_the_room_does_not_widen_the_scene_window(self):
        """The action budget answers a question; it does not buy more room.

        Scene facts evicted by the scene budget are scene-REACHABLE, so they
        must not compete in the action budget. Otherwise the most ordinary
        action there is — naming the NPC you are talking to — silently
        raises the standing fact block from 20 to 26 without introducing a
        single new subject.
        """
        ws = WorldState(current_location="Copper Finch")
        mara = NPCState(name="Mara Venn", location="Copper Finch")
        ws.npcs[mara.id] = mara
        ws.established_facts = [f"Mara Venn fact {n}." for n in range(40)]

        scene_only = ws.get_scene_relevant_facts()
        naming_her = ws.get_scene_relevant_facts(
            action_text="I ask Mara Venn a question.",
        )

        assert len(scene_only) == 20
        assert naming_her == scene_only

    def test_the_distinctive_offscene_fact_beats_a_scene_fact_for_the_budget(
        self,
    ):
        """Recency decides within an anchor, never across them."""
        ws = WorldState(
            current_location="Copper Finch", connected_locations=["Ash Gate"],
        )
        ws.established_facts = [
            "The Ash Gate's keystone bears the seal of the drowned king.",
            "The Ash Gate road runs past the Copper Finch.",
            "The Ash Gate caravan stops at the Copper Finch.",
            "Copper Finch A.",
            "Copper Finch B.",
        ]

        facts = ws.get_scene_relevant_facts(
            max_facts=2,
            max_action_facts=1,
            action_text="What is known about the Ash Gate?",
        )

        # Index 0 is the only fact scene anchoring could never reach.
        assert facts[0] == ws.established_facts[0]
        assert facts == [
            ws.established_facts[0],
            ws.established_facts[3],
            ws.established_facts[4],
        ]

    def test_facts_are_projected_in_ledger_order(self):
        ws = self._tavern()

        facts = ws.get_scene_relevant_facts(
            action_text="I ask Mara Venn about the Ash Gate.",
        )

        assert facts == ws.established_facts

    def test_asking_after_someone_does_not_put_them_on_stage(self):
        """Anchoring widens FACTS only — never scene membership."""
        ws = self._tavern()
        sera = NPCState(name="Sera Vellian", location="Tallow Market")
        ws.npcs[sera.id] = sera

        data = yaml.safe_load(
            ws.to_yaml(action_text="I ask around after Sera Vellian.")
        )

        assert [n["name"] for n in data["npcs_here"]] == ["Mara Venn"]

    def test_no_action_text_projects_exactly_the_scene(self):
        ws = self._tavern()

        assert ws.get_scene_relevant_facts() == [ws.established_facts[0]]

    def test_action_anchoring_alone_projects_facts_in_an_empty_scene(self):
        """A scene with no anchors is not a reason to withhold the answer."""
        ws = WorldState()
        ws.established_facts = ["The Ash Gate has been shut for years."]

        facts = ws.get_scene_relevant_facts(
            action_text="I ask what is known about the Ash Gate.",
            action_entities=["Ash Gate"],
        )

        assert facts == ws.established_facts

    def test_action_facts_can_be_disabled_by_budget(self):
        ws = self._tavern()

        facts = ws.get_scene_relevant_facts(
            action_text="I ask Mara Venn about the Ash Gate.",
            action_entities=["Ash Gate"],
            max_action_facts=0,
        )

        assert facts == [ws.established_facts[0]]

    def test_action_entities_naming_nothing_in_the_ledger_add_no_facts(self):
        ws = self._tavern()

        facts = ws.get_scene_relevant_facts(
            action_text="I ask Toran Vex about the weather.",
            action_entities=["Toran Vex"],
        )

        assert facts == [ws.established_facts[0]]


class TestCanonOutranksRecency:
    """Authored canon must survive a long campaign about its own subject.

    Both budgets were pure recency slices, and canon is installed at turn 0
    — so the book held the oldest ledger positions and was always evicted
    first, irreversibly. Six later mentions of a subject were enough to make
    the book's own line about that subject permanently unreachable, and the
    evidence the feature shipped with was all turn-1 on a fresh ledger,
    where recency and correctness happen to agree.

    Every fixture here is therefore DEEP, and every "canon is reached"
    assertion is paired with the same ledger unmarked — which fails on this
    code, so the marking is provably the only reason canon is in the output.
    """

    AUTHORED = "Old Bram the ferryman drowned the night the Ash Gate was shut."
    GATE = "The Ash Gate has been shut for years; its keystone is cracked."

    @classmethod
    def _bram(cls, newer: int, *, canon: bool) -> WorldState:
        """Off stage, named by the action, buried under `newer` play facts."""
        ws = WorldState(current_location="Copper Finch")
        bram = NPCState(name="Old Bram", location="Ash Gate")
        ws.npcs[bram.id] = bram
        ws.established_facts = [cls.AUTHORED] + [
            f"Old Bram was mentioned in passing, note {n}."
            for n in range(newer)
        ]
        if canon:
            ws.canon_facts = [cls.AUTHORED]
        return ws

    @classmethod
    def _gate(cls, *, at_the_gate: bool, newer: int, canon: bool) -> WorldState:
        """The gate's canon, buried under facts about its WARDEN.

        The warden facts name Toran Vex, never the gate — so they are scene
        facts at the gate and invisible from the tavern. That asymmetry is
        what made the standpoint decide reachability.
        """
        ws = WorldState(
            current_location="Ash Gate" if at_the_gate else "Copper Finch",
            connected_locations=["Ash Gate"],
        )
        warden = NPCState(name="Toran Vex", location="Ash Gate")
        ws.npcs[warden.id] = warden
        ws.established_facts = [cls.GATE] + [
            f"Toran Vex did something unremarkable, note {n}."
            for n in range(newer)
        ]
        if canon:
            ws.canon_facts = [cls.GATE]
        return ws

    @staticmethod
    def _ask_about_bram(ws: WorldState) -> list[str]:
        return ws.get_scene_relevant_facts(action_text="I ask about Old Bram.")

    @staticmethod
    def _ask_about_the_gate(ws: WorldState) -> list[str]:
        return ws.get_scene_relevant_facts(
            action_text="I ask what is known about the Ash Gate.",
        )

    # ── The action budget ────────────────────────────────────────────────

    def test_canon_survives_any_depth_of_play_about_its_subject(self):
        """The measured ceiling: it used to fall at exactly six."""
        for newer in (6, 12, 40, 200):
            ws = self._bram(newer, canon=True)

            assert self.AUTHORED in self._ask_about_bram(ws), (
                f"authored canon lost under {newer} newer facts"
            )

    def test_recency_alone_loses_canon_at_six(self):
        """Positive control: the same ledgers, unmarked, still fail.

        Without this the test above would pass on a build that simply
        returned more facts, and would keep passing if the marking silently
        stopped being applied.
        """
        assert self.AUTHORED in self._ask_about_bram(self._bram(5, canon=False))
        for newer in (6, 12, 40, 200):
            ws = self._bram(newer, canon=False)

            assert self.AUTHORED not in self._ask_about_bram(ws)

    def test_the_newest_play_facts_still_arrive_beside_the_canon(self):
        """Canon takes a floor, not the budget. Recent events still land."""
        ws = self._bram(40, canon=True)

        facts = self._ask_about_bram(ws)

        assert self.AUTHORED in facts
        assert "Old Bram was mentioned in passing, note 39." in facts
        assert len(facts) == 6

    # ── The scene budget, and the standpoint inversion ───────────────────

    def test_standing_at_a_place_never_hides_that_place_s_canon(self):
        """Closer must not be worse.

        The scene anchor reaches this fact and the scene budget evicted it;
        the action budget then excluded it for being scene-reachable, so it
        was reachable by nothing — while the identical question asked from
        the tavern reached it.
        """
        for newer in (25, 100):
            at_the_gate = self._gate(
                at_the_gate=True, newer=newer, canon=True
            )
            from_the_tavern = self._gate(
                at_the_gate=False, newer=newer, canon=True
            )

            assert self.GATE in self._ask_about_the_gate(at_the_gate), (
                f"standing at the gate hid its canon under {newer} facts"
            )
            assert self.GATE in self._ask_about_the_gate(from_the_tavern)

    def test_the_inversion_is_what_the_marking_removed(self):
        """Positive control: unmarked, the standpoint still decides.

        Reached from the tavern, unreachable at the gate — the same ledger,
        the same question, opposite answers.
        """
        unreachable = self._gate(at_the_gate=True, newer=25, canon=False)
        reachable = self._gate(at_the_gate=False, newer=25, canon=False)

        assert self.GATE not in self._ask_about_the_gate(unreachable)
        assert self.GATE in self._ask_about_the_gate(reachable)

    def test_canon_reaches_the_scene_projection_with_no_action_at_all(self):
        """Not only the answer to a question — the standing room block too."""
        ws = self._gate(at_the_gate=True, newer=100, canon=True)

        assert self.GATE in ws.get_scene_relevant_facts()

    # ── The floor is a floor, in both directions ─────────────────────────

    def test_canon_cannot_crowd_out_what_just_happened(self):
        """A book with more facts about a room than the budget holds.

        Canon-always-wins would leave the narrator describing a room by the
        book while blind to the fight that just ended in it.
        """
        ws = WorldState(current_location="Ash Gate")
        ws.established_facts = (
            [f"The Ash Gate authored line {n}." for n in range(60)]
            + [f"The Ash Gate saw something happen just now, {n}." for n in range(5)]
        )
        ws.canon_facts = ws.established_facts[:60]

        facts = ws.get_scene_relevant_facts()

        assert len(facts) == 20
        assert len([f for f in facts if "just now" in f]) == 5

    def test_canon_takes_the_slack_when_play_underfills(self):
        """The other direction: an unspent play share flows back to canon."""
        ws = WorldState(current_location="Ash Gate")
        ws.established_facts = (
            [f"The Ash Gate authored line {n}." for n in range(60)]
            + ["The Ash Gate saw something happen just now."]
        )
        ws.canon_facts = ws.established_facts[:60]

        facts = ws.get_scene_relevant_facts()

        assert len(facts) == 20
        # 19 canon, not the bare floor of 10 — play left the room.
        assert len([f for f in facts if "authored line" in f]) == 19

    def test_canon_is_taken_in_authored_order_and_holds_still(self):
        """A stable window, not a jittering one.

        Authored order is the author's own ordering, and it does not move as
        play appends — the narrator sees the same lines turn after turn
        instead of a churn that costs prompt cache for nothing.
        """
        ws = WorldState(current_location="Ash Gate")
        ws.established_facts = [
            f"The Ash Gate authored line {n}." for n in range(60)
        ]
        ws.canon_facts = list(ws.established_facts)

        before = ws.get_scene_relevant_facts()
        for n in range(30):
            ws.established_facts.append(f"The Ash Gate then saw event {n}.")
        after = ws.get_scene_relevant_facts()

        assert before == [f"The Ash Gate authored line {n}." for n in range(20)]
        assert [f for f in after if "authored line" in f] == before[:10]

    # ── Campaigns with no canon marking behave exactly as before ─────────

    def test_an_unmarked_ledger_still_ranks_purely_by_recency(self):
        """No canon is not a new code path — it is the old behaviour.

        Campaigns seeded before this field existed carry no marks, and a
        scene the book never touched carries none either.
        """
        ws = WorldState(current_location="Copper Finch")
        ws.established_facts = [
            f"The Copper Finch detail number {n}." for n in range(40)
        ]

        facts = ws.get_scene_relevant_facts()

        assert facts == ws.established_facts[-20:]

    def test_marking_a_fact_that_is_not_in_the_ledger_changes_nothing(self):
        """`canon_facts` is provenance on the ledger, never a second ledger."""
        ws = WorldState(current_location="Copper Finch")
        ws.established_facts = ["The Copper Finch keeps a copper lamp burning."]
        ws.canon_facts = ["The Ash Gate has been shut for years."]

        assert ws.get_scene_relevant_facts() == ws.established_facts


class TestPhaseStateMachine:
    """Test phase FSM transitions."""

    def test_exploration_can_enter_combat(self):
        assert is_valid_phase_transition("exploration", "combat")

    def test_combat_cannot_enter_shopping(self):
        assert not is_valid_phase_transition("combat", "shopping")

    def test_combat_cannot_enter_rest(self):
        assert not is_valid_phase_transition("combat", "rest")

    def test_rest_can_be_interrupted_by_combat(self):
        assert is_valid_phase_transition("rest", "combat")

    def test_all_phases_have_transitions(self):
        for phase in ["exploration", "combat", "dialogue", "rest", "shopping"]:
            assert phase in PHASE_TRANSITIONS

    def test_all_phases_have_style_hints(self):
        for phase in ["exploration", "combat", "dialogue", "rest", "shopping"]:
            assert phase in PHASE_STYLE_HINTS
            assert len(PHASE_STYLE_HINTS[phase]) > 10


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


class TestRepeatedNameClaims:
    """A name owns ALL the ground it covers, not just its first occurrence."""

    def test_repeating_a_longer_name_does_not_free_a_span_for_a_shorter_one(self):
        from dnd_bot.game.identity import names_addressed_in_text

        addressed, unrelated = names_addressed_in_text(
            "Mara Venn, tell Mara Venn to wait.", [["Mara Venn"], ["Mara"]],
        )

        assert addressed == ["Mara Venn"]
        assert unrelated == []

    def test_naming_both_people_still_names_both(self):
        """The control: shadowing must not swallow a genuine second subject."""
        from dnd_bot.game.identity import names_addressed_in_text

        addressed, other = names_addressed_in_text(
            "I ask Mara Venn about Mara.", [["Mara Venn"], ["Mara"]],
        )

        assert addressed == ["Mara Venn"]
        assert other == ["Mara"]
