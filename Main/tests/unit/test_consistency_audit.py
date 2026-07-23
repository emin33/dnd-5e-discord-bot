"""Unit tests for the cross-store consistency audit and the pinned-fact
resurrection-loop fixes it exists to catch."""

from types import SimpleNamespace

from dnd_bot.game.consistency_audit import run_consistency_audit
from dnd_bot.game.world_state import NPCState, WorldState
from dnd_bot.game.world_store import WorldStateStore


class _EntityType:
    def __init__(self, value):
        self.value = value


def _node(node_id, name, typ="npc", description=""):
    return SimpleNamespace(
        node_id=node_id,
        name=name,
        entity_type=_EntityType(typ),
        properties={"description": description} if description else {},
        aliases=[],
    )


def _graph(*nodes):
    return SimpleNamespace(_entities={n.node_id: n for n in nodes})


class TestHardInvariants:
    def test_clean_state_passes(self):
        npc = NPCState(name="Pell")
        world = WorldState(
            npcs={npc.id: npc},
            established_facts=["Pell tends the shrine."],
            superseded_facts=[{
                "fact": "Pell naps at noon.",
                "superseded_by": "Pell no longer naps.",
                "turn": 4,
            }],
        )
        report = run_consistency_audit(
            world_state=world,
            knowledge_graph=_graph(_node(npc.id, "Pell")),
        )
        assert report.passed
        assert report.coverage["world_npcs_with_kg_node"] == "1/1"

    def test_fact_in_both_ledgers_violates(self):
        world = WorldState(
            established_facts=["the wax is soft"],
            superseded_facts=[{
                "fact": "the wax is soft", "superseded_by": "x", "turn": 1,
            }],
        )
        report = run_consistency_audit(world_state=world)
        assert any("fact_in_both_ledgers" in v for v in report.violations)

    def test_stale_pinned_fact_violates(self):
        world = WorldState(superseded_facts=[{
            "fact": "old truth", "superseded_by": "new truth", "turn": 2,
        }])
        memory = SimpleNamespace(
            buffer=SimpleNamespace(pinned_facts=["old truth", "still fine"])
        )
        report = run_consistency_audit(world_state=world, memory=memory)
        assert any("pinned_fact_is_superseded" in v for v in report.violations)

    def test_dangling_scene_link_violates(self):
        registry = SimpleNamespace(get_all=lambda: [
            SimpleNamespace(name="Pell", npc_id="missing-id"),
        ])
        report = run_consistency_audit(
            world_state=WorldState(), scene_registry=registry,
        )
        assert any("scene_link_dangling" in v for v in report.violations)

    def test_kg_name_collision_violates(self):
        report = run_consistency_audit(
            world_state=WorldState(),
            knowledge_graph=_graph(
                _node("a", "Elara"), _node("b", "Elara"),
            ),
        )
        assert any("kg_npc_name_collision" in v for v in report.violations)

    def test_generic_kg_labels_do_not_collide(self):
        report = run_consistency_audit(
            world_state=WorldState(),
            knowledge_graph=_graph(
                _node("a", "the older woman"), _node("b", "the older woman"),
            ),
        )
        assert report.passed


class TestCoverage:
    def test_chroma_coverage_reported(self):
        class _VS:
            def indexed_entity_ids(self, campaign_id, node_ids):
                return {"a"}

        report = run_consistency_audit(
            world_state=WorldState(),
            knowledge_graph=_graph(
                _node("a", "Pell", description="tends the shrine"),
                _node("b", "Vex", description="the market courier"),
            ),
            vector_store=_VS(),
            campaign_id="camp",
        )
        assert report.coverage["described_kg_entities_indexed"] == "1/2"
        assert report.coverage["unindexed_entity_ids"] == ["b"]
        # Coverage gaps report, they do not hard-fail.
        assert report.passed


class TestResurrectionLoopFixes:
    def test_store_refuses_to_resurrect_superseded_fact(self):
        world = WorldState(superseded_facts=[{
            "fact": "old truth", "superseded_by": "new truth", "turn": 3,
        }])
        WorldStateStore(world).add_established_fact("old truth")
        assert "old truth" not in world.established_facts

    def test_buffer_retire_facts_drops_only_superseded(self):
        from dnd_bot.memory.blocks import MessageBuffer

        buffer = MessageBuffer()
        buffer._pinned_facts = ["keep", "retire me"]
        removed = buffer.retire_facts({"retire me", "never pinned"})
        assert removed == ["retire me"]
        assert buffer.pinned_facts == ["keep"]

    def test_buffer_retire_empty_set_noop(self):
        from dnd_bot.memory.blocks import MessageBuffer

        buffer = MessageBuffer()
        buffer._pinned_facts = ["keep"]
        assert buffer.retire_facts(set()) == []
        assert buffer.pinned_facts == ["keep"]
