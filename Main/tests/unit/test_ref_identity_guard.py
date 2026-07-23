"""Unit tests for the ref_entity alias misbinding guard and the SRD
fuzzy-match token requirement.

Both defects came from the 80-turn soak (20260722_230128): the narrator
bound alias "Elara" onto Lyra's roster id for three turns, and the scene
registry auto-matched the NPC "Elara" to the SRD monster "Lamia" at a 0.60
fuzzy score.
"""

from types import SimpleNamespace

from dnd_bot.game.world_state import NPCState, WorldState
from dnd_bot.llm.effects import EffectType, EffectValidator, ProposedEffect


def _world_with(*npcs: NPCState) -> WorldState:
    return WorldState(npcs={npc.id: npc for npc in npcs})


def _ref(entity_id: str, alias: str = "") -> ProposedEffect:
    return ProposedEffect(
        effect_type=EffectType.REF_ENTITY,
        ref_entity_id=entity_id,
        ref_alias_used=alias or None,
    )


class TestAliasMisbindingGuard:
    def test_alias_owned_by_different_npc_rejected(self):
        lyra = NPCState(name="Lyra")
        elara = NPCState(name="Elara")
        validator = EffectValidator(
            session=SimpleNamespace(world_state=_world_with(lyra, elara))
        )
        result = validator.validate(_ref(lyra.id, "Elara"))
        assert result.valid is False
        assert elara.id in result.rejection_reason

    def test_alias_matching_target_itself_allowed(self):
        elara = NPCState(name="Elara")
        validator = EffectValidator(
            session=SimpleNamespace(world_state=_world_with(elara))
        )
        assert validator.validate(_ref(elara.id, "Elara")).valid is True

    def test_alias_matching_target_prior_alias_allowed(self):
        figure = NPCState(name="Lyra", aliases=["the figure", "Elara"])
        other = NPCState(name="Elara")
        validator = EffectValidator(
            session=SimpleNamespace(world_state=_world_with(figure, other))
        )
        # Pollution already recorded on the target abstains rather than
        # breaking established references.
        assert validator.validate(_ref(figure.id, "Elara")).valid is True

    def test_unowned_nickname_allowed(self):
        lyra = NPCState(name="Lyra")
        validator = EffectValidator(
            session=SimpleNamespace(world_state=_world_with(lyra))
        )
        assert validator.validate(_ref(lyra.id, "Duchess")).valid is True

    def test_generic_alias_allowed(self):
        lyra = NPCState(name="Lyra")
        woman = NPCState(name="the young woman")
        validator = EffectValidator(
            session=SimpleNamespace(world_state=_world_with(lyra, woman))
        )
        assert validator.validate(_ref(lyra.id, "the woman")).valid is True

    def test_structural_validator_unaffected(self):
        assert EffectValidator().validate(_ref("anyone", "Elara")).valid is True

    def test_graph_side_owner_detected(self):
        lyra = NPCState(name="Lyra")

        class _EntityType:
            value = "npc"

        graph_elara = SimpleNamespace(
            node_id="graph-elara-id",
            name="Elara",
            aliases=[],
            entity_type=_EntityType(),
        )
        graph = SimpleNamespace(
            _entities={"graph-elara-id": graph_elara},
            get_entity=lambda entity_id: (
                graph_elara if entity_id == "graph-elara-id" else None
            ),
        )
        validator = EffectValidator(
            session=SimpleNamespace(
                world_state=_world_with(lyra),
                knowledge_graph=graph,
            )
        )
        result = validator.validate(_ref(lyra.id, "Elara"))
        assert result.valid is False
        assert "graph-elara-id" in result.rejection_reason

    def test_ambiguous_ownership_abstains(self):
        lyra = NPCState(name="Lyra")
        elara_one = NPCState(name="Elara")
        elara_two = NPCState(name="Elara")
        validator = EffectValidator(
            session=SimpleNamespace(
                world_state=_world_with(lyra, elara_one, elara_two)
            )
        )
        assert validator.validate(_ref(lyra.id, "Elara")).valid is True


class TestFuzzyMonsterTokenGuard:
    def test_personal_name_no_longer_matches_unrelated_monster(self):
        from dnd_bot.data.srd import get_srd

        assert get_srd().fuzzy_match_monster("Elara") is None

    def test_shared_token_fuzzy_still_matches(self):
        from dnd_bot.data.srd import get_srd

        match = get_srd().fuzzy_match_monster("goblin chief")
        assert match is not None
        assert "goblin" in match.get("name", "").lower()

    def test_exact_match_unchanged(self):
        from dnd_bot.data.srd import get_srd

        match = get_srd().fuzzy_match_monster("lamia")
        assert match is not None
        assert match.get("name", "").lower() == "lamia"
