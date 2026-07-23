"""Regression fences for long-horizon memory relevance."""

from dnd_bot.llm.orchestrator import _explicit_narrative_tags


def test_narrative_tags_include_only_mentions_and_explicit_refs():
    tags = _explicit_narrative_tags(
        mentioned_entity_ids=["mentioned-npc"],
        narrator_ref_ids=["tool-ref", "mentioned-npc"],
    )

    assert tags == ["mentioned-npc", "tool-ref"]


def test_ambient_retrieval_seeds_are_not_an_input_to_narrative_tags():
    ambient_scene_seeds = ["offscene-vara", "historical-location"]

    tags = _explicit_narrative_tags([], [])

    assert tags == []
    assert not set(tags) & set(ambient_scene_seeds)
