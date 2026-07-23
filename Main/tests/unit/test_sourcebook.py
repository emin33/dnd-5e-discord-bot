"""Executable contract tests for provider-neutral sourcebook packs."""

import pytest
from pydantic import ValidationError

from dnd_bot.models.sourcebook import CampaignSourcebook


def _minimal_pack() -> dict:
    return {
        "metadata": {
            "sourcebook_id": "book.glasswake",
            "title": "The Glasswake",
            "pitch": "A city pays for survival with memories.",
        },
        "locations": [
            {
                "id": "loc.veyr",
                "name": "Veyr",
                "location_kind": "settlement",
            }
        ],
        "items": [
            {
                "id": "item.brass_compass",
                "name": "Living Brass Compass",
                "significance": "Points toward memories that were taken unjustly.",
            }
        ],
        "npcs": [
            {
                "id": "npc.orla_hask",
                "name": "Orla Hask",
                "current_location_id": "loc.veyr",
                "inventory": [{"item_id": "item.brass_compass"}],
                "behavior": {
                    "goals": ["Learn who erased her apprenticeship."],
                    "decision_rules": ["Protect apprentices before institutions."],
                },
            }
        ],
        "factions": [
            {
                "id": "faction.choir",
                "name": "Choir of Anchors",
                "leader_ids": ["npc.orla_hask"],
                "headquarters_id": "loc.veyr",
            }
        ],
        "relationships": [
            {
                "id": "rel.orla_choir",
                "source_id": "npc.orla_hask",
                "target_id": "faction.choir",
                "kind": "member_of",
            }
        ],
        "claims": [
            {
                "id": "claim.orla_memory",
                "subject_id": "npc.orla_hask",
                "text": "Orla's apprenticeship was erased from the Archive.",
                "visibility": "discoverable",
            }
        ],
        "quests": [
            {
                "id": "quest.restore_orla",
                "name": "The Missing Apprenticeship",
                "giver_ids": ["npc.orla_hask"],
                "reveal_claim_ids": ["claim.orla_memory"],
                "objectives": [
                    {
                        "id": "objective.find_ledger",
                        "description": "Find the original apprenticeship ledger.",
                        "location_ids": ["loc.veyr"],
                    }
                ],
            }
        ],
        "story_arcs": [
            {
                "id": "arc.glasswake",
                "name": "What the City Forgets",
                "premise": "The memory tithe is failing.",
                "central_question": "Who deserves to choose what survives?",
                "involved_entity_ids": ["npc.orla_hask", "faction.choir"],
                "beats": [
                    {
                        "id": "beat.ledger",
                        "title": "The Altered Ledger",
                        "purpose": "Reveal the personal cost of the tithe.",
                        "reveal_claim_ids": ["claim.orla_memory"],
                    }
                ],
            }
        ],
        "starting_state": {
            "location_id": "loc.veyr",
            "opening_situation": "Upward rain whispers stolen secrets.",
            "active_quest_ids": ["quest.restore_orla"],
            "active_story_arc_ids": ["arc.glasswake"],
        },
    }


def test_sourcebook_validates_a_deeply_linked_pack():
    book = CampaignSourcebook.model_validate(_minimal_pack())

    assert book.npcs[0].inventory[0].item_id == "item.brass_compass"
    assert book.story_arcs[0].beats[0].reveal_claim_ids == ["claim.orla_memory"]
    assert book.model_json_schema()["properties"]["claims"]


def test_sourcebook_rejects_dangling_references():
    pack = _minimal_pack()
    pack["npcs"][0]["current_location_id"] = "loc.nowhere"

    with pytest.raises(ValidationError, match="references missing id 'loc.nowhere'"):
        CampaignSourcebook.model_validate(pack)


def test_sourcebook_rejects_duplicate_ownership_of_unique_items():
    pack = _minimal_pack()
    pack["npcs"].append(
        {
            "id": "npc.rival",
            "name": "Veyra Sorn",
            "current_location_id": "loc.veyr",
            "inventory": [{"item_id": "item.brass_compass"}],
        }
    )

    with pytest.raises(ValidationError, match="unique item .* is held by both"):
        CampaignSourcebook.model_validate(pack)


def test_sourcebook_rejects_location_hierarchy_cycles():
    pack = _minimal_pack()
    pack["locations"] = [
        {
            "id": "loc.upper",
            "name": "Upper Veyr",
            "location_kind": "district",
            "parent_location_id": "loc.lower",
        },
        {
            "id": "loc.lower",
            "name": "Lower Veyr",
            "location_kind": "district",
            "parent_location_id": "loc.upper",
        },
    ]
    pack["npcs"][0]["current_location_id"] = "loc.upper"
    pack["factions"][0]["headquarters_id"] = "loc.upper"
    pack["quests"][0]["objectives"][0]["location_ids"] = ["loc.upper"]
    pack["starting_state"]["location_id"] = "loc.upper"

    with pytest.raises(ValidationError, match="location hierarchy contains a cycle"):
        CampaignSourcebook.model_validate(pack)


def test_sourcebook_rejects_valid_but_wrong_reference_types():
    pack = _minimal_pack()
    pack["npcs"][0]["current_location_id"] = "faction.choir"

    with pytest.raises(ValidationError, match="npc npc.orla_hask current location"):
        CampaignSourcebook.model_validate(pack)


def test_sourcebook_rejects_cyclic_quest_objectives():
    pack = _minimal_pack()
    pack["quests"][0]["objectives"] = [
        {
            "id": "objective.find_ledger",
            "description": "Find the ledger.",
            "prerequisite_objective_ids": ["objective.decode_ledger"],
        },
        {
            "id": "objective.decode_ledger",
            "description": "Decode the ledger.",
            "prerequisite_objective_ids": ["objective.find_ledger"],
        },
    ]

    with pytest.raises(ValidationError, match="objective graph contains a cycle"):
        CampaignSourcebook.model_validate(pack)


def test_sourcebook_rejects_two_current_placements_for_a_unique_item():
    pack = _minimal_pack()
    pack["items"][0]["default_location_id"] = "loc.veyr"

    with pytest.raises(ValidationError, match="both default_location_id and NPC holder"):
        CampaignSourcebook.model_validate(pack)
