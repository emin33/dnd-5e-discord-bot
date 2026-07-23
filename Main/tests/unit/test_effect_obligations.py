"""Pins for conservative resolved-action effect obligations."""

from __future__ import annotations

import pytest

from dnd_bot.llm.effect_obligations import (
    infer_effect_coherence_obligations,
    infer_effect_obligations,
    infer_narration_effect_obligations,
)
from dnd_bot.llm.effects import EffectType, ProposedEffect


@pytest.mark.parametrize(
    ("action", "expected"),
    [
        (
            "This is an uncontested item transfer with no roll: I hand my "
            "brass compass to Mara Venn, she accepts it, and it is now in "
            "her coat rather than my pack.",
            {EffectType.UPDATE_ENTITY, EffectType.UPDATE_PLAYER},
        ),
        (
            "This is an uncontested return with no roll: Mara Venn takes "
            "the brass compass from her coat, hands it back to me, and it "
            "is now in my pack rather than hers.",
            {EffectType.UPDATE_ENTITY, EffectType.UPDATE_PLAYER},
        ),
        (
            "I pay Mara Venn exactly two gold pieces for her help, and she "
            "accepts them.",
            {EffectType.UPDATE_PLAYER},
        ),
        (
            "Mara Venn sets a newly revealed obsidian key on the table; it "
            "is a distinct object I can pick up.",
            {EffectType.SPAWN_OBJECT},
        ),
        (
            "I pick up the obsidian key and put it in my pack.",
            {EffectType.UPDATE_PLAYER},
        ),
        (
            "Its already-armed charge destroys the sealed reliquary completely.",
            {EffectType.REMOVE_ENTITY},
        ),
        (
            "Mara Venn chooses to flee immediately, taking nothing else.",
            {EffectType.UPDATE_ENTITY},
        ),
        (
            "I leave the Copper Finch and travel to the Ash Gate, arriving "
            "beneath its cracked black arch.",
            {EffectType.CHANGE_LOCATION},
        ),
        (
            "At the Ash Gate I meet a new, physically present courier named "
            "Sable Quill.",
            {EffectType.ADD_NPC},
        ),
        (
            "Mara reveals her scar and swears to become my ally.",
            {EffectType.UPDATE_ENTITY},
        ),
    ],
)
def test_infers_high_confidence_resolved_outcomes(
    action: str,
    expected: set[EffectType],
) -> None:
    assert infer_effect_obligations(action).required_types == expected


@pytest.mark.parametrize(
    "action",
    [
        "I try to give Mara the compass.",
        "I offer Mara the compass and wait to see if she accepts.",
        "I attack the rope bridge.",
        "I ask whether Sable Quill is nearby.",
        "I start walking toward the Ash Gate.",
    ],
)
def test_ambiguous_attempts_do_not_create_obligations(action: str) -> None:
    assert not infer_effect_obligations(action).required_types


def test_transfer_denial_is_detected_but_completed_transfer_is_not() -> None:
    obligations = infer_effect_obligations(
        "I hand my compass to Mara, she accepts it, and it is now in her coat."
    )

    denied = obligations.contradiction_reasons(
        "Mara checks her coat. 'I never had it,' she says. You lost the compass."
    )
    completed = obligations.contradiction_reasons(
        "You hand over the compass. Mara accepts it and tucks it into her coat."
    )

    assert denied == ("narration denied the resolved item transfer",)
    assert completed == ()


@pytest.mark.parametrize(
    "prose",
    [
        "Your hand comes up empty. The brass compass is gone.",
        "Mara checks every pocket. 'I don't have it. I never took it.'",
        (
            "Mara says, 'I put it in my coat after you handed it to me, but "
            "I don't have it now. It's gone.'"
        ),
    ],
)
def test_live_transfer_denial_wordings_are_detected(prose: str) -> None:
    obligations = infer_effect_obligations(
        "I hand my compass to Mara, she accepts it, and it is now in her coat."
    )
    assert obligations.contradiction_reasons(prose) == (
        "narration denied the resolved item transfer",
    )


def test_completed_transfer_can_truthfully_say_item_left_source_inventory() -> None:
    obligations = infer_effect_obligations(
        "I hand my compass to Mara, she accepts it, and it is now in her coat."
    )
    prose = (
        "You hand the compass to Mara. She accepts it. The compass is gone "
        "from your pack and tucked into her coat."
    )
    assert obligations.contradiction_reasons(prose) == ()


def test_narrator_possession_correction_requires_npc_inventory_write() -> None:
    obligations = infer_narration_effect_obligations(
        "I turn the compass over in my hand and question the Cartographer.",
        (
            "Wait. The compass is not in your hand. It's in Cinder Vex's "
            "hand, clutched in her grip."
        ),
    )

    assert obligations.required_types == {EffectType.UPDATE_ENTITY}


def test_ordinary_possession_description_creates_no_narration_obligation() -> None:
    obligations = infer_narration_effect_obligations(
        "I ask Cinder what the compass means.",
        "Cinder Vex turns the compass over in her hand before answering.",
    )

    assert not obligations.required_types


@pytest.mark.parametrize(
    ("action", "prose"),
    [
        (
            "I tell Doran, \"Let's get you out of this rain and find "
            "Harrow's Drippings.\"",
            "You guide him out of the alley, into the grey morning light "
            "of the Tallow Rows. Two streets east stands Harrow's Drippings.",
        ),
        (
            "I leave this scene and ask at the nearest public crossroads "
            "where I can find Archivist Valerius, then follow the first "
            "credible direction toward them.",
            "You step out into the Tallow Rows and reach a public crossroads.",
        ),
    ],
)
def test_completed_requested_travel_requires_location_write(
    action: str,
    prose: str,
) -> None:
    obligations = infer_narration_effect_obligations(action, prose)

    assert obligations.required_types == {EffectType.CHANGE_LOCATION}


def test_incidental_motion_within_scene_creates_no_location_obligation() -> None:
    obligations = infer_narration_effect_obligations(
        "I step closer and reach for the cup.",
        "You step across the room and stop beside the table.",
    )

    assert EffectType.CHANGE_LOCATION not in obligations.required_types


def test_active_npc_deterioration_requires_entity_write() -> None:
    obligations = infer_narration_effect_obligations(
        "I ask Liraen how much longer she can endure.",
        "Liraen winces. 'The ink is eating through me. I can feel it.'",
    )

    assert obligations.required_types == {EffectType.UPDATE_ENTITY}


def test_direct_proper_name_self_introduction_requires_npc_creation() -> None:
    obligations = infer_narration_effect_obligations(
        "I ask the injured woman who she is.",
        'She steadies herself. "I\'m Elara. Elara Venn."',
    )

    assert obligations.required_types == {EffectType.ADD_NPC}


def test_generic_self_description_does_not_require_npc_creation() -> None:
    obligations = infer_narration_effect_obligations(
        "I ask the courier what happened.",
        'The courier coughs. "I\'m very badly hurt."',
    )

    assert EffectType.ADD_NPC not in obligations.required_types


def test_known_npc_reference_satisfies_self_introduction_identity_write() -> None:
    obligations = infer_narration_effect_obligations(
        "I ask the woman who she is.",
        'She answers, "I\'m Elara Venn."',
    )
    effects = [ProposedEffect(
        effect_type=EffectType.REF_ENTITY,
        ref_entity_id="elara-id",
    )]

    assert obligations.missing_from(effects) == frozenset()


def test_quoted_full_name_answer_to_direct_question_requires_npc_creation() -> None:
    obligations = infer_narration_effect_obligations(
        "I ask, 'What is your exact name, and what do you want from me?'",
        (
            'She tilts her head. "**Riven Marchetti.** " Her voice is low. '
            '"I think you want something from me."'
        ),
    )

    assert obligations.required_types == {EffectType.ADD_NPC}


def test_markdown_name_is_and_onstage_introductions_require_npc_creation() -> None:
    direct = infer_narration_effect_obligations(
        "I ask the vendor who she is.",
        'She smiles. "Name\'s **Elena Voss.** Buy something or move along."',
    )
    group = infer_narration_effect_obligations(
        "I ask the three strangers to identify themselves.",
        'The scarred man nods. "Name\'s **Garret.** This is **Sula.**"',
    )

    assert direct.required_types == {EffectType.ADD_NPC}
    assert group.required_types == {EffectType.ADD_NPC}


def test_spawned_object_visibly_held_by_npc_requires_inventory_write() -> None:
    effects = [
        ProposedEffect(
            effect_type=EffectType.REF_ENTITY,
            ref_entity_id="dorn-id",
        ),
        ProposedEffect(
            effect_type=EffectType.SPAWN_OBJECT,
            object_name="rusted iron key",
        ),
    ]
    obligations = infer_effect_coherence_obligations(
        (
            "When his hand comes back up, it's holding a rusted iron key. "
            "Dorn turns the key over in his hand."
        ),
        effects,
    )

    assert obligations.required_types == {EffectType.UPDATE_ENTITY}
