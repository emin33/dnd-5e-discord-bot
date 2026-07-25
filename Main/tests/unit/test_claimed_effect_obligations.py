"""Pin: a deterministically CLAIMED mutation satisfies the obligation net.

The orchestrator merges a handler's ``authoritative_effects`` AFTER narration
returns, so the obligation net inside NarrationStrategy.run cannot see them.
Judging the narrator's tool calls alone, it treated an already-owned write as
missing and entered a prose+tools repair leg for it — a wasted narrator round
trip whose failure branch fails the turn closed, and a fail-closed turn now
DISCARDS the claim (so a validated transfer would silently do nothing).

Probed live against the real pipeline before the fix: a handler-driven
transfer whose narrator emitted no tools reported
``effect_obligation_missing_initial: ['update_entity', 'update_player']`` and
``resolved_outcome_repair_attempted: True``.

Reuses the Step-2 strategy harness from test_narration_strategy.
"""

from __future__ import annotations

import pytest

from dnd_bot.llm.brains.base import BrainContext
from dnd_bot.llm.effects import EffectType, ProposedEffect

from tests.fakes import narration_response
from tests.unit.test_narration_strategy import (  # noqa: F401
    _ChatOnlyClient,
    _Harness,
    _context,
    _spec,
)


TRANSFER_ACTION = (
    "This is an uncontested item transfer with no roll: I hand my brass "
    "compass to Mara Venn, she accepts it, and it is now in her coat "
    "rather than my pack."
)
TRANSFER_PROSE = "Mara takes the compass and tucks it into her coat."


def _claimed_transfer_effects() -> list[ProposedEffect]:
    """What _handle_inventory's transfer_to_npc branch hands the orchestrator."""
    return [
        ProposedEffect(
            effect_type=EffectType.UPDATE_PLAYER,
            player_item_remove=[{
                "name": "brass compass",
                "quantity": 1,
                "destination": "npc:mara-id",
            }],
        ),
        ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="mara-id",
            update_add_items=["brass compass"],
        ),
    ]


def _transfer_spec(**overrides):
    return _spec(
        action=TRANSFER_ACTION,
        player_action=TRANSFER_ACTION,
        prompt="Narrate the completed transfer.",
        prompt_role="user",
        **overrides,
    )


@pytest.mark.asyncio
async def test_claimed_effects_satisfy_obligations_and_skip_the_repair_legs():
    client = _ChatOnlyClient([
        narration_response(TRANSFER_PROSE),
        narration_response(TRANSFER_PROSE),
    ])
    harness = _Harness(client)

    prose, effects = await harness.strategy.run(
        _transfer_spec(claimed_effects=_claimed_transfer_effects()),
        _context(),
        triage=None,
    )

    diag = harness.strategy.last_diagnostics
    assert diag["effect_obligation_missing_initial"] == []
    assert diag["effect_obligation_missing_final"] == []
    # Neither obligation-driven leg fires: the write is already owned. (The
    # generic zero-effect tool followup is a separate, pre-existing mechanism
    # and still runs — the narrator may owe unrelated effects like ref_entity.)
    assert not diag.get("effect_obligation_repair_attempted")
    assert not diag.get("resolved_outcome_repair_attempted")
    assert not diag.get("resolved_outcome_failed_closed")
    # The claim is NOT re-proposed here — the orchestrator merges it after.
    assert effects == []
    assert prose == TRANSFER_PROSE


@pytest.mark.asyncio
async def test_unclaimed_transfer_still_reports_the_obligation():
    """The net keeps its teeth when nothing owns the mutation."""
    client = _ChatOnlyClient([
        narration_response(TRANSFER_PROSE),
        narration_response(TRANSFER_PROSE),
        narration_response(TRANSFER_PROSE),
    ])
    harness = _Harness(client)

    await harness.strategy.run(_transfer_spec(), _context(), triage=None)

    diag = harness.strategy.last_diagnostics
    assert "update_player" in diag["effect_obligation_missing_initial"]
    # Unowned, so the net does spend its repair budget here.
    assert len(client.calls) > 1


@pytest.mark.asyncio
async def test_partial_claim_leaves_the_unclaimed_family_visible():
    """Claiming one family must not silence the other."""
    client = _ChatOnlyClient([
        narration_response(TRANSFER_PROSE),
        narration_response(TRANSFER_PROSE),
        narration_response(TRANSFER_PROSE),
    ])
    harness = _Harness(client)

    await harness.strategy.run(
        # update_player claimed; update_entity is not.
        _transfer_spec(claimed_effects=[_claimed_transfer_effects()[0]]),
        _context(),
        triage=None,
    )

    missing = harness.strategy.last_diagnostics["effect_obligation_missing_initial"]
    assert "update_player" not in missing
    assert "update_entity" in missing
