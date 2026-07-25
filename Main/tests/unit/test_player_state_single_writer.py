"""Pins for the one-writer-per-narrated-event player-state seam.

The receipt-vs-state agreement gate's first live runs (player_state_sweep,
20260724_231350 .. 20260724_234241) exposed deterministic handlers writing
player currency/items directly — receipt-less and stacked on top of the
narrator's update_player for the same narrated event:

T2   "I pay Mara Venn two gold for her help" triaged `purchase`:
     _handle_purchase charged 2gp AND minted the service phrase
     'investigation-assistance-from-mara-venn' as an inventory row, while
     the narrator's currency_delta -2gp ALSO executed. Paid twice.
T8   The player RECEIVING five silver triaged `purchase` with no item_name:
     the handler bought "unknown item", charging the player and minting an
     'unknown-item' row.
T9   The transfer fallback item name "item" substring-matched that junk
     'unknown item' row and transferred IT instead of the named compass.
T10  A non-pickup ("…a distinct object I can pick up") minted a literal
     'Item' row.
T2'  (run 20260724_234241) Step 5's _consume_currency charged triage's
     currency_spent on top of the narrator's receipted -2gp — double-charge
     even with gift phrasing.

The contract pinned here: deterministic handlers either claim the write as
authoritative effects (receipted update_player, replacing the narrator's) or
defer the turn to the narrator entirely; Step 5's consumption dedupes
against the turn's receipts and routes residual writes through the effect
pipeline. Every test fails against the pre-fix handlers.
"""

import pytest

from dnd_bot.llm import orchestrator as orchestrator_module
from dnd_bot.llm.brains.base import BrainContext
from dnd_bot.llm.effects import EffectExecutionResult, EffectType, ProposedEffect
from dnd_bot.llm.orchestrator import (
    DMOrchestrator,
    TriageResult,
    _scrub_claimed_mutations,
    _srd_goods_match,
)
from dnd_bot.game.session import GameSession
from dnd_bot.game.world_state import NPCState, WorldState
from dnd_bot.models import InventoryItem
from dnd_bot.models.inventory import Currency


def _async_return(value):
    async def _coro():
        return value
    return _coro()


class _LedgerRepo:
    """Fake inventory repo that records every write."""

    def __init__(self, items: list[InventoryItem], gold: int = 10):
        self.items = items
        self.gold = gold
        self.writes: list[tuple] = []

    async def get_all_items(self, character_id: str) -> list[InventoryItem]:
        return [i for i in self.items if i.character_id == character_id]

    async def get_currency(self, character_id: str) -> Currency:
        return Currency(character_id=character_id, gold=self.gold)

    async def add_item(self, item: InventoryItem) -> InventoryItem:
        self.writes.append(("add_item", item.item_name, item.quantity))
        return item

    async def remove_item(self, item_id: str, quantity: int = 1) -> bool:
        self.writes.append(("remove_item", item_id, quantity))
        return True

    async def remove_gold(self, character_id: str, amount: int):
        self.writes.append(("remove_gold", amount))
        return True, Currency(character_id=character_id, gold=self.gold - amount)

    async def update_currency(self, currency: Currency) -> None:
        self.writes.append(("update_currency", currency.gold))


class _RecordingExecutor:
    """Succeeds every effect and remembers it, echoing an applied payload."""

    def __init__(self):
        self.executed: list[ProposedEffect] = []
        self.keys: list[str] = []
        self.acting_character_id = None

    async def execute(self, effect, idempotency_key=None):
        self.executed.append(effect)
        self.keys.append(idempotency_key)
        applied = {}
        if effect.player_currency_delta:
            applied["currency_delta"] = dict(effect.player_currency_delta)
        if effect.player_item_remove:
            applied["items_removed"] = list(effect.player_item_remove)
        if effect.player_item_grant:
            applied["items_granted"] = list(effect.player_item_grant)
        return EffectExecutionResult(
            effect=effect, success=True, details={"applied": applied},
        )


@pytest.fixture
def rig(mock_character, monkeypatch):
    """Session + orchestrator + ledger repo + recording executor."""
    session = GameSession(
        id="sess", channel_id=902_201, guild_id=1, campaign_id="camp"
    )
    session.add_player(12345, "Test Hero", mock_character)
    session.world_state = WorldState(current_location="Copper Finch")
    mara = NPCState(
        id="mara-id",
        name="Mara Venn",
        location="Copper Finch",
        inventory=["test draught", "silver antidote"],
    )
    session.world_state.npcs[mara.id] = mara

    orch = DMOrchestrator()
    orch.set_session(session)
    executor = _RecordingExecutor()
    orch._effect_executor = executor

    repo = _LedgerRepo(
        [
            InventoryItem(
                character_id=mock_character.id,
                item_index="unknown-item",
                item_name="unknown item",
                quantity=1,
            ),
            InventoryItem(
                character_id=mock_character.id,
                item_index="brass-compass",
                item_name="brass compass",
                quantity=1,
            ),
            InventoryItem(
                character_id=mock_character.id,
                item_index="arrow",
                item_name="Arrow",
                quantity=20,
            ),
        ],
        gold=10,
    )
    monkeypatch.setattr(
        orchestrator_module, "get_inventory_repo", lambda: _async_return(repo)
    )
    context = BrainContext(campaign_id="camp", session_id="sess")
    return orch, repo, executor, context


# ── Purchase: claim the write or stay out of the turn ────────────────────────


@pytest.mark.asyncio
async def test_purchase_emits_authoritative_effects_not_direct_writes(rig):
    orch, repo, _, context = rig
    triage = TriageResult(
        action_type="purchase", reasoning="",
        item_name="Dagger", item_cost=2, quantity=1,
    )

    result = await orch._handle_purchase(triage, "Test Hero", context)

    assert result["success"] is True
    assert result["gold_after"] == 8
    (effect,) = result["authoritative_effects"]
    assert effect.effect_type == EffectType.UPDATE_PLAYER
    assert effect.player_currency_delta == {"gp": -2}
    assert effect.player_item_grant == [{"name": "Dagger", "quantity": 1}]
    # Read-only until the effect pipeline commits it: the old handler's
    # direct remove_gold/add_item here was the second, unreceipted writer.
    assert repo.writes == []


@pytest.mark.asyncio
async def test_purchase_without_item_name_defers_to_narrator(rig):
    # Live T8: the player RECEIVING money triaged `purchase` with no
    # item_name — the old handler bought "unknown item" with real gold.
    orch, repo, _, context = rig
    triage = TriageResult(action_type="purchase", reasoning="", item_cost=2)

    result = await orch._handle_purchase(triage, "Test Hero", context)

    assert result is None
    assert repo.writes == []


@pytest.mark.asyncio
async def test_purchase_of_service_phrase_defers_to_narrator(rig):
    # Live T2: a social payment mistriaged as `purchase` minted the service
    # phrase as goods and charged the player on top of the narrator's delta.
    orch, repo, _, context = rig
    triage = TriageResult(
        action_type="purchase", reasoning="",
        item_name="investigation assistance from Mara Venn", item_cost=2,
    )

    result = await orch._handle_purchase(triage, "Test Hero", context)

    assert result is None
    assert repo.writes == []


@pytest.mark.asyncio
async def test_purchase_insufficient_gold_fails_closed_without_writes(rig):
    orch, repo, _, context = rig
    triage = TriageResult(
        action_type="purchase", reasoning="",
        item_name="Chain Mail", item_cost=75, quantity=1,
    )

    result = await orch._handle_purchase(triage, "Test Hero", context)

    assert result["success"] is False
    assert "Not enough gold" in result["error"]
    assert "authoritative_effects" not in result
    assert repo.writes == []


# ── Inventory: resolution from real holdings, mutations as effects ───────────


@pytest.mark.asyncio
async def test_transfer_resolves_item_named_in_action_text(rig):
    # Live T9: with no triage item_name the old "item" fallback
    # substring-matched the junk 'unknown item' row and transferred THAT;
    # the named compass never moved.
    orch, repo, _, context = rig
    triage = TriageResult(action_type="inventory", reasoning="")

    result = await orch._handle_inventory(
        triage, "Test Hero", context,
        "I hand my brass compass to Mara Venn, she accepts it, and it is "
        "now in her coat rather than my pack.",
    )

    assert result["operation"] == "transfer_to_npc"
    assert result["success"] is True
    assert result["item"] == "brass compass"
    player_effect, npc_effect = result["authoritative_effects"]
    assert player_effect.player_item_remove == [{
        "name": "brass compass",
        "quantity": 1,
        "destination": "npc:mara-id",
    }]
    assert npc_effect.update_add_items == ["brass compass"]
    assert repo.writes == []


@pytest.mark.asyncio
async def test_transfer_with_unresolvable_item_defers_to_narrator(rig):
    orch, repo, _, context = rig
    triage = TriageResult(action_type="inventory", reasoning="")

    result = await orch._handle_inventory(
        triage, "Test Hero", context,
        "Mara Venn takes something small from her coat and hands it to me.",
    )

    assert result["operation"] == "transfer_from_npc"
    assert result["success"] is True
    assert "authoritative_effects" not in result
    assert repo.writes == []


@pytest.mark.asyncio
async def test_pickup_emits_authoritative_grant(rig):
    orch, repo, _, context = rig
    triage = TriageResult(
        action_type="inventory", reasoning="", item_name="obsidian key",
    )

    result = await orch._handle_inventory(
        triage, "Test Hero", context,
        "I pick up the obsidian key from the table and put it in my pack.",
    )

    assert result["operation"] == "pickup"
    assert result["success"] is True
    (effect,) = result["authoritative_effects"]
    assert effect.effect_type == EffectType.UPDATE_PLAYER
    assert effect.player_item_grant == [{"name": "Obsidian Key", "quantity": 1}]
    assert repo.writes == []


@pytest.mark.asyncio
async def test_pickup_without_resolvable_item_defers(rig):
    # Live T10: "…it is a distinct object I can pick up" is not a pickup —
    # the old fallback minted a literal 'Item' inventory row.
    orch, repo, _, context = rig
    triage = TriageResult(action_type="inventory", reasoning="")

    result = await orch._handle_inventory(
        triage, "Test Hero", context,
        "Mara Venn sets a newly revealed obsidian key on the table between "
        "us; it is a distinct object I can pick up.",
    )

    assert result is None
    assert repo.writes == []


@pytest.mark.asyncio
async def test_use_defers_consumption_to_narrator(rig):
    # Live T6 class: the old `use` branch fed [RESULT: FAILURE] to the
    # narrator for an established automatic effect. Consumption is
    # narrator-owned, with Step 5's receipted net as backstop.
    orch, repo, _, context = rig
    triage = TriageResult(
        action_type="inventory", reasoning="", item_name="silver antidote",
    )

    result = await orch._handle_inventory(
        triage, "Test Hero", context,
        "I drink the silver antidote from my pack.",
    )

    assert result is None
    assert repo.writes == []


# ── Step 5: consumption dedupes against receipts and receipts its writes ─────


def _receipt(applied: dict) -> dict:
    return {
        "type": "update_player",
        "was_duplicate": False,
        "details": {"applied": applied},
    }


@pytest.mark.asyncio
async def test_consume_currency_skips_when_turn_already_receipted(rig):
    # Live run 20260724_234241 T2: narrator receipted -2gp AND Step 5
    # silently removed two more for the same narrated payment.
    orch, repo, executor, _ = rig
    orch._last_effect_executions = [_receipt({"currency_delta": {"gp": -2}})]

    await orch._consume_currency({"gold": 2}, "Test Hero")

    assert executor.executed == []
    assert repo.writes == []


@pytest.mark.asyncio
async def test_consume_currency_writes_once_with_receipt(rig):
    orch, repo, executor, _ = rig
    orch._last_effect_executions = []

    await orch._consume_currency({"gold": 2}, "Test Hero")

    (effect,) = executor.executed
    assert effect.effect_type == EffectType.UPDATE_PLAYER
    assert effect.player_currency_delta == {"gp": -2}
    # The write is receipted in the turn's executed-effects ledger (and
    # counted as proposed, keeping effect accounting balanced).
    (record,) = orch._last_effect_executions
    assert record["type"] == "update_player"
    assert record["details"]["applied"] == {"currency_delta": {"gp": -2}}
    assert len(orch._last_deterministic_proposed) == 1
    # The repo write happens inside the (stubbed) executor — never directly.
    assert repo.writes == []


@pytest.mark.asyncio
async def test_consume_resources_dedupes_receipted_removal(rig):
    orch, repo, executor, _ = rig
    orch._last_effect_executions = [
        _receipt({"items_removed": [{"name": "Arrow", "quantity": 2}]})
    ]

    await orch._consume_resources([{"item": "Arrow", "quantity": 2}], "Test Hero")

    assert executor.executed == []
    assert repo.writes == []


@pytest.mark.asyncio
async def test_consume_resources_writes_residual_with_receipt(rig):
    orch, repo, executor, _ = rig
    orch._last_effect_executions = []

    await orch._consume_resources([{"item": "Arrow", "quantity": 2}], "Test Hero")

    (effect,) = executor.executed
    assert effect.player_item_remove == [{"name": "Arrow", "quantity": 2}]
    (record,) = orch._last_effect_executions
    assert record["type"] == "update_player"
    assert record["details"]["applied"] == {
        "items_removed": [{"name": "Arrow", "quantity": 2}]
    }
    assert repo.writes == []


@pytest.mark.asyncio
async def test_repeated_resource_entry_dedupes_against_its_own_write(rig):
    """The dedup set must see writes this loop just made.

    Snapshotting receipts once before the loop let a repeated entry consume
    the item a second time — the same double-write shape, self-inflicted.
    """
    orch, repo, executor, _ = rig
    orch._last_effect_executions = []

    await orch._consume_resources(
        [{"item": "Arrow", "quantity": 2}, {"item": "Arrow", "quantity": 2}],
        "Test Hero",
    )

    assert len(executor.executed) == 1


@pytest.mark.asyncio
async def test_step5_idempotency_keys_are_turn_stable(rig):
    """Keys derive from session+turn, so a retried turn collapses instead
    of consuming twice — matching the narrator path's derivation."""
    orch, _, executor, _ = rig

    orch._last_effect_executions = []
    orch._last_deterministic_proposed = []
    await orch._consume_currency({"gold": 2}, "Test Hero")

    # Same turn replayed (receipts cleared as they would be on a retry).
    orch._last_effect_executions = []
    orch._last_deterministic_proposed = []
    await orch._consume_currency({"gold": 2}, "Test Hero")

    assert len(executor.keys) == 2
    assert executor.keys[0] == executor.keys[1]
    assert "turn-" in executor.keys[0] and "step5" in executor.keys[0]


# ── Claim scrubbing: per field/entity, not per effect type ───────────────────


def test_scrub_drops_only_the_claimed_player_field():
    """The narrator's mirror of a claimed family goes; an unrelated
    mutation on the same turn survives.

    Type-level replacement would have thrown away narrated HP damage taken
    during a purchase turn — a silent state loss traded for the dedup.
    """
    authoritative = [ProposedEffect(
        effect_type=EffectType.UPDATE_PLAYER,
        player_currency_delta={"gp": -2},
        player_item_grant=[{"name": "Dagger", "quantity": 1}],
    )]
    narrator = [
        ProposedEffect(  # the duplicate payment + grant: both claimed
            effect_type=EffectType.UPDATE_PLAYER,
            player_currency_delta={"gp": -2},
            player_item_grant=[{"name": "Dagger", "quantity": 1}],
        ),
        ProposedEffect(  # unrelated: the shopkeeper's thug hit the player
            effect_type=EffectType.UPDATE_PLAYER,
            player_hp_delta=-3,
            player_damage_type="bludgeoning",
        ),
    ]

    kept = _scrub_claimed_mutations(narrator, authoritative)

    assert len(kept) == 1
    assert kept[0].player_hp_delta == -3
    assert kept[0].player_currency_delta == {}
    assert kept[0].player_item_grant == []


def test_scrub_keeps_unclaimed_fields_of_a_partially_claimed_effect():
    authoritative = [ProposedEffect(
        effect_type=EffectType.UPDATE_PLAYER,
        player_item_remove=[{"name": "brass compass", "quantity": 1}],
    )]
    narrator = [ProposedEffect(
        effect_type=EffectType.UPDATE_PLAYER,
        player_item_remove=[{"name": "brass compass", "quantity": 1}],
        player_add_conditions=["poisoned"],
    )]

    (kept,) = _scrub_claimed_mutations(narrator, authoritative)

    assert kept.player_item_remove == []
    assert kept.player_add_conditions == ["poisoned"]


def test_scrub_drops_entity_updates_only_for_the_claimed_entity():
    authoritative = [ProposedEffect(
        effect_type=EffectType.UPDATE_ENTITY,
        update_entity_id="mara-id",
        update_add_items=["brass compass"],
    )]
    narrator = [
        ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="mara-id",
            update_add_items=["brass compass"],
        ),
        ProposedEffect(
            effect_type=EffectType.UPDATE_ENTITY,
            update_entity_id="barkeep-id",
            update_disposition="hostile",
        ),
    ]

    (kept,) = _scrub_claimed_mutations(narrator, authoritative)

    assert kept.update_entity_id == "barkeep-id"


def test_scrub_leaves_unrelated_effect_types_alone():
    authoritative = [ProposedEffect(
        effect_type=EffectType.UPDATE_PLAYER,
        player_currency_delta={"gp": -2},
    )]
    narrator = [ProposedEffect(
        effect_type=EffectType.REF_ENTITY, ref_entity_id="mara-id",
    )]

    assert _scrub_claimed_mutations(narrator, authoritative) == narrator


# ── Goods classification: false negatives are free, false positives cost ─────


@pytest.mark.parametrize("phrase", [
    "Dagger",
    "daggers",
    "Healing Potion",       # token-set equality finds "Potion of Healing"
    "a healing potion",
    "longsword",
    "Chain Mail",
])
def test_real_srd_goods_are_transactable(phrase):
    assert _srd_goods_match(phrase) is True


@pytest.mark.parametrize("phrase", [
    "investigation assistance from Mara Venn",  # live T2: a service phrase
    "unknown item",                             # live T8: the old fallback
    "her help",
    "information",
    "five silver pieces",                       # money, not goods
    "two gold pieces",
    # Bare currency words must not partial-match a catalog name: subset
    # matching sold "Silver Dragon Scale Mail" to a player being PAID.
    "silver",
    "gold",
])
def test_non_goods_phrases_are_not_transactable(phrase):
    assert _srd_goods_match(phrase) is False
