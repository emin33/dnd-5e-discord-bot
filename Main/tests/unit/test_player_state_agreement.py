"""Pin the player-state receipt-vs-state agreement (tool preflight sweep).

evaluate_player_state_agreement replays executed update_player receipts
over the initial DB snapshot and requires the result to equal the final DB
snapshot. Both failure directions are pinned: a receipt whose write never
landed (including the executor's clamp writing less than the receipt
claims), and a write that produced no receipt.
"""

from test_tool_reliability import (
    _update_player_receipts,
    evaluate_player_state_agreement,
)


def _initial():
    return {
        "currency": {
            "copper": 0, "silver": 0, "electrum": 0, "gold": 10, "platinum": 0,
        },
        "inventory": {"brass-compass": 1},
        "conditions": [],
        "spell_slots": {
            "1": [2, 2], **{str(level): [0, 0] for level in range(2, 10)},
        },
    }


def _receipts():
    # The sweep's canonical ledger: currency both directions, item grant and
    # removal, condition add then remove, one first-level slot expenditure.
    return [
        {"currency_delta": {"gp": -2}},
        {"items_granted": [{"name": "test draught", "quantity": 1}]},
        {"conditions_added": ["poisoned"]},
        {"items_granted": [{"name": "silver antidote", "quantity": 1}]},
        {"conditions_removed": ["poisoned"]},
        {"spell_slot_used": 1},
        {"currency_delta": {"sp": 5}},
        {"items_removed": [{"name": "brass compass", "quantity": 1}]},
    ]


def _consistent_final():
    return {
        "currency": {
            "copper": 0, "silver": 5, "electrum": 0, "gold": 8, "platinum": 0,
        },
        # The removed compass row is simply absent — the repo deletes
        # zero-quantity rows, and the evaluator treats absent as 0.
        "inventory": {"test-draught": 1, "silver-antidote": 1},
        "conditions": [],
        "spell_slots": {
            "1": [1, 2], **{str(level): [0, 0] for level in range(2, 10)},
        },
    }


def test_consistent_run_passes_every_check():
    checks = evaluate_player_state_agreement(
        _initial(), _consistent_final(), _receipts()
    )
    assert set(checks) == {
        "currency_receipts_match_state",
        "inventory_receipts_match_state",
        "condition_receipts_match_state",
        "spell_slot_receipts_match_state",
        "receipts_cover_all_player_state_families",
    }
    failures = {name: c["detail"] for name, c in checks.items() if not c["passed"]}
    assert not failures, failures


def test_currency_write_without_receipt_fails_and_names_denomination():
    final = _consistent_final()
    final["currency"]["gold"] = 9  # one gold moved with no receipt
    checks = evaluate_player_state_agreement(_initial(), final, _receipts())
    assert not checks["currency_receipts_match_state"]["passed"]
    assert "gold" in checks["currency_receipts_match_state"]["detail"]


def test_clamped_write_smaller_than_receipt_fails():
    # The executor clamps currency at zero but the receipt records the
    # requested delta; if they ever diverge the gate must say so.
    receipts = [{"currency_delta": {"gp": -15}}]
    final = _initial()
    final["currency"] = dict(final["currency"], gold=0)
    checks = evaluate_player_state_agreement(_initial(), final, receipts)
    assert not checks["currency_receipts_match_state"]["passed"]


def test_missing_granted_item_fails_and_names_index():
    final = _consistent_final()
    final["inventory"] = {"test-draught": 1}  # antidote receipt never landed
    checks = evaluate_player_state_agreement(_initial(), final, _receipts())
    assert not checks["inventory_receipts_match_state"]["passed"]
    assert "silver-antidote" in checks["inventory_receipts_match_state"]["detail"]


def test_unreceipted_inventory_write_fails():
    final = _consistent_final()
    final["inventory"] = dict(final["inventory"], **{"mystery-rock": 1})
    checks = evaluate_player_state_agreement(_initial(), final, _receipts())
    assert not checks["inventory_receipts_match_state"]["passed"]
    assert "mystery-rock" in checks["inventory_receipts_match_state"]["detail"]


def test_condition_replay_is_order_aware():
    # add then remove nets to absent; a lingering condition is a mismatch.
    final = _consistent_final()
    final["conditions"] = ["poisoned"]
    checks = evaluate_player_state_agreement(_initial(), final, _receipts())
    assert not checks["condition_receipts_match_state"]["passed"]

    # add-only receipts expect the condition present.
    receipts = [{"conditions_added": ["poisoned"]}]
    final_present = _initial() | {"conditions": ["poisoned"]}
    checks = evaluate_player_state_agreement(_initial(), final_present, receipts)
    assert checks["condition_receipts_match_state"]["passed"]


def test_spell_slot_mismatch_fails_and_names_level():
    final = _consistent_final()
    final["spell_slots"] = dict(final["spell_slots"], **{"1": [2, 2]})
    checks = evaluate_player_state_agreement(_initial(), final, _receipts())
    assert not checks["spell_slot_receipts_match_state"]["passed"]
    assert "L1" in checks["spell_slot_receipts_match_state"]["detail"]


def test_family_coverage_names_missing_families():
    receipts = [r for r in _receipts() if "items_removed" not in r]
    final = _consistent_final()
    final["inventory"] = dict(final["inventory"], **{"brass-compass": 1})
    checks = evaluate_player_state_agreement(_initial(), final, receipts)
    coverage = checks["receipts_cover_all_player_state_families"]
    assert not coverage["passed"]
    assert "items_removed" in coverage["detail"]


def test_receipt_collection_skips_duplicates_and_other_families():
    turn_rows = [
        {"turn": 1, "executed": [
            {"type": "update_player", "was_duplicate": False,
             "details": {"applied": {"currency_delta": {"gp": -2}}}},
            {"type": "ref_entity",
             "details": {"applied": {"currency_delta": {"gp": -99}}}},
        ]},
        {"turn": 2, "executed": [
            {"type": "update_player", "was_duplicate": True,
             "details": {"applied": {"currency_delta": {"gp": -2}}}},
            {"type": "update_player", "details": {"applied": {}}},
            {"type": "update_player",
             "details": {"applied": {"spell_slot_used": 1}}},
        ]},
    ]
    assert _update_player_receipts(turn_rows) == [
        {"currency_delta": {"gp": -2}},
        {"spell_slot_used": 1},
    ]
