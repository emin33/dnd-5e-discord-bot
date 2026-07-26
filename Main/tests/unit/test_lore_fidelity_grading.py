"""Pin the lore_recall grader — the gate that reads ground truth.

Every other harness gate grades self-consistency. This one grades against an
authored book, which is the only way to ask two questions that otherwise have
no answer: did the right canon reach the prose, and did canon the party never
earned leak anyway.

Leak detection is exact-substring on CANARY tokens — invented proper nouns
that appear only inside withheld claims. The grader must be strict about
those (a paraphrase cannot invent "Vesk Harrow") and must not be fooled by
prose that merely discusses the same subject.
"""

from __future__ import annotations

from test_tool_reliability import _CANARIES, _lore_book, evaluate_lore_fidelity

from dnd_bot.game.knowledge.sourcebook_compiler import compile_sourcebook


def _compiled():
    book = _lore_book()
    return book, compile_sourcebook(book, "camp")


def _world(names):
    return {"npcs": {f"id-{i}": {"name": n} for i, n in enumerate(names)}}


_GOOD_PROSE = {
    1: "Mara leans back. 'Closed since Old Bram died — he was the ferryman.'",
    2: "'Toran Vex kept that lock for years,' she says. 'A lockwright's work.'",
    3: "'If I knew who sold it,' she says, 'I'd have said so by now.'",
    4: "You walk to the arch.",
    5: "No answer comes. Old Bram is dead, and the wind says nothing.",
    6: "The ground is packed hard. Nothing gives.",
    7: "The Copper Finch is as you left it.",
}

_CLEAN_WORLD = _world(["Mara Venn", "Toran Vex"])


def test_a_clean_run_passes_every_lore_gate():
    book, compiled = _compiled()

    checks = evaluate_lore_fidelity(book, compiled, _GOOD_PROSE, _CLEAN_WORLD)

    failures = {k: v["detail"] for k, v in checks.items() if not v["passed"]}
    assert not failures, failures


def test_a_leaked_canary_fails_and_names_the_turn():
    """The headline gate: unearned canon in the prose, however phrased."""
    book, compiled = _compiled()
    prose = dict(_GOOD_PROSE)
    prose[3] = (
        "Mara's jaw tightens. 'It was Vesk Harrow,' she admits. "
        "'He sold the passage.'"
    )

    checks = evaluate_lore_fidelity(book, compiled, prose, _CLEAN_WORLD)

    leak = checks["no_withheld_canon_leaked"]
    assert not leak["passed"]
    assert "Vesk Harrow" in leak["detail"]
    assert "3" in leak["detail"]          # the turn is named


def test_the_second_secret_is_watched_too():
    book, compiled = _compiled()
    prose = dict(_GOOD_PROSE)
    prose[6] = "Your fingers find a hollow — the Marrow Cist, and something in it."

    checks = evaluate_lore_fidelity(book, compiled, prose, _CLEAN_WORLD)

    assert not checks["no_withheld_canon_leaked"]["passed"]
    assert "Marrow Cist" in checks["no_withheld_canon_leaked"]["detail"]


def test_discussing_the_subject_without_the_secret_is_not_a_leak():
    """The party may ask about the sale all day; only the ANSWER is canon."""
    book, compiled = _compiled()
    prose = dict(_GOOD_PROSE)
    prose[3] = (
        "'Someone sold passage through that gate,' Mara says, 'and someone "
        "filed the lock to hide it. Names are what I'm short of.'"
    )

    checks = evaluate_lore_fidelity(book, compiled, prose, _CLEAN_WORLD)

    assert checks["no_withheld_canon_leaked"]["passed"]


def test_forgotten_public_canon_fails_recall():
    book, compiled = _compiled()
    prose = dict(_GOOD_PROSE)
    prose[2] = "'Someone kept it up,' she shrugs. 'Long time ago now.'"

    checks = evaluate_lore_fidelity(book, compiled, prose, _CLEAN_WORLD)

    recall = checks["book_public_canon_recalled"]
    assert not recall["passed"]
    assert "toran_is_the_lockwright" in recall["detail"]


def test_residents_missing_after_return_fails_hydration():
    book, compiled = _compiled()

    checks = evaluate_lore_fidelity(
        book, compiled, _GOOD_PROSE, _world(["Mara Venn"]),
    )

    assert not checks["authored_residents_on_stage_after_return"]["passed"]


def test_an_authored_dead_npc_on_stage_fails():
    book, compiled = _compiled()

    checks = evaluate_lore_fidelity(
        book, compiled, _GOOD_PROSE,
        _world(["Mara Venn", "Toran Vex", "Old Bram"]),
    )

    assert not checks["authored_dead_never_on_stage"]["passed"]


def test_canaries_appear_nowhere_the_party_can_reach():
    """Guards the instrument itself.

    The gate is only meaningful if a canary cannot reach the prose by any
    honest route — so it must not appear in the seeded world, the scenario
    script, or the public claims.
    """
    from test_tool_reliability import LORE_ACTIONS

    book, compiled = _compiled()
    reachable = " ".join([
        *compiled.established_facts,
        compiled.current_location,
        compiled.location_description,
        compiled.opening_situation,
        *LORE_ACTIONS,
        *(n.name + " " + n.appearance + " " + n.summary for n in book.npcs),
        *(l.description for l in book.locations),
    ]).casefold()

    for token in _CANARIES.values():
        assert token.casefold() not in reachable, token
    # And each canary really is in a withheld claim, or the gate tests nothing.
    withheld_text = " ".join(c.text for c in compiled.withheld).casefold()
    for token in _CANARIES.values():
        assert token.casefold() in withheld_text, token
