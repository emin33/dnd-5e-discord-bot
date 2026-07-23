"""Unit tests for the offline narrative-quality grader (deterministic parts)."""

from test_narrative_grader import (
    DIMENSIONS,
    GATE_REPEAT_RATIO_MAX,
    GradeReport,
    TurnRow,
    WindowGrade,
    _excerpt,
    opening_bigrams,
    repeat_ratios,
)


class TestExcerpt:
    def test_short_text_untouched(self):
        assert _excerpt("A short line.", 100) == "A short line."

    def test_long_text_cuts_at_sentence_and_announces(self):
        text = ("First sentence here. Second sentence follows. " * 10).strip()
        out = _excerpt(text, 120)
        assert "TRUNCATED BY GRADER" in out
        # The kept portion ends at a sentence boundary, not mid-word.
        kept = out.split(" [EXCERPT")[0]
        assert kept.endswith(".")


def _row(turn: int, narration: str) -> TurnRow:
    return TurnRow(turn=turn, action="I act.", narration=narration)


def _grade(first: int, last: int, score: int, contradictions=None) -> WindowGrade:
    return WindowGrade(
        first_turn=first,
        last_turn=last,
        scores={d: score for d in DIMENSIONS},
        contradictions=list(contradictions or []),
        flags=[],
        window_summary="s",
    )


class TestRepeatRatios:
    def test_fresh_prose_scores_low(self):
        rows = [
            _row(1, "The tavern hums with low conversation and pipe smoke "
                    "curling beneath blackened rafters while a fiddler tunes."),
            _row(2, "Rain needles the harbor stones as gulls wheel over the "
                    "mast forest and a customs bell tolls twice from the pier."),
        ]
        ratios = repeat_ratios(rows)
        assert ratios[2] == 0.0

    def test_recycled_paragraph_scores_high(self):
        text = ("The rain falls upward past the chained leviathan while "
                "memory lanterns gutter along the sagging rooftops of Veyr "
                "and the ninth bell tolls its warning across the district.")
        rows = [_row(1, text), _row(2, text + " You press onward anyway.")]
        ratios = repeat_ratios(rows)
        assert ratios[2] > GATE_REPEAT_RATIO_MAX

    def test_first_turn_has_no_ratio(self):
        rows = [_row(1, "Words enough to form several eight grams of text "
                        "for the sliding window calculation here.")]
        assert repeat_ratios(rows) == {}


class TestOpeningBigrams:
    def test_counts_recurring_openers(self):
        rows = [
            _row(1, "You step into the alley."),
            _row(2, "You step over the gutter."),
            _row(3, "Rain hammers the rooftops."),
        ]
        counts = opening_bigrams(rows)
        assert counts["you step"] == 2
        assert counts["rain hammers"] == 1


class TestGates:
    def _report(self, windows, failed=0, ratios=None) -> GradeReport:
        report = GradeReport(turn_log="x", judge_model="m")
        report.windows = windows
        report.failed_windows = failed
        report.repeat_ratios = ratios or {2: 0.1}
        return report

    def test_healthy_report_passes(self):
        report = self._report([_grade(1, 6, 4), _grade(7, 12, 5)])
        assert report.overall_average() == 4.5
        assert report.passed()

    def test_low_dimension_mean_fails(self):
        low = _grade(1, 6, 4)
        low.scores["prose_freshness"] = 2
        second = _grade(7, 12, 4)
        second.scores["prose_freshness"] = 3
        report = self._report([low, second])
        names = {n: ok for n, ok, _ in report.gates()}
        assert names["no_dimension_below_minimum"] is False
        assert not report.passed()

    def test_severe_contradiction_fails(self):
        window = _grade(1, 6, 5, contradictions=[
            {"turn": 3, "severity": "severe", "detail": "dead NPC speaks"},
        ])
        report = self._report([window])
        names = {n: ok for n, ok, _ in report.gates()}
        assert names["zero_severe_contradictions"] is False

    def test_minor_contradictions_do_not_fail_gate(self):
        window = _grade(1, 6, 5, contradictions=[
            {"turn": 3, "severity": "minor", "detail": "eye color drifted"},
        ])
        report = self._report([window])
        names = {n: ok for n, ok, _ in report.gates()}
        assert names["zero_severe_contradictions"] is True

    def test_failed_judge_window_fails_closed(self):
        report = self._report([_grade(1, 6, 5)], failed=1)
        names = {n: ok for n, ok, _ in report.gates()}
        assert names["judge_coverage_complete"] is False
        assert not report.passed()

    def test_no_windows_fails_closed(self):
        report = self._report([], failed=0)
        assert not report.passed()

    def test_repetition_gate_uses_worst_turn(self):
        report = self._report(
            [_grade(1, 6, 5)], ratios={2: 0.05, 9: 0.6},
        )
        names = {n: ok for n, ok, _ in report.gates()}
        assert names["prose_repetition_bounded"] is False
