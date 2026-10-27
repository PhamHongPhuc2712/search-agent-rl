"""EM/F1 must match the official scorers.

HotpotQA: hotpot_evaluate_v1.py (yes/no/noanswer special case).
MuSiQue and TopiOCQA: SQuAD v2 style (empty answers, no yes/no rule).
Expected values below were worked out from those scripts by hand.
"""

import pytest

from search_agent_rl.eval.metrics import (
    Scheme,
    exact_match,
    f1,
    max_over_golds,
    normalize_answer,
)


class TestNormalize:
    def test_lowercases_and_strips_punctuation_articles_whitespace(self):
        assert normalize_answer("  The  Eiffel-Tower! ") == "eiffeltower"

    def test_articles_removed_only_as_whole_words(self):
        assert normalize_answer("An anthem at the theatre") == "anthem at theatre"

    def test_empty(self):
        assert normalize_answer("") == ""


class TestExactMatch:
    def test_equal_after_normalization(self):
        assert exact_match("the Beatles.", "Beatles") == 1.0

    def test_different(self):
        assert exact_match("Rolling Stones", "Beatles") == 0.0


class TestF1Squad:
    def test_partial_overlap(self):
        # pred tokens: christopher nolan; gold: nolan -> p=1/2, r=1 -> f1=2/3
        assert f1("Christopher Nolan", "Nolan", Scheme.SQUAD) == pytest.approx(2 / 3)

    def test_repeated_tokens_count_once_per_match(self):
        # pred: paris paris; gold: paris -> common=1, p=1/2, r=1
        assert f1("Paris Paris", "Paris", Scheme.SQUAD) == pytest.approx(2 / 3)

    def test_no_overlap(self):
        assert f1("London", "Paris", Scheme.SQUAD) == 0.0

    def test_both_empty_scores_one(self):
        assert f1("", "", Scheme.SQUAD) == 1.0

    def test_one_empty_scores_zero(self):
        assert f1("", "Paris", Scheme.SQUAD) == 0.0
        assert f1("Paris", "", Scheme.SQUAD) == 0.0

    def test_yes_no_partial_credit_allowed(self):
        assert f1("yes it is", "yes", Scheme.SQUAD) == pytest.approx(0.5)


class TestF1Hotpot:
    def test_partial_overlap_same_as_squad(self):
        assert f1("Christopher Nolan", "Nolan", Scheme.HOTPOT) == pytest.approx(2 / 3)

    def test_yes_no_gold_requires_exact(self):
        assert f1("yes it is", "yes", Scheme.HOTPOT) == 0.0

    def test_yes_no_prediction_against_other_gold_is_zero(self):
        assert f1("no", "no way out", Scheme.HOTPOT) == 0.0

    def test_yes_no_exact_is_one(self):
        assert f1("Yes.", "yes", Scheme.HOTPOT) == 1.0

    def test_both_empty_scores_zero(self):
        # hotpot_evaluate_v1 has no empty special case: no common tokens -> 0
        assert f1("", "", Scheme.HOTPOT) == 0.0


class TestMaxOverGolds:
    def test_takes_best_alias(self):
        golds = ["Barack Obama", "Obama", "President Obama"]
        assert max_over_golds(exact_match, "obama", golds) == 1.0

    def test_f1_best_alias(self):
        score = max_over_golds(
            lambda p, g: f1(p, g, Scheme.SQUAD), "Barack H. Obama", ["Obama", "Barack Obama"]
        )
        # vs "barack obama": pred tokens barack h obama (3), common 2 -> p=2/3, r=1 -> 0.8
        assert score == pytest.approx(0.8)

    def test_no_golds_raises(self):
        with pytest.raises(ValueError):
            max_over_golds(exact_match, "x", [])
