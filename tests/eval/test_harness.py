import json

import pytest

from search_agent_rl.data.records import QARecord
from search_agent_rl.eval.harness import (
    EmptySystem,
    GoldSystem,
    build_parser,
    evaluate,
    save_result,
)
from search_agent_rl.eval.stats import bootstrap_ci


def rec(
    i: int, answers: tuple[str, ...], hops: int | None = 2, dataset: str = "musique"
) -> QARecord:
    rid = f"{dataset}:{i}"
    return QARecord(
        id=rid, dataset=dataset, question=f"q{i}", answers=answers, group=rid, hops=hops
    )


class FixedSystem:
    name = "fixed"

    def __init__(self, answers: dict[str, str]):
        self._answers = answers

    def answer(self, record: QARecord) -> str:
        return self._answers[record.id]


class TestBootstrap:
    def test_constant_values_have_zero_width(self):
        assert bootstrap_ci([1.0] * 20) == (1.0, 1.0)

    def test_deterministic_for_seed(self):
        values = [0.0, 1.0] * 50
        assert bootstrap_ci(values, seed=3) == bootstrap_ci(values, seed=3)

    def test_interval_contains_mean(self):
        values = [1.0] * 30 + [0.0] * 70
        lo, hi = bootstrap_ci(values)
        assert lo < 0.3 < hi

    def test_empty_rejected(self):
        with pytest.raises(ValueError):
            bootstrap_ci([])


class TestEvaluate:
    def test_gold_system_scores_perfectly(self):
        records = [rec(i, ("Paris", "City of Paris")) for i in range(5)]
        result = evaluate(GoldSystem(), records, dataset="musique", split="validation")
        assert result["summary"]["em"] == 1.0
        assert result["summary"]["f1"] == 1.0

    def test_empty_system_scores_zero(self):
        result = evaluate(
            EmptySystem(), [rec(0, ("Paris",))], dataset="musique", split="validation"
        )
        assert result["summary"]["em"] == 0.0

    def test_records_without_answers_are_skipped_and_counted(self):
        records = [rec(0, ("Paris",)), rec(1, ())]
        result = evaluate(GoldSystem(), records, dataset="musique", split="validation")
        assert result["summary"]["n"] == 1
        assert result["summary"]["n_skipped_no_answer"] == 1

    def test_breakdown_by_hops(self):
        records = [rec(0, ("a",), hops=2), rec(1, ("b",), hops=3), rec(2, ("c",), hops=3)]
        system = FixedSystem({"musique:0": "a", "musique:1": "b", "musique:2": "wrong"})
        by_hops = evaluate(system, records, dataset="musique", split="validation")["summary"][
            "by_hops"
        ]
        assert by_hops["2"] == {"n": 1, "em": 1.0, "f1": 1.0}
        assert by_hops["3"]["n"] == 2 and by_hops["3"]["em"] == 0.5

    def test_uses_hotpot_scheme_for_hotpotqa(self):
        record = rec(0, ("yes",), dataset="hotpotqa")
        result = evaluate(
            FixedSystem({"hotpotqa:0": "yes it is"}),
            [record],
            dataset="hotpotqa",
            split="validation",
        )
        assert result["summary"]["f1"] == 0.0  # hotpot rule: yes/no must match exactly

    def test_per_question_rows_kept(self):
        result = evaluate(GoldSystem(), [rec(0, ("a",))], dataset="musique", split="validation")
        assert result["per_question"] == [
            {"id": "musique:0", "prediction": "a", "em": 1.0, "f1": 1.0, "hops": 2}
        ]

    def test_empty_records_rejected(self):
        with pytest.raises(ValueError, match="no records"):
            evaluate(GoldSystem(), [], dataset="musique", split="validation")


class TestSave:
    def test_writes_under_split_dataset_system(self, tmp_path):
        result = evaluate(GoldSystem(), [rec(0, ("a",))], dataset="musique", split="validation")
        path = save_result(result, tmp_path)
        assert path == tmp_path / "validation" / "musique" / "gold.json"
        assert json.loads(path.read_text())["summary"]["em"] == 1.0


class TestCli:
    def test_final_flag_defaults_off(self):
        args = build_parser().parse_args(["--system", "gold", "--dataset", "musique"])
        assert args.final is False and args.split == "validation"
