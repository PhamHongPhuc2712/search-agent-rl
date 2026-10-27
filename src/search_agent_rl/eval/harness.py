"""Run any system on any split and score it with EM/F1.

uv run python -m search_agent_rl.eval.harness --system gold --dataset musique
uv run python -m search_agent_rl.eval.harness --system ... --split test --final
"""

import argparse
import json
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path
from typing import Protocol

from search_agent_rl.data.loaders import DATASETS
from search_agent_rl.data.records import QARecord
from search_agent_rl.data.splits import load_split
from search_agent_rl.eval.metrics import Scheme, exact_match, f1, max_over_golds
from search_agent_rl.eval.stats import bootstrap_ci

DEFAULT_RESULTS_DIR = Path(__file__).resolve().parents[3] / "docs" / "results"
SCHEMES = {"hotpotqa": Scheme.HOTPOT, "musique": Scheme.SQUAD, "topiocqa": Scheme.SQUAD}


class System(Protocol):
    name: str

    def answer(self, record: QARecord) -> str: ...


class GoldSystem:
    """Returns the first gold answer. A sanity check: must score EM = 1."""

    name = "gold"

    def answer(self, record: QARecord) -> str:
        return record.answers[0]


class EmptySystem:
    """Returns an empty answer. A sanity check: scores EM ~ 0.

    Not exactly 0 under the SQuAD scheme: a few real gold answers normalize to
    the empty string ("$", "The The", "!!!"), and the official scorer counts an
    empty prediction as matching them. Kept for comparability with published
    numbers; the RL reward penalizes empty answers separately.
    """

    name = "empty"

    def answer(self, record: QARecord) -> str:
        return ""


SYSTEMS: dict[str, type] = {"gold": GoldSystem, "empty": EmptySystem}


def _mean(values: Sequence[float]) -> float:
    return sum(values) / len(values)


def _score(prediction: str, record: QARecord, scheme: Scheme) -> tuple[float, float]:
    em = max_over_golds(exact_match, prediction, record.answers)
    f = max_over_golds(lambda p, g: f1(p, g, scheme), prediction, record.answers)
    return em, f


def _by_hops(rows: list[dict]) -> dict[str, dict]:
    groups: dict[int, list[dict]] = defaultdict(list)
    for row in rows:
        if row["hops"] is not None:
            groups[row["hops"]].append(row)
    return {
        str(hops): {
            "n": len(g),
            "em": _mean([r["em"] for r in g]),
            "f1": _mean([r["f1"] for r in g]),
        }
        for hops, g in sorted(groups.items())
    }


def evaluate(system: System, records: Sequence[QARecord], *, dataset: str, split: str) -> dict:
    if not records:
        raise ValueError(f"no records to evaluate for {dataset}/{split}")
    scheme = SCHEMES[dataset]
    answerable = [r for r in records if r.answers]
    rows = []
    for record in answerable:
        prediction = system.answer(record)
        em, f = _score(prediction, record, scheme)
        rows.append(
            {"id": record.id, "prediction": prediction, "em": em, "f1": f, "hops": record.hops}
        )
    if not rows:
        raise ValueError(f"no records with gold answers for {dataset}/{split}")
    ems, f1s = [r["em"] for r in rows], [r["f1"] for r in rows]
    summary = {
        "n": len(rows),
        "n_skipped_no_answer": len(records) - len(answerable),
        "em": _mean(ems),
        "f1": _mean(f1s),
        "em_ci95": bootstrap_ci(ems),
        "f1_ci95": bootstrap_ci(f1s),
        "by_hops": _by_hops(rows),
    }
    return {
        "system": system.name,
        "dataset": dataset,
        "split": split,
        "scheme": scheme.value,
        "summary": summary,
        "per_question": rows,
    }


def save_result(result: dict, results_dir: Path = DEFAULT_RESULTS_DIR) -> Path:
    path = results_dir / result["split"] / result["dataset"] / f"{result['system']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2) + "\n")
    return path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--system", required=True, choices=sorted(SYSTEMS))
    parser.add_argument("--dataset", required=True, choices=DATASETS)
    parser.add_argument("--split", default="validation", choices=("validation", "test"))
    parser.add_argument("--final", action="store_true", help="required to touch the test split")
    parser.add_argument("--limit", type=int, help="evaluate only the first N records")
    parser.add_argument("--out", type=Path, default=DEFAULT_RESULTS_DIR)
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    records = load_split(args.dataset, args.split, final=args.final)
    if args.limit is not None:
        records = records[: args.limit]
    result = evaluate(SYSTEMS[args.system](), records, dataset=args.dataset, split=args.split)
    path = save_result(result, args.out)
    s = result["summary"]
    print(
        f"{args.system} on {args.dataset}/{args.split}: n={s['n']} "
        f"EM={s['em']:.3f} {s['em_ci95']} F1={s['f1']:.3f} -> {path}"
    )


if __name__ == "__main__":
    main()
