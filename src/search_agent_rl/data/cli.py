"""Data commands: build and check frozen splits, report train/test overlap.

uv run python -m search_agent_rl.data.cli make-splits
uv run python -m search_agent_rl.data.cli verify
uv run python -m search_agent_rl.data.cli overlap
"""

import argparse
import json
from collections.abc import Sequence

from search_agent_rl.data import splits
from search_agent_rl.data.loaders import DATASETS, load_dataset
from search_agent_rl.data.overlap import find_overlap
from search_agent_rl.eval.harness import DEFAULT_RESULTS_DIR


def _make_splits() -> None:
    manifest = splits.make_splits()
    for dataset, entry in manifest.items():
        print(f"{dataset}: validation={entry['n_validation']} train={entry['n_train']}")


def _verify() -> None:
    splits.verify()
    print("splits and raw files match the manifest")


def _overlap() -> None:
    # Reads test questions only to compare text; nothing is scored, so the seal holds.
    all_train = [r for d in DATASETS for r in load_dataset(d, "train")]
    report = {}
    for dataset in DATASETS:
        test = load_dataset(dataset, "dev")
        same = find_overlap(load_dataset(dataset, "train"), test)
        cross = find_overlap(all_train, test)
        report[dataset] = {"same_dataset_train": same, "any_dataset_train": cross}
        print(
            f"{dataset}: test={same['n_test']} exact={len(same['exact'])} "
            f"reordered={len(same['bag_of_words'])} | any train: exact={len(cross['exact'])} "
            f"reordered={len(cross['bag_of_words'])}"
        )
    path = DEFAULT_RESULTS_DIR / "overlap.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2) + "\n")
    print(f"-> {path}")


COMMANDS = {"make-splits": _make_splits, "verify": _verify, "overlap": _overlap}


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("command", choices=sorted(COMMANDS))
    COMMANDS[parser.parse_args(argv).command]()


if __name__ == "__main__":
    main()
