import json
from pathlib import Path

import pytest

from search_agent_rl.data import splits
from search_agent_rl.data.records import QARecord
from search_agent_rl.data.splits import (
    ChecksumMismatchError,
    SealedSplitError,
    load_split,
    make_splits,
    select_validation_groups,
)


def rec(i: int, group: str | None = None) -> QARecord:
    rid = f"hotpotqa:train_{i}"
    return QARecord(
        id=rid, dataset="hotpotqa", question=f"q{i}", answers=("a",), group=group or rid
    )


def write_raw(raw_dir: Path, n_train: int = 50, n_dev: int = 5) -> None:
    d = raw_dir / "hotpotqa"
    d.mkdir(parents=True)
    for name, n in (("train", n_train), ("dev", n_dev)):
        rows = [
            {
                "id": f"{name}_{i}",
                "question": f"{name} q{i}",
                "golden_answers": ["a"],
                "metadata": {},
            }
            for i in range(n)
        ]
        (d / f"{name}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))


@pytest.fixture
def built(tmp_path):
    raw, out = tmp_path / "raw", tmp_path / "splits"
    write_raw(raw)
    make_splits(raw_dir=raw, out_dir=out, sizes={"hotpotqa": 10})
    return raw, out


class TestSelectValidationGroups:
    def test_deterministic_and_sized(self):
        records = [rec(i) for i in range(100)]
        a = select_validation_groups(records, 10)
        assert a == select_validation_groups(list(reversed(records)), 10)
        assert len(a) == 10

    def test_selects_whole_groups(self):
        records = [rec(i, group=f"conv{i // 5}") for i in range(50)]  # 10 groups of 5
        chosen = select_validation_groups(records, 3)
        assert len(chosen) == 3
        assert sum(r.group in chosen for r in records) == 15

    def test_too_many_requested(self):
        with pytest.raises(ValueError, match="only"):
            select_validation_groups([rec(0)], 2)


class TestMakeAndLoad:
    def test_validation_and_train_partition_the_train_file(self, built):
        raw, out = built
        val = load_split("hotpotqa", "validation", raw_dir=raw, splits_dir=out)
        train = load_split("hotpotqa", "train", raw_dir=raw, splits_dir=out)
        assert len(val) == 10 and len(train) == 40
        assert not {r.id for r in val} & {r.id for r in train}

    def test_manifest_records_counts_and_hashes(self, built):
        _, out = built
        manifest = json.loads((out / "manifest.json").read_text())["hotpotqa"]
        assert manifest["n_validation"] == 10
        assert manifest["n_train"] == 40
        assert len(manifest["validation_ids_sha256"]) == 64
        assert len(manifest["raw_train_sha256"]) == 64

    def test_rebuilding_is_identical(self, built, tmp_path):
        raw, out = built
        out2 = tmp_path / "splits2"
        make_splits(raw_dir=raw, out_dir=out2, sizes={"hotpotqa": 10})
        assert (out / "manifest.json").read_text() == (out2 / "manifest.json").read_text()

    def test_test_split_needs_final_flag(self, built):
        raw, out = built
        with pytest.raises(SealedSplitError):
            load_split("hotpotqa", "test", raw_dir=raw, splits_dir=out)
        assert len(load_split("hotpotqa", "test", raw_dir=raw, splits_dir=out, final=True)) == 5

    def test_edited_ids_file_detected(self, built):
        raw, out = built
        ids_file = out / "hotpotqa.validation.txt"
        ids_file.write_text(ids_file.read_text() + "hotpotqa:train_999\n")
        with pytest.raises(ChecksumMismatchError):
            load_split("hotpotqa", "validation", raw_dir=raw, splits_dir=out)

    def test_changed_raw_data_detected_by_verify(self, built):
        raw, out = built
        train_file = raw / "hotpotqa" / "train.jsonl"
        train_file.write_text(train_file.read_text().replace('train q1"', 'train q1 edited"'))
        with pytest.raises(ChecksumMismatchError):
            splits.verify(raw_dir=raw, splits_dir=out)

    def test_unknown_split(self, built):
        raw, out = built
        with pytest.raises(ValueError, match="split"):
            load_split("hotpotqa", "dev", raw_dir=raw, splits_dir=out)
