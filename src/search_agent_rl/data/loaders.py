"""Read the raw downloads in data/raw/ into QARecords.

HotpotQA and MuSiQue come from FlashRAG's preprocessed release, TopiOCQA from
McGill-NLP's Hugging Face release. Each raw split is one JSONL file.
"""

import json
from collections.abc import Callable, Iterator
from pathlib import Path

from search_agent_rl.data.records import QARecord

DEFAULT_RAW_DIR = Path(__file__).resolve().parents[3] / "data" / "raw"
UNANSWERABLE = "UNANSWERABLE"


def parse_hotpotqa(row: dict) -> QARecord:
    rec_id = f"hotpotqa:{row['id']}"
    return QARecord(
        id=rec_id,
        dataset="hotpotqa",
        question=row["question"],
        answers=tuple(row["golden_answers"]),
        group=rec_id,
        hops=2,
    )


def parse_musique(row: dict) -> QARecord:
    meta = row["metadata"]
    if not meta.get("answerable", True):
        raise ValueError(f"musique {row['id']}: unanswerable questions are not supported")
    rec_id = f"musique:{row['id']}"
    return QARecord(
        id=rec_id,
        dataset="musique",
        question=row["question"],
        answers=tuple(row["golden_answers"]),
        group=rec_id,
        hops=len(meta["question_decomposition"]),
    )


def parse_topiocqa(row: dict) -> QARecord:
    context = row["Context"]
    if len(context) % 2:
        turn = f"{row['Conversation_no']}_{row['Turn_no']}"
        raise ValueError(f"topiocqa {turn}: context must alternate question/answer")
    history = tuple(zip(context[0::2], context[1::2], strict=True))
    candidates = [row["Answer"], *(a["Answer"] for a in row["Additional_answers"])]
    answers = tuple(dict.fromkeys(a for a in candidates if a != UNANSWERABLE))
    conv = f"topiocqa:{row['Conversation_no']}"
    return QARecord(
        id=f"{conv}_{row['Turn_no']}",
        dataset="topiocqa",
        question=row["Question"],
        answers=answers,
        group=conv,
        history=history,
    )


_PARSERS: dict[str, Callable[[dict], QARecord]] = {
    "hotpotqa": parse_hotpotqa,
    "musique": parse_musique,
    "topiocqa": parse_topiocqa,
}
_FILES = {
    ("hotpotqa", "train"): "hotpotqa/train.jsonl",
    ("hotpotqa", "dev"): "hotpotqa/dev.jsonl",
    ("musique", "train"): "musique/train.jsonl",
    ("musique", "dev"): "musique/dev.jsonl",
    ("topiocqa", "train"): "topiocqa/train.jsonl",
    ("topiocqa", "dev"): "topiocqa/valid.jsonl",
}

DATASETS = tuple(_PARSERS)


def raw_path(dataset: str, split: str, raw_dir: Path = DEFAULT_RAW_DIR) -> Path:
    if dataset not in _PARSERS:
        raise ValueError(f"unknown dataset {dataset!r}; expected one of {DATASETS}")
    if (dataset, split) not in _FILES:
        raise ValueError(f"unknown split {split!r} for {dataset}; expected 'train' or 'dev'")
    return raw_dir / _FILES[(dataset, split)]


def _read_jsonl(path: Path) -> Iterator[dict]:
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def load_dataset(dataset: str, split: str, raw_dir: Path = DEFAULT_RAW_DIR) -> list[QARecord]:
    path = raw_path(dataset, split, raw_dir)
    parse = _PARSERS[dataset]
    records = [parse(row) for row in _read_jsonl(path)]
    seen: set[str] = set()
    for rec in records:
        if rec.id in seen:
            raise ValueError(f"{path}: duplicate record id {rec.id}")
        seen.add(rec.id)
    return records
