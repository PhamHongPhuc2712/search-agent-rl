"""Frozen splits: train, validation (carved from train), test (official dev).

Validation is chosen by hashing group ids, so it is the same on every machine
and every run. The chosen ids and the hashes of the raw files are committed in
splits/; loading checks them so a changed file can't silently move a number.

The test split is sealed: it loads only with final=True (the --final flag).
"""

import hashlib
import json
from collections.abc import Iterable, Mapping
from pathlib import Path

from search_agent_rl.data.loaders import DEFAULT_RAW_DIR, load_dataset, raw_path
from search_agent_rl.data.records import QARecord

DEFAULT_SPLITS_DIR = Path(__file__).resolve().parents[3] / "splits"
MANIFEST = "manifest.json"
SALT = "search-agent-rl/validation/v1"

# Number of groups held out: questions for HotpotQA/MuSiQue, conversations for TopiOCQA.
VALIDATION_SIZES = {"hotpotqa": 1000, "musique": 1000, "topiocqa": 100}


class SealedSplitError(RuntimeError):
    """Raised when the test split is requested without final=True."""


class ChecksumMismatchError(RuntimeError):
    """Raised when a split file or raw file no longer matches the manifest."""


def _sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _group_rank(group: str) -> str:
    return _sha256_bytes(f"{SALT}:{group}".encode())


def select_validation_groups(records: Iterable[QARecord], n_groups: int) -> frozenset[str]:
    groups = sorted({r.group for r in records})
    if n_groups > len(groups):
        raise ValueError(f"requested {n_groups} validation groups but only {len(groups)} exist")
    return frozenset(sorted(groups, key=_group_rank)[:n_groups])


def _ids_path(splits_dir: Path, dataset: str) -> Path:
    return splits_dir / f"{dataset}.validation.txt"


def make_splits(
    raw_dir: Path = DEFAULT_RAW_DIR,
    out_dir: Path = DEFAULT_SPLITS_DIR,
    sizes: Mapping[str, int] = VALIDATION_SIZES,
) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, dict] = {}
    for dataset, n_groups in sorted(sizes.items()):
        train = load_dataset(dataset, "train", raw_dir)
        chosen = select_validation_groups(train, n_groups)
        val_ids = sorted(r.id for r in train if r.group in chosen)
        ids_bytes = "".join(f"{i}\n" for i in val_ids).encode()
        _ids_path(out_dir, dataset).write_bytes(ids_bytes)
        manifest[dataset] = {
            "salt": SALT,
            "n_validation_groups": n_groups,
            "n_validation": len(val_ids),
            "n_train": len(train) - len(val_ids),
            "validation_ids_sha256": _sha256_bytes(ids_bytes),
            "raw_train_sha256": _sha256_file(raw_path(dataset, "train", raw_dir)),
            "raw_dev_sha256": _sha256_file(raw_path(dataset, "dev", raw_dir)),
        }
    (out_dir / MANIFEST).write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def _read_manifest(splits_dir: Path) -> dict:
    path = splits_dir / MANIFEST
    if not path.exists():
        raise FileNotFoundError(f"{path} not found; run `make splits` first")
    return json.loads(path.read_text())


def _validation_ids(dataset: str, splits_dir: Path, manifest: dict) -> frozenset[str]:
    if dataset not in manifest:
        raise ValueError(f"no frozen split for {dataset!r}; run `make splits`")
    data = _ids_path(splits_dir, dataset).read_bytes()
    if _sha256_bytes(data) != manifest[dataset]["validation_ids_sha256"]:
        raise ChecksumMismatchError(f"{dataset} validation ids changed since the split was frozen")
    return frozenset(data.decode().split())


def load_split(
    dataset: str,
    split: str,
    raw_dir: Path = DEFAULT_RAW_DIR,
    splits_dir: Path = DEFAULT_SPLITS_DIR,
    final: bool = False,
) -> list[QARecord]:
    if split == "test":
        if not final:
            raise SealedSplitError(
                f"{dataset} test split is sealed; pass final=True (--final) for a final number"
            )
        return load_dataset(dataset, "dev", raw_dir)
    if split not in ("train", "validation"):
        raise ValueError(f"unknown split {split!r}; expected 'train', 'validation' or 'test'")
    val_ids = _validation_ids(dataset, splits_dir, _read_manifest(splits_dir))
    records = load_dataset(dataset, "train", raw_dir)
    in_validation = split == "validation"
    return [r for r in records if (r.id in val_ids) == in_validation]


def verify(raw_dir: Path = DEFAULT_RAW_DIR, splits_dir: Path = DEFAULT_SPLITS_DIR) -> None:
    """Check every raw file and ids file against the manifest."""
    manifest = _read_manifest(splits_dir)
    for dataset, entry in sorted(manifest.items()):
        _validation_ids(dataset, splits_dir, manifest)
        for split in ("train", "dev"):
            if _sha256_file(raw_path(dataset, split, raw_dir)) != entry[f"raw_{split}_sha256"]:
                raise ChecksumMismatchError(
                    f"{dataset} raw {split} file changed since splits were made"
                )
