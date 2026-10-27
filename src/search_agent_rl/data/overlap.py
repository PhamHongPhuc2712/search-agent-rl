"""Train/test question overlap, reported rather than silently ignored.

Two levels: exact match after answer-style normalization, and the same set of
words in any order (catches reworded copies). Fuzzier matching can come later.
"""

from collections.abc import Sequence

from search_agent_rl.eval.metrics import normalize_answer

from .records import QARecord


def _bag(text: str) -> frozenset[str]:
    return frozenset(text.split())


def find_overlap(train: Sequence[QARecord], test: Sequence[QARecord]) -> dict:
    train_exact = {normalize_answer(r.question) for r in train}
    train_bags = {_bag(q) for q in train_exact}
    exact, bag_of_words = [], []
    for record in test:
        norm = normalize_answer(record.question)
        if norm in train_exact:
            exact.append(record.id)
        elif _bag(norm) in train_bags:
            bag_of_words.append(record.id)
    return {"n_test": len(test), "exact": exact, "bag_of_words": bag_of_words}
