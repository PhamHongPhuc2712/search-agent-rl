"""Answer-level exact match and F1, matching the official scorers.

Two schemes differ only in edge cases:
- HOTPOT follows hotpot_evaluate_v1.py: a yes/no/noanswer answer on either
  side must match exactly, and empty answers score 0.
- SQUAD follows the SQuAD v2 script used by MuSiQue: no yes/no rule, and two
  empty answers score 1.
"""

import re
import string
from collections import Counter
from collections.abc import Callable, Sequence
from enum import Enum

_PUNCTUATION = frozenset(string.punctuation)
_ARTICLES = re.compile(r"\b(a|an|the)\b")
_HOTPOT_SPECIAL = frozenset({"yes", "no", "noanswer"})


class Scheme(Enum):
    HOTPOT = "hotpot"
    SQUAD = "squad"


def normalize_answer(text: str) -> str:
    text = text.lower()
    text = "".join(ch for ch in text if ch not in _PUNCTUATION)
    text = _ARTICLES.sub(" ", text)
    return " ".join(text.split())


def exact_match(prediction: str, gold: str) -> float:
    return float(normalize_answer(prediction) == normalize_answer(gold))


def f1(prediction: str, gold: str, scheme: Scheme) -> float:
    pred_norm = normalize_answer(prediction)
    gold_norm = normalize_answer(gold)

    special = pred_norm in _HOTPOT_SPECIAL or gold_norm in _HOTPOT_SPECIAL
    if scheme is Scheme.HOTPOT and special and pred_norm != gold_norm:
        return 0.0

    pred_tokens = pred_norm.split()
    gold_tokens = gold_norm.split()

    if scheme is Scheme.SQUAD and (not pred_tokens or not gold_tokens):
        return float(pred_tokens == gold_tokens)

    num_same = sum((Counter(pred_tokens) & Counter(gold_tokens)).values())
    if num_same == 0:
        return 0.0
    precision = num_same / len(pred_tokens)
    recall = num_same / len(gold_tokens)
    return 2 * precision * recall / (precision + recall)


def max_over_golds(
    metric: Callable[[str, str], float], prediction: str, golds: Sequence[str]
) -> float:
    """Score against every gold alias and keep the best, as the official scripts do."""
    if not golds:
        raise ValueError("at least one gold answer is required")
    return max(metric(prediction, gold) for gold in golds)
