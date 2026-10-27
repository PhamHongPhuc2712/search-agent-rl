from dataclasses import dataclass


@dataclass(frozen=True)
class QARecord:
    """One question with its gold answers, in a format shared by all datasets.

    `group` is the unit for splitting: the record itself for single-turn data,
    the whole conversation for TopiOCQA, so turns of one conversation never
    land on both sides of a split.
    """

    id: str
    dataset: str
    question: str
    answers: tuple[str, ...]
    group: str
    hops: int | None = None
    history: tuple[tuple[str, str], ...] = ()
