import json
from pathlib import Path

import pytest

from search_agent_rl.data.loaders import load_dataset, parse_hotpotqa, parse_musique, parse_topiocqa


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in rows))


HOTPOT_ROW = {
    "id": "dev_0",
    "question": "Were Scott Derrickson and Ed Wood of the same nationality?",
    "golden_answers": ["yes"],
    "metadata": {"type": "comparison", "level": "hard"},
}

MUSIQUE_ROW = {
    "id": "dev_0",
    "question": "Who is the spouse of the Green performer?",
    "golden_answers": ["Miquette Giraudy", "M. Giraudy"],
    "metadata": {
        "answerable": True,
        "question_decomposition": [{"answer": "Steve Hillage"}, {"answer": "Miquette Giraudy"}],
    },
}


def topiocqa_row(turn: int, context: list[str], answer: str, extra: list[str]) -> dict:
    return {
        "Context": context,
        "Conversation_no": 7,
        "Turn_no": turn,
        "Question": f"question {turn}",
        "Answer": answer,
        "Additional_answers": [{"Answer": a} for a in extra],
    }


class TestParsers:
    def test_hotpotqa(self):
        rec = parse_hotpotqa(HOTPOT_ROW)
        assert rec.id == "hotpotqa:dev_0"
        assert rec.dataset == "hotpotqa"
        assert rec.answers == ("yes",)
        assert rec.hops == 2
        assert rec.group == rec.id
        assert rec.history == ()

    def test_musique_keeps_aliases_and_counts_hops(self):
        rec = parse_musique(MUSIQUE_ROW)
        assert rec.answers == ("Miquette Giraudy", "M. Giraudy")
        assert rec.hops == 2

    def test_musique_rejects_unanswerable(self):
        row = {**MUSIQUE_ROW, "metadata": {**MUSIQUE_ROW["metadata"], "answerable": False}}
        with pytest.raises(ValueError, match="unanswerable"):
            parse_musique(row)

    def test_topiocqa_history_pairs_and_answers(self):
        row = topiocqa_row(
            2, ["question 1", "answer 1"], "Dunkirk", ["UNANSWERABLE", "Dunkirk, France", "Dunkirk"]
        )
        rec = parse_topiocqa(row)
        assert rec.id == "topiocqa:7_2"
        assert rec.group == "topiocqa:7"
        assert rec.history == (("question 1", "answer 1"),)
        # UNANSWERABLE dropped, duplicates removed, order kept
        assert rec.answers == ("Dunkirk", "Dunkirk, France")
        assert rec.hops is None

    def test_topiocqa_unanswerable_turn_has_no_answers(self):
        rec = parse_topiocqa(topiocqa_row(1, [], "UNANSWERABLE", ["UNANSWERABLE"]))
        assert rec.answers == ()

    def test_topiocqa_odd_context_rejected(self):
        with pytest.raises(ValueError, match="context"):
            parse_topiocqa(topiocqa_row(2, ["only a question"], "x", []))


class TestLoadDataset:
    def test_reads_raw_file_for_split(self, tmp_path):
        write_jsonl(
            tmp_path / "hotpotqa" / "dev.jsonl", [HOTPOT_ROW, {**HOTPOT_ROW, "id": "dev_1"}]
        )
        records = load_dataset("hotpotqa", "dev", raw_dir=tmp_path)
        assert [r.id for r in records] == ["hotpotqa:dev_0", "hotpotqa:dev_1"]

    def test_topiocqa_dev_maps_to_valid_file(self, tmp_path):
        write_jsonl(tmp_path / "topiocqa" / "valid.jsonl", [topiocqa_row(1, [], "a", [])])
        assert len(load_dataset("topiocqa", "dev", raw_dir=tmp_path)) == 1

    def test_unknown_dataset(self, tmp_path):
        with pytest.raises(ValueError, match="unknown dataset"):
            load_dataset("nq", "dev", raw_dir=tmp_path)

    def test_duplicate_ids_rejected(self, tmp_path):
        write_jsonl(tmp_path / "hotpotqa" / "dev.jsonl", [HOTPOT_ROW, HOTPOT_ROW])
        with pytest.raises(ValueError, match="duplicate"):
            load_dataset("hotpotqa", "dev", raw_dir=tmp_path)
