from search_agent_rl.data.overlap import find_overlap
from search_agent_rl.data.records import QARecord


def rec(rid: str, question: str) -> QARecord:
    return QARecord(id=rid, dataset="hotpotqa", question=question, answers=("a",), group=rid)


def test_exact_after_normalization():
    train = [rec("t1", "Who directed Inception?")]
    test = [rec("d1", "who directed  inception"), rec("d2", "Where was Nolan born?")]
    report = find_overlap(train, test)
    assert report["exact"] == ["d1"]
    assert report["n_test"] == 2


def test_same_words_reordered_is_near_duplicate_not_exact():
    train = [rec("t1", "Inception was directed by whom?")]
    test = [rec("d1", "Whom was Inception directed by?")]
    report = find_overlap(train, test)
    assert report["exact"] == []
    assert report["bag_of_words"] == ["d1"]


def test_no_overlap():
    report = find_overlap([rec("t1", "a b c")], [rec("d1", "x y z")])
    assert report == {"n_test": 1, "exact": [], "bag_of_words": []}
