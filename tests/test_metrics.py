from raglab.evalharness.metrics import aggregate, mrr, recall_at_k

FILES = ["a.md", "b.md", "c.md", "d.md", "e.md"]


def test_recall_at_k_truth_at_rank_1():
    assert recall_at_k(FILES, "a.md", k=1) == 1.0
    assert recall_at_k(FILES, "a.md", k=3) == 1.0


def test_recall_at_k_truth_at_rank_3():
    assert recall_at_k(FILES, "c.md", k=1) == 0.0
    assert recall_at_k(FILES, "c.md", k=3) == 1.0
    assert recall_at_k(FILES, "c.md", k=5) == 1.0


def test_recall_at_k_truth_absent():
    assert recall_at_k(FILES, "zzz.md", k=5) == 0.0


def test_recall_at_k_k_larger_than_list():
    assert recall_at_k(["a.md"], "a.md", k=10) == 1.0


def test_recall_at_k_empty_retrieved():
    assert recall_at_k([], "a.md", k=5) == 0.0


def test_mrr_rank_1():
    assert mrr(FILES, "a.md") == 1.0


def test_mrr_rank_3():
    assert mrr(FILES, "c.md") == 1.0 / 3


def test_mrr_absent():
    assert mrr(FILES, "zzz.md") == 0.0


def test_mrr_duplicate_truth_counts_first_occurrence():
    # Several chunks of the same file: rank of the *first* one counts.
    assert mrr(["b.md", "a.md", "a.md"], "a.md") == 1.0 / 2


def test_mrr_empty_retrieved():
    assert mrr([], "a.md") == 0.0


def test_aggregate_means_per_key():
    rows = [
        {"recall@1": 1.0, "mrr": 1.0},
        {"recall@1": 0.0, "mrr": 0.5},
    ]
    assert aggregate(rows) == {"recall@1": 0.5, "mrr": 0.75}


def test_aggregate_empty_returns_empty():
    assert aggregate([]) == {}
