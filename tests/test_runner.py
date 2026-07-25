from unittest.mock import Mock

import pytest

import raglab.retrieval
from raglab.config import RagConfig
from raglab.evalharness.golden import GoldenExample
from raglab.evalharness.judge import judge_answer
from raglab.evalharness.runner import evaluate_retrieval, results_table, run_grid
from raglab.store import RetrievedChunk

EXAMPLES = [
    GoldenExample("q1", "What is A?", "A is A.", "a.md"),
    GoldenExample("q2", "What is Z?", "Z is Z.", "z.md"),
]

CONFIG = RagConfig("fixed__minilm", "fixed", "model")


def fake_retrieve(question, config, k=10, client=None, embedder=None):
    # q1's truth (a.md) is at rank 1; q2's truth (z.md) is never retrieved.
    # Duplicate files simulate overlapping chunks from the same document -
    # ranks must collapse to distinct files (a, b, c, d, e), not chunk slots.
    files = ["a.md", "a.md", "b.md", "b.md", "c.md", "d.md", "d.md", "e.md"]
    return [
        RetrievedChunk(f"chunk about {f}", f, f"{f}::{i}", 0.9 - i * 0.05)
        for i, f in enumerate(files)
    ]


@pytest.fixture(autouse=True)
def patch_retrieve(monkeypatch):
    monkeypatch.setattr(raglab.retrieval, "retrieve", fake_retrieve)


def test_evaluate_retrieval_metrics_hand_computed():
    result = evaluate_retrieval(EXAMPLES, CONFIG)
    assert result["config"] == "fixed__minilm"
    assert result["n"] == 2
    # q1 hits at rank 1, q2 never hits -> means are 0.5 each, MRR 0.5.
    assert result["recall@1"] == 0.5
    assert result["recall@5"] == 0.5
    assert result["recall@10"] == 0.5
    assert result["mrr"] == 0.5


def test_ranks_collapse_duplicate_files():
    # Truth c.md sits at chunk slot 5 but is the 3rd *distinct* file, so its
    # rank must be 3: recall@3 hits and MRR contribution is 1/3.
    examples = [GoldenExample("q1", "What is C?", "C is C.", "c.md")]
    result = evaluate_retrieval(examples, CONFIG)
    assert result["recall@3"] == 1.0
    assert result["recall@1"] == 0.0
    assert result["mrr"] == pytest.approx(1.0 / 3)


def test_k_larger_than_k_max_raises():
    with pytest.raises(ValueError, match="k_max"):
        evaluate_retrieval(EXAMPLES, CONFIG, k_values=(1, 20), k_max=10)


def test_evaluate_retrieval_collects_misses_with_top5():
    result = evaluate_retrieval(EXAMPLES, CONFIG)
    assert len(result["misses"]) == 1
    miss = result["misses"][0]
    assert miss["id"] == "q2"
    assert miss["truth"] == "z.md"
    assert len(miss["retrieved"]) == 5
    assert miss["retrieved"][0]["file"] == "a.md"
    assert "snippet" in miss["retrieved"][0]


def test_results_table_one_row_per_config():
    results = run_grid(
        EXAMPLES,
        configs=[RagConfig(f"cfg{i}", "fixed", "m") for i in range(4)],
    )
    table = results_table(results)
    lines = table.splitlines()
    assert len(lines) == 2 + 4  # header + divider + 4 rows
    assert "recall@1" in lines[0]
    assert all(f"cfg{i}" in table for i in range(4))


def make_judge_client(reply: str):
    client = Mock()
    client.generate.return_value = reply
    return client


def test_judge_parses_correct():
    result = judge_answer("q", "ref", "ans", make_judge_client("CORRECT - matches."))
    assert result["verdict"] == "correct"


def test_judge_parses_lowercase_and_punctuation():
    result = judge_answer("q", "ref", "ans", make_judge_client("correct."))
    assert result["verdict"] == "correct"


def test_judge_parses_wrapped_verdicts():
    for reply in ("`CORRECT`", "**CORRECT** - matches", "Verdict: CORRECT", "(CORRECT)"):
        assert judge_answer("q", "r", "a", make_judge_client(reply))["verdict"] == "correct"


def test_judge_wrapped_incorrect_not_misread_as_correct():
    for reply in ("`INCORRECT`", "Verdict: INCORRECT - contradicts"):
        assert judge_answer("q", "r", "a", make_judge_client(reply))["verdict"] == "incorrect"


def test_judge_unparseable_counts_as_incorrect():
    result = judge_answer("q", "ref", "ans", make_judge_client("It is right."))
    assert result["verdict"] == "incorrect"


def test_judge_empty_reply_counts_as_incorrect():
    result = judge_answer("q", "ref", "ans", make_judge_client("   "))
    assert result["verdict"] == "incorrect"
