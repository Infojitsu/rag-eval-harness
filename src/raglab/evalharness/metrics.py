"""Hand-written retrieval metrics.

Each golden question has exactly one ground-truth source file, so recall@k
degenerates to hit@k: 1.0 if the truth file appears in the top-k retrieved
chunks' files, else 0.0. We keep the name recall@k because that is what it
measures in the single-relevant-document case.
"""


def recall_at_k(retrieved_files: list[str], truth_file: str, k: int) -> float:
    """1.0 if truth_file is among the first k retrieved files, else 0.0."""
    return 1.0 if truth_file in retrieved_files[:k] else 0.0


def mrr(retrieved_files: list[str], truth_file: str) -> float:
    """Reciprocal rank of the first chunk from truth_file; 0.0 if absent."""
    for rank, file in enumerate(retrieved_files, start=1):
        if file == truth_file:
            return 1.0 / rank
    return 0.0


def aggregate(per_question: list[dict[str, float]]) -> dict[str, float]:
    """Mean of every metric key across per-question rows."""
    if not per_question:
        return {}
    keys = per_question[0].keys()
    n = len(per_question)
    return {key: sum(row[key] for row in per_question) / n for key in keys}
