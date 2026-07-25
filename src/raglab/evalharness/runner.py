"""Run the golden set through the retrieval grid and (optionally) the judge."""
import csv
import json
from pathlib import Path

from raglab import retrieval
from raglab.config import CONFIGS, RESULTS_DIR, RagConfig
from raglab.evalharness.golden import GoldenExample
from raglab.evalharness.judge import judge_answer
from raglab.evalharness.metrics import aggregate, mrr, recall_at_k
from raglab.generate import OllamaClient, answer

K_VALUES = (1, 3, 5, 10)
K_MAX = 10
# Retrieve deeper than k_max so that after collapsing duplicate files we still
# have k_max distinct candidates to rank.
RETRIEVE_CHUNKS = 50
METRIC_COLUMNS = [f"recall@{k}" for k in K_VALUES] + ["mrr"]


def evaluate_retrieval(
    examples: list[GoldenExample],
    config: RagConfig,
    k_values: tuple[int, ...] = K_VALUES,
    k_max: int = K_MAX,
    client=None,
) -> dict:
    """File-level retrieval metrics for one config, plus its missed questions.

    Ranks are computed over *distinct files* (first occurrence wins), not raw
    chunk slots. Otherwise a chunker that splits files into many overlapping
    chunks fills top-k slots with duplicates of the same file, and the
    cross-chunker comparison measures slot accounting, not retrieval quality.
    """
    if max(k_values) > k_max:
        raise ValueError(
            f"k_max={k_max} is smaller than the largest k in k_values={k_values}; "
            "metrics beyond k_max would be silently truncated"
        )
    rows, misses = [], []
    for example in examples:
        chunks = retrieval.retrieve(
            example.question, config, k=RETRIEVE_CHUNKS, client=client
        )
        files = list(dict.fromkeys(c.source_file for c in chunks))[:k_max]
        row = {f"recall@{k}": recall_at_k(files, example.source_file, k) for k in k_values}
        row["mrr"] = mrr(files, example.source_file)
        rows.append(row)
        if example.source_file not in files:
            misses.append(
                {
                    "id": example.id,
                    "question": example.question,
                    "truth": example.source_file,
                    "retrieved": [
                        {
                            "file": c.source_file,
                            "chunk_id": c.chunk_id,
                            "score": round(c.score, 4),
                            "snippet": c.text[:200],
                        }
                        for c in chunks[:5]
                    ],
                }
            )
    return {"config": config.name, "n": len(examples), **aggregate(rows), "misses": misses}


def run_grid(examples: list[GoldenExample], configs=None, client=None) -> list[dict]:
    configs = list(CONFIGS.values()) if configs is None else list(configs)
    return [evaluate_retrieval(examples, config, client=client) for config in configs]


def results_table(results: list[dict]) -> str:
    header = "| config | " + " | ".join(METRIC_COLUMNS) + " |"
    divider = "|---" * (len(METRIC_COLUMNS) + 1) + "|"
    lines = [header, divider]
    for result in results:
        cells = " | ".join(f"{result[col]:.3f}" for col in METRIC_COLUMNS)
        lines.append(f"| {result['config']} | {cells} |")
    return "\n".join(lines)


def save_results(results: list[dict], results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "retrieval_results.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["config", "n"] + METRIC_COLUMNS)
        writer.writeheader()
        for result in results:
            writer.writerow({k: result[k] for k in ["config", "n"] + METRIC_COLUMNS})
    (results_dir / "retrieval_results.md").write_text(results_table(results), encoding="utf-8")
    for result in results:
        misses_path = results_dir / f"misses_{result['config']}.jsonl"
        misses_path.write_text(
            "\n".join(json.dumps(m) for m in result["misses"]), encoding="utf-8"
        )


def evaluate_answers(
    examples: list[GoldenExample], config: RagConfig, client: OllamaClient | None = None
) -> dict:
    """Generate an answer per golden question and let the LLM judge grade it."""
    llm = client if client is not None else OllamaClient()
    details = []
    for example in examples:
        generated, chunks = answer(example.question, config, client=llm)
        verdict = judge_answer(example.question, example.reference_answer, generated, llm)
        details.append(
            {
                "id": example.id,
                "question": example.question,
                "reference": example.reference_answer,
                "generated": generated,
                "verdict": verdict["verdict"],
                "judge_raw": verdict["raw"],
                "sources": [c.source_file for c in chunks],
            }
        )
    correct = sum(1 for d in details if d["verdict"] == "correct")
    return {
        "config": config.name,
        "n": len(details),
        "judged_correct": correct / len(details) if details else 0.0,
        "details": details,
    }


def save_answers(result: dict, results_dir: Path = RESULTS_DIR) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"answers_{result['config']}.jsonl"
    path.write_text("\n".join(json.dumps(d) for d in result["details"]), encoding="utf-8")
