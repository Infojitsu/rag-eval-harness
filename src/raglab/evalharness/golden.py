"""Load and validate the golden QA set."""
import json
from dataclasses import dataclass
from pathlib import Path

from raglab.config import GOLDEN_PATH

REQUIRED_FIELDS = ("id", "question", "reference_answer", "source_file")


@dataclass(frozen=True)
class GoldenExample:
    id: str
    question: str
    reference_answer: str
    source_file: str


def load_golden(
    path: Path = GOLDEN_PATH, corpus_files: set[str] | None = None
) -> list[GoldenExample]:
    """Parse the JSONL golden set, failing loudly on any malformed example."""
    examples: list[GoldenExample] = []
    seen_ids: set[str] = set()
    for lineno, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        row = json.loads(line)
        for field in REQUIRED_FIELDS:
            if not str(row.get(field, "")).strip():
                raise ValueError(f"golden set line {lineno}: missing or empty '{field}'")
        if row["id"] in seen_ids:
            raise ValueError(f"golden set line {lineno}: duplicate id '{row['id']}'")
        seen_ids.add(row["id"])
        if corpus_files is not None and row["source_file"] not in corpus_files:
            raise ValueError(
                f"golden set line {lineno} ({row['id']}): source_file "
                f"'{row['source_file']}' is not in the corpus"
            )
        examples.append(
            GoldenExample(
                id=row["id"],
                question=row["question"],
                reference_answer=row["reference_answer"],
                source_file=row["source_file"],
            )
        )
    if not examples:
        raise ValueError(f"golden set at {path} is empty")
    return examples
