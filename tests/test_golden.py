import json

import pytest

from raglab.evalharness.golden import load_golden

VALID_ROWS = [
    {"id": "q001", "question": "What is X?", "reference_answer": "X is Y.", "source_file": "a.md"},
    {"id": "q002", "question": "What is Z?", "reference_answer": "Z is W.", "source_file": "b.md"},
]


def write_jsonl(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")
    return path


def test_valid_file_loads(tmp_path):
    path = write_jsonl(tmp_path / "g.jsonl", VALID_ROWS)
    examples = load_golden(path)
    assert [e.id for e in examples] == ["q001", "q002"]
    assert examples[0].question == "What is X?"


def test_duplicate_id_raises_with_id_in_message(tmp_path):
    rows = VALID_ROWS + [dict(VALID_ROWS[0])]
    path = write_jsonl(tmp_path / "g.jsonl", rows)
    with pytest.raises(ValueError, match="q001"):
        load_golden(path)


def test_empty_question_raises(tmp_path):
    rows = [dict(VALID_ROWS[0], question="   ")]
    path = write_jsonl(tmp_path / "g.jsonl", rows)
    with pytest.raises(ValueError, match="question"):
        load_golden(path)


def test_unknown_source_file_raises_only_when_corpus_given(tmp_path):
    path = write_jsonl(tmp_path / "g.jsonl", VALID_ROWS)
    assert len(load_golden(path, corpus_files=None)) == 2
    with pytest.raises(ValueError, match="b.md"):
        load_golden(path, corpus_files={"a.md"})


def test_empty_golden_set_raises(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text("\n  \n", encoding="utf-8")
    with pytest.raises(ValueError, match="empty"):
        load_golden(path)


def test_blank_lines_ignored(tmp_path):
    path = tmp_path / "g.jsonl"
    path.write_text(
        json.dumps(VALID_ROWS[0]) + "\n\n" + json.dumps(VALID_ROWS[1]) + "\n",
        encoding="utf-8",
    )
    assert len(load_golden(path)) == 2
