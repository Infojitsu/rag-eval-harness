from pathlib import Path

import pytest

from raglab.corpus import corpus_files, load_documents

FIXTURES = Path(__file__).parent / "fixtures" / "mini_corpus"


def test_loads_three_docs_skips_tiny_and_excluded_files():
    # tiny.md is under the size floor; release-notes.md is on the exclusion list.
    docs = load_documents(FIXTURES)
    assert {d.path for d in docs} == {"index.md", "tutorial.md", "advanced.md"}


def test_title_comes_from_first_heading():
    docs = {d.path: d for d in load_documents(FIXTURES)}
    assert docs["index.md"].title == "Teapot API"
    assert docs["tutorial.md"].title == "Brewing Tutorial"


def test_include_directives_stripped():
    docs = load_documents(FIXTURES)
    for doc in docs:
        assert "{*" not in doc.text
        assert "{!" not in doc.text


def test_admonition_fences_stripped_but_content_kept():
    docs = {d.path: d for d in load_documents(FIXTURES)}
    text = docs["advanced.md"].text
    assert not any(line.startswith("///") for line in text.splitlines())
    assert "Keep water below boiling for green tea." in text


def test_html_tags_stripped_but_content_kept():
    docs = {d.path: d for d in load_documents(FIXTURES)}
    text = docs["index.md"].text
    assert "<abbr" not in text
    assert "API" in text


def test_code_blocks_survive_cleaning():
    docs = {d.path: d for d in load_documents(FIXTURES)}
    assert 'requests.post(' in docs["tutorial.md"].text


def test_corpus_files_returns_relative_posix_paths():
    assert corpus_files(FIXTURES) == {"index.md", "tutorial.md", "advanced.md"}


def test_missing_corpus_dir_raises_helpful_error():
    with pytest.raises(RuntimeError, match="rag fetch"):
        load_documents(FIXTURES / "does_not_exist")
