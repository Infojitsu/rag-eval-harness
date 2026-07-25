from raglab.chunking import chunk_fixed, chunk_sections

# 400 numbered 8-char tokens ("tok00000 tok00001 ...") -> 3600 chars of text.
TOKENS = [f"tok{i:05d}" for i in range(400)]
TOKEN_TEXT = " ".join(TOKENS)


def test_fixed_respects_size_bound():
    chunks = chunk_fixed(TOKEN_TEXT, "a.md", size=800, overlap=150)
    assert len(chunks) > 1
    assert all(len(c.text) <= 800 for c in chunks)


def test_fixed_no_mid_word_cuts():
    chunks = chunk_fixed(TOKEN_TEXT, "a.md", size=800, overlap=150)
    vocab = set(TOKENS)
    for chunk in chunks:
        assert set(chunk.text.split()) <= vocab


def test_fixed_overlap_continuity():
    chunks = chunk_fixed(TOKEN_TEXT, "a.md", size=800, overlap=150)
    for prev, nxt in zip(chunks, chunks[1:]):
        # The first token of the next chunk must sit inside the previous
        # chunk's tail -- that is what "overlap" means.
        first_token = nxt.text.split()[0]
        assert first_token in prev.text


def test_fixed_short_text_single_chunk():
    chunks = chunk_fixed("just a few words", "a.md")
    assert len(chunks) == 1
    assert chunks[0].text == "just a few words"


def test_fixed_whitespace_free_text_terminates():
    chunks = chunk_fixed("a" * 1000, "a.md", size=800, overlap=150)
    assert [len(c.text) for c in chunks] == [800, 350]


def test_fixed_empty_text():
    assert chunk_fixed("   ", "a.md") == []


def test_fixed_ids_sequential():
    chunks = chunk_fixed(TOKEN_TEXT, "a.md", size=800, overlap=150)
    assert [c.chunk_id for c in chunks] == [
        f"a.md::{i}" for i in range(len(chunks))
    ]


SECTIONED = (
    "Some intro text before any heading. " * 10
    + "\n# Guide\n"
    + "Opening paragraph of the guide. " * 10
    + "\n## First topic\n"
    + "Body of the first topic. " * 20
    + "\n## Tiny\nOne short line.\n"
    + "## Second topic\n"
    + "Body of the second topic. " * 20
)


def test_sections_split_on_headings():
    chunks = chunk_sections(SECTIONED, "g.md", title="Guide")
    joined_headers = [c.text.splitlines()[0] for c in chunks]
    assert any("First topic" in h for h in joined_headers)
    assert any("Second topic" in h for h in joined_headers)


def test_sections_intro_text_kept():
    chunks = chunk_sections(SECTIONED, "g.md", title="Guide")
    assert any("Intro" in c.text.splitlines()[0] for c in chunks)
    assert any("intro text before any heading" in c.text for c in chunks)


def test_sections_tiny_section_merged():
    chunks = chunk_sections(SECTIONED, "g.md", title="Guide")
    headers = [c.text.splitlines()[0] for c in chunks]
    # "Tiny" is under min_chars, so it must not get its own chunk...
    assert not any("Tiny" in h for h in headers)
    # ...but its content survives inside the previous section's chunk.
    assert any("One short line." in c.text for c in chunks)


def test_sections_prefix_has_title_and_heading():
    chunks = chunk_sections(SECTIONED, "g.md", title="Guide")
    assert all(c.text.startswith("Guide - ") for c in chunks)


def test_sections_huge_section_resplit():
    huge = "# Doc\n## Big\n" + "word " * 1000  # 5000-char body
    chunks = chunk_sections(huge, "h.md", title="Doc", max_chars=2000)
    assert len(chunks) > 1
    assert all(len(c.text) <= 2000 + len("Doc - Big\n") for c in chunks)


def test_sections_ids_sequential():
    chunks = chunk_sections(SECTIONED, "g.md", title="Guide")
    assert [c.chunk_id for c in chunks] == [
        f"g.md::{i}" for i in range(len(chunks))
    ]
