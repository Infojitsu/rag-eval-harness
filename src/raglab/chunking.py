"""Two chunking strategies for the A/B evaluation.

- fixed: sliding window of ~`size` characters with `overlap`, cutting at word
  boundaries. The classic baseline.
- sections: split on Markdown headings, merge tiny sections, re-split huge
  ones, and prefix every chunk with "{doc title} - {section heading}" so the
  chunk carries its own context.
"""
import re
from dataclasses import dataclass

HEADING_RE = re.compile(r"^#{1,3}\s+(.*)$")


@dataclass(frozen=True)
class Chunk:
    text: str
    source_file: str
    chunk_id: str


def _split_fixed(text: str, size: int, overlap: int) -> list[str]:
    """Split text into <=size pieces at word boundaries, with overlap."""
    text = text.strip()
    pieces = []
    start = 0
    while start < len(text):
        end = min(start + size, len(text))
        if end < len(text):
            cut = max(text.rfind(" ", start, end), text.rfind("\n", start, end))
            if cut > start:
                end = cut
        piece = text[start:end].strip()
        if piece:
            pieces.append(piece)
        if end >= len(text):
            break
        # Step back by `overlap`, then snap forward to the next word start.
        # A chunk can still begin mid-word only when a single unbroken token
        # (e.g. a very long URL) spans the entire overlap region.
        start = max(end - overlap, start + 1)
        if not text[start - 1].isspace():
            boundary = min(
                (p for p in (text.find(" ", start), text.find("\n", start)) if p != -1),
                default=-1,
            )
            if boundary != -1 and boundary < end:
                start = boundary + 1
    return pieces


def _make_chunks(pieces: list[str], source_file: str) -> list[Chunk]:
    return [
        Chunk(text=piece, source_file=source_file, chunk_id=f"{source_file}::{i}")
        for i, piece in enumerate(pieces)
    ]


def chunk_fixed(
    text: str, source_file: str, size: int = 800, overlap: int = 150
) -> list[Chunk]:
    return _make_chunks(_split_fixed(text, size, overlap), source_file)


def chunk_documents(docs, chunker: str) -> list[Chunk]:
    """Chunk a list of corpus Documents with the named strategy."""
    chunks: list[Chunk] = []
    for doc in docs:
        if chunker == "fixed":
            chunks.extend(chunk_fixed(doc.text, doc.path))
        elif chunker == "sections":
            chunks.extend(chunk_sections(doc.text, doc.path, doc.title))
        else:
            raise ValueError(f"Unknown chunker '{chunker}' (expected 'fixed' or 'sections')")
    return chunks


def _split_headings(text: str) -> list[tuple[str, str]]:
    """Split into (heading, body) pairs; text before any heading -> 'Intro'."""
    sections = []
    heading = "Intro"
    lines: list[str] = []
    for line in text.splitlines():
        match = HEADING_RE.match(line)
        if match:
            sections.append((heading, "\n".join(lines).strip()))
            heading = match.group(1).strip()
            lines = []
        else:
            lines.append(line)
    sections.append((heading, "\n".join(lines).strip()))
    return [(h, b) for h, b in sections if b]


def chunk_sections(
    text: str,
    source_file: str,
    title: str,
    max_chars: int = 2000,
    min_chars: int = 200,
) -> list[Chunk]:
    merged: list[tuple[str, str]] = []
    pending = ""  # tiny leading sections waiting for a home
    for heading, body in _split_headings(text):
        if pending:
            body = f"{pending}\n\n{heading}\n{body}"
            heading, pending = "Intro", ""
        if len(body) < min_chars:
            if merged:
                prev_heading, prev_body = merged[-1]
                merged[-1] = (prev_heading, f"{prev_body}\n\n{heading}\n{body}")
            else:
                pending = f"{heading}\n{body}"
            continue
        merged.append((heading, body))
    if pending:  # document had only tiny sections
        merged.append(("Intro", pending))

    pieces = []
    for heading, body in merged:
        prefix = f"{title} - {heading}\n"
        if len(body) > max_chars:
            pieces.extend(prefix + part for part in _split_fixed(body, max_chars, 150))
        else:
            pieces.append(prefix + body)
    return _make_chunks(pieces, source_file)
