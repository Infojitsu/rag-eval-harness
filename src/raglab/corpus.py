"""Fetch and parse the FastAPI documentation corpus.

The corpus is downloaded at a pinned git tag (see config.FASTAPI_TAG) so the
evaluation numbers in docs/eval-report.md stay reproducible. It is never
committed to this repository.
"""
import io
import re
import tarfile
from dataclasses import dataclass
from pathlib import Path

import requests

from raglab.config import CORPUS_DIR, FASTAPI_TAG

DOCS_PREFIX = "docs/en/docs/"
MIN_DOC_CHARS = 200

# Not documentation content: the changelog alone is ~690 KB and would add
# ~1,000 chunks that mention every feature in passing - pure retrieval noise.
# Community/meta pages carry no answerable technical content either.
EXCLUDED_FILES = {
    "release-notes.md",
    "fastapi-people.md",
    "external-links.md",
    "management.md",
    "translation-banner.md",
    "translations.md",
    "contributing.md",
    "_llm-test.md",
}

INCLUDE_LINE_RE = re.compile(r"^\s*(\{\*.*\*\}|\{!.*!\})\s*$")
ADMONITION_FENCE_RE = re.compile(r"^///")
HTML_TAG_RE = re.compile(r"</?[a-zA-Z][a-zA-Z0-9-]*(\s[^<>]*)?/?>")
TITLE_RE = re.compile(r"^# (.+)$", re.MULTILINE)


@dataclass(frozen=True)
class Document:
    path: str  # relative posix path inside the corpus, e.g. "tutorial/body.md"
    title: str
    text: str


def fetch_corpus(dest: Path = CORPUS_DIR, tag: str = FASTAPI_TAG) -> int:
    """Download the FastAPI repo tarball at `tag`, keep only the English docs."""
    url = f"https://codeload.github.com/fastapi/fastapi/tar.gz/{tag}"
    response = requests.get(url, timeout=120)
    response.raise_for_status()

    dest.mkdir(parents=True, exist_ok=True)
    count = 0
    with tarfile.open(fileobj=io.BytesIO(response.content), mode="r:gz") as tar:
        for member in tar.getmembers():
            # Member names look like "fastapi-0.139.2/docs/en/docs/index.md".
            _, _, rel = member.name.partition("/")
            if not (member.isfile() and rel.startswith(DOCS_PREFIX) and rel.endswith(".md")):
                continue
            out_path = (dest / rel[len(DOCS_PREFIX):]).resolve()
            if not out_path.is_relative_to(dest.resolve()):
                continue  # tarball path traversal guard
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_bytes(tar.extractfile(member).read())
            count += 1
    return count


def _clean_markdown(raw: str) -> str:
    lines = []
    for line in raw.splitlines():
        if INCLUDE_LINE_RE.match(line) or ADMONITION_FENCE_RE.match(line):
            continue
        lines.append(HTML_TAG_RE.sub("", line))
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def load_documents(corpus_dir: Path = CORPUS_DIR) -> list[Document]:
    if not corpus_dir.is_dir():
        raise RuntimeError(f"Corpus not found at {corpus_dir} - run: rag fetch")
    docs = []
    for path in sorted(corpus_dir.rglob("*.md")):
        if path.relative_to(corpus_dir).as_posix() in EXCLUDED_FILES:
            continue
        raw = path.read_text(encoding="utf-8")
        text = _clean_markdown(raw)
        if len(text) < MIN_DOC_CHARS:
            continue  # redirects and stubs carry no answerable content
        title_match = TITLE_RE.search(raw)
        title = HTML_TAG_RE.sub("", title_match.group(1)).strip() if title_match else path.stem
        docs.append(Document(path.relative_to(corpus_dir).as_posix(), title, text))
    return docs


def corpus_files(corpus_dir: Path = CORPUS_DIR) -> set[str]:
    """Relative paths of every loadable (non-stub) corpus document."""
    return {doc.path for doc in load_documents(corpus_dir)}
