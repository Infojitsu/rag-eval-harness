"""The repo is pure ASCII by policy - no emojis, no typographic dashes."""
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
SKIP_DIRS = {".git", ".venv", "__pycache__", ".pytest_cache", "data"}
CHECKED_SUFFIXES = {".py", ".md", ".txt", ".toml", ".jsonl", ".tex"}


def checked_files():
    for path in REPO_ROOT.rglob("*"):
        if any(part in SKIP_DIRS or part.endswith(".egg-info") for part in path.parts):
            continue
        if path.is_file() and path.suffix in CHECKED_SUFFIXES:
            yield path
    yield REPO_ROOT / "data" / "golden_set.jsonl"  # the one tracked file in data/


def test_all_tracked_text_is_pure_ascii():
    offenders = []
    for path in checked_files():
        text = path.read_text(encoding="utf-8")
        for lineno, line in enumerate(text.splitlines(), start=1):
            bad = [ch for ch in line if ord(ch) > 127]
            if bad:
                offenders.append(
                    f"{path.relative_to(REPO_ROOT)}:{lineno} "
                    f"{[f'U+{ord(c):04X}' for c in bad]}"
                )
    assert not offenders, "Non-ASCII characters found:\n" + "\n".join(offenders)
