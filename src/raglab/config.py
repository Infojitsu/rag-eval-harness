"""Central configuration: the 2x2 eval grid, paths, and model settings."""
from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = REPO_ROOT / "data"
CORPUS_DIR = DATA_DIR / "corpus"
CHROMA_DIR = DATA_DIR / "chroma"
RESULTS_DIR = DATA_DIR / "results"
GOLDEN_PATH = DATA_DIR / "golden_set.jsonl"

OLLAMA_URL = "http://localhost:11434"
GENERATION_MODEL = "llama3.1:8b"

# Corpus is fetched at this exact tag so eval numbers are reproducible.
FASTAPI_TAG = "0.139.2"

TOP_K = 5


@dataclass(frozen=True)
class RagConfig:
    name: str
    chunker: str  # "fixed" | "sections"
    embed_model: str


CONFIGS = {
    "fixed__minilm": RagConfig(
        "fixed__minilm", "fixed", "sentence-transformers/all-MiniLM-L6-v2"
    ),
    "fixed__bge": RagConfig("fixed__bge", "fixed", "BAAI/bge-small-en-v1.5"),
    "sections__minilm": RagConfig(
        "sections__minilm", "sections", "sentence-transformers/all-MiniLM-L6-v2"
    ),
    "sections__bge": RagConfig("sections__bge", "sections", "BAAI/bge-small-en-v1.5"),
}

# Which config `rag ask` and the Streamlit app use. Winner of the 2x2 eval
# grid by MRR - see docs/eval-report.md.
DEFAULT_CONFIG = "fixed__bge"
