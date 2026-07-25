"""ChromaDB persistent store: one collection per (chunker, embedder) config.

Embeddings are always computed by us and passed in explicitly - Chroma is
used purely as a vector index, never with its default embedding function.
"""
import json
from dataclasses import dataclass
from pathlib import Path

import chromadb

from raglab.chunking import Chunk
from raglab.config import CHROMA_DIR

BATCH_SIZE = 256
MANIFEST_PATH = CHROMA_DIR / "ingest_manifest.json"


@dataclass(frozen=True)
class RetrievedChunk:
    text: str
    source_file: str
    chunk_id: str
    score: float  # 1 - cosine distance, higher is better


def get_client(path: Path = CHROMA_DIR):
    path.mkdir(parents=True, exist_ok=True)
    return chromadb.PersistentClient(path=str(path))


def build_collection(client, name: str, chunks: list[Chunk], embeddings) -> None:
    """(Re)build a collection from scratch - ingest is idempotent."""
    try:
        client.delete_collection(name)
    except Exception:
        pass  # collection did not exist yet
    collection = client.create_collection(name, metadata={"hnsw:space": "cosine"})
    for i in range(0, len(chunks), BATCH_SIZE):
        batch = chunks[i : i + BATCH_SIZE]
        collection.add(
            ids=[c.chunk_id for c in batch],
            documents=[c.text for c in batch],
            embeddings=[[float(x) for x in e] for e in embeddings[i : i + BATCH_SIZE]],
            metadatas=[{"source_file": c.source_file} for c in batch],
        )


def record_ingest(
    name: str, tag: str, n_docs: int, n_chunks: int, manifest_path: Path = MANIFEST_PATH
) -> None:
    """Remember what corpus state a collection was built from."""
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest[name] = {"tag": tag, "n_docs": n_docs, "n_chunks": n_chunks}
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(manifest, indent=1), encoding="utf-8")


def verify_index(
    client, name: str, tag: str, n_docs: int, manifest_path: Path = MANIFEST_PATH
) -> None:
    """Refuse to evaluate a collection built from a different corpus state.

    Without this, `rag ingest --config X` after a corpus change would let the
    eval silently compare configs indexed from different corpus snapshots.
    """
    manifest = {}
    if manifest_path.exists():
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    entry = manifest.get(name)
    if entry is None:
        raise RuntimeError(f"No ingest record for '{name}' - run: rag ingest")
    if entry["tag"] != tag or entry["n_docs"] != n_docs:
        raise RuntimeError(
            f"Index '{name}' was built from corpus tag {entry['tag']} "
            f"({entry['n_docs']} docs) but the current corpus is tag {tag} "
            f"({n_docs} docs) - run: rag ingest"
        )
    try:
        count = client.get_collection(name).count()
    except Exception:
        raise RuntimeError(f"Index '{name}' is missing - run: rag ingest") from None
    if count != entry["n_chunks"]:
        raise RuntimeError(
            f"Index '{name}' has {count} chunks but its ingest record says "
            f"{entry['n_chunks']} - run: rag ingest"
        )


def query_collection(client, name: str, query_embedding, k: int) -> list[RetrievedChunk]:
    empty_error = RuntimeError(f"Index '{name}' is empty - run: rag ingest")
    try:
        collection = client.get_collection(name)
    except Exception:
        raise empty_error from None
    count = collection.count()
    if count == 0:
        raise empty_error
    result = collection.query(
        query_embeddings=[[float(x) for x in query_embedding]],
        n_results=min(k, count),
        include=["documents", "metadatas", "distances"],
    )
    return [
        RetrievedChunk(text=text, source_file=meta["source_file"], chunk_id=cid, score=1.0 - dist)
        for text, meta, dist, cid in zip(
            result["documents"][0],
            result["metadatas"][0],
            result["distances"][0],
            result["ids"][0],
        )
    ]
