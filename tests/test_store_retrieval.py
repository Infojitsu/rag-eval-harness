import numpy as np
import pytest

from raglab import retrieval, store
from raglab.chunking import Chunk
from raglab.config import RagConfig

CHUNKS = [Chunk(f"text {i}", f"f{i}.md", f"f{i}.md::0") for i in range(6)]
# Orthogonal one-hot embeddings make nearest-neighbor results exact.
EMBEDDINGS = np.eye(8)[:6]


@pytest.fixture()
def client(tmp_path):
    return store.get_client(tmp_path / "chroma")


def test_round_trip_nearest_first(client):
    store.build_collection(client, "cfg", CHUNKS, EMBEDDINGS)
    hits = store.query_collection(client, "cfg", np.eye(8)[3], k=3)
    assert hits[0].chunk_id == "f3.md::0"
    assert hits[0].source_file == "f3.md"
    assert hits[0].text == "text 3"
    assert hits[0].score == pytest.approx(1.0, abs=1e-5)
    assert len(hits) == 3


def test_k_is_respected(client):
    store.build_collection(client, "cfg", CHUNKS, EMBEDDINGS)
    assert len(store.query_collection(client, "cfg", np.eye(8)[0], k=2)) == 2


def test_k_larger_than_collection_is_clamped(client):
    store.build_collection(client, "cfg", CHUNKS, EMBEDDINGS)
    assert len(store.query_collection(client, "cfg", np.eye(8)[0], k=50)) == 6


def test_missing_collection_raises_helpful_error(client):
    with pytest.raises(RuntimeError, match="rag ingest"):
        store.query_collection(client, "nope", np.eye(8)[0], k=3)


def test_rebuild_is_idempotent(client):
    store.build_collection(client, "cfg", CHUNKS, EMBEDDINGS)
    store.build_collection(client, "cfg", CHUNKS, EMBEDDINGS)
    assert client.get_collection("cfg").count() == 6


def test_verify_index_passes_on_matching_manifest(client, tmp_path):
    manifest = tmp_path / "manifest.json"
    store.build_collection(client, "cfg", CHUNKS, EMBEDDINGS)
    store.record_ingest("cfg", "0.139.2", 140, 6, manifest_path=manifest)
    store.verify_index(client, "cfg", "0.139.2", 140, manifest_path=manifest)


def test_verify_index_rejects_missing_record(client, tmp_path):
    with pytest.raises(RuntimeError, match="rag ingest"):
        store.verify_index(client, "cfg", "0.139.2", 140, manifest_path=tmp_path / "m.json")


def test_verify_index_rejects_different_corpus_state(client, tmp_path):
    manifest = tmp_path / "manifest.json"
    store.build_collection(client, "cfg", CHUNKS, EMBEDDINGS)
    store.record_ingest("cfg", "0.139.2", 140, 6, manifest_path=manifest)
    with pytest.raises(RuntimeError, match="built from corpus tag"):
        store.verify_index(client, "cfg", "0.140.0", 140, manifest_path=manifest)
    with pytest.raises(RuntimeError, match="built from corpus tag"):
        store.verify_index(client, "cfg", "0.139.2", 141, manifest_path=manifest)


def test_verify_index_rejects_chunk_count_mismatch(client, tmp_path):
    manifest = tmp_path / "manifest.json"
    store.build_collection(client, "cfg", CHUNKS, EMBEDDINGS)
    store.record_ingest("cfg", "0.139.2", 140, 999, manifest_path=manifest)
    with pytest.raises(RuntimeError, match="999"):
        store.verify_index(client, "cfg", "0.139.2", 140, manifest_path=manifest)


class FakeEmbedder:
    def encode_query(self, text):
        return np.eye(8)[1]


def test_retrieve_uses_embedder_and_config_name(client):
    store.build_collection(client, "test__cfg", CHUNKS, EMBEDDINGS)
    config = RagConfig("test__cfg", "fixed", "unused-model")
    hits = retrieval.retrieve(
        "anything", config, k=2, client=client, embedder=FakeEmbedder()
    )
    assert hits[0].chunk_id == "f1.md::0"
