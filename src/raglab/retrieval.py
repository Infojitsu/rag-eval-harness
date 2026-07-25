"""Embed a question and fetch the top-k chunks for a config."""
from raglab import store
from raglab.config import TOP_K, RagConfig
from raglab.embeddings import Embedder
from raglab.store import RetrievedChunk

_EMBEDDERS: dict[str, Embedder] = {}


def get_embedder(model_name: str) -> Embedder:
    """Cache embedders - loading a sentence-transformers model is slow."""
    if model_name not in _EMBEDDERS:
        _EMBEDDERS[model_name] = Embedder(model_name)
    return _EMBEDDERS[model_name]


def retrieve(
    question: str,
    config: RagConfig,
    k: int = TOP_K,
    client=None,
    embedder=None,
) -> list[RetrievedChunk]:
    client = client if client is not None else store.get_client()
    embedder = embedder if embedder is not None else get_embedder(config.embed_model)
    query_embedding = embedder.encode_query(question)
    return store.query_collection(client, config.name, query_embedding, k)
