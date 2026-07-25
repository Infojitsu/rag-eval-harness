"""Thin wrapper around sentence-transformers with model-specific query handling."""
from sentence_transformers import SentenceTransformer

# bge-small-en-v1.5 was trained to see this prefix on short queries (but not
# on passages); skipping it measurably hurts its retrieval quality.
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "


class Embedder:
    def __init__(self, model_name: str):
        self.model_name = model_name
        # SentenceTransformer picks CUDA automatically when available.
        self.model = SentenceTransformer(model_name)

    def encode_passages(self, texts: list[str]):
        return self.model.encode(
            texts, normalize_embeddings=True, batch_size=64, show_progress_bar=False
        )

    def encode_query(self, text: str):
        if "bge" in self.model_name.lower():
            text = BGE_QUERY_PREFIX + text
        return self.model.encode(
            [text], normalize_embeddings=True, show_progress_bar=False
        )[0]
