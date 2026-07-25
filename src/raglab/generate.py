"""Prompt construction and the Ollama REST client."""
import requests

from raglab import retrieval
from raglab.config import GENERATION_MODEL, OLLAMA_URL, TOP_K, RagConfig
from raglab.store import RetrievedChunk

REFUSAL = "I don't know based on the provided documentation."

# Local generation is slow but not THIS slow; past this, something is wrong.
TIMEOUT_SECONDS = 300

PROMPT_TEMPLATE = """You are a precise assistant answering questions about the FastAPI documentation.

Sources:
{sources}

Rules:
- Answer using ONLY the numbered sources above.
- After each claim, cite the source that supports it, like [1] or [2].
- If the sources do not contain the answer, reply exactly: "{refusal}"
- Be concise: a few sentences, plus a short code example only if the sources contain one.

Question: {question}

Answer:"""


def build_prompt(question: str, chunks: list[RetrievedChunk]) -> str:
    sources = "\n\n".join(
        f"[{i}] ({chunk.source_file})\n{chunk.text}"
        for i, chunk in enumerate(chunks, start=1)
    )
    return PROMPT_TEMPLATE.format(sources=sources, refusal=REFUSAL, question=question)


class OllamaClient:
    def __init__(self, base_url: str = OLLAMA_URL):
        self.base_url = base_url

    def generate(
        self, prompt: str, model: str = GENERATION_MODEL, temperature: float = 0.0
    ) -> str:
        try:
            response = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": model,
                    "prompt": prompt,
                    "stream": False,
                    "options": {"temperature": temperature},
                },
                timeout=TIMEOUT_SECONDS,
            )
        except requests.ReadTimeout as exc:
            raise RuntimeError(
                f"Ollama accepted the request but produced no answer within "
                f"{TIMEOUT_SECONDS} seconds. The model is probably starved for "
                "GPU - close GPU-heavy applications (games, video encoders) "
                "and try again."
            ) from exc
        except requests.ConnectionError as exc:
            raise RuntimeError(
                f"Ollama is not reachable at {self.base_url}. Install it from "
                f"https://ollama.com, start it, then run: ollama pull {model}"
            ) from exc
        if response.status_code != 200:
            raise RuntimeError(f"Ollama returned {response.status_code}: {response.text}")
        return response.json()["response"].strip()


def answer(
    question: str, config: RagConfig, k: int = TOP_K, client: OllamaClient | None = None
) -> tuple[str, list[RetrievedChunk]]:
    """Retrieve top-k chunks, prompt the model, return (answer, sources)."""
    chunks = retrieval.retrieve(question, config, k)
    llm = client if client is not None else OllamaClient()
    return llm.generate(build_prompt(question, chunks)), chunks
