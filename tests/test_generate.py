from unittest.mock import Mock, patch

import pytest
import requests

from raglab.generate import REFUSAL, OllamaClient, build_prompt
from raglab.store import RetrievedChunk

CHUNKS = [
    RetrievedChunk("Query params are declared as function args.", "tutorial/query-params.md", "tutorial/query-params.md::0", 0.9),
    RetrievedChunk("Use Annotated for validation metadata.", "tutorial/query-params-str-validations.md", "tutorial/query-params-str-validations.md::2", 0.8),
]


def test_prompt_contains_numbered_sources_and_texts():
    prompt = build_prompt("How do query params work?", CHUNKS)
    assert "[1] (tutorial/query-params.md)" in prompt
    assert "[2] (tutorial/query-params-str-validations.md)" in prompt
    for chunk in CHUNKS:
        assert chunk.text in prompt


def test_prompt_contains_refusal_instruction_and_question_last():
    prompt = build_prompt("How do query params work?", CHUNKS)
    assert REFUSAL in prompt
    assert prompt.rfind("How do query params work?") > prompt.rfind(CHUNKS[-1].text)


def test_generate_success_posts_expected_payload():
    mock_response = Mock(status_code=200)
    mock_response.json.return_value = {"response": "  the answer  "}
    with patch("raglab.generate.requests.post", return_value=mock_response) as post:
        result = OllamaClient().generate("PROMPT")
    assert result == "the answer"
    url = post.call_args.args[0] if post.call_args.args else post.call_args.kwargs["url"]
    assert url.endswith("/api/generate")
    payload = post.call_args.kwargs["json"]
    assert payload["stream"] is False
    assert payload["options"]["temperature"] == 0.0
    assert payload["prompt"] == "PROMPT"


def test_generate_connection_error_gives_install_instructions():
    with patch("raglab.generate.requests.post", side_effect=requests.ConnectionError):
        with pytest.raises(RuntimeError, match="ollama pull"):
            OllamaClient().generate("PROMPT")


def test_generate_read_timeout_gives_gpu_hint():
    # A saturated GPU makes Ollama accept the request but never answer;
    # that must surface as a clear message, not a raw ReadTimeout traceback.
    with patch("raglab.generate.requests.post", side_effect=requests.ReadTimeout):
        with pytest.raises(RuntimeError, match="GPU"):
            OllamaClient().generate("PROMPT")


def test_generate_http_error_surfaces_status():
    mock_response = Mock(status_code=404, text="model not found")
    with patch("raglab.generate.requests.post", return_value=mock_response):
        with pytest.raises(RuntimeError, match="404"):
            OllamaClient().generate("PROMPT")
