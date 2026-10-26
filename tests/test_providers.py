"""OpenAI-compatible providers: right URL, right key, right token field, no key for Ollama."""

import asyncio
import json

import httpx
import pytest

from mistrial.cache import Cache
from mistrial.cases import Case
from mistrial.judges import LLM_PROVIDERS, PROVIDERS, LLMJudge, build_judge

CASE = Case("c1", "What is 2+2?", "4", "4", True)


def _reply(content='{"score": true}') -> dict:
    return {"choices": [{"finish_reason": "stop", "message": {"role": "assistant", "content": content}}]}


def _run(judge: LLMJudge, handler):
    judge._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    return asyncio.run(judge.judge(CASE, 0, Cache(enabled=False)))


def _capture(seen: list):
    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((str(request.url), request.headers.get("authorization"), json.loads(request.content)))
        return httpx.Response(200, json=_reply())

    return handler


def test_gemini_uses_google_openai_endpoint_and_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "g-test")
    seen: list = []
    raw, _, _ = _run(LLMJudge("gemini", "gemini-2.5-flash"), _capture(seen))
    url, auth, body = seen[0]
    assert raw["score"] is True
    assert url == "https://generativelanguage.googleapis.com/v1beta/openai/chat/completions"
    assert auth == "Bearer g-test"
    assert body["max_tokens"] == 1024 and "max_completion_tokens" not in body


def test_gemini_accepts_google_api_key_alias(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_API_KEY", "goog")
    seen: list = []
    _run(LLMJudge("gemini", "gemini-2.5-flash"), _capture(seen))
    assert seen[0][1] == "Bearer goog"


def test_ollama_needs_no_key_and_defaults_to_localhost(monkeypatch):
    monkeypatch.delenv("OLLAMA_BASE_URL", raising=False)
    seen: list = []
    _run(LLMJudge("ollama", "llama3.2"), _capture(seen))
    url, auth, body = seen[0]
    assert url == "http://localhost:11434/v1/chat/completions" and auth is None
    assert body["model"] == "llama3.2"


def test_base_url_override(monkeypatch):
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://gpu-box:11434/v1/")
    seen: list = []
    _run(LLMJudge("ollama", "llama3.2"), _capture(seen))
    assert seen[0][0] == "http://gpu-box:11434/v1/chat/completions"


def test_custom_requires_base_url(monkeypatch):
    monkeypatch.delenv("MISTRIAL_BASE_URL", raising=False)
    monkeypatch.delenv("CUSTOM_BASE_URL", raising=False)
    with pytest.raises(RuntimeError, match="MISTRIAL_BASE_URL"):
        _run(LLMJudge("custom", "my-model"), _capture([]))
    monkeypatch.setenv("MISTRIAL_BASE_URL", "http://localhost:8000/v1")
    seen: list = []
    _run(LLMJudge("custom", "my-model"), _capture(seen))
    assert seen[0][0] == "http://localhost:8000/v1/chat/completions" and seen[0][1] is None


def test_hosted_provider_without_key_fails_clearly(monkeypatch):
    monkeypatch.delenv("GROQ_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="GROQ_API_KEY"):
        _run(LLMJudge("groq", "llama-3.3-70b"), _capture([]))


def test_json_mode_dropped_when_server_rejects_it(monkeypatch):
    bodies: list = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "response_format" in body:
            return httpx.Response(400, json={"error": {"message": "response_format is not supported"}})
        return httpx.Response(200, json=_reply('Sure: {"score": false}'))

    raw, _, _ = _run(LLMJudge("ollama", "tiny"), handler)
    assert raw["score"] is False and len(bodies) == 2 and "response_format" not in bodies[1]


def test_build_judge_knows_every_provider():
    for p in LLM_PROVIDERS:
        j = build_judge(f"{p}:some-model")
        assert isinstance(j, LLMJudge) and j.provider == p and j.name == f"{p}:some-model"
    with pytest.raises(ValueError, match="unknown provider"):
        LLMJudge("bedrock", "x")


def test_every_hosted_provider_has_https_base_url():
    for name, (url, envs) in PROVIDERS.items():
        if name in ("ollama", "custom"):
            continue
        assert url.startswith("https://") and envs, name
