"""The OpenAI and Anthropic adapters, exercised against a fake HTTP server (no network)."""

import asyncio
import json

import httpx
import pytest

from judgeflake.cache import Cache
from judgeflake.cases import Case
from judgeflake.judges import LLMJudge

CASE = Case("c1", "What is 2+2?", "4", "4", True)


def _openai_reply(content: str, finish: str = "stop") -> dict:
    return {"choices": [{"finish_reason": finish, "message": {"role": "assistant", "content": content}}]}


def _anthropic_reply(text: str, stop: str = "end_turn") -> dict:
    return {"stop_reason": stop, "content": [{"type": "text", "text": text}]}


def _client(handler) -> httpx.AsyncClient:
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_openai_request_shape_and_parse(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        seen.append((request.url.path, request.headers.get("authorization"), body))
        return httpx.Response(200, json=_openai_reply('{"score": true, "reasoning": "4 is right"}'))

    j = LLMJudge("openai", "gpt-5-mini", rubric="correctness", reasoning_effort="low")
    j._client = _client(handler)
    raw, rationale, cached = asyncio.run(j.judge(CASE, 0, Cache(enabled=False)))
    assert raw["score"] is True and rationale == "4 is right" and cached is False
    path, auth, body = seen[0]
    assert path == "/v1/chat/completions" and auth == "Bearer sk-test"
    assert body["model"] == "gpt-5-mini" and body["max_completion_tokens"] == 1024
    assert body["response_format"] == {"type": "json_object"} and body["reasoning_effort"] == "low"
    assert "temperature" not in body
    assert body["messages"][0]["role"] == "system"
    assert "REFERENCE:\n4" in body["messages"][1]["content"] and "OUTPUT:\n4" in body["messages"][1]["content"]


def test_openai_drops_rejected_params_and_retries_once(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    bodies = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        bodies.append(body)
        if "temperature" in body:
            return httpx.Response(400, json={"error": {"message": "Unsupported parameter: 'temperature' is not supported with this model."}})
        return httpx.Response(200, json=_openai_reply('{"score": false}'))

    j = LLMJudge("openai", "gpt-5.1", temperature=0.0)
    j._client = _client(handler)
    raw, _, _ = asyncio.run(j.judge(CASE, 0, Cache(enabled=False)))
    assert raw["score"] is False
    assert "temperature" in bodies[0] and "temperature" not in bodies[1] and len(bodies) == 2


def test_openai_retries_on_429_then_succeeds(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        if calls["n"] < 3:
            return httpx.Response(429, json={"error": "slow down"})
        return httpx.Response(200, json=_openai_reply('```json\n{"score": true}\n```'))

    j = LLMJudge("openai", "gpt-5-mini", retries=4)
    j._client = _client(handler)
    monkeypatch.setattr("judgeflake.judges.asyncio.sleep", _no_sleep)
    raw, _, _ = asyncio.run(j.judge(CASE, 0, Cache(enabled=False)))
    assert raw["score"] is True and calls["n"] == 3


def test_openai_hard_errors_surface(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    j = LLMJudge("openai", "gpt-5-mini")
    j._client = _client(lambda r: httpx.Response(401, json={"error": {"message": "bad key"}}))
    with pytest.raises(RuntimeError, match="401"):
        asyncio.run(j.judge(CASE, 0, Cache(enabled=False)))
    j._client = _client(lambda r: httpx.Response(200, json=_openai_reply("{", finish="length")))
    with pytest.raises(RuntimeError, match="cut off"):
        asyncio.run(j.judge(CASE, 0, Cache(enabled=False)))


def test_anthropic_request_shape_and_refusal(monkeypatch):
    monkeypatch.setenv("ANTHROPIC_API_KEY", "ak-test")
    seen = []

    def handler(request: httpx.Request) -> httpx.Response:
        seen.append((request.url.path, dict(request.headers), json.loads(request.content)))
        return httpx.Response(200, json=_anthropic_reply('Here: {"score": 4, "reasoning": "good"}'))

    j = LLMJudge("anthropic", "claude-sonnet-5-5", rubric="quality")
    j._client = _client(handler)
    raw, rationale, _ = asyncio.run(j.judge(Case("q", "Explain DNS", "DNS maps names to addresses.", None), 0, Cache(enabled=False)))
    assert raw["score"] == 4 and rationale == "good"
    path, headers, body = seen[0]
    assert path == "/v1/messages" and headers["x-api-key"] == "ak-test" and headers["anthropic-version"]
    assert body["model"] == "claude-sonnet-5-5" and body["max_tokens"] == 1024 and body["system"]
    assert body["messages"] == [{"role": "user", "content": body["messages"][0]["content"]}]
    assert "1 to 5" in body["messages"][0]["content"] and "(none)" not in body["messages"][0]["content"]

    j._client = _client(lambda r: httpx.Response(200, json=_anthropic_reply("", stop="refusal")))
    with pytest.raises(RuntimeError, match="refused"):
        asyncio.run(j.judge(CASE, 0, Cache(enabled=False)))


def test_llm_cache_hits_skip_the_network(monkeypatch, tmp_path):
    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    calls = {"n": 0}

    def handler(request: httpx.Request) -> httpx.Response:
        calls["n"] += 1
        return httpx.Response(200, json=_openai_reply('{"score": true}'))

    j = LLMJudge("openai", "gpt-5-mini")
    j._client = _client(handler)
    cache = Cache(tmp_path / "c.sqlite")
    asyncio.run(j.judge(CASE, 0, cache))
    _, _, cached = asyncio.run(j.judge(CASE, 0, cache))
    asyncio.run(j.judge(CASE, 1, cache))  # a different run index is a different sample
    assert cached is True and calls["n"] == 2
    cache.close()


async def _no_sleep(_s: float) -> None:
    return None
