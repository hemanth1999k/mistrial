"""Judges: anything that turns a Case into a verdict.

Three kinds:
- LLMJudge      an OpenAI or Anthropic model with a rubric (direct REST via httpx, no SDK)
- CallableJudge any Python function with the LangSmith / openevals evaluator signature
- DemoJudge     a deterministic fake with tunable flakiness, so the tool works with no keys
"""

from __future__ import annotations

import asyncio
import hashlib
import importlib
import inspect
import json
import os
import random
import re
import time
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Protocol

import httpx

from .cache import Cache, key_for
from .cases import Case
from .prompts import SYSTEM, load_rubric, render


@dataclass
class Verdict:
    case_id: str
    judge: str
    run: int
    value: Any = None  # normalized by stats.normalize; None when errored
    raw: Any = None
    rationale: str | None = None
    error: str | None = None
    latency_ms: float = 0.0
    cached: bool = False

    def to_dict(self) -> dict:
        return {
            "case_id": self.case_id,
            "judge": self.judge,
            "run": self.run,
            "value": self.value,
            "raw": self.raw,
            "rationale": self.rationale,
            "error": self.error,
            "latency_ms": round(self.latency_ms, 1),
            "cached": self.cached,
        }


class Judge(Protocol):
    name: str

    async def judge(self, case: Case, run: int, cache: Cache) -> tuple[Any, str | None, bool]:
        """Return (raw result, rationale, came_from_cache). Raise on failure."""
        ...

    def describe(self) -> dict: ...


# ----------------------------------------------------------------------------------- demo --


@dataclass
class DemoJudge:
    """A fake judge for demos and tests.

    Gives the human label when the case has one (else a stable pseudo-random verdict), and flips
    it with probability `flakiness` on each run. `bias` is leniency: that share of cases get a
    True verdict every run no matter what (consistent, but wrong). Deterministic for a given
    (name, seed, case, run).
    """

    name: str = "demo"
    flakiness: float = 0.15
    bias: float = 0.0
    seed: int = 0
    kind: str = "bool"  # "bool" or "score" (1-5)
    latency_ms: float = 0.0

    async def judge(self, case: Case, run: int, cache: Cache) -> tuple[Any, str | None, bool]:
        if self.latency_ms:
            await asyncio.sleep(self.latency_ms / 1000)
        rng = random.Random(f"{self.name}:{self.seed}:{case.id}:{run}")
        base_rng = random.Random(f"{self.name}:{self.seed}:{case.id}")
        if self.kind == "score":
            truth = case.label if isinstance(case.label, (int, float)) and not isinstance(case.label, bool) else base_rng.randint(1, 5)
            value = int(truth)
            if rng.random() < self.flakiness:
                value = max(1, min(5, value + rng.choice([-1, 1])))
            return {"score": value, "reasoning": "demo judge"}, "demo judge", False
        truth = case.label if isinstance(case.label, bool) else base_rng.random() < 0.6
        value = bool(truth)
        if self.bias and base_rng.random() < self.bias:
            value = True  # lenient on this case, every run
        elif rng.random() < self.flakiness:
            value = not value
        return {"score": value, "reasoning": "demo judge"}, "demo judge", False

    def describe(self) -> dict:
        return {"kind": "demo", "name": self.name, "flakiness": self.flakiness, "bias": self.bias, "seed": self.seed, "output": self.kind}


# ------------------------------------------------------------------------------- callable --


@dataclass
class CallableJudge:
    """Wraps `module:function` with the evaluator signature fn(inputs=, outputs=, reference_outputs=)."""

    name: str
    fn: Callable[..., Any | Awaitable[Any]]
    spec: str = ""

    @classmethod
    def from_spec(cls, spec: str, name: str | None = None) -> "CallableJudge":
        if ":" not in spec:
            raise ValueError(f"callable judge must look like module:function, got {spec!r}")
        mod_name, _, attr = spec.rpartition(":")
        import sys

        if os.getcwd() not in sys.path:
            sys.path.insert(0, os.getcwd())
        mod = importlib.import_module(mod_name)
        fn = getattr(mod, attr)
        if not callable(fn):
            raise ValueError(f"{spec} is not callable")
        return cls(name=name or attr, fn=fn, spec=spec)

    async def judge(self, case: Case, run: int, cache: Cache) -> tuple[Any, str | None, bool]:
        key = key_for("callable", self.spec or self.name, case.to_dict(), run)
        hit = cache.get(key)
        if hit is not None:
            return hit["raw"], hit.get("rationale"), True
        kwargs = _accepted_kwargs(self.fn, {"inputs": case.inputs, "outputs": case.outputs, "reference_outputs": case.reference_outputs})
        result = self.fn(**kwargs)
        if inspect.isawaitable(result):
            result = await result
        rationale = None
        if isinstance(result, dict):
            rationale = result.get("comment") or result.get("reasoning") or result.get("rationale")
        elif isinstance(result, list) and result and isinstance(result[0], dict):
            rationale = result[0].get("comment")
        raw = _jsonable(result)
        cache.put(key, {"raw": raw, "rationale": rationale})
        return raw, rationale, False

    def describe(self) -> dict:
        return {"kind": "callable", "name": self.name, "spec": self.spec}


def _accepted_kwargs(fn: Callable[..., Any], candidates: dict[str, Any]) -> dict[str, Any]:
    try:
        sig = inspect.signature(fn)
    except (TypeError, ValueError):
        return candidates
    params = sig.parameters
    if any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values()):
        return candidates
    return {k: v for k, v in candidates.items() if k in params}


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value)
        return value
    except TypeError:
        if hasattr(value, "model_dump"):
            return value.model_dump()
        if hasattr(value, "__dict__"):
            return {k: _jsonable(v) for k, v in vars(value).items()}
        return str(value)


# ------------------------------------------------------------------------------------ llm --

ANTHROPIC_URL = "https://api.anthropic.com/v1/messages"
RETRY_STATUSES = {408, 409, 425, 429, 500, 502, 503, 504}

# Every provider except Anthropic speaks the OpenAI chat-completions format, so one adapter
# covers them all: (default base URL, env vars that may hold the key). An empty env tuple means
# no key is needed. `<PROVIDER>_BASE_URL` in the environment overrides the base URL.
PROVIDERS: dict[str, tuple[str, tuple[str, ...]]] = {
    "openai": ("https://api.openai.com/v1", ("OPENAI_API_KEY",)),
    "gemini": ("https://generativelanguage.googleapis.com/v1beta/openai", ("GEMINI_API_KEY", "GOOGLE_API_KEY")),
    "groq": ("https://api.groq.com/openai/v1", ("GROQ_API_KEY",)),
    "mistral": ("https://api.mistral.ai/v1", ("MISTRAL_API_KEY",)),
    "deepseek": ("https://api.deepseek.com/v1", ("DEEPSEEK_API_KEY",)),
    "xai": ("https://api.x.ai/v1", ("XAI_API_KEY",)),
    "together": ("https://api.together.xyz/v1", ("TOGETHER_API_KEY",)),
    "openrouter": ("https://openrouter.ai/api/v1", ("OPENROUTER_API_KEY",)),
    "ollama": ("http://localhost:11434/v1", ()),
    "custom": ("", ("JUDGEFLAKE_API_KEY",)),  # any OpenAI-compatible server: set JUDGEFLAKE_BASE_URL
}
LLM_PROVIDERS = ("anthropic", *PROVIDERS)


@dataclass
class LLMJudge:
    """`<provider>:<model>` with a rubric. Asks for JSON, parses it leniently.

    Providers: anthropic (native messages API) and every OpenAI-compatible one in PROVIDERS.
    """

    provider: str
    model: str
    rubric: str = "correctness"
    name: str = ""
    temperature: float | None = None
    reasoning_effort: str | None = None
    max_tokens: int = 1024
    timeout: float = 60.0
    retries: int = 5
    template: str = field(default="", repr=False)
    _client: httpx.AsyncClient | None = field(default=None, repr=False)

    def __post_init__(self) -> None:
        if self.provider not in LLM_PROVIDERS:
            raise ValueError(f"unknown provider {self.provider!r} (use one of {', '.join(LLM_PROVIDERS)})")
        self.template = self.template or load_rubric(self.rubric)
        self.name = self.name or f"{self.provider}:{self.model}"

    @classmethod
    def from_spec(cls, spec: str, **kw: Any) -> "LLMJudge":
        if ":" not in spec:
            raise ValueError(f"judge must look like <provider>:<model>, e.g. openai:gpt-5-mini, got {spec!r}")
        provider, _, model = spec.partition(":")
        return cls(provider=provider, model=model, **kw)

    def base_url(self) -> str:
        """OpenAI-compatible providers only. `<PROVIDER>_BASE_URL` overrides the default."""
        default, _ = PROVIDERS[self.provider]
        url = os.environ.get(f"{self.provider.upper()}_BASE_URL") or (os.environ.get("JUDGEFLAKE_BASE_URL") if self.provider == "custom" else None) or default
        if not url:
            raise RuntimeError(f"{self.name}: set JUDGEFLAKE_BASE_URL to your OpenAI-compatible server, e.g. http://localhost:8000/v1")
        return url.rstrip("/")

    def describe(self) -> dict:
        return {
            "kind": "llm",
            "name": self.name,
            "provider": self.provider,
            "model": self.model,
            "rubric": self.rubric,
            "temperature": self.temperature,
            "reasoning_effort": self.reasoning_effort,
        }

    def prompt_for(self, case: Case) -> str:
        return render(self.template, inputs=case.inputs, outputs=case.outputs, reference_outputs=case.reference_outputs)

    def api_key(self) -> str | None:
        """The key from the first matching env var, or None when the provider needs none (Ollama)."""
        envs = ("ANTHROPIC_API_KEY",) if self.provider == "anthropic" else PROVIDERS[self.provider][1]
        for env in envs:
            if os.environ.get(env):
                return os.environ[env]
        if not envs:
            return None
        if self.provider == "custom" and len(envs) == 1:
            return None  # a local OpenAI-compatible server often has no auth
        raise RuntimeError(f"{self.name}: set {' or '.join(envs)} in the environment")

    async def judge(self, case: Case, run: int, cache: Cache) -> tuple[Any, str | None, bool]:
        prompt = self.prompt_for(case)
        key = key_for("llm", self.provider, self.model, self.temperature, self.reasoning_effort, SYSTEM, prompt, run)
        hit = cache.get(key)
        if hit is not None:
            return hit["raw"], hit.get("rationale"), True
        text = await self._complete(prompt)
        raw = parse_json_verdict(text)
        rationale = raw.get("reasoning") if isinstance(raw, dict) else None
        cache.put(key, {"raw": raw, "rationale": rationale, "text": text})
        return raw, rationale, False

    async def _complete(self, prompt: str) -> str:
        client = self._client or httpx.AsyncClient(timeout=self.timeout)
        own = self._client is None
        try:
            if self.provider == "anthropic":
                return await self._anthropic(client, prompt)
            return await self._openai(client, prompt)
        finally:
            if own:
                await client.aclose()

    async def _post(self, client: httpx.AsyncClient, url: str, headers: dict, body: dict) -> httpx.Response:
        delay = 1.0
        last: Exception | None = None
        for attempt in range(self.retries):
            try:
                r = await client.post(url, headers=headers, json=body)
            except httpx.HTTPError as e:  # network hiccup
                last = e
            else:
                if r.status_code < 400:
                    return r
                if r.status_code not in RETRY_STATUSES:
                    return r
                last = RuntimeError(f"HTTP {r.status_code}: {r.text[:200]}")
            await asyncio.sleep(delay + random.random() * 0.25)
            delay = min(delay * 2, 16)
        raise RuntimeError(f"{self.name}: gave up after {self.retries} attempts: {last}")

    async def _openai(self, client: httpx.AsyncClient, prompt: str) -> str:
        """OpenAI itself and every OpenAI-compatible provider (Gemini, Groq, Ollama, ...)."""
        url = f"{self.base_url()}/chat/completions"
        key = self.api_key()
        headers = {"Authorization": f"Bearer {key}"} if key else {}
        body: dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
            "response_format": {"type": "json_object"},
        }
        # OpenAI renamed the limit; most compatible servers still only know the old name.
        body["max_completion_tokens" if self.provider == "openai" else "max_tokens"] = self.max_tokens
        if self.temperature is not None:
            body["temperature"] = self.temperature
        if self.reasoning_effort:
            body["reasoning_effort"] = self.reasoning_effort
        r = await self._post(client, url, headers, body)
        # Some models reject some params (temperature, an effort word, json mode). Drop the ones
        # the error names and retry once.
        optional = ("temperature", "reasoning_effort", "response_format")
        if r.status_code == 400 and any(p in body for p in optional):
            msg = r.text
            dropped = [body.pop(p) for p in optional if p in body and p in msg]
            if dropped:
                r = await self._post(client, url, headers, body)
        if r.status_code >= 400:
            raise RuntimeError(f"{self.name}: HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        choice = data["choices"][0]
        if choice.get("finish_reason") == "length":
            raise RuntimeError(f"{self.name}: response cut off (raise --max-tokens)")
        return choice["message"]["content"] or ""

    async def _anthropic(self, client: httpx.AsyncClient, prompt: str) -> str:
        headers = {"x-api-key": self.api_key(), "anthropic-version": "2023-06-01"}
        body: dict[str, Any] = {
            "model": self.model,
            "max_tokens": self.max_tokens,
            "system": SYSTEM,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self.temperature is not None:
            body["temperature"] = self.temperature
        r = await self._post(client, ANTHROPIC_URL, headers, body)
        if r.status_code >= 400:
            raise RuntimeError(f"{self.name}: HTTP {r.status_code}: {r.text[:300]}")
        data = r.json()
        if data.get("stop_reason") == "refusal":
            raise RuntimeError(f"{self.name}: model refused")
        return "".join(block.get("text", "") for block in data.get("content", []) if block.get("type") == "text")


_JSON_RE = re.compile(r"\{.*\}", re.S)


def parse_json_verdict(text: str) -> Any:
    """Find the JSON object in a model reply. Tolerates code fences and chatter around it."""
    s = text.strip()
    if s.startswith("```"):
        s = re.sub(r"^```(?:json)?\s*|\s*```$", "", s, flags=re.S)
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        pass
    m = _JSON_RE.search(s)
    if m:
        try:
            return json.loads(m.group(0))
        except json.JSONDecodeError:
            pass
    raise ValueError(f"no JSON verdict in reply: {text[:120]!r}")


# -------------------------------------------------------------------------------- factory --


def build_judge(spec: str, *, rubric: str = "correctness", temperature: float | None = None, reasoning_effort: str | None = None, max_tokens: int = 1024, name: str | None = None) -> Judge:
    """Turn a CLI spec into a judge.

    - demo, demo:flaky, demo:steady, demo:score   -> DemoJudge presets
    - <provider>:<model> for any LLM_PROVIDERS    -> LLMJudge
    - anything else with a colon                  -> CallableJudge module:function
    """
    if spec == "demo" or spec.startswith("demo:"):
        preset = spec.partition(":")[2] or "default"
        presets = {
            "default": dict(name="demo", flakiness=0.06, seed=1),
            "flaky": dict(name="demo-flaky", flakiness=0.3, seed=2),
            "steady": dict(name="demo-steady", flakiness=0.01, seed=3),
            "lenient": dict(name="demo-lenient", flakiness=0.01, bias=0.35, seed=4),
            "score": dict(name="demo-score", flakiness=0.2, seed=5, kind="score"),
        }
        if preset not in presets:
            raise ValueError(f"unknown demo preset {preset!r}; use {', '.join(presets)}")
        j = DemoJudge(**presets[preset])  # type: ignore[arg-type]
        if name:
            j.name = name
        return j
    provider = spec.partition(":")[0]
    if provider in LLM_PROVIDERS:
        return LLMJudge.from_spec(spec, rubric=rubric, temperature=temperature, reasoning_effort=reasoning_effort, max_tokens=max_tokens, name=name or "")
    return CallableJudge.from_spec(spec, name=name)


def short_hash(text: str) -> str:
    return hashlib.sha1(text.encode()).hexdigest()[:8]
