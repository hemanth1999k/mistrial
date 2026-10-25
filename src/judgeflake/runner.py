"""Run every judge over every case N times, concurrently, and keep every verdict."""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import httpx

from .cache import Cache
from .cases import Case
from .judges import Judge, LLMJudge, Verdict
from .stats import normalize

Progress = Callable[[int, int, Verdict], None]


@dataclass
class RunResult:
    created: str
    runs: int
    cases: list[Case]
    judges: list[dict]
    verdicts: list[Verdict] = field(default_factory=list)
    duration_s: float = 0.0
    rubric: str | None = None
    label: str | None = None

    def to_dict(self) -> dict:
        return {
            "version": 1,
            "created": self.created,
            "label": self.label,
            "runs": self.runs,
            "rubric": self.rubric,
            "duration_s": round(self.duration_s, 2),
            "judges": self.judges,
            "cases": [c.to_dict() for c in self.cases],
            "verdicts": [v.to_dict() for v in self.verdicts],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "RunResult":
        from .cases import case_from_dict

        return cls(
            created=d["created"],
            runs=d["runs"],
            cases=[case_from_dict(c, i) for i, c in enumerate(d["cases"])],
            judges=d["judges"],
            verdicts=[Verdict(**v) for v in d["verdicts"]],
            duration_s=d.get("duration_s", 0.0),
            rubric=d.get("rubric"),
            label=d.get("label"),
        )


async def run(
    cases: Sequence[Case],
    judges: Sequence[Judge],
    runs: int = 3,
    concurrency: int = 8,
    cache: Cache | None = None,
    progress: Progress | None = None,
    rubric: str | None = None,
    label: str | None = None,
) -> RunResult:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    cache = cache or Cache()
    sem = asyncio.Semaphore(max(1, concurrency))
    total = len(cases) * len(judges) * runs
    done = 0
    started = time.perf_counter()
    result = RunResult(
        created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        runs=runs,
        cases=list(cases),
        judges=[j.describe() for j in judges],
        rubric=rubric,
        label=label,
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        for j in judges:
            if isinstance(j, LLMJudge):
                j._client = client

        async def one(judge: Judge, case: Case, run: int) -> Verdict:
            nonlocal done
            async with sem:
                t0 = time.perf_counter()
                v = Verdict(case_id=case.id, judge=judge.name, run=run)
                try:
                    raw, rationale, cached = await judge.judge(case, run, cache)
                    v.raw, v.rationale, v.cached = raw, rationale, cached
                    v.value = normalize(raw)
                except Exception as e:  # keep going; the report shows errors per judge
                    v.error = f"{type(e).__name__}: {e}"[:500]
                v.latency_ms = (time.perf_counter() - t0) * 1000
                done += 1
                if progress:
                    progress(done, total, v)
                return v

        tasks = [one(j, c, r) for j in judges for c in cases for r in range(runs)]
        result.verdicts = list(await asyncio.gather(*tasks))
        for j in judges:
            if isinstance(j, LLMJudge):
                j._client = None

    result.duration_s = time.perf_counter() - started
    cache.close()
    return result


def by_judge(result: RunResult) -> dict[str, dict[str, list[Any]]]:
    """judge -> case id -> values across runs (errors left out), in run order."""
    out: dict[str, dict[str, list[Any]]] = {j["name"]: {c.id: [] for c in result.cases} for j in result.judges}
    for v in sorted(result.verdicts, key=lambda v: v.run):
        if v.error is None and v.value is not None:
            out.setdefault(v.judge, {}).setdefault(v.case_id, []).append(v.value)
    return out
