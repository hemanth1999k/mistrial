"""Run every judge over every case N times, concurrently, and keep every verdict.

Two modes:
- fixed:    every (judge, case) gets exactly `runs` samples.
- adaptive: every (judge, case) gets two samples; only the ones that disagreed (or errored)
            get the rest, up to `runs`. Same self-agreement estimate where it matters, a
            fraction of the calls where it doesn't.
"""

from __future__ import annotations

import asyncio
import datetime as dt
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Sequence

import httpx

from .cache import Cache
from .cases import Case
from .judges import Judge, LLMJudge, Verdict, as_response
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
    adaptive: bool = False
    calls_saved: int = 0  # adaptive mode: calls a fixed run would have made and this one skipped

    def to_dict(self) -> dict:
        return {
            "version": 2,
            "created": self.created,
            "label": self.label,
            "runs": self.runs,
            "rubric": self.rubric,
            "adaptive": self.adaptive,
            "calls_saved": self.calls_saved,
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
            adaptive=d.get("adaptive", False),
            calls_saved=d.get("calls_saved", 0),
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
    adaptive: bool = False,
) -> RunResult:
    if runs < 1:
        raise ValueError("runs must be at least 1")
    cache = cache or Cache()
    sem = asyncio.Semaphore(max(1, concurrency))
    adaptive = adaptive and runs > 2
    first = 2 if adaptive else runs
    total = len(cases) * len(judges) * first
    done = 0
    started = time.perf_counter()
    result = RunResult(
        created=dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        runs=runs,
        cases=list(cases),
        judges=[j.describe() for j in judges],
        rubric=rubric,
        label=label,
        adaptive=adaptive,
    )

    async with httpx.AsyncClient(timeout=120.0) as client:
        lent: list[LLMJudge] = []
        for j in judges:
            if isinstance(j, LLMJudge) and j._client is None:
                j._client = client  # share one connection pool; a judge with its own client keeps it
                lent.append(j)

        async def one(judge: Judge, case: Case, run_ix: int) -> Verdict:
            nonlocal done
            async with sem:
                t0 = time.perf_counter()
                v = Verdict(case_id=case.id, judge=judge.name, run=run_ix)
                try:
                    r = as_response(await judge.judge(case, run_ix, cache))
                    v.raw, v.rationale, v.cached, v.text, v.parse = r.raw, r.rationale, r.cached, r.text, r.parse
                    try:
                        v.value = normalize(r.raw)
                    except ValueError:
                        v.parse = "no-verdict"
                        raise
                except Exception as e:  # keep going; the report shows errors per judge
                    v.error = f"{type(e).__name__}: {e}"[:500]
                    if v.parse is None:
                        v.parse = "error"
                v.latency_ms = (time.perf_counter() - t0) * 1000
                done += 1
                if progress:
                    progress(done, total, v)
                return v

        tasks = [one(j, c, r) for j in judges for c in cases for r in range(first)]
        verdicts = list(await asyncio.gather(*tasks))

        if adaptive:
            # Re-ask only where the first two samples disagreed or something failed.
            seen: dict[tuple[str, str], list[Verdict]] = {}
            for v in verdicts:
                seen.setdefault((v.judge, v.case_id), []).append(v)
            unsettled = [(j, c) for j in judges for c in cases if _unsettled(seen.get((j.name, c.id), []))]
            extra = [one(j, c, r) for j, c in unsettled for r in range(first, runs)]
            total += len(extra)
            result.calls_saved = len(cases) * len(judges) * (runs - first) - len(extra)
            verdicts += list(await asyncio.gather(*extra))

        result.verdicts = verdicts
        for j in lent:
            j._client = None

    result.duration_s = time.perf_counter() - started
    cache.close()
    return result


def _unsettled(vs: list[Verdict]) -> bool:
    if any(v.error for v in vs):
        return True
    values = [v.value for v in vs]
    return len(set(values)) > 1


def by_judge(result: RunResult) -> dict[str, dict[str, list[Any]]]:
    """judge -> case id -> values across runs (errors left out), in run order."""
    out: dict[str, dict[str, list[Any]]] = {j["name"]: {c.id: [] for c in result.cases} for j in result.judges}
    for v in sorted(result.verdicts, key=lambda v: v.run):
        if v.error is None and v.value is not None:
            out.setdefault(v.judge, {}).setdefault(v.case_id, []).append(v.value)
    return out
