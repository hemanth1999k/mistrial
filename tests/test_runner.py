import asyncio

import pytest

from mistrial.cache import Cache
from mistrial.cases import Case
from mistrial.judges import CallableJudge, DemoJudge, LLMJudge, build_judge, parse_json_verdict
from mistrial.report import analyze, compare, headline, to_html
from mistrial.runner import by_judge, run

CASES = [
    Case("a", "q1", "yes", "yes", True),
    Case("b", "q2", "no", "yes", False),
    Case("c", "q3", "maybe", "yes", False),
]


def test_demo_judge_is_deterministic():
    j = DemoJudge(name="d", flakiness=0.5, seed=7)
    c = Cache(enabled=False)
    r1 = asyncio.run(j.judge(CASES[0], 0, c))
    r2 = asyncio.run(j.judge(CASES[0], 0, c))
    assert r1 == r2
    steady = DemoJudge(name="s", flakiness=0.0)
    for case in CASES:
        raw, _, _ = asyncio.run(steady.judge(case, 3, c))
        assert raw["score"] is case.label


def test_callable_judge_filters_kwargs_and_caches(tmp_path):
    calls = []

    def fn(outputs, reference_outputs):
        calls.append(outputs)
        return {"key": "exact", "score": outputs == reference_outputs, "comment": "exact match"}

    j = CallableJudge(name="exact", fn=fn, spec="t:fn")
    cache = Cache(tmp_path / "c.sqlite")
    raw, rationale, cached = asyncio.run(j.judge(CASES[0], 0, cache))
    assert raw["score"] is True and rationale == "exact match" and cached is False
    raw2, _, cached2 = asyncio.run(j.judge(CASES[0], 0, cache))
    assert cached2 is True and raw2 == raw
    assert calls == ["yes"]
    cache.close()


def test_callable_judge_async_and_bare_values():
    async def fn(inputs, outputs, reference_outputs):
        return 0.5

    j = CallableJudge(name="half", fn=fn)
    raw, rationale, _ = asyncio.run(j.judge(CASES[0], 0, Cache(enabled=False)))
    assert raw == 0.5 and rationale is None


def test_parse_json_verdict_tolerates_fences_and_chatter():
    assert parse_json_verdict('{"score": true, "reasoning": "ok"}')["score"] is True
    assert parse_json_verdict('```json\n{"score": 4}\n```')["score"] == 4
    assert parse_json_verdict('Sure! Here you go: {"score": false} hope that helps')["score"] is False
    with pytest.raises(ValueError):
        parse_json_verdict("no json here")


def test_build_judge_specs():
    assert build_judge("demo").name == "demo"
    assert build_judge("demo:flaky").flakiness > build_judge("demo:steady").flakiness
    llm = build_judge("openai:gpt-5-mini", rubric="quality")
    assert isinstance(llm, LLMJudge) and llm.name == "openai:gpt-5-mini" and "1 to 5" in llm.template
    assert build_judge("anthropic:claude-sonnet-5-5").provider == "anthropic"
    with pytest.raises(ValueError):
        build_judge("demo:nope")
    with pytest.raises(ValueError):
        build_judge("nocolon")


def test_llm_judge_needs_a_key(monkeypatch):
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    j = build_judge("openai:gpt-5-mini")
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        asyncio.run(j.judge(CASES[0], 0, Cache(enabled=False)))


def test_run_collects_every_verdict_and_errors_do_not_stop_it():
    def boom(outputs):
        if outputs == "maybe":
            raise RuntimeError("nope")
        return True

    judges = [DemoJudge(name="d", flakiness=0.0), CallableJudge(name="boom", fn=boom)]
    seen = []
    result = asyncio.run(run(CASES, judges, runs=2, cache=Cache(enabled=False), progress=lambda d, t, v: seen.append((d, t))))
    assert len(result.verdicts) == 3 * 2 * 2
    assert seen[-1] == (12, 12)
    errs = [v for v in result.verdicts if v.error]
    assert len(errs) == 2 and all(v.judge == "boom" and v.case_id == "c" for v in errs)
    vals = by_judge(result)
    assert vals["d"]["a"] == [True, True]
    assert vals["boom"]["c"] == []  # errors are left out of the stats
    report = analyze(result)
    boom = next(j for j in report["judges"] if j["name"] == "boom")
    assert boom["errors"] == 2 and boom["cases"] == 2
    d = next(j for j in report["judges"] if j["name"] == "d")
    assert d["flip_rate"] == 0 and d["accuracy"]["accuracy"] == 1.0
    assert report["agreement"]["pairs"][0]["cases"] == 2
    assert any("stable" in line for line in headline(report))
    html = to_html(report)
    assert "<title>mistrial report</title>" in html and "boom" in html


def test_round_trip_and_compare():
    flaky = DemoJudge(name="d", flakiness=0.6, seed=1)
    r1 = asyncio.run(run(CASES, [flaky], runs=3, cache=Cache(enabled=False)))
    from mistrial.runner import RunResult

    back = RunResult.from_dict(r1.to_dict())
    assert [v.value for v in back.verdicts] == [v.value for v in r1.verdicts]
    rep1 = analyze(r1)
    steady = DemoJudge(name="d", flakiness=0.0, seed=1)
    rep2 = analyze(asyncio.run(run(CASES, [steady], runs=3, cache=Cache(enabled=False))))
    cmp = compare(rep1, rep2)
    assert cmp["judges"][0]["name"] == "d" and cmp["judges"][0]["cases"] == 3
    assert cmp["judges"][0]["flip_rate_new"] == 0
