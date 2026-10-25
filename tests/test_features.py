"""The ten additions: rubric check, gate risk, migrate, sentinel, parser audit, adaptive,
escalate, stabilize, reversal, markdown. Everything offline."""

import asyncio
import csv
import json

import httpx
import pytest
from typer.testing import CliRunner

from judgeflake import stats
from judgeflake.cache import Cache
from judgeflake.cases import Case
from judgeflake.cli import app
from judgeflake.judges import DemoJudge, LLMJudge, StabilizedJudge, stabilize
from judgeflake.probes import push_to_langsmith, rubric_check, sentinel_baseline, sentinel_check
from judgeflake.report import analyze, escalation, to_markdown
from judgeflake.runner import run

runner = CliRunner()
CASES = [Case(f"c{i}", f"q{i}", "a", "a", i % 3 != 0) for i in range(24)]


def _write_cases(tmp_path, n=24):
    p = tmp_path / "cases.jsonl"
    p.write_text("\n".join(json.dumps({"id": f"c{i}", "inputs": f"q{i}", "outputs": "a", "reference_outputs": "a", "label": i % 3 != 0}) for i in range(n)))
    return p


# ------------------------------------------------------------------------------- stats --


def test_gate_risk_false_alarm_rises_with_flakiness_and_falls_with_runs():
    steady = {f"c{i}": [True] * 5 for i in range(20)}
    g = stats.gate_risk("steady", steady, pass_mark=0.9)
    assert g is not None and g.false_alarm == 0.0 and g.expected == 1.0
    wobbly = {f"c{i}": [True, True, True, False, False] if i % 2 else [True] * 5 for i in range(20)}
    g1 = stats.gate_risk("wobbly", wobbly, pass_mark=0.8, runs_in_ci=1, draws=4000)
    g5 = stats.gate_risk("wobbly", wobbly, pass_mark=0.8, runs_in_ci=5, draws=4000)
    assert g1 is not None and g5 is not None
    assert 0.0 < g1.false_alarm and g5.false_alarm < g1.false_alarm
    assert g1.safe_mark <= g1.expected and g1.p5 <= g1.p95
    assert g1.runs_for_5pct is None or g1.runs_for_5pct >= 3


def test_gate_risk_numeric_scores_use_the_mean():
    scores = {f"c{i}": [4, 4, 5] for i in range(10)}
    g = stats.gate_risk("q", scores, pass_mark=4.0, draws=2000)
    assert g is not None and g.expected == pytest.approx(4.33, abs=0.05) and g.false_alarm == 0.0


def test_translate_threshold_bool_and_numeric():
    old = {f"c{i}": i < 8 for i in range(10)}  # 80% pass
    new = {f"c{i}": i < 6 for i in range(10)}  # 60% pass
    t = stats.translate_threshold("old", "new", old, new, pass_mark=0.7)
    assert t.offset == pytest.approx(-0.2) and t.translated_mark == pytest.approx(0.5)
    assert len(t.changed) == 2 and t.percent_agreement == 0.8
    old_n = {f"c{i}": s for i, s in enumerate([5, 5, 4, 4, 3, 3, 2, 2, 1, 1])}
    new_n = {f"c{i}": s for i, s in enumerate([4, 4, 3, 3, 3, 2, 2, 1, 1, 1])}
    t2 = stats.translate_threshold("old", "new", old_n, new_n, pass_mark=3.0, per_case_threshold=4)
    assert t2.offset == pytest.approx(-0.6) and t2.translated_mark == pytest.approx(2.4)
    assert t2.translated_per_case == 3  # 40% of cases scored >= 4 on old; >= 3 on new passes 50%, the nearest share


def test_sensitivity_and_reversal_stats():
    base = {f"c{i}": i % 2 == 0 for i in range(10)}
    v1 = dict(base)
    v1["c0"] = not v1["c0"]
    v2 = dict(base)
    s = stats.sensitivity("j", {"v0": base, "v1": v1, "v2": v2})
    assert s.changed_cases == ["c0"] and s.changed_rate == 0.1 and s.vs_baseline == {"v1": 0.1, "v2": 0.0}
    assert s.alpha is not None and 0.8 < s.alpha <= 1.0
    rev = {c: (not v) for c, v in base.items()}
    rev["c3"] = base["c3"]
    r = stats.reversal("j", base, rev)
    assert r.inverted == 9 and r.unchanged == 1 and r.ignored_cases == ["c3"] and r.inversion_rate == 0.9


# ------------------------------------------------------------------------- stabilize --


def test_stabilized_judge_is_steadier_than_the_raw_one():
    raw = DemoJudge(name="wobbly", flakiness=0.3, seed=7)
    stab = StabilizedJudge(raw, k=5)
    assert stab.name == "wobbly×5"
    result = asyncio.run(run(CASES, [raw, stab], runs=4, cache=Cache(enabled=False)))
    rep = analyze(result)
    by = {j["name"]: j for j in rep["judges"]}
    assert by["wobbly×5"]["self_agreement"] > by["wobbly"]["self_agreement"]
    assert by["wobbly×5"]["describe"]["kind"] == "stabilized" and by["wobbly×5"]["describe"]["k"] == 5


def test_stabilize_api_returns_an_evaluator_with_votes():
    def flaky(inputs, outputs, reference_outputs):
        return {"key": "x", "score": outputs == reference_outputs}

    ev = stabilize(flaky, k=3)
    out = ev(inputs="q", outputs="a", reference_outputs="a")
    assert out["score"] is True and out["judgeflake"]["votes"] == [True, True, True] and out["judgeflake"]["k"] == 3
    ev2 = stabilize("demo:score", k=3)
    out2 = ev2(inputs="q", outputs="a")
    assert isinstance(out2["score"], (int, float)) and len(out2["judgeflake"]["votes"]) == 3


# ---------------------------------------------------------------------------- adaptive --


def test_adaptive_run_skips_settled_cases():
    steady = DemoJudge(name="steady", flakiness=0.0, seed=1)
    wobbly = DemoJudge(name="wobbly", flakiness=0.5, seed=2)
    result = asyncio.run(run(CASES, [steady, wobbly], runs=6, cache=Cache(enabled=False), adaptive=True))
    assert result.adaptive and result.calls_saved > 0
    per_steady = [v for v in result.verdicts if v.judge == "steady"]
    assert len(per_steady) == len(CASES) * 2  # never disagreed, never re-asked
    per_wobbly = [v for v in result.verdicts if v.judge == "wobbly"]
    assert len(per_wobbly) > len(CASES) * 2
    rep = analyze(result)
    assert rep["meta"]["adaptive"] and rep["meta"]["calls_saved"] == result.calls_saved
    assert next(j for j in rep["judges"] if j["name"] == "steady")["self_agreement"] == 1.0


def test_adaptive_with_two_runs_is_just_fixed():
    result = asyncio.run(run(CASES[:4], [DemoJudge(flakiness=0.5)], runs=2, cache=Cache(enabled=False), adaptive=True))
    assert not result.adaptive and len(result.verdicts) == 8


# ------------------------------------------------------------------------ parser audit --


def test_parser_audit_counts_lenient_and_failed_replies(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    replies = iter(['{"score": true}', '```json\n{"score": false}\n```', 'Sure! {"score": true} hope that helps', "no json here", '{"reasoning": "forgot the score"}'])

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": next(replies)}}]})

    j = LLMJudge("openai", "m")
    j._client = httpx.AsyncClient(transport=httpx.MockTransport(handler))
    cases = [Case(f"c{i}", "q", "a", "a") for i in range(5)]
    result = asyncio.run(run(cases, [j], runs=1, cache=Cache(enabled=False)))
    modes = {v.case_id: v.parse for v in result.verdicts}
    assert modes == {"c0": "json", "c1": "fenced", "c2": "embedded", "c3": "error", "c4": "no-verdict"}
    rep = analyze(result)
    p = rep["judges"][0]["parser"]
    assert p["counts"]["json"] == 1 and p["counts"]["fenced"] == 1 and p["counts"]["embedded"] == 1
    assert p["counts"]["error"] == 1 and p["counts"]["no-verdict"] == 1 and p["reading"] == "fragile"
    assert any(ex["parse"] == "no-verdict" for ex in p["examples"])
    assert analyze(asyncio.run(run(cases[:2], [DemoJudge()], runs=1, cache=Cache(enabled=False))))["judges"][0]["parser"] is None


# ------------------------------------------------------------------------- escalation --


def test_escalation_rows_and_csv_and_langsmith(tmp_path):
    result = asyncio.run(run(CASES, [DemoJudge(name="a", flakiness=0.3, seed=1), DemoJudge(name="b", flakiness=0.0, bias=0.4, seed=2)], runs=3, cache=Cache(enabled=False)))
    rep = analyze(result, top=3)
    rows = escalation(rep)
    assert len(rows) > 3  # not capped at the card's top N
    ids = {c.id for c in CASES}
    assert all(r["case_id"] in ids and r["reason"] and r["inputs"] for r in rows)
    assert rows[0]["disagreement"] >= rows[-1]["disagreement"]
    assert any("disagree" in r["reason"] for r in rows) and any("label says" in r["reason"] for r in rows)
    from judgeflake.probes import write_escalation_csv

    out = tmp_path / "h.csv"
    write_escalation_csv(rows, out, ["a", "b"])
    with out.open() as f:
        header = next(csv.reader(f))
    assert header[:3] == ["case_id", "reason", "disagreement"] and "verdicts:a" in header and "your_verdict" in header

    class FakeDS:
        id = "ds-1"

    class FakeClient:
        def __init__(self):
            self.examples = None

        def read_dataset(self, dataset_name):
            raise RuntimeError("missing")

        def create_dataset(self, dataset_name, description):
            return FakeDS()

        def create_examples(self, inputs, outputs, metadata, dataset_id):
            self.examples = (inputs, outputs, metadata, dataset_id)

    fc = FakeClient()
    info = push_to_langsmith(rows, "needs-human", client=fc)
    assert info["examples"] == len(rows) and fc.examples[3] == "ds-1" and fc.examples[2][0]["judgeflake_case_id"] == rows[0]["case_id"]


# ---------------------------------------------------------------------------- markdown --


def test_markdown_card_has_table_and_gate_lines():
    result = asyncio.run(run(CASES, [DemoJudge(name="a", flakiness=0.2, seed=3)], runs=3, cache=Cache(enabled=False)))
    rep = analyze(result, pass_mark=0.6)
    md = to_markdown(rep, "t", ["a flip rate 50% > 10%"])
    assert md.startswith("<!-- judgeflake -->") and "🔴" in md and "| `a` |" in md and "Gate risk at pass mark 0.6" in md and "**Gate failures:**" in md
    assert "🟢" in to_markdown(rep, "t", [])


# ------------------------------------------------------------------------ rubric check --


def test_rubric_check_with_demo_judges():
    out = rubric_check(CASES, ["demo:flaky", "demo:steady"], rubric="correctness", runs=1, cache=Cache(enabled=False))
    assert out["variants"][0].startswith("v0") and len(out["variants"]) == 4
    by = {j["judge"]: j for j in out["judges"]}
    assert by["demo:flaky"]["sensitivity"]["changed_rate"] > by["demo:steady"]["sensitivity"]["changed_rate"]
    assert by["demo:steady"]["sensitivity"]["reading"] == "robust"
    r = by["demo:steady"]["reversal"]
    assert 0.6 <= r["inversion_rate"] < 1.0 and r["unchanged"] == len(r["ignored_cases"])
    assert out["calls"] == len(CASES) * 2 * 5  # 4 wordings + reversed, per judge


def test_rubric_check_llm_uses_each_wording(monkeypatch):
    monkeypatch.setenv("OPENAI_API_KEY", "k")
    prompts = []

    def handler(request: httpx.Request) -> httpx.Response:
        body = json.loads(request.content)
        prompts.append(body["messages"][1]["content"])
        return httpx.Response(200, json={"choices": [{"finish_reason": "stop", "message": {"content": '{"score": true}'}}]})

    import judgeflake.runner as rn

    real = rn.httpx.AsyncClient
    monkeypatch.setattr(rn.httpx, "AsyncClient", lambda **kw: real(transport=httpx.MockTransport(handler)))
    out = rubric_check(CASES[:3], ["openai:m"], rubric="correctness", runs=1, cache=Cache(enabled=False))
    assert len(prompts) == 3 * 5
    assert any("Candidate answer" in p for p in prompts) and any("INCORRECT" in p for p in prompts)
    assert out["judges"][0]["sensitivity"]["changed_rate"] == 0.0 and out["judges"][0]["reversal"]["inversion_rate"] == 0.0


def test_rubric_check_rejects_function_judges():
    with pytest.raises(ValueError, match="no rubric"):
        rubric_check(CASES, ["examples.my_evals:exact"], runs=1, cache=Cache(enabled=False))


# -------------------------------------------------------------------------- sentinel --


def test_sentinel_baseline_and_check_detect_drift():
    result = asyncio.run(run(CASES, [DemoJudge(name="j", flakiness=0.0, seed=1)], runs=2, cache=Cache(enabled=False)))
    rep = analyze(result)
    base = sentinel_baseline({"report": rep, "run": result.to_dict()})
    assert base["cases"] == 24 and "j" in base["judges"] and base["judges"]["j"]["kappa_vs_human"] == 1.0
    same = sentinel_check(base, rep, CASES)
    assert same["verdict"] == "STABLE" and same["judges"][0]["changed"] == []
    drifted = asyncio.run(run(CASES, [DemoJudge(name="j", flakiness=0.0, bias=0.5, seed=9)], runs=2, cache=Cache(enabled=False)))
    out = sentinel_check(base, analyze(drifted), CASES)
    assert out["verdict"] == "JUDGE_DRIFT" and out["judges"][0]["changed_rate"] > 0.1 and out["judges"][0]["reasons"]
    other_cases = CASES[:-1]
    assert sentinel_check(base, rep, other_cases)["verdict"] == "ANCHORS_CHANGED"


# -------------------------------------------------------------------------------- cli --


def test_cli_run_with_adaptive_stabilize_pass_mark_markdown(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = _write_cases(tmp_path)
    r = runner.invoke(app, ["run", str(cases), "-j", "demo:flaky", "-n", "5", "--adaptive", "--stabilize", "3", "--pass-mark", "0.6", "--html", "none", "--markdown", "c.md", "--out-dir", ".jf"])
    assert r.exit_code == 0, r.output
    assert "demo-flaky×3" in r.output and "Gate risk" in r.output and "adaptive" in r.output
    md = (tmp_path / "c.md").read_text()
    assert "<!-- judgeflake -->" in md and "demo-flaky×3" in md


def test_cli_gate_risk_command(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = _write_cases(tmp_path)
    assert runner.invoke(app, ["run", str(cases), "-j", "demo:flaky", "-n", "4", "-q", "--html", "none", "--out-dir", ".jf"]).exit_code == 0
    saved = next((tmp_path / ".jf" / "runs").glob("*.json"))
    ok = runner.invoke(app, ["gate-risk", str(saved), "--pass-mark", "0.3"])
    assert ok.exit_code == 0 and "false alarm" in ok.output
    bad = runner.invoke(app, ["gate-risk", str(saved), "--pass-mark", "0.65", "--max-false-alarm", "0.01"])
    assert bad.exit_code == 1


def test_cli_rubric_check_and_gates(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = _write_cases(tmp_path)
    r = runner.invoke(app, ["rubric-check", str(cases), "-j", "demo:steady", "--out-dir", ".jf", "--json", "rc.json"])
    assert r.exit_code == 0, r.output
    assert "Rubric sensitivity" in r.output and "Rubric reversal" in r.output
    assert json.loads((tmp_path / "rc.json").read_text())["judges"][0]["judge"] == "demo:steady"
    strict = runner.invoke(app, ["rubric-check", str(cases), "-j", "demo:flaky", "--out-dir", ".jf", "-q", "--max-changed", "0.0"])
    assert strict.exit_code == 1


def test_cli_migrate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = _write_cases(tmp_path)
    assert runner.invoke(app, ["run", str(cases), "-j", "demo:steady", "-j", "demo:lenient", "-n", "2", "-q", "--html", "none", "--out-dir", ".jf"]).exit_code == 0
    saved = next((tmp_path / ".jf" / "runs").glob("*.json"))
    r = runner.invoke(app, ["migrate", str(saved), "--pass-mark", "0.7", "--json", "m.json"])
    assert r.exit_code == 0, r.output
    m = json.loads((tmp_path / "m.json").read_text())
    assert m["old_judge"] == "demo-steady" and m["new_judge"] == "demo-lenient" and m["translated_mark"] >= 0.7
    assert "your pass mark 0.7" in r.output


def test_cli_sentinel_and_escalate(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = _write_cases(tmp_path)
    assert runner.invoke(app, ["run", str(cases), "-j", "demo:steady", "-n", "2", "-q", "--html", "none", "--out-dir", ".jf"]).exit_code == 0
    saved = next((tmp_path / ".jf" / "runs").glob("*.json"))
    b = runner.invoke(app, ["sentinel", "baseline", str(saved), "--out", "base.json"])
    assert b.exit_code == 0 and (tmp_path / "base.json").exists()
    ok = runner.invoke(app, ["sentinel", "check", str(cases), "-j", "demo:steady", "--baseline", "base.json", "-n", "2", "--out-dir", ".jf", "-q"])
    assert ok.exit_code == 0, ok.output
    base = json.loads((tmp_path / "base.json").read_text())
    for cid in list(base["judges"]["demo-steady"]["majorities"])[:8]:
        base["judges"]["demo-steady"]["majorities"][cid] = not base["judges"]["demo-steady"]["majorities"][cid]
    (tmp_path / "base.json").write_text(json.dumps(base))
    drift = runner.invoke(app, ["sentinel", "check", str(cases), "-j", "demo:steady", "--baseline", "base.json", "-n", "2", "--out-dir", ".jf"])
    assert drift.exit_code == 2 and "JUDGE_DRIFT" in drift.output
    e = runner.invoke(app, ["escalate", str(saved), "--out", "h.csv"])
    assert e.exit_code == 0 and (tmp_path / "h.csv").exists() and "need a human" in e.output
