import json
import os

from typer.testing import CliRunner

from judgeflake.cli import app

runner = CliRunner()


def test_demo_writes_html_and_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["demo", "--runs", "3", "--out-dir", str(tmp_path / ".jf")])
    assert r.exit_code == 0, r.output
    assert "Consistency" in r.output and "demo-flaky" in r.output
    assert (tmp_path / "judgeflake-demo.html").exists()
    runs = list((tmp_path / ".jf" / "runs").glob("*.json"))
    assert len(runs) == 1
    data = json.loads(runs[0].read_text())
    assert data["report"]["meta"]["runs"] == 3 and len(data["run"]["verdicts"]) == 20 * 3 * 3


def test_gate_fails_on_flaky_judge(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = tmp_path / "c.jsonl"
    cases.write_text("\n".join(json.dumps({"id": f"c{i}", "outputs": "x", "label": i % 2 == 0}) for i in range(12)))
    r = runner.invoke(app, ["run", str(cases), "-j", "demo:flaky", "-n", "4", "--max-flip-rate", "0.05", "--html", "none", "-q", "--out-dir", str(tmp_path / ".jf")])
    assert r.exit_code == 1
    assert "gate failed" in (r.output + (r.stderr if hasattr(r, "stderr") else ""))
    ok = runner.invoke(app, ["run", str(cases), "-j", "demo:steady", "-n", "2", "--max-flip-rate", "0.5", "--html", "none", "-q", "--out-dir", str(tmp_path / ".jf")])
    assert ok.exit_code == 0, ok.output


def test_compare_and_report_commands(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = tmp_path / "c.jsonl"
    cases.write_text("\n".join(json.dumps({"id": f"c{i}", "outputs": "x", "label": True}) for i in range(8)))
    out = tmp_path / ".jf"
    a = runner.invoke(app, ["run", str(cases), "-j", "demo:flaky", "-n", "3", "-q", "--html", "none", "--out-dir", str(out), "--json", str(tmp_path / "a.json")])
    assert a.exit_code == 0, a.output
    b = runner.invoke(app, ["run", str(cases), "-j", "demo:flaky", "-n", "3", "-q", "--html", "none", "--out-dir", str(out), "--json", str(tmp_path / "b.json"), "--fresh"])
    assert b.exit_code == 0, b.output
    c = runner.invoke(app, ["compare", str(tmp_path / "a.json"), str(tmp_path / "b.json")])
    assert c.exit_code == 0 and "demo-flaky" in c.output
    rep = runner.invoke(app, ["report", str(tmp_path / "a.json"), "--html", str(tmp_path / "again.html")])
    assert rep.exit_code == 0 and (tmp_path / "again.html").exists()


def test_rubrics_and_bad_judge():
    assert "correctness" in runner.invoke(app, ["rubrics"]).output
    r = runner.invoke(app, ["run", "nope.jsonl", "-j", "whatever:nomodule"])
    assert r.exit_code != 0
