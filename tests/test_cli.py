import json
import os

from typer.testing import CliRunner

from mistrial.cli import app

runner = CliRunner()


def test_demo_writes_html_and_run(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    r = runner.invoke(app, ["demo", "--runs", "3", "--out-dir", str(tmp_path / ".jf")])
    assert r.exit_code == 0, r.output
    assert "Consistency" in r.output and "demo-flaky" in r.output
    assert (tmp_path / "mistrial-demo.html").exists()
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
    monkeypatch.chdir(tmp_path)
    skip = runner.invoke(app, ["report", str(tmp_path / "a.json"), "--html", "none", "--markdown", str(tmp_path / "a.md")])
    assert skip.exit_code == 0 and (tmp_path / "a.md").exists()
    assert not (tmp_path / "none").exists(), "report --html none must skip the HTML, not write a file called none"


def test_rubrics_and_bad_judge():
    assert "correctness" in runner.invoke(app, ["rubrics"]).output
    r = runner.invoke(app, ["run", "nope.jsonl", "-j", "whatever:nomodule"])
    assert r.exit_code != 0


def test_version_matches_pyproject():
    import re, pathlib
    import mistrial
    pyproject = pathlib.Path(__file__).resolve().parents[1] / "pyproject.toml"
    declared = re.search(r'^version = "([^"]+)"', pyproject.read_text(), re.M).group(1)
    assert mistrial.__version__ == declared
    out = runner.invoke(app, ["--version"])
    assert out.exit_code == 0 and declared in out.output


def test_same_second_runs_do_not_overwrite(tmp_path, monkeypatch):
    from mistrial import cli

    class Frozen(cli.dt.datetime):
        @classmethod
        def now(cls, tz=None):
            return cls(2026, 10, 26, 4, 23, 19)

    monkeypatch.setattr(cli.dt, "datetime", Frozen)
    out = tmp_path / ".mt"
    paths = []
    for _ in range(3):
        p = cli._run_path(out)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text("{}")
        paths.append(p.name)
    assert paths == ["20261026-042319.json", "20261026-042319-2.json", "20261026-042319-3.json"]
    assert cli._run_path(out, "sentinel-").name == "sentinel-20261026-042319.json"


def test_probes_reject_wrong_file_kind(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    cases = tmp_path / "c.jsonl"
    cases.write_text(json.dumps({"id": "a", "inputs": "2+2", "outputs": "4", "reference_outputs": "4", "label": True}) + "\n")
    out = tmp_path / ".mt"
    rc = runner.invoke(app, ["rubric-check", str(cases), "-j", "demo:flaky", "-q", "--out-dir", str(out)])
    assert rc.exit_code == 0
    rubric_file = next((out / "runs").glob("rubric-check-*.json"))
    for args in (["migrate", str(rubric_file), "--pass-mark", "0.5"], ["gate-risk", str(rubric_file), "--pass-mark", "0.5"],
                 ["escalate", str(rubric_file), "--out", str(tmp_path / "h.csv")], ["report", str(rubric_file), "--html", "none"],
                 ["sentinel", "baseline", str(rubric_file), "--out", str(tmp_path / "b.json")]):
        r = runner.invoke(app, args)
        assert r.exit_code == 2, args
        assert "rubric-check result" in (r.output + str(r.exception or "")), args
        assert "Traceback" not in r.output, args
    missing = runner.invoke(app, ["report", str(tmp_path / "nope.json"), "--html", "none"])
    assert missing.exit_code == 2 and "no such file" in missing.output
