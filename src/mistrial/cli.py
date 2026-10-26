"""mistrial command line."""

from __future__ import annotations

import asyncio
import datetime as dt
import json
from importlib import resources
from pathlib import Path
from typing import Optional

import typer
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn

from . import __version__
from .cache import DEFAULT_DIR, Cache
from .cases import load_cases
from .judges import PROVIDERS, StabilizedJudge, build_judge
from .probes import (
    migrate,
    print_migrate,
    print_rubric_check,
    print_sentinel,
    push_to_langsmith,
    rubric_check,
    sentinel_baseline,
    sentinel_check,
    write_escalation_csv,
)
from .prompts import RUBRICS
from .report import analyze, compare, escalation, print_compare, print_report, write_outputs
from .runner import RunResult, run

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Is your LLM judge flaky? Re-run any evaluator across runs and judge models and get a reliability card.",
)
console = Console()
err = Console(stderr=True)

JUDGE_HELP = "Judge spec, repeatable: <provider>:<model> (openai, anthropic, gemini, groq, mistral, deepseek, xai, together, openrouter, ollama, custom; see `mistrial providers`), module:function, demo[:flaky|steady|lenient|score]."


def _version(value: bool) -> None:
    if value:
        console.print(f"mistrial {__version__}")
        raise typer.Exit()


@app.callback()
def main(version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show the version and exit.")) -> None:
    pass


def _html_path(value: str | None) -> Optional[Path]:
    if value is None or value.strip().lower() in ("", "none", "-", "."):
        return None
    return Path(value)


def _run_path(out_dir: Path, prefix: str = "") -> Path:
    """A fresh file under <out_dir>/runs/. Stamps are per second, so two runs in the same
    second get -2, -3, ... instead of silently overwriting each other."""
    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    runs = out_dir / "runs"
    path = runs / f"{prefix}{stamp}.json"
    n = 2
    while path.exists():
        path = runs / f"{prefix}{stamp}-{n}.json"
        n += 1
    return path


def _load_run(path: Path, what: str = "a saved run") -> dict:
    """Read a file written by `mistrial run` (or `sentinel check`) and fail cleanly on anything else."""
    try:
        data = json.loads(path.read_text())
    except FileNotFoundError:
        err.print(f"[red]no such file: {path}[/red]")
        raise typer.Exit(2)
    except (OSError, ValueError) as e:
        err.print(f"[red]could not read {path}: {e}[/red]")
        raise typer.Exit(2)
    if isinstance(data, dict) and "run" in data and "report" in data:
        return data
    if isinstance(data, dict) and "variants" in data:
        err.print(f"[red]{path} is a rubric-check result, not {what}. Point at a file written by `mistrial run` (.mistrial/runs/<stamp>.json).[/red]")
    else:
        err.print(f"[red]{path} is not {what} written by `mistrial run` (expected JSON with 'run' and 'report').[/red]")
    raise typer.Exit(2)


def _build_judges(judge_specs: list[str], rubric: str, temperature, reasoning_effort, max_tokens: int, stabilize: int = 0) -> list:
    try:
        judges = [build_judge(s, rubric=rubric, temperature=temperature, reasoning_effort=reasoning_effort, max_tokens=max_tokens) for s in judge_specs]
    except (ValueError, ImportError, AttributeError) as e:
        err.print(f"[red]judge error:[/red] {e}")
        raise typer.Exit(2)
    if stabilize and stabilize > 1:
        judges = judges + [StabilizedJudge(j, k=stabilize) for j in judges]
    names = [j.name for j in judges]
    if len(set(names)) != len(names):
        err.print("[red]two judges have the same name; give one a different model or preset[/red]")
        raise typer.Exit(2)
    return judges


def _progress(quiet: bool, total: int):
    bar = Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console, disable=quiet)
    task = bar.add_task("judging", total=total)

    def tick(done: int, new_total: int, _v: object) -> None:
        bar.update(task, completed=done, total=new_total)

    return bar, tick


def _gate_failures(report: dict, max_flip_rate, min_self_agreement, min_agreement) -> list[str]:
    failed = []
    if max_flip_rate is not None:
        for j in report["judges"]:
            if j["flip_rate"] > max_flip_rate:
                failed.append(f"{j['name']} flip rate {j['flip_rate']:.0%} > {max_flip_rate:.0%}")
    if min_self_agreement is not None:
        for j in report["judges"]:
            sa = j["self_agreement"]
            if sa is not None and sa < min_self_agreement:
                failed.append(f"{j['name']} self-agreement {sa:.0%} < {min_self_agreement:.0%}")
    if min_agreement is not None and report.get("agreement") and report["agreement"]["alpha_majority"] is not None:
        a = report["agreement"]["alpha_majority"]
        if a < min_agreement:
            failed.append(f"judge agreement alpha {a:.2f} < {min_agreement:.2f}")
    for j in report["judges"]:
        if j["errors"] and j["errors"] == j["cases"] * j["runs"]:
            failed.append(f"{j['name']}: every call failed")
    return failed


def _run_and_report(
    cases_path: Path,
    judge_specs: list[str],
    runs: int,
    rubric: str,
    concurrency: int,
    fresh: bool,
    out_dir: Path,
    html: Optional[Path],
    json_out: Optional[Path],
    max_flip_rate: Optional[float],
    min_agreement: Optional[float],
    min_self_agreement: Optional[float],
    top: int,
    temperature: Optional[float],
    reasoning_effort: Optional[str],
    max_tokens: int,
    label: Optional[str],
    quiet: bool,
    title: str,
    adaptive: bool = False,
    stabilize: int = 0,
    pass_mark: Optional[float] = None,
    runs_in_ci: int = 1,
    markdown: Optional[Path] = None,
) -> int:
    cases = load_cases(cases_path)
    judges = _build_judges(judge_specs, rubric, temperature, reasoning_effort, max_tokens, stabilize)
    cache = Cache(out_dir / "cache.sqlite", enabled=not fresh)
    planned = len(cases) * len(judges) * (2 if adaptive and runs > 2 else runs)
    if not quiet:
        mode = " (adaptive: 2 runs each, more only where they disagree)" if adaptive and runs > 2 else ""
        console.print(f"mistrial · {len(cases)} cases × {len(judges)} judges × {runs} runs{mode}" + (" · fresh, no cache" if fresh else ""))
    bar, tick = _progress(quiet, planned)
    with bar:
        result = asyncio.run(run(cases, judges, runs=runs, concurrency=concurrency, cache=cache, progress=tick, rubric=rubric, label=label, adaptive=adaptive))

    report = analyze(result, top=top, pass_mark=pass_mark, runs_in_ci=runs_in_ci)
    if not quiet:
        print_report(report, console)
    failed = _gate_failures(report, max_flip_rate, min_self_agreement, min_agreement)

    saved = _run_path(out_dir)
    write_outputs(report, result, html_path=html, json_path=saved, title=title, markdown_path=markdown, gate_failures=failed)
    if json_out and json_out != saved:
        write_outputs(report, result, html_path=None, json_path=json_out, title=title)
    if not quiet:
        console.print(f"saved run → {saved}" + (f"\nreport  → {html}" if html else "") + (f"\nmarkdown → {markdown}" if markdown else ""))
    if failed:
        err.print("[red]gate failed:[/red] " + "; ".join(failed))
        return 1
    return 0


@app.command("run")
def run_cmd(
    cases: Path = typer.Argument(..., help="Cases file: .jsonl, .json or .csv with inputs/outputs(/reference_outputs/label)."),
    judge: list[str] = typer.Option(..., "--judge", "-j", help=JUDGE_HELP),
    runs: int = typer.Option(3, "--runs", "-n", min=1, help="How many times to ask each judge about each case."),
    rubric: str = typer.Option("correctness", "--rubric", "-r", help="Built-in rubric name (correctness, quality) or a prompt file for LLM judges."),
    adaptive: bool = typer.Option(False, "--adaptive", help="Ask twice everywhere, then spend the remaining runs only on cases where the two disagreed. Same numbers where they matter, far fewer calls."),
    stabilize: int = typer.Option(0, "--stabilize", help="Also measure each judge wrapped in a K-sample majority vote (the fix), next to the raw judge."),
    pass_mark: Optional[float] = typer.Option(None, "--pass-mark", help="Your eval's pass mark on the aggregate score (0.8 = 80% of cases pass). Adds the gate false-alarm section."),
    runs_in_ci: int = typer.Option(1, "--runs-in-ci", min=1, help="How many runs per case your CI gate uses (majority vote). For --pass-mark."),
    concurrency: int = typer.Option(8, "--concurrency", "-c", min=1),
    fresh: bool = typer.Option(False, "--fresh", help="Ignore the cache and make new calls (use this to measure drift over time)."),
    out_dir: Path = typer.Option(DEFAULT_DIR, "--out-dir", help="Where runs and the cache live."),
    html: str = typer.Option("report.html", "--html", help="Write the HTML report here (pass 'none' to skip)."),
    markdown: Optional[Path] = typer.Option(None, "--markdown", help="Also write the card as markdown (what the GitHub Action posts on a PR)."),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Also write the JSON report here."),
    max_flip_rate: Optional[float] = typer.Option(None, "--max-flip-rate", help="CI gate: exit 1 if any judge flipped on more than this share of cases (0.1 = 10%). Depends on --runs."),
    min_self_agreement: Optional[float] = typer.Option(None, "--min-self-agreement", help="CI gate: exit 1 if the chance two runs of a judge agree is below this (0.95 = 95%). Independent of --runs."),
    min_agreement: Optional[float] = typer.Option(None, "--min-agreement", help="CI gate: exit 1 if Krippendorff's alpha between judges is below this."),
    top: int = typer.Option(10, "--top", help="How many flaky cases to list."),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="LLM judges only. Leave unset to use the model default."),
    reasoning_effort: Optional[str] = typer.Option(None, "--reasoning-effort", help="OpenAI reasoning models only (e.g. low, medium)."),
    max_tokens: int = typer.Option(1024, "--max-tokens"),
    label: Optional[str] = typer.Option(None, "--label", help="A name for this run (shows up in compare)."),
    title: str = typer.Option("mistrial report", "--title"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Run the judges over the cases and print the reliability card."""
    code = _run_and_report(
        cases, judge, runs, rubric, concurrency, fresh, out_dir, _html_path(html), json_out, max_flip_rate, min_agreement, min_self_agreement, top, temperature, reasoning_effort, max_tokens, label, quiet, title,
        adaptive=adaptive, stabilize=stabilize, pass_mark=pass_mark, runs_in_ci=runs_in_ci, markdown=markdown,
    )
    raise typer.Exit(code)


@app.command("demo")
def demo_cmd(
    runs: int = typer.Option(5, "--runs", "-n", min=1),
    html: str = typer.Option("mistrial-demo.html", "--html", help="Write the HTML report here (pass 'none' to skip)."),
    out_dir: Path = typer.Option(DEFAULT_DIR, "--out-dir"),
    stabilize: int = typer.Option(0, "--stabilize", help="Also show each demo judge wrapped in a K-sample majority vote."),
    pass_mark: Optional[float] = typer.Option(None, "--pass-mark", help="Show the gate false-alarm section at this pass mark (try 0.8)."),
    adaptive: bool = typer.Option(False, "--adaptive"),
) -> None:
    """Try mistrial in ten seconds with three fake judges and twenty bundled cases. No API keys."""
    console.print("[dim]Demo: three fake judges built to be flaky, lenient and steady on purpose, so the card has something to show. Nothing here is broken.[/dim]")
    with resources.as_file(resources.files("mistrial").joinpath("data/demo_cases.jsonl")) as p:
        code = _run_and_report(
            Path(p), ["demo", "demo:flaky", "demo:lenient"], runs, "correctness", 8, True, out_dir, _html_path(html), None, None, None, None, 10, None, None, 1024, "demo", False, "mistrial demo",
            adaptive=adaptive, stabilize=stabilize, pass_mark=pass_mark,
        )
    console.print("[dim]These are fake judges. Point `mistrial run` at your own cases with --judge openai:<model>, --judge ollama:<model> or --judge module:function.[/dim]")
    raise typer.Exit(code)


@app.command("compare")
def compare_cmd(
    old: Path = typer.Argument(..., help="An earlier run (.mistrial/runs/<stamp>.json)."),
    new: Path = typer.Argument(..., help="A later run of the same cases and judges."),
    json_out: Optional[Path] = typer.Option(None, "--json"),
) -> None:
    """Drift: which cases did each judge change its mind on between two runs?"""
    o = _load_run(old, "the older run")["report"]
    n = _load_run(new, "the newer run")["report"]
    cmp = compare(o, n)
    print_compare(cmp, console)
    if json_out:
        json_out.write_text(json.dumps(cmp, indent=2, default=str))


@app.command("report")
def report_cmd(
    run_file: Path = typer.Argument(..., help="A saved run (.mistrial/runs/<stamp>.json)."),
    html: str = typer.Option("report.html", "--html", help="Write the HTML report here (pass 'none' to skip)."),
    markdown: Optional[Path] = typer.Option(None, "--markdown"),
    pass_mark: Optional[float] = typer.Option(None, "--pass-mark", help="Add the gate false-alarm section at this pass mark."),
    runs_in_ci: int = typer.Option(1, "--runs-in-ci", min=1),
    top: int = typer.Option(10, "--top"),
) -> None:
    """Re-render the card from a saved run (no API calls)."""
    data = _load_run(run_file)
    result = RunResult.from_dict(data["run"])
    report = analyze(result, top=top, pass_mark=pass_mark, runs_in_ci=runs_in_ci)
    print_report(report, console)
    html_path = _html_path(html)
    write_outputs(report, result, html_path=html_path, json_path=None, markdown_path=markdown)
    if html_path:
        console.print(f"report → {html_path}")
    if markdown:
        console.print(f"markdown → {markdown}")


@app.command("gate-risk")
def gate_risk_cmd(
    run_file: Path = typer.Argument(..., help="A saved run."),
    pass_mark: float = typer.Option(..., "--pass-mark", help="The pass mark your CI gate uses on the aggregate score (0.8 = 80% of cases must pass; for 1-5 scores, the mean)."),
    runs_in_ci: int = typer.Option(1, "--runs-in-ci", min=1, help="Runs per case your gate uses (majority vote)."),
    max_false_alarm: Optional[float] = typer.Option(None, "--max-false-alarm", help="Exit 1 if any judge's gate would go red more often than this with nothing changed (0.05 = 5%)."),
) -> None:
    """How often would your CI gate go red with nothing changed? And what fixes it."""
    data = _load_run(run_file)
    result = RunResult.from_dict(data["run"])
    report = analyze(result, top=0, pass_mark=pass_mark, runs_in_ci=runs_in_ci)
    report["flaky_cases"] = []
    print_report(report, console)
    if max_false_alarm is not None:
        bad = [g for g in report["gate"]["judges"] if g["false_alarm"] > max_false_alarm]
        if bad:
            err.print("[red]gate risk too high:[/red] " + "; ".join(f"{g['judge']} {g['false_alarm']:.0%}" for g in bad))
            raise typer.Exit(1)


@app.command("rubric-check")
def rubric_check_cmd(
    cases: Path = typer.Argument(..., help="Cases file."),
    judge: list[str] = typer.Option(..., "--judge", "-j", help="LLM or demo judges (functions have no rubric to reword)."),
    rubric: str = typer.Option("correctness", "--rubric", "-r", help="Built-in rubric (has rewordings and a reversed form) or a prompt file (bring --variant files)."),
    variant: list[Path] = typer.Option([], "--variant", help="Extra rubric wordings as prompt files, repeatable."),
    reversal: bool = typer.Option(True, "--reversal/--no-reversal", help="Also ask with the criterion flipped and check the verdicts invert."),
    runs: int = typer.Option(1, "--runs", "-n", min=1, help="Runs per wording; the majority is used."),
    max_changed: Optional[float] = typer.Option(None, "--max-changed", help="Exit 1 if rewording changes more than this share of verdicts (0.1 = 10%)."),
    min_inversion: Optional[float] = typer.Option(None, "--min-inversion", help="Exit 1 if fewer than this share of verdicts invert under the reversed rubric (0.9 = 90%)."),
    concurrency: int = typer.Option(8, "--concurrency", "-c", min=1),
    fresh: bool = typer.Option(False, "--fresh"),
    out_dir: Path = typer.Option(DEFAULT_DIR, "--out-dir"),
    json_out: Optional[Path] = typer.Option(None, "--json"),
    temperature: Optional[float] = typer.Option(None, "--temperature"),
    reasoning_effort: Optional[str] = typer.Option(None, "--reasoning-effort"),
    max_tokens: int = typer.Option(1024, "--max-tokens"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Does the verdict depend on the wording of the rubric? Rewordings with the same meaning, plus the criterion flipped."""
    cs = load_cases(cases)
    cache = Cache(out_dir / "cache.sqlite", enabled=not fresh)
    bar, tick = _progress(quiet, 1)
    try:
        with bar:
            out = rubric_check(cs, judge, rubric=rubric, variant_files=variant, reversal=reversal, runs=runs, concurrency=concurrency, cache=cache, progress=tick, temperature=temperature, reasoning_effort=reasoning_effort, max_tokens=max_tokens)
    except ValueError as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(2)
    if not quiet:
        print_rubric_check(out, console)
    saved = _run_path(out_dir, "rubric-check-")
    saved.parent.mkdir(parents=True, exist_ok=True)
    saved.write_text(json.dumps(out, indent=2, default=str))
    if json_out:
        json_out.write_text(json.dumps(out, indent=2, default=str))
    if not quiet:
        console.print(f"saved → {saved}")
    failed = []
    for j in out["judges"]:
        s, r = j.get("sensitivity"), j.get("reversal")
        if max_changed is not None and s and s["changed_rate"] > max_changed:
            failed.append(f"{j['judge']} rewording changed {s['changed_rate']:.0%} of verdicts > {max_changed:.0%}")
        if min_inversion is not None and r and r["inversion_rate"] < min_inversion:
            failed.append(f"{j['judge']} only {r['inversion_rate']:.0%} of verdicts inverted < {min_inversion:.0%}")
    if failed:
        err.print("[red]gate failed:[/red] " + "; ".join(failed))
        raise typer.Exit(1)


@app.command("migrate")
def migrate_cmd(
    old_run: Path = typer.Argument(..., help="A saved run with the old judge (and the new one, if both ran together)."),
    new_run: Optional[Path] = typer.Argument(None, help="A saved run with the new judge on the same cases, if separate."),
    old_judge: Optional[str] = typer.Option(None, "--old-judge", help="Judge name in the old run (default: first judge)."),
    new_judge: Optional[str] = typer.Option(None, "--new-judge", help="Judge name in the new run (default: second judge of the same run, or first of the new run)."),
    pass_mark: Optional[float] = typer.Option(None, "--pass-mark", help="Your pass mark on the aggregate score under the old judge; printed translated for the new one."),
    threshold: Optional[float] = typer.Option(None, "--threshold", help="A per-case score cutoff (1-5 rubrics) under the old judge; translated by quantile."),
    json_out: Optional[Path] = typer.Option(None, "--json"),
) -> None:
    """Switching judge models? See what moved and what your pass mark becomes on the new judge."""
    old = _load_run(old_run, "the run with the old judge")
    new = _load_run(new_run, "the run with the new judge") if new_run else None
    try:
        out = migrate(old, new, old_judge, new_judge, pass_mark, threshold)
    except ValueError as e:
        err.print(f"[red]{e}[/red]")
        raise typer.Exit(2)
    print_migrate(out, console)
    if json_out:
        json_out.write_text(json.dumps(out, indent=2, default=str))


sentinel_app = typer.Typer(help="Catch a judge that changed under you. Freeze a baseline on anchor cases, then check against it on a schedule.", no_args_is_help=True)
app.add_typer(sentinel_app, name="sentinel")


@sentinel_app.command("baseline")
def sentinel_baseline_cmd(
    run_file: Path = typer.Argument(..., help="A saved run on your anchor cases (small, labeled, never changes)."),
    out: Path = typer.Option(Path(".mistrial/baseline.json"), "--out"),
) -> None:
    """Freeze a run as the baseline the sentinel compares against."""
    data = _load_run(run_file, "a run on your anchor cases")
    base = sentinel_baseline(data)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(base, indent=2, default=str))
    console.print(f"baseline frozen → {out} · {base['cases']} anchors · judges: {', '.join(base['judges'])}")


@sentinel_app.command("check")
def sentinel_check_cmd(
    cases: Path = typer.Argument(..., help="The same anchor cases the baseline was frozen on."),
    judge: list[str] = typer.Option(..., "--judge", "-j", help=JUDGE_HELP),
    baseline: Path = typer.Option(Path(".mistrial/baseline.json"), "--baseline"),
    runs: int = typer.Option(3, "--runs", "-n", min=1),
    rubric: str = typer.Option("correctness", "--rubric", "-r"),
    max_changed: float = typer.Option(0.10, "--max-changed", help="JUDGE_DRIFT when more than this share of anchor verdicts changed."),
    max_kappa_drop: float = typer.Option(0.15, "--max-kappa-drop", help="JUDGE_DRIFT when kappa vs human labels fell by more than this."),
    concurrency: int = typer.Option(8, "--concurrency", "-c", min=1),
    out_dir: Path = typer.Option(DEFAULT_DIR, "--out-dir"),
    json_out: Optional[Path] = typer.Option(None, "--json"),
    temperature: Optional[float] = typer.Option(None, "--temperature"),
    reasoning_effort: Optional[str] = typer.Option(None, "--reasoning-effort"),
    max_tokens: int = typer.Option(1024, "--max-tokens"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Re-judge the anchors fresh and compare to the baseline. Exit 0 STABLE, 2 JUDGE_DRIFT, 3 anchors changed."""
    try:
        base = json.loads(baseline.read_text())
    except FileNotFoundError:
        err.print(f"[red]no baseline at {baseline}. Freeze one first: mistrial sentinel baseline <run>.json[/red]")
        raise typer.Exit(2)
    cs = load_cases(cases)
    judges = _build_judges(judge, rubric, temperature, reasoning_effort, max_tokens)
    cache = Cache(out_dir / "cache.sqlite", enabled=False)  # always fresh: the point is to see today's judge
    bar, tick = _progress(quiet, len(cs) * len(judges) * runs)
    with bar:
        result = asyncio.run(run(cs, judges, runs=runs, concurrency=concurrency, cache=cache, progress=tick, rubric=rubric, label="sentinel"))
    report = analyze(result, top=0)
    out = sentinel_check(base, report, cs, max_changed=max_changed, max_kappa_drop=max_kappa_drop)
    if not quiet:
        print_sentinel(out, console)
    saved = _run_path(out_dir, "sentinel-")
    write_outputs(report, result, html_path=None, json_path=saved)
    if json_out:
        json_out.write_text(json.dumps(out, indent=2, default=str))
    raise typer.Exit({"STABLE": 0, "JUDGE_DRIFT": 2, "ANCHORS_CHANGED": 3}[out["verdict"]])


@app.command("escalate")
def escalate_cmd(
    run_file: Path = typer.Argument(..., help="A saved run."),
    out: Path = typer.Option(Path("needs-human.csv"), "--out", help="CSV for a reviewer: one row per case, with a column for their verdict."),
    min_disagreement: float = typer.Option(0.0, "--min-disagreement", help="Only cases with pooled disagreement above this."),
    misses: bool = typer.Option(True, "--misses/--no-misses", help="Include cases where a judge missed the human label."),
    langsmith_dataset: Optional[str] = typer.Option(None, "--langsmith-dataset", help="Also push the cases to this LangSmith dataset (needs `pip install langsmith` and LANGSMITH_API_KEY)."),
) -> None:
    """The cases a human should look at: judges disagreed, flipped, or missed the label."""
    data = _load_run(run_file)
    report = data["report"]
    rows = escalation(report, min_disagreement=min_disagreement, include_misses=misses)
    names = [j["name"] for j in report["judges"]]
    write_escalation_csv(rows, out, names)
    console.print(f"{len(rows)} of {report['meta']['cases']} cases need a human → {out}")
    for r in rows[:10]:
        console.print(f"  {r['case_id']}: {r['reason']}" + (f" ({r['disagreement']:.0%} disagreement)" if r["disagreement"] else ""))
    if len(rows) > 10:
        console.print(f"  … and {len(rows) - 10} more")
    if langsmith_dataset:
        try:
            info = push_to_langsmith(rows, langsmith_dataset)
        except Exception as e:
            err.print(f"[red]LangSmith push failed:[/red] {e}")
            raise typer.Exit(2)
        console.print(f"pushed {info['examples']} examples to LangSmith dataset {info['dataset']} ({info['id']})")


@app.command("rubrics")
def rubrics_cmd() -> None:
    """List the built-in rubrics."""
    from .prompts import RUBRIC_VARIANTS, REVERSED_RUBRICS

    for name, r in RUBRICS.items():
        extras = f"  ({len(RUBRIC_VARIANTS.get(name, []))} rewordings{', reversed form' if name in REVERSED_RUBRICS else ''})"
        console.print(f"[bold]{name}[/bold]  {r['description']}" + ("  (needs reference_outputs)" if r["needs_reference"] == "true" else "") + extras)


@app.command("providers")
def providers_cmd() -> None:
    """List the LLM providers, the env var that holds each key, and the base URL."""
    console.print("[bold]anthropic[/bold]  ANTHROPIC_API_KEY  (native messages API)")
    for name, (url, envs) in PROVIDERS.items():
        key = " or ".join(envs) if envs else "no key"
        where = url or "set MISTRIAL_BASE_URL"
        console.print(f"[bold]{name}[/bold]  {key}  {where}")
    console.print("[dim]<PROVIDER>_BASE_URL overrides a base URL. All of these use --judge <provider>:<model>.[/dim]")


if __name__ == "__main__":
    app()
