"""judgeflake command line."""

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
from .judges import PROVIDERS, build_judge
from .prompts import RUBRICS
from .report import analyze, compare, print_compare, print_report, write_outputs
from .runner import RunResult, run

app = typer.Typer(
    add_completion=False,
    no_args_is_help=True,
    help="Is your LLM judge flaky? Re-run any evaluator across runs and judge models and get a reliability card.",
)
console = Console()
err = Console(stderr=True)


def _version(value: bool) -> None:
    if value:
        console.print(f"judgeflake {__version__}")
        raise typer.Exit()


@app.callback()
def main(version: bool = typer.Option(False, "--version", callback=_version, is_eager=True, help="Show the version and exit.")) -> None:
    pass


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
) -> int:
    cases = load_cases(cases_path)
    try:
        judges = [build_judge(s, rubric=rubric, temperature=temperature, reasoning_effort=reasoning_effort, max_tokens=max_tokens) for s in judge_specs]
    except (ValueError, ImportError, AttributeError) as e:
        err.print(f"[red]judge error:[/red] {e}")
        raise typer.Exit(2)
    names = [j.name for j in judges]
    if len(set(names)) != len(names):
        err.print("[red]two judges have the same name; give one a different model or preset[/red]")
        raise typer.Exit(2)

    cache = Cache(out_dir / "cache.sqlite", enabled=not fresh)
    total = len(cases) * len(judges) * runs
    if not quiet:
        console.print(f"judgeflake · {len(cases)} cases × {len(judges)} judges × {runs} runs = {total} verdicts" + (" (fresh, no cache)" if fresh else ""))

    with Progress(TextColumn("[progress.description]{task.description}"), BarColumn(), MofNCompleteColumn(), TimeElapsedColumn(), console=console, disable=quiet) as bar:
        task = bar.add_task("judging", total=total)

        def tick(done: int, _total: int, _v: object) -> None:
            bar.update(task, completed=done)

        result = asyncio.run(run(cases, judges, runs=runs, concurrency=concurrency, cache=cache, progress=tick, rubric=rubric, label=label))

    report = analyze(result, top=top)
    if not quiet:
        print_report(report, console)

    stamp = dt.datetime.now().strftime("%Y%m%d-%H%M%S")
    saved = out_dir / "runs" / f"{stamp}.json"
    write_outputs(report, result, html_path=html, json_path=saved, title=title)
    if json_out and json_out != saved:
        write_outputs(report, result, html_path=None, json_path=json_out, title=title)
    if not quiet:
        console.print(f"saved run → {saved}" + (f"\nreport  → {html}" if html else ""))

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
    errored = [j for j in report["judges"] if j["errors"] and j["errors"] == j["cases"] * j["runs"]]
    for j in errored:
        failed.append(f"{j['name']}: every call failed")
    if failed:
        err.print("[red]gate failed:[/red] " + "; ".join(failed))
        return 1
    return 0


@app.command("run")
def run_cmd(
    cases: Path = typer.Argument(..., help="Cases file: .jsonl, .json or .csv with inputs/outputs(/reference_outputs/label)."),
    judge: list[str] = typer.Option(..., "--judge", "-j", help="Judge spec, repeatable: <provider>:<model> (openai, anthropic, gemini, groq, mistral, deepseek, xai, together, openrouter, ollama, custom; see `judgeflake providers`), module:function, demo[:flaky|steady|lenient|score]."),
    runs: int = typer.Option(3, "--runs", "-n", min=1, help="How many times to ask each judge about each case."),
    rubric: str = typer.Option("correctness", "--rubric", "-r", help="Built-in rubric name (correctness, quality) or a prompt file for LLM judges."),
    concurrency: int = typer.Option(8, "--concurrency", "-c", min=1),
    fresh: bool = typer.Option(False, "--fresh", help="Ignore the cache and make new calls (use this to measure drift over time)."),
    out_dir: Path = typer.Option(DEFAULT_DIR, "--out-dir", help="Where runs and the cache live."),
    html: str = typer.Option("report.html", "--html", help="Write the HTML report here (pass 'none' to skip)."),
    json_out: Optional[Path] = typer.Option(None, "--json", help="Also write the JSON report here."),
    max_flip_rate: Optional[float] = typer.Option(None, "--max-flip-rate", help="CI gate: exit 1 if any judge flipped on more than this share of cases (0.1 = 10%). Depends on --runs."),
    min_self_agreement: Optional[float] = typer.Option(None, "--min-self-agreement", help="CI gate: exit 1 if the chance two runs of a judge agree is below this (0.95 = 95%). Independent of --runs."),
    min_agreement: Optional[float] = typer.Option(None, "--min-agreement", help="CI gate: exit 1 if Krippendorff's alpha between judges is below this."),
    top: int = typer.Option(10, "--top", help="How many flaky cases to list."),
    temperature: Optional[float] = typer.Option(None, "--temperature", help="LLM judges only. Leave unset to use the model default."),
    reasoning_effort: Optional[str] = typer.Option(None, "--reasoning-effort", help="OpenAI reasoning models only (e.g. low, medium)."),
    max_tokens: int = typer.Option(1024, "--max-tokens"),
    label: Optional[str] = typer.Option(None, "--label", help="A name for this run (shows up in compare)."),
    title: str = typer.Option("judgeflake report", "--title"),
    quiet: bool = typer.Option(False, "--quiet", "-q"),
) -> None:
    """Run the judges over the cases and print the reliability card."""
    html_path = _html_path(html)
    code = _run_and_report(cases, judge, runs, rubric, concurrency, fresh, out_dir, html_path, json_out, max_flip_rate, min_agreement, min_self_agreement, top, temperature, reasoning_effort, max_tokens, label, quiet, title)
    raise typer.Exit(code)


def _html_path(value: str | None) -> Optional[Path]:
    if value is None or value.strip().lower() in ("", "none", "-", "."):
        return None
    return Path(value)


@app.command("demo")
def demo_cmd(
    runs: int = typer.Option(5, "--runs", "-n", min=1),
    html: str = typer.Option("judgeflake-demo.html", "--html", help="Write the HTML report here (pass 'none' to skip)."),
    out_dir: Path = typer.Option(DEFAULT_DIR, "--out-dir"),
) -> None:
    """Try judgeflake in ten seconds with three fake judges and twenty bundled cases. No API keys."""
    with resources.as_file(resources.files("judgeflake").joinpath("data/demo_cases.jsonl")) as p:
        code = _run_and_report(
            Path(p), ["demo", "demo:flaky", "demo:lenient"], runs, "correctness", 8, True, out_dir, _html_path(html), None, None, None, None, 10, None, None, 1024, "demo", False, "judgeflake demo"
        )
    console.print("[dim]These are fake judges. Point `judgeflake run` at your own cases with --judge openai:<model>, --judge ollama:<model> or --judge module:function.[/dim]")
    raise typer.Exit(code)


@app.command("compare")
def compare_cmd(
    old: Path = typer.Argument(..., help="An earlier run (.judgeflake/runs/<stamp>.json)."),
    new: Path = typer.Argument(..., help="A later run of the same cases and judges."),
    json_out: Optional[Path] = typer.Option(None, "--json"),
) -> None:
    """Drift: which cases did each judge change its mind on between two runs?"""
    o = json.loads(old.read_text())["report"]
    n = json.loads(new.read_text())["report"]
    cmp = compare(o, n)
    print_compare(cmp, console)
    if json_out:
        json_out.write_text(json.dumps(cmp, indent=2, default=str))


@app.command("report")
def report_cmd(
    run_file: Path = typer.Argument(..., help="A saved run (.judgeflake/runs/<stamp>.json)."),
    html: Optional[Path] = typer.Option(Path("report.html"), "--html"),
    top: int = typer.Option(10, "--top"),
) -> None:
    """Re-render the card from a saved run (no API calls)."""
    data = json.loads(run_file.read_text())
    result = RunResult.from_dict(data["run"])
    report = analyze(result, top=top)
    print_report(report, console)
    if html:
        write_outputs(report, result, html_path=html, json_path=None)
        console.print(f"report → {html}")


@app.command("rubrics")
def rubrics_cmd() -> None:
    """List the built-in rubrics."""
    for name, r in RUBRICS.items():
        console.print(f"[bold]{name}[/bold]  {r['description']}" + ("  (needs reference_outputs)" if r["needs_reference"] == "true" else ""))


@app.command("providers")
def providers_cmd() -> None:
    """List the LLM providers, the env var that holds each key, and the base URL."""
    console.print("[bold]anthropic[/bold]  ANTHROPIC_API_KEY  (native messages API)")
    for name, (url, envs) in PROVIDERS.items():
        key = " or ".join(envs) if envs else "no key"
        where = url or "set JUDGEFLAKE_BASE_URL"
        console.print(f"[bold]{name}[/bold]  {key}  {where}")
    console.print("[dim]<PROVIDER>_BASE_URL overrides a base URL. All of these use --judge <provider>:<model>.[/dim]")


if __name__ == "__main__":
    app()
