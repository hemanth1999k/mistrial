"""The probes: rubric sensitivity and reversal, judge migration, the sentinel, and escalation.

Each one is a function that returns plain data plus a printer. The CLI is a thin wrapper.
"""

from __future__ import annotations

import asyncio
import csv
import datetime as dt
import hashlib
import json
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Sequence

from rich.console import Console
from rich.table import Table

from . import stats
from .cache import Cache
from .cases import Case
from .judges import LLM_PROVIDERS, Judge, build_judge
from .prompts import load_rubric, reversed_rubric, rubric_variants
from .report import analyze, escalation
from .runner import RunResult, by_judge, run

Progress = Callable[[int, int, Any], None]


# ----------------------------------------------------------------------------- rubric check --


def _is_llm(spec: str) -> bool:
    return spec.partition(":")[0] in LLM_PROVIDERS


def _is_demo(spec: str) -> bool:
    return spec == "demo" or spec.startswith("demo:")


def rubric_check(
    cases: Sequence[Case],
    judge_specs: Sequence[str],
    rubric: str = "correctness",
    variant_files: Sequence[Path] = (),
    reversal: bool = True,
    runs: int = 1,
    concurrency: int = 8,
    cache: Cache | None = None,
    progress: Progress | None = None,
    **judge_kw: Any,
) -> dict:
    """Run each judge under reworded rubrics (and a reversed one) and measure what moved.

    Variants: the built-in rewordings of a built-in rubric, plus any prompt files given. The
    first variant is always the rubric itself. Demo judges simulate rewording with a salt.
    Function judges have no rubric, so they are skipped with a note.
    """
    base_template = load_rubric(rubric)
    variants: list[tuple[str, str]] = [("v0 (as used)", base_template)]
    for i, t in enumerate(rubric_variants(rubric), 1):
        variants.append((f"v{i}", t))
    for p in variant_files:
        variants.append((p.name, Path(p).read_text()))
    rev_template = reversed_rubric(rubric)
    if len(variants) < 2 and not reversal:
        raise ValueError("no rubric variants: use a built-in rubric or pass --variant files")

    judges: list[Judge] = []
    skipped: list[str] = []
    plan: dict[str, dict[str, str]] = {}  # spec -> {variant name -> judge name}
    for spec in judge_specs:
        if not (_is_llm(spec) or _is_demo(spec)):
            skipped.append(spec)
            continue
        plan[spec] = {}
        for vname, template in variants:
            jname = f"{spec}@{vname.split(' ')[0]}"
            if _is_demo(spec):
                j = build_judge(spec, name=jname, salt="" if vname.startswith("v0") else vname, **judge_kw)
            else:
                j = build_judge(spec, rubric=rubric, name=jname, template=template, **judge_kw)
            judges.append(j)
            plan[spec][vname] = jname
        if reversal and (rev_template or _is_demo(spec)):
            jname = f"{spec}@reversed"
            if _is_demo(spec):
                j = build_judge(spec, name=jname, invert=True, **judge_kw)
            else:
                j = build_judge(spec, rubric=rubric, name=jname, template=rev_template, **judge_kw)
            judges.append(j)
            plan[spec]["reversed"] = jname

    if not judges:
        raise ValueError("rubric-check needs LLM or demo judges (functions have no rubric): " + ", ".join(skipped))

    result = asyncio.run(run(cases, judges, runs=runs, concurrency=concurrency, cache=cache, progress=progress, rubric=rubric, label="rubric-check"))
    values = by_judge(result)
    majorities = {name: {cid: stats.majority(vs)[0] for cid, vs in per.items() if vs} for name, per in values.items()}
    errors = {name: sum(1 for v in result.verdicts if v.judge == name and v.error) for name in values}

    out: dict[str, Any] = {"rubric": rubric, "variants": [v for v, _ in variants], "reversal": reversal, "runs": runs, "judges": [], "skipped": skipped, "created": result.created, "calls": len(result.verdicts)}
    for spec, names in plan.items():
        entry: dict[str, Any] = {"judge": spec, "errors": sum(errors.get(n, 0) for n in names.values())}
        by_variant = {v: majorities.get(n, {}) for v, n in names.items() if v != "reversed"}
        if len(by_variant) >= 2:
            s = stats.sensitivity(spec, by_variant)
            entry["sensitivity"] = asdict(s)
            entry["sensitivity"]["reading"] = _read_sensitivity(s.changed_rate)
        if "reversed" in names:
            r = stats.reversal(spec, by_variant[next(iter(by_variant))], majorities.get(names["reversed"], {}))
            entry["reversal"] = asdict(r)
            entry["reversal"]["reading"] = _read_reversal(r.inversion_rate)
        entry["majorities"] = {v: majorities.get(n, {}) for v, n in names.items()}
        out["judges"].append(entry)
    out["run"] = result.to_dict()
    return out


def _read_sensitivity(rate: float) -> str:
    if rate <= 0.05:
        return "robust"
    if rate <= 0.15:
        return "wording-sensitive"
    return "measures the wording"


def _read_reversal(rate: float) -> str:
    if rate >= 0.95:
        return "reads the rubric"
    if rate >= 0.7:
        return "mostly reads the rubric"
    return "ignores the rubric"


def print_rubric_check(out: dict, console: Console | None = None) -> None:
    con = console or Console()
    con.print()
    con.print(f"[bold]judgeflake rubric-check[/bold] · rubric {out['rubric']} · {len(out['variants'])} wordings{' + reversed' if out['reversal'] else ''} · {out['run']['runs']} run(s) each · {out['calls']} verdicts")
    con.print()
    t = Table(title="Rubric sensitivity (same meaning, different words)", title_justify="left")
    t.add_column("judge", style="bold")
    t.add_column("cases", justify="right")
    t.add_column("verdicts that changed", justify="right")
    t.add_column("alpha across wordings", justify="right")
    t.add_column("per wording vs v0")
    t.add_column("reading")
    for j in out["judges"]:
        s = j.get("sensitivity")
        if not s:
            continue
        color = {"robust": "green", "wording-sensitive": "yellow"}.get(s["reading"], "red")
        per = ", ".join(f"{v} {r:.0%}" for v, r in s["vs_baseline"].items())
        t.add_row(j["judge"], str(s["cases"]), f"{len(s['changed_cases'])} ({s['changed_rate']:.0%})", f"{s['alpha']:.2f}" if s["alpha"] is not None else "—", per, f"[{color}]{s['reading']}[/{color}]")
    con.print(t)
    if out["reversal"] and any(j.get("reversal") for j in out["judges"]):
        con.print()
        t2 = Table(title="Rubric reversal (ask for INCORRECT instead of correct)", title_justify="left")
        t2.add_column("judge", style="bold")
        t2.add_column("cases", justify="right")
        t2.add_column("verdicts inverted", justify="right")
        t2.add_column("unchanged", justify="right")
        t2.add_column("reading")
        for j in out["judges"]:
            r = j.get("reversal")
            if not r:
                continue
            color = {"reads the rubric": "green", "mostly reads the rubric": "yellow"}.get(r["reading"], "red")
            t2.add_row(j["judge"], str(r["cases"]), f"{r['inverted']} ({r['inversion_rate']:.0%})", str(r["unchanged"]), f"[{color}]{r['reading']}[/{color}]")
        con.print(t2)
        for j in out["judges"]:
            r = j.get("reversal")
            if r and r["ignored_cases"]:
                con.print(f"  [dim]{j['judge']} gave the same verdict under both rubrics on:[/dim] {', '.join(r['ignored_cases'][:12])}{' …' if len(r['ignored_cases']) > 12 else ''}")
    for j in out["judges"]:
        s = j.get("sensitivity")
        if s and s["changed_cases"]:
            con.print(f"  [dim]{j['judge']} changed its verdict under rewording on:[/dim] {', '.join(s['changed_cases'][:12])}{' …' if len(s['changed_cases']) > 12 else ''}")
    if out["skipped"]:
        con.print(f"[dim]skipped (no rubric to reword): {', '.join(out['skipped'])}[/dim]")
    con.print()


# ---------------------------------------------------------------------------------- migrate --


def migrate(old_run: dict, new_run: dict | None, old_judge: str | None, new_judge: str | None, pass_mark: float | None, per_case_threshold: float | None) -> dict:
    """Re-anchor a pass mark from one judge to another on the same cases."""
    old_rep = old_run["report"]
    new_rep = (new_run or old_run)["report"]
    names_old = [j["name"] for j in old_rep["judges"]]
    names_new = [j["name"] for j in new_rep["judges"]]
    if old_judge is None or new_judge is None:
        if new_run is None and len(names_old) >= 2:
            old_judge, new_judge = old_judge or names_old[0], new_judge or names_old[1]
        elif new_run is not None:
            old_judge, new_judge = old_judge or names_old[0], new_judge or names_new[0]
        else:
            raise ValueError("say which judges to compare with --old-judge and --new-judge")
    o = next((j for j in old_rep["judges"] if j["name"] == old_judge), None)
    n = next((j for j in new_rep["judges"] if j["name"] == new_judge), None)
    if not o or not n:
        raise ValueError(f"judges not found: have {names_old} and {names_new}")
    t = stats.translate_threshold(old_judge, new_judge, o["majorities"], n["majorities"], pass_mark, per_case_threshold)
    out = asdict(t)
    out["changed"] = [{"case_id": c, "old": a, "new": b} for c, a, b in t.changed]
    out["old_accuracy"] = o["accuracy"]["accuracy"] if o.get("accuracy") else None
    out["new_accuracy"] = n["accuracy"]["accuracy"] if n.get("accuracy") else None
    out["old_self_agreement"] = o.get("self_agreement")
    out["new_self_agreement"] = n.get("self_agreement")
    out["kappa_reading"] = stats.read_kappa(t.kappa)
    return out


def print_migrate(out: dict, console: Console | None = None) -> None:
    con = console or Console()
    con.print()
    con.print(f"[bold]judgeflake migrate[/bold] · {out['old_judge']} → {out['new_judge']} · {out['cases']} shared cases")
    t = Table(show_header=True)
    t.add_column("")
    t.add_column(out["old_judge"], justify="right")
    t.add_column(out["new_judge"], justify="right")
    t.add_row("aggregate score (pass rate or mean)", f"{out['old_mean']:.2f}", f"{out['new_mean']:.2f}")
    if out["old_accuracy"] is not None:
        t.add_row("vs human labels", f"{out['old_accuracy']:.0%}", f"{out['new_accuracy']:.0%}")
    if out["old_self_agreement"] is not None:
        t.add_row("self-agreement", f"{out['old_self_agreement']:.0%}", f"{out['new_self_agreement']:.0%}" if out["new_self_agreement"] is not None else "—")
    con.print(t)
    con.print(f"agree on {out['percent_agreement']:.0%} of cases · kappa {out['kappa']:.2f} ({out['kappa_reading']}) · shift {out['offset']:+.2f}")
    if out["pass_mark"] is not None:
        con.print(f"[bold]your pass mark {out['pass_mark']:g} on {out['old_judge']} is about {out['translated_mark']:.2f} on {out['new_judge']}[/bold]")
    if out["per_case_threshold"] is not None and out["translated_per_case"] is not None:
        con.print(f"[bold]a per-case cutoff of {out['per_case_threshold']:g} on {out['old_judge']} passes the same share of cases as {out['translated_per_case']:.2f} on {out['new_judge']}[/bold]")
    if out["changed"]:
        con.print(f"\n{len(out['changed'])} cases get a different verdict:")
        for c in out["changed"][:20]:
            con.print(f"  {c['case_id']}: {_fmt(c['old'])} → {_fmt(c['new'])}")
        if len(out["changed"]) > 20:
            con.print(f"  … and {len(out['changed']) - 20} more")
    con.print("[dim]The translated mark keeps your pass/fail rate; it does not make the two judges agree. Look at the changed cases before you trust it.[/dim]")
    con.print()


def _fmt(v: Any) -> str:
    if isinstance(v, bool):
        return "T" if v else "F"
    return str(v)


# --------------------------------------------------------------------------------- sentinel --
# Design borrowed with thanks from judge-drift-sentinel (homayoun-safarpour): a frozen anchor
# set and a baseline, then every later run is compared to the baseline, not to last week.


def anchor_hash(cases: Sequence[Case]) -> str:
    h = hashlib.sha256()
    for c in sorted(cases, key=lambda c: c.id):
        h.update(json.dumps([c.id, c.inputs, c.outputs, c.reference_outputs, c.label], sort_keys=True, default=str).encode())
    return h.hexdigest()[:16]


def sentinel_baseline(run_data: dict) -> dict:
    """Freeze a run as the baseline: per judge, the majority verdict on every anchor case."""
    rep = run_data["report"]
    result = RunResult.from_dict(run_data["run"])
    return {
        "created": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "source_run": rep["meta"]["created"],
        "anchor_hash": anchor_hash(result.cases),
        "cases": len(result.cases),
        "judges": {
            j["name"]: {
                "describe": j["describe"],
                "majorities": j["majorities"],
                "self_agreement": j.get("self_agreement"),
                "accuracy": j["accuracy"]["accuracy"] if j.get("accuracy") else None,
                "kappa_vs_human": j["accuracy"]["kappa"] if j.get("accuracy") else None,
            }
            for j in rep["judges"]
        },
    }


def sentinel_check(baseline: dict, report: dict, cases: Sequence[Case], max_changed: float = 0.10, max_kappa_drop: float = 0.15) -> dict:
    """Compare a fresh run against the frozen baseline. Verdict per judge: STABLE or JUDGE_DRIFT."""
    out: dict[str, Any] = {"anchor_ok": anchor_hash(cases) == baseline["anchor_hash"], "judges": [], "verdict": "STABLE"}
    for j in report["judges"]:
        b = baseline["judges"].get(j["name"])
        if not b:
            out["judges"].append({"name": j["name"], "verdict": "NO_BASELINE"})
            continue
        common = [c for c in b["majorities"] if c in j["majorities"]]
        changed = [c for c in common if b["majorities"][c] != j["majorities"][c]]
        rate = len(changed) / len(common) if common else 0.0
        a = [b["majorities"][c] for c in common]
        n = [j["majorities"][c] for c in common]
        kappa_vs_baseline = stats.cohens_kappa(a, n) if common else None
        k_old = b.get("kappa_vs_human")
        k_new = j["accuracy"]["kappa"] if j.get("accuracy") else None
        drop = (k_old - k_new) if (k_old is not None and k_new is not None) else None
        reasons = []
        if rate > max_changed:
            reasons.append(f"{len(changed)} of {len(common)} anchor verdicts changed ({rate:.0%} > {max_changed:.0%})")
        if drop is not None and drop > max_kappa_drop:
            reasons.append(f"kappa vs human fell {k_old:.2f} → {k_new:.2f}")
        verdict = "JUDGE_DRIFT" if reasons else "STABLE"
        if verdict == "JUDGE_DRIFT":
            out["verdict"] = "JUDGE_DRIFT"
        out["judges"].append(
            {
                "name": j["name"],
                "cases": len(common),
                "changed": changed,
                "changed_rate": round(rate, 4),
                "kappa_vs_baseline": round(kappa_vs_baseline, 4) if kappa_vs_baseline is not None else None,
                "kappa_vs_human_old": k_old,
                "kappa_vs_human_new": k_new,
                "self_agreement_old": b.get("self_agreement"),
                "self_agreement_new": j.get("self_agreement"),
                "verdict": verdict,
                "reasons": reasons,
            }
        )
    if not out["anchor_ok"]:
        out["verdict"] = "ANCHORS_CHANGED"
    return out


def print_sentinel(out: dict, console: Console | None = None) -> None:
    con = console or Console()
    con.print()
    color = {"STABLE": "green", "JUDGE_DRIFT": "red", "ANCHORS_CHANGED": "yellow"}[out["verdict"]]
    con.print(f"[bold]judgeflake sentinel[/bold] · [{color}]{out['verdict']}[/{color}]")
    if not out["anchor_ok"]:
        con.print("[yellow]the anchor cases differ from the ones the baseline was frozen on; re-freeze with `sentinel baseline`[/yellow]")
    t = Table()
    t.add_column("judge", style="bold")
    t.add_column("anchors", justify="right")
    t.add_column("verdicts changed", justify="right")
    t.add_column("kappa vs baseline", justify="right")
    t.add_column("kappa vs human", justify="right")
    t.add_column("verdict")
    for j in out["judges"]:
        if j["verdict"] == "NO_BASELINE":
            t.add_row(j["name"], "—", "—", "—", "—", "[yellow]not in baseline[/yellow]")
            continue
        kh = "—" if j["kappa_vs_human_old"] is None else f"{j['kappa_vs_human_old']:.2f} → {j['kappa_vs_human_new']:.2f}"
        c = {"STABLE": "green", "JUDGE_DRIFT": "red"}[j["verdict"]]
        t.add_row(j["name"], str(j["cases"]), f"{len(j['changed'])} ({j['changed_rate']:.0%})", f"{j['kappa_vs_baseline']:.2f}" if j["kappa_vs_baseline"] is not None else "—", kh, f"[{c}]{j['verdict']}[/{c}]")
    con.print(t)
    for j in out["judges"]:
        for r in j.get("reasons", []):
            con.print(f"  [red]{j['name']}:[/red] {r}")
        if j.get("changed"):
            con.print(f"  [dim]{j['name']} changed on:[/dim] {', '.join(j['changed'][:12])}{' …' if len(j['changed']) > 12 else ''}")
    con.print("[dim]Anchors never change, so a change here is the judge, not your system. Idea from judge-drift-sentinel.[/dim]")
    con.print()


# --------------------------------------------------------------------------------- escalate --


def _csv_safe(value: str) -> str:
    """Stop a cell that starts with = + - @ from running as a formula when the CSV opens in a spreadsheet.

    Model outputs are untrusted text; "=HYPERLINK(...)" in an answer must stay text.
    """
    if value and value[0] in "=+-@\t\r":
        return "'" + value
    return value


def write_escalation_csv(rows: list[dict], path: Path, judge_names: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["case_id", "reason", "disagreement", *[f"verdicts:{j}" for j in judge_names], "human_label", "inputs", "outputs", "reference_outputs", "your_verdict", "notes"])
        for r in rows:
            cells = [r["case_id"], r["reason"], r["disagreement"], *[r["verdicts"].get(j, "") for j in judge_names], _fmt(r["label"]) if r["label"] is not None else "", _s(r["inputs"]), _s(r["outputs"]), _s(r["reference_outputs"]), "", ""]
            w.writerow([_csv_safe(c) if isinstance(c, str) else c for c in cells])


def push_to_langsmith(rows: list[dict], dataset_name: str, client: Any = None) -> dict:
    """Create (or extend) a LangSmith dataset with the cases that need a human. Needs `langsmith` installed and LANGSMITH_API_KEY."""
    if client is None:
        try:
            from langsmith import Client  # type: ignore
        except ImportError as e:
            raise RuntimeError("pip install langsmith, and set LANGSMITH_API_KEY") from e
        client = Client()
    try:
        ds = client.read_dataset(dataset_name=dataset_name)
    except Exception:
        ds = client.create_dataset(dataset_name=dataset_name, description="Cases judgeflake flagged for human review: judges disagreed or missed the label.")
    client.create_examples(
        inputs=[{"inputs": r["inputs"]} for r in rows],
        outputs=[{"outputs": r["outputs"], "reference_outputs": r["reference_outputs"]} for r in rows],
        metadata=[{"judgeflake_case_id": r["case_id"], "reason": r["reason"], "disagreement": r["disagreement"], "verdicts": r["verdicts"], "human_label": r["label"]} for r in rows],
        dataset_id=ds.id,
    )
    return {"dataset": dataset_name, "id": str(ds.id), "examples": len(rows)}


def _s(v: Any) -> str:
    if v is None:
        return ""
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False)


__all__ = ["rubric_check", "print_rubric_check", "migrate", "print_migrate", "sentinel_baseline", "sentinel_check", "print_sentinel", "escalation", "write_escalation_csv", "push_to_langsmith", "analyze"]
