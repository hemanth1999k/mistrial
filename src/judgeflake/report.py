"""Turn a RunResult into the reliability card: a dict, a terminal view, and an HTML file."""

from __future__ import annotations

import html
import json
from collections import Counter
from dataclasses import asdict
from pathlib import Path
from typing import Any

from rich.console import Console
from rich.table import Table

from . import stats
from .runner import RunResult, by_judge


# -------------------------------------------------------------------------------- analyze --


PARSE_MODES = ("json", "fenced", "embedded", "no-verdict", "error")


def _parser_audit(name: str, verdicts: list) -> dict | None:
    """How the judge's replies were read. None for judges that don't produce text (functions, demo)."""
    mine = [v for v in verdicts if v.judge == name]
    if not any(v.text is not None or v.parse not in (None, "error") for v in mine):
        return None
    counts = Counter((v.parse or "error") for v in mine)
    total = sum(counts.values())
    strict = counts.get("json", 0)
    lenient = counts.get("fenced", 0) + counts.get("embedded", 0)
    failed = counts.get("no-verdict", 0) + counts.get("error", 0)
    examples = [
        {"case_id": v.case_id, "run": v.run, "parse": v.parse, "text": (v.text or v.error or "")[:240]}
        for v in mine
        if v.parse in ("embedded", "no-verdict", "error")
    ][:5]
    reading = "clean" if failed == 0 and lenient == 0 else "lenient" if failed == 0 else "fragile"
    return {
        "total": total,
        "counts": {m: counts.get(m, 0) for m in PARSE_MODES},
        "strict_rate": round(strict / total, 4) if total else None,
        "lenient_rate": round(lenient / total, 4) if total else None,
        "failed_rate": round(failed / total, 4) if total else None,
        "reading": reading,
        "examples": examples,
    }


def analyze(result: RunResult, top: int = 10, pass_mark: float | None = None, runs_in_ci: int = 1) -> dict:
    """Everything the report shows, as plain data. Formulas live in stats.py."""
    values = by_judge(result)
    labels = {c.id: c.label for c in result.cases if c.label is not None}
    errors = Counter(v.judge for v in result.verdicts if v.error)
    cached = Counter(v.judge for v in result.verdicts if v.cached)
    latency: dict[str, list[float]] = {}
    for v in result.verdicts:
        if not v.error and not v.cached:
            latency.setdefault(v.judge, []).append(v.latency_ms)

    judges = []
    majorities: dict[str, dict[str, Any]] = {}
    gates = []
    for j in result.judges:
        name = j["name"]
        cons = stats.consistency(name, values.get(name, {}))
        maj = {c.case_id: c.majority for c in cons.per_case}
        majorities[name] = maj
        acc = stats.accuracy(name, maj, labels) if labels else None
        lat = latency.get(name, [])
        if pass_mark is not None:
            g = stats.gate_risk(name, values.get(name, {}), pass_mark, runs_in_ci=runs_in_ci)
            if g:
                gates.append(asdict(g))
        judges.append(
            {
                "name": name,
                "describe": j,
                "cases": cons.cases,
                "runs": cons.runs,
                "errors": errors.get(name, 0),
                "cached": cached.get(name, 0),
                "parser": _parser_audit(name, result.verdicts),
                "mean_latency_ms": round(sum(lat) / len(lat), 1) if lat else None,
                "numeric": cons.numeric,
                "flip_rate": round(cons.flip_rate, 4),
                "flip_cases": cons.flip_cases,
                "self_agreement": round(cons.self_agreement, 4) if cons.self_agreement is not None else None,
                "mean_agreement": round(cons.mean_agreement, 4),
                "mean_spread": round(cons.mean_spread, 4) if cons.mean_spread is not None else None,
                "reading": stats.read_self_agreement(cons.self_agreement),
                "majorities": maj,
                "all_values": {c.case_id: list(c.values) for c in cons.per_case},
                "accuracy": (
                    {
                        "cases": acc.cases,
                        "accuracy": round(acc.accuracy, 4),
                        "kappa": round(acc.kappa, 4),
                        "kappa_reading": stats.read_kappa(acc.kappa),
                        "mean_abs_error": round(acc.mean_abs_error, 4) if acc.mean_abs_error is not None else None,
                        "misses": [{"case_id": c, "judge": jv, "label": lv} for c, jv, lv in acc.misses],
                    }
                    if acc
                    else None
                ),
            }
        )

    agreement = None
    if len(result.judges) >= 2:
        ag = stats.agreement(majorities, values)
        agreement = {
            "level": ag.level,
            "alpha_majority": round(ag.alpha_majority, 4) if ag.alpha_majority is not None else None,
            "alpha_all_runs": round(ag.alpha_all_runs, 4) if ag.alpha_all_runs is not None else None,
            "alpha_reading": stats.read_kappa(ag.alpha_majority),
            "pairs": [
                {**asdict(p), "percent": round(p.percent, 4), "kappa": round(p.kappa, 4), "kappa_reading": stats.read_kappa(p.kappa)}
                for p in ag.pairs
            ],
        }

    flaky = [
        {"case_id": f.case_id, "disagreement": f.disagreement, "verdicts": f.verdicts, "label": f.label}
        for f in stats.flaky_cases(values, labels, top=top)
    ]

    return {
        "meta": {
            "created": result.created,
            "label": result.label,
            "runs": result.runs,
            "rubric": result.rubric,
            "cases": len(result.cases),
            "judges": len(result.judges),
            "labels": len(labels),
            "duration_s": round(result.duration_s, 2),
            "calls": len(result.verdicts),
            "adaptive": result.adaptive,
            "calls_saved": result.calls_saved,
        },
        "judges": judges,
        "agreement": agreement,
        "gate": {"pass_mark": pass_mark, "runs_in_ci": runs_in_ci, "judges": gates} if pass_mark is not None else None,
        "flaky_cases": flaky,
        "cases": {c.id: {"inputs": c.inputs, "outputs": c.outputs, "reference_outputs": c.reference_outputs, "label": c.label} for c in result.cases},
    }


# ------------------------------------------------------------------------------- readings --


def headline(report: dict) -> list[str]:
    """One line per judge, plus one for agreement. Plain words, no grades."""
    lines = []
    for j in report["judges"]:
        n_flip = len(j["flip_cases"])
        sa = f"two runs agree {j['self_agreement']:.0%} of the time; " if j["self_agreement"] is not None else ""
        line = f"{j['name']}: {j['reading']} — {sa}flipped on {n_flip} of {j['cases']} cases across {j['runs']} runs"
        if j["accuracy"]:
            line += f"; matches human labels on {j['accuracy']['accuracy']:.0%} ({j['accuracy']['kappa_reading']} kappa)"
        if j["errors"]:
            line += f"; {j['errors']} errors"
        lines.append(line)
    ag = report.get("agreement")
    if ag and ag["alpha_majority"] is not None:
        lines.append(f"judges agree with each other: alpha {ag['alpha_majority']:.2f} ({ag['alpha_reading']})")
    return lines


def _fmt_value(v: Any) -> str:
    if isinstance(v, bool):
        return "T" if v else "F"
    if isinstance(v, float):
        return f"{v:g}"
    return str(v)


# ------------------------------------------------------------------------------- terminal --


def print_report(report: dict, console: Console | None = None) -> None:
    con = console or Console()
    m = report["meta"]
    con.print()
    con.print(f"[bold]judgeflake[/bold] · {m['cases']} cases · {m['judges']} judge{'s' if m['judges'] != 1 else ''} · {m['runs']} runs each · {m['calls']} verdicts in {m['duration_s']}s")
    con.print()

    t = Table(title="Consistency (same judge, asked again)", title_justify="left", show_lines=False)
    t.add_column("judge", style="bold")
    t.add_column("self-agreement", justify="right")
    t.add_column("flipped cases", justify="right")
    t.add_column("vs human", justify="right")
    t.add_column("errors", justify="right")
    t.add_column("reading")
    for j in report["judges"]:
        acc = j["accuracy"]
        t.add_row(
            j["name"],
            f"{j['self_agreement']:.0%}" if j["self_agreement"] is not None else "—",
            f"{len(j['flip_cases'])}/{j['cases']} ({j['flip_rate']:.0%})",
            f"{acc['accuracy']:.0%} (κ {acc['kappa']:.2f})" if acc else "—",
            str(j["errors"]) if j["errors"] else "0",
            _color_reading(j["reading"]),
        )
    con.print(t)

    ag = report.get("agreement")
    if ag:
        con.print()
        t2 = Table(title="Agreement (different judges)", title_justify="left")
        t2.add_column("pair", style="bold")
        t2.add_column("cases", justify="right")
        t2.add_column("agree", justify="right")
        t2.add_column("kappa", justify="right")
        t2.add_column("reading")
        for p in ag["pairs"]:
            t2.add_row(f"{p['judge_a']} vs {p['judge_b']}", str(p["cases"]), f"{p['percent']:.0%}", f"{p['kappa']:.2f}", p["kappa_reading"])
        con.print(t2)
        if ag["alpha_majority"] is not None:
            con.print(f"Krippendorff's alpha: {ag['alpha_majority']:.2f} on majority verdicts, {ag['alpha_all_runs']:.2f} counting every run ({ag['level']})")

    if report["flaky_cases"]:
        con.print()
        t3 = Table(title="Flakiest cases (everyone's verdicts pooled)", title_justify="left")
        t3.add_column("case", style="bold")
        t3.add_column("disagreement", justify="right")
        for j in report["judges"]:
            t3.add_column(j["name"])
        t3.add_column("human")
        for f in report["flaky_cases"]:
            row = [f["case_id"], f"{f['disagreement']:.0%}"]
            for j in report["judges"]:
                row.append(" ".join(_fmt_value(v) for v in f["verdicts"].get(j["name"], [])))
            row.append(_fmt_value(f["label"]) if f["label"] is not None else "—")
            t3.add_row(*row)
        con.print(t3)

    audits = [(j["name"], j["parser"]) for j in report["judges"] if j.get("parser")]
    if audits:
        con.print()
        t4 = Table(title="Parser audit (how the replies were read)", title_justify="left")
        t4.add_column("judge", style="bold")
        t4.add_column("strict JSON", justify="right")
        t4.add_column("fenced/embedded", justify="right")
        t4.add_column("no verdict", justify="right")
        t4.add_column("reading")
        for name, p in audits:
            c = p["counts"]
            color = {"clean": "green", "lenient": "yellow", "fragile": "red"}[p["reading"]]
            t4.add_row(name, f"{c['json']}/{p['total']}", str(c["fenced"] + c["embedded"]), str(c["no-verdict"] + c["error"]), f"[{color}]{p['reading']}[/{color}]")
        con.print(t4)
        for name, p in audits:
            for ex in p["examples"][:3]:
                con.print(f"  [dim]{name} · {ex['case_id']} · run {ex['run']} · {ex['parse']}:[/dim] {ex['text'][:120]!r}")

    gate = report.get("gate")
    if gate and gate["judges"]:
        con.print()
        t5 = Table(title=f"Gate risk (pass mark {gate['pass_mark']:g}, {gate['runs_in_ci']} run{'s' if gate['runs_in_ci'] != 1 else ''} per case in CI)", title_justify="left")
        t5.add_column("judge", style="bold")
        t5.add_column("expected score", justify="right")
        t5.add_column("90% range", justify="right")
        t5.add_column("false alarm", justify="right")
        t5.add_column("safe mark", justify="right")
        t5.add_column("fix")
        for g in gate["judges"]:
            fa = g["false_alarm"]
            color = "green" if fa <= 0.05 else "yellow" if fa <= 0.2 else "red"
            if g["expected"] < g["pass_mark"]:
                fix = "already below the mark on average"
            elif fa <= 0.05:
                fix = "fine as is"
            elif g["runs_for_5pct"]:
                fix = f"{g['runs_for_5pct']} runs per case, or mark ≤ {g['safe_mark']:.2f}"
            else:
                fix = f"lower the mark to {g['safe_mark']:.2f}"
            t5.add_row(g["judge"], f"{g['expected']:.2f}", f"{g['p5']:.2f}–{g['p95']:.2f}", f"[{color}]{fa:.0%}[/{color}]", f"{g['safe_mark']:.2f}", fix)
        con.print(t5)
        con.print("[dim]false alarm = how often this gate goes red with nothing changed, from re-sampling each case's observed verdicts.[/dim]")

    con.print()
    for line in headline(report):
        con.print(f"• {line}")
    if m.get("adaptive"):
        con.print(f"• adaptive: re-asked only where the first two runs disagreed; skipped {m['calls_saved']} calls")
    con.print()


def _color_reading(reading: str) -> str:
    color = {"stable": "green", "mostly stable": "green", "noticeably flaky": "yellow", "flaky": "red", "very flaky": "red"}.get(reading, "white")
    return f"[{color}]{reading}[/{color}]"


# ----------------------------------------------------------------------------------- html --

_CSS = """
:root{--bg:#fbfaf7;--ink:#1b1a17;--soft:#5f5b53;--line:#e4e0d6;--card:#ffffff;--ok:#2f7d4f;--warn:#b7791f;--bad:#b3362b;--acc:#c8421f;--mono:ui-monospace,SFMono-Regular,Menlo,monospace}
@media(prefers-color-scheme:dark){:root{--bg:#121417;--ink:#ecebe6;--soft:#a39f96;--line:#2a2e35;--card:#191c21;--ok:#5fbf86;--warn:#e0a63a;--bad:#ef6d5f;--acc:#ff7b4f}}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:15px/1.5 -apple-system,Inter,Segoe UI,sans-serif}
main{max-width:1040px;margin:0 auto;padding:40px 20px 80px}h1{font-size:28px;margin:0 0 4px;letter-spacing:-.01em}h2{font-size:17px;margin:40px 0 12px}
.meta{color:var(--soft);font-size:13px}.grid{display:grid;gap:14px;grid-template-columns:repeat(auto-fit,minmax(260px,1fr))}
.card{background:var(--card);border:1px solid var(--line);border-radius:10px;padding:16px 18px}.card h3{margin:0 0 10px;font-size:15px;font-family:var(--mono);font-weight:600}
.big{font-size:34px;font-weight:600;letter-spacing:-.02em;line-height:1}.lbl{color:var(--soft);font-size:12px;text-transform:uppercase;letter-spacing:.08em}
.bar{height:6px;background:var(--line);border-radius:99px;overflow:hidden;margin:10px 0 6px}.bar i{display:block;height:100%;background:var(--acc)}
.row{display:flex;justify-content:space-between;gap:12px;font-size:13px;color:var(--soft);margin-top:6px}.row b{color:var(--ink);font-weight:500}
table{width:100%;border-collapse:collapse;font-size:14px}th,td{text-align:left;padding:8px 10px;border-bottom:1px solid var(--line);vertical-align:top}th{color:var(--soft);font-weight:500;font-size:12px;text-transform:uppercase;letter-spacing:.06em}
td.r,th.r{text-align:right;font-family:var(--mono)}.v{display:inline-block;min-width:22px;text-align:center;font-family:var(--mono);font-size:12px;padding:1px 5px;border-radius:4px;background:var(--line);margin-right:3px}
.v.t{background:color-mix(in srgb,var(--ok) 22%,transparent)}.v.f{background:color-mix(in srgb,var(--bad) 22%,transparent)}.ok{color:var(--ok)}.warn{color:var(--warn)}.bad{color:var(--bad)}
.read{font-weight:600}details{margin-top:8px}summary{cursor:pointer;color:var(--soft);font-size:13px}pre{white-space:pre-wrap;font:12px/1.5 var(--mono);background:var(--bg);border:1px solid var(--line);border-radius:6px;padding:10px;margin:6px 0 0}
ul.head{padding-left:18px}ul.head li{margin:4px 0}footer{margin-top:60px;color:var(--soft);font-size:12px}
"""


def _cls(reading: str) -> str:
    return {"stable": "ok", "mostly stable": "ok", "noticeably flaky": "warn"}.get(reading, "bad")


def _chip(v: Any) -> str:
    s = _fmt_value(v)
    c = "t" if v is True else "f" if v is False else ""
    return f'<span class="v {c}">{html.escape(s)}</span>'


def to_html(report: dict, title: str = "judgeflake report") -> str:
    m = report["meta"]
    e = html.escape
    cards = []
    for j in report["judges"]:
        acc = j["accuracy"]
        acc_html = (
            f'<div class="row"><span>vs human labels</span><b>{acc["accuracy"]:.0%} · κ {acc["kappa"]:.2f} ({e(acc["kappa_reading"])})</b></div>' if acc else ""
        )
        spread = f'<div class="row"><span>mean spread across runs</span><b>{j["mean_spread"]}</b></div>' if j["mean_spread"] is not None else ""
        errs = f'<div class="row"><span>errors</span><b class="bad">{j["errors"]}</b></div>' if j["errors"] else ""
        lat = f'<div class="row"><span>mean latency</span><b>{j["mean_latency_ms"]:.0f} ms</b></div>' if j["mean_latency_ms"] else ""
        sa = j["self_agreement"]
        cards.append(
            f"""<div class="card"><h3>{e(j['name'])}</h3>
<div class="lbl">two runs agree</div><div class="big {_cls(j['reading'])}">{f"{sa:.0%}" if sa is not None else "—"}</div>
<div class="bar"><i style="width:{(sa or 0) * 100:.1f}%"></i></div>
<div class="row"><span>flipped on</span><b>{len(j['flip_cases'])} of {j['cases']} cases · {j['runs']} runs ({j['flip_rate']:.0%})</b></div>
{acc_html}{spread}{lat}{errs}
<div class="row"><span>reading</span><b class="read {_cls(j['reading'])}">{e(j['reading'])}</b></div></div>"""
        )

    ag = report.get("agreement")
    ag_html = ""
    if ag:
        rows = "".join(
            f"<tr><td>{e(p['judge_a'])} vs {e(p['judge_b'])}</td><td class='r'>{p['cases']}</td><td class='r'>{p['percent']:.0%}</td><td class='r'>{p['kappa']:.2f}</td><td>{e(p['kappa_reading'])}</td></tr>"
            for p in ag["pairs"]
        )
        alpha = (
            f"<p class='meta'>Krippendorff's alpha {ag['alpha_majority']:.2f} on majority verdicts, {ag['alpha_all_runs']:.2f} counting every run as a rater ({e(ag['level'])} level).</p>"
            if ag["alpha_majority"] is not None
            else ""
        )
        ag_html = f"<h2>Agreement between judges</h2><table><tr><th>pair</th><th class='r'>cases</th><th class='r'>agree</th><th class='r'>kappa</th><th>reading</th></tr>{rows}</table>{alpha}"

    flaky_html = ""
    if report["flaky_cases"]:
        head = "".join(f"<th>{e(j['name'])}</th>" for j in report["judges"])
        rows = []
        for f in report["flaky_cases"]:
            c = report["cases"].get(f["case_id"], {})
            cells = "".join(f"<td>{''.join(_chip(v) for v in f['verdicts'].get(j['name'], []))}</td>" for j in report["judges"])
            detail = (
                f"<details><summary>show case</summary><pre>INPUT: {e(_s(c.get('inputs')))}\n\nOUTPUT: {e(_s(c.get('outputs')))}\n\nREFERENCE: {e(_s(c.get('reference_outputs')))}</pre></details>"
            )
            rows.append(
                f"<tr><td><b>{e(f['case_id'])}</b>{detail}</td><td class='r'>{f['disagreement']:.0%}</td>{cells}<td>{_chip(f['label']) if f['label'] is not None else '—'}</td></tr>"
            )
        flaky_html = f"<h2>Flakiest cases</h2><p class='meta'>Every judge's verdicts for the case, one chip per run. Disagreement is 1 minus the share of the overall majority.</p><table><tr><th>case</th><th class='r'>disagreement</th>{head}<th>human</th></tr>{''.join(rows)}</table>"

    audit_html = ""
    audits = [(j["name"], j["parser"]) for j in report["judges"] if j.get("parser")]
    if audits:
        rows = []
        for name, p in audits:
            c = p["counts"]
            ex = "".join(f"<pre>{e(x['case_id'])} · run {x['run']} · {e(str(x['parse']))}\n{e(x['text'])}</pre>" for x in p["examples"][:3])
            det = f"<details><summary>examples</summary>{ex}</details>" if ex else ""
            rows.append(
                f"<tr><td><b>{e(name)}</b>{det}</td><td class='r'>{c['json']}/{p['total']}</td><td class='r'>{c['fenced'] + c['embedded']}</td><td class='r'>{c['no-verdict'] + c['error']}</td><td class='{ {'clean': 'ok', 'lenient': 'warn', 'fragile': 'bad'}[p['reading']] } read'>{e(p['reading'])}</td></tr>"
            )
        audit_html = f"<h2>Parser audit</h2><p class='meta'>How each reply was read. Replies that needed the lenient path (a code fence, JSON buried in prose) are one bad day from unparseable, and a parsing change can flip verdicts the model never changed.</p><table><tr><th>judge</th><th class='r'>strict JSON</th><th class='r'>fenced / embedded</th><th class='r'>no verdict</th><th>reading</th></tr>{''.join(rows)}</table>"

    gate_html = ""
    gate = report.get("gate")
    if gate and gate["judges"]:
        rows = []
        for g in gate["judges"]:
            fa = g["false_alarm"]
            cls = "ok" if fa <= 0.05 else "warn" if fa <= 0.2 else "bad"
            if g["expected"] < g["pass_mark"]:
                fix = "already below the mark on average"
            elif fa <= 0.05:
                fix = "fine as is"
            elif g["runs_for_5pct"]:
                fix = f"{g['runs_for_5pct']} runs per case, or a mark of {g['safe_mark']:.2f}"
            else:
                fix = f"lower the mark to {g['safe_mark']:.2f}"
            rows.append(f"<tr><td><b>{e(g['judge'])}</b></td><td class='r'>{g['expected']:.2f}</td><td class='r'>{g['p5']:.2f}–{g['p95']:.2f}</td><td class='r {cls}'><b>{fa:.0%}</b></td><td class='r'>{g['safe_mark']:.2f}</td><td>{e(fix)}</td></tr>")
        gate_html = f"<h2>Gate risk</h2><p class='meta'>Pass mark {gate['pass_mark']:g}, {gate['runs_in_ci']} run(s) per case in CI. False alarm is how often the gate goes red with nothing changed, from re-sampling each case's observed verdicts {gate['judges'][0]['draws']} times.</p><table><tr><th>judge</th><th class='r'>expected</th><th class='r'>90% range</th><th class='r'>false alarm</th><th class='r'>safe mark</th><th>fix</th></tr>{''.join(rows)}</table>"

    adaptive = f"<p class='meta'>Adaptive: re-asked only where the first two runs disagreed; skipped {m['calls_saved']} calls.</p>" if m.get("adaptive") else ""
    heads = "".join(f"<li>{e(line)}</li>" for line in headline(report))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>{e(title)}</title><style>{_CSS}</style></head>
<body><main>
<h1>{e(title)}</h1>
<p class="meta">{m['cases']} cases · {m['judges']} judge{'s' if m['judges'] != 1 else ''} · {m['runs']} runs each · {m['calls']} verdicts · rubric {e(str(m['rubric'] or '—'))} · {e(m['created'])}</p>
<ul class="head">{heads}</ul>{adaptive}
<h2>Consistency (same judge, asked again)</h2>
<div class="grid">{''.join(cards)}</div>
{ag_html}
{gate_html}
{flaky_html}
{audit_html}
<footer>Made with judgeflake. Formulas: "two runs agree" = mean over cases of the share of run pairs with equal verdicts; flip rate = cases with more than one distinct verdict across runs; Cohen's kappa and Krippendorff's alpha as standard; kappa readings use Landis &amp; Koch bands.</footer>
</main></body></html>"""


def _s(v: Any) -> str:
    if v is None:
        return "(none)"
    return v if isinstance(v, str) else json.dumps(v, ensure_ascii=False, indent=2)


def write_outputs(report: dict, result: RunResult, html_path: Path | None, json_path: Path | None, title: str = "judgeflake report", markdown_path: Path | None = None, gate_failures: list[str] | None = None) -> None:
    if json_path:
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text(json.dumps({"report": report, "run": result.to_dict()}, indent=2, default=str))
    if html_path:
        html_path.parent.mkdir(parents=True, exist_ok=True)
        html_path.write_text(to_html(report, title))
    if markdown_path:
        markdown_path.parent.mkdir(parents=True, exist_ok=True)
        markdown_path.write_text(to_markdown(report, title, gate_failures or []))


# ------------------------------------------------------------------------------- markdown --


def to_markdown(report: dict, title: str = "judgeflake report", gate_failures: list[str] | None = None) -> str:
    """The card as a GitHub-flavoured markdown comment (what the GitHub Action posts on a PR)."""
    m = report["meta"]
    failures = gate_failures or []
    status = "🔴 gate failed" if failures else "🟢 judges within limits"
    lines = [
        "<!-- judgeflake -->",
        f"### {title} · {status}",
        "",
        f"{m['cases']} cases · {m['judges']} judge{'s' if m['judges'] != 1 else ''} · {m['runs']} runs each · {m['calls']} verdicts",
        "",
        "| judge | self-agreement | flipped | vs human | reading |",
        "|---|---:|---:|---:|---|",
    ]
    for j in report["judges"]:
        acc = j["accuracy"]
        sa = f"{j['self_agreement']:.0%}" if j["self_agreement"] is not None else "—"
        vs_human = "{:.0%} (κ {:.2f})".format(acc["accuracy"], acc["kappa"]) if acc else "—"
        lines.append(f"| `{j['name']}` | {sa} | {len(j['flip_cases'])}/{j['cases']} | {vs_human} | {j['reading']} |")
    ag = report.get("agreement")
    if ag and ag["alpha_majority"] is not None:
        lines += ["", f"Judges agree with each other: alpha {ag['alpha_majority']:.2f} ({ag['alpha_reading']})."]
    gate = report.get("gate")
    if gate and gate["judges"]:
        lines += ["", f"Gate risk at pass mark {gate['pass_mark']:g}:"]
        for g in gate["judges"]:
            lines.append(f"- `{g['judge']}`: expected {g['expected']:.2f}, false alarm {g['false_alarm']:.0%}, safe mark {g['safe_mark']:.2f}")
    if report["flaky_cases"]:
        lines += ["", "<details><summary>Flakiest cases</summary>", "", "| case | disagreement | " + " | ".join(j["name"] for j in report["judges"]) + " | human |", "|---|---:|" + "---|" * len(report["judges"]) + "---|"]
        for f in report["flaky_cases"][:10]:
            cells = " | ".join(" ".join(_fmt_value(v) for v in f["verdicts"].get(j["name"], [])) for j in report["judges"])
            lines.append(f"| `{f['case_id']}` | {f['disagreement']:.0%} | {cells} | {_fmt_value(f['label']) if f['label'] is not None else '—'} |")
        lines += ["", "</details>"]
    if failures:
        lines += ["", "**Gate failures:** " + "; ".join(failures)]
    lines += ["", "<sub>judgeflake · self-agreement is the chance two runs give the same verdict on a case</sub>"]
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------- escalation --


def escalation(report: dict, min_disagreement: float = 0.0, include_misses: bool = True) -> list[dict]:
    """The cases a human should look at: everything the judges disagreed on, plus judge-vs-label misses."""
    rows: dict[str, dict] = {}
    cases = report["cases"]
    for f in stats_flaky_all(report):
        if f["disagreement"] > min_disagreement:
            c = cases.get(f["case_id"], {})
            rows[f["case_id"]] = {
                "case_id": f["case_id"],
                "reason": "judges disagree",
                "disagreement": f["disagreement"],
                "verdicts": {j: " ".join(_fmt_value(v) for v in vs) for j, vs in f["verdicts"].items()},
                "label": f["label"],
                "inputs": c.get("inputs"),
                "outputs": c.get("outputs"),
                "reference_outputs": c.get("reference_outputs"),
            }
    if include_misses:
        for j in report["judges"]:
            if not j["accuracy"]:
                continue
            for miss in j["accuracy"]["misses"]:
                cid = miss["case_id"]
                if cid in rows:
                    rows[cid]["reason"] = "judges disagree, and a judge misses the label"
                    continue
                c = cases.get(cid, {})
                rows[cid] = {
                    "case_id": cid,
                    "reason": f"{j['name']} says {_fmt_value(miss['judge'])}, label says {_fmt_value(miss['label'])}",
                    "disagreement": 0.0,
                    "verdicts": {j["name"]: _fmt_value(miss["judge"])},
                    "label": miss["label"],
                    "inputs": c.get("inputs"),
                    "outputs": c.get("outputs"),
                    "reference_outputs": c.get("reference_outputs"),
                }
    return sorted(rows.values(), key=lambda r: (-r["disagreement"], r["case_id"]))


def stats_flaky_all(report: dict) -> list[dict]:
    """Every case with any disagreement, not just the top N the card shows."""
    pooled: dict[str, dict[str, list]] = {j["name"]: dict(j.get("all_values", {})) for j in report["judges"]}
    labels = {cid: c["label"] for cid, c in report["cases"].items() if c.get("label") is not None}
    out = []
    for f in stats.flaky_cases(pooled, labels, top=len(pooled) or 1):
        out.append({"case_id": f.case_id, "disagreement": f.disagreement, "verdicts": f.verdicts, "label": f.label})
    return out


# -------------------------------------------------------------------------------- compare --


def compare(old: dict, new: dict) -> dict:
    """Drift between two reports: per judge, the cases whose majority verdict changed."""
    out = {"judges": [], "old": old["meta"], "new": new["meta"]}
    new_by = {j["name"]: j for j in new["judges"]}
    for oj in old["judges"]:
        nj = new_by.get(oj["name"])
        if not nj:
            continue
        common = [c for c in oj["majorities"] if c in nj["majorities"]]
        changed = [
            {"case_id": c, "old": oj["majorities"][c], "new": nj["majorities"][c]}
            for c in common
            if oj["majorities"][c] != nj["majorities"][c]
        ]
        out["judges"].append(
            {
                "name": oj["name"],
                "cases": len(common),
                "changed": changed,
                "drift_rate": round(len(changed) / len(common), 4) if common else 0.0,
                "flip_rate_old": oj["flip_rate"],
                "flip_rate_new": nj["flip_rate"],
                "self_agreement_old": oj.get("self_agreement"),
                "self_agreement_new": nj.get("self_agreement"),
                "accuracy_old": oj["accuracy"]["accuracy"] if oj["accuracy"] else None,
                "accuracy_new": nj["accuracy"]["accuracy"] if nj["accuracy"] else None,
            }
        )
    return out


def print_compare(cmp: dict, console: Console | None = None) -> None:
    con = console or Console()
    con.print()
    con.print(f"[bold]judgeflake compare[/bold] · {cmp['old']['created']} → {cmp['new']['created']}")
    t = Table(show_lines=False)
    t.add_column("judge", style="bold")
    t.add_column("cases", justify="right")
    t.add_column("verdicts changed", justify="right")
    t.add_column("flip rate", justify="right")
    t.add_column("vs human", justify="right")
    for j in cmp["judges"]:
        acc = "—" if j["accuracy_old"] is None else f"{j['accuracy_old']:.0%} → {j['accuracy_new']:.0%}"
        t.add_row(j["name"], str(j["cases"]), f"{len(j['changed'])} ({j['drift_rate']:.0%})", f"{j['flip_rate_old']:.0%} → {j['flip_rate_new']:.0%}", acc)
    con.print(t)
    for j in cmp["judges"]:
        if j["changed"]:
            con.print(f"\n[bold]{j['name']}[/bold] changed its mind on:")
            for c in j["changed"][:25]:
                con.print(f"  {c['case_id']}: {_fmt_value(c['old'])} → {_fmt_value(c['new'])}")
            if len(j["changed"]) > 25:
                con.print(f"  … and {len(j['changed']) - 25} more")
    con.print()
