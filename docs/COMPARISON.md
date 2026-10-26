# How mistrial compares

Checked 2026-10-25 against each project's README, design doc or docs page. If something here is wrong or out of date, open an issue and it gets fixed.

Projects: [judgecheck](https://github.com/adhammomen/judgecheck), [judgebench](https://github.com/fakkad/judgebench), [judgekeeper](https://github.com/judgekeeper/judgekeeper), [Assay](https://github.com/tap222/assay-evals) (`assay rejudge`), [LangSmith repetitions](https://docs.langchain.com/langsmith/repetition), [promptfoo `repeat`](https://www.promptfoo.dev/docs/configuration/reference).

## Feature by feature

| | mistrial | judgecheck | judgebench | judgekeeper | Assay rejudge | LangSmith / promptfoo |
|---|---|---|---|---|---|---|
| Calls the judge for you, N times | yes | no, you bring a ratings file | 2 runs (A/B order swap) | no | yes, `--times N` | yes, `num_repetitions` / `repeat` |
| Self-agreement of one judge across runs | yes | yes, from your file | no | no | yes ("splits with itself") | average and std dev per score only |
| Flip rate and the list of flipped cases | yes | per-item dispersion | no | no | yes | no |
| Agreement between two judges (kappa) | yes | yes | no, judge vs human only | no | vs saved verdicts | no |
| Agreement across many judges (Krippendorff's alpha) | nominal, interval | nominal, ordinal, interval, ratio | yes | no | no | no |
| Accuracy and kappa vs human labels | yes | yes | yes | yes, you label in a browser | no | no |
| Confidence intervals on the coefficients | no | yes, cluster bootstrap | no | no | no | no |
| Position bias (swap A and B) | no | yes, simulated probe | yes, the main feature | no | no | no |
| Verbosity, self-preference, formatting, sycophancy probes | no | yes | verbosity, self-enhancement, leniency | no | no | no |
| Calibration (ECE, reliability diagram) | no | yes | no | no | no | no |
| Drift: compare two saved runs | yes | no | no | "re-check" after a rule change | no | compare experiments in the UI |
| CI gate with exit code | yes, three gates | no | no | no | yes, `--min-agreement` | no |
| Takes your existing evaluator function | yes, LangSmith / openevals signature | no | no | reads result files from promptfoo, DeepEval, Inspect, MLflow | yes, `file.py:fn` | LangSmith: yes, its own |
| Built-in LLM judges | 11 providers + local Ollama | none | Anthropic | none | none | LangSmith: yes |
| Report | terminal, single-file HTML, JSON | markdown, JSON | HTML dashboard | browser page | pytest report, PR comment | hosted UI |
| Dependencies | typer, rich, httpx | numpy, matplotlib | – | – | pytest | SaaS / node |
| Try without an API key | `pipx run mistrial demo`, 10 s | clone and install, ~30 s | no | yes, on existing results | no | no |
| On PyPI | pending | yes | no | yes | yes | n/a |
| Activity (stars, last push) | new | 0, created 2026-10-25 | 0, dormant since 2026-02 | 0, 2026-10 | 23, active | large |

## What the others have that mistrial does not

Being honest about this is cheaper than being corrected in a comment thread.

- **judgecheck** is the serious statistics package. Alpha at every level, weighted and Fleiss kappa, ICC, Gwet's AC1, bootstrap confidence intervals, sample-size planning, five bias probes, calibration. If you need a defensible number for a paper or an audit, use it. It does not call any model; you produce the ratings and hand them over.
- **judgebench** is the position-bias tool. It runs every pair in both orders and reports how often the verdict depends on order. mistrial has no pairwise mode yet.
- **judgekeeper** solves a different problem: getting human labels in the first place. It opens a page where you mark pass or fail, then measures the judge against you, and helps you rewrite the rule. mistrial assumes you already have labels, or runs without them.
- **Assay** does re-judging inside a full regression-testing framework with pytest, PR comments and a McNemar test. If you already run Assay, `rejudge` is right there.
- **LangSmith** re-runs evaluators with `num_repetitions` and shows the mean and standard deviation per score in its UI. It does not compute agreement coefficients, list flipped cases, or compare judges to each other.

## What mistrial does that none of them do together

- One command against the evaluator you already have, with the judge called for you, repeated, across any mix of providers and plain functions.
- Self-agreement as the headline, because flip rate grows with the number of runs and self-agreement does not.
- The flakiest cases with every verdict visible, which is the thing you act on.
- Drift between two saved runs.
- A CI gate on judge reliability, not just on the score.
- A report you can email as one file.

## Roadmap items taken from this table

Position-bias swap test, confidence intervals on kappa and alpha, ordinal alpha, cost per run. In that order, by how often people are likely to ask.

## Proposed additions and who already has them

Checked 2026-10-25 against the same READMEs plus judge-drift-sentinel, haizelabs verdict and truescore.

| idea | status | closest elsewhere |
|---|---|---|
| 1. Rubric sensitivity: paraphrase the rubric three ways, count verdicts that change | **built in 0.2.0** (`rubric-check`) | papers only; Assay rewords the request, not the rubric |
| 2. Gate false-alarm calculator: how often CI goes red on no change | **built in 0.2.0** (`--pass-mark`, `gate-risk`) | judgecheck plans run counts (Spearman-Brown) |
| 3. Judge migration report with threshold translation | **built in 0.2.0** (`migrate`) | judge-drift-sentinel says judge vs system moved; Assay rejudge with a new judge |
| 4. Silent-upgrade sentinel on an anchor set | **built in 0.2.0** (`sentinel`), design credited to judge-drift-sentinel | borrow the design, credit it |
| 5. Parser audit: raw text next to parsed verdict | **built in 0.2.0** (on the card) | Assay counts "no verdict" apart from flips |
| 6. Adaptive re-ask: extra runs only where runs disagreed | **built in 0.2.0** (`--adaptive`) | verdict repeats a fixed k |
| 7. Escalation list of flaky cases to a human queue | **built in 0.2.0** (`escalate`, `--langsmith-dataset`) | LangSmith has annotation queues, nothing fills them by reliability |
| 8. Stabilized evaluator: wrap into a k-sample majority vote, same signature | **built in 0.2.0** (`--stabilize`, `mistrial.stabilize`) | haizelabs verdict builds voting judges as a framework |
| 9. Rubric reversal check: flip the criterion, confirm verdicts invert | **built in 0.2.0** (`rubric-check`) | EMNLP 2026 paper only |
| 10. GitHub Action with the card as a PR comment | **built in 0.2.0** (`action.yml`, `--markdown`) | Assay and promptfoo-action post eval results, not judge reliability |
| 11. Transitivity check for pairwise judges | no | TrustJudge paper only |

Ten of eleven are now in mistrial 0.2.0. The eleventh (transitivity) waits for a pairwise mode.

## Other close projects

- [judge-drift-sentinel](https://pypi.org/project/judge-drift-sentinel/): frozen human-labeled anchor set, tells you whether the judge or the system moved, exit codes. v0.1.0, PyPI upload pending.
- [verdict](https://github.com/haizelabs/verdict) (haizelabs): framework for compound judges with repeat and vote. 349 stars, last push 2025-11. Builds judges rather than measuring them.
- [truescore](https://pypi.org/project/truescore/): bias-corrected scores and confidence intervals for LLM-judged benchmarks from a results file. Statistics after the fact.
- [promptfoo-action](https://github.com/promptfoo/promptfoo-action): posts eval results on PRs. 77 stars.
