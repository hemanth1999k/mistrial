<p align="center">
  <h1 align="center">mistrial</h1>
  <p align="center"><strong>Is your LLM judge flaky?</strong><br>
  Re-run any evaluator across runs, judge models and rubric wordings, and get a reliability card.<br>
  When the verdict can't be trusted, declare a mistrial.</p>
</p>

<p align="center">
  <a href="https://pypi.org/project/mistrial/"><img alt="PyPI" src="https://img.shields.io/pypi/v/mistrial.svg"></a>
  <a href="https://github.com/hemanth1999k/mistrial/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/hemanth1999k/mistrial/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/mistrial/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/mistrial.svg"></a>
  <a href="LICENSE"><img alt="MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/hemanth1999k/mistrial/main/docs/report.png" alt="mistrial HTML report: three judges, their self-agreement, agreement with each other, and the flakiest cases" width="820">
</p>

You ship AI features based on a score from an LLM judge. Ask that judge the same question twice and it changes its verdict on a tenth of your cases. Swap the model and it changes its mind on a quarter. One eval run is one sample from a noisy process, and almost nobody measures the noise.

mistrial does. Point it at the evaluator you already use, run it a few times across one or more judge models, and get a **reliability card**: how often the judge agrees with itself, how much judges agree with each other, how they match human labels, and the exact cases that flip.

## Quickstart

No API key needed for the demo.

```bash
pipx run mistrial demo
# or
pip install mistrial && mistrial demo
```

Three fake judges, twenty bundled cases, five runs each:

<p align="center"><img src="https://raw.githubusercontent.com/hemanth1999k/mistrial/main/docs/demo.gif" alt="mistrial terminal demo: the reliability card, the stabilized judge, rubric-check and migrate" width="900"></p>

```
mistrial · 20 cases · 3 judges · 5 runs each · 300 verdicts in 0.03s

Consistency (same judge, asked again)
┏━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━┓
┃ judge        ┃ self-agreement ┃ flipped cases ┃      vs human ┃ errors ┃ reading    ┃
┡━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━┩
│ demo         │            85% │    7/20 (35%) │ 100% (κ 1.00) │      0 │ flaky      │
│ demo-flaky   │            58% │   16/20 (80%) │  80% (κ 0.60) │      0 │ very flaky │
│ demo-lenient │            98% │     1/20 (5%) │  80% (κ 0.55) │      0 │ stable     │
└──────────────┴────────────────┴───────────────┴───────────────┴────────┴────────────┘

Agreement (different judges)
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━┳━━━━━━━━━━━━━┓
┃ pair                       ┃ cases ┃ agree ┃ kappa ┃ reading     ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━╇━━━━━━━━━━━━━┩
│ demo vs demo-flaky         │    20 │   80% │  0.60 │ substantial │
│ demo vs demo-lenient       │    20 │   80% │  0.55 │ moderate    │
│ demo-flaky vs demo-lenient │    20 │   60% │  0.20 │ slight      │
└────────────────────────────┴───────┴───────┴───────┴─────────────┘
Krippendorff's alpha: 0.44 on majority verdicts, 0.32 counting every run (nominal)
```

Look at `demo-lenient`: 98% self-agreement, "stable", and wrong on a fifth of the cases. Now look at `demo`: matches humans 100% of the time on average, and still flips on a third of the cases, so it will hand you a different release decision on Tuesday than on Monday. You need both numbers, and today you get neither.

With a real judge:

```bash
export OPENAI_API_KEY=...        # and/or ANTHROPIC_API_KEY
mistrial run cases.jsonl --judge openai:gpt-5-mini --judge anthropic:claude-sonnet-5-5 --runs 5
```

You get the card in the terminal, a single-file `report.html` you can send to anyone, and JSON for machines.

## Why

LLM-as-a-judge is how most teams grade their AI outputs now. The score it produces is treated as ground truth: it gates CI, decides which prompt wins, and goes on the dashboard. But the judge is itself a model sampling from a distribution.

- Judges have low intra-rater reliability: the same judge gives different scores to the same answer across runs ([Rating Roulette, EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1361.pdf)).
- Raw percent agreement overstates reliability by 30 to 40 points compared with chance-corrected kappa, across 21 judges ([Reliability without Validity, 2026](https://arxiv.org/abs/2606.19544)).
- None of the mainstream eval platforms re-run a judge and report its self-agreement. They report one score and stop.

The statistics for this problem are a century old. Test-retest reliability, Cohen's kappa and Krippendorff's alpha are standard in any field that uses human raters. They just never made it into a tool you can run in ten seconds against the evaluator you already have. That is the whole point of mistrial.

**Before "how good is my model?" you need "can I trust the number that told me?"**

## What you get

The card:

- **Self-agreement** per judge: the chance two runs give the same verdict on a case. Independent of how many runs you did. The headline number.
- **Flipped cases**: which cases had more than one verdict across runs.
- **Agreement between judges**: percent agreement and Cohen's kappa for every pair, Krippendorff's alpha across all of them.
- **Accuracy vs human labels**: accuracy and kappa against your `label` column, plus the misses.
- **Flakiest cases**: every verdict from every judge and run, as chips, ranked by disagreement. This is where you go to fix your rubric.
- **Parser audit**: how each reply was read (strict JSON, code fence, buried in prose, no verdict), so a parsing problem never gets blamed on the model.

The probes, each one answering a way judges go wrong in production:

- **Gate risk**: how often your CI gate would go red with nothing changed, and the pass mark or run count that fixes it.
- **Rubric sensitivity**: reword the rubric three ways with the same meaning and count the verdicts that move.
- **Rubric reversal**: ask for "incorrect" instead of "correct" and check the verdicts invert.
- **Migration**: switching judge models? See what moved and what your pass mark becomes on the new judge.
- **Sentinel**: freeze a baseline on anchor cases and catch a judge that changed under you, on a schedule.
- **Drift**: compare any two saved runs, case by case.

The fixes:

- **Stabilize**: wrap the judge in a K-sample majority vote, with the same evaluator signature, and measure it next to the raw one.
- **Adaptive re-ask**: two runs everywhere, extra runs only where the two disagreed. Same numbers, a fraction of the calls.
- **Escalate**: a CSV of the cases a human should look at, or push them to a LangSmith dataset.
- **CI gate and GitHub Action**: fail the build when a judge is flakier than you allow, and post the card on the pull request.

Responses are cached on disk, so re-rendering a report is free and `--fresh` is explicit.

## Usage

### Cases file

One case per line. JSONL, JSON or CSV.

```json
{"id": "capital-france", "inputs": "What is the capital of France?", "outputs": "Paris.", "reference_outputs": "Paris", "label": true}
{"id": "seven-times-eight", "inputs": "What is 7 × 8?", "outputs": "54", "reference_outputs": "56", "label": false}
```

`inputs` and `outputs` are required. `reference_outputs` and the human `label` are optional but unlock the correctness rubric and the accuracy column. Common aliases work: `input`, `question`, `answer`, `expected`, `ground_truth`.

### Judges

Three kinds, mixed freely with repeated `--judge`:

| spec | what it is |
|---|---|
| `openai:<model>` | OpenAI, needs `OPENAI_API_KEY` |
| `anthropic:<model>` | Anthropic, needs `ANTHROPIC_API_KEY` |
| `gemini:<model>` | Google Gemini, needs `GEMINI_API_KEY` |
| `groq:`, `mistral:`, `deepseek:`, `xai:`, `together:`, `openrouter:` | hosted OpenAI-compatible providers, each with its own `<PROVIDER>_API_KEY` |
| `ollama:<model>` | a local Ollama model, no key |
| `custom:<model>` | any OpenAI-compatible server: set `MISTRIAL_BASE_URL` (vLLM, LM Studio, llama.cpp, a proxy) |
| `module:function` | any Python function with the LangSmith / openevals signature |
| `demo`, `demo:flaky`, `demo:steady`, `demo:lenient` | deterministic fake judges, no key |

All LLM judges use the same built-in rubric and ask for a JSON verdict. `mistrial providers` lists the providers with their env vars and base URLs; `<PROVIDER>_BASE_URL` overrides a base URL, which is how you point `ollama:` at another machine or `openai:` at a proxy.

```bash
# GPT vs Claude vs a local model, five runs each
mistrial run cases.jsonl --judge openai:gpt-5-mini --judge anthropic:claude-sonnet-5-5 --judge ollama:llama3.2 --runs 5
```

### Your own evaluator

Any function that takes `inputs`, `outputs`, `reference_outputs` and returns a score, a bool, or `{"score": ...}`:

```python
# my_evals.py
def correctness(inputs, outputs, reference_outputs):
    return {"key": "correctness", "score": outputs.strip() == reference_outputs.strip()}
```

```bash
mistrial run cases.jsonl --judge my_evals:correctness --runs 5
```

### openevals and LangSmith

`create_llm_as_judge` returns exactly that kind of function, so an openevals judge needs no adapter:

```python
# my_evals.py
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
judge = create_llm_as_judge(prompt=CORRECTNESS_PROMPT, model="openai:gpt-5-mini")
```

```bash
mistrial run cases.jsonl --judge my_evals:judge --judge openai:gpt-5-mini --runs 5
```

Any LangSmith custom evaluator with the same signature works the same way. Export your dataset as JSONL and point at it.

### Rubrics

`--rubric correctness` (boolean, needs `reference_outputs`) and `--rubric quality` (1 to 5, no reference). Or `--rubric my_prompt.txt` with `{inputs}`, `{outputs}`, `{reference_outputs}` placeholders; the model must reply with JSON `{"score": ..., "reasoning": ...}`. Run `mistrial rubrics` to list them.

### Drift

```bash
mistrial run cases.jsonl --judge openai:gpt-5-mini --runs 5 --fresh
mistrial compare .mistrial/runs/20261025-101500.json .mistrial/runs/20261101-093000.json
```

Per judge: the cases whose majority verdict changed, and flip rate and accuracy before and after.

### CI gate

```bash
mistrial run cases.jsonl --judge openai:gpt-5-mini --runs 3 \
  --min-self-agreement 0.95 --max-flip-rate 0.10 --min-agreement 0.6 --quiet
```

Exit code 1 and a one-line reason on stderr when any gate fails. Use `--min-self-agreement` as the primary gate; it does not depend on `--runs`.

### Re-render

```bash
mistrial report .mistrial/runs/20261025-101500.json --html card.html --markdown card.md
```

No API calls. Re-renders the card from a saved run, as HTML and as the markdown the GitHub Action posts.

## The probes

Everything below works on the demo judges too, so you can try each one before spending a token: swap `openai:gpt-5-mini` for `demo:flaky`.

### Gate risk: will my CI gate go red on its own?

```bash
mistrial run cases.jsonl --judge openai:gpt-5-mini --runs 5 --pass-mark 0.8
# or on a saved run
mistrial gate-risk .mistrial/runs/20261025-101500.json --pass-mark 0.8 --runs-in-ci 1 --max-false-alarm 0.05
```

```
Gate risk (pass mark 0.8, 1 run per case in CI)
┃ judge       ┃ expected score ┃ 90% range ┃ false alarm ┃ safe mark ┃ fix                                 ┃
│ gpt-5-mini  │           0.84 │ 0.75–0.90 │         22% │      0.75 │ 5 runs per case, or mark ≤ 0.75     │
```

Each case's observed verdicts are treated as its distribution. The gate score is re-sampled 20,000 times and "false alarm" is how often it lands below your pass mark with nothing changed. The fix column gives the pass mark the judge clears 95% of the time, or the runs-per-case majority vote that gets the alarm rate under 5%. Pure Monte Carlo, so it is right for 10 cases as well as 1,000.

### Rubric sensitivity and reversal: is the score measuring the wording?

```bash
mistrial rubric-check cases.jsonl --judge openai:gpt-5-mini --rubric correctness
```

```
Rubric sensitivity (same meaning, different words)
┃ judge       ┃ cases ┃ verdicts that changed ┃ alpha across wordings ┃ per wording vs v0   ┃ reading               ┃
│ gpt-5-mini  │    20 │               5 (25%) │                  0.61 │ v1 15%, v2 20%, v3 10% │ measures the wording │

Rubric reversal (ask for INCORRECT instead of correct)
┃ judge       ┃ cases ┃ verdicts inverted ┃ unchanged ┃ reading                 ┃
│ gpt-5-mini  │    20 │          17 (85%) │         3 │ mostly reads the rubric │
```

The built-in rubrics ship with three rewordings that mean the same thing and a reversed form. The judge runs under each; `changed` is the share of cases whose verdict depends on which wording you used, and `inverted` is the share that flipped when the criterion flipped. Bring your own wordings with `--variant my_rubric_v2.txt`. Gates: `--max-changed 0.10`, `--min-inversion 0.90`. Function judges have no rubric, so they are skipped.

### Migrate: my judge model is being retired

```bash
mistrial run cases.jsonl --judge openai:gpt-4o --judge openai:gpt-5-mini --runs 3
mistrial migrate .mistrial/runs/20261025-101500.json --pass-mark 0.7
```

```
agree on 80% of cases · kappa 0.55 (moderate) · shift +0.20
your pass mark 0.7 on openai:gpt-4o is about 0.90 on openai:gpt-5-mini
4 cases get a different verdict: …
```

Both judges on the same cases, the shift in aggregate score, your pass mark translated so the pass rate stays the same, and the cases that get a different verdict. For 1-to-5 rubrics, `--threshold 4` maps a per-case cutoff by quantile. The translated mark keeps your pass rate. It does not make the judges agree, so read the changed cases.

### Sentinel: did the judge change under me?

```bash
# once: freeze a baseline on a small labeled anchor set that never changes
mistrial run anchors.jsonl --judge openai:gpt-5-mini --runs 3
mistrial sentinel baseline .mistrial/runs/20261025-101500.json

# on a schedule (nightly cron, CI): re-judge fresh and compare
mistrial sentinel check anchors.jsonl --judge openai:gpt-5-mini
```

Exit 0 `STABLE`, exit 2 `JUDGE_DRIFT` (more than `--max-changed` of anchor verdicts changed, or kappa vs human fell by more than `--max-kappa-drop`), exit 3 if the anchors themselves changed. The anchors never change, so any movement is the judge, which is how you catch a vendor's quiet model update before it blocks a release. Design credit: [judge-drift-sentinel](https://pypi.org/project/judge-drift-sentinel/).

### Stabilize: the fix, as a judge

```bash
mistrial run cases.jsonl --judge openai:gpt-5-mini --runs 5 --stabilize 5
```

Adds `openai:gpt-5-mini×5` next to the raw judge: the same judge asked five times with a majority vote (median for scores). The card shows both, so you see what K buys you before you pay for it. In code:

```python
from mistrial import stabilize

judge = stabilize("openai:gpt-5-mini", k=5)               # or stabilize(my_evaluator_fn, k=5)
judge(inputs="2+2?", outputs="4", reference_outputs="4")
# {"key": "openai:gpt-5-mini×5", "score": True, "comment": "...", "mistrial": {"votes": [...], "agreement": 1.0, "k": 5}}
```

Same signature as a LangSmith or openevals evaluator, so it drops in where the flaky one was.

### Adaptive re-ask: the same numbers for a third of the calls

```bash
mistrial run cases.jsonl --judge openai:gpt-5-mini --runs 7 --adaptive
```

Every case gets two runs. Only cases where the two disagreed (or a call failed) get the remaining five. Self-agreement is estimated from pairs within each case, so settled cases lose nothing. The card reports the calls skipped.

### Escalate: what should a human look at?

```bash
mistrial escalate .mistrial/runs/20261025-101500.json --out needs-human.csv
mistrial escalate .mistrial/runs/20261025-101500.json --langsmith-dataset needs-human   # pip install langsmith
```

One row per case where judges disagreed, flipped, or missed the human label, with every verdict, the case itself, and an empty `your_verdict` column for the reviewer. The LangSmith option creates or extends a dataset with the same rows so your team can annotate them there.

### GitHub Action: the card on every pull request

```yaml
# .github/workflows/mistrial.yml
on: [pull_request]
jobs:
  judge:
    runs-on: ubuntu-latest
    permissions: { contents: read, pull-requests: write }
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with: { python-version: "3.12" }
      - uses: hemanth1999k/mistrial@v0.2.0
        with:
          cases: evals/cases.jsonl
          judges: "openai:gpt-5-mini my_evals:correctness"
          runs: "3"
          adaptive: "true"
          min-self-agreement: "0.95"
          pass-mark: "0.8"
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

Posts the card as a comment on the PR (updated in place on later pushes), writes it to the job summary, uploads the HTML report as an artifact, and fails the job when a gate fails. Outside GitHub, `mistrial run … --markdown card.md` writes the same comment for whatever posts it.

## How the numbers are computed

Everything lives in `src/mistrial/stats.py`, is pure Python, and is tested against the textbook examples (Krippendorff's own data for alpha, the standard two-rater example for kappa). Nothing is a "grade".

| metric | formula | notes |
|---|---|---|
| self-agreement | mean over cases of the share of run pairs that agree | run-count independent; the headline |
| flip rate | cases with more than one distinct verdict ÷ cases | grows with `--runs`, compare at equal run counts |
| percent agreement | cases where both judges' majority verdicts match ÷ cases | raw, uncorrected |
| Cohen's kappa | (p_o − p_e) ÷ (1 − p_e) | chance-corrected, two raters |
| Krippendorff's alpha | 1 − D_o ÷ D_e from the coincidence matrix | nominal for labels, interval for scores, missing allowed, any number of raters |
| accuracy | judge majority verdict == human label ÷ labeled cases | plus kappa vs the label |
| disagreement (per case) | 1 − share of the overall majority across every judge and run | ranks the flakiest cases |
| gate false alarm | share of 20,000 re-samples of the aggregate score that fall below the pass mark | each case re-sampled from its observed verdicts; majority of `--runs-in-ci` |
| rubric sensitivity | cases whose majority verdict differs across wordings ÷ cases | plus alpha with wordings as raters |
| inversion rate | cases where the reversed rubric gave the opposite verdict ÷ cases | boolean rubrics |
| migration shift | mean score under the new judge − mean under the old | pass mark moves by the shift; per-case cutoffs map by quantile |

Readings use the Landis and Koch bands for kappa and alpha: slight, fair, moderate, substantial, almost perfect. They are labels for humans, not thresholds for machines. Set your own gates.

Known limits: kappa on a handful of cases is not evidence, and a judge that is perfectly consistent can be consistently wrong. The card shows both numbers so neither hides the other.

## How is this different

Other people have looked at this problem, and some of their tools go deeper than mistrial on statistics. The short version:

| | mistrial | judgecheck | judgebench | Assay rejudge | LangSmith repetitions |
|---|---|---|---|---|---|
| calls the judge for you, N times | yes, adaptive | no, you bring ratings | 2 runs, A/B swap | yes | yes |
| self-agreement and flipped cases | yes | from your file | no | yes | mean and std dev only |
| agreement between judges (kappa, alpha) | yes | yes, with confidence intervals | judge vs human | vs saved verdicts | no |
| position and verbosity bias probes | no | yes | yes | no | no |
| rubric sensitivity and reversal | yes | no | no | no | no |
| gate false-alarm rate | yes | no | no | no | no |
| judge migration with threshold translation | yes | no | no | partly | no |
| sentinel on anchor cases | yes | no | no | no | no |
| parser audit | yes | no | no | partly | no |
| stabilized evaluator (k-vote, same signature) | yes | no | no | no | no |
| escalation list for humans | yes | no | no | no | annotation queues exist |
| drift between saved runs | yes | no | no | no | UI compare |
| CI gate, GitHub Action with PR comment | yes | no | no | yes | no |
| takes your existing evaluator function | yes | no | no | yes | its own |
| try with no key | 10 s demo | ~30 s | no | no | no |

The full table, with what each of them does better, is in [docs/COMPARISON.md](docs/COMPARISON.md). judgecheck is the one to use when you need confidence intervals and bias probes for a paper or audit. mistrial is the one to run against the evaluator you already have, in a minute, and put in CI.

## Roadmap

- Position-bias swap test and transitivity check for pairwise judges
- Confidence intervals on kappa and alpha
- Cost per run per provider
- Pull a LangSmith dataset directly instead of exporting JSONL
- `mistrial serve` to browse reports on localhost

Open an issue if you want one of these first. See [CHANGELOG.md](CHANGELOG.md) for what has shipped.

## Contributing

Bug reports, rubric ideas, and adapters for other providers are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). The rule for new metrics: it needs a published formula, a test against a known example, and a sentence in this README saying what it does and does not mean.

## License

[MIT](LICENSE).
