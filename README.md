<p align="center">
  <h1 align="center">Mistrial</h1>
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

```bash
pip install mistrial
export OPENAI_API_KEY=...        # or ANTHROPIC_API_KEY, GEMINI_API_KEY, GROQ_API_KEY ... see `mistrial providers`
mistrial run cases.jsonl -j openai:gpt-5.4-mini -j openai:gpt-5.5 --runs 5
```

No key? `pipx run mistrial demo` runs three fake judges that are built to be flaky on purpose.

## The problem

You ship AI features based on a score from an LLM judge. The score gates CI, decides which prompt wins, and goes on the dashboard. But the judge is a model sampling from a distribution. Ask it the same question twice and it changes its verdict on some of your cases. Swap the model and it changes its mind on more. Reword the rubric without changing its meaning and it moves again. One eval run is one sample from a noisy process, and almost nobody measures the noise.

<p align="center"><img src="https://raw.githubusercontent.com/hemanth1999k/mistrial/main/docs/img/problem.png" alt="Three ways a judge's verdict moves with nothing changed: across runs, across judges, across rubric wordings. Real replies from gpt-5.4-nano, gpt-5.4-mini and gpt-5.5." width="1000"></p>

These are real replies. On the first case, gpt-5.4-nano wrote "the added mention of 1811 does not contradict the core requirement" on one run and "1811 is not part of the reference, so the answer is incorrect" on the next. On the second, it worked out that 239,000 miles is 384,000 km and then failed the answer for being in miles. Each of those judges reports a clean score on its own. The flakiness only shows up when you ask again and compare.

The research agrees. The same judge gives different scores to the same answer across runs ([Rating Roulette, EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1361.pdf)). Raw percent agreement overstates reliability by 30 to 40 points compared with chance-corrected kappa, across 21 judges ([Reliability without Validity, 2026](https://arxiv.org/abs/2606.19544)).

## Why nothing catches it today

- **Eval platforms report one number and stop.** LangSmith, Braintrust, Langfuse and friends run the judge once per case. Some can repeat an experiment and show a standard deviation, none report self-agreement, kappa between judges, or which cases flipped.
- **The statistics exist, but not as a tool.** Test-retest reliability, Cohen's kappa and Krippendorff's alpha are a century old and standard in every field that uses human raters. They live in research notebooks, not in something you can run against your evaluator in a minute.
- **The projects that tried stop at the card.** judgecheck, judgebench, Assay rejudge and judge-drift-sentinel each cover a piece of this well, and [docs/COMPARISON.md](docs/COMPARISON.md) credits what each does better. None of them take your existing evaluator function, run in CI, or tell you what to do when the judge is flaky.

**Before "how good is my model?" you need "can I trust the number that told me?"**

## What Mistrial does

<p align="center"><img src="https://raw.githubusercontent.com/hemanth1999k/mistrial/main/docs/img/how.png" alt="You bring cases, any judge and a rubric. mistrial re-asks across runs, judges and rubric wordings. You get a reliability card, a verdict on the judge, and a stabilized judge as the fix." width="1000"></p>

Point it at the evaluator you already use: a model name, or the function you run in LangSmith or openevals. It asks the judge several times, across one or more judge models, and under rewordings of the rubric, and reports the spread: self-agreement, flipped cases, kappa and alpha between judges, accuracy against your human labels, and the exact cases that move. Then it does something about it: a gate for CI, a stabilized judge with the same signature, and a list of the cases a human should look at.

**The 90 second version.** Every number in it is from the runs below.

https://github.com/user-attachments/assets/3ff4653d-6fcd-4848-993f-df576a8c8044

## Real numbers

Three current OpenAI judges on `examples/tricky.jsonl`, 32 cases built to be hard to judge (hedges, rounding, unit conversions, extra details, equivalent code), five runs each. 480 live calls on 25 October 2026. Every number below is from that run, and the GIF replays those replies.

<p align="center"><img src="https://raw.githubusercontent.com/hemanth1999k/mistrial/main/docs/img/results.png" alt="gpt-5.4-nano 92% self-agreement and 81% vs human, gpt-5.4-mini 95% and 88%, gpt-5.5 100% and 91%. Gate false alarm 90% on gpt-5.4-mini. Rubric rewording moves 16% of gpt-5.4-nano verdicts. A 5-vote majority takes gpt-5.4-nano to 99% self-agreement." width="1000"></p>

<p align="center"><img src="https://raw.githubusercontent.com/hemanth1999k/mistrial/main/docs/demo.gif" alt="mistrial terminal demo with real OpenAI judges: the reliability card, the stabilized judge, rubric-check and migrate" width="900"></p>

```
mistrial · 32 cases · 3 judges · 5 runs each · 480 verdicts in 76.66s

Consistency (same judge, asked again)
┏━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┳━━━━━━━━┳━━━━━━━━━━━━━━━━━━┓
┃ judge               ┃ self-agreement ┃ flipped cases ┃      vs human ┃ errors ┃ reading          ┃
┡━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━╇━━━━━━━━╇━━━━━━━━━━━━━━━━━━┩
│ openai:gpt-5.4-nano │            92% │    5/32 (16%) │  81% (κ 0.63) │      0 │ noticeably flaky │
│ openai:gpt-5.4-mini │            95% │    4/32 (12%) │  88% (κ 0.75) │      0 │ mostly stable    │
│ openai:gpt-5.5      │           100% │    0/32 (0%)  │  91% (κ 0.81) │      0 │ stable           │
└─────────────────────┴────────────────┴───────────────┴───────────────┴────────┴──────────────────┘

Flakiest cases (everyone's verdicts pooled)
┏━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━┳━━━━━━━┓
┃ case                  ┃ disagreement ┃ openai:gpt-5.4-nano ┃ openai:gpt-5.4-mini ┃ openai:gpt-5.5 ┃ human ┃
┡━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━╇━━━━━━━┩
│ arigato               │          40% │ F F F F F           │ F T T T T           │ T T T T T      │ T     │
│ austen-wrong-year     │          40% │ T F F F T           │ T T T T F           │ F F F F F      │ F     │
│ largest-country-hedge │          40% │ F F F F F           │ T F F F F           │ T T T T T      │ F     │
│ moon-miles            │          33% │ F F F F F           │ F F F F F           │ T T T T T      │ T     │
│ percent-unit          │          33% │ T T T T T           │ F F F F F           │ F F F F F      │ F     │
└───────────────────────┴──────────────┴─────────────────────┴─────────────────────┴────────────────┴───────┘
```

What the run says, in order:

- **gpt-5.4-nano flips on 5 of 32 cases** with nothing changed, and agrees with the human label on 81%. gpt-5.5 never flips and agrees on 91%. The judges agree with each other at kappa 0.69 to 0.81, Krippendorff's alpha 0.75.
- **A CI gate at a 0.5 pass mark on gpt-5.4-mini goes red 90% of the time** with nothing changed, because its score sits in a 0.41 to 0.50 band. On gpt-5.4-nano, 100%. On gpt-5.5, 0%.
- **Rewording the rubric moves 16% of gpt-5.4-nano's verdicts.** gpt-5.5 moves none. Asked for "incorrect" instead of "correct", gpt-5.4-nano fails to invert on 4 cases.
- **Switching from gpt-5.4-nano to gpt-5.5 changes 6 verdicts**, and a 0.5 pass mark on the old judge is about 0.62 on the new one.
- **The fix works and you can see by how much.** gpt-5.4-nano wrapped in a 5-vote majority goes from 92% to 99% self-agreement, and adaptive re-asking got there with 177 fewer calls than asking five times everywhere.

### Stronger judges are not immune, they are harder to catch

On the 32 tricky cases, gpt-5.5 and the three gpt-6 judges never flip. So we wrote `examples/hard.jsonl`: 64 cases that pit precision against reasonableness. Rounding that is close enough or not, unit conversions, hedges that commit to the right answer and hedges that don't, extra facts that are right and extra facts that are wrong, equivalent code, multi-step answers with a wrong step and a right total. Six judges, five runs each, 1,920 live verdicts on 26 October 2026.

| judge | self-agreement | flipped | vs human | rubric rewording moves |
|---|---|---|---|---|
| gpt-5.4-nano | 93% | 9/64 | 81% (κ 0.63) | — |
| gpt-5.4-mini | 98% | 3/64 | 91% (κ 0.81) | — |
| gpt-5.5 | 99% | 1/64 | 100% (κ 1.00) | — |
| gpt-6-astra | 99% | 1/64 | 98% (κ 0.97) | 2% |
| gpt-6-luna | 98% | 2/64 | 97% (κ 0.94) | 5% |
| gpt-6.1-sol | 99% | 2/64 | 98% (κ 0.97) | 2% |

Every gpt-6 judge flipped, and the rationales show what they flip on:

- **gpt-6.1-sol on "6.02 x 10^23" for Avogadro's number**: fails it four runs out of five for "omits the required unit, per mole", then passes it because "per mole is implicit in the context."
- **gpt-6-astra on a shirt-price answer with the right total and wrong working**: fails it four times, "25% of $20 is $5, not $4", then passes it once because "the final price matches the reference."
- **gpt-6-luna on "two hydrogen atoms bonded to one oxygen atom" for H2O**: F, T, F, T, F across five runs. gpt-5.5 flipped on the same case.

The pattern: weak judges flip everywhere, strong judges flip exactly where a reasonable person could go either way, at 1 to 3% of cases. On an eval of a few hundred cases that is still several verdicts that depend on which run you got. And the bigger number is between tiers: Avogadro without "per mole" passes on every gpt-5 judge and fails on every gpt-6 judge; "32 degrees Fahrenheit" for the freezing point fails on both 5.4 judges and passes on everything above. Kappa between gpt-5.4-nano and gpt-6.1-sol is 0.60. Krippendorff's alpha across all six is 0.81, across the three gpt-6 judges 0.97.

One more thing the run caught: OpenAI's moderation rejected one of gpt-6-luna's 320 calls, on a question about `sum(xs) / len(xs)`, as a flagged prompt. The parser audit recorded it and marked that judge "fragile". A pipeline without error accounting would have counted that as a fail or skipped it silently.

Both sets ship with the package. `examples/tricky.jsonl` is where cheap judges fall apart; `examples/hard.jsonl` is where the expensive ones do.

You get the card in the terminal, a single-file `report.html` you can send to anyone, JSON for machines, and markdown for a pull request.

<p align="center">
  <img src="https://raw.githubusercontent.com/hemanth1999k/mistrial/main/docs/report.png" alt="mistrial HTML report: three judges, their self-agreement, agreement with each other, and the flakiest cases" width="820">
</p>

## What you get

The card:

- **Self-agreement** per judge: the chance two runs give the same verdict on a case. Independent of how many runs you did. The headline number.
- **Flipped cases**: which cases had more than one verdict across runs.
- **Agreement between judges**: percent agreement and Cohen's kappa for every pair, Krippendorff's alpha across all of them.
- **Accuracy vs human labels**: accuracy and kappa against your `label` column, plus the misses.
- **Flakiest cases**: every verdict from every judge and run, as chips, ranked by disagreement. This is where you go to fix your rubric.
- **Parser audit**: how each reply was read (strict JSON, code fence, buried in prose, no verdict), so a parsing problem never gets blamed on the model.

Every run is saved as JSON under `.mistrial/runs/` and every response is cached on disk, so the probes below can read a saved run, re-rendering is free, and `--fresh` is explicit. Beyond the card: [the probes](#the-probes) for the ways judges go wrong in production, [the fixes](#the-fixes) for the flakiness they find, and [the CI setup](#in-ci) to keep it that way.

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
mistrial run cases.jsonl --judge openai:gpt-5.4-mini --judge anthropic:claude-sonnet-5-5 --judge ollama:llama3.2 --runs 5
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
judge = create_llm_as_judge(prompt=CORRECTNESS_PROMPT, model="openai:gpt-5.4-mini")
```

```bash
mistrial run cases.jsonl --judge my_evals:judge --judge openai:gpt-5.4-mini --runs 5
```

Any LangSmith custom evaluator with the same signature works the same way. Export your dataset as JSONL and point at it.

### Rubrics

`--rubric correctness` (boolean, needs `reference_outputs`) and `--rubric quality` (1 to 5, no reference). Or `--rubric my_prompt.txt` with `{inputs}`, `{outputs}`, `{reference_outputs}` placeholders; the model must reply with JSON `{"score": ..., "reasoning": ...}`. Run `mistrial rubrics` to list them.

## The probes

Each probe answers one way judges go wrong in production. Most read a saved run, so they cost nothing after the `run` that produced it. Everything below works on the demo judges too, so you can try each one before spending a token: swap `openai:gpt-5.4-mini` for `demo:flaky`.

| Question | Command |
|---|---|
| Will my CI gate go red on its own? | `mistrial gate-risk .mistrial/runs/<run>.json --pass-mark 0.5` |
| Is the verdict just the rubric wording? | `mistrial rubric-check cases.jsonl -j openai:gpt-5.4-mini` |
| My judge model is being retired, what moves? | `mistrial migrate .mistrial/runs/<run>.json --pass-mark 0.5` |
| Did the judge change under me? | `mistrial sentinel baseline <run>.json`, then `mistrial sentinel check anchors.jsonl -j openai:gpt-5.4-mini` |
| What changed between two runs? | `mistrial compare <old-run>.json <new-run>.json` |
| Which cases need a human? | `mistrial escalate .mistrial/runs/<run>.json --out needs-human.csv` |

### Gate risk: will my CI gate go red on its own?

```bash
mistrial run examples/tricky.jsonl --judge openai:gpt-5.4-mini --runs 5 --pass-mark 0.5
# or on a saved run
mistrial gate-risk .mistrial/runs/20261025-101500.json --pass-mark 0.5 --runs-in-ci 1 --max-false-alarm 0.05
```

```
Gate risk (pass mark 0.5, 1 run per case in CI)
┃ judge               ┃ expected score ┃ 90% range ┃ false alarm ┃ safe mark ┃ fix                               ┃
│ openai:gpt-5.4-nano │           0.38 │ 0.34–0.44 │        100% │      0.34 │ already below the mark on average │
│ openai:gpt-5.4-mini │           0.46 │ 0.41–0.50 │         90% │      0.41 │ already below the mark on average │
│ openai:gpt-5.5      │           0.50 │ 0.50–0.50 │          0% │      0.50 │ fine as is                        │
```

Each case's observed verdicts are treated as its distribution. The gate score is re-sampled 20,000 times and "false alarm" is how often it lands below your pass mark with nothing changed. The fix column gives the pass mark the judge clears 95% of the time, or the runs-per-case majority vote that gets the alarm rate under 5%. Pure Monte Carlo, so it is right for 10 cases as well as 1,000.

### Rubric sensitivity and reversal: is the score measuring the wording?

```bash
mistrial rubric-check examples/tricky.jsonl --judge openai:gpt-5.4-nano --judge openai:gpt-5.5 --rubric correctness
```

```
Rubric sensitivity (same meaning, different words)
┃ judge               ┃ cases ┃ verdicts that changed ┃ alpha across wordings ┃ per wording vs v0      ┃ reading              ┃
│ openai:gpt-5.4-nano │    32 │               5 (16%) │                  0.81 │ v1 12%, v2 12%, v3 6%  │ measures the wording │
│ openai:gpt-5.5      │    32 │                0 (0%) │                  1.00 │ v1 0%, v2 0%, v3 0%    │ robust               │

Rubric reversal (ask for INCORRECT instead of correct)
┃ judge               ┃ cases ┃ verdicts inverted ┃ unchanged ┃ reading                 ┃
│ openai:gpt-5.4-nano │    32 │          28 (88%) │         4 │ mostly reads the rubric │
│ openai:gpt-5.5      │    32 │         32 (100%) │         0 │ reads the rubric        │
```

The built-in rubrics ship with three rewordings that mean the same thing and a reversed form. The judge runs under each; `changed` is the share of cases whose verdict depends on which wording you used, and `inverted` is the share that flipped when the criterion flipped. Bring your own wordings with `--variant my_rubric_v2.txt`. Gates: `--max-changed 0.10`, `--min-inversion 0.90`. Function judges have no rubric, so they are skipped.

### Migrate: my judge model is being retired

```bash
mistrial run examples/tricky.jsonl --judge openai:gpt-5.4-nano --judge openai:gpt-5.5 --runs 3
mistrial migrate .mistrial/runs/20261025-101500.json --pass-mark 0.5
```

```
agree on 81% of cases · kappa 0.62 (substantial) · shift +0.12
your pass mark 0.5 on openai:gpt-5.4-nano is about 0.62 on openai:gpt-5.5
6 cases get a different verdict:
  arigato: F → T
  moon-miles: F → T
  percent-unit: T → F
  …
```

Both judges on the same cases, the shift in aggregate score, your pass mark translated so the pass rate stays the same, and the cases that get a different verdict. For 1-to-5 rubrics, `--threshold 4` maps a per-case cutoff by quantile. The translated mark keeps your pass rate. It does not make the judges agree, so read the changed cases.

### Sentinel: did the judge change under me?

```bash
# once: freeze a baseline on a small labeled anchor set that never changes
mistrial run anchors.jsonl --judge openai:gpt-5.4-mini --runs 3
mistrial sentinel baseline .mistrial/runs/20261025-101500.json

# on a schedule (nightly cron, CI): re-judge fresh and compare
mistrial sentinel check anchors.jsonl --judge openai:gpt-5.4-mini
```

Exit 0 `STABLE`, exit 2 `JUDGE_DRIFT` (more than `--max-changed` of anchor verdicts changed, or kappa vs human fell by more than `--max-kappa-drop`), exit 3 if the anchors themselves changed. The anchors never change, so any movement is the judge, which is how you catch a vendor's quiet model update before it blocks a release. Design credit: [judge-drift-sentinel](https://pypi.org/project/judge-drift-sentinel/).

### Drift: what changed between these two runs?

```bash
mistrial run cases.jsonl --judge openai:gpt-5.4-mini --runs 5 --fresh
mistrial compare .mistrial/runs/20261025-101500.json .mistrial/runs/20261101-093000.json
```

Per judge: the cases whose majority verdict changed, and flip rate and accuracy before and after. Sentinel is this, on a schedule, against anchors that never change; `compare` is the ad hoc version for any two runs you have.

### Escalate: what should a human look at?

```bash
mistrial escalate .mistrial/runs/20261025-101500.json --out needs-human.csv
mistrial escalate .mistrial/runs/20261025-101500.json --langsmith-dataset needs-human   # pip install langsmith
```

One row per case where judges disagreed, flipped, or missed the human label, with every verdict, the case itself, and an empty `your_verdict` column for the reviewer. The LangSmith option creates or extends a dataset with the same rows so your team can annotate them there.

## The fixes

| What | Command |
|---|---|
| Majority vote of K samples, measured next to the raw judge | `mistrial run cases.jsonl -j openai:gpt-5.4-mini --runs 5 --stabilize 5` |
| The same numbers for a fraction of the calls | `mistrial run cases.jsonl -j openai:gpt-5.4-mini --runs 7 --adaptive` |

### Stabilize: the fix, as a judge

```bash
mistrial run cases.jsonl --judge openai:gpt-5.4-mini --runs 5 --stabilize 5
```

Adds `openai:gpt-5.4-mini×5` next to the raw judge: the same judge asked five times with a majority vote (median for scores). The card shows both, so you see what K buys you before you pay for it. In code:

```python
from mistrial import stabilize

judge = stabilize("openai:gpt-5.4-mini", k=5)               # or stabilize(my_evaluator_fn, k=5)
judge(inputs="2+2?", outputs="4", reference_outputs="4")
# {"key": "openai:gpt-5.4-mini×5", "score": True, "comment": "...", "mistrial": {"votes": [...], "agreement": 1.0, "k": 5}}
```

Same signature as a LangSmith or openevals evaluator, so it drops in where the flaky one was.

### Adaptive re-ask: the same numbers for a third of the calls

```bash
mistrial run cases.jsonl --judge openai:gpt-5.4-mini --runs 7 --adaptive
```

Every case gets two runs. Only cases where the two disagreed (or a call failed) get the remaining five. Self-agreement is estimated from pairs within each case, so settled cases lose nothing. The card reports the calls skipped.

## In CI

### The gate

```bash
mistrial run cases.jsonl --judge openai:gpt-5.4-mini --runs 3 \
  --min-self-agreement 0.95 --max-flip-rate 0.10 --min-agreement 0.6 --quiet
```

Exit code 1 and a one-line reason on stderr when any gate fails. Use `--min-self-agreement` as the primary gate; it does not depend on `--runs`. Add `--pass-mark` and the card also reports the [gate risk](#gate-risk-will-my-ci-gate-go-red-on-its-own), so you learn the false-alarm rate of the gate before it fires.

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
      - uses: hemanth1999k/mistrial@v0.2.1
        with:
          cases: evals/cases.jsonl
          judges: "openai:gpt-5.4-mini my_evals:correctness"
          runs: "3"
          adaptive: "true"
          min-self-agreement: "0.95"
          pass-mark: "0.8"
        env:
          OPENAI_API_KEY: ${{ secrets.OPENAI_API_KEY }}
```

Posts the card as a comment on the PR (updated in place on later pushes), writes it to the job summary, uploads the HTML report as an artifact, and fails the job when a gate fails.

### Elsewhere

```bash
mistrial run cases.jsonl --judge openai:gpt-5.4-mini --runs 3 --markdown card.md --html card.html
mistrial report .mistrial/runs/20261025-101500.json --html card.html --markdown card.md   # from a saved run, no API calls
```

The markdown is the same comment the Action posts, for whatever posts it in GitLab, Buildkite or a Slack hook. `report` re-renders any saved run without a single API call.

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

## Commands

| What you want to do | Command |
|---|---|
| Try it, no API key | `mistrial demo` |
| Judge your own cases, five runs | `mistrial run cases.jsonl -j openai:gpt-5.4-mini --runs 5` |
| Compare two judge models | `mistrial run cases.jsonl -j openai:gpt-5.5 -j anthropic:claude-sonnet-5-5 --runs 5` |
| Use your own evaluator function | `mistrial run cases.jsonl -j my_evals:correctness --runs 5` |
| Fail CI when the judge is flaky | `mistrial run cases.jsonl -j openai:gpt-5.4-mini --min-self-agreement 0.95 --quiet` |
| Fix it: majority vote of five samples | `mistrial run cases.jsonl -j openai:gpt-5.4-mini --stabilize 5` |
| Re-render a saved run as HTML or markdown | `mistrial report .mistrial/runs/<run>.json --html card.html` |
| List providers and their env vars | `mistrial providers` |
| List rubrics | `mistrial rubrics` |
| Every flag | `mistrial --help`, `mistrial run --help` |

The [probes](#the-probes) (gate risk, rubric wording, migration, sentinel, drift, escalate) each have a command of their own, and all of them work on the demo judges.

## How is this different

Other people have looked at this problem, and some of their tools go deeper than Mistrial on statistics. judgecheck is the one to use when you need confidence intervals and bias probes for a paper or audit. judgebench owns position bias. Assay is a full regression framework. Mistrial is the one to run against the evaluator you already have, in a minute, and put in CI.

<p align="center"><img src="https://raw.githubusercontent.com/hemanth1999k/mistrial/main/docs/img/compare.png" alt="Feature comparison: Mistrial, judgecheck, judgebench, Assay rejudge, judge-drift-sentinel, LangSmith/promptfoo, across measuring the judge, protecting the pipeline, fixing it, and getting started." width="1000"></p>

The full table, with links and what each of them does better, is in [docs/COMPARISON.md](docs/COMPARISON.md). If a cell is wrong, open an issue and it gets fixed.

## Credits, and repos to check out

Mistrial puts century-old reliability statistics in front of a new kind of rater. None of the ideas are new; the packaging is. This section is the credit: the papers the methods come from, the posts that shaped the framing, and the projects that got to parts of this first. All worth following if you work on this.

**Papers to read**

- [Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685), Zheng et al., 2023. The paper that named the practice and first measured judge agreement with humans, and the position, verbosity and self-enhancement biases that come with it.
- [Large Language Models are not Fair Evaluators](https://arxiv.org/abs/2305.17926), Wang et al., 2023. Swap the order of two answers and the verdict changes. Position bias is on Mistrial's roadmap because of this paper.
- [Large Language Models are Inconsistent and Biased Evaluators](https://arxiv.org/abs/2405.01724), Stureborg et al., 2024. Run-to-run inconsistency of the same judge, measured. This is the problem the card's self-agreement column reports.
- [Rating Roulette: Self-Inconsistency in LLM-as-a-Judge Frameworks](https://aclanthology.org/2025.findings-emnlp.1361.pdf), EMNLP 2025 Findings ([arXiv](https://arxiv.org/abs/2510.27106)). The same judge gives different scores to the same answer across runs; the paper that made "ask again" a method.
- [The Coin Flip Judge? Reliability and Bias in LLM-as-a-Judge Evaluation](https://arxiv.org/abs/2606.13685), Yagubyan et al., 2026. Identical reruns flip the pairwise winner about 14% of the time, temperature zero helps but does not fix it, and it takes around eleven trials for a majority vote to settle. The closest paper to what the card measures.
- [All Verdicts are Not Equal: Rethinking LLM Judge Reliability](https://arxiv.org/abs/2610.12083), Kumar et al., 2026. Verdicts change across identical reruns at temperature zero, and the most consistent judge was consistent mainly by repeating wrong answers. Why the card shows self-agreement and accuracy side by side.
- [Judge Reliability Harness](https://arxiv.org/abs/2603.05399), Dev et al., 2026. Stress-tests judges with reformatting, paraphrasing, verbosity and flipped labels; no judge was reliable across all four benchmarks. The rubric probes in `rubric-check` are the same instinct applied to the rubric instead of the answer.
- [An Empirical Study of LLM-as-a-Judge: How Design Choices Impact Evaluation Reliability](https://arxiv.org/abs/2506.13639), Yamauchi et al., 2025. Clear criteria matter most, and sampling beats deterministic decoding for agreement with humans.
- [Reliability without Validity](https://arxiv.org/abs/2606.19544), 2026. Raw percent agreement overstates judge reliability by 30 to 40 points against chance-corrected kappa, across 21 judges. Why the card reports kappa and alpha, not just agreement.
- [Who Drifted: the System or the Judge?](https://arxiv.org/abs/2606.15474), Li et al., 2026. A frozen human-labeled anchor set re-scored on a schedule, with a sequential test on the judge-versus-human gap, to tell a judge update from a real regression. The design behind `sentinel`, together with judge-drift-sentinel below.
- [LLM Judges as Raters](https://arxiv.org/abs/2608.29517), Sunkavalli, 2026. Treats judges as human raters are treated in psychometrics: severity, halo, reliability, and the shift when the model version changes.
- [JudgeBench: A Benchmark for Evaluating LLM-based Judges](https://arxiv.org/abs/2410.12784), Tan et al., 2024. Hard, verifiable pairwise cases where a judge only gets credit if it is right in both orders. A different thing from the judgebench CLI below, which shares the name.
- [Replacing Judges with Juries](https://arxiv.org/abs/2404.18796), Verga et al. (Cohere), 2024. A panel of smaller, diverse judges beats one large judge and costs less. The jury mode planned for v0.3 starts here.
- [Who Validates the Validators?](https://arxiv.org/abs/2404.12272), Shankar et al., 2024. EvalGen: humans and LLM evaluators drift apart unless the rubric is aligned with human judgment case by case. The reason `escalate` exists.
- [G-Eval](https://arxiv.org/abs/2303.16634), Liu et al., 2023, and [Prometheus 2](https://arxiv.org/abs/2405.01535), Kim et al., 2024. Two ways to make a judge: prompt a frontier model with a rubric, or train an open model to be one. Mistrial measures either.
- [Judging the Judges](https://arxiv.org/abs/2406.12624), Thakur et al., 2024, and [Justice or Prejudice?](https://arxiv.org/abs/2410.02736), Ye et al., 2024. Systematic audits of judge alignment and of twelve bias types. Good checklists for probes Mistrial does not have yet.
- [Agent-as-a-Judge](https://arxiv.org/abs/2410.10934), Zhuge et al., 2024. Judging multi-step agent trajectories, where reliability is worst and least measured.
- [A Survey on LLM-as-a-Judge](https://arxiv.org/abs/2411.15594), Gu et al., 2024. The map of the field.
- Classics: [Cohen's kappa](https://en.wikipedia.org/wiki/Cohen%27s_kappa) (1960), [Krippendorff's alpha](https://en.wikipedia.org/wiki/Krippendorff%27s_alpha), and the [Landis and Koch](https://en.wikipedia.org/wiki/Cohen%27s_kappa#Interpreting_magnitude) bands the card uses as readings. Every field with human raters has used these for decades.

**Posts worth reading**

- [Using LLM-as-a-Judge for Evaluation: A Complete Guide](https://hamel.dev/blog/posts/llm-judge/), Hamel Husain. Start with one domain expert making pass/fail calls with written critiques, then build the judge from those. The labels Mistrial measures against should come from this process.
- [Evaluating the Effectiveness of LLM-Evaluators](https://eugeneyan.com/writing/llm-evaluators/), Eugene Yan. A review of two dozen papers on judges: scoring methods, metrics, alignment, critiques.
- [LLM-as-Judge Is the New Flaky Test](https://dikrana.dev/blog/llm-as-judge-is-flaky/), Dik Rana, 2026. Treat a green judge verdict as a hypothesis; re-run, perturb, use panels. The framing this README borrows.
- [LLM-as-Judge Drift: When Your Evaluator Upgrades and All Your Numbers Move](https://tianpan.co/blog/2026/04/23/llm-judge-drift-evaluator-upgrade-phantom-regression), Tian Pan, 2026. Pin judge versions, record judge metadata on every run, calibrate against a second judge.

**Repos to check out**

- [judgecheck](https://github.com/adhammomen/judgecheck) (adhammomen): the serious statistics package for judge ratings, with confidence intervals, five bias probes and calibration. Bring your ratings file.
- [judgecheck](https://github.com/theruviparambil/judgecheck) (theruviparambil, the one on PyPI): a different project with the same name. Panel reliability from rating files: Fleiss' kappa, how many effective independent judges you really have, coincident errors between raters, and a keep/review/drop triage per rater.
- [judgebench](https://github.com/fakkad/judgebench): position-bias testing by running every pair in both orders.
- [judge-kappa](https://github.com/williamcaban/judge-kappa-eval) (on PyPI as `judge-kappa`): Krippendorff's alpha with confidence intervals, Cohen's kappa, panel and jury judges, pairwise preference and Elo, position and verbosity bias detection. The closest package in spirit; it builds and scores panels, Mistrial re-runs the one you have.
- [llm-judge-bench](https://github.com/hankimis/llm-judge-bench): one command runs five frontier judges against ground truth and reports truth accuracy, self-consistency, and verbosity, position and self-preference bias.
- [judge-drift-sentinel](https://github.com/homayoun-safarpour/judge-drift-sentinel): a frozen human-labeled anchor set and a baseline, so you can tell whether the judge or your system moved. Mistrial's `sentinel` borrows this design, with credit.
- [judgekeeper](https://github.com/judgekeeper/judgekeeper): get human labels in the first place, in a browser, then measure the judge against them.
- [Assay](https://github.com/tap222/assay-evals): behavioral regression testing for AI apps with a pytest plugin, PR comments, and a `rejudge` command.
- [verdict](https://github.com/haizelabs/verdict) (Haize Labs): a framework for building compound judges with repeat-and-vote, debate and verification. Builds the judge that Mistrial then measures.
- [llm-council](https://github.com/karpathy/llm-council) (Andrej Karpathy): several models answer, review each other anonymously, and a chairman synthesizes. The council idea applied to answering; Mistrial's planned jury mode applies it to judging.
- [openevals](https://github.com/langchain-ai/openevals) and [LangSmith](https://docs.langchain.com/langsmith/evaluation): ready-made evaluators and the evaluator signature Mistrial speaks, plus Align Evals for calibrating a judge against human corrections. Point `--judge module:function` at any of them.
- [TruLens](https://github.com/truera/trulens): evaluation and tracing with a `Jury` aggregator for feedback functions, and an open pull request adding a repeated-trial `SelfConsistency` wrapper. The same fix as `stabilize`, inside that framework.
- [autoevals](https://github.com/braintrustdata/autoevals) (Braintrust), [DeepEval](https://github.com/confident-ai/deepeval), [Ragas](https://github.com/explodinggradients/ragas), [promptfoo](https://github.com/promptfoo/promptfoo), [Phoenix](https://github.com/Arize-ai/phoenix): the eval frameworks most judges live in. Wrap their scorers as a function and Mistrial will re-run them.
- [Inspect](https://github.com/UKGovernmentBEIS/inspect_ai) (UK AI Security Institute), [lm-evaluation-harness](https://github.com/EleutherAI/lm-evaluation-harness), [AlpacaEval](https://github.com/tatsu-lab/alpaca_eval), [FastChat](https://github.com/lm-sys/FastChat) (MT-Bench): the benchmark harnesses where model-graded evaluation grew up.

If your project belongs here and is missing, open a PR with one line.

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
