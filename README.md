<p align="center">
  <h1 align="center">judgeflake</h1>
  <p align="center"><strong>Is your LLM judge flaky?</strong><br>
  Re-run any evaluator across runs and judge models and get a reliability card.</p>
</p>

<p align="center">
  <a href="https://pypi.org/project/judgeflake/"><img alt="PyPI" src="https://img.shields.io/pypi/v/judgeflake.svg"></a>
  <a href="https://github.com/hemanth1999k/judgeflake/actions/workflows/ci.yml"><img alt="CI" src="https://github.com/hemanth1999k/judgeflake/actions/workflows/ci.yml/badge.svg"></a>
  <a href="https://pypi.org/project/judgeflake/"><img alt="Python" src="https://img.shields.io/pypi/pyversions/judgeflake.svg"></a>
  <a href="LICENSE"><img alt="MIT" src="https://img.shields.io/badge/license-MIT-blue.svg"></a>
</p>

<p align="center">
  <img src="https://raw.githubusercontent.com/hemanth1999k/judgeflake/main/docs/report.png" alt="judgeflake HTML report: three judges, their self-agreement, agreement with each other, and the flakiest cases" width="820">
</p>

You ship AI features based on a score from an LLM judge. Ask that judge the same question twice and it changes its verdict on a tenth of your cases. Swap the model and it changes its mind on a quarter. One eval run is one sample from a noisy process, and almost nobody measures the noise.

judgeflake does. Point it at the evaluator you already use, run it a few times across one or more judge models, and get a **reliability card**: how often the judge agrees with itself, how much judges agree with each other, how they match human labels, and the exact cases that flip.

## Quickstart

No API key needed for the demo.

```bash
pipx run judgeflake demo
# or
pip install judgeflake && judgeflake demo
```

Three fake judges, twenty bundled cases, five runs each:

```
judgeflake · 20 cases · 3 judges · 5 runs each · 300 verdicts in 0.03s

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
judgeflake run cases.jsonl --judge openai:gpt-5-mini --judge anthropic:claude-sonnet-5-5 --runs 5
```

You get the card in the terminal, a single-file `report.html` you can send to anyone, and JSON for machines.

## Why

LLM-as-a-judge is how most teams grade their AI outputs now. The score it produces is treated as ground truth: it gates CI, decides which prompt wins, and goes on the dashboard. But the judge is itself a model sampling from a distribution.

- Judges have low intra-rater reliability: the same judge gives different scores to the same answer across runs ([Rating Roulette, EMNLP 2025](https://aclanthology.org/2025.findings-emnlp.1361.pdf)).
- Raw percent agreement overstates reliability by 30 to 40 points compared with chance-corrected kappa, across 21 judges ([Reliability without Validity, 2026](https://arxiv.org/abs/2606.19544)).
- None of the mainstream eval platforms re-run a judge and report its self-agreement. They report one score and stop.

The statistics for this problem are a century old. Test-retest reliability, Cohen's kappa and Krippendorff's alpha are standard in any field that uses human raters. They just never made it into a tool you can run in ten seconds against the evaluator you already have. That is the whole point of judgeflake.

**Before "how good is my model?" you need "can I trust the number that told me?"**

## What you get

- **Self-agreement** per judge: the chance two runs give the same verdict on a case. Independent of how many runs you did. The headline number.
- **Flipped cases**: which cases had more than one verdict across runs.
- **Agreement between judges**: percent agreement and Cohen's kappa for every pair, Krippendorff's alpha across all of them.
- **Accuracy vs human labels**: accuracy and kappa against your `label` column, plus the misses.
- **Flakiest cases**: every verdict from every judge and run, as chips, ranked by disagreement. This is where you go to fix your rubric.
- **Drift**: every run is saved. Compare this week's run with last week's and see which cases the judge changed its mind on.
- **CI gate**: fail the build when a judge is flakier than you allow.
- **Cache**: responses are stored on disk, so re-rendering a report is free and `--fresh` is explicit.

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
| `custom:<model>` | any OpenAI-compatible server: set `JUDGEFLAKE_BASE_URL` (vLLM, LM Studio, llama.cpp, a proxy) |
| `module:function` | any Python function with the LangSmith / openevals signature |
| `demo`, `demo:flaky`, `demo:steady`, `demo:lenient` | deterministic fake judges, no key |

All LLM judges use the same built-in rubric and ask for a JSON verdict. `judgeflake providers` lists the providers with their env vars and base URLs; `<PROVIDER>_BASE_URL` overrides a base URL, which is how you point `ollama:` at another machine or `openai:` at a proxy.

```bash
# GPT vs Claude vs a local model, five runs each
judgeflake run cases.jsonl --judge openai:gpt-5-mini --judge anthropic:claude-sonnet-5-5 --judge ollama:llama3.2 --runs 5
```

### Your own evaluator

Any function that takes `inputs`, `outputs`, `reference_outputs` and returns a score, a bool, or `{"score": ...}`:

```python
# my_evals.py
def correctness(inputs, outputs, reference_outputs):
    return {"key": "correctness", "score": outputs.strip() == reference_outputs.strip()}
```

```bash
judgeflake run cases.jsonl --judge my_evals:correctness --runs 5
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
judgeflake run cases.jsonl --judge my_evals:judge --judge openai:gpt-5-mini --runs 5
```

Any LangSmith custom evaluator with the same signature works the same way. Export your dataset as JSONL and point at it.

### Rubrics

`--rubric correctness` (boolean, needs `reference_outputs`) and `--rubric quality` (1 to 5, no reference). Or `--rubric my_prompt.txt` with `{inputs}`, `{outputs}`, `{reference_outputs}` placeholders; the model must reply with JSON `{"score": ..., "reasoning": ...}`. Run `judgeflake rubrics` to list them.

### Drift

```bash
judgeflake run cases.jsonl --judge openai:gpt-5-mini --runs 5 --fresh
judgeflake compare .judgeflake/runs/20261025-101500.json .judgeflake/runs/20261101-093000.json
```

Per judge: the cases whose majority verdict changed, and flip rate and accuracy before and after.

### CI gate

```bash
judgeflake run cases.jsonl --judge openai:gpt-5-mini --runs 3 \
  --min-self-agreement 0.95 --max-flip-rate 0.10 --min-agreement 0.6 --quiet
```

Exit code 1 and a one-line reason on stderr when any gate fails. Use `--min-self-agreement` as the primary gate; it does not depend on `--runs`.

### Re-render

```bash
judgeflake report .judgeflake/runs/20261025-101500.json --html card.html
```

No API calls. Re-renders the card from a saved run.

## How the numbers are computed

Everything lives in `src/judgeflake/stats.py`, is pure Python, and is tested against the textbook examples (Krippendorff's own data for alpha, the standard two-rater example for kappa). Nothing is a "grade".

| metric | formula | notes |
|---|---|---|
| self-agreement | mean over cases of the share of run pairs that agree | run-count independent; the headline |
| flip rate | cases with more than one distinct verdict ÷ cases | grows with `--runs`, compare at equal run counts |
| percent agreement | cases where both judges' majority verdicts match ÷ cases | raw, uncorrected |
| Cohen's kappa | (p_o − p_e) ÷ (1 − p_e) | chance-corrected, two raters |
| Krippendorff's alpha | 1 − D_o ÷ D_e from the coincidence matrix | nominal for labels, interval for scores, missing allowed, any number of raters |
| accuracy | judge majority verdict == human label ÷ labeled cases | plus kappa vs the label |
| disagreement (per case) | 1 − share of the overall majority across every judge and run | ranks the flakiest cases |

Readings use the Landis and Koch bands for kappa and alpha: slight, fair, moderate, substantial, almost perfect. They are labels for humans, not thresholds for machines. Set your own gates.

Known limits: kappa on a handful of cases is not evidence, and a judge that is perfectly consistent can be consistently wrong. The card shows both numbers so neither hides the other.

## How is this different

| | judgeflake | eval platforms (LangSmith, Braintrust, promptfoo, DeepEval) | judgecheck | judgekeeper |
|---|---|---|---|---|
| re-runs the judge and reports self-agreement | yes | no | yes | no |
| takes the evaluator function you already have | yes | n/a | no, you prepare a ratings file | no, you hand-label in a browser |
| agreement between judges (kappa, alpha) | yes | no | yes, with bootstrap CIs | partial |
| drift between saved runs | yes | no | no | re-check |
| CI gate | yes | some | no | no |
| shareable single-file HTML | yes | hosted UI | markdown | browser page |
| dependencies | typer, rich, httpx | heavy | numpy, matplotlib | – |
| time to first result | ten seconds, no key | account | ~30 s, clone | hand-label first |

judgecheck is more rigorous where it overlaps (confidence intervals, bias probes). judgeflake is the one you can run against your existing evaluator in a minute and put in CI. They are complementary.

## Roadmap

- Position-bias swap test for pairwise judges
- Cost per run per provider
- Pull a LangSmith dataset directly instead of exporting JSONL
- `judgeflake serve` to browse reports on localhost

Open an issue if you want one of these first. See [CHANGELOG.md](CHANGELOG.md) for what has shipped.

## Contributing

Bug reports, rubric ideas, and adapters for other providers are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md). The rule for new metrics: it needs a published formula, a test against a known example, and a sentence in this README saying what it does and does not mean.

## License

[MIT](LICENSE).
