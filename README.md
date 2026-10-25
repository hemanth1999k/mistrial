# judgeflake

**Is your LLM judge flaky?**

You gate releases on an LLM-as-a-judge score. Run the same judge twice and it changes its mind on a tenth of your cases. Swap the model and it changes its mind on a quarter. One eval run is one sample from a noisy process, and almost nobody measures the noise.

judgeflake does. Point it at any evaluator, run it a few times across one or more judge models, and get a reliability card:

- **Consistency** — how often the *same* judge flips its verdict when asked again (flip rate, per case)
- **Agreement** — how much *different* judges agree (percent agreement, Cohen's kappa, Krippendorff's alpha)
- **Accuracy** — how well each judge matches *human* labels, when you have them
- **Flakiest cases** — the exact cases everyone disagrees about, with every verdict
- **Drift** — which cases a judge changed its mind on since last week
- **CI gate** — fail the build when a judge is flakier than you allow

Works with LangSmith and openevals evaluators, plain Python functions, or OpenAI / Anthropic models with a built-in rubric. Terminal output, a shareable single-file HTML report, and JSON.

## Ten-second demo (no API keys)

```bash
pipx run judgeflake demo
# or: pip install judgeflake && judgeflake demo
```

Three fake judges, twenty bundled cases, five runs each. You get the full card and a `judgeflake-demo.html`.

## Real use

```bash
export OPENAI_API_KEY=...        # and/or ANTHROPIC_API_KEY
judgeflake run cases.jsonl \
  --judge openai:gpt-5-mini \
  --judge anthropic:claude-sonnet-5-5 \
  --runs 5 --rubric correctness
```

`cases.jsonl` is one case per line:

```json
{"id": "capital-france", "inputs": "What is the capital of France?", "outputs": "Paris.", "reference_outputs": "Paris", "label": true}
```

`inputs`, `outputs`, optional `reference_outputs`, optional human `label`. Common aliases (`input`, `answer`, `expected`, `ground_truth`) work. CSV and JSON too.

### Your own evaluator

Any function with the LangSmith / openevals signature:

```python
# my_evals.py
def correctness(inputs, outputs, reference_outputs):
    return {"key": "correctness", "score": outputs.strip() == reference_outputs.strip()}
```

```bash
judgeflake run cases.jsonl --judge my_evals:correctness --runs 5
```

An openevals judge works the same way, because `create_llm_as_judge(...)` returns exactly that kind of function:

```python
# my_evals.py
from openevals.llm import create_llm_as_judge
from openevals.prompts import CORRECTNESS_PROMPT
judge = create_llm_as_judge(prompt=CORRECTNESS_PROMPT, model="openai:gpt-5-mini")
```

```bash
judgeflake run cases.jsonl --judge my_evals:judge --judge openai:gpt-5-mini --runs 5
```

### Reading the card

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

- **self-agreement**: the chance that two runs of the same judge give the same verdict on a case. Independent of how many runs you did. The number to watch.
- **flipped cases**: how many cases had more than one distinct verdict across the runs. This one grows with `--runs`, so compare it only at the same run count.
- **vs human**: accuracy and Cohen's kappa against your `label` column.
- **kappa / alpha**: 1 is perfect, 0 is chance. Readings use the Landis and Koch bands (slight, fair, moderate, substantial, almost perfect).

Look at `demo-lenient`: 98% self-agreement, "stable", and wrong on a fifth of the cases. A judge that is perfectly consistent can still be consistently wrong, and a judge that matches humans 100% of the time on average will still hand you a different release decision on Tuesday than on Monday if it flips on a third of the cases. You want both numbers.

### Drift

Every run is saved under `.judgeflake/runs/`. Later:

```bash
judgeflake run cases.jsonl --judge openai:gpt-5-mini --runs 5 --fresh   # new calls, no cache
judgeflake compare .judgeflake/runs/20261025-101500.json .judgeflake/runs/20261101-093000.json
```

You get the cases whose majority verdict changed, per judge, plus flip rate and accuracy before and after.

### CI gate

```bash
judgeflake run cases.jsonl --judge openai:gpt-5-mini --runs 3 --max-flip-rate 0.10 --quiet
```

Exits 1 when any judge flips on more than 10% of cases, or when `--min-agreement` alpha isn't met. Responses are cached on disk, so re-running a report is free; use `--fresh` when you actually want new samples.

## Rubrics

`--rubric correctness` (boolean, needs `reference_outputs`) and `--rubric quality` (1 to 5, no reference). Or `--rubric my_prompt.txt` with `{inputs}`, `{outputs}`, `{reference_outputs}` placeholders; the model must reply with JSON `{"score": ..., "reasoning": ...}`.

## How the numbers are computed

All formulas live in `src/judgeflake/stats.py` and are tested against the textbook examples (Krippendorff's own data, the standard kappa example). Nothing is invented and nothing is a "grade".

- flip rate = cases with more than one distinct verdict across runs ÷ cases
- Cohen's kappa = (p_o − p_e) ÷ (1 − p_e)
- Krippendorff's alpha from the coincidence matrix, nominal for labels, interval for numbers, missing values allowed; reported on majority verdicts and again with every (judge, run) as its own rater

## Not yet

Position bias (pairwise swap tests), cost per provider, pulling datasets straight from LangSmith. Open an issue if you want one of these first.

## License

MIT.
