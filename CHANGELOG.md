# Changelog

All notable changes to mistrial are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- `examples/hard.jsonl`: 64 cases that pit precision against reasonableness (rounding, units, hedges, extra facts, equivalent code, multi-step answers with a wrong step). Built because gpt-5.5 and the gpt-6 judges never flip on the tricky set; on this one every one of them does. README has the six-judge run.

## [0.2.1] - 2026-10-26

### Changed
- The source distribution contains only the package, tests, examples and the top-level docs. The wheel is unchanged.

## [0.2.0] - 2026-10-26

First public release. Ten additions over 0.1.0, each answering a documented way LLM judges go wrong in production.

### Added
- **Rubric sensitivity** (`mistrial rubric-check`): run the judge under three rewordings of the rubric that mean the same thing, report the share of verdicts that change and Krippendorff's alpha across wordings. Built-in rewordings for `correctness` and `quality`; bring your own with `--variant`.
- **Rubric reversal** (same command): ask with the criterion flipped and check that verdicts invert. Reports the cases where the judge gave the same answer to "is it correct?" and "is it incorrect?".
- **Gate false-alarm rate** (`--pass-mark` on `run`/`report`, and `mistrial gate-risk`): how often your CI gate would go red with nothing changed, by re-sampling each case's observed verdicts; plus a safe mark and the runs per case that bring the rate under 5%.
- **Judge migration** (`mistrial migrate`): old judge vs new judge on the same cases, with your pass mark translated to the new judge and a per-case cutoff mapped by quantile.
- **Sentinel** (`mistrial sentinel baseline` / `check`): freeze a run on anchor cases, re-judge fresh on a schedule, exit 2 on JUDGE_DRIFT. Design credit: judge-drift-sentinel.
- **Parser audit**: every LLM verdict keeps the raw reply and how it was read (strict JSON, fenced, embedded in prose, no verdict). The card shows the counts and examples.
- **Adaptive re-ask** (`--adaptive`): two runs everywhere, extra runs only where the two disagreed. The card reports the calls skipped.
- **Escalation** (`mistrial escalate`): a CSV of the cases a human should look at, with a column for their verdict; `--langsmith-dataset` pushes them to LangSmith.
- **Stabilized evaluator** (`--stabilize K` and `mistrial.stabilize()`): the judge wrapped in a K-sample majority vote with the same evaluator signature, measured next to the raw one.
- **GitHub Action** (`action.yml`): runs the card in CI, posts it as a PR comment, uploads the HTML report, fails the job on a gate. `--markdown` writes the same card anywhere.
- `examples/tricky.jsonl`: 32 labeled cases built to be hard to judge (hedges, rounding, unit conversions, extra details, equivalent code). The bundled `examples/cases.jsonl` stays as the easy set.
- `mistrial demo` says up front that its three judges are fake and flaky on purpose.
- `docs/img/compare.png`, a feature comparison against judgecheck, judgebench, Assay, judge-drift-sentinel and LangSmith/promptfoo, and a "Standing on shoulders" README section with the papers and repos this builds on.
- Judges for every OpenAI-compatible provider: `gemini:`, `groq:`, `mistral:`, `deepseek:`, `xai:`, `together:`, `openrouter:`, `ollama:` (local, no key) and `custom:` with `MISTRIAL_BASE_URL`. `<PROVIDER>_BASE_URL` overrides any base URL. `mistrial providers` lists them.
- An 88 second animated explainer video (`docs/explainer.mp4`, not committed) rendered from `docs/explainer/explainer.html`: the problem, why it slips through, the command, the real numbers, the fix. Narration script and re-render steps in `docs/explainer/SCRIPT.md`.

### Changed
- The project is called **mistrial**. Package, command, env vars (`MISTRIAL_BASE_URL`, `MISTRIAL_API_KEY`) and the `.mistrial/` directory all follow. It was briefly called judgeflake before any release.
- README, demo GIF and report screenshot now show real, current OpenAI judges (`gpt-5.4-nano`, `gpt-5.4-mini`, `gpt-5.5`) instead of the fake `demo` judges. The numbers in the README come from 480 live calls made on 2026-10-25.
- README opens with the problem, why nothing catches it today, and what mistrial does, with three diagrams in `docs/img/`.
- Saved runs are format version 2 (adds `adaptive`, `calls_saved`, and `text`/`parse` per verdict). Version 1 files still load.

## [0.1.0] - 2026-10-25

First tagged version, never published.

### Added
- `mistrial run`: run one or more judges over a cases file N times and print the reliability card.
- Judges: `openai:<model>`, `anthropic:<model>`, any Python `module:function` with the LangSmith / openevals signature, and deterministic `demo` judges.
- Cases from JSONL, JSON or CSV with common field aliases and an optional human `label`.
- Metrics: self-agreement, flip rate, per-case disagreement, pairwise percent agreement and Cohen's kappa, Krippendorff's alpha (nominal and interval, missing values allowed), accuracy and kappa against human labels.
- Built-in rubrics `correctness` and `quality`, or a prompt file.
- Terminal card, single-file HTML report (light and dark), JSON report.
- On-disk response cache with `--fresh` to bypass it.
- `mistrial compare` for drift between two saved runs.
- CI gates: `--min-self-agreement`, `--max-flip-rate`, `--min-agreement`.
- `mistrial demo`: three fake judges on twenty bundled cases, no API key.
- `mistrial report` to re-render a saved run and `mistrial rubrics` to list rubrics.

[Unreleased]: https://github.com/hemanth1999k/mistrial/compare/v0.2.1...HEAD
[0.2.1]: https://github.com/hemanth1999k/mistrial/compare/v0.2.0...v0.2.1
[0.2.0]: https://github.com/hemanth1999k/mistrial/compare/v0.1.0...v0.2.0
[0.1.0]: https://github.com/hemanth1999k/mistrial/releases/tag/v0.1.0
