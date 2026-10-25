# Changelog

All notable changes to judgeflake are recorded here. The format follows [Keep a Changelog](https://keepachangelog.com/en/1.1.0/) and the project uses [Semantic Versioning](https://semver.org/).

## [Unreleased]

### Added
- Judges for every OpenAI-compatible provider: `gemini:`, `groq:`, `mistral:`, `deepseek:`, `xai:`, `together:`, `openrouter:`, `ollama:` (local, no key) and `custom:` with `JUDGEFLAKE_BASE_URL`. `<PROVIDER>_BASE_URL` overrides any base URL.
- `judgeflake providers` lists them with their env vars.

## [0.1.0] - 2026-10-25

First release.

### Added
- `judgeflake run`: run one or more judges over a cases file N times and print the reliability card.
- Judges: `openai:<model>`, `anthropic:<model>`, any Python `module:function` with the LangSmith / openevals signature, and deterministic `demo` judges.
- Cases from JSONL, JSON or CSV with common field aliases and an optional human `label`.
- Metrics: self-agreement, flip rate, per-case disagreement, pairwise percent agreement and Cohen's kappa, Krippendorff's alpha (nominal and interval, missing values allowed), accuracy and kappa against human labels.
- Built-in rubrics `correctness` and `quality`, or a prompt file.
- Terminal card, single-file HTML report (light and dark), JSON report.
- On-disk response cache with `--fresh` to bypass it.
- `judgeflake compare` for drift between two saved runs.
- CI gates: `--min-self-agreement`, `--max-flip-rate`, `--min-agreement`.
- `judgeflake demo`: three fake judges on twenty bundled cases, no API key.
- `judgeflake report` to re-render a saved run and `judgeflake rubrics` to list rubrics.

[Unreleased]: https://github.com/hemanth1999k/judgeflake/compare/v0.1.0...HEAD
[0.1.0]: https://github.com/hemanth1999k/judgeflake/releases/tag/v0.1.0
