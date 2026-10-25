# Contributing

Thanks for looking. judgeflake is small on purpose, so most contributions are quick.

## Setup

```bash
git clone https://github.com/hemanth1999k/judgeflake.git
cd judgeflake
python3 -m venv .venv && .venv/bin/pip install -e ".[dev]"
.venv/bin/pytest -q
.venv/bin/judgeflake demo
```

Python 3.10 or newer. No API key is needed for the tests or the demo; the OpenAI and Anthropic adapters are tested against a fake HTTP server.

## What is welcome

- Bug reports with the command you ran and the output you got.
- Adapters for other providers. Follow `LLMJudge._openai` in `src/judgeflake/judges.py`: plain `httpx`, retries on 429 and 5xx, one JSON verdict out.
- Rubrics. Add to `RUBRICS` in `src/judgeflake/prompts.py` with a one-line description.
- Loaders for other case formats, in `src/judgeflake/cases.py`.
- Report improvements, in `src/judgeflake/report.py`. Keep it one file with no external assets.

## Rules for metrics

A new number on the card needs all three:

1. A published formula, cited in the docstring.
2. A test in `tests/test_stats.py` against a known example from the literature or a hand-computed case.
3. A line in the README table saying what it means and what it does not mean.

No invented scores and no letter grades. The card shows numbers and a one-line reading of them.

## Pull requests

- One change per PR, with a test.
- Run `pytest -q` and `judgeflake demo --runs 3 --html none` before opening it.
- Add a line under `[Unreleased]` in `CHANGELOG.md`.
- Keep the PR description short: what changed and why.

## The demo GIF

`docs/demo.gif` is rendered from `docs/demo.tape` with [VHS](https://github.com/charmbracelet/vhs): `brew install vhs && vhs docs/demo.tape`. Re-render it when the card's layout changes.

## Releasing (maintainers)

1. Bump `version` in `pyproject.toml`, move `[Unreleased]` entries under the new version in `CHANGELOG.md`.
2. `git tag v0.x.y && git push --tags`.
3. `python -m build && python -m twine upload dist/*`.
4. Create the GitHub release from the tag with the changelog section as notes.
