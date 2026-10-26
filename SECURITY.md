# Security

mistrial is a command-line tool. It has no server, no accounts, no telemetry, and makes no network calls except to the LLM providers you name.

## What it touches

- **API keys** are read from environment variables (`OPENAI_API_KEY`, `ANTHROPIC_API_KEY`, `<PROVIDER>_API_KEY`) and sent only in the request header to that provider's base URL. They are never written to disk, never put in a saved run, and never shown in a report. `mistrial providers` prints the env var names, not the values.
- **Base URLs** can be overridden with `<PROVIDER>_BASE_URL` or `MISTRIAL_BASE_URL`. Your key goes wherever that points, so set it only to hosts you control or trust. TLS certificates are verified (httpx defaults).
- **Your data**: case inputs, outputs, references and labels are sent to the judge providers you choose, and stored locally in `.mistrial/cache.sqlite` and `.mistrial/runs/*.json`, along with the model's raw replies. Treat that directory like the data it holds; it is in `.gitignore` by default.
- **Reports**: the HTML report has no JavaScript and escapes all content. The markdown card contains judge names, case ids and numbers only, escaped, never case text or model text. The escalation CSV neutralises cells that start with `=`, `+`, `-` or `@` so a model reply cannot run as a spreadsheet formula.
- **`module:function` judges** import and run your Python code from the current directory, the same way pytest does. Only point mistrial at code you would run yourself.
- **The GitHub Action** passes every input through environment variables, never into script text, and posts comments with the token you give it (`github.token` by default, needing `pull-requests: write`). Secrets are not available to workflows triggered by pull requests from forks, which is GitHub's default and the right one.
- **LangSmith push** (`escalate --langsmith-dataset`) uses the official `langsmith` client and your `LANGSMITH_API_KEY`, only when you ask for it.

## What it does not protect you from

- **Prompt injection into the judge.** An output under evaluation can contain text that tries to steer the judge. mistrial measures the judge; it does not sanitise what the judge reads. The rubric-reversal and sensitivity probes will often expose a judge that is easily steered, which is the point.
- **A compromised provider or proxy.** If the endpoint you send cases to is hostile, it sees your cases and your key.

## Dependencies

`typer`, `rich`, `httpx`. Nothing else at runtime. `langsmith` only if you install it for the push.

## Reporting a problem

Open a private security advisory on GitHub, or email the maintainer from the address on the profile. You will get a reply within a few days. Please do not open a public issue for anything that could expose other users' keys or data.
