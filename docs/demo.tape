# VHS tape for the README demo: `vhs docs/demo.tape` writes docs/demo.gif and docs/demo.mp4
# https://github.com/charmbracelet/vhs
#
# The judges are real OpenAI models. Before recording, warm the cache with live calls:
#   export OPENAI_API_KEY=...
#   mistrial run examples/tricky.jsonl -j openai:gpt-5.4-nano -j openai:gpt-5.4-mini -j openai:gpt-5.5 --runs 5 --pass-mark 0.5 --out-dir /tmp/mt2 --html none
#   mistrial run examples/tricky.jsonl -j openai:gpt-5.4-nano --runs 5 --stabilize 5 --adaptive --out-dir /tmp/mt2 --html none
#   mistrial rubric-check examples/tricky.jsonl -j openai:gpt-5.4-nano -j openai:gpt-5.5 --out-dir /tmp/mt2
#   mkdir -p /tmp/mt2b && cp /tmp/mt2/cache.sqlite /tmp/mt2b/
#   mistrial run examples/tricky.jsonl -j openai:gpt-5.4-nano -j openai:gpt-5.5 --runs 3 -q --html none --out-dir /tmp/mt2b
# The recording then replays those real replies from the cache, so the GIF stays short.

Output docs/demo.gif
Output docs/demo.mp4

Set Shell "bash"
Set FontSize 13
Set FontFamily "Menlo"
Set Width 1280
Set Height 1000
Set Padding 22
Set Theme "Catppuccin Mocha"
Set TypingSpeed 25ms

Hide
Type "export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib; export PATH=$PWD/.venv/bin:$PATH; export PS1='$ '; clear"
Enter
Sleep 500ms
Show

Type "# mistrial: is your LLM judge flaky?  gpt-5.4-nano, gpt-5.4-mini and gpt-5.5 on 32 tricky cases, 5 runs each.  (real replies, replayed from cache)"
Enter
Sleep 1.5s
Type "mistrial run examples/tricky.jsonl -j openai:gpt-5.4-nano -j openai:gpt-5.4-mini -j openai:gpt-5.5 --runs 5 --top 6 --pass-mark 0.5 --html none --out-dir /tmp/mt2"
Enter
Sleep 12s

Hide
Type "clear"
Enter
Show
Type "# The fix, measured: the flaky judge wrapped in a 5-vote majority, asked adaptively (extra calls only where the first two runs disagreed)."
Enter
Sleep 1.5s
Type "mistrial run examples/tricky.jsonl -j openai:gpt-5.4-nano --runs 5 --stabilize 5 --adaptive --top 0 --html none --out-dir /tmp/mt2"
Enter
Sleep 8s

Hide
Type "clear"
Enter
Show
Type "# Does the verdict depend on how the rubric is worded?  Reword it three ways, then flip the criterion."
Enter
Sleep 1.5s
Type "mistrial rubric-check examples/tricky.jsonl -j openai:gpt-5.4-nano -j openai:gpt-5.5 --out-dir /tmp/mt2"
Enter
Sleep 9s

Hide
Type "clear"
Enter
Show
Type "# Switching judge models?  See which cases move, and what your pass mark becomes on the new judge."
Enter
Sleep 1.5s
Type "mistrial migrate /tmp/mt2b/runs/*.json --pass-mark 0.5"
Enter
Sleep 8s

Hide
Type "clear"
Enter
Show
Type "# Works with openai:*  anthropic:*  gemini:*  groq:*  ollama:*  or your own my_evals:fn.  No key?  'mistrial demo' runs fake judges."
Enter
Sleep 1s
Type "pip install mistrial"
Sleep 3s
