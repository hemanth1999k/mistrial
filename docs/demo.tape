# VHS tape for the README demo: `vhs docs/demo.tape` writes docs/demo.gif and docs/demo.mp4
# https://github.com/charmbracelet/vhs
#
# The judges are real OpenAI models. Before recording, warm the cache with live calls:
#   export OPENAI_API_KEY=...
#   mistrial run examples/cases.jsonl -j openai:gpt-4o-mini -j openai:gpt-5-mini -j openai:gpt-4.1-mini --runs 5 --pass-mark 0.45 --out-dir /tmp/mtdemo --html none
#   mistrial rubric-check examples/cases.jsonl -j openai:gpt-4o-mini -j openai:gpt-5-mini --out-dir /tmp/mtdemo
#   mkdir -p /tmp/mtdemo2 && cp /tmp/mtdemo/cache.sqlite /tmp/mtdemo2/
#   mistrial run examples/cases.jsonl -j openai:gpt-4o-mini -j openai:gpt-5-mini --runs 3 -q --html none --out-dir /tmp/mtdemo2
# The recording then replays those real replies from the cache, so the GIF stays short.

Output docs/demo.gif
Output docs/demo.mp4

Set Shell "bash"
Set FontSize 13
Set FontFamily "Menlo"
Set Width 1280
Set Height 960
Set Padding 22
Set Theme "Catppuccin Mocha"
Set TypingSpeed 25ms

Hide
Type "export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib; export PATH=$PWD/.venv/bin:$PATH; export PS1='$ '; clear"
Enter
Sleep 500ms
Show

Type "# mistrial: is your LLM judge flaky?  Three real OpenAI judges, 20 cases, 5 runs each.  (replies replayed from cache)"
Enter
Sleep 1.5s
Type "mistrial run examples/cases.jsonl -j openai:gpt-4o-mini -j openai:gpt-5-mini -j openai:gpt-4.1-mini --runs 5 --top 4 --pass-mark 0.45 --html none --out-dir /tmp/mtdemo"
Enter
Sleep 11s

Hide
Type "clear"
Enter
Show
Type "# gpt-4o-mini looks 'stable' and is wrong on a fifth of the cases.  Does its verdict depend on how the rubric is worded?"
Enter
Sleep 1.5s
Type "mistrial rubric-check examples/cases.jsonl -j openai:gpt-4o-mini -j openai:gpt-5-mini --out-dir /tmp/mtdemo"
Enter
Sleep 9s

Hide
Type "clear"
Enter
Show
Type "# Switching from gpt-4o-mini to gpt-5-mini?  See which cases move, and what your pass mark becomes on the new judge."
Enter
Sleep 1.5s
Type "mistrial migrate /tmp/mtdemo2/runs/*.json --pass-mark 0.6"
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
