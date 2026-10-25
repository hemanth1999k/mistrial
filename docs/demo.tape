# VHS tape for the README demo: `vhs docs/demo.tape` writes docs/demo.gif and docs/demo.mp4
# https://github.com/charmbracelet/vhs

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
Type "export DYLD_LIBRARY_PATH=/opt/homebrew/opt/expat/lib; export PATH=$PWD/.venv/bin:$PATH; export PS1='$ '; rm -rf /tmp/jfdemo /tmp/jfdemo2; clear"
Enter
Sleep 500ms
Show

Type "# Is your LLM judge flaky?  Three fake judges, twenty cases, five runs each. No API key."
Enter
Sleep 1.5s
Type "judgeflake run examples/cases.jsonl -j demo -j demo:flaky -j demo:lenient --runs 5 --top 4 --pass-mark 0.55 --html none --out-dir /tmp/jfdemo"
Enter
Sleep 9s

Hide
Type "clear"
Enter
Show
Type "# The fix, measured: the same flaky judge wrapped in a 5-vote majority, asked adaptively."
Enter
Sleep 1.5s
Type "judgeflake run examples/cases.jsonl -j demo:flaky --runs 5 --stabilize 5 --adaptive --top 0 --html none --out-dir /tmp/jfdemo"
Enter
Sleep 7s

Hide
Type "clear"
Enter
Show
Type "# Does the verdict depend on how the rubric is worded?  Reword it three ways, then flip the criterion."
Enter
Sleep 1.5s
Type "judgeflake rubric-check examples/cases.jsonl -j demo -j demo:flaky --out-dir /tmp/jfdemo"
Enter
Sleep 8s

Hide
Type "clear"
Enter
Show
Type "# Switching judge models?  See what moved, and what your pass mark becomes on the new judge."
Enter
Sleep 1.5s
Type "judgeflake run examples/cases.jsonl -j demo:steady -j demo:lenient --runs 3 -q --html none --out-dir /tmp/jfdemo2"
Enter
Sleep 800ms
Type "judgeflake migrate /tmp/jfdemo2/runs/*.json --pass-mark 0.7"
Enter
Sleep 6s

Hide
Type "clear"
Enter
Show
Type "# Real judges: openai:gpt-5-mini  anthropic:claude-sonnet-5-5  gemini:gemini-2.5-flash  ollama:llama3.1  my_evals:fn"
Enter
Sleep 1s
Type "pip install judgeflake"
Sleep 3s
