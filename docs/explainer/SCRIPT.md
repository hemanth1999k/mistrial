# Explainer video: script and how to re-render

`docs/explainer.mp4` is an 88 second, 1080p animated explainer. It is rendered from `explainer.html` in this folder, frame by frame, so every number on screen is typed into one HTML file and the video never drifts from the README.

## Narration (for a voice track)

Timestamps match the scenes in the video. Read at a normal pace; each line fits its scene with a beat to spare.

| at | say |
|---|---|
| 0:00 | mistrial. Is your LLM judge flaky? |
| 0:06 | You ship AI features on a score from an LLM judge. A model reads your model's answers and says pass or fail. That score gates the build, picks the winning prompt, and goes on the dashboard. It looks like a fact. |
| 0:15 | Ask the judge again. Nothing changed: not the prompt, not the case, not the model. On this case, gpt-5.4-nano said the extra year "does not contradict the core requirement", then on the next run said the same year makes the answer incorrect. Same model, same case, opposite reasoning. Which run did your dashboard get? |
| 0:28 | The verdict moves in three directions. Across runs of the same judge. Across judges reading the same rubric, where two models fail a correct answer for being in miles. And across rewordings of the rubric that mean the same thing. One eval run is one sample from a noisy process, and almost nobody measures the noise. |
| 0:40 | Why does nothing catch it? Eval platforms report one number and stop. The statistics exist, kappa and alpha are a century old, but they live in notebooks, not in a tool. And the projects that tried stop at the card. Before "how good is my model?" you need "can I trust the number that told me?" |
| 0:50 | Point mistrial at the judge you already use: a model name, or the function you run in LangSmith or openevals. One command. It asks each judge five times, and the card shows how much the number moves, and which cases move it. |
| 1:02 | Real numbers. Three current OpenAI judges, thirty-two tricky cases, five runs each. gpt-5.4-nano agrees with itself ninety-two percent of the time. A CI gate on gpt-5.4-mini goes red ninety percent of the time with nothing changed. Rewording the rubric moves sixteen percent of nano's verdicts. And a pass mark of 0.5 on one judge is 0.62 on the next. |
| 1:12 | Then it does something about it. Wrap the flaky judge in a five-vote majority and it goes from ninety-two to ninety-nine percent self-agreement, with the same evaluator signature. Adaptive re-asking gets there with a hundred and seventy-seven fewer calls. There is a CI gate, a GitHub Action that posts the card on every pull request, and an escalation list for the cases a human should see. |
| 1:21 | pip install mistrial. No key? pipx run mistrial demo. When the verdict can't be trusted, declare a mistrial. |

## Re-render

Needs Chrome, Node, `@playwright/test` and ffmpeg. From a folder where `@playwright/test` is installed:

```bash
node docs/explainer/frames.mjs "$PWD/docs/explainer/explainer.html" video docs/explainer.mp4 88 30
node docs/explainer/frames.mjs "$PWD/docs/explainer/explainer.html" stills docs/explainer 3 26 66   # single frames, for thumbnails
```

The page exposes `window.__seek(seconds)`, which pauses every animation at that moment and sets the counters and typed text. The renderer steps through time at 30 frames per second and pipes PNGs into ffmpeg, so the output is sharp and never drops a frame. Change the numbers or the copy in `explainer.html` and render again.

To add a voice track:

```bash
ffmpeg -i docs/explainer.mp4 -i voice.m4a -c:v copy -c:a aac -shortest docs/explainer-voiced.mp4
```
