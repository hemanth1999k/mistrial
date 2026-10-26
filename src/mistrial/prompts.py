"""Built-in rubrics for the LLM judges. Keep them short, strict and reference-aware."""

from __future__ import annotations

import json
from typing import Any

SYSTEM = (
    "You are a careful, consistent evaluator. Read the task, then judge the output exactly as the "
    "rubric says. Reply with a single JSON object and nothing else."
)

RUBRICS: dict[str, dict[str, str]] = {
    "correctness": {
        "description": "Is the output correct, judged against the reference? Boolean.",
        "template": (
            "Judge whether OUTPUT is correct, using REFERENCE as the ground truth. Minor wording "
            "differences do not matter; factual or logical differences do. If OUTPUT is missing "
            "something the REFERENCE requires, it is incorrect.\n\n"
            "INPUT:\n{inputs}\n\nOUTPUT:\n{outputs}\n\nREFERENCE:\n{reference_outputs}\n\n"
            'Reply as JSON: {{"score": true or false, "reasoning": "<one or two sentences>"}}'
        ),
        "needs_reference": "true",
    },
    "quality": {
        "description": "Overall answer quality for the input, 1 (unusable) to 5 (excellent). No reference needed.",
        "template": (
            "Rate the quality of OUTPUT as an answer to INPUT on a 1 to 5 scale: 1 unusable, 2 poor, "
            "3 acceptable, 4 good, 5 excellent. Judge accuracy, completeness and clarity. "
            "Use the whole scale.\n\n"
            "INPUT:\n{inputs}\n\nOUTPUT:\n{outputs}\n\n"
            'Reply as JSON: {{"score": <integer 1-5>, "reasoning": "<one or two sentences>"}}'
        ),
        "needs_reference": "false",
    },
}


# Rewordings of each built-in rubric that mean the same thing. `mistrial rubric-check` runs
# every variant and counts the verdicts that change: if the wording moves the score, the score
# was measuring the wording. Research puts this at a quarter of cases for many judges.
RUBRIC_VARIANTS: dict[str, list[str]] = {
    "correctness": [
        (
            "You are given a question, a candidate answer, and the accepted answer. Decide if the "
            "candidate answer is correct. Treat the accepted answer as the truth. Ignore phrasing; "
            "count factual or logical mismatches and anything required that the candidate left out.\n\n"
            "Question:\n{inputs}\n\nCandidate answer:\n{outputs}\n\nAccepted answer:\n{reference_outputs}\n\n"
            'Answer in JSON only: {{"score": true if correct else false, "reasoning": "<brief>"}}'
        ),
        (
            "Compare the RESPONSE to the GOLD answer for the given TASK. Mark it correct when it "
            "says the same thing as GOLD in substance, even if worded differently. Mark it incorrect "
            "when it contradicts GOLD, gets a fact wrong, or omits something GOLD requires.\n\n"
            "TASK: {inputs}\nRESPONSE: {outputs}\nGOLD: {reference_outputs}\n\n"
            'Return JSON: {{"score": <true|false>, "reasoning": "<why, briefly>"}}'
        ),
        (
            "Grade the following.\n\n"
            "PROMPT\n{inputs}\n\nMODEL OUTPUT\n{outputs}\n\nEXPECTED\n{reference_outputs}\n\n"
            "Is MODEL OUTPUT correct with respect to EXPECTED? Differences in wording are fine. "
            "Differences in facts, logic, or missing required content are not.\n"
            'Respond with a JSON object: {{"score": true or false, "reasoning": "one sentence"}}'
        ),
    ],
    "quality": [
        (
            "Score how good OUTPUT is as a response to INPUT, from 1 to 5 (1 unusable, 2 poor, 3 "
            "acceptable, 4 good, 5 excellent). Consider accuracy, completeness and clarity, and use "
            "the full range.\n\nINPUT:\n{inputs}\n\nOUTPUT:\n{outputs}\n\n"
            'JSON only: {{"score": <1-5>, "reasoning": "<brief>"}}'
        ),
        (
            "You are grading an answer. Question: {inputs}\n\nAnswer: {outputs}\n\n"
            "Give an overall quality rating between 1 and 5 where 1 means unusable and 5 means "
            "excellent, weighing whether it is accurate, complete and clear. Do not cluster at 3 or 4.\n"
            'Reply as JSON: {{"score": <integer 1 to 5>, "reasoning": "<short justification>"}}'
        ),
        (
            "Evaluate the response below.\n\nUser request:\n{inputs}\n\nResponse:\n{outputs}\n\n"
            "On a five-point scale (1 = unusable, 5 = excellent), how good is the response in terms "
            "of accuracy, completeness and clarity?\n"
            'Output a JSON object of the form {{"score": N, "reasoning": "..."}} with N from 1 to 5.'
        ),
    ],
}

# The same rubric with the criterion flipped. A judge that reads the rubric should give the
# opposite verdict on every case. Many don't: they answer "is it good?" whatever you ask.
REVERSED_RUBRICS: dict[str, str] = {
    "correctness": (
        "Judge whether OUTPUT is INCORRECT, using REFERENCE as the ground truth. Minor wording "
        "differences do not make it incorrect; factual or logical differences do, and so does "
        "missing something the REFERENCE requires.\n\n"
        "INPUT:\n{inputs}\n\nOUTPUT:\n{outputs}\n\nREFERENCE:\n{reference_outputs}\n\n"
        'Reply as JSON: {{"score": true if OUTPUT is INCORRECT, false if it is correct, "reasoning": "<one or two sentences>"}}'
    ),
}


def rubric_variants(name_or_path: str) -> list[str]:
    """Built-in rewordings for a built-in rubric; empty for a prompt file (pass your own)."""
    return list(RUBRIC_VARIANTS.get(name_or_path, []))


def reversed_rubric(name_or_path: str) -> str | None:
    return REVERSED_RUBRICS.get(name_or_path)


def _fmt(value: Any) -> str:
    if value is None:
        return "(none)"
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True)


def render(template: str, *, inputs: Any, outputs: Any, reference_outputs: Any) -> str:
    """Fill a rubric template. Templates use {inputs}, {outputs}, {reference_outputs}."""
    return template.format(
        inputs=_fmt(inputs),
        outputs=_fmt(outputs),
        reference_outputs=_fmt(reference_outputs),
    )


def load_rubric(name_or_path: str) -> str:
    """A built-in rubric by name, or the contents of a prompt file."""
    if name_or_path in RUBRICS:
        return RUBRICS[name_or_path]["template"]
    from pathlib import Path

    p = Path(name_or_path)
    if p.exists():
        return p.read_text()
    raise ValueError(f"unknown rubric {name_or_path!r}; built-ins: {', '.join(RUBRICS)} (or pass a file path)")
