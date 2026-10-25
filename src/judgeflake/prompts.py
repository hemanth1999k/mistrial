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
