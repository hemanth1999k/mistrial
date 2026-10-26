"""Cases: the things a judge is asked to score.

A case is what you'd hand to a LangSmith or openevals evaluator: `inputs`, `outputs`, and
optionally `reference_outputs`. A human `label` is optional too; when present the report
measures the judges against it.
"""

from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

INPUT_KEYS = ("inputs", "input", "question", "prompt", "query")
OUTPUT_KEYS = ("outputs", "output", "answer", "response", "completion", "prediction")
REFERENCE_KEYS = (
    "reference_outputs",
    "reference_output",
    "reference",
    "expected",
    "expected_output",
    "expected_outputs",
    "ground_truth_answer",
)
LABEL_KEYS = ("label", "labels", "human", "human_label", "ground_truth", "expected_score", "gold")
ID_KEYS = ("id", "case_id", "example_id", "name")


@dataclass(frozen=True)
class Case:
    id: str
    inputs: Any
    outputs: Any
    reference_outputs: Any = None
    label: Any = None
    metadata: dict = field(default_factory=dict)

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "inputs": self.inputs,
            "outputs": self.outputs,
            "reference_outputs": self.reference_outputs,
            "label": self.label,
            "metadata": self.metadata,
        }


def _pick(row: dict, keys: Iterable[str]) -> tuple[bool, Any]:
    for k in keys:
        if k in row:
            return True, row[k]
    return False, None


def case_from_dict(row: dict, index: int) -> Case:
    """Build a Case from a loosely-shaped dict, accepting the usual alias keys."""
    if not isinstance(row, dict):
        raise ValueError(f"case {index}: expected an object, got {type(row).__name__}")
    has_in, inputs = _pick(row, INPUT_KEYS)
    has_out, outputs = _pick(row, OUTPUT_KEYS)
    if not has_out:
        raise ValueError(f"case {index}: needs an 'outputs' (or 'output'/'answer'/'response') field")
    _, reference = _pick(row, REFERENCE_KEYS)
    _, label = _pick(row, LABEL_KEYS)
    has_id, cid = _pick(row, ID_KEYS)
    known = set(INPUT_KEYS) | set(OUTPUT_KEYS) | set(REFERENCE_KEYS) | set(LABEL_KEYS) | set(ID_KEYS)
    metadata = {k: v for k, v in row.items() if k not in known}
    return Case(
        id=str(cid) if has_id and cid not in (None, "") else f"case-{index + 1:03d}",
        inputs=inputs if has_in else None,
        outputs=outputs,
        reference_outputs=reference,
        label=_coerce_label(label),
        metadata=metadata,
    )


def _coerce_label(value: Any) -> Any:
    """Labels from CSV arrive as strings; turn 'true'/'1'/'0.5' into real values."""
    if isinstance(value, str):
        s = value.strip()
        if s == "":
            return None
        if s.lower() in ("true", "yes", "pass", "correct"):
            return True
        if s.lower() in ("false", "no", "fail", "incorrect"):
            return False
        try:
            return int(s)
        except ValueError:
            pass
        try:
            return float(s)
        except ValueError:
            return s
    return value


def load_cases(path: str | Path) -> list[Case]:
    """Load cases from .jsonl, .json (a list or {"cases": [...]}), or .csv."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(p)
    suffix = p.suffix.lower()
    rows: list[dict]
    if suffix == ".jsonl":
        rows = [json.loads(line) for line in p.read_text().splitlines() if line.strip()]
    elif suffix == ".json":
        data = json.loads(p.read_text())
        if isinstance(data, dict):
            data = data.get("cases") or data.get("examples") or data.get("data")
        if not isinstance(data, list):
            raise ValueError(f"{p}: expected a list of cases (or an object with a 'cases' list)")
        rows = data
    elif suffix == ".csv":
        with p.open(newline="") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
    else:
        raise ValueError(f"{p}: unsupported format {suffix!r} (use .jsonl, .json or .csv)")
    cases = [case_from_dict(r, i) for i, r in enumerate(rows)]
    ids = [c.id for c in cases]
    dupes = sorted({i for i in ids if ids.count(i) > 1})
    if dupes:
        raise ValueError(f"{p}: duplicate case ids: {', '.join(dupes[:5])}")
    if not cases:
        raise ValueError(f"{p}: no cases found")
    return cases
