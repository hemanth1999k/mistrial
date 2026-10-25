"""Reliability statistics. Pure functions, no I/O, every formula tested.

Vocabulary used throughout the report:
- consistency: does the SAME judge give the SAME verdict when asked again? (run-to-run)
- agreement:   do DIFFERENT judges give the same verdict?               (judge-to-judge)
- accuracy:    does a judge match the HUMAN label, when one exists?      (judge-to-human)
"""

from __future__ import annotations

import math
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from itertools import combinations
from typing import Any, Hashable, Iterable, Mapping, Sequence

Value = Hashable

# ------------------------------------------------------------------------------ normalize --


def normalize(value: Any) -> Value:
    """Reduce an evaluator result to a comparable verdict value.

    - {"score": x, ...} / {"value": x} / {"passed": x}  -> x (LangSmith / openevals results)
    - bool stays bool; ints stay ints; floats round to 3 places; strings are stripped
    - lists of results take the first one
    """
    if isinstance(value, dict):
        for k in ("score", "value", "passed", "label", "verdict"):
            if k in value:
                return normalize(value[k])
        raise ValueError(f"cannot read a verdict out of {value!r}")
    if isinstance(value, (list, tuple)):
        if not value:
            raise ValueError("empty result")
        return normalize(value[0])
    if isinstance(value, bool):
        return value
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if math.isnan(value):
            raise ValueError("NaN score")
        return round(value, 3)
    if isinstance(value, str):
        s = value.strip()
        low = s.lower()
        if low in ("true", "yes", "pass", "correct"):
            return True
        if low in ("false", "no", "fail", "incorrect"):
            return False
        return s
    if value is None:
        raise ValueError("None is not a verdict")
    return value


def is_numeric(values: Iterable[Value]) -> bool:
    vs = list(values)
    return bool(vs) and all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in vs)


def as_number(value: Value) -> float:
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    raise TypeError(f"{value!r} is not numeric")


# ---------------------------------------------------------------------------- consistency --


@dataclass
class CaseConsistency:
    case_id: str
    values: list[Value]
    majority: Value
    share: float  # fraction of runs that gave the majority verdict (1.0 = every run agreed)
    distinct: int
    spread: float | None = None  # numeric only: max - min across runs


@dataclass
class JudgeConsistency:
    judge: str
    cases: int
    runs: int
    flip_cases: list[str]
    flip_rate: float  # fraction of cases with more than one distinct verdict (grows with runs)
    self_agreement: float | None  # chance two runs of this judge agree on a case (independent of runs)
    mean_agreement: float  # mean of per-case majority share
    numeric: bool
    mean_spread: float | None = None
    per_case: list[CaseConsistency] = field(default_factory=list)


def pairwise_agreement(values: Sequence[Value]) -> float | None:
    """Fraction of pairs of values that are equal. None with fewer than two values."""
    n = len(values)
    if n < 2:
        return None
    same = sum(c * (c - 1) for c in Counter(values).values())
    return same / (n * (n - 1))


def majority(values: Sequence[Value]) -> tuple[Value, float]:
    """The most common value and the share of values that equal it. Ties go to the first seen."""
    if not values:
        raise ValueError("no values")
    counts = Counter(values)
    best = max(counts.values())
    for v in values:
        if counts[v] == best:
            return v, best / len(values)
    raise AssertionError("unreachable")


def consistency(judge: str, verdicts: Mapping[str, Sequence[Value]]) -> JudgeConsistency:
    """Run-to-run consistency for one judge. `verdicts` maps case id -> values across runs."""
    per_case: list[CaseConsistency] = []
    numeric = all(is_numeric(vs) for vs in verdicts.values() if vs)
    for cid, vs in verdicts.items():
        if not vs:
            continue
        maj, share = majority(vs)
        spread = (max(as_number(v) for v in vs) - min(as_number(v) for v in vs)) if numeric else None
        per_case.append(CaseConsistency(cid, list(vs), maj, share, len(set(vs)), spread))
    flips = [c.case_id for c in per_case if c.distinct > 1]
    n = len(per_case)
    runs = max((len(c.values) for c in per_case), default=0)
    pairwise = [p for p in (pairwise_agreement(c.values) for c in per_case) if p is not None]
    return JudgeConsistency(
        judge=judge,
        cases=n,
        runs=runs,
        flip_cases=flips,
        flip_rate=len(flips) / n if n else 0.0,
        self_agreement=sum(pairwise) / len(pairwise) if pairwise else None,
        mean_agreement=sum(c.share for c in per_case) / n if n else 0.0,
        numeric=numeric,
        mean_spread=(sum(c.spread or 0.0 for c in per_case) / n) if (numeric and n) else None,
        per_case=per_case,
    )


# ------------------------------------------------------------------------------ agreement --


def percent_agreement(a: Sequence[Value], b: Sequence[Value]) -> float:
    if len(a) != len(b):
        raise ValueError("sequences differ in length")
    if not a:
        return 0.0
    return sum(1 for x, y in zip(a, b) if x == y) / len(a)


def cohens_kappa(a: Sequence[Value], b: Sequence[Value]) -> float:
    """Cohen's kappa for two raters over the same items. 1 = perfect, 0 = chance, <0 = worse."""
    if len(a) != len(b):
        raise ValueError("sequences differ in length")
    n = len(a)
    if n == 0:
        return 0.0
    po = percent_agreement(a, b)
    ca, cb = Counter(a), Counter(b)
    pe = sum(ca[v] * cb[v] for v in set(a) | set(b)) / (n * n)
    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def krippendorff_alpha(units: Mapping[str, Sequence[Value]], level: str = "nominal") -> float | None:
    """Krippendorff's alpha over units with any number of raters each (missing values allowed).

    `units` maps unit id -> the values given by whichever raters rated it. Units with fewer than
    two values are ignored. `level` is "nominal" (any labels) or "interval" (numbers).
    Returns None when there is nothing to compute (fewer than two pairable values).
    """
    if level not in ("nominal", "interval"):
        raise ValueError("level must be 'nominal' or 'interval'")
    pairable = {u: list(vs) for u, vs in units.items() if len(vs) >= 2}
    n = sum(len(vs) for vs in pairable.values())
    if n <= 1:
        return None

    def delta(c: Value, k: Value) -> float:
        if level == "nominal":
            return 0.0 if c == k else 1.0
        return (as_number(c) - as_number(k)) ** 2

    # Coincidence matrix o[c][k]: within each unit, every ordered pair of values counts 1/(m-1).
    o: dict[Value, dict[Value, float]] = defaultdict(lambda: defaultdict(float))
    for vs in pairable.values():
        m = len(vs)
        for i, c in enumerate(vs):
            for j, k in enumerate(vs):
                if i != j:
                    o[c][k] += 1.0 / (m - 1)
    values = set(o)
    n_c = {c: sum(o[c].values()) for c in values}
    d_o = sum(o[c][k] * delta(c, k) for c in values for k in values)
    d_e = sum(n_c[c] * n_c[k] * delta(c, k) for c in values for k in values) / (n - 1)
    if d_e == 0:
        return 1.0  # everybody said the same thing
    return 1.0 - d_o / d_e


@dataclass
class PairAgreement:
    judge_a: str
    judge_b: str
    cases: int
    percent: float
    kappa: float
    mean_abs_diff: float | None = None  # numeric only


@dataclass
class Agreement:
    pairs: list[PairAgreement]
    alpha_majority: float | None  # alpha over each judge's majority verdict per case
    alpha_all_runs: float | None  # alpha over every (judge, run) as its own rater
    level: str


def agreement(
    majorities: Mapping[str, Mapping[str, Value]],
    all_runs: Mapping[str, Mapping[str, Sequence[Value]]] | None = None,
) -> Agreement:
    """Judge-to-judge agreement.

    `majorities` maps judge -> {case id -> majority verdict};
    `all_runs` (optional) maps judge -> {case id -> values across runs}.
    """
    judges = list(majorities)
    every_value = [v for m in majorities.values() for v in m.values()]
    level = "interval" if is_numeric(every_value) else "nominal"
    pairs: list[PairAgreement] = []
    for ja, jb in combinations(judges, 2):
        common = [cid for cid in majorities[ja] if cid in majorities[jb]]
        a = [majorities[ja][c] for c in common]
        b = [majorities[jb][c] for c in common]
        mad = None
        if level == "interval" and common:
            mad = sum(abs(as_number(x) - as_number(y)) for x, y in zip(a, b)) / len(common)
        pairs.append(PairAgreement(ja, jb, len(common), percent_agreement(a, b), cohens_kappa(a, b), mad))
    units_maj: dict[str, list[Value]] = defaultdict(list)
    for j in judges:
        for cid, v in majorities[j].items():
            units_maj[cid].append(v)
    alpha_maj = krippendorff_alpha(units_maj, level) if len(judges) >= 2 else None
    alpha_runs = None
    if all_runs:
        units_runs: dict[str, list[Value]] = defaultdict(list)
        for j in all_runs:
            for cid, vs in all_runs[j].items():
                units_runs[cid].extend(vs)
        alpha_runs = krippendorff_alpha(units_runs, level)
    return Agreement(pairs, alpha_maj, alpha_runs, level)


# ------------------------------------------------------------------------------- accuracy --


@dataclass
class Accuracy:
    judge: str
    cases: int
    accuracy: float
    kappa: float
    misses: list[tuple[str, Value, Value]]  # (case id, judge majority, human label)
    mean_abs_error: float | None = None  # numeric only


def accuracy(judge: str, majorities: Mapping[str, Value], labels: Mapping[str, Value]) -> Accuracy | None:
    common = [cid for cid in majorities if cid in labels and labels[cid] is not None]
    if not common:
        return None
    a = [majorities[c] for c in common]
    b = [normalize(labels[c]) for c in common]
    mae = None
    if is_numeric(a) and is_numeric(b):
        mae = sum(abs(as_number(x) - as_number(y)) for x, y in zip(a, b)) / len(common)
    misses = [(c, x, y) for c, x, y in zip(common, a, b) if x != y]
    return Accuracy(judge, len(common), percent_agreement(a, b), cohens_kappa(a, b), misses, mae)


# ---------------------------------------------------------------------------- flaky cases --


@dataclass
class FlakyCase:
    case_id: str
    disagreement: float  # 1 - share of the overall majority across every judge and run
    verdicts: dict[str, list[Value]]  # judge -> values
    label: Value | None = None


def flaky_cases(all_runs: Mapping[str, Mapping[str, Sequence[Value]]], labels: Mapping[str, Value] | None = None, top: int = 10) -> list[FlakyCase]:
    """Cases ranked by how much everyone disagreed about them."""
    by_case: dict[str, dict[str, list[Value]]] = defaultdict(dict)
    for j, cases in all_runs.items():
        for cid, vs in cases.items():
            if vs:
                by_case[cid][j] = list(vs)
    out: list[FlakyCase] = []
    for cid, per_judge in by_case.items():
        pooled = [v for vs in per_judge.values() for v in vs]
        _, share = majority(pooled)
        lab = None
        if labels and labels.get(cid) is not None:
            lab = normalize(labels[cid])
        out.append(FlakyCase(cid, round(1.0 - share, 4), per_judge, lab))
    out.sort(key=lambda f: (-f.disagreement, f.case_id))
    return [f for f in out[:top] if f.disagreement > 0]


# ------------------------------------------------------------------------------- readings --


def read_kappa(k: float | None) -> str:
    """A plain-English reading of a kappa / alpha value (Landis & Koch bands)."""
    if k is None:
        return "not computable"
    if k < 0:
        return "worse than chance"
    if k < 0.2:
        return "slight"
    if k < 0.4:
        return "fair"
    if k < 0.6:
        return "moderate"
    if k < 0.8:
        return "substantial"
    return "almost perfect"


def read_self_agreement(p: float | None) -> str:
    """Reading of the chance that two runs of the same judge agree on a case."""
    if p is None:
        return "one run only"
    if p >= 0.98:
        return "stable"
    if p >= 0.95:
        return "mostly stable"
    if p >= 0.90:
        return "noticeably flaky"
    if p >= 0.80:
        return "flaky"
    return "very flaky"
