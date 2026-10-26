import math

import pytest

from mistrial import stats


def test_normalize_reads_langsmith_and_openevals_results():
    assert stats.normalize({"key": "correctness", "score": True}) is True
    assert stats.normalize({"score": 0.75, "comment": "ok"}) == 0.75
    assert stats.normalize([{"key": "x", "score": 3}]) == 3
    assert stats.normalize("PASS") is True
    assert stats.normalize(" no ") is False
    assert stats.normalize("B") == "B"
    assert stats.normalize(0.123456) == 0.123
    with pytest.raises(ValueError):
        stats.normalize({"comment": "no score here"})
    with pytest.raises(ValueError):
        stats.normalize(None)


def test_majority_breaks_ties_by_first_seen():
    assert stats.majority([True, False]) == (True, 0.5)
    assert stats.majority([2, 3, 3]) == (3, pytest.approx(2 / 3))


def test_consistency_flip_rate_and_agreement():
    c = stats.consistency(
        "j",
        {
            "a": [True, True, True],
            "b": [True, False, True],
            "c": [False, False, False],
            "d": [1, 1, 1],  # mixed with bools is fine for categorical
        },
    )
    assert c.cases == 4 and c.runs == 3
    assert c.flip_cases == ["b"]
    assert c.flip_rate == 0.25
    assert c.mean_agreement == pytest.approx((1 + 2 / 3 + 1 + 1) / 4)
    assert c.numeric is False


def test_consistency_numeric_spread():
    c = stats.consistency("j", {"a": [4, 5, 4], "b": [2, 2, 2]})
    assert c.numeric is True
    assert c.mean_spread == pytest.approx(0.5)
    assert c.per_case[0].spread == 1


def test_cohens_kappa_known_values():
    # Perfect and perfectly-constant raters.
    assert stats.cohens_kappa([1, 0, 1], [1, 0, 1]) == 1.0
    assert stats.cohens_kappa([1, 1, 1], [1, 1, 1]) == 1.0
    # Known: Wikipedia example, 50 items, po=0.70, pe=0.50 -> kappa 0.40
    a = ["y"] * 20 + ["n"] * 15 + ["y"] * 5 + ["n"] * 10
    b = ["y"] * 20 + ["n"] * 15 + ["n"] * 5 + ["y"] * 10
    assert stats.cohens_kappa(a, b) == pytest.approx(0.4)


def test_krippendorff_alpha_matches_the_canonical_example():
    # Krippendorff's own example (also on Wikipedia): three coders, missing values.
    A = [None, None, None, None, None, 3, 4, 1, 2, 1, 1, 3, 3, None, 3]
    B = [1, None, 2, 1, 3, 3, 4, 3, None, None, None, None, None, None, None]
    C = [None, None, 2, 1, 3, 4, 4, None, 2, 1, 1, 3, 3, None, 4]
    units = {}
    for i, vals in enumerate(zip(A, B, C)):
        units[str(i)] = [v for v in vals if v is not None]
    assert stats.krippendorff_alpha(units, "nominal") == pytest.approx(0.691, abs=0.001)
    assert stats.krippendorff_alpha(units, "interval") == pytest.approx(0.811, abs=0.001)


def test_krippendorff_alpha_edges():
    assert stats.krippendorff_alpha({"a": [1, 1], "b": [0, 0]}, "nominal") == 1.0
    assert stats.krippendorff_alpha({"a": [1]}, "nominal") is None
    assert stats.krippendorff_alpha({"a": [1, 0], "b": [0, 1]}, "nominal") < 0


def test_agreement_between_judges():
    maj = {
        "gpt": {"a": True, "b": False, "c": True, "d": True},
        "claude": {"a": True, "b": False, "c": False, "d": True},
    }
    runs = {
        "gpt": {"a": [True, True], "b": [False, False], "c": [True, True], "d": [True, True]},
        "claude": {"a": [True, True], "b": [False, False], "c": [False, True], "d": [True, True]},
    }
    ag = stats.agreement(maj, runs)
    assert ag.level == "nominal"
    assert len(ag.pairs) == 1
    assert ag.pairs[0].percent == 0.75
    assert ag.alpha_majority is not None and ag.alpha_all_runs is not None
    assert ag.pairs[0].mean_abs_diff is None


def test_agreement_numeric_uses_interval_level():
    maj = {"g": {"a": 4, "b": 2}, "c": {"a": 5, "b": 2}}
    ag = stats.agreement(maj)
    assert ag.level == "interval"
    assert ag.pairs[0].mean_abs_diff == 0.5


def test_accuracy_against_labels():
    acc = stats.accuracy("g", {"a": True, "b": False, "c": True}, {"a": True, "b": True, "c": True, "zzz": False})
    assert acc is not None
    assert acc.cases == 3
    assert acc.accuracy == pytest.approx(2 / 3)
    assert acc.misses == [("b", False, True)]
    assert stats.accuracy("g", {"a": True}, {}) is None


def test_flaky_cases_ranked_by_pooled_disagreement():
    runs = {
        "g": {"a": [True, True], "b": [True, False], "c": [True, True]},
        "c": {"a": [True, True], "b": [False, True], "c": [True, False]},
    }
    flaky = stats.flaky_cases(runs, labels={"b": True})
    assert [f.case_id for f in flaky] == ["b", "c"]
    assert flaky[0].disagreement == 0.5  # 2 of 4 said True
    assert flaky[0].label is True
    assert flaky[1].disagreement == 0.25


def test_self_agreement_is_independent_of_run_count():
    # A judge that says True on 3 of 4 runs: two random runs agree with probability 0.5.
    c = stats.consistency("j", {"a": [True, True, True, False]})
    assert c.self_agreement == 0.5
    assert stats.consistency("j", {"a": [True, True], "b": [False, False]}).self_agreement == 1.0
    assert stats.consistency("j", {"a": [True]}).self_agreement is None  # one run: nothing to compare
    assert stats.read_self_agreement(0.99) == "stable"
    assert stats.read_self_agreement(0.5) == "very flaky"


def test_readings():
    assert stats.read_kappa(0.85) == "almost perfect"
    assert stats.read_kappa(-0.1) == "worse than chance"
    assert stats.read_self_agreement(None) == "one run only"
    assert stats.read_self_agreement(0.92) == "noticeably flaky"
    assert math.isfinite(stats.as_number(True))
