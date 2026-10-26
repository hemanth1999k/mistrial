import json

import pytest

from mistrial.cases import load_cases


def test_jsonl_with_aliases(tmp_path):
    p = tmp_path / "c.jsonl"
    p.write_text(
        "\n".join(
            [
                json.dumps({"question": "2+2?", "answer": "4", "expected": "4", "label": "true"}),
                json.dumps({"id": "x", "inputs": {"q": "cap of France"}, "outputs": {"a": "Lyon"}, "reference_outputs": {"a": "Paris"}, "label": False, "topic": "geo"}),
            ]
        )
    )
    cases = load_cases(p)
    assert [c.id for c in cases] == ["case-001", "x"]
    assert cases[0].inputs == "2+2?" and cases[0].reference_outputs == "4" and cases[0].label is True
    assert cases[1].metadata == {"topic": "geo"} and cases[1].label is False


def test_json_object_wrapper_and_csv(tmp_path):
    p = tmp_path / "c.json"
    p.write_text(json.dumps({"cases": [{"output": "hi"}]}))
    assert load_cases(p)[0].outputs == "hi"
    c = tmp_path / "c.csv"
    c.write_text("id,input,output,expected,label\n1,q,a,a,1\n2,q2,b,c,0.5\n")
    cases = load_cases(c)
    assert cases[0].label == 1 and cases[1].label == 0.5
    assert cases[1].reference_outputs == "c"


def test_errors(tmp_path):
    p = tmp_path / "bad.jsonl"
    p.write_text(json.dumps({"inputs": "no outputs here"}))
    with pytest.raises(ValueError, match="outputs"):
        load_cases(p)
    d = tmp_path / "dupe.jsonl"
    d.write_text("\n".join([json.dumps({"id": "a", "outputs": 1}), json.dumps({"id": "a", "outputs": 2})]))
    with pytest.raises(ValueError, match="duplicate"):
        load_cases(d)
    with pytest.raises(FileNotFoundError):
        load_cases(tmp_path / "missing.jsonl")
