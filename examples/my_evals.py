"""Two plain-Python evaluators with the LangSmith / openevals signature.

    judgeflake run examples/cases.jsonl --judge examples.my_evals:exact --judge examples.my_evals:contains --runs 3
"""


def exact(inputs, outputs, reference_outputs):
    """Strict: the output must equal the reference."""
    return {"key": "exact", "score": str(outputs).strip().lower() == str(reference_outputs).strip().lower()}


def contains(inputs, outputs, reference_outputs):
    """Lenient: the reference just has to appear somewhere in the output."""
    return {"key": "contains", "score": str(reference_outputs).strip().lower() in str(outputs).lower()}
