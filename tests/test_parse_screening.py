import sys
import os

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from app.criteria.criteria import parse_llm_screening_output


def test_parse_llm_screening_output_basic():
    raw_output = (
        "Decision: Include\n"
        "Confidence: 4\n"
        "Population: Adults with hypertension\n"
        "Intervention: Drug A\n"
        "Comparison: Drug B\n"
        "Outcome: Blood pressure reduction\n"
        "Rationale: The study met inclusion criteria."
    )
    criteria = ["Population", "Intervention", "Comparison", "Outcome"]

    result = parse_llm_screening_output(raw_output, criteria)

    assert result["decision"] == "Include"
    assert result["confidence"] == 4
    assert result["rationale"] == "The study met inclusion criteria."
    assert result["criteria_matches"]["Population"] == "Adults with hypertension"
    assert result["criteria_matches"]["Intervention"] == "Drug A"
    assert result["criteria_matches"]["Comparison"] == "Drug B"
    assert result["criteria_matches"]["Outcome"] == "Blood pressure reduction"