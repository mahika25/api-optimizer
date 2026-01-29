import sys
import os
from typing import Dict, Any

# Allow running directly
current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.insert(0, repo_root)

from src.analyzer.model_rec import ModelRecommender


def test_pricing_contains_expected_models():
    mr = ModelRecommender()
    assert "gpt-4o-mini" in mr.PRICING
    assert "gpt-4o" in mr.PRICING
    assert "gpt-4" in mr.PRICING
    assert "claude-3-haiku" in mr.PRICING

    for model, rates in mr.PRICING.items():
        assert "input" in rates and "output" in rates
        assert rates["input"] >= 0
        assert rates["output"] >= 0


def test_estimate_cost_monotonicity():
    mr = ModelRecommender()
    m = "gpt-4o-mini"
    c1 = mr.estimate_cost(m, 1000, 1000)
    c2 = mr.estimate_cost(m, 2000, 1000)
    c3 = mr.estimate_cost(m, 1000, 2000)
    assert c2 >= c1
    assert c3 >= c1
    assert c1 >= 0


def test_estimate_tokens_openai_vs_claude_sanity():
    mr = ModelRecommender()
    text = "Hello world! This is a test with punctuation, numbers (123), and symbols: $%."

    t_openai = mr.estimate_tokens(text, model="gpt-4o-mini")
    t_claude = mr.estimate_tokens(text, model="claude-3-haiku")

    assert t_openai > 0
    assert t_claude > 0

    ratio = t_openai / max(t_claude, 1)
    assert 0.25 <= ratio <= 4.0, f"Token estimate ratio seems off: {ratio:.2f}"


def test_estimate_response_tokens_rules():
    mr = ModelRecommender()

    p1 = "Answer yes or no: is the sky blue?"
    out1 = mr.estimate_response_tokens(p1)
    assert out1 == 10

    p2 = "Classify the sentiment as positive or negative: I love it."
    out2 = mr.estimate_response_tokens(p2)
    assert out2 == 20

    p3 = "Summarize:\n" + ("This is a long paragraph. " * 200)
    in3 = mr.estimate_tokens(p3, model="gpt-4o-mini")
    out3 = mr.estimate_response_tokens(p3)
    assert out3 < in3, f"Expected summary output smaller than input: out={out3}, in={in3}"

    p4 = "Write code: implement a binary search tree class in Python."
    out4 = mr.estimate_response_tokens(p4)
    assert out4 >= 200


def test_keyword_analysis_outputs_expected_shape():
    mr = ModelRecommender()
    res = mr.keyword_analysis("Is this spam? yes or no.")
    assert isinstance(res, dict)
    assert "complexity" in res and "confidence" in res
    assert res["complexity"] in {"simple", "medium", "complex"}
    assert 0.0 <= float(res["confidence"]) <= 1.0


def test_analyze_task_complexity_matches_keyword_analysis():
    mr = ModelRecommender()
    p = "Is this spam? yes or no."
    c1 = mr.analyze_task_complexity(p)
    c2 = mr.keyword_analysis(p)["complexity"]
    assert c1 == c2


def test_recommend_model_simple_from_gpt4():
    mr = ModelRecommender()
    current_model = "gpt-4"
    prompt = "Is this email spam? 'Congratulations, you won a prize!'"

    rec = mr.recommend_model(current_model, prompt)
    assert rec is not None

    for k in ["current_model", "recommended_model", "reason", "confidence", "task_complexity", "cost_savings_per_1m_tokens"]:
        assert k in rec, f"Missing key {k} in recommendation"

    assert rec["current_model"] == current_model
    assert rec["recommended_model"] in mr.PRICING

    savings = rec["cost_savings_per_1m_tokens"]
    assert "input" in savings and "output" in savings
    assert savings["input"] >= 0
    assert savings["output"] >= 0


def test_recommend_model_returns_none_when_no_reco():
    mr = ModelRecommender()
    rec = mr.recommend_model("gpt-4o-mini", "Classify sentiment: I love it.")
    assert rec is None


def test_batch_analyze_structure_and_totals():
    mr = ModelRecommender()
    calls = [
        {"model": "gpt-4", "prompt": "Is this spam? Buy now!"},
        {"model": "gpt-4o", "prompt": "Summarize: " + ("hello " * 150)},
        {"model": "claude-3-opus", "prompt": "Classify sentiment: I enjoy this."},
        {"model": "unknown-model", "prompt": "Is this spam?"}
    ]

    out = mr.batch_analyze(calls)
    assert isinstance(out, dict)

    for k in ["recommendations", "total_current_cost", "total_recommended_cost", "total_savings", "savings_percent", "count"]:
        assert k in out, f"Missing key {k} in batch result"

    assert isinstance(out["recommendations"], list)
    assert isinstance(out["count"], int)

    assert out["total_current_cost"] >= 0
    assert out["total_recommended_cost"] >= 0

    expected_savings = out["total_current_cost"] - out["total_recommended_cost"]
    assert abs(out["total_savings"] - expected_savings) < 1e-9

    for item in out["recommendations"]:
        assert "original" in item and "recommendation" in item
        rec: Dict[str, Any] = item["recommendation"]
        for k in ["input_tokens", "output_tokens", "current_cost", "recommended_cost", "savings", "savings_percent"]:
            assert k in rec, f"Missing {k} in per-call recommendation"


def test_batch_analyze_skips_when_no_recommendation():
    mr = ModelRecommender()
    calls = [{"model": "gpt-4o-mini", "prompt": "Classify sentiment: good/bad"}]
    out = mr.batch_analyze(calls)
    assert out["count"] == 0
    assert out["recommendations"] == []


if __name__ == "__main__":
    test_pricing_contains_expected_models()
    test_estimate_cost_monotonicity()
    test_estimate_tokens_openai_vs_claude_sanity()
    test_estimate_response_tokens_rules()
    test_keyword_analysis_outputs_expected_shape()
    test_analyze_task_complexity_matches_keyword_analysis()
    test_recommend_model_simple_from_gpt4()
    test_recommend_model_returns_none_when_no_reco()
    test_batch_analyze_structure_and_totals()
    test_batch_analyze_skips_when_no_recommendation()
    print("\nALL MODEL_REC TESTS PASSED")