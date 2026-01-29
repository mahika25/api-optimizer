import sys
import os
from unittest.mock import MagicMock

sys.modules["openai"] = MagicMock()

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.insert(0, repo_root)

from src.runtime.wrapper import APIOptimizerWrapper


def test_cache_hit_skips_api_call():

    wrapper = APIOptimizerWrapper(
        enable_caching=True,
        enable_model_downgrade=True,
        auto_apply_optimizations=True,
        verbose=False
    )

    prompt = "Is this email spam?"
    model = "gpt-4"
    cached_response = {"cached": True}

    wrapper.cache.set(prompt, model, cached_response, temperature=1.0, max_tokens=50)

    api_called = {"count": 0}
    def fake_create(**kwargs):
        api_called["count"] += 1
        return {"live": True}

    out = wrapper._optimized_call(
        fake_create,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=50
    )

    print("Returned:", out)
    assert out == cached_response
    assert api_called["count"] == 0
    assert wrapper.stats["cache_hits"] == 1


def test_model_downgrade_applied():

    wrapper = APIOptimizerWrapper(
        enable_caching=False,
        enable_model_downgrade=True,
        auto_apply_optimizations=True,
        verbose=False
    )

    prompt = "Translate hello to Spanish"
    starting_model = "gpt-4"

    received_model = {"value": None}

    def fake_create(**kwargs):
        received_model["value"] = kwargs.get("model")
        return {"ok": True}

    out = wrapper._optimized_call(
        fake_create,
        model=starting_model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.3,
        max_tokens=20
    )

    print("Original model:", starting_model)
    print("Used model:", received_model["value"])

    assert received_model["value"] != starting_model
    assert received_model["value"] == "gpt-4o-mini"
    assert wrapper.stats["model_downgrades"] == 1


def test_model_not_downgraded_for_complex_prompt():

    wrapper = APIOptimizerWrapper(
        enable_caching=False,
        enable_model_downgrade=True,
        auto_apply_optimizations=True,
        verbose=False
    )

    prompt = "Design a distributed system with fault tolerance and replication"
    model = "gpt-4"

    received_model = {"value": None}

    def fake_create(**kwargs):
        received_model["value"] = kwargs.get("model")
        return {"ok": True}

    wrapper._optimized_call(
        fake_create,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=1.0,
        max_tokens=500
    )

    print("Used model:", received_model["value"])
    assert received_model["value"] == model
    assert wrapper.stats["model_downgrades"] == 0


def test_cost_savings_tracked():

    wrapper = APIOptimizerWrapper(
        enable_caching=False,
        enable_model_downgrade=True,
        auto_apply_optimizations=True,
        verbose=False
    )

    prompt = "Is this text positive or negative sentiment?"
    model = "gpt-4"

    def fake_create(**kwargs):
        return {"ok": True}

    wrapper._optimized_call(
        fake_create,
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=30
    )

    print("Total cost saved:", wrapper.stats["total_cost_saved"])
    assert wrapper.stats["total_cost_saved"] > 0


if __name__ == "__main__":
    test_cache_hit_skips_api_call()
    test_model_downgrade_applied()
    test_model_not_downgraded_for_complex_prompt()
    test_cost_savings_tracked()

    print("\nALL WRAPPER TESTS PASSED")
