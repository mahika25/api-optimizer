import sys
import os
import tempfile

current_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(current_dir)
sys.path.insert(0, repo_root)

from src.runtime.cache import ResponseCache


def test_cache_set_get_exact_hit():

    c = ResponseCache(ttl_minutes=60)
    prompt = "Hello"
    model = "gpt-4o-mini"
    resp = {"ok": True}

    assert c.get(prompt, model) is None
    c.set(prompt, model, resp)
    assert c.get(prompt, model) == resp


def test_cache_key_changes_with_temperature_and_max_tokens():

    c = ResponseCache(ttl_minutes=60)
    prompt = "Same prompt"
    model = "gpt-4o-mini"

    c.set(prompt, model, "resp_t1", temperature=0.2, max_tokens=50)
    assert c.get(prompt, model, temperature=0.2, max_tokens=50) == "resp_t1"
    assert c.get(prompt, model, temperature=0.9, max_tokens=50) is None
    assert c.get(prompt, model, temperature=0.2, max_tokens=200) is None



def test_cache_ttl_expiration():
    c = ResponseCache(ttl_minutes=0)
    prompt = "Expire me"
    model = "gpt-4o-mini"

    c.set(prompt, model, "resp")
    assert c.get(prompt, model) is None


def test_cache_stats_counts_expired():
    c = ResponseCache(ttl_minutes=0)  # everything expires immediately
    c.set("p1", "m1", "r1")
    c.set("p2", "m2", "r2")

    stats = c.stats()
    print("Stats:", stats)

    assert stats["total_entries"] == 2
    assert stats["expired_entries"] == 2
    assert stats["valid_entries"] == 0



def test_cache_save_load_roundtrip():
    c = ResponseCache(ttl_minutes=60)
    c.set("p", "m", {"x": 1}, temperature=0.1, max_tokens=12)

    with tempfile.TemporaryDirectory() as td:
        path = os.path.join(td, "cache.pkl")
        c.save(path)

        c2 = ResponseCache(ttl_minutes=60)
        c2.load(path)

        loaded = c2.get("p", "m", temperature=0.1, max_tokens=12)
        print("Loaded:", loaded)
        assert loaded == {"x": 1}



if __name__ == "__main__":
    test_cache_set_get_exact_hit()
    test_cache_key_changes_with_temperature_and_max_tokens()
    test_cache_ttl_expiration()
    test_cache_stats_counts_expired()
    test_cache_save_load_roundtrip()
    print("\nALL CACHE TESTS PASSED")
