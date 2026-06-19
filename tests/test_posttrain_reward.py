"""
Tests for the Track A reward path (shared sandbox + verl reward adapter).

Standalone:  .venv/bin/python tests/test_posttrain_reward.py
"""

from shared.sandbox.execution import pass_rate_reward, run_unit_tests
from posttrain.envs.code_reward import score, compute_score
from posttrain.data.prep_rl_tasks import build_records


ADD_TESTS = "def check(f):\n    assert f(1, 2) == 3\n    assert f(0, 0) == 0\n"


def test_pass_rate_reward_levels():
    assert pass_rate_reward("def add(a,b):\n    return a+b", ADD_TESTS, "add") == 1.0
    assert pass_rate_reward("def add(a,b):\n    return a-b", ADD_TESTS, "add") == 0.1
    assert pass_rate_reward("def add(a,b)\n    return a", ADD_TESTS, "add") == 0.0
    assert pass_rate_reward("", ADD_TESTS, "add") == 0.0


def test_verl_compute_score_with_extra_info():
    extra = {"tests": ADD_TESTS, "entry_point": "add"}
    # model wrapped its code in a fence -> extract_program handles it
    good = "Here you go:\n```python\ndef add(a, b):\n    return a + b\n```"
    bad = "```python\ndef add(a, b):\n    return a - b\n```"
    broken = "```python\ndef add(a, b)\n    return a\n```"
    assert compute_score(solution_str=good, extra_info=extra) == 1.0
    assert compute_score(solution_str=bad, extra_info=extra) == 0.1
    assert compute_score(solution_str=broken, extra_info=extra) == 0.0
    # missing test metadata -> 0 (defensive)
    assert compute_score(solution_str=good, extra_info={}) == 0.0


def test_build_records_schema():
    recs = build_records(split="test", suite="coding", jsonl_path=None)
    assert len(recs) >= 1
    r = recs[0]
    for key in ("data_source", "prompt", "reward_model", "extra_info"):
        assert key in r, key
    assert r["prompt"][0]["role"] == "user"
    assert "tests" in r["extra_info"] and "entry_point" in r["extra_info"]
    # the record's own reward path scores its reference logic correctly
    ep = r["extra_info"]["entry_point"]
    assert score(f"```python\ndef {ep}(): pass\n```", r["extra_info"]["tests"], ep) in (0.0, 0.1)


if __name__ == "__main__":
    import sys
    tests = sorted((k, v) for k, v in globals().items()
                   if k.startswith("test_") and callable(v))
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
