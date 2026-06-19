"""
Tests for tasks/coding_env.py (verifiable coding RL environment).

Standalone:  .venv/bin/python tests/test_coding_env.py
(Spawns sandbox subprocesses, so takes a few seconds.)
"""

from tasks.coding_env import CodingEnv


def _completion(code):
    return f"```python\n{code}\n```"


def test_example_structure():
    env = CodingEnv(split="train")
    conv = env[0]
    assert "messages" in conv and conv["messages"][0]["role"] == "user"
    assert "entry_point" in conv and "tests" in conv
    assert "python tool" in conv["messages"][0]["content"]


def test_all_reference_solutions_pass():
    # Strong self-consistency check: every bundled problem's reference solution
    # must pass its own hidden tests (reward 1.0, evaluate True).
    env = CodingEnv(split="all", as_sft=True)
    for i in range(len(env)):
        conv = env[i]
        completion = conv["messages"][-1]["content"]
        r = env.reward(conv, completion)
        assert r == 1.0, f"problem {i} ({conv['entry_point']}) reference reward={r}"
    # check evaluate on a few
    for i in (0, 5, len(env) - 1):
        conv = env[i]
        assert env.evaluate(conv, conv["messages"][-1]["content"]) is True


def test_reward_partial_credit_for_wrong_but_valid():
    env = CodingEnv(split="all")
    conv = env[0]  # add
    # valid Python, wrong logic -> 0.1 partial credit (anti-collapse)
    r = env.reward(conv, _completion("def add(a, b):\n    return a - b"))
    assert r == 0.1, r


def test_reward_zero_for_syntax_error():
    env = CodingEnv(split="all")
    conv = env[0]  # add
    r = env.reward(conv, _completion("def add(a, b)\n    return a + b"))  # missing colon
    assert r == 0.0, r


def test_reward_full_for_correct():
    env = CodingEnv(split="all")
    conv = env[0]
    r = env.reward(conv, _completion("def add(a, b):\n    return a + b"))
    assert r == 1.0, r


def test_tool_bonus_applies_only_when_used_and_runnable():
    env = CodingEnv(split="all", tool_bonus=0.05)
    conv = env[0]
    # correct + tool used -> capped at 1.0 (already max)
    used_correct = "<|python_start|>print(add(1,2))<|python_end|>" + _completion("def add(a, b):\n    return a + b")
    assert env.reward(conv, used_correct) == 1.0
    # wrong-but-valid + tool used -> 0.1 + 0.05 bonus
    used_wrong = "<|python_start|>print(add(1,2))<|python_end|>" + _completion("def add(a, b):\n    return a - b")
    assert abs(env.reward(conv, used_wrong) - 0.15) < 1e-9
    # syntax error + tool used -> still 0.0 (no bonus on unusable code)
    used_broken = "<|python_start|>x<|python_end|>" + _completion("def add(a, b)\n    return a")
    assert env.reward(conv, used_broken) == 0.0


def test_split_sizes():
    train = CodingEnv(split="train")
    test = CodingEnv(split="test")
    allp = CodingEnv(split="all")
    assert len(train) + len(test) == len(allp)
    assert len(test) >= 1 and len(train) >= 1


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
