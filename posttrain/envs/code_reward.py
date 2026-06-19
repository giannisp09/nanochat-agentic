"""
verl-compatible reward function for agentic coding RL (Track A).

verl calls a custom reward function per rollout. The exact signature has varied
across verl versions; recent versions use roughly:

    def compute_score(data_source, solution_str, ground_truth, extra_info=None, **kwargs) -> float

We keep the real scoring logic in `score()` (framework-agnostic + unit-tested) and
expose thin `compute_score(...)` / `reward_func(...)` adapters. Point verl's
`custom_reward_function.path` / `.name` at this module's `compute_score`.

The dataset (see posttrain/data/prep_rl_tasks.py) carries the hidden tests and the
entry-point in `extra_info`, so the reward can run them in the shared sandbox.
"""

from shared.sandbox.execution import pass_rate_reward
from tasks.humaneval import extract_program


def score(solution_str, tests, entry_point, imports="", timeout=8.0):
    """Core, framework-agnostic scorer: extract code from the completion, run tests."""
    code = extract_program(solution_str or "")
    if not tests or not entry_point:
        return 0.0
    return pass_rate_reward(code, tests, entry_point, imports=imports, timeout=timeout)


def compute_score(data_source=None, solution_str="", ground_truth=None, extra_info=None, **kwargs):
    """verl entry point. Hidden tests + entry_point travel in extra_info."""
    info = extra_info or {}
    return score(
        solution_str,
        tests=info.get("tests"),
        entry_point=info.get("entry_point"),
        imports=info.get("imports", ""),
        timeout=float(info.get("timeout", 8.0)),
    )


# Some verl versions look for `reward_func` instead; alias for convenience.
reward_func = compute_score
