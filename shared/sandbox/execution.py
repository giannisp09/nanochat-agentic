"""
Shared code-execution + unit-test reward, used by BOTH tracks:
  - Track A (posttrain/envs/code_reward.py -> verl reward fn)
  - Track B (tasks/coding_env.py uses the same shaped-reward shape)

It reuses nanochat.execution.execute_code (subprocess + timeout + memory limit +
reliability_guard), which is pure-Python (no torch), so it imports fine inside the
separate Track-A verl/SGLang environment as long as the repo root is on PYTHONPATH.

SECURITY: nanochat.execution is explicitly NOT a security sandbox (network is not
blocked; ctypes can escape). For RL at volume — especially on shared infra — run
the whole job inside a container/VM, or swap execute_code for verl Sandbox Fusion.
The function boundary here (run_unit_tests / pass_rate_reward) is exactly where you
would redirect to a remote sandbox service.
"""

from nanochat.execution import execute_code, ExecutionResult  # noqa: F401


def run_unit_tests(solution_code: str, tests: str, entry_point: str,
                   imports: str = "", timeout: float = 8.0) -> ExecutionResult:
    """Assemble `solution + tests + check(entry_point)` and run it in the sandbox."""
    program = (
        (imports + "\n\n" if imports else "")
        + solution_code + "\n\n"
        + tests + "\n"
        + f"check({entry_point})"
    )
    return execute_code(program, timeout=timeout)


def pass_rate_reward(solution_code: str, tests: str, entry_point: str,
                     imports: str = "", timeout: float = 8.0, partial: float = 0.1) -> float:
    """
    Shaped reward (matches tasks/coding_env.py):
      1.0      all hidden tests pass
      partial  valid Python but fails / errors at runtime  (anti-collapse)
      0.0      syntax error / unusable

    `tests` must define `check(f)` that asserts against the candidate function.
    Swap the body for fractional per-test pass-rate when your suite exposes
    individual test cases (verl's Prime reward manager does this).
    """
    if not solution_code.strip():
        return 0.0
    if run_unit_tests(solution_code, tests, entry_point, imports=imports, timeout=timeout).success:
        return 1.0
    try:
        compile(solution_code, "<solution>", "exec")
        return partial
    except SyntaxError:
        return 0.0
