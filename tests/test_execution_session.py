"""
Tests for PersistentPythonSession (multi-turn stateful sandbox).

Standalone:  .venv/bin/python tests/test_execution_session.py
"""

from nanochat.execution import (
    PersistentPythonSession,
    format_execution_output,
    ExecutionResult,
)


def test_state_persists_across_calls():
    sess = PersistentPythonSession()
    try:
        r1 = sess.run("x = 40")
        assert r1.success, r1
        r2 = sess.run("def add(a, b):\n    return a + b")
        assert r2.success, r2
        r3 = sess.run("print(add(x, 2))")
        assert r3.success, r3
        assert r3.stdout.strip() == "42", repr(r3.stdout)
    finally:
        sess.close()


def test_error_is_captured_not_raised():
    sess = PersistentPythonSession()
    try:
        r = sess.run("print(undefined_name)")
        assert not r.success
        assert "NameError" in (r.error or ""), r
        # session survives the error and keeps state
        r2 = sess.run("print('still alive')")
        assert r2.success and r2.stdout.strip() == "still alive"
    finally:
        sess.close()


def test_partial_stdout_before_error():
    sess = PersistentPythonSession()
    try:
        r = sess.run("print('before'); raise ValueError('boom')")
        assert not r.success
        assert "before" in r.stdout
        assert "ValueError" in (r.error or "")
    finally:
        sess.close()


def test_timeout_recovers_and_keeps_state():
    sess = PersistentPythonSession(timeout=1.0)
    try:
        assert sess.run("marker = 123").success
        r = sess.run("while True:\n    pass")  # should time out (~1s)
        assert r.timeout or (r.error is not None), r
        # after a caught timeout the namespace is still intact
        r2 = sess.run("print(marker)")
        assert r2.success and r2.stdout.strip() == "123", r2
    finally:
        sess.close()


def test_format_execution_output():
    ok = ExecutionResult(success=True, stdout="hello\n", stderr="")
    assert format_execution_output(ok) == "hello"
    err = ExecutionResult(success=False, stdout="partial\n", stderr="", error="ValueError: x")
    out = format_execution_output(err)
    assert "partial" in out and "Error: ValueError: x" in out
    empty = ExecutionResult(success=True, stdout="", stderr="")
    assert format_execution_output(empty) == "(no output)"
    long = ExecutionResult(success=True, stdout="A" * 5000, stderr="")
    assert len(format_execution_output(long, max_chars=100)) <= 100 + len("\n...(truncated)")


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
