"""
Tests for nanochat/tools.py (Tool ABC, CalculatorTool, PythonReplTool).

Standalone:  .venv/bin/python tests/test_tools.py
"""

from nanochat.tools import CalculatorTool, PythonReplTool, use_calculator, Tool


def test_calculator_basic():
    tool = CalculatorTool()
    assert tool.run(None, "2+2") == "4"
    assert tool.run(None, "12*60") == "720"
    assert tool.run(None, "1,000+1") == "1001"   # commas stripped
    assert tool.run(None, "2**8") is None          # power operator disallowed
    assert tool.run(None, "__import__('os')") is None
    # string .count() is supported
    assert tool.run(None, "'strawberry'.count('r')") == "3"


def test_calculator_matches_original_engine_behavior():
    # tools.use_calculator is a verbatim move of engine.use_calculator; confirm parity.
    from nanochat.engine import use_calculator as engine_use_calculator
    cases = ["2+2", "12*60", "1,000+1", "2**8", "abc", "'mississippi'.count('s')",
             "open('x')", "3.5*2", "(1+2)*3"]
    for c in cases:
        assert use_calculator(c) == engine_use_calculator(c), f"mismatch on {c!r}"


def test_tool_tokens_present():
    for tool in (CalculatorTool(), PythonReplTool()):
        assert isinstance(tool, Tool)
        assert tool.start_token == "<|python_start|>"
        assert tool.end_token == "<|python_end|>"
        assert tool.output_start_token == "<|output_start|>"
        assert tool.output_end_token == "<|output_end|>"


def test_python_repl_runs_and_persists_state():
    tool = PythonReplTool()
    sess = tool.make_session()
    try:
        assert tool.run(sess, "a = 10") == "(no output)"
        assert tool.run(sess, "print(a * 2)") == "20"
        assert "Error: NameError" in tool.run(sess, "print(missing)")
    finally:
        tool.close_session(sess)


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
