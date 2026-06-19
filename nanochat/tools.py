"""
Pluggable tools for the inference Engine.

nanochat's Engine originally hard-coded a single calculator tool inline. This
module generalizes that into a tiny `Tool` interface so the Engine can drive
*any* tool (and multiple turns of it) with the same token state machine:

    sample <|python_start|> ... <|python_end|>
      -> Tool.run(session, text) -> output string
      -> force-inject <|output_start|> output <|output_end|>
      -> keep sampling (next turn)

All tools reuse the same special tokens the tokenizer already defines
(<|python_start|> / <|python_end|> / <|output_start|> / <|output_end|>), so a
model trained once can be served with either the safe CalculatorTool (GSM8K)
or the full PythonReplTool (agentic coding) — you just pass a different tool
list to the Engine.

This module imports only from nanochat.execution (never from engine), so there
is no import cycle when engine.py imports the tools here.
"""

import signal
import warnings
from contextlib import contextmanager
from typing import Optional

from nanochat.execution import PersistentPythonSession, format_execution_output

# -----------------------------------------------------------------------------
# Calculator backend (moved here from engine.py, behavior preserved verbatim)

@contextmanager
def timeout(duration, formula):
    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")
    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception:
        signal.alarm(0)
        return None


def use_calculator(expr):
    """
    Evaluate a Python expression safely.
    Supports both math expressions and string operations like .count()
    """
    expr = expr.replace(",", "")
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # disallow power operator
            return None
        return eval_with_timeout(expr)
    allowed_chars = "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    if not all([x in allowed_chars for x in expr]):
        return None
    dangerous_patterns = ['__', 'import', 'exec', 'eval', 'compile', 'open', 'file',
                          'input', 'raw_input', 'globals', 'locals', 'vars', 'dir',
                          'getattr', 'setattr', 'delattr', 'hasattr']
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None
    if '.count(' not in expr:
        return None
    return eval_with_timeout(expr)

# -----------------------------------------------------------------------------
# Tool interface

class Tool:
    """
    Minimal tool interface. A tool is identified by the special token that opens
    its block; the Engine buffers tokens until the closing token, decodes them
    to text, calls run(), and force-injects the result between the output tokens.
    """
    name = "tool"
    start_token = "<|python_start|>"
    end_token = "<|python_end|>"
    output_start_token = "<|output_start|>"
    output_end_token = "<|output_end|>"

    def make_session(self):
        """Return a fresh per-row session object (for stateful tools), or None."""
        return None

    def run(self, session, text) -> Optional[str]:
        """
        Execute the tool on `text`. Return the output string to feed back to the
        model, or None to inject nothing (e.g. invalid/empty calculator input).
        """
        raise NotImplementedError

    def close_session(self, session):
        """Clean up a per-row session (override for stateful tools)."""
        if session is not None and hasattr(session, "close"):
            session.close()


class CalculatorTool(Tool):
    """Safe arithmetic / string-.count() calculator (the original nanochat tool)."""
    name = "calculator"

    def run(self, session, text) -> Optional[str]:
        result = use_calculator(text)
        return None if result is None else str(result)


class PythonReplTool(Tool):
    """
    Full sandboxed Python REPL with state persisting across turns. Backed by
    PersistentPythonSession (spawned subprocess, timeout + memory limits).
    Use this for agentic coding RL/eval; use CalculatorTool for GSM8K.
    """
    name = "python"

    def __init__(self, timeout: float = 5.0, max_output_chars: int = 1500,
                 maximum_memory_bytes: Optional[int] = 256 * 1024 * 1024):
        self.timeout = timeout
        self.max_output_chars = max_output_chars
        self.maximum_memory_bytes = maximum_memory_bytes

    def make_session(self):
        return PersistentPythonSession(timeout=self.timeout,
                                       maximum_memory_bytes=self.maximum_memory_bytes)

    def run(self, session, text) -> Optional[str]:
        if session is None:
            session = self.make_session()
        result = session.run(text, timeout=self.timeout)
        return format_execution_output(result, max_chars=self.max_output_chars)
