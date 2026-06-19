"""
Tests for the generalized multi-turn pluggable-tool state machine in engine.py.

Drives Engine._handle_tool_token directly with a fake tokenizer + fake tool, so
no model / GPU is needed.

Standalone:  .venv/bin/python tests/test_engine_tools.py
"""

from nanochat.engine import Engine, RowState
from nanochat.tools import Tool


class FakeTokenizer:
    """Maps each character to its ord() as a token id; special tokens use 1000+."""
    def __init__(self):
        self.specials = {
            "<|python_start|>": 1000,
            "<|python_end|>": 1001,
            "<|output_start|>": 1002,
            "<|output_end|>": 1003,
            "<|assistant_end|>": 1004,
        }
    def encode_special(self, s):
        return self.specials[s]
    def get_bos_token_id(self):
        return 1005
    def decode(self, ids):
        return "".join(chr(i) for i in ids)
    def encode(self, s):
        return [ord(c) for c in s]


class EchoTool(Tool):
    name = "echo"
    def __init__(self):
        self.sessions_made = 0
        self.calls = []
    def make_session(self):
        self.sessions_made += 1
        return {"id": self.sessions_made}
    def run(self, session, text):
        self.calls.append((session["id"], text))
        return f"OUT[{text}]"


def _emit(eng, state, token, max_tool_turns=None):
    eng._handle_tool_token(state, token, max_tool_turns)


def test_single_tool_turn_injects_output():
    eng = Engine(model=None, tokenizer=FakeTokenizer(), tools=[EchoTool()])
    state = RowState([])
    _emit(eng, state, 1000)               # <|python_start|>
    for ch in "hi":
        _emit(eng, state, ord(ch))
    _emit(eng, state, 1001)               # <|python_end|> -> run tool
    expected = [1002] + [ord(c) for c in "OUT[hi]"] + [1003]
    assert list(state.forced_tokens) == expected, list(state.forced_tokens)
    assert state.num_tool_turns == 1


def test_multi_turn_reuses_same_session():
    tool = EchoTool()
    eng = Engine(model=None, tokenizer=FakeTokenizer(), tools=[tool])
    state = RowState([])
    # turn 1
    _emit(eng, state, 1000)
    for ch in "aa":
        _emit(eng, state, ord(ch))
    _emit(eng, state, 1001)
    state.forced_tokens.clear()  # pretend they were emitted
    # turn 2
    _emit(eng, state, 1000)
    for ch in "bb":
        _emit(eng, state, ord(ch))
    _emit(eng, state, 1001)
    assert state.num_tool_turns == 2
    assert tool.sessions_made == 1, "a row must reuse one session across turns"
    assert tool.calls == [(1, "aa"), (1, "bb")]


def test_max_tool_turns_cap():
    tool = EchoTool()
    eng = Engine(model=None, tokenizer=FakeTokenizer(), tools=[tool])
    state = RowState([])
    # turn 1 (allowed)
    _emit(eng, state, 1000); _emit(eng, state, ord("x")); _emit(eng, state, 1001, max_tool_turns=1)
    state.forced_tokens.clear()
    # turn 2 (over the cap -> no execution, no output)
    _emit(eng, state, 1000); _emit(eng, state, ord("y")); _emit(eng, state, 1001, max_tool_turns=1)
    assert state.num_tool_turns == 1
    assert len(state.forced_tokens) == 0
    assert tool.calls == [(1, "x")]


def test_empty_block_does_nothing():
    eng = Engine(model=None, tokenizer=FakeTokenizer(), tools=[EchoTool()])
    state = RowState([])
    _emit(eng, state, 1000)
    _emit(eng, state, 1001)  # immediately close, empty buffer
    assert len(state.forced_tokens) == 0
    assert state.num_tool_turns == 0


def test_default_engine_uses_calculator():
    # No tools arg -> CalculatorTool, byte-identical to the original behavior.
    eng = Engine(model=None, tokenizer=FakeTokenizer())
    state = RowState([])
    _emit(eng, state, 1000)
    for ch in "2+2":
        _emit(eng, state, ord(ch))
    _emit(eng, state, 1001)
    expected = [1002] + [ord(c) for c in "4"] + [1003]
    assert list(state.forced_tokens) == expected, list(state.forced_tokens)


def test_non_tool_tokens_are_ignored_outside_block():
    eng = Engine(model=None, tokenizer=FakeTokenizer(), tools=[EchoTool()])
    state = RowState([])
    _emit(eng, state, ord("z"))   # random token, no active block
    assert state.active_start is None and len(state.forced_tokens) == 0


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
