"""
A self-contained, from-scratch coding environment with verifiable rewards.

This is the Track-B agentic-coding RL task. It bundles a small set of
function-completion problems (HumanEval-shaped: signature + docstring + hidden
unit tests) so the milestone is runnable with zero external downloads, then can
be swapped for MBPP+/HumanEval+ later via `from_jsonl`.

The reward is SHAPED to avoid the classic sparse-reward GRPO collapse at small
scale, where most early rollouts score 0 and the group advantage degenerates:

    1.0  all hidden tests pass
    0.1  code is valid Python but fails tests / errors at runtime  (partial credit)
    0.0  syntax error / no usable code

The hidden tests are executed in the sandbox and never shown to the model.
"""

from nanochat.execution import execute_code
from tasks.common import Task
from tasks.humaneval import extract_program, extract_imports

# -----------------------------------------------------------------------------
# Bundled starter problems. Each: prompt (stub the model completes), entry_point,
# a reference solution (for SFT), and hidden `check(f)` tests.
# Keep these small and unambiguous; extend freely or load a bigger suite via from_jsonl.

PROBLEMS = [
    {
        "prompt": 'def add(a, b):\n    """Return the sum of a and b."""\n',
        "entry_point": "add",
        "solution": "    return a + b\n",
        "tests": "def check(f):\n    assert f(1, 2) == 3\n    assert f(-1, 1) == 0\n    assert f(0, 0) == 0\n",
    },
    {
        "prompt": 'def is_even(n):\n    """Return True if n is even, else False."""\n',
        "entry_point": "is_even",
        "solution": "    return n % 2 == 0\n",
        "tests": "def check(f):\n    assert f(2) is True\n    assert f(3) is False\n    assert f(0) is True\n",
    },
    {
        "prompt": 'def factorial(n):\n    """Return n! for n >= 0."""\n',
        "entry_point": "factorial",
        "solution": "    r = 1\n    for i in range(2, n + 1):\n        r *= i\n    return r\n",
        "tests": "def check(f):\n    assert f(0) == 1\n    assert f(1) == 1\n    assert f(5) == 120\n",
    },
    {
        "prompt": 'def reverse_string(s):\n    """Return the string s reversed."""\n',
        "entry_point": "reverse_string",
        "solution": "    return s[::-1]\n",
        "tests": "def check(f):\n    assert f('abc') == 'cba'\n    assert f('') == ''\n    assert f('a') == 'a'\n",
    },
    {
        "prompt": 'def count_vowels(s):\n    """Return the number of vowels (aeiou) in s, case-insensitive."""\n',
        "entry_point": "count_vowels",
        "solution": "    return sum(c.lower() in 'aeiou' for c in s)\n",
        "tests": "def check(f):\n    assert f('hello') == 2\n    assert f('XYZ') == 0\n    assert f('AeIoU') == 5\n",
    },
    {
        "prompt": 'def max_of_list(xs):\n    """Return the maximum element of a non-empty list xs."""\n',
        "entry_point": "max_of_list",
        "solution": "    m = xs[0]\n    for x in xs[1:]:\n        if x > m:\n            m = x\n    return m\n",
        "tests": "def check(f):\n    assert f([1, 2, 3]) == 3\n    assert f([-5, -2, -9]) == -2\n    assert f([42]) == 42\n",
    },
    {
        "prompt": 'def fib(n):\n    """Return the n-th Fibonacci number (fib(0)=0, fib(1)=1)."""\n',
        "entry_point": "fib",
        "solution": "    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n",
        "tests": "def check(f):\n    assert f(0) == 0\n    assert f(1) == 1\n    assert f(10) == 55\n",
    },
    {
        "prompt": 'def is_palindrome(s):\n    """Return True if s reads the same forwards and backwards."""\n',
        "entry_point": "is_palindrome",
        "solution": "    return s == s[::-1]\n",
        "tests": "def check(f):\n    assert f('racecar') is True\n    assert f('abc') is False\n    assert f('') is True\n",
    },
    {
        "prompt": 'def sum_list(xs):\n    """Return the sum of all numbers in xs."""\n',
        "entry_point": "sum_list",
        "solution": "    total = 0\n    for x in xs:\n        total += x\n    return total\n",
        "tests": "def check(f):\n    assert f([1, 2, 3]) == 6\n    assert f([]) == 0\n    assert f([-1, 1]) == 0\n",
    },
    {
        "prompt": 'def gcd(a, b):\n    """Return the greatest common divisor of a and b."""\n',
        "entry_point": "gcd",
        "solution": "    while b:\n        a, b = b, a % b\n    return a\n",
        "tests": "def check(f):\n    assert f(12, 8) == 4\n    assert f(17, 5) == 1\n    assert f(100, 10) == 10\n",
    },
    {
        "prompt": 'def unique(xs):\n    """Return a list of the unique elements of xs, preserving first-seen order."""\n',
        "entry_point": "unique",
        "solution": "    seen = set()\n    out = []\n    for x in xs:\n        if x not in seen:\n            seen.add(x)\n            out.append(x)\n    return out\n",
        "tests": "def check(f):\n    assert f([1, 1, 2, 3, 3]) == [1, 2, 3]\n    assert f([]) == []\n    assert f([2, 1, 2]) == [2, 1]\n",
    },
    {
        "prompt": 'def title_case(s):\n    """Capitalize the first letter of each space-separated word in s."""\n',
        "entry_point": "title_case",
        "solution": "    return ' '.join(w[:1].upper() + w[1:] for w in s.split(' '))\n",
        "tests": "def check(f):\n    assert f('hello world') == 'Hello World'\n    assert f('a b c') == 'A B C'\n",
    },
    {
        "prompt": 'def second_largest(xs):\n    """Return the second largest distinct value in xs (len(set(xs)) >= 2)."""\n',
        "entry_point": "second_largest",
        "solution": "    s = sorted(set(xs))\n    return s[-2]\n",
        "tests": "def check(f):\n    assert f([1, 2, 3]) == 2\n    assert f([5, 5, 4]) == 4\n    assert f([-1, -2, -3]) == -2\n",
    },
    {
        "prompt": 'def flatten(xss):\n    """Flatten a list of lists into a single list."""\n',
        "entry_point": "flatten",
        "solution": "    out = []\n    for xs in xss:\n        out.extend(xs)\n    return out\n",
        "tests": "def check(f):\n    assert f([[1, 2], [3]]) == [1, 2, 3]\n    assert f([]) == []\n    assert f([[], [1]]) == [1]\n",
    },
    {
        "prompt": 'def count_words(s):\n    """Return the number of whitespace-separated words in s."""\n',
        "entry_point": "count_words",
        "solution": "    return len(s.split())\n",
        "tests": "def check(f):\n    assert f('hello world') == 2\n    assert f('') == 0\n    assert f('  a  b  c ') == 3\n",
    },
    {
        "prompt": 'def clamp(x, lo, hi):\n    """Return x clamped to the inclusive range [lo, hi]."""\n',
        "entry_point": "clamp",
        "solution": "    return max(lo, min(x, hi))\n",
        "tests": "def check(f):\n    assert f(5, 0, 10) == 5\n    assert f(-1, 0, 10) == 0\n    assert f(99, 0, 10) == 10\n",
    },
]

# Default split: most for training, a held-out tail for eval.
_DEFAULT_TRAIN = 12  # PROBLEMS[:12] train, PROBLEMS[12:] test

USER_INSTRUCTION = (
    "Complete the following Python function. You may write and run code with the "
    "python tool to test your work, then give your final solution in a ```python code block.\n\n"
)


class CodingEnv(Task):
    """
    Verifiable coding task. eval_type='generative'.

    split: 'train' | 'test' | 'all'
    as_sft: if True, include a reference assistant solution (for SFT seeding).
    tool_bonus: small reward added when the model actually used the python tool
                AND produced runnable code (>0 base). Default 0 (off) — the 0.1
                partial credit already prevents advantage collapse.
    """

    def __init__(self, split="train", as_sft=False, tool_bonus=0.0,
                 problems=None, **kwargs):
        super().__init__(**kwargs)
        all_problems = problems if problems is not None else PROBLEMS
        if split == "train":
            self.problems = all_problems[:_DEFAULT_TRAIN]
        elif split == "test":
            self.problems = all_problems[_DEFAULT_TRAIN:]
        elif split == "all":
            self.problems = all_problems
        else:
            raise ValueError(f"unknown split: {split!r}")
        self.as_sft = as_sft
        self.tool_bonus = tool_bonus

    @classmethod
    def from_jsonl(cls, path, **kwargs):
        """Load a larger suite (e.g. MBPP+/HumanEval+ converted to our schema)."""
        import json
        with open(path) as f:
            problems = [json.loads(line) for line in f if line.strip()]
        return cls(problems=problems, **kwargs)

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.problems)

    def get_example(self, index):
        p = self.problems[index]
        user = USER_INSTRUCTION + f"```python\n{p['prompt']}```"
        messages = [{"role": "user", "content": user}]
        if self.as_sft:
            full = p["prompt"] + p["solution"]
            messages.append({"role": "assistant", "content": f"```python\n{full}```"})
        return {
            "messages": messages,
            "entry_point": p["entry_point"],
            "tests": p["tests"],
            "imports": p.get("imports", ""),
        }

    def _build_program(self, conversation, completion):
        code = extract_program(completion)
        imports = conversation.get("imports", "")
        program = (
            imports + "\n\n" + code + "\n\n"
            + conversation["tests"] + "\n"
            + f"check({conversation['entry_point']})"
        )
        return code, program

    def evaluate(self, conversation, completion):
        """Binary pass: do all hidden tests pass?"""
        _, program = self._build_program(conversation, completion)
        return execute_code(program).success

    def reward(self, conversation, completion):
        """Shaped reward: 1.0 pass / 0.1 valid-but-fails / 0.0 unusable (+ optional tool bonus)."""
        code, program = self._build_program(conversation, completion)
        if execute_code(program).success:
            base = 1.0
        else:
            try:
                compile(code, "<solution>", "exec")
                base = 0.1  # valid syntax but wrong / runtime error
            except SyntaxError:
                base = 0.0
        if self.tool_bonus > 0.0 and base > 0.0 and self._used_tool(completion):
            base = min(1.0, base + self.tool_bonus)
        return base

    @staticmethod
    def _used_tool(completion):
        return "<|python_start|>" in completion or "python_start" in completion
