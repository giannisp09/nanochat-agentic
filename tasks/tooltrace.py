"""
SFT-seed data: verified multi-turn tool-use trajectories for coding.

RL from a cold policy that never emits <|python_start|> almost never earns
reward, so GRPO stalls. This task manufactures SFT conversations that teach the
*format and reflex* of tool use: write code -> run it with the python tool ->
read the output -> give the final answer. Crucially, the python_output part is
REAL — produced by actually running the code in the sandbox (execute_code) — not
hallucinated, mirroring how dev/gen_synthetic_data.py builds SFT data.

These render through tokenizer.render_conversation for free: the 'python' part is
supervised, the 'python_output' part is masked out (mask=0), exactly as for
GSM8K's calculator traces.
"""

import re

from tasks.common import Task
from tasks.coding_env import CodingEnv, USER_INSTRUCTION
from nanochat.execution import execute_code

# Pull "print(entry(args))" demo calls out of the hidden `assert f(args) == ...` tests.
_ASSERT_RE = re.compile(r"assert\s+f\((.*?)\)\s*(?:==|is)")


def _demo_calls(tests, entry_point, max_calls=3):
    args_list = _ASSERT_RE.findall(tests)[:max_calls]
    return "\n".join(f"print({entry_point}({args}))" for args in args_list)


class CodingToolTrace(Task):
    """Generative SFT task of authentic write->run->finalize coding trajectories."""

    def __init__(self, split="train", **kwargs):
        super().__init__(**kwargs)
        self.problems = CodingEnv(split=split).problems
        self._output_cache = {}

    @property
    def eval_type(self):
        return "generative"

    def num_examples(self):
        return len(self.problems)

    def _real_output(self, index, demo_code):
        if index not in self._output_cache:
            res = execute_code(demo_code)
            self._output_cache[index] = res.stdout if res.success else f"Error: {res.error}"
        return self._output_cache[index]

    def get_example(self, index):
        p = self.problems[index]
        full = p["prompt"] + p["solution"]              # the reference solution
        entry = p["entry_point"]
        demo = full + "\n" + _demo_calls(p["tests"], entry)
        output_text = self._real_output(index, demo)    # genuine sandbox stdout

        user = USER_INSTRUCTION + f"```python\n{p['prompt']}```"
        assistant_parts = [
            {"type": "text", "text": "Let me implement this and check it with the python tool.\n"},
            {"type": "python", "text": demo},
            {"type": "python_output", "text": output_text},
            {"type": "text", "text": f"The outputs look correct. Final solution:\n```python\n{full}```"},
        ]
        messages = [
            {"role": "user", "content": user},
            {"role": "assistant", "content": assistant_parts},
        ]
        return {"messages": messages, "entry_point": entry, "tests": p["tests"]}
