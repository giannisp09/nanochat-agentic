"""
Held-out pass@1 (and pass@k) for a post-trained coding model served over an
OpenAI-compatible endpoint (SGLang / vLLM). This is the headline Track A metric:
run it on the SFT checkpoint and the GRPO checkpoint and report the delta.

Serve first, e.g.:
    python -m sglang.launch_server --model checkpoints/grpo_qwen3_8b --port 30000

Then:
    python -m posttrain.eval.pass_at_1 --base-url http://localhost:30000/v1 \
        --model checkpoints/grpo_qwen3_8b --n 1

Uses the bundled CodingEnv test split by default (swap to MBPP+/HumanEval+ via
CodingEnv.from_jsonl). Scoring runs the hidden tests in the shared sandbox.
Stdlib-only HTTP (urllib), so it has no extra deps.
"""

import argparse
import json
import urllib.request

from tasks.coding_env import CodingEnv
from tasks.humaneval import extract_program
from shared.sandbox.execution import run_unit_tests


def chat_complete(base_url, model, messages, n, temperature, max_tokens):
    body = json.dumps({
        "model": model, "messages": messages, "n": n,
        "temperature": temperature, "max_tokens": max_tokens,
    }).encode()
    req = urllib.request.Request(base_url.rstrip("/") + "/chat/completions",
                                 data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req) as resp:
        out = json.loads(resp.read())
    return [c["message"]["content"] for c in out["choices"]]


def main():
    ap = argparse.ArgumentParser(description="pass@1/pass@k for a served coding model")
    ap.add_argument("--base-url", required=True, help="OpenAI-compatible base url, e.g. http://localhost:30000/v1")
    ap.add_argument("--model", required=True)
    ap.add_argument("--split", default="test", choices=["test", "train", "all"])
    ap.add_argument("--n", type=int, default=1, help="samples per problem (pass@k over k=1..n)")
    ap.add_argument("--temperature", type=float, default=0.2)
    ap.add_argument("--max-tokens", type=int, default=1024)
    args = ap.parse_args()

    env = CodingEnv(split=args.split)
    passk = [0] * args.n
    for i in range(len(env)):
        conv = env[i]
        completions = chat_complete(args.base_url, args.model, conv["messages"],
                                    args.n, args.temperature, args.max_tokens)
        corrects = []
        for c in completions:
            code = extract_program(c)
            corrects.append(run_unit_tests(code, conv["tests"], conv["entry_point"]).success)
        for k in range(1, args.n + 1):
            if any(corrects[:k]):
                passk[k - 1] += 1

    n = len(env)
    for k in range(1, args.n + 1):
        print(f"pass@{k}: {passk[k - 1] / max(n, 1):.4f}  ({passk[k - 1]}/{n})")


if __name__ == "__main__":
    main()
