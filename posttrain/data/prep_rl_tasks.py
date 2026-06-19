"""
Build a verl-style RL dataset for agentic coding.

Each row carries the chat `prompt` plus the hidden tests in `extra_info`, which
posttrain/envs/code_reward.py reads to score rollouts via the shared sandbox.

Default source is the bundled CodingEnv (runs anywhere, no downloads). For the
real milestone, convert MBPP+/HumanEval+ (EvalPlus) into the same schema and pass
--suite jsonl --jsonl-path <file> (each line: {prompt, solution, entry_point,
tests, imports}). Writes JSONL by default; --format parquet if pandas+pyarrow are
present (verl reads parquet).

    python -m posttrain.data.prep_rl_tasks --split train --out data/rl_train.jsonl
    python -m posttrain.data.prep_rl_tasks --split test  --out data/rl_eval.jsonl
"""

import argparse
import json
import os

from tasks.coding_env import CodingEnv


def build_records(split, suite, jsonl_path):
    if suite == "coding":
        env = CodingEnv(split=split)
    elif suite == "jsonl":
        assert jsonl_path, "--jsonl-path required for --suite jsonl"
        env = CodingEnv.from_jsonl(jsonl_path, split="all")
    else:
        raise ValueError(f"unknown suite: {suite}")

    records = []
    for i in range(len(env)):
        conv = env[i]
        user_msg = conv["messages"][0]  # CodingEnv emits a single user message in RL mode
        records.append({
            "data_source": f"nanochat_coding_{suite}",
            "ability": "coding",
            "prompt": [{"role": "user", "content": user_msg["content"]}],
            "reward_model": {"style": "rule", "ground_truth": conv["entry_point"]},
            "extra_info": {
                "index": i,
                "split": split,
                "entry_point": conv["entry_point"],
                "tests": conv["tests"],
                "imports": conv.get("imports", ""),
            },
        })
    return records


def main():
    ap = argparse.ArgumentParser(description="Build a verl RL dataset for agentic coding")
    ap.add_argument("--split", default="train", choices=["train", "test", "all"])
    ap.add_argument("--suite", default="coding", choices=["coding", "jsonl"])
    ap.add_argument("--jsonl-path", default=None, help="external problems jsonl (for --suite jsonl)")
    ap.add_argument("--out", required=True, help="output path (.jsonl or .parquet)")
    ap.add_argument("--format", default="jsonl", choices=["jsonl", "parquet"])
    args = ap.parse_args()

    records = build_records(args.split, args.suite, args.jsonl_path)
    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)

    if args.format == "parquet":
        import pandas as pd  # optional; verl consumes parquet
        pd.DataFrame(records).to_parquet(args.out)
    else:
        with open(args.out, "w") as f:
            for r in records:
                f.write(json.dumps(r) + "\n")

    print(f"Wrote {len(records)} {args.split} records -> {args.out}")


if __name__ == "__main__":
    main()
