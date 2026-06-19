#!/bin/bash

# Track A milestone (one command): post-train Qwen3-8B for agentic coding via verl.
# Assumes `bash posttrain/setup.sh` has been run (separate .venv_posttrain).
# See posttrain/README.md for the reasoning, memory math, and what's real vs template.
#
#   bash runs/trackA_milestone1.sh
set -euo pipefail
source .venv_posttrain/bin/activate

mkdir -p data

# 1) Data (runs anywhere). Swap in MBPP+/HumanEval+ via prep_rl_tasks --suite jsonl.
python -m posttrain.data.prep_sft      --out data/sft_toolcode.jsonl --repeat 50
python -m posttrain.data.prep_rl_tasks --split train --out data/rl_train.parquet --format parquet
python -m posttrain.data.prep_rl_tasks --split test  --out data/rl_eval.parquet  --format parquet

# 2) Light SFT primer (~1-3h), then 3) agentic coding GRPO (~3-6h)
bash posttrain/sft/sft_qwen3_8b.sh
bash posttrain/rl/grpo_qwen3_8b.sh

# 4) Headline metric: held-out pass@1 before vs after RL
bash posttrain/eval/run_eval.sh checkpoints/sft_qwen3_8b
bash posttrain/eval/run_eval.sh checkpoints/grpo_qwen3_8b
