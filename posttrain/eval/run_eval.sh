#!/bin/bash
# Serve a checkpoint and measure held-out pass@1. Usage:
#   bash posttrain/eval/run_eval.sh checkpoints/sft_qwen3_8b   # baseline
#   bash posttrain/eval/run_eval.sh checkpoints/grpo_qwen3_8b  # after RL
set -euo pipefail
source .venv_posttrain/bin/activate

CKPT=${1:?usage: run_eval.sh <checkpoint_dir>}
PORT=${PORT:-30000}

python -m sglang.launch_server --model "$CKPT" --port "$PORT" >/tmp/sglang_$PORT.log 2>&1 &
SERVER_PID=$!
trap 'kill $SERVER_PID 2>/dev/null || true' EXIT

# wait for the server to come up
for _ in $(seq 1 60); do
    curl -sf "http://localhost:$PORT/v1/models" >/dev/null 2>&1 && break || sleep 5
done

python -m posttrain.eval.pass_at_1 --base-url "http://localhost:$PORT/v1" --model "$CKPT" --n 1
