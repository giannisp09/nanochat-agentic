#!/bin/bash
# Track A — STRETCH: 14B LoRA GRPO. verl ships an official 14B-LoRA-on-2xH100 recipe,
# so on 8xH100 this is roomy (bigger groups / faster wall-clock). Use AFTER the 8B
# full-FT loop is green. TEMPLATE: confirm LoRA keys against your verl version.
set -euo pipefail
source .venv_posttrain/bin/activate

MODEL=${MODEL:-Qwen/Qwen2.5-Coder-14B-Instruct}   # or Qwen/Qwen3-14B

python -m verl.trainer.main_ppo \
    --config-path "$(pwd)/posttrain/rl" --config-name config_grpo \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.model.lora_rank=32 \
    actor_rollout_ref.model.lora_alpha=32 \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.5 \
    data.train_batch_size=64 \
    "$@"

# Reference: verl/examples/tuning/14b/qwen2-14b_grpo-lora_2_h100_fsdp_vllm.sh
