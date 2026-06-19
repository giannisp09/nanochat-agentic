#!/bin/bash
# Track A — Stage 2 (THE MILESTONE): agentic coding GRPO on Qwen3-8B, 8xH100, ~3-6h.
# TEMPLATE: reconcile override keys with your verl version (see config_grpo.yaml).
#
# Prereqs:
#   bash posttrain/setup.sh
#   python -m posttrain.data.prep_rl_tasks --split train --out data/rl_train.parquet --format parquet
#   python -m posttrain.data.prep_rl_tasks --split test  --out data/rl_eval.parquet  --format parquet
#   bash posttrain/sft/sft_qwen3_8b.sh        # -> checkpoints/sft_qwen3_8b
set -euo pipefail
source .venv_posttrain/bin/activate

MODEL=${MODEL:-checkpoints/sft_qwen3_8b}

# verl reads config_grpo.yaml; CLI overrides shown for the knobs you'll tune per burst.
python -m verl.trainer.main_ppo \
    --config-path "$(pwd)/posttrain/rl" --config-name config_grpo \
    actor_rollout_ref.model.path="$MODEL" \
    actor_rollout_ref.rollout.n=8 \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.55 \
    data.train_batch_size=64 \
    data.max_response_length=1024 \
    trainer.save_freq=50 \
    "$@"

# To switch GRPO -> GSPO (e.g. if you move to a MoE policy):
#   algorithm.adv_estimator=grpo actor_rollout_ref.actor.policy_loss.importance_sampling_level=sequence
