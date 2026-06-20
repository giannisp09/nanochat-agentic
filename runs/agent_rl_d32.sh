#!/bin/bash
# Agentic RL DIRECTLY on the released d32 — skips chat_sft entirely.
#
# d32 (karpathy/nanochat-d32) is ALREADY a chat-SFT model that knows the calculator
# tool, so we don't need to re-run SFT. This runs the agentic RL loop straight on it:
#   1) GSM8K calculator RL  -> sanity-checks GRPO + the multi-turn tool engine
#   2) coding RL            -> the milestone (sandboxed exec + verifiable reward)
#
#   bash runs/agent_rl_d32.sh                          # 8-GPU node
#   NPROC_PER_NODE=1 bash runs/agent_rl_d32.sh         # single-GPU box
#   WANDB_RUN=agentrl NPROC_PER_NODE=1 bash runs/agent_rl_d32.sh
set -euo pipefail

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
source .venv/bin/activate
WANDB_RUN="${WANDB_RUN:-dummy}"
NPROC="${NPROC_PER_NODE:-8}"

# agent_rl calls load_model("sft", ...), which reads chatsft_checkpoints/. d32 was
# downloaded into base_checkpoints/, so mirror it across (it IS an SFT model already).
SFT_DIR="$NANOCHAT_BASE_DIR/chatsft_checkpoints/d32"
BASE_DIR_D32="$NANOCHAT_BASE_DIR/base_checkpoints/d32"
if [ ! -f "$SFT_DIR/model_000650.pt" ]; then
    if [ ! -f "$BASE_DIR_D32/model_000650.pt" ]; then
        echo "ERROR: d32 not found in $BASE_DIR_D32. Run: bash runs/fetch_d32.sh" >&2
        exit 1
    fi
    mkdir -p "$SFT_DIR"
    cp "$BASE_DIR_D32/model_000650.pt" "$BASE_DIR_D32/meta_000650.json" "$SFT_DIR/"
    echo "Mirrored d32 into $SFT_DIR (agent_rl loads from chatsft_checkpoints)"
fi

# 1) SANITY: GSM8K calculator RL (dense reward) — validates the whole loop fast.
torchrun --standalone --nproc_per_node=$NPROC -m scripts.agent_rl -- \
    --task=gsm8k --objective=grpo --adv-norm=zscore \
    --group-size=16 --max-tool-turns=2 --num-epochs=1 --run=$WANDB_RUN

# 2) MILESTONE: agentic CODING RL with sandboxed execution + verifiable reward.
torchrun --standalone --nproc_per_node=$NPROC -m scripts.agent_rl -- \
    --task=coding --objective=grpo --adv-norm=zscore \
    --group-size=16 --max-tool-turns=4 --max-new-tokens=768 \
    --num-epochs=2 --run=$WANDB_RUN

echo "Done. Inspect pass@k in the agent_rl logs / wandb run '$WANDB_RUN'."
