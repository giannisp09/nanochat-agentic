#!/bin/bash

# Track B milestone: agentic tool-use + coding RL FROM SCRATCH on a nanochat model.
# Picks up AFTER pretraining — run `bash runs/speedrun.sh` first to get a base model
# (this script re-does SFT with a coding/tool-use seed, then runs agentic RL).
#
# Designed for an 8XH100 node. Rough time on d24: SFT seed ~20-40m, GSM8K calc RL
# ~1-2h, coding RL ~3-6h => comfortably one ~1-day burst.
#
#   bash runs/agent_speedrun.sh
#   WANDB_RUN=agentrl bash runs/agent_speedrun.sh   # with logging

export OMP_NUM_THREADS=1
export NANOCHAT_BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
source .venv/bin/activate
WANDB_RUN="${WANDB_RUN:-dummy}"
NPROC="${NPROC_PER_NODE:-8}"

# -----------------------------------------------------------------------------
# 0) SFT with a coding/tool-use SEED so RL starts from a warm, format-correct policy.
#    --coding-epochs oversamples the bundled CodingToolTrace; for a stronger primer,
#    point CodingEnv/CodingToolTrace at a larger corpus (e.g. MBPP+/HumanEval+).
torchrun --standalone --nproc_per_node=$NPROC -m scripts.chat_sft -- \
    --device-batch-size=16 --coding-epochs=30 --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# 1) SANITY: multi-turn CALCULATOR RL on GSM8K (dense reward) — validates the whole
#    GRPO + multi-turn-tool loop quickly before the sparse coding env.
torchrun --standalone --nproc_per_node=$NPROC -m scripts.agent_rl -- \
    --task=gsm8k --objective=grpo --adv-norm=zscore \
    --group-size=16 --max-tool-turns=2 --num-epochs=1 --run=$WANDB_RUN

# -----------------------------------------------------------------------------
# 2) THE MILESTONE: agentic CODING RL with sandboxed execution + verifiable reward.
#    Partial-credit reward (1.0 / 0.1 / 0.0) prevents sparse-reward GRPO collapse.
torchrun --standalone --nproc_per_node=$NPROC -m scripts.agent_rl -- \
    --task=coding --objective=grpo --adv-norm=zscore \
    --group-size=16 --max-tool-turns=4 --max-new-tokens=768 \
    --num-epochs=2 --run=$WANDB_RUN

# Pass@k on the held-out coding split is logged by agent_rl during training (--eval-every).
# Try GSPO instead of GRPO: add --objective=gspo. Add a trust region: --kl-beta=1e-3.
echo "Done. Inspect pass@k in the agent_rl logs / wandb run '$WANDB_RUN'."
