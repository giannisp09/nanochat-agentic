#!/bin/bash
# One-shot setup for a fresh cloud GPU VM (Ubuntu + NVIDIA).
# Installs uv, all repo deps, and the HuggingFace CLI, then prints next steps.
#
# Usage (run from inside the cloned repo):
#   bash setup_vm.sh          # GPU box (CUDA 12.8 torch)  [default]
#   bash setup_vm.sh cpu      # CPU-only / MPS box
#
# Or bootstrap from scratch on a bare box:
#   git clone https://github.com/giannisp09/nanochat-agentic.git
#   cd nanochat-agentic && bash setup_vm.sh
set -euo pipefail

EXTRA="${1:-gpu}"   # gpu | cpu

echo "==> nanochat-agentic VM setup (extra=$EXTRA)"

# 1) system basics (skip silently if no apt / no sudo)
if command -v apt-get >/dev/null 2>&1; then
    echo "==> apt: git curl build-essential"
    SUDO=""; [ "$(id -u)" -ne 0 ] && command -v sudo >/dev/null 2>&1 && SUDO="sudo"
    $SUDO apt-get update -y && $SUDO apt-get install -y git curl build-essential || \
        echo "   (apt step skipped/failed — continuing; usually already present)"
fi

# 2) install uv if missing, put it on PATH for this shell
if ! command -v uv >/dev/null 2>&1; then
    echo "==> installing uv"
    curl -LsSf https://astral.sh/uv/install.sh | sh
fi
[ -f "$HOME/.local/bin/env" ] && source "$HOME/.local/bin/env"
export PATH="$HOME/.local/bin:$PATH"
command -v uv >/dev/null 2>&1 || { echo "ERROR: uv not on PATH after install"; exit 1; }
echo "==> uv: $(uv --version)"

# 3) sanity: show GPU (warn, don't fail — useful on a CPU box)
if [ "$EXTRA" = "gpu" ]; then
    if command -v nvidia-smi >/dev/null 2>&1; then
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true
    else
        echo "   WARNING: nvidia-smi not found. If this is a GPU box, the driver is missing."
        echo "            Pick an image with CUDA/PyTorch preinstalled, or rerun with: bash setup_vm.sh cpu"
    fi
fi

# 4) install all repo deps into .venv
echo "==> uv sync --extra $EXTRA  (this pulls torch + cuda, ~5-10 min)"
uv sync --extra "$EXTRA"

# 5) HuggingFace CLI (for runs/fetch_d32.sh) — not in pyproject
echo "==> installing huggingface_hub[cli]"
uv pip install "huggingface_hub[cli]"

echo ""
echo "================================================================"
echo " Setup complete. Next steps:"
echo ""
echo "   source .venv/bin/activate"
echo "   bash runs/fetch_d32.sh                       # download d32 (skip pretraining)"
echo "   NPROC_PER_NODE=1 bash runs/agent_speedrun.sh # 1-GPU box (use 8 on an 8xH100 node)"
echo "================================================================"
