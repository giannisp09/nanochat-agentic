#!/bin/bash
# Download Karpathy's released nanochat d32 checkpoint into nanochat's cache layout,
# so you can SKIP base pretraining and go straight to the agentic speedrun.
#
#   bash runs/fetch_d32.sh
#   then: bash runs/agent_speedrun.sh
#
# The d32 release (huggingface.co/karpathy/nanochat-d32) is a *chat-SFT* checkpoint
# (step 650, already trained with the GSM8K calculator tool -> it knows the
# <|python_start|> format). We place the model under base_checkpoints/d32 so the
# agent_speedrun's coding/tool-use SFT seed (chat_sft) continues on top of it.
set -euo pipefail

# Activate the repo venv if present (so `hf` / python deps are available).
[ -f ".venv/bin/activate" ] && source .venv/bin/activate || true
PY="$(command -v python3 || command -v python)"

BASE_DIR="${NANOCHAT_BASE_DIR:-$HOME/.cache/nanochat}"
REPO="karpathy/nanochat-d32"

CKPT_DIR="$BASE_DIR/base_checkpoints/d32"
TOK_DIR="$BASE_DIR/tokenizer"
mkdir -p "$CKPT_DIR" "$TOK_DIR"

# Prefer the `hf` CLI; fall back to python huggingface_hub (version-stable).
if command -v hf >/dev/null 2>&1; then
    hf download "$REPO" model_000650.pt meta_000650.json --local-dir "$CKPT_DIR"
    hf download "$REPO" tokenizer.pkl token_bytes.pt    --local-dir "$TOK_DIR"
else
    "$PY" - "$REPO" "$CKPT_DIR" "$TOK_DIR" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo, ckpt_dir, tok_dir = sys.argv[1:4]
for fn, dst in [("model_000650.pt", ckpt_dir), ("meta_000650.json", ckpt_dir),
                ("tokenizer.pkl", tok_dir), ("token_bytes.pt", tok_dir)]:
    p = hf_hub_download(repo, fn, local_dir=dst)
    print("downloaded:", p)
PY
fi

echo ""
echo "Done. Layout:"
echo "  $CKPT_DIR/{model_000650.pt,meta_000650.json}"
echo "  $TOK_DIR/{tokenizer.pkl,token_bytes.pt}"
echo "Next:  bash runs/agent_speedrun.sh"
