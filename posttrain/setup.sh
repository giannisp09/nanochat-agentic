#!/bin/bash
# Track A environment: verl + SGLang + flash-attn, kept SEPARATE from nanochat's uv env.
# Run once on the 8xH100 node. PIN versions (verl's config schema & tool API move fast).
#
# Do this on a cheap/idle session if possible — setup friction (Ray, flash-attn build,
# version matrix) can otherwise eat into a paid GPU burst.
set -euo pipefail

# A dedicated venv so Track A (HF/verl) never collides with nanochat's deps.
python -m venv .venv_posttrain
source .venv_posttrain/bin/activate
pip install --upgrade pip

# --- PIN THESE to a known-good matrix for your CUDA/driver before a real run ---
# verl pulls torch/vllm/sglang transitively; confirm the trio is mutually compatible.
pip install "verl"                 # e.g. verl==0.5.x  (agentic rollout + GRPO/GSPO)
pip install "sglang"               # rollout engine (verl default for multi-turn tools)
pip install flash-attn --no-build-isolation
pip install datasets pandas pyarrow huggingface_hub

echo
echo "Setup done. Confirm versions and the verl config schema for THIS verl release:"
pip show verl sglang | grep -E "Name|Version" || true
echo "Then edit rl/config_grpo.yaml + rl/*.sh to match (keys differ across verl versions)."
