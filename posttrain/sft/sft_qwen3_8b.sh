#!/bin/bash
# Track A — Stage 1: light SFT (format primer) on Qwen3-8B, full fine-tune, 8xH100.
# TEMPLATE: confirm flags against your installed verl/TRL version.
#
# Qwen3-8B already does tool use, so this is a short run (~1-3h) that locks in the
# write->run->finalize format before GRPO. Build the data first:
#   python -m posttrain.data.prep_sft --out data/sft_toolcode.jsonl --repeat 50
set -euo pipefail
source .venv_posttrain/bin/activate

MODEL=${MODEL:-Qwen/Qwen3-8B}
DATA=${DATA:-data/sft_toolcode.jsonl}
OUT=${OUT:-checkpoints/sft_qwen3_8b}

# Option A (simplest): TRL SFTTrainer with FSDP via accelerate.
#   accelerate launch --config_file posttrain/sft/fsdp.yaml -m trl.scripts.sft \
#       --model_name_or_path "$MODEL" --dataset_name "$DATA" \
#       --bf16 --gradient_checkpointing --packing --max_seq_length 4096 \
#       --per_device_train_batch_size 4 --gradient_accumulation_steps 8 \
#       --learning_rate 1e-5 --num_train_epochs 2 --output_dir "$OUT"
#
# Option B: verl's SFT trainer (keeps one toolchain for SFT+RL):
#   torchrun --standalone --nproc_per_node=8 -m verl.trainer.fsdp_sft_trainer \
#       data.train_files="$DATA" model.partial_pretrain="$MODEL" \
#       trainer.default_local_dir="$OUT" trainer.total_epochs=2

echo "Edit this script for your SFT toolchain (TRL or verl). MODEL=$MODEL DATA=$DATA OUT=$OUT"
