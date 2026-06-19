# Track A — Agentic coding capability via post-training

Fast path to a genuinely useful agentic **coding** model by post-training an open
model (Qwen3-8B) with **verl + SGLang**, on a single **8×H100** node in bursts.
This is a *config layer over verl* — we don't reimplement RL infra.

> Realism anchor: DeepSWE (Qwen3-32B, R2E-Gym, GRPO++) used **64×H100 for 6 days
> (~$28K)** for 42% SWE-bench. Your budget is ~30–100× smaller, so **milestone 1
> is 8B + a light local coding suite** (MBPP+/HumanEval+-scale), not SWE-bench.

## Why these choices (see the root plan for full reasoning)
- **Qwen3-8B, full fine-tune** — fits full-FT GRPO + rollouts on 8×H100 (~16 GB/GPU
  training state; tune `gpu_memory_utilization≈0.55` for the colocated rollout).
  Stretch: 14B **LoRA** (`rl/grpo_lora_qwen_14b.sh`). Reach: 32B QLoRA / GLM-4.5-Air.
- **verl + SGLang, colocated** — first-class agentic RL (GRPO/GSPO, multi-turn
  tools); colocate train+rollout on one node. TRL `GRPOTrainer` is the escape hatch.
- **Reward = unit-test pass/fail** with partial credit (`envs/code_reward.py` →
  `shared/sandbox`). Tier 2 (SWE-style): swap in verl Sandbox Fusion.

## Milestone A1 (one ~6–10 h burst, ~$150–350)
```bash
bash posttrain/setup.sh                                  # verl+sglang+flash-attn (PIN versions)

# data (runs anywhere; bundled coding set, or point prep at MBPP+/HumanEval+)
python -m posttrain.data.prep_sft       --out data/sft_toolcode.jsonl --repeat 50
python -m posttrain.data.prep_rl_tasks  --split train --out data/rl_train.parquet --format parquet
python -m posttrain.data.prep_rl_tasks  --split test  --out data/rl_eval.parquet  --format parquet

bash posttrain/sft/sft_qwen3_8b.sh                       # light SFT primer (~1–3 h)
bash posttrain/rl/grpo_qwen3_8b.sh                       # agentic coding GRPO (~3–6 h)

bash posttrain/eval/run_eval.sh checkpoints/sft_qwen3_8b   # baseline pass@1
bash posttrain/eval/run_eval.sh checkpoints/grpo_qwen3_8b  # post-RL pass@1  (report the delta)
```
Headline metric: **held-out pass@1 before vs after GRPO.** Honest target at this
budget: single → low-double-digit points. Then demo via SGLang + nanochat's `ui.html`.

## What's real here vs. template
- **Real & unit-tested:** `envs/code_reward.py`, `shared/sandbox/execution.py`, the
  `data/prep_*.py` scripts (they produce datasets from the bundled coding env).
- **Templates (confirm against your pinned verl version):** `setup.sh`,
  `sft/*.sh`, `rl/*.sh`, `rl/config_grpo.yaml`, `envs/python_tool.py`. verl's
  config keys and tool API change across releases — reconcile after `setup.sh`.

## Interop with nanochat
- **Tokenizer:** NOT shared — Qwen3 uses its own.
- **Sandbox:** shared (`shared/sandbox`), reused by Track B too.
- **Demo UI:** reuse `nanochat/ui.html` pointed at the SGLang OpenAI endpoint.
