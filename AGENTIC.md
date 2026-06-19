# Agentic expansion of nanochat (Track A + Track B)

This fork extends nanochat toward **agentic, GLM-class** capability along two tracks,
both starting at the same milestone — **agentic tool-use + coding RL** — from opposite
ends. The full reasoning and roadmap live in the approved plan
(`~/.claude/plans/i-want-to-use-greedy-dongarra.md`); this is the implementation map.

> **Honest scope.** You cannot train a GLM-5.1-class model by scaling nanochat
> (frontier = 355B+ MoE, 15–30T tokens, thousands of GPUs, $10M+). On an **8×H100
> burst** budget the realistic wins are: (A) real agentic coding capability by
> *post-training open weights*, and (B) *learning the machinery* by building it
> from scratch at small scale.

## Track A — capability via post-training (priority)
`posttrain/` — a config layer over **verl + SGLang**: Qwen3-8B → light SFT → GRPO with
a sandbox unit-test reward → held-out pass@1. Runs on the GPU node.
See **`posttrain/README.md`**. One command: `bash runs/trackA_milestone1.sh`.

## Track B — from scratch on nanochat (learning)
Extends nanochat itself, preserving the single-`--depth` dial and minimal style.
One command (after `runs/speedrun.sh`): **`bash runs/agent_speedrun.sh`**
(coding-seeded SFT → GSM8K calculator-RL sanity → agentic coding RL).

What was added/changed for Track B:
| File | What |
|---|---|
| `nanochat/rl_core.py` (+) | GRPO/GSPO/KL math as pure functions; reduces exactly to chat_rl REINFORCE |
| `nanochat/tools.py` (+) | `Tool` ABC, `CalculatorTool`, `PythonReplTool` (calculator moved here) |
| `nanochat/execution.py` (~) | `PersistentPythonSession` — stateful REPL (spawn-safe under CUDA) |
| `nanochat/engine.py` (~) | multi-turn **pluggable** tool state machine; `Engine(model, tokenizer)` unchanged |
| `tasks/coding_env.py` (+) | verifiable coding task, shaped reward (1.0 / 0.1 / 0.0) anti-collapse |
| `tasks/tooltrace.py` (+) | SFT seed: authentic write→run→finalize trajectories (real sandbox output) |
| `scripts/agent_rl.py` (+) | real GRPO/GSPO loop w/ multi-turn tools (`scripts/chat_rl.py` kept as baseline) |
| `scripts/chat_sft.py` (~) | `--coding-epochs` hook (default 0 → stock SFT unchanged) |

Later Track B stages (in the plan, not yet built): reasoning RLVR/thinking mode →
MoE learning module → long-context (RoPE/YaRN) → FSDP2/TP.

## Shared
`shared/sandbox/execution.py` — unit-test pass-rate reward used by both tracks.

## What's verified locally (CPU/MPS) vs. needs a GPU
Verified now (no GPU) — run `bash runs/test_agentic.sh` or:
```
for t in tests/test_rl_core.py tests/test_execution_session.py tests/test_tools.py \
         tests/test_coding_env.py tests/test_engine_tools.py tests/test_posttrain_reward.py; do
  PYTHONPATH=. .venv/bin/python "$t"; done
```
Covers: the GRPO/GSPO/KL math, the stateful sandbox, the pluggable multi-turn tool
state machine, the verifiable coding reward, and the Track A reward path — **all green**.

Needs the 8×H100 burst (can't run on a Mac): `scripts/agent_rl.py` (Track B RL),
the whole `posttrain/` verl pipeline (Track A), and any real training. The verl
launchers/configs in `posttrain/` are **templates** to reconcile with your pinned
verl version after `posttrain/setup.sh`.
