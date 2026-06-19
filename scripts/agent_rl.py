"""
Agentic RL: real GRPO/GSPO with multi-turn tool use.

scripts/chat_rl.py is kept verbatim as the minimal REINFORCE *teaching baseline*.
This script generalizes it into proper GRPO/GSPO, where each upgrade over plain
REINFORCE is a single CLI toggle (the math lives in nanochat/rl_core.py):

    --objective {grpo,gspo}   token- vs sequence-level importance (GSPO ~ MoE)
    --adv-norm  {mean,zscore} group-normalized advantages (mean == chat_rl)
    --kl-beta   B             KL-to-reference trust region (loads a frozen ref model)
    --clip-eps  E             PPO ratio+clip (on-policy ratio==1, so this only
                              bites with off-policy rollout reuse; wired & ready)

Plus multi-turn tool use via the pluggable Engine tools, and a task switch:

    --task {gsm8k,coding,humaneval}
        gsm8k     -> CalculatorTool   (dense reward; sanity-checks the whole loop)
        coding    -> PythonReplTool    (sandboxed code exec; the agentic milestone)
        humaneval -> PythonReplTool    (small; train==eval)

With defaults (--objective grpo --adv-norm mean --kl-beta 0 --clip-eps 0) and
--task gsm8k this reproduces the chat_rl.py objective. See tests/test_rl_core.py.

1 GPU:
    python -m scripts.agent_rl --task=coding
8 GPUs:
    torchrun --standalone --nproc_per_node=8 -m scripts.agent_rl -- --task=coding --run=agentrl
"""

import argparse
import os
import itertools
import wandb
import torch
import torch.distributed as dist
from nanochat.common import compute_init, compute_cleanup, print0, get_base_dir, DummyWandb, autodetect_device_type
from nanochat.checkpoint_manager import save_checkpoint, load_model
from nanochat.engine import Engine
from nanochat.tools import CalculatorTool, PythonReplTool
from nanochat import rl_core

# -----------------------------------------------------------------------------
# CLI arguments
parser = argparse.ArgumentParser(description="Agentic RL (GRPO/GSPO) with multi-turn tool use")
# Logging
parser.add_argument("--run", type=str, default="dummy", help="wandb run name ('dummy' disables wandb logging)")
# Runtime
parser.add_argument("--device-type", type=str, default="", help="cuda|cpu|mps (empty = autodetect)")
# Model loading
parser.add_argument("--model-tag", type=str, default=None, help="model tag to load from")
parser.add_argument("--model-step", type=int, default=None, help="model step to load from")
# Task / environment
parser.add_argument("--task", type=str, default="gsm8k", choices=["gsm8k", "coding", "humaneval"], help="RL task/environment")
parser.add_argument("--max-tool-turns", type=int, default=4, help="max tool calls per rollout (None-like cap)")
parser.add_argument("--tool-bonus", type=float, default=0.0, help="(coding) small reward bonus for using the tool on runnable code")
# Training horizon
parser.add_argument("--num-epochs", type=int, default=1, help="number of epochs over the training task")
# Batch sizes / sampling
parser.add_argument("--device-batch-size", type=int, default=8, help="max batch size per forward pass")
parser.add_argument("--examples-per-step", type=int, default=16, help="total examples (prompts) per optimization step across all ranks")
parser.add_argument("--group-size", type=int, default=16, help="rollouts sampled per prompt (the GRPO group)")
# GRPO / GSPO knobs
parser.add_argument("--objective", type=str, default="grpo", choices=["grpo", "gspo"], help="policy-gradient objective")
parser.add_argument("--adv-norm", type=str, default="mean", choices=["mean", "zscore"], help="group advantage normalization")
parser.add_argument("--clip-eps", type=float, default=0.0, help="PPO clip epsilon (0 = off; only matters off-policy)")
parser.add_argument("--kl-beta", type=float, default=0.0, help="KL-to-reference penalty weight (0 = no reference model)")
# Generation
parser.add_argument("--max-new-tokens", type=int, default=512, help="max tokens to generate per rollout")
parser.add_argument("--temperature", type=float, default=1.0, help="sampling temperature")
parser.add_argument("--top-k", type=int, default=50, help="top-k sampling (0 = disabled)")
# Optimization
parser.add_argument("--embedding-lr", type=float, default=0.2, help="learning rate for embedding parameters (Adam)")
parser.add_argument("--unembedding-lr", type=float, default=0.004, help="learning rate for unembedding parameters (Adam)")
parser.add_argument("--matrix-lr", type=float, default=0.02, help="learning rate for matrix parameters (Muon)")
parser.add_argument("--weight-decay", type=float, default=0.0, help="weight decay (Adam)")
parser.add_argument("--init-lr-frac", type=float, default=0.05, help="initial LR as fraction of base LR")
# Evaluation / checkpointing
parser.add_argument("--eval-every", type=int, default=60, help="evaluate pass@k every N steps")
parser.add_argument("--eval-examples", type=int, default=400, help="number of examples for pass@k evaluation")
parser.add_argument("--save-every", type=int, default=60, help="save checkpoint every N steps")
args = parser.parse_args()
user_config = vars(args).copy()
# Use num_samples as an alias for group_size throughout (keep parity with chat_rl naming)
num_samples = args.group_size
# -----------------------------------------------------------------------------

# Init compute/precision
device_type = autodetect_device_type() if args.device_type == "" else args.device_type
ddp, ddp_rank, ddp_local_rank, ddp_world_size, device = compute_init(device_type)
master_process = ddp_rank == 0

# wandb logging init
use_dummy_wandb = args.run == "dummy" or not master_process
wandb_run = DummyWandb() if use_dummy_wandb else wandb.init(project="nanochat-rl", name=args.run, config=user_config)

# Init model and tokenizer
model, tokenizer, meta = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step)

# Optional frozen reference model for the KL trust region (only if --kl-beta > 0)
ref_model = None
if args.kl_beta > 0.0:
    ref_model, _, _ = load_model("sft", device, phase="eval", model_tag=args.model_tag, step=args.model_step)
    for p in ref_model.parameters():
        p.requires_grad_(False)
    ref_model.eval()
    print0("Loaded frozen reference model for KL penalty")

# -----------------------------------------------------------------------------
# Task / environment + tool selection

def build_task_and_tools(task_name):
    """Returns (train_task, val_task, tools). gsm8k uses the calculator; coding/humaneval the python REPL."""
    if task_name == "gsm8k":
        from tasks.gsm8k import GSM8K
        train = GSM8K(subset="main", split="train")
        val = GSM8K(subset="main", split="test")
        tools = [CalculatorTool()]
    elif task_name == "coding":
        from tasks.coding_env import CodingEnv
        train = CodingEnv(split="train", tool_bonus=args.tool_bonus)
        val = CodingEnv(split="test")
        tools = [PythonReplTool()]
    elif task_name == "humaneval":
        from tasks.humaneval import HumanEval
        train = HumanEval()
        val = HumanEval()  # tiny dataset; train==eval, just to exercise the loop
        tools = [PythonReplTool()]
    else:
        raise ValueError(f"unknown task: {task_name}")
    return train, val, tools

train_task, val_task, tools = build_task_and_tools(args.task)
engine = Engine(model, tokenizer, tools=tools)  # for sampling rollouts (with tools)

# reward(conversation, text) -> float. Tasks expose reward(); fall back to float(evaluate()).
def get_reward_fn(task):
    fn = getattr(task, "reward", None)
    if fn is not None:
        return fn
    return lambda conv, text: float(task.evaluate(conv, text))
reward_fn = get_reward_fn(train_task)

num_steps = (len(train_task) // args.examples_per_step) * args.num_epochs
print0(f"Task: {args.task} | train examples: {len(train_task)} | calculated number of steps: {num_steps}")

# -----------------------------------------------------------------------------
# Rollout / sampling generator: yields one prompt's group of `num_samples` rollouts

@torch.no_grad()
def get_batch():
    assistant_end = tokenizer.encode_special("<|assistant_end|>")  # used only for padding
    rank_indices = range(ddp_rank, len(train_task), ddp_world_size)
    for example_idx in itertools.cycle(rank_indices):
        conversation = train_task[example_idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)

        # Generate num_samples rollouts (with multi-turn tool use), in chunks to avoid OOM
        model.eval()
        generated_token_sequences = []
        masks = []
        assert num_samples % args.device_batch_size == 0, "group-size must be divisible by device-batch-size"
        num_sampling_steps = num_samples // args.device_batch_size
        for sampling_step in range(num_sampling_steps):
            seed = hash((step, example_idx, sampling_step)) & 0x7FFFFFFF
            seqs, msks = engine.generate_batch(
                tokens,
                num_samples=args.device_batch_size,
                max_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_k=args.top_k,
                seed=seed,
                max_tool_turns=args.max_tool_turns,
            )
            generated_token_sequences.extend(seqs)
            masks.extend(msks)

        # Rewards for each rollout (sandbox / verifier runs here)
        rewards = []
        for sample_tokens in generated_token_sequences:
            generated_tokens = sample_tokens[prefix_length:]
            generated_text = tokenizer.decode(generated_tokens)
            rewards.append(reward_fn(conversation, generated_text))

        # Pad to equal length in time
        max_length = max(len(seq) for seq in generated_token_sequences)
        padded = [seq + [assistant_end] * (max_length - len(seq)) for seq in generated_token_sequences]
        padded_masks = [m + [0] * (max_length - len(m)) for m in masks]
        ids = torch.tensor(padded, dtype=torch.long, device=device)
        mask_ids = torch.tensor(padded_masks, dtype=torch.long, device=device)
        inputs = ids[:, :-1]
        targets = ids[:, 1:].clone()
        targets[mask_ids[:, 1:] == 0] = -1  # ignore prompt + forced tool-output tokens in the loss
        rewards = torch.tensor(rewards, dtype=torch.float, device=device)
        # Group-relative advantages (the GR in GRPO). One prompt == one group of num_samples.
        advantages = rl_core.group_normalize_advantages(rewards, group_size=num_samples, mode=args.adv_norm)
        yield generated_token_sequences, inputs, targets, rewards, advantages

# -----------------------------------------------------------------------------
# Generic pass@k evaluation (works for any Task with .evaluate)

def run_eval(task, max_examples=None, num_samples=1, max_completion_tokens=512, temperature=0.0, top_k=50):
    max_examples = min(max_examples, len(task)) if max_examples is not None else len(task)
    for idx in range(ddp_rank, max_examples, ddp_world_size):
        conversation = task[idx]
        tokens = tokenizer.render_for_completion(conversation)
        prefix_length = len(tokens)
        assert num_samples <= args.device_batch_size
        seqs, _ = engine.generate_batch(
            tokens, num_samples=num_samples, max_tokens=max_completion_tokens,
            temperature=temperature, top_k=top_k, max_tool_turns=args.max_tool_turns,
        )
        outcomes = []
        for sample_tokens in seqs:
            generated_text = tokenizer.decode(sample_tokens[prefix_length:])
            outcomes.append({"is_correct": task.evaluate(conversation, generated_text)})
        yield {"idx": idx, "outcomes": outcomes}

# -----------------------------------------------------------------------------
# Training loop

optimizer = model.setup_optimizer(
    unembedding_lr=args.unembedding_lr,
    embedding_lr=args.embedding_lr,
    matrix_lr=args.matrix_lr,
    weight_decay=args.weight_decay,
)
for group in optimizer.param_groups:
    group["lr"] = group["lr"] * args.init_lr_frac
    group["initial_lr"] = group["lr"]

def get_lr_multiplier(it):
    return 1.0 - it / num_steps

print0(f"Total sequences per step: {args.examples_per_step * num_samples}")
assert args.examples_per_step % ddp_world_size == 0, "examples_per_step must be divisible by world size"
examples_per_rank = args.examples_per_step // ddp_world_size
print0(f"Calculated examples per rank: {examples_per_rank}")

batch_iterator = get_batch()
for step in range(num_steps):

    # Periodic pass@k evaluation
    if step % args.eval_every == 0:
        model.eval()
        passk = torch.zeros(args.device_batch_size, device=device)
        records = list(run_eval(val_task, max_examples=args.eval_examples, num_samples=args.device_batch_size, temperature=1.0))
        for k in range(1, args.device_batch_size + 1):
            passk[k - 1] = sum(any(o["is_correct"] for o in r["outcomes"][:k]) for r in records)
        num_records = torch.tensor(len(records), dtype=torch.long, device=device)
        if ddp:
            dist.all_reduce(num_records, op=dist.ReduceOp.SUM)
            dist.all_reduce(passk, op=dist.ReduceOp.SUM)
        passk = passk / max(num_records.item(), 1)
        print_passk = [f"Pass@{k}: {passk[k - 1].item():.4f}" for k in range(1, args.device_batch_size + 1)]
        print0(f"Step {step} | {', '.join(print_passk)}")
        wandb_run.log({"step": step, **{f"pass@{k}": passk[k - 1].item() for k in range(1, args.device_batch_size + 1)}})

    # Rollouts + policy-gradient update over examples_per_rank prompts (gradient accumulation)
    rewards_list = []
    sequence_lengths = []
    for example_step in range(examples_per_rank):
        sequences_all, inputs_all, targets_all, rewards_all, advantages_all = next(batch_iterator)
        model.train()
        assert inputs_all.size(0) % args.device_batch_size == 0
        num_passes = inputs_all.size(0) // args.device_batch_size
        for pass_idx in range(num_passes):
            b0, b1 = pass_idx * args.device_batch_size, (pass_idx + 1) * args.device_batch_size
            inputs = inputs_all[b0:b1]
            targets = targets_all[b0:b1]
            advantages = advantages_all[b0:b1]
            mask = (targets >= 0).float()
            # current-policy log-probs of taken tokens (0 at ignored positions)
            logp = -model(inputs, targets, loss_reduction='none').view_as(inputs)
            logp_old = logp.detach() if args.clip_eps > 0.0 else None  # on-policy: ratio==1
            logp_ref = None
            if args.kl_beta > 0.0:
                with torch.no_grad():
                    logp_ref = -ref_model(inputs, targets, loss_reduction='none').view_as(inputs)
            loss = rl_core.policy_loss(
                logp, advantages, mask,
                objective=args.objective,
                logp_old=logp_old, clip_eps=args.clip_eps,
                logp_ref=logp_ref, kl_beta=args.kl_beta,
            )
            # scale for gradient accumulation across passes and prompts
            loss = loss / (num_passes * examples_per_rank)
            loss.backward()
        rewards_list.append(rewards_all.mean().item())
        sequence_lengths.extend(len(seq) for seq in sequences_all)

    # Logging of rollout stats
    mean_reward = sum(rewards_list) / len(rewards_list)
    mean_sequence_length = sum(sequence_lengths) / len(sequence_lengths)
    if ddp:
        mr = torch.tensor(mean_reward, dtype=torch.float, device=device)
        ml = torch.tensor(mean_sequence_length, dtype=torch.float, device=device)
        dist.all_reduce(mr, op=dist.ReduceOp.AVG)
        dist.all_reduce(ml, op=dist.ReduceOp.AVG)
        mean_reward, mean_sequence_length = mr.item(), ml.item()
    print0(f"Step {step}/{num_steps} | objective {args.objective} | reward {mean_reward:.4f} | seqlen {mean_sequence_length:.1f}")
    wandb_run.log({"step": step, "reward": mean_reward, "sequence_length": mean_sequence_length})

    # Optimizer step (one per training step; gradients accumulated above)
    lrm = get_lr_multiplier(step)
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * lrm
    optimizer.step()
    model.zero_grad(set_to_none=True)
    wandb_run.log({"step": step, "lrm": lrm})

    # Checkpointing
    if master_process and ((step > 0 and step % args.save_every == 0) or step == num_steps - 1):
        base_dir = get_base_dir()
        depth = model.config.n_layer
        output_dirname = args.model_tag if args.model_tag else f"d{depth}"
        checkpoint_dir = os.path.join(base_dir, "agentrl_checkpoints", output_dirname)
        save_checkpoint(
            checkpoint_dir, step, model.state_dict(), None,
            {"model_config": model.config.__dict__},
        )
        print(f"✅ Saved model checkpoint to {checkpoint_dir}")

# Report + cleanup
from nanochat.report import get_report
get_report().log(section="Agentic RL", data=[user_config])
wandb_run.finish()
compute_cleanup()
