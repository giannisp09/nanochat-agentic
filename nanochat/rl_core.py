"""
Policy-gradient math for RL fine-tuning, isolated as pure functions.

The point of this module is pedagogical *and* practical: the toy RL in
`scripts/chat_rl.py` is REINFORCE (advantage = r - mean, no groups, no clip,
no KL). Real GRPO/GSPO add four things on top, each a one-line toggle here:

    1) group-normalized advantages         -> group_normalize_advantages
    2) PPO ratio + clip (off-policy reuse)  -> token_pg_loss / sequence_pg_loss
    3) KL-to-reference trust region         -> kl_penalty
    4) sequence-level importance (GSPO)     -> sequence_pg_loss

Design notes
------------
- Everything operates on plain tensors so it is trivially unit-testable on CPU
  (see tests/test_rl_core.py), with no model or GPU required.
- `logp` is the log-prob of the *taken* token at each position, shape (B, T).
  In `chat_rl.py` this is `-model(inputs, targets, loss_reduction='none')`.
- `mask` is (B, T) in {0, 1}: 1 for tokens we train on (assistant, non-tool),
  0 for prompt / forced tool-output / padding tokens. This is exactly
  `(targets >= 0)` in the training script.
- `advantages` is (B,) -- one scalar per sampled trajectory.

Equivalence guarantee: with `objective="grpo"`, `clip_eps=0.0`, `kl_beta=0.0`
and a single group, `policy_loss` produces the *same* objective as the current
`chat_rl.py` (token-level normalization, advantage = r - mean). Turning each
knob on is what upgrades REINFORCE -> GRPO -> GSPO. See test_rl_core.py.
"""

import torch

# -----------------------------------------------------------------------------
# 1) Advantages

def group_normalize_advantages(rewards, group_size, mode="mean", eps=1e-6):
    """
    Group Relative advantages (the "GR" in GRPO).

    rewards: (N,) float tensor, where N is divisible by group_size and the
             samples are laid out so that each contiguous block of `group_size`
             entries are the rollouts for one prompt.
    group_size: number of rollouts per prompt (the group).
    mode: "mean"   -> A = r - mean(group)                 (chat_rl.py default)
          "zscore" -> A = (r - mean(group)) / (std + eps)  (canonical GRPO)
    Returns: (N,) advantages.

    With group_size == N this reduces to a single group, i.e. exactly
    `rewards - rewards.mean()` (mode="mean"), matching chat_rl.py:143-144.
    """
    assert rewards.dim() == 1, f"rewards must be 1D, got shape {tuple(rewards.shape)}"
    n = rewards.shape[0]
    assert group_size >= 1 and n % group_size == 0, \
        f"N={n} must be divisible by group_size={group_size}"
    groups = rewards.view(n // group_size, group_size)
    mu = groups.mean(dim=1, keepdim=True)
    adv = groups - mu
    if mode == "zscore":
        # population std (unbiased=False) so a single-element / constant group
        # yields std=0 -> adv=0 rather than NaN.
        std = groups.std(dim=1, keepdim=True, unbiased=False)
        adv = adv / (std + eps)
    elif mode != "mean":
        raise ValueError(f"unknown adv-norm mode: {mode!r} (use 'mean' or 'zscore')")
    return adv.reshape(n)

# -----------------------------------------------------------------------------
# 2) Per-token policy loss (GRPO / REINFORCE / token-level PPO)

def token_pg_loss(logp, advantages, mask, logp_old=None, clip_eps=0.0):
    """
    Token-level policy-gradient loss, returned per token as (B, T) (already
    multiplied by `mask`, so masked positions contribute exactly 0).

    logp:        (B, T) log-prob of the taken token under the current policy.
    advantages:  (B,) per-sequence advantage (broadcast across time).
    mask:        (B, T) in {0, 1}.
    logp_old:    (B, T) log-prob under the rollout (behavior) policy. Required
                 only when clip_eps > 0 (off-policy reuse of rollouts).
    clip_eps:    PPO clip epsilon. If 0 (default) -> plain on-policy REINFORCE,
                 i.e. -(logp * adv). This is the chat_rl.py objective.

    On-policy note: with clip_eps>0 and logp_old = logp.detach() (a single grad
    step on fresh rollouts), the *gradient* of this loss equals the REINFORCE
    gradient at ratio==1 (verified in tests). So clipping only changes behavior
    once you actually go off-policy (ppo_epochs>1 or stale rollouts).
    """
    adv = advantages.unsqueeze(1)  # (B, 1)
    if clip_eps and clip_eps > 0.0:
        assert logp_old is not None, "clip_eps>0 requires logp_old (the rollout log-probs)"
        ratio = torch.exp(logp - logp_old)          # (B, T)
        unclipped = ratio * adv
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * adv
        per_token = -torch.min(unclipped, clipped)
    else:
        per_token = -(logp * adv)                   # REINFORCE
    return per_token * mask

# -----------------------------------------------------------------------------
# 2b) Sequence-level policy loss (GSPO)

def sequence_pg_loss(logp, advantages, mask, logp_old=None, clip_eps=0.2):
    """
    GSPO (Group Sequence Policy Optimization): the importance ratio is computed
    once per *sequence* from the length-normalized sequence log-prob, instead of
    per token. Returns per-sequence loss (B,). This is the token-vs-sequence
    importance-weighting distinction between GRPO and GSPO.

    GSPO is the more stable choice for MoE policies; for dense models GRPO
    (token_pg_loss) is the usual default.
    """
    seq_len = mask.sum(dim=1).clamp(min=1)                 # (B,)
    seq_logp = (logp * mask).sum(dim=1) / seq_len          # (B,)
    if clip_eps and clip_eps > 0.0:
        assert logp_old is not None, "GSPO clip requires logp_old"
        seq_logp_old = (logp_old * mask).sum(dim=1) / seq_len
        ratio = torch.exp(seq_logp - seq_logp_old)         # (B,)
        unclipped = ratio * advantages
        clipped = torch.clamp(ratio, 1.0 - clip_eps, 1.0 + clip_eps) * advantages
        return -torch.min(unclipped, clipped)
    else:
        return -(seq_logp * advantages)

# -----------------------------------------------------------------------------
# 3) KL-to-reference penalty (optional trust region)

def kl_penalty(logp, logp_ref, mask):
    """
    Per-token KL penalty using the low-variance, unbiased k3 estimator
    (Schulman): kl = exp(r) - r - 1, where r = logp_ref - logp. This is always
    >= 0 and == 0 iff logp == logp_ref. Returned per token as (B, T), masked.

    Add `kl_beta * kl_penalty(...)` to the per-token loss to pull the policy
    toward a frozen reference (the SFT checkpoint). Off by default (beta=0) to
    preserve chat_rl.py's "no reference model" simplicity.
    """
    diff = logp_ref - logp
    kl = torch.exp(diff) - diff - 1.0
    return kl * mask

# -----------------------------------------------------------------------------
# Reductions

def masked_token_mean(per_token, mask):
    """
    Sum over all tokens divided by the number of valid tokens (DAPO / chat_rl
    token-level normalization). Returns a scalar.
    """
    denom = mask.sum().clamp(min=1)
    return per_token.sum() / denom

# -----------------------------------------------------------------------------
# Convenience: compose into a single scalar loss

def policy_loss(logp, advantages, mask, *, objective="grpo",
                logp_old=None, clip_eps=0.0, logp_ref=None, kl_beta=0.0):
    """
    Compose the pieces above into a single scalar loss to minimize.

    objective="grpo": token-level loss, normalized by valid token count.
    objective="gspo": sequence-level loss, averaged over sequences.

    With objective="grpo", clip_eps=0, kl_beta=0 and a single advantage group,
    this is exactly the chat_rl.py objective (token-level REINFORCE). Each kwarg
    is the corresponding GRPO/GSPO upgrade.
    """
    if objective == "grpo":
        per_token = token_pg_loss(logp, advantages, mask, logp_old=logp_old, clip_eps=clip_eps)
        if kl_beta and kl_beta > 0.0:
            assert logp_ref is not None, "kl_beta>0 requires logp_ref"
            per_token = per_token + kl_beta * kl_penalty(logp, logp_ref, mask)
        return masked_token_mean(per_token, mask)
    elif objective == "gspo":
        per_seq = sequence_pg_loss(logp, advantages, mask, logp_old=logp_old, clip_eps=clip_eps)
        loss = per_seq.mean()
        if kl_beta and kl_beta > 0.0:
            assert logp_ref is not None, "kl_beta>0 requires logp_ref"
            # sequence-averaged KL to match the sequence-level objective
            seq_len = mask.sum(dim=1).clamp(min=1)
            kl_seq = (kl_penalty(logp, logp_ref, mask).sum(dim=1) / seq_len)
            loss = loss + kl_beta * kl_seq.mean()
        return loss
    else:
        raise ValueError(f"unknown objective: {objective!r} (use 'grpo' or 'gspo')")
