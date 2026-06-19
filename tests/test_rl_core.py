"""
Tests for nanochat/rl_core.py — the GRPO/GSPO/KL policy-gradient math.

Runs on CPU with no model required. Works under pytest, and also standalone:

    .venv/bin/python tests/test_rl_core.py
"""

import torch

from nanochat.rl_core import (
    group_normalize_advantages,
    token_pg_loss,
    sequence_pg_loss,
    kl_penalty,
    masked_token_mean,
    policy_loss,
)

torch.manual_seed(0)


def _chat_rl_reference_loss(logp, rewards, mask):
    """Replicates the exact objective in scripts/chat_rl.py:143-272 (single group)."""
    mu = rewards.mean()
    advantages = rewards - mu                      # (B,)
    pg_obj = (logp * advantages.unsqueeze(-1) * mask).sum()
    num_valid = mask.sum().clamp(min=1)
    pg_obj = pg_obj / num_valid
    return -pg_obj


# -----------------------------------------------------------------------------
# group_normalize_advantages

def test_group_normalize_mean_subtracts_group_mean():
    rewards = torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 1.0])  # 2 groups of 3
    adv = group_normalize_advantages(rewards, group_size=3, mode="mean")
    g = adv.view(2, 3)
    # each group should be zero-mean
    assert torch.allclose(g.mean(dim=1), torch.zeros(2), atol=1e-6)
    # group 0: [1,0,0] - 1/3
    assert torch.allclose(g[0], torch.tensor([2 / 3, -1 / 3, -1 / 3]), atol=1e-6)


def test_group_normalize_single_group_matches_chat_rl():
    rewards = torch.tensor([1.0, 0.0, 1.0, 0.0, 0.0, 1.0, 0.0, 0.0])
    # chat_rl uses one group over all num_samples: advantages = rewards - mean
    adv = group_normalize_advantages(rewards, group_size=rewards.numel(), mode="mean")
    assert torch.allclose(adv, rewards - rewards.mean(), atol=1e-7)


def test_group_normalize_zscore_unit_variance():
    rewards = torch.tensor([3.0, 1.0, 2.0, 0.0, 4.0, 4.0, 4.0, 4.0])  # group0 varies, group1 constant
    adv = group_normalize_advantages(rewards, group_size=4, mode="zscore")
    g = adv.view(2, 4)
    # group 0 has variance -> normalized to ~unit population std and zero mean
    assert torch.allclose(g[0].mean(), torch.tensor(0.0), atol=1e-6)
    assert abs(g[0].std(unbiased=False).item() - 1.0) < 1e-4
    # group 1 is constant -> std 0 -> advantages all 0 (no NaN)
    assert torch.allclose(g[1], torch.zeros(4), atol=1e-6)
    assert not torch.isnan(adv).any()


def test_group_normalize_rejects_bad_shapes():
    try:
        group_normalize_advantages(torch.zeros(5), group_size=2)
        raise AssertionError("should have rejected N not divisible by group_size")
    except AssertionError as e:
        assert "divisible" in str(e)


# -----------------------------------------------------------------------------
# token_pg_loss reduces to REINFORCE / chat_rl

def test_token_pg_loss_is_reinforce_when_clip_off():
    B, T = 4, 7
    logp = torch.randn(B, T)
    mask = (torch.rand(B, T) > 0.3).float()
    adv = torch.randn(B)
    per_tok = token_pg_loss(logp, adv, mask, clip_eps=0.0)
    expected = -(logp * adv.unsqueeze(1)) * mask
    assert torch.allclose(per_tok, expected, atol=1e-6)
    # masked positions contribute exactly zero
    assert torch.allclose(per_tok[mask == 0], torch.zeros_like(per_tok[mask == 0]))


def test_policy_loss_grpo_matches_chat_rl_reference():
    B, T = 5, 9
    logp = torch.randn(B, T)
    mask = (torch.rand(B, T) > 0.2).float()
    rewards = torch.bernoulli(torch.full((B,), 0.5))
    adv = group_normalize_advantages(rewards, group_size=B, mode="mean")
    loss = policy_loss(logp, adv, mask, objective="grpo", clip_eps=0.0, kl_beta=0.0)
    ref = _chat_rl_reference_loss(logp, rewards, mask)
    assert torch.allclose(loss, ref, atol=1e-6), f"{loss.item()} vs {ref.item()}"


# -----------------------------------------------------------------------------
# PPO clip: gradient equivalence on-policy, and clamping behavior

def test_ppo_clip_gradient_matches_reinforce_on_policy():
    B, T = 3, 5
    base = torch.randn(B, T)
    mask = torch.ones(B, T)
    adv = torch.tensor([1.5, -2.0, 0.5])

    # REINFORCE gradient
    logp1 = base.clone().requires_grad_(True)
    loss1 = masked_token_mean(token_pg_loss(logp1, adv, mask, clip_eps=0.0), mask)
    (g1,) = torch.autograd.grad(loss1, logp1)

    # PPO clip with logp_old = logp.detach() (single on-policy step) -> ratio == 1
    logp2 = base.clone().requires_grad_(True)
    loss2 = masked_token_mean(
        token_pg_loss(logp2, adv, mask, logp_old=base.clone().detach(), clip_eps=0.2), mask
    )
    (g2,) = torch.autograd.grad(loss2, logp2)

    assert torch.allclose(g1, g2, atol=1e-6), f"grad mismatch: {(g1 - g2).abs().max()}"


def test_ppo_clip_clamps_positive_advantage():
    # advantage > 0, ratio pushed well above 1+eps -> objective clipped at (1+eps)*adv
    eps = 0.2
    adv = torch.tensor([2.0])
    mask = torch.ones(1, 1)
    logp = torch.tensor([[0.0]])        # ratio = exp(logp - logp_old)
    logp_old = torch.tensor([[-1.0]])   # logp - logp_old = 1.0 -> ratio = e ~ 2.718 > 1.2
    per_tok = token_pg_loss(logp, adv, mask, logp_old=logp_old, clip_eps=eps)
    expected = -(1.0 + eps) * adv  # clipped branch wins for positive advantage
    assert torch.allclose(per_tok.view(-1), expected, atol=1e-6)


def test_ppo_clip_unclipped_inside_band():
    # ratio inside [1-eps, 1+eps] -> unclipped == clipped, equals ratio*adv
    eps = 0.5
    adv = torch.tensor([1.0])
    mask = torch.ones(1, 1)
    logp = torch.tensor([[0.1]])
    logp_old = torch.tensor([[0.0]])    # ratio = e^0.1 ~ 1.105, within [0.5, 1.5]
    per_tok = token_pg_loss(logp, adv, mask, logp_old=logp_old, clip_eps=eps)
    ratio = torch.exp(logp - logp_old)
    assert torch.allclose(per_tok.view(-1), (-ratio * adv).view(-1), atol=1e-6)


# -----------------------------------------------------------------------------
# KL penalty (k3 estimator)

def test_kl_penalty_zero_when_equal_and_positive_otherwise():
    B, T = 2, 4
    logp = torch.randn(B, T)
    mask = torch.ones(B, T)
    # equal -> exactly zero
    assert torch.allclose(kl_penalty(logp, logp.clone(), mask), torch.zeros(B, T), atol=1e-7)
    # different -> strictly non-negative (k3 is always >= 0)
    logp_ref = logp + 0.5 * torch.randn(B, T)
    kl = kl_penalty(logp, logp_ref, mask)
    assert (kl >= -1e-7).all()
    # known value: r = 0.5 -> exp(0.5) - 0.5 - 1
    r = torch.tensor(0.5)
    known = torch.exp(r) - r - 1.0
    one = kl_penalty(torch.tensor([[0.0]]), torch.tensor([[0.5]]), torch.ones(1, 1))
    assert torch.allclose(one.view(-1), known.view(-1), atol=1e-6)


def test_kl_penalty_respects_mask():
    logp = torch.randn(2, 3)
    logp_ref = logp + 1.0
    mask = torch.tensor([[1.0, 0.0, 1.0], [0.0, 1.0, 0.0]])
    kl = kl_penalty(logp, logp_ref, mask)
    assert torch.allclose(kl[mask == 0], torch.zeros_like(kl[mask == 0]))


# -----------------------------------------------------------------------------
# GSPO (sequence-level)

def test_gspo_shapes_and_on_policy_value():
    B, T = 4, 6
    logp = torch.randn(B, T)
    mask = (torch.rand(B, T) > 0.2).float()
    adv = torch.randn(B)
    per_seq = sequence_pg_loss(logp, adv, mask, clip_eps=0.0)
    assert per_seq.shape == (B,)
    # on-policy (no clip): -(seq_logp * adv)
    seq_len = mask.sum(1).clamp(min=1)
    seq_logp = (logp * mask).sum(1) / seq_len
    assert torch.allclose(per_seq, -(seq_logp * adv), atol=1e-6)


def test_policy_loss_gspo_runs_and_is_scalar():
    B, T = 6, 8
    logp = torch.randn(B, T)
    mask = torch.ones(B, T)
    rewards = torch.bernoulli(torch.full((B,), 0.5))
    adv = group_normalize_advantages(rewards, group_size=3, mode="zscore")  # 2 groups of 3
    loss = policy_loss(logp, adv, mask, objective="gspo",
                       logp_old=logp.detach(), clip_eps=0.2,
                       logp_ref=logp.detach(), kl_beta=0.01)
    assert loss.dim() == 0 and torch.isfinite(loss)


# -----------------------------------------------------------------------------
# standalone runner (no pytest dependency)

if __name__ == "__main__":
    import sys
    tests = sorted((k, v) for k, v in globals().items()
                   if k.startswith("test_") and callable(v))
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception as e:
            failed += 1
            print(f"FAIL  {name}: {type(e).__name__}: {e}")
    print(f"\n{len(tests) - failed}/{len(tests)} passed")
    sys.exit(1 if failed else 0)
