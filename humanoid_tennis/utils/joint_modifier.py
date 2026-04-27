from __future__ import annotations

from typing import Sequence

import torch


def _smoothstep(x: torch.Tensor) -> torch.Tensor:
    return x * x * (3 - 2 * x)


def _estimate_vel_from_pos(
    q: torch.Tensor,
    t: torch.Tensor,
    lengths_safe: torch.Tensor,
    fps: float,
) -> torch.Tensor:
    """Centered finite-difference velocity with per-env valid-length clipping."""
    dtype = q.dtype
    t_prev = (t - 1).clamp_min(0)
    t_next = torch.minimum(t + 1, (lengths_safe - 1).unsqueeze(1))
    idx_prev = t_prev.unsqueeze(-1).expand(-1, -1, q.shape[-1])
    idx_next = t_next.unsqueeze(-1).expand(-1, -1, q.shape[-1])
    q_prev = torch.gather(q, dim=1, index=idx_prev)
    q_next = torch.gather(q, dim=1, index=idx_next)
    dt = (t_next - t_prev).to(dtype=dtype).unsqueeze(-1).clamp_min(1.0)
    return (q_next - q_prev) * (float(fps) / dt)


def _sample_int_inclusive(
    low: torch.Tensor,
    high: torch.Tensor,
    *,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Vectorized inclusive integer sampler for per-element [low, high]."""
    span = (high - low + 1).clamp_min(1)
    u = torch.rand(span.shape, device=span.device, generator=generator)
    return low + torch.floor(u * span.to(dtype=u.dtype)).to(torch.long)


def sample_joint_abc_points(
    lengths: torch.Tensor,
    T: int,
    *,
    ac_len_range: Sequence[int],
    b_ratio_range: Sequence[float],
    generator: torch.Generator | None = None,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    lengths_safe = lengths.clamp_min(1).clamp_max(T)

    ac_min = int(ac_len_range[0])
    ac_max = int(ac_len_range[1])
    b_min = float(b_ratio_range[0])
    b_max = float(b_ratio_range[1])

    max_span = torch.minimum(torch.full_like(lengths_safe, ac_max), lengths_safe - 1)
    span = _sample_int_inclusive(torch.full_like(lengths_safe, ac_min), max_span, generator=generator)

    max_a = (lengths_safe - 1 - span).clamp_min(0)
    a = _sample_int_inclusive(torch.zeros_like(lengths_safe), max_a, generator=generator)
    c = (a + span).clamp_max(lengths_safe - 1)

    b_lo = a + torch.ceil(span.to(torch.float32) * b_min).to(torch.long)
    b_hi = a + torch.floor(span.to(torch.float32) * b_max).to(torch.long)
    b_lo = torch.maximum(a + 1, torch.minimum(b_lo, c - 1))
    b_hi = torch.maximum(a + 1, torch.minimum(b_hi, c - 1))

    b_rand_1 = _sample_int_inclusive(b_lo, b_hi, generator=generator)
    b_rand_2 = _sample_int_inclusive(b_lo, b_hi, generator=generator)
    b1 = torch.minimum(b_rand_1, b_rand_2)
    b2 = torch.maximum(b_rand_1, b_rand_2)
    t_mid = _sample_int_inclusive(b1, b2, generator=generator)

    return lengths_safe, a, b1, b2, c, t_mid


def apply_joint_abc_curve_(
    q_sel: torch.Tensor,
    v_sel: torch.Tensor,
    *,
    lengths_safe: torch.Tensor,
    a: torch.Tensor,
    b1: torch.Tensor,
    b2: torch.Tensor,
    c: torch.Tensor,
    q_b: torch.Tensor,
    fps: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    E, T, _ = q_sel.shape
    dev = q_sel.device
    dtype = q_sel.dtype

    env_idx = torch.arange(E, device=dev, dtype=torch.long)
    q_a = q_sel[env_idx, a]
    q_c = q_sel[env_idx, c]
    v_a_orig = v_sel[env_idx, a]
    v_c_orig = v_sel[env_idx, c]

    t = torch.arange(T, device=dev, dtype=torch.long).unsqueeze(0).expand(E, T)
    ab_mask = (t >= a.unsqueeze(1)) & (t <= b1.unsqueeze(1))
    bb_mask = (t >= b1.unsqueeze(1)) & (t <= b2.unsqueeze(1))
    bc_mask = (t >= b2.unsqueeze(1)) & (t <= c.unsqueeze(1))
    ac_mask = (t >= a.unsqueeze(1)) & (t <= c.unsqueeze(1))

    ab_denom = (b1 - a).clamp_min(1).to(dtype=dtype).unsqueeze(1)
    bc_denom = (c - b2).clamp_min(1).to(dtype=dtype).unsqueeze(1)
    s_ab = ((t - a.unsqueeze(1)).to(dtype=dtype) / ab_denom).clamp(0.0, 1.0)
    s_bc = ((t - b2.unsqueeze(1)).to(dtype=dtype) / bc_denom).clamp(0.0, 1.0)
    s_ab = _smoothstep(s_ab).unsqueeze(-1)
    s_bc = _smoothstep(s_bc).unsqueeze(-1)

    q_ab = (1.0 - s_ab) * q_a.unsqueeze(1) + s_ab * q_b.unsqueeze(1)
    q_bc = (1.0 - s_bc) * q_b.unsqueeze(1) + s_bc * q_c.unsqueeze(1)

    q_new = torch.where(ab_mask.unsqueeze(-1), q_ab, q_sel)
    q_new = torch.where(bb_mask.unsqueeze(-1), q_b.unsqueeze(1), q_new)
    q_new = torch.where(bc_mask.unsqueeze(-1), q_bc, q_new)

    v_est = _estimate_vel_from_pos(q_new, t, lengths_safe, fps)
    v_new = torch.where(ac_mask.unsqueeze(-1), v_est, v_sel)

    a_mask = (t == a.unsqueeze(1))
    c_mask = (t == c.unsqueeze(1))
    q_new = torch.where(a_mask.unsqueeze(-1), q_a.unsqueeze(1), q_new)
    q_new = torch.where(c_mask.unsqueeze(-1), q_c.unsqueeze(1), q_new)
    v_new = torch.where(a_mask.unsqueeze(-1), v_a_orig.unsqueeze(1), v_new)
    v_new = torch.where(c_mask.unsqueeze(-1), v_c_orig.unsqueeze(1), v_new)
    return q_new, v_new, ac_mask


def apply_joint_abc_modification_(
    joint_pos: torch.Tensor,
    joint_vel: torch.Tensor,
    lengths: torch.Tensor,
    *,
    left_joint_ids: torch.Tensor,
    right_joint_ids: torch.Tensor,
    left_prob: float,
    right_prob: float,
    b_dataset_prob: float,
    joint_pos_bank: torch.Tensor,
    ac_len_range: Sequence[int],
    b_ratio_range: Sequence[float],
    fps: float,
    generator: torch.Generator | None = None,
) -> torch.Tensor:
    """In-place A-B1-B2-C segment modification for batched trajectories.

    Args:
        joint_pos: [E, T, J]
        joint_vel: [E, T, J]
        lengths: [E], effective trajectory lengths
        left_joint_ids: left-side joint ids to modify
        right_joint_ids: right-side joint ids to modify
    """
    E, T, J = joint_pos.shape
    dev = joint_pos.device

    if joint_pos_bank is None or joint_pos_bank.numel() == 0:
        raise RuntimeError("apply_joint_abc_modification_ requires non-empty joint_pos_bank.")

    q_sel = joint_pos
    v_sel = joint_vel

    lengths_safe, a, b1, b2, c, t_mid = sample_joint_abc_points(
        lengths,
        T,
        ac_len_range=ac_len_range,
        b_ratio_range=b_ratio_range,
        generator=generator,
    )

    env_idx = torch.arange(E, device=dev, dtype=torch.long)
    q_b_tmid = q_sel[env_idx, t_mid]
    bank_rows = torch.randint(
        0,
        joint_pos_bank.shape[0],
        (E,),
        device=dev,
        generator=generator,
    )
    q_b_bank = joint_pos_bank.index_select(0, bank_rows)
    p_dataset = float(max(0.0, min(1.0, b_dataset_prob)))
    use_bank = torch.rand((E,), device=dev, generator=generator) < p_dataset
    q_b = torch.where(use_bank.unsqueeze(-1), q_b_bank, q_b_tmid)

    q_new, v_new, ac_mask = apply_joint_abc_curve_(
        q_sel,
        v_sel,
        lengths_safe=lengths_safe,
        a=a,
        b1=b1,
        b2=b2,
        c=c,
        q_b=q_b,
        fps=fps,
    )

    left_applied = (torch.rand((E,), device=dev, generator=generator) <= left_prob)
    right_applied = (torch.rand((E,), device=dev, generator=generator) <= right_prob)

    joint_write_mask = torch.zeros((E, J), device=dev, dtype=torch.bool)
    if left_joint_ids.numel() > 0:
        joint_write_mask[:, left_joint_ids] |= left_applied.unsqueeze(1)
    if right_joint_ids.numel() > 0:
        joint_write_mask[:, right_joint_ids] |= right_applied.unsqueeze(1)

    q_out = torch.where(joint_write_mask.unsqueeze(1), q_new, q_sel)
    v_out = torch.where(joint_write_mask.unsqueeze(1), v_new, v_sel)
    joint_pos[:] = q_out
    joint_vel[:] = v_out
    env_has_modified_joint = joint_write_mask.any(dim=1)
    return ac_mask & env_has_modified_joint.unsqueeze(1)
