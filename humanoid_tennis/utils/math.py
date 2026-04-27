# This file contains additional math utilities
# that are not covered by IsaacLab

import torch
import torch.distributions as D


def wrap_to_pi(x: torch.Tensor) -> torch.Tensor:
    two_pi = 2 * torch.pi
    return (x + torch.pi) % two_pi - torch.pi


def quat_conjugate(q: torch.Tensor) -> torch.Tensor:
    out = q.clone()
    out[..., 1:] = -out[..., 1:]
    return out


def quat_from_angle_axis(angle: torch.Tensor, axis: torch.Tensor) -> torch.Tensor:
    half = 0.5 * angle
    w = torch.cos(half)
    xyz = axis * torch.sin(half).unsqueeze(-1)
    return torch.cat([w.unsqueeze(-1), xyz], dim=-1)

def clamp_norm(x: torch.Tensor, min: float=0., max: float=torch.inf):
    x_norm = x.norm(dim=-1, keepdim=True).clamp(1e-6)
    x = torch.where(x_norm < min, x / x_norm * min, x)
    x = torch.where(x_norm > max, x / x_norm * max, x)
    return x

def clamp_along(x: torch.Tensor, axis: torch.Tensor, min: float, max: float):
    projection = (x * axis).sum(dim=-1, keepdim=True)
    return x - projection * axis + projection.clamp(min, max) * axis

def normalize(x: torch.Tensor):
    return x / x.norm(dim=-1, keepdim=True).clamp_min(1e-6)

@torch.jit.script
def quat_apply(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., 1:]    # (..., 3)
    w = quat[..., :1]      # (..., 1)
    t = torch.cross(xyz, vec, dim=-1) * 2   # (..., 3)
    return vec + w * t + torch.cross(xyz, t, dim=-1)  # (..., 3)

@torch.jit.script
def quat_apply_inverse(quat: torch.Tensor, vec: torch.Tensor) -> torch.Tensor:
    xyz = quat[..., 1:]    # (..., 3)
    w = quat[..., :1]      # (..., 1)
    t = torch.cross(xyz, vec, dim=-1) * 2
    return vec - w * t + torch.cross(xyz, t, dim=-1)

@torch.jit.script
def axis_angle_from_quat(quat: torch.Tensor) -> torch.Tensor:
    quat = quat * (1.0 - 2.0 * (quat[..., 0:1] < 0.0))
    mag = torch.linalg.norm(quat[..., 1:], dim=-1)
    half_angle = torch.atan2(mag, quat[..., 0])
    angle = 2.0 * half_angle
    sin_half_angles_over_angles = torch.where(
        angle.abs() > 1.0e-6, torch.sin(half_angle) / angle, 0.5 - angle * angle / 48
    )
    return quat[..., 1:4] / sin_half_angles_over_angles.unsqueeze(-1)

def yaw_quat(quat: torch.Tensor) -> torch.Tensor:
    qw = quat[..., 0]
    qx = quat[..., 1]
    qy = quat[..., 2]
    qz = quat[..., 3]

    yaw = torch.atan2(
        2.0 * (qw * qz + qx * qy),
        1.0 - 2.0 * (qy * qy + qz * qz),
    )

    half_yaw = yaw * 0.5
    out = torch.zeros_like(quat)
    out[..., 0] = torch.cos(half_yaw)  # w
    out[..., 3] = torch.sin(half_yaw)  # z

    return normalize(out)

@torch.jit.script
def quat_mul(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
    # TorchScript-friendly checks
    torch._assert(q1.size(-1) == 4, "q1 last dim must be 4 (w, x, y, z)")
    torch._assert(q2.size(-1) == 4, "q2 last dim must be 4 (w, x, y, z)")

    # (..., 4) -> (...,)
    w1 = q1[..., 0]
    x1 = q1[..., 1]
    y1 = q1[..., 2]
    z1 = q1[..., 3]

    w2 = q2[..., 0]
    x2 = q2[..., 1]
    y2 = q2[..., 2]
    z2 = q2[..., 3]

    # Hamilton product (broadcasts over leading dims)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2
    z = w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2

    return torch.stack((w, x, y, z), dim=-1)

@torch.jit.script
def matrix_from_quat(quaternions: torch.Tensor) -> torch.Tensor:
    r, i, j, k = torch.unbind(quaternions, -1)
    two_s = 2.0 / (quaternions * quaternions).sum(-1)

    o = torch.stack(
        (
            1 - two_s * (j * j + k * k),
            two_s * (i * j - k * r),
            two_s * (i * k + j * r),
            two_s * (i * j + k * r),
            1 - two_s * (i * i + k * k),
            two_s * (j * k - i * r),
            two_s * (i * k - j * r),
            two_s * (j * k + i * r),
            1 - two_s * (i * i + j * j),
        ),
        -1,
    )
    return o.reshape(quaternions.shape[:-1] + (3, 3))

__all__ = [
    "yaw_quat", "wrap_to_pi", "quat_mul", "quat_conjugate", "quat_from_angle_axis",
    "quat_apply", "quat_apply_inverse", "axis_angle_from_quat",
    "clamp_norm", "clamp_along", "normalize", "matrix_from_quat",
]
