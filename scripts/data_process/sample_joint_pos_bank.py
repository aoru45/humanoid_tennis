#!/usr/bin/env python3
import argparse
from pathlib import Path

import torch

from active_adaptation.utils.motion import MotionDataset


def _sample_from_single_dataset(
    ds: MotionDataset,
    num_frames: int,
    *,
    generator: torch.Generator,
):
    starts = ds.starts.to(torch.long)
    lengths = ds.lengths.to(torch.long)
    total_frames = int(lengths.sum().item())
    if total_frames <= 0:
        raise RuntimeError("Dataset has no valid frames.")

    flat_ids = torch.randint(0, total_frames, (num_frames,), device=starts.device, generator=generator)
    cdf = lengths.cumsum(dim=0)
    motion_ids = torch.searchsorted(cdf, flat_ids, right=True)
    prev_cdf = torch.zeros_like(flat_ids)
    valid_mid = motion_ids > 0
    prev_cdf[valid_mid] = cdf[motion_ids[valid_mid] - 1]
    local_offsets = flat_ids - prev_cdf
    frame_ids = starts[motion_ids] + local_offsets
    return ds.data.joint_pos[frame_ids].to(dtype=torch.float32)


def sample_joint_pos_bank(
    mem_paths: list[str],
    num_frames: int,
    *,
    path_weights: list[float] | None = None,
    device: str = "cpu",
    seed: int = 0,
):
    dev = torch.device(device)
    datasets = [MotionDataset.create_from_path_lazy(p, device=dev) for p in mem_paths]
    if len(datasets) == 0:
        raise ValueError("mem_paths is empty.")

    g = torch.Generator(device=dev)
    g.manual_seed(int(seed))

    ref_joint_names = list(datasets[0].joint_names)
    for i, ds in enumerate(datasets[1:], start=1):
        if list(ds.joint_names) != ref_joint_names:
            raise ValueError(
                f"joint_names mismatch between mem_paths[0]='{mem_paths[0]}' and mem_paths[{i}]='{mem_paths[i]}'"
            )

    if path_weights is None:
        totals = torch.tensor(
            [int(ds.lengths.to(torch.long).sum().item()) for ds in datasets],
            dtype=torch.float64,
            device=dev,
        )
        if (totals <= 0).any():
            bad = [mem_paths[i] for i, v in enumerate(totals.tolist()) if v <= 0]
            raise RuntimeError(f"Datasets have no valid frames: {bad}")
        probs = totals / totals.sum()
    else:
        if len(path_weights) != len(mem_paths):
            raise ValueError("path_weights length must match mem_paths length.")
        w = torch.tensor(path_weights, dtype=torch.float64, device=dev)
        if (w < 0).any() or float(w.sum().item()) <= 0.0:
            raise ValueError("path_weights must be non-negative and sum > 0.")
        probs = w / w.sum()

    ds_ids = torch.multinomial(probs.to(torch.float32), num_frames, replacement=True, generator=g)
    counts = torch.bincount(ds_ids, minlength=len(datasets))
    chunks = []
    for i, cnt in enumerate(counts.tolist()):
        if cnt <= 0:
            continue
        chunks.append(_sample_from_single_dataset(datasets[i], cnt, generator=g))
    joint_pos = torch.cat(chunks, dim=0)
    perm = torch.randperm(joint_pos.shape[0], device=joint_pos.device, generator=g)
    joint_pos = joint_pos[perm].cpu().contiguous()

    return {
        "joint_pos": joint_pos,
        "joint_names": ref_joint_names,
        "mem_paths": list(mem_paths),
        "path_weights": probs.cpu().tolist(),
        "num_frames": int(num_frames),
        "seed": int(seed),
    }


def main():
    parser = argparse.ArgumentParser(
        description="Sample random joint positions from one or more motion datasets into a single .pt bank"
    )
    parser.add_argument(
        "--mem-paths",
        nargs="+",
        required=True,
        help="One or more dataset mem_paths used by MotionDataset.create_from_path_lazy",
    )
    parser.add_argument(
        "--path-weights",
        type=float,
        nargs="+",
        default=None,
        help="Optional sampling weights for each mem_path (same length as --mem-paths)",
    )
    parser.add_argument("--num-frames", type=int, default=20000, help="Number of random frames to sample")
    parser.add_argument("--out", required=True, help="Output .pt path")
    parser.add_argument("--device", default="cpu", help="Sampling device, default cpu")
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    args = parser.parse_args()

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = sample_joint_pos_bank(
        mem_paths=args.mem_paths,
        num_frames=args.num_frames,
        path_weights=args.path_weights,
        device=args.device,
        seed=args.seed,
    )
    torch.save(payload, out)
    print(
        "Saved joint bank: "
        f"{out} | frames={payload['joint_pos'].shape[0]} joints={payload['joint_pos'].shape[1]} "
        f"| datasets={len(payload['mem_paths'])}"
    )


if __name__ == "__main__":
    main()
