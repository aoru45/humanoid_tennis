import argparse
import json
from functools import partial
from pathlib import Path

import numpy as np
import torch

from active_adaptation.utils.motion import MotionDataset

EXCLUDE_LABEL_PATH = Path(__file__).parent / "label.txt"
SEED_KEEP_FILENAMES_PATH = Path("/home/axell/Desktop/dataset_new/retarget_g1/seed/keep_filenames.txt")
EXCLUDED_PATHS = set()
SEED_KEEP_FILENAMES = set()
ENABLE_AMASS_FILTER = False
ENABLE_SEED_FILTER = False
EXCLUDED_SUBSTRINGS = [
    "CMU/94",
    "CMU/126",
    "chair",
    "HDM05/tr",
    "HDM05/bk",
    "SShapeRL",
    "SShapeLR",
    "CircleCCW",
    "KIT/1226",
]


def preprocess_motion(motion, foot_idx, always_on_ground: bool = False):
    root_pos = motion["qpos"][:, :3]  # (T,3)
    offset_xy = root_pos[0, :2].copy()  # 首帧 x,y
    motion["qpos"][:, 0] -= offset_xy[0]
    motion["qpos"][:, 1] -= offset_xy[1]
    motion["xpos"][:, :, 0] -= offset_xy[0]
    motion["xpos"][:, :, 1] -= offset_xy[1]

    z_l = motion["xpos"][:, foot_idx[0], 2]
    z_r = motion["xpos"][:, foot_idx[1], 2]

    if not always_on_ground:
        z_min = float(min(z_l.min(), z_r.min()))
        target_z0 = 0.0
        dz = target_z0 - z_min
        motion["qpos"][:, 2] += dz
        motion["xpos"][:, :, 2] += dz
    else:
        z_min = np.min(
            np.concatenate([z_l.reshape(-1, 1), z_r.reshape(-1, 1)], axis=1),
            axis=-1,
            keepdims=True,
        )
        target_z0 = 0.0
        dz = target_z0 - z_min
        motion["qpos"][:, 2] += dz.reshape(-1)
        motion["xpos"][:, :, 2] += dz
    return motion


def none_callback(_ctx, m):
    m["metadata"] = None

def load_excluded_paths(label_path: Path) -> set[str]:
    if not label_path.exists():
        return set()
    paths = set()
    with label_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            token = line.split("\t", 1)[0]
            token = token.split(" ", 1)[0]
            if token:
                paths.add(token)
    return paths

def load_keep_filenames(list_path: Path) -> set[str]:
    if not list_path.exists():
        return set()
    names = set()
    with list_path.open("r", encoding="utf-8") as f:
        for line in f:
            name = line.strip()
            if name:
                names.add(name)
    return names

def check_motion(motion, foot_idx, path, start_idx, end_idx) -> bool:
    """Return False when the motion violates basic physical sanity checks."""

    qvel = motion["qvel"]
    qpos = motion["qpos"]
    xpos = motion["xpos"]

    path_str = str(path)
    if ENABLE_AMASS_FILTER:
        if EXCLUDED_PATHS and path_str in EXCLUDED_PATHS:
            print("Invalid motion due to excluded path")
            return False
        if any(s in path_str for s in EXCLUDED_SUBSTRINGS):
            print("Invalid motion due to excluded substring")
            return False
    if ENABLE_SEED_FILTER and SEED_KEEP_FILENAMES and path.stem not in SEED_KEEP_FILENAMES:
        print("Invalid motion due to missing keep filename")
        return False

    if np.any(np.abs(qvel[:, :6]) > 10):
        print("Invalid motion due to high velocity spike")
        return False
    if qpos.shape[0] < 250:
        print("Invalid motion due to short length")
        return False

    min_body_z = np.min(xpos[:, :, 2], axis=1)
    all_off = min_body_z > 0.2
    fps = int(motion.get("fps", 0))
    if fps <= 0:
        fps = 50
    if np.any(all_off):
        padded = np.concatenate(([0], all_off.astype(np.int8), [0]))
        edges = np.diff(padded)
        run_starts = np.where(edges == 1)[0]
        run_ends = np.where(edges == -1)[0]
        max_run = (run_ends - run_starts).max() if run_starts.size else 0
        if max_run > fps:
            print("Invalid motion due to all bodies off ground > 1s")
            return False
    max_body_z = float(np.max(xpos[:, :, 2]))
    if max_body_z <= 0.2:
        print("Invalid motion due to low max body height")
        return False
    return True

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", required=True, help="NPZ file or directory to convert")
    ap.add_argument("--mem-path", required=True, help="Output memmap directory")
    ap.add_argument("--amass-filter", action="store_true", help="Enable AMASS-specific path/name filters")
    ap.add_argument("--seed-filter", action="store_true", help="Enable seed keep_filenames allowlist filter")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    global EXCLUDED_PATHS, SEED_KEEP_FILENAMES, ENABLE_AMASS_FILTER, ENABLE_SEED_FILTER
    ENABLE_AMASS_FILTER = args.amass_filter
    ENABLE_SEED_FILTER = args.seed_filter
    EXCLUDED_PATHS = load_excluded_paths(EXCLUDE_LABEL_PATH) if ENABLE_AMASS_FILTER else set()
    SEED_KEEP_FILENAMES = load_keep_filenames(SEED_KEEP_FILENAMES_PATH) if ENABLE_SEED_FILTER else set()

    MotionDataset.create_from_path(
        str(dataset_root),
        target_fps=50,
        mem_path=args.mem_path,
        callback=none_callback,
        motion_processer=partial(preprocess_motion, always_on_ground=False),
        motion_filter=check_motion,
        segment_len=1000,
        storage_float_dtype=torch.float16,
        storage_int_dtype=torch.int32,
    )


if __name__ == "__main__":
    main()
