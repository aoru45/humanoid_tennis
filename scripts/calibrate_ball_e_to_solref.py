#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import importlib.util
import math
import sys
from pathlib import Path

import numpy as np


def _load_calibrator_module():
    script_path = Path(__file__).resolve().parent / "calibrate_tennis_contacts_mjlab.py"
    spec = importlib.util.spec_from_file_location("calib_contacts", script_path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Failed to load module from {script_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _frange(start: float, end: float, step: float) -> list[float]:
    vals: list[float] = []
    x = float(start)
    for _ in range(200000):
        if x > float(end) + 1.0e-12:
            break
        vals.append(float(round(x, 8)))
        x += float(step)
    return vals


def _interp_x_from_y(xs: np.ndarray, ys: np.ndarray, y_target: float) -> float:
    if ys.size < 2:
        raise ValueError("Need at least 2 samples for interpolation.")
    y_min = float(np.min(ys))
    y_max = float(np.max(ys))
    if y_target < y_min or y_target > y_max:
        raise ValueError(f"target e={y_target:.6f} out of sampled range [{y_min:.6f}, {y_max:.6f}]")

    order = np.argsort(ys)
    ys_s = ys[order]
    xs_s = xs[order]

    # For nearly duplicated ys, keep last x to avoid zero-division.
    uniq_y, uniq_idx = np.unique(np.round(ys_s, decimals=8), return_index=True)
    xs_u = xs_s[uniq_idx]
    ys_u = ys_s[uniq_idx]
    if ys_u.size < 2:
        return float(xs_u[0])
    return float(np.interp(y_target, ys_u, xs_u))


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Inverse-map target bounce COR(e) to MuJoCo ground solref[1].")
    p.add_argument("--e-min", type=float, default=0.71, help="Target effective COR lower bound.")
    p.add_argument("--e-max", type=float, default=0.79, help="Target effective COR upper bound.")
    p.add_argument("--drop-height-m", type=float, default=2.54)
    p.add_argument("--max-time-s", type=float, default=2.5)
    p.add_argument("--max-episode-steps", type=int, default=0)

    p.add_argument("--ball-solref2", type=float, default=None, help="Fix ball solref[1]. Default: current XML value.")
    p.add_argument("--ground-solref2-min", type=float, default=0.02)
    p.add_argument("--ground-solref2-max", type=float, default=0.30)
    p.add_argument("--ground-solref2-step", type=float, default=0.005)

    p.add_argument("--physics-dt", type=float, default=0.0005)
    p.add_argument("--iterations", type=int, default=24)
    p.add_argument("--ls-iterations", type=int, default=48)
    p.add_argument("--ccd-iterations", type=int, default=50)
    p.add_argument("--nconmax", type=int, default=600)
    p.add_argument("--njmax", type=int, default=2000)
    p.add_argument("--use-training-cfg", action="store_true")

    p.add_argument("--csv-out", type=str, default="", help="Optional csv output path for sampled curve.")
    return p


def main() -> None:
    args = build_parser().parse_args()
    if args.e_max < args.e_min:
        raise ValueError(f"Require e-max >= e-min, got [{args.e_min}, {args.e_max}]")
    if args.ground_solref2_step <= 0.0:
        raise ValueError("--ground-solref2-step must be > 0.")

    mod = _load_calibrator_module()
    if bool(args.use_training_cfg):
        mod._apply_training_sim_defaults(args)

    cal = mod.MujocoBounceCalibrator(
        physics_dt=float(args.physics_dt),
        iterations=int(args.iterations),
        ls_iterations=int(args.ls_iterations),
        ccd_iterations=int(args.ccd_iterations),
        nconmax=int(args.nconmax),
        njmax=int(args.njmax),
        max_episode_steps=int(args.max_episode_steps),
    )

    if args.ball_solref2 is None:
        ball_solref2 = float(cal.model.geom_solref[cal.ball_gid, 1])
    else:
        ball_solref2 = float(args.ball_solref2)

    ground_vals = _frange(args.ground_solref2_min, args.ground_solref2_max, args.ground_solref2_step)
    if len(ground_vals) < 2:
        raise RuntimeError("Need at least 2 ground_solref2 samples.")

    rows: list[tuple[float, float, float]] = []
    for g in ground_vals:
        cal.set_contact_solref2(ball_solref2=ball_solref2, ground_solref2=float(g))
        pos = np.array([0.0, -4.0, cal.contact_center_z + float(args.drop_height_m)], dtype=np.float64)
        vel = np.zeros((3,), dtype=np.float64)
        ang = np.zeros((3,), dtype=np.float64)
        _, _, rebound_h = cal._rollout_first_rebound(pos, vel, ang, max_time_s=float(args.max_time_s))
        rebound_h = float(rebound_h) if np.isfinite(rebound_h) else float("nan")
        if np.isfinite(rebound_h) and rebound_h >= 0.0:
            e_eff = float(math.sqrt(rebound_h / max(float(args.drop_height_m), 1.0e-9)))
        else:
            e_eff = float("nan")
        rows.append((float(g), float(rebound_h), float(e_eff)))

    finite = [(g, h, e) for g, h, e in rows if np.isfinite(e)]
    if len(finite) < 2:
        raise RuntimeError("Insufficient finite samples for inverse mapping.")

    g_arr = np.asarray([r[0] for r in finite], dtype=np.float64)
    e_arr = np.asarray([r[2] for r in finite], dtype=np.float64)

    g_for_e_min = _interp_x_from_y(g_arr, e_arr, float(args.e_min))
    g_for_e_max = _interp_x_from_y(g_arr, e_arr, float(args.e_max))
    g_low = min(g_for_e_min, g_for_e_max)
    g_high = max(g_for_e_min, g_for_e_max)

    corr = float(np.corrcoef(g_arr, e_arr)[0, 1]) if g_arr.size >= 2 else float("nan")

    print("[INFO] e->solref inverse mapping (drop-test effective COR)")
    print(
        f"[INFO] sampled ground_solref2=[{min(g_arr):.6f}, {max(g_arr):.6f}], "
        f"e_range=[{float(np.min(e_arr)):.6f}, {float(np.max(e_arr)):.6f}], corr={corr:.6f}"
    )
    print(
        f"[RESULT] target_e=[{float(args.e_min):.6f}, {float(args.e_max):.6f}] "
        f"=> ground_solref2=[{g_low:.6f}, {g_high:.6f}] (ball_solref2_fixed={ball_solref2:.6f})"
    )

    if args.csv_out:
        out_path = Path(args.csv_out).expanduser().resolve()
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with out_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["ground_solref2", "rebound_h_m", "e_eff"])
            for g, h, e in rows:
                writer.writerow([f"{g:.8f}", f"{h:.8f}", f"{e:.8f}"])
        print(f"[INFO] wrote curve csv: {out_path}")


if __name__ == "__main__":
    main()
