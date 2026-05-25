from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path

import hydra
import numpy as np
import onnxruntime as ort
import torch
from omegaconf import OmegaConf
from tensordict.nn import TensorDictSequential

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.utils.helpers import ObsNorm, make_env_policy


def _load_cfg(checkpoint: Path, bank_file: Path, seed: int):
    with hydra.initialize(config_path="../cfg", job_name="compare_highlevel_onnx_pytorch", version_base=None):
        cfg = hydra.compose(config_name="train", overrides=["task=G1/G1_tennis_highlevel", "+exp=highlevel"])

    OmegaConf.set_struct(cfg, False)
    cfg.seed = int(seed)
    cfg.checkpoint_path = str(checkpoint)
    cfg.vecnorm = "eval"
    cfg.rollout_mode = "eval"
    cfg.task.num_envs = 1
    cfg.app.headless = True
    cfg.task.viewer.headless = True

    cfg.task.command.config.launch.bank.file = str(bank_file)
    cfg.task.command.config.launch.bank.easy_file = None
    cfg.task.command.config.launch.bank.medium_file = None
    cfg.task.command.config.launch.bank.hard_file = None
    cfg.task.command.config.launch.bank.use_curriculum = False
    return cfg


def _build_onnx_session(path: Path) -> ort.InferenceSession:
    sess_opt = ort.SessionOptions()
    sess_opt.intra_op_num_threads = 4
    sess_opt.inter_op_num_threads = 1
    return ort.InferenceSession(str(path), sess_options=sess_opt, providers=["CPUExecutionProvider"])


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare highlevel policy action parity between PyTorch and ONNX.")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="outputs/tennis_highlevel/checkpoints/checkpoint_final.pt",
    )
    parser.add_argument(
        "--onnx",
        type=str,
        default="exports/tennis_highlevel/policy.onnx",
    )
    parser.add_argument(
        "--bank-file",
        type=str,
        default="data/tennis_launch_bank/highlevel_subsets/launch_bank_easy.npz",
    )
    parser.add_argument("--steps", type=int, default=300)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--seed", type=int, default=1)
    args = parser.parse_args()

    root = Path(__file__).resolve().parents[1]
    ckpt_path = Path(args.checkpoint).expanduser()
    if not ckpt_path.is_absolute():
        ckpt_path = (root / ckpt_path).resolve()
    onnx_path = Path(args.onnx).expanduser()
    if not onnx_path.is_absolute():
        onnx_path = (root / onnx_path).resolve()
    bank_path = Path(args.bank_file).expanduser()
    if not bank_path.is_absolute():
        bank_path = (root / bank_path).resolve()
    meta_path = onnx_path.with_suffix(".json")

    if not ckpt_path.exists():
        raise FileNotFoundError(f"checkpoint not found: {ckpt_path}")
    if not onnx_path.exists():
        raise FileNotFoundError(f"onnx not found: {onnx_path}")
    if not meta_path.exists():
        raise FileNotFoundError(f"onnx meta json not found: {meta_path}")
    if not bank_path.exists():
        raise FileNotFoundError(f"launch bank not found: {bank_path}")

    cfg = _load_cfg(checkpoint=ckpt_path, bank_file=bank_path, seed=int(args.seed))
    env, policy, vecnorm, _ = make_env_policy(cfg)
    deploy_policy = copy.deepcopy(policy.get_rollout_policy("deploy")).cpu().eval()
    obs_norm = ObsNorm.from_vecnorm(vecnorm, deploy_policy.in_keys).cpu().eval()
    pt_td_model = TensorDictSequential(obs_norm, deploy_policy).cpu().eval()

    with open(meta_path, "r") as f:
        meta = json.load(f)
    in_keys = [k if isinstance(k, str) else tuple(k) for k in meta.get("in_keys", [])]
    out_keys = [k if isinstance(k, str) else tuple(k) for k in meta.get("out_keys", [])]
    if "action" not in out_keys:
        raise RuntimeError(f"ONNX outputs missing 'action', out_keys={out_keys}")
    action_idx = out_keys.index("action")

    ort_sess = _build_onnx_session(onnx_path)
    ort_input_names = [x.name for x in ort_sess.get_inputs()]

    td = env.reset()
    max_abs = 0.0
    mean_abs_sum = 0.0
    eval_steps = 0
    worst_step = -1

    for step in range(int(args.steps)):
        inp_td = td.select(*[k for k in in_keys if isinstance(k, str)]).cpu()
        with torch.inference_mode():
            out_td = pt_td_model(inp_td.clone())
        pt_action = out_td["action"].detach().cpu().numpy().astype(np.float32)

        ort_inputs = {}
        for name in ort_input_names:
            if name not in inp_td.keys():
                raise RuntimeError(f"ONNX expects input '{name}', but tensordict has keys={list(inp_td.keys())}")
            v = inp_td[name]
            if isinstance(v, torch.Tensor):
                v = v.detach().cpu().numpy()
            ort_inputs[name] = np.asarray(v, dtype=np.float32)
        ort_out = ort_sess.run(None, ort_inputs)
        onnx_action = np.asarray(ort_out[action_idx], dtype=np.float32)

        diff = np.abs(pt_action - onnx_action)
        step_max = float(diff.max())
        if step >= int(args.warmup):
            eval_steps += 1
            mean_abs_sum += float(diff.mean())
            if step_max > max_abs:
                max_abs = step_max
                worst_step = step

        td.set("action", torch.from_numpy(pt_action).to(td.device))
        td = env.step(td)["next"]

    mean_abs = mean_abs_sum / max(1, eval_steps)
    print("=== PT vs ONNX Action Parity ===")
    print(f"steps={int(args.steps)} warmup={int(args.warmup)} eval_steps={eval_steps}")
    print(f"action: max_abs={max_abs:.6e} mean_abs={mean_abs:.6e} worst_step={worst_step}")
    env.close()


if __name__ == "__main__":
    main()
