#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'EOF'
Usage:
  ./export_highlevel.sh [checkpoint.pt] [output.onnx]

Example:
  ./export_highlevel.sh \
    /path/to/checkpoint_final.pt \
    exports/tennis_highlevel/policy.onnx
EOF
}

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  usage
  exit 0
fi

if [[ $# -gt 2 ]]; then
  usage
  exit 1
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

DEFAULT_CKPT_PATH="outputs/tennis_highlevel/checkpoints/checkpoint_final.pt"
CKPT_PATH="${1:-$DEFAULT_CKPT_PATH}"
OUT_PATH="${2:-exports/tennis_highlevel/policy.onnx}"

if [[ ! -f "${CKPT_PATH}" ]]; then
  echo "[ERROR] checkpoint not found: ${CKPT_PATH}" >&2
  exit 1
fi

mkdir -p "$(dirname "${OUT_PATH}")"

uv run python - "${CKPT_PATH}" "${OUT_PATH}" <<'PY'
from __future__ import annotations

import copy
import os
import sys
from pathlib import Path

import hydra
import torch
from omegaconf import OmegaConf
from tensordict.nn import TensorDictSequential

from humanoid_tennis.utils.export import export_onnx
from scripts.utils.helpers import ObsNorm, make_env_policy


def main() -> None:
    ckpt_path = Path(sys.argv[1]).expanduser().resolve()
    out_path = Path(sys.argv[2]).expanduser()
    if not out_path.is_absolute():
        out_path = (Path.cwd() / out_path).resolve()
    out_path.parent.mkdir(parents=True, exist_ok=True)

    with hydra.initialize(config_path="cfg", job_name="export_highlevel", version_base=None):
        cfg = hydra.compose(
            config_name="train",
            overrides=["task=G1/G1_tennis_highlevel", "+exp=highlevel"],
        )

    OmegaConf.set_struct(cfg, False)
    cfg.checkpoint_path = str(ckpt_path)
    cfg.vecnorm = "eval"
    cfg.export_policy = False
    cfg.perf_test = False
    cfg.rollout_mode = "eval"
    cfg.headless = True
    cfg.app.headless = True
    cfg.task.viewer.headless = True
    cfg.task.num_envs = 1

    # Highlevel command requires offline launch bank files during env init.
    bank_cfg = cfg.task.command.config.launch.bank
    default_easy = (Path.cwd() / "data/tennis_launch_bank/highlevel_subsets/launch_bank_easy.npz").resolve()
    default_medium = (Path.cwd() / "data/tennis_launch_bank/highlevel_subsets/launch_bank_medium.npz").resolve()
    default_hard = (Path.cwd() / "data/tennis_launch_bank/highlevel_subsets/launch_bank_hard.npz").resolve()
    default_single = (Path.cwd() / "data/tennis_launch_bank/highlevel_launch_bank.npz").resolve()

    easy = Path(os.environ.get("LAUNCH_BANK_EASY_FILE", str(default_easy))).expanduser().resolve()
    medium = Path(os.environ.get("LAUNCH_BANK_MEDIUM_FILE", str(default_medium))).expanduser().resolve()
    hard = Path(os.environ.get("LAUNCH_BANK_HARD_FILE", str(default_hard))).expanduser().resolve()
    single = Path(os.environ.get("LAUNCH_BANK_FILE", str(default_single))).expanduser().resolve()

    if easy.exists() and medium.exists() and hard.exists():
        bank_cfg.easy_file = str(easy)
        bank_cfg.medium_file = str(medium)
        bank_cfg.hard_file = str(hard)
        bank_cfg.file = None
    elif single.exists():
        bank_cfg.file = str(single)
        bank_cfg.easy_file = None
        bank_cfg.medium_file = None
        bank_cfg.hard_file = None
    else:
        raise FileNotFoundError(
            "Launch bank files are required but not found. "
            f"Tried easy/medium/hard under {default_easy.parent} and single {default_single}. "
            "You can override with env vars: LAUNCH_BANK_FILE or "
            "LAUNCH_BANK_EASY_FILE/LAUNCH_BANK_MEDIUM_FILE/LAUNCH_BANK_HARD_FILE."
        )

    env, policy, vecnorm, _ = make_env_policy(cfg)
    if hasattr(policy, "step_schedule"):
        policy.step_schedule(1.0, 0)
    if hasattr(env, "step_schedule"):
        env.step_schedule(1.0, 0)

    fake_input = env.observation_spec[0].rand().cpu()
    fake_input["is_init"] = torch.tensor(1, dtype=bool)
    fake_input["context_adapt_hx"] = torch.zeros(128)
    fake_input = fake_input.unsqueeze(0)

    deploy_policy = copy.deepcopy(policy.get_rollout_policy("deploy"))
    obs_norm = ObsNorm.from_vecnorm(vecnorm, deploy_policy.in_keys)
    export_policy = TensorDictSequential(obs_norm, deploy_policy).cpu()

    meta = {"action_scaling": dict(cfg.task.action.get("action_scaling"))}
    export_onnx(export_policy, fake_input, str(out_path), meta)
    env.close()

    print(f"[OK] exported ONNX: {out_path}")
    data_path = Path(str(out_path) + ".data")
    if data_path.exists():
        print(f"[OK] exported ONNX external data: {data_path}")


if __name__ == "__main__":
    main()
PY
