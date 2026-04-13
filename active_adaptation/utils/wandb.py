# MIT License
#
# Copyright (c) 2023 Botian Xu, Tsinghua University
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence
from urllib.parse import urlparse

import requests
import wandb
from omegaconf import OmegaConf


DEFAULT_CFG_FILE_NAMES = ("cfg.yaml", "files/cfg.yaml", "config.yaml")


def _select(cfg_like: Any, key: str, default: Any = None):
    if cfg_like is None:
        return default
    if isinstance(cfg_like, Mapping):
        cur = cfg_like
        for part in key.split("."):
            if not isinstance(cur, Mapping) or part not in cur:
                return default
            cur = cur[part]
        return cur
    try:
        return OmegaConf.select(cfg_like, key, default=default)
    except Exception:
        return default


def _to_container(cfg_like: Any):
    if cfg_like is None:
        return None
    if isinstance(cfg_like, Mapping):
        return dict(cfg_like)
    return OmegaConf.to_container(cfg_like, resolve=False)


def _normalize_proxy(value: Any) -> str | None:
    if value is None:
        return None
    value = str(value).strip()
    if not value:
        return None
    if "://" not in value:
        value = f"http://{value}"
    parsed = urlparse(value)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid proxy URL: {value}")
    return value.rstrip("/")


def _proxy_pair_from_cfg(wandb_cfg: Any = None):
    http_proxy = _normalize_proxy(_select(wandb_cfg, "settings.http_proxy", None))
    https_proxy = _normalize_proxy(_select(wandb_cfg, "settings.https_proxy", None))
    if http_proxy is None and https_proxy is None:
        return None, None
    # Fill both protocols explicitly to avoid fallback to process-level proxies.
    if http_proxy is None:
        http_proxy = https_proxy
    if https_proxy is None:
        https_proxy = http_proxy
    return http_proxy, https_proxy


def load_wandb_cfg_from_yaml(cfg_path: str | os.PathLike[str]):
    path = Path(cfg_path)
    if not path.exists():
        return None
    cfg = OmegaConf.load(path)
    return cfg.get("wandb", None)


def build_wandb_settings(wandb_cfg: Any = None):
    http_proxy, https_proxy = _proxy_pair_from_cfg(wandb_cfg)
    settings_kwargs = {}
    if http_proxy is not None:
        settings_kwargs["http_proxy"] = http_proxy
    if https_proxy is not None:
        settings_kwargs["https_proxy"] = https_proxy
    if not settings_kwargs:
        return None
    return wandb.Settings(**settings_kwargs)


def build_wandb_init_kwargs(wandb_cfg: Any = None, **kwargs):
    settings = build_wandb_settings(wandb_cfg)
    if settings is not None and "settings" not in kwargs:
        kwargs["settings"] = settings
    return kwargs


def build_wandb_api_overrides(wandb_cfg: Any = None):
    # wandb.Api() in wandb==0.25.0 reads proxy config from overrides["_proxies"].
    http_proxy, https_proxy = _proxy_pair_from_cfg(wandb_cfg)
    proxies = {}
    if http_proxy is not None:
        proxies["http"] = http_proxy
    if https_proxy is not None:
        proxies["https"] = https_proxy
    if not proxies:
        return None
    return {"_proxies": proxies}


def make_wandb_api(wandb_cfg: Any = None, **kwargs):
    overrides = kwargs.pop("overrides", None)
    merged_overrides = {}
    if overrides:
        merged_overrides.update(overrides)
    proxy_overrides = build_wandb_api_overrides(wandb_cfg)
    if proxy_overrides:
        merged_overrides.update(proxy_overrides)
    if merged_overrides:
        kwargs["overrides"] = merged_overrides
    api = wandb.Api(**kwargs)
    _apply_proxy_to_api_session(api, wandb_cfg)
    return api


def _configured_proxies(wandb_cfg: Any = None):
    http_proxy, https_proxy = _proxy_pair_from_cfg(wandb_cfg)
    proxies = {}
    if http_proxy is not None:
        proxies["http"] = http_proxy
    if https_proxy is not None:
        proxies["https"] = https_proxy
    return proxies


def _apply_proxy_to_api_session(api, wandb_cfg: Any = None):
    # Force requests session to ignore process env proxies. Otherwise global
    # http_proxy/https_proxy can override the session-level proxy map.
    session = None
    try:
        session = api.client._session
    except Exception:
        session = None
    if session is None:
        return
    session.trust_env = False
    session.proxies.clear()
    proxies = _configured_proxies(wandb_cfg)
    if proxies:
        session.proxies.update(proxies)


def _make_download_session(wandb_cfg: Any = None):
    session = requests.Session()
    session.trust_env = False
    proxies = _configured_proxies(wandb_cfg)
    if proxies:
        session.proxies.update(proxies)
    return session


def _download_file_via_session(
    file_obj,
    api,
    dest_path: Path,
    wandb_cfg: Any = None,
    replace: bool = True,
):
    if dest_path.exists() and not replace:
        return
    dest_path.parent.mkdir(parents=True, exist_ok=True)

    session = _make_download_session(wandb_cfg)
    auth = ("api", api.api_key or "")
    with session.get(file_obj.url, auth=auth, stream=True, timeout=10) as response:
        response.raise_for_status()
        with open(dest_path, "wb") as f:
            for chunk in response.iter_content(chunk_size=1024):
                if chunk:
                    f.write(chunk)


def init_wandb_run(wandb_cfg: Any, config: Any = None, **kwargs):
    default_keys = (
        "project",
        "entity",
        "group",
        "name",
        "mode",
        "tags",
        "id",
        "notes",
        "job_type",
    )
    for key in default_keys:
        if key not in kwargs:
            value = _select(wandb_cfg, key, None)
            if value is not None:
                kwargs[key] = value
    kwargs = build_wandb_init_kwargs(wandb_cfg, **kwargs)
    requested_name = kwargs.get("name", None)
    run = wandb.init(**kwargs)
    if requested_name is not None and str(requested_name).strip():
        run.name = str(requested_name)
    config_dict = _to_container(config)
    if config_dict is not None:
        run.config.update(config_dict)
    return run


def finish_wandb_run(run=None):
    if run is not None:
        run.finish()
    else:
        wandb.finish()


@dataclass(frozen=True)
class RunCheckpointDownload:
    run_path: str
    run_name: str
    run_root: str
    checkpoint_name: str
    checkpoint_path: str
    cfg_path: str | None


def _checkpoint_step_from_name(file_name: str) -> int | None:
    stem = Path(file_name).stem
    suffix = stem.split("_")[-1]
    if suffix == "final":
        return 100000
    if suffix.isdigit():
        return int(suffix)
    return None


def _select_checkpoint_file(checkpoint_files: Sequence[Any], iteration: int | None):
    if not checkpoint_files:
        raise ValueError("No checkpoint files found in run.")

    if iteration is not None:
        for file in checkpoint_files:
            if _checkpoint_step_from_name(file.name) == iteration:
                return file
        raise ValueError(f"No checkpoint found for iteration {iteration}")

    parsed = [(file, _checkpoint_step_from_name(file.name)) for file in checkpoint_files]
    valid = [(file, step) for file, step in parsed if step is not None]
    if valid:
        return max(valid, key=lambda x: x[1])[0]
    return sorted(checkpoint_files, key=lambda x: x.name)[-1]


def _first_existing_path(candidates: Sequence[Path]) -> str | None:
    for path in candidates:
        if path.exists():
            return str(path)
    return None


def download_run_checkpoint(
    run_path: str,
    wandb_cfg: Any = None,
    root_dir: str | os.PathLike[str] | None = None,
    iteration: int | None = None,
    cfg_file_names: Sequence[str] = DEFAULT_CFG_FILE_NAMES,
    replace: bool = True,
) -> RunCheckpointDownload:
    api = make_wandb_api(wandb_cfg)
    run = api.run(run_path)

    if root_dir is None:
        root_dir = Path.cwd() / "wandb"
    run_root = Path(root_dir) / run.name
    run_root.mkdir(parents=True, exist_ok=True)

    checkpoint_files = []
    cfg_name_set = set(cfg_file_names)
    for file in run.files():
        if "checkpoint" in file.name:
            checkpoint_files.append(file)
        elif file.name in cfg_name_set:
            _download_file_via_session(
                file_obj=file,
                api=api,
                dest_path=run_root / file.name,
                wandb_cfg=wandb_cfg,
                replace=replace,
            )

    checkpoint = _select_checkpoint_file(checkpoint_files, iteration)
    _download_file_via_session(
        file_obj=checkpoint,
        api=api,
        dest_path=run_root / checkpoint.name,
        wandb_cfg=wandb_cfg,
        replace=replace,
    )

    cfg_candidates = [run_root / name for name in cfg_file_names]
    cfg_path = _first_existing_path(cfg_candidates)
    checkpoint_path = run_root / checkpoint.name

    return RunCheckpointDownload(
        run_path=run_path,
        run_name=run.name,
        run_root=str(run_root),
        checkpoint_name=checkpoint.name,
        checkpoint_path=str(checkpoint_path),
        cfg_path=cfg_path,
    )


def load_run_cfg_and_checkpoint(
    run_path: str,
    wandb_cfg: Any = None,
    root_dir: str | os.PathLike[str] | None = None,
    iteration: int | None = None,
    cfg_file_names: Sequence[str] = DEFAULT_CFG_FILE_NAMES,
):
    result = download_run_checkpoint(
        run_path=run_path,
        wandb_cfg=wandb_cfg,
        root_dir=root_dir,
        iteration=iteration,
        cfg_file_names=cfg_file_names,
    )
    if result.cfg_path is None:
        raise FileNotFoundError(
            f"Could not find downloaded cfg file in run root: {result.run_root}"
        )
    cfg = OmegaConf.load(result.cfg_path)
    return cfg, result


def _parse_run_reference(path: str):
    # Supports: run:<run_path> or run:<run_path>:<iteration>
    parts = path.split(":")
    if len(parts) < 2 or not parts[1]:
        raise ValueError(f"Invalid wandb run reference: {path}")
    run_path = parts[1]
    iteration = None
    if len(parts) >= 3 and parts[2]:
        if not parts[2].isdigit():
            raise ValueError(f"Invalid checkpoint iteration in run reference: {path}")
        iteration = int(parts[2])
    return run_path, iteration


def parse_checkpoint_path(
    path: str | None = None,
    wandb_cfg: Any = None,
    download_root: str | os.PathLike[str] | None = None,
):
    if path is None:
        return None
    if not path.startswith("run:"):
        return path

    run_path, iteration = _parse_run_reference(path)
    if download_root is None:
        download_root = Path(__file__).resolve().parent / "wandb"
    result = download_run_checkpoint(
        run_path=run_path,
        wandb_cfg=wandb_cfg,
        root_dir=download_root,
        iteration=iteration,
    )
    return result.checkpoint_path


parse_path = parse_checkpoint_path
