from __future__ import annotations

from dataclasses import dataclass, field, fields, is_dataclass, replace
from typing import Any, Mapping, Sequence


def _as_mapping(raw: Any) -> Mapping[str, Any]:
    if isinstance(raw, Mapping):
        return raw
    if hasattr(raw, "items"):
        return dict(raw.items())
    raise TypeError(f"Expected mapping-like config, got {type(raw)!r}")


def _merge_dataclass(instance, overrides: Any):
    if overrides is None:
        return instance
    mapping = _as_mapping(overrides)
    updates: dict[str, Any] = {}
    for f in fields(instance):
        if f.name not in mapping:
            continue
        raw = mapping[f.name]
        if raw is None:
            continue
        cur = getattr(instance, f.name)
        if is_dataclass(cur):
            updates[f.name] = _merge_dataclass(cur, raw)
            continue
        if isinstance(cur, tuple) and isinstance(raw, Sequence) and not isinstance(raw, (str, bytes)):
            updates[f.name] = tuple(raw)
            continue
        updates[f.name] = raw
    return replace(instance, **updates)


@dataclass
class LaunchBankConfig:
    file: str | None = None
    shuffle: bool = True
    easy_file: str | None = None
    medium_file: str | None = None
    hard_file: str | None = None
    use_curriculum: bool = False
    curriculum_start_probs: tuple[float, float, float] = (0.80, 0.15, 0.05)
    curriculum_target_probs: tuple[float, float, float] = (0.20, 0.35, 0.45)
    curriculum_progress_up: float = 0.04
    curriculum_progress_down: float = 0.0
    curriculum_ema_alpha: float = 0.05
    curriculum_min_level_prob: float = 0.05


@dataclass
class EpisodeConfig:
    max_task_steps: int = 1000
    max_consecutive_returns_before_finish: int = 8
    relaunch_on_success: bool = True
    launch_interval_s: float = 2.0
    relaunch_require_recovery: bool = True
    relaunch_recovery_hold_steps: int = 8
    relaunch_recovery_timeout_s: float = 2.5


@dataclass
class SpawnConfig:
    pos: tuple[float, float, float] = (0.0, -10.0, 0.81)
    yaw: float = 1.5707963267948966
    xy_noise: tuple[float, float] = (0.0, 0.0)
    yaw_noise_deg: float = 0.0


@dataclass
class ApproachConfig:
    contact_lead_time: float = 0.16
    contact_min_t: float = 0.06
    contact_max_t: float = 1.20
    hitting_zone_xy_radius: float = 0.40
    hitting_zone_z_tol: float = 0.26
    hitting_zone_time_min: float = 0.05
    hitting_zone_time_max: float = 0.65


@dataclass
class LaunchConfig:
    bank: LaunchBankConfig = field(default_factory=LaunchBankConfig)


@dataclass
class CourtConfig:
    x_limit: float = 4.2
    y_limit: float = 12.2
    y_min_success: float = 0.0
    net_height: float = 0.914
    net_half_thickness: float = 0.12
    net_clearance_reward_margin: float = 0.12
    miss_margin_y: float = 0.55
    out_margin_z: float = -0.25


@dataclass
class RecoverConfig:
    pre_hit_dead_ball_speed_thres: float = 1.0
    pre_hit_dead_ball_height_margin: float = 0.05
    pre_hit_dead_ball_patience_steps: int = 20
    pre_hit_dead_ball_min_steps: int = 24
    post_hit_dead_ball_speed_thres: float = -1.0
    post_hit_dead_ball_height_margin: float = 0.05
    post_hit_dead_ball_patience_steps: int = 30
    post_hit_dead_ball_min_steps: int = 24


@dataclass
class HighLevelTennisConfig:
    debug_draw: bool = False
    highlevel_latent_dim: int = 32
    ball_obs_history_steps: tuple[int, ...] = (0, 1, 2, 3, 4)
    ball_obs_prediction_horizon_s: float = 1.5
    wrist_joint_patterns: tuple[str, ...] = ("right_wrist_.*_joint",)
    stroke_style_min_racket_speed: float = 2.5
    stroke_style_min_forward_speed: float = 0.2
    post_hit_clean_bonus_window_steps: int = 32
    post_hit_stability_window_steps: int = 24
    recovery_outer_xy_radius: float = 1.10
    recovery_inner_xy_radius: float = 1.10
    recovery_outer_heading_cos: float = 0.65
    recovery_inner_heading_cos: float = 0.82
    recovery_upper_joint_patterns: tuple[str, ...] = (
        "waist_.*_joint",
        ".*_shoulder_.*_joint",
        ".*_elbow_joint",
        ".*_wrist_.*_joint",
    )

    ball_radius: float = 0.0335
    ball_mass: float = 0.057
    air_density: float = 1.21
    air_drag_k: float = 1.0
    drag_coef: float = 0.55
    lift_spin_scale: float = 5.0
    spin_damping_coef: float = 0.003

    outgoing_speed_minmax: tuple[float, float] = (8.0, 26.0)

    episode: EpisodeConfig = field(default_factory=EpisodeConfig)
    spawn: SpawnConfig = field(default_factory=SpawnConfig)
    approach: ApproachConfig = field(default_factory=ApproachConfig)
    launch: LaunchConfig = field(default_factory=LaunchConfig)
    court: CourtConfig = field(default_factory=CourtConfig)
    recover: RecoverConfig = field(default_factory=RecoverConfig)

    @classmethod
    def from_any(cls, raw: Mapping[str, Any] | "HighLevelTennisConfig" | None) -> "HighLevelTennisConfig":
        if raw is None:
            return cls()
        if isinstance(raw, cls):
            return raw
        return _merge_dataclass(cls(), raw)
