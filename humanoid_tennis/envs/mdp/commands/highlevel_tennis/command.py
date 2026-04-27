from __future__ import annotations

import math
from typing import Mapping

import torch

from ..base import Command
from .action_contact import HighLevelTennisActionContactMixin
from .common import ContactSensorHandles, LaunchBankBuffer
from .config import HighLevelTennisConfig
from .launch import HighLevelTennisLaunchMixin
from .observations import HighLevelTennisObservationMixin
from .rewards import HighLevelTennisRewardMixin
from .runtime_flow import HighLevelTennisRuntimeFlowMixin
from .runtime_metrics import HighLevelTennisRuntimeMetricsMixin
from .state import HighLevelTennisStateMixin
from .terminations import HighLevelTennisTerminationMixin


class HighLevelTennisCommand(
    HighLevelTennisObservationMixin,
    HighLevelTennisRewardMixin,
    HighLevelTennisTerminationMixin,
    HighLevelTennisRuntimeFlowMixin,
    HighLevelTennisRuntimeMetricsMixin,
    HighLevelTennisStateMixin,
    HighLevelTennisActionContactMixin,
    HighLevelTennisLaunchMixin,
    Command,
):
    def __init__(
        self,
        env,
        config: Mapping[str, object] | HighLevelTennisConfig | None = None,
    ):
        super().__init__(env)
        cfg = HighLevelTennisConfig.from_any(config)
        if "tennis_ball" not in self.env.scene.entities:
            raise RuntimeError(
                "HighLevelTennisCommand requires task.tennis.add_ball=true "
                "so entity 'tennis_ball' is available."
            )
        self.ball = self.env.scene["tennis_ball"]
        self.debug_draw_enabled = bool(cfg.debug_draw)

        episode_cfg = cfg.episode
        spawn_cfg = cfg.spawn
        approach_cfg = cfg.approach
        launch_cfg = cfg.launch
        court_cfg = cfg.court
        recover_cfg = cfg.recover
        launch_bank_cfg = launch_cfg.bank

        self.robot_spawn_pos = torch.tensor(spawn_cfg.pos, device=self.device, dtype=torch.float32)
        self.robot_spawn_yaw = float(spawn_cfg.yaw)
        self.robot_spawn_xy_noise = torch.tensor(spawn_cfg.xy_noise, device=self.device, dtype=torch.float32)
        if self.robot_spawn_xy_noise.numel() != 2:
            raise ValueError(
                f"robot_spawn_xy_noise must have length 2, got shape={tuple(self.robot_spawn_xy_noise.shape)}"
            )
        self.robot_spawn_yaw_noise_rad = math.radians(float(spawn_cfg.yaw_noise_deg))

        self.max_task_steps = int(episode_cfg.max_task_steps)
        self.max_consecutive_returns_before_finish = max(0, int(episode_cfg.max_consecutive_returns_before_finish))
        self.relaunch_on_success = bool(episode_cfg.relaunch_on_success)
        self.launch_interval_s = max(float(episode_cfg.launch_interval_s), 0.0)
        self.launch_interval_steps = int(round(self.launch_interval_s / float(self.env.step_dt)))
        self.highlevel_latent_dim = int(cfg.highlevel_latent_dim)
        self.ball_obs_history_steps = tuple(int(s) for s in cfg.ball_obs_history_steps)
        if len(self.ball_obs_history_steps) == 0:
            raise ValueError("ball_obs_history_steps must contain at least one index (e.g. [0,1,2,3]).")
        if min(self.ball_obs_history_steps) < 0:
            raise ValueError(f"ball_obs_history_steps must be non-negative, got {self.ball_obs_history_steps}.")
        self.ball_obs_buffer_size = int(max(self.ball_obs_history_steps) + 1)
        self.ball_obs_prediction_horizon_s = max(float(cfg.ball_obs_prediction_horizon_s), 0.1)
        self.approach_contact_lead_time = max(float(approach_cfg.contact_lead_time), 0.0)
        self.approach_contact_min_t = max(float(approach_cfg.contact_min_t), 0.01)
        self.approach_contact_max_t = max(float(approach_cfg.contact_max_t), self.approach_contact_min_t + 0.01)
        self.hitting_zone_xy_radius = max(float(approach_cfg.hitting_zone_xy_radius), 0.05)
        self.hitting_zone_z_tol = max(float(approach_cfg.hitting_zone_z_tol), 0.05)
        self.hitting_zone_time_min = max(float(approach_cfg.hitting_zone_time_min), 0.01)
        self.hitting_zone_time_max = max(float(approach_cfg.hitting_zone_time_max), self.hitting_zone_time_min + 0.01)
        self.wrist_joint_patterns = tuple(str(p) for p in cfg.wrist_joint_patterns)
        self.stroke_style_min_racket_speed = float(cfg.stroke_style_min_racket_speed)
        self.stroke_style_min_forward_speed = float(cfg.stroke_style_min_forward_speed)
        self.post_hit_clean_bonus_window_steps = max(1, int(cfg.post_hit_clean_bonus_window_steps))
        self.post_hit_recovery_window_steps = max(1, int(cfg.post_hit_recovery_window_steps))
        self.post_hit_stability_window_steps = max(1, int(cfg.post_hit_stability_window_steps))
        self.recovery_upper_joint_patterns = tuple(str(p) for p in cfg.recovery_upper_joint_patterns)
        self.stroke_mode_lateral_deadzone = max(0.0, float(cfg.stroke_mode_lateral_deadzone))
        face_axis_local = torch.tensor(cfg.racket_face_axis_local, device=self.device, dtype=torch.float32)
        if face_axis_local.numel() != 3:
            raise ValueError(
                f"racket_face_axis_local must have length 3, got shape={tuple(face_axis_local.shape)}"
            )
        self.racket_face_axis_local = face_axis_local / face_axis_local.norm().clamp_min(1.0e-6)
        self.forehand_uses_negative_face_axis = bool(cfg.forehand_uses_negative_face_axis)
        self.STROKE_MODE_NEUTRAL = 0
        self.STROKE_MODE_FOREHAND = 1
        self.STROKE_MODE_BACKHAND = 2
        self.ball_radius = float(cfg.ball_radius)
        self.ball_mass = float(cfg.ball_mass)
        self.air_density = float(cfg.air_density)
        self.air_drag_k = float(cfg.air_drag_k)
        self.drag_coef = float(cfg.drag_coef)
        self.lift_spin_scale = float(cfg.lift_spin_scale)
        self.spin_damping_coef = float(cfg.spin_damping_coef)
        self.aero_force_k = 0.5 * self.air_density * math.pi * (self.ball_radius ** 2) * self.air_drag_k
        self.net_height = float(court_cfg.net_height)
        self.net_half_thickness = float(court_cfg.net_half_thickness)
        self.net_clearance_reward_margin = float(court_cfg.net_clearance_reward_margin)
        self.miss_margin_y = float(court_cfg.miss_margin_y)
        self.out_margin_z = float(court_cfg.out_margin_z)
        self.pre_hit_dead_ball_speed_thres = float(recover_cfg.pre_hit_dead_ball_speed_thres)
        self.pre_hit_dead_ball_height_margin = float(recover_cfg.pre_hit_dead_ball_height_margin)
        self.pre_hit_dead_ball_patience_steps = max(1, int(recover_cfg.pre_hit_dead_ball_patience_steps))
        self.pre_hit_dead_ball_min_steps = max(1, int(recover_cfg.pre_hit_dead_ball_min_steps))
        self.post_hit_dead_ball_speed_thres = float(recover_cfg.post_hit_dead_ball_speed_thres)
        self.post_hit_dead_ball_height_margin = float(recover_cfg.post_hit_dead_ball_height_margin)
        self.post_hit_dead_ball_patience_steps = max(1, int(recover_cfg.post_hit_dead_ball_patience_steps))
        self.post_hit_dead_ball_min_steps = max(1, int(recover_cfg.post_hit_dead_ball_min_steps))
        self.court_x_limit = float(court_cfg.x_limit)
        self.court_y_limit = float(court_cfg.y_limit)
        self.court_y_min_success = float(court_cfg.y_min_success)

        self.launch_bank_file = str(launch_bank_cfg.file).strip() if launch_bank_cfg.file is not None else ""
        self.launch_bank_easy_file = (
            str(launch_bank_cfg.easy_file).strip() if launch_bank_cfg.easy_file is not None else ""
        )
        self.launch_bank_medium_file = (
            str(launch_bank_cfg.medium_file).strip() if launch_bank_cfg.medium_file is not None else ""
        )
        self.launch_bank_hard_file = (
            str(launch_bank_cfg.hard_file).strip() if launch_bank_cfg.hard_file is not None else ""
        )
        self.launch_bank_shuffle = bool(launch_bank_cfg.shuffle)
        self.launch_bank = LaunchBankBuffer(device=self.device, shuffle=self.launch_bank_shuffle)
        has_multi = any(
            [
                bool(self.launch_bank_easy_file),
                bool(self.launch_bank_medium_file),
                bool(self.launch_bank_hard_file),
            ]
        )
        if has_multi:
            self.launch_bank.load_levels(
                easy_file=(self.launch_bank_easy_file or None),
                medium_file=(self.launch_bank_medium_file or None),
                hard_file=(self.launch_bank_hard_file or None),
            )
            if bool(launch_bank_cfg.use_curriculum):
                self.launch_bank.enable_curriculum(
                    start_probs=tuple(float(v) for v in launch_bank_cfg.curriculum_start_probs),
                    target_probs=tuple(float(v) for v in launch_bank_cfg.curriculum_target_probs),
                    progress_up=float(launch_bank_cfg.curriculum_progress_up),
                    progress_down=float(launch_bank_cfg.curriculum_progress_down),
                    ema_alpha=float(launch_bank_cfg.curriculum_ema_alpha),
                    min_level_prob=float(launch_bank_cfg.curriculum_min_level_prob),
                )
        else:
            if not self.launch_bank_file:
                raise ValueError(
                    "HighLevelTennisCommand requires offline launch banks. "
                    "Set launch.bank.file (single) or launch.bank.easy/medium/hard_file (multi-level)."
                )
            self.launch_bank.load(self.launch_bank_file)

        self.outgoing_speed_minmax = torch.tensor(cfg.outgoing_speed_minmax, device=self.device, dtype=torch.float32)

        racket_body_name = str(cfg.racket_body_name)
        racket_body_ids, racket_names = self.asset.find_bodies(racket_body_name)
        if len(racket_body_ids) != 1:
            raise ValueError(
                f"Expected exactly one racket body from '{racket_body_name}', got {racket_names}."
            )
        self.racket_body_id = int(racket_body_ids[0])
        self.racket_center_offset = torch.tensor(cfg.racket_center_offset, device=self.device, dtype=torch.float32)
        self.use_racket_body_contact_sensor = bool(cfg.use_racket_body_contact_sensor)
        self.enable_racket_body_direct_contact_guard = bool(cfg.enable_racket_body_direct_contact_guard)
        self.contact_sensors = ContactSensorHandles.from_scene(self.env.scene)

        self.racket_contact_geom_ids = torch.zeros((0,), dtype=torch.int32, device=self.device)
        self.racket_body_contact_geom_ids = torch.zeros((0,), dtype=torch.int32, device=self.device)
        racket_geom_ids, racket_geom_names = self.asset.find_geoms("tennis_racket_collision")
        if len(racket_geom_ids) == 0:
            raise ValueError(
                f"Racket collision geom not found for robot. matched={racket_geom_names}"
            )
        self.racket_contact_geom_ids = torch.tensor(
            [int(v) for v in racket_geom_ids], dtype=torch.int32, device=self.device
        )
        body_geom_ids, body_geom_names = self.asset.find_geoms(".*_collision")
        body_ids_kept: list[int] = []
        for gid, gname in zip(body_geom_ids, body_geom_names):
            name = str(gname)
            if name == "tennis_racket_collision":
                continue
            # Exclude only mounting hand, keep all other body parts.
            if name == "right_hand_collision":
                continue
            body_ids_kept.append(int(gid))
        if len(body_ids_kept) > 0:
            self.racket_body_contact_geom_ids = torch.tensor(
                body_ids_kept, dtype=torch.int32, device=self.device
            )

        ball_body_ids, _ = self.ball.find_bodies("tennis_ball")
        if len(ball_body_ids) != 1:
            raise ValueError("Tennis ball entity must contain exactly one body named 'tennis_ball'.")
        self.ball_body_ids = torch.tensor(ball_body_ids, device=self.device, dtype=torch.long)

        gravity_z = self._read_gravity_z_value(self.env.sim.model.opt.gravity)
        self.gravity = torch.full((self.num_envs, 1), gravity_z, dtype=torch.float32, device=self.device)
        self.root_default_joint_pos = self.asset.data.default_joint_pos.clone()
        self.root_default_joint_vel = self.asset.data.default_joint_vel.clone()
        self.spawn_root_pos_w = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.spawn_root_quat_w = torch.zeros((self.num_envs, 4), dtype=torch.float32, device=self.device)
        self.spawn_root_quat_w[:, 0] = 1.0
        self.spawn_root_forward_xy = torch.zeros((self.num_envs, 2), dtype=torch.float32, device=self.device)

        self.task_step = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.finished = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.success_done = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.timeout = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hit_limit_reached = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.consecutive_return_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._rally_relaunch_mask = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._rally_launch_delay = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self._rally_launch_ready = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.has_hit = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.has_bounce = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.has_pass_net = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.bounce_in = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fail_miss = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fail_net = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fail_out = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fail_style = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.fail_racket_body = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.first_hit_step = torch.full(
            (self.num_envs,), self.max_task_steps, dtype=torch.int32, device=self.device
        )
        self.first_bounce_step = torch.full(
            (self.num_envs,), self.max_task_steps, dtype=torch.int32, device=self.device
        )

        self.hit_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.bounce_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.pass_net_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.net_clearance_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hit_stroke_mode_match_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hit_stroke_mode_mismatch_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.racket_ball_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_net_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_court_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.racket_body_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.racket_ball_contact_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_net_contact_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_court_contact_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.racket_body_contact_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stroke_style_violation_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prehit_zone = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prehit_zone_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prehit_zone_entered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hit_cooldown = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.pre_hit_dead_ball_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.post_hit_dead_ball_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.hit_racket_speed = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self._last_hit_mask = None
        self.highlevel_action = torch.zeros(self.num_envs, self.highlevel_latent_dim, dtype=torch.float32, device=self.device)
        self.correction_action = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self.prev_correction_action = torch.zeros_like(self.correction_action)
        self.correction_action_rate = torch.zeros_like(self.correction_action)
        self.racket_acc_norm = torch.zeros(self.num_envs, 1, dtype=torch.float32, device=self.device)
        self._prev_racket_vel_w = torch.zeros(self.num_envs, 3, dtype=torch.float32, device=self.device)
        self.ball_pos_w_history = torch.zeros(
            (self.num_envs, self.ball_obs_buffer_size, 3), dtype=torch.float32, device=self.device
        )
        self.ball_vel_w_history = torch.zeros(
            (self.num_envs, self.ball_obs_buffer_size, 3), dtype=torch.float32, device=self.device
        )
        self._prime_ball_obs_history(torch.arange(self.num_envs, device=self.device, dtype=torch.long))

        self.target_bounce_w = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self.launch_level_ids = torch.full((self.num_envs,), -1, device=self.device, dtype=torch.long)
        self.stroke_mode_target = torch.zeros(self.num_envs, device=self.device, dtype=torch.long)
        self.stroke_mode_contact_lateral = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self.bounce_pos_w = torch.zeros(self.num_envs, 3, device=self.device, dtype=torch.float32)
        self.prev_ball_target_dist = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self.ball_target_dist = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self.ball_target_progress_buf = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self.prev_net_dist = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self.net_dist = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self.net_dist_progress_buf = torch.zeros(self.num_envs, 1, device=self.device, dtype=torch.float32)
        self.gravity_dir = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)

        self._forward_dir_b = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=torch.float32).expand(self.num_envs, -1)
        self._left_dir_b = torch.tensor([0.0, 1.0, 0.0], device=self.device, dtype=torch.float32).expand(self.num_envs, -1)

        self.action_joint_names: list[str] = []
        self.lower_body_action_ids = torch.zeros((0,), dtype=torch.long, device=self.device)
        self.wrist_action_ids = torch.zeros((0,), dtype=torch.long, device=self.device)
        self.wrist_joint_ids_asset = self._resolve_asset_joint_ids(list(self.wrist_joint_patterns))
        self.recovery_upper_joint_ids_asset = self._resolve_asset_joint_ids(list(self.recovery_upper_joint_patterns))
        self.wrist_actuator_ids = torch.zeros((0,), dtype=torch.long, device=self.device)
        self._action_layout_ready = False
        self.step_schedule(0.0, None)
