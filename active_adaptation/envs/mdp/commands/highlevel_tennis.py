from __future__ import annotations

import math
import os
import re
from typing import Sequence

import numpy as np
import torch

from active_adaptation.envs.mdp import observation, reward, termination
from active_adaptation.utils import symmetry as sym_utils
from active_adaptation.utils.math import quat_apply, quat_apply_inverse

from .base import Command


def _exp_reward(error: torch.Tensor, sigma: Sequence[float] | None = None) -> torch.Tensor:
    if sigma is None or len(sigma) == 0:
        sigma = (0.2,)
    rewards = [torch.exp(-error / float(s)) for s in sigma]
    return sum(rewards) / float(len(rewards))


def _yaw_to_quat(yaw: float, device: str) -> torch.Tensor:
    half = 0.5 * yaw
    return torch.tensor(
        [math.cos(half), 0.0, 0.0, math.sin(half)],
        dtype=torch.float32,
        device=device,
    )


class HighLevelTennisCommand(Command):
    def __init__(
        self,
        env,
        robot_spawn_pos: Sequence[float] = (0.0, -8, 0.81),
        robot_spawn_yaw: float = math.pi / 2.0,
        robot_spawn_xy_noise: Sequence[float] = (0.0, 0.0),
        robot_spawn_yaw_noise_deg: float = 0.0,
        max_task_steps: int = 1000,
        max_consecutive_hits_before_finish: int = 8,
        max_consecutive_returns_before_finish: int | None = None,
        relaunch_on_success: bool = True,
        launch_interval_s: float = 2.0,
        highlevel_latent_dim: int = 32,
        ball_obs_history_steps: Sequence[int] = (0, 1, 2, 3, 4),
        ball_obs_prediction_horizon_s: float = 1.5,
        approach_oracle_mix_start: float = 1.0,
        approach_oracle_mix_end: float = 0.0,
        approach_oracle_mix_full_progress: float = 0.5,
        approach_contact_lead_time: float = 0.16,
        approach_contact_min_t: float = 0.06,
        approach_contact_max_t: float = 1.20,
        hitting_zone_xy_radius: float = 0.40,
        hitting_zone_z_tol: float = 0.26,
        hitting_zone_time_min: float = 0.05,
        hitting_zone_time_max: float = 0.65,
        wrist_joint_patterns: Sequence[str] = ("right_wrist_.*_joint",),
        stroke_style_min_racket_speed: float = 2.5,
        stroke_style_min_forward_speed: float = 0.2,
        ball_radius: float = 0.0335,
        ball_mass: float = 0.057,
        air_density: float = 1.21,
        air_drag_k: float = 1.0,
        drag_coef: float = 0.55,
        lift_spin_scale: float = 5.0,
        spin_damping_coef: float = 0.003,
        bounce_restitution: float = 0.76,
        bounce_friction: float = 0.88,
        bounce_spin_friction: float = 0.12,
        bounce_spin_decay: float = 0.78,
        launcher_x_range: Sequence[float] = (-2.8, 2.8),
        launcher_y_range: Sequence[float] = (8.5, 10.5),
        launcher_z_range: Sequence[float] = (1.5, 2.4),
        strike_x_range: Sequence[float] = (-1.0, 1.0),
        strike_y_range: Sequence[float] = (-1.3, -0.2),
        strike_z_range: Sequence[float] = (0.9, 1.3),
        flight_t_range: Sequence[float] = (0.45, 0.75),
        launch_speed_range: Sequence[float] = (14.0, 32.0),
        launch_spin_rps_range: Sequence[float] = (-9.0, 9.0),
        launch_solver_iters: int = 2,
        launch_resample_attempts: int = 5,
        launch_strike_tolerance: float = 0.22,
        launch_prediction_substeps: int = 2,
        launch_predict_dt: float | None = 0.01,
        enforce_launch_net_clearance: bool = True,
        launch_net_clearance_margin: float = 0.05,
        launch_angle_deg_range: Sequence[float] = (7.0, 18.0),
        launch_min_vz: float = 3.0,
        launch_min_forward_speed: float = 10.0,
        launch_clearance_correction_iters: int = 2,
        enforce_launch_incoming_bounce_in: bool = True,
        incoming_bounce_x_range: Sequence[float] = (-3.8, 3.8),
        incoming_bounce_y_range: Sequence[float] = (-10.8, -0.4),
        launch_bank_file: str | None = None,
        require_launch_bank: bool = True,
        launch_bank_shuffle: bool = True,
        target_x_range: Sequence[float] = (-3.5, 3.5),
        target_y_range: Sequence[float] = (7.2, 11.2),
        court_x_limit: float = 4.2,
        court_y_limit: float = 12.2,
        court_y_min_success: float = 0.0,
        success_requires_ball_exit: bool = True,
        success_exit_x_margin: float = 0.0,
        success_exit_y_margin: float = 0.0,
        net_height: float = 0.914,
        net_half_thickness: float = 0.12,
        net_clearance_reward_margin: float = 0.12,
        miss_margin_y: float = 0.55,
        out_margin_z: float = -0.25,
        pre_hit_dead_ball_speed_thres: float = 1.0,
        pre_hit_dead_ball_height_margin: float = 0.05,
        pre_hit_dead_ball_patience_steps: int = 20,
        pre_hit_dead_ball_min_steps: int = 24,
        post_hit_dead_ball_speed_thres: float = -1.0,
        post_hit_dead_ball_height_margin: float = 0.05,
        post_hit_dead_ball_patience_steps: int = 30,
        post_hit_dead_ball_min_steps: int = 24,
        ground_recover_enable: bool = False,
        ground_recover_trigger_z: float = 0.0,
        ground_recover_surface_z: float = 0.016,
        ground_recover_speed_thres: float = 1.0,
        ground_recover_bounce_restitution: float = 0.7,
        ground_recover_min_upward_speed: float = 1.2,
        ground_recover_horizontal_damping: float = 0.9,
        ground_recover_spin_damping: float = 0.85,
        ground_recover_max_speed: float = 5.0,
        racket_body_name: str = "tennis_racket_mount",
        racket_center_offset: Sequence[float] = (0.1025, -0.004, 0.4),
        outgoing_base_speed: float = 10.0,
        outgoing_speed_gain: float = 0.85,
        outgoing_speed_minmax: Sequence[float] = (8.0, 26.0),
        outgoing_vz_minmax: Sequence[float] = (1.8, 8.0),
        hit_spin_gain: float = 20.0,
        max_spin_rad_s: float = 260.0,
        debug_draw: bool = False,
    ):
        super().__init__(env)
        if "tennis_ball" not in self.env.scene.entities:
            raise RuntimeError(
                "HighLevelTennisCommand requires task.tennis.add_ball=true "
                "so entity 'tennis_ball' is available."
            )
        self.ball = self.env.scene["tennis_ball"]
        self.debug_draw_enabled = bool(debug_draw)

        self.robot_spawn_pos = torch.tensor(robot_spawn_pos, device=self.device, dtype=torch.float32)
        self.robot_spawn_yaw = float(robot_spawn_yaw)
        self.robot_spawn_quat = _yaw_to_quat(self.robot_spawn_yaw, self.device)
        self.robot_spawn_xy_noise = torch.tensor(robot_spawn_xy_noise, device=self.device, dtype=torch.float32)
        if self.robot_spawn_xy_noise.numel() != 2:
            raise ValueError(
                f"robot_spawn_xy_noise must have length 2, got shape={tuple(self.robot_spawn_xy_noise.shape)}"
            )
        self.robot_spawn_yaw_noise_rad = math.radians(float(robot_spawn_yaw_noise_deg))

        self.max_task_steps = int(max_task_steps)
        if max_consecutive_returns_before_finish is None:
            max_consecutive_returns_before_finish = max_consecutive_hits_before_finish
        self.max_consecutive_returns_before_finish = max(0, int(max_consecutive_returns_before_finish))
        self.relaunch_on_success = bool(relaunch_on_success)
        self.launch_interval_s = float(max(launch_interval_s, 0.0))
        self.launch_interval_steps = int(round(self.launch_interval_s / float(self.env.step_dt)))
        self.highlevel_latent_dim = int(highlevel_latent_dim)
        self.ball_obs_history_steps = tuple(int(s) for s in ball_obs_history_steps)
        if len(self.ball_obs_history_steps) == 0:
            raise ValueError("ball_obs_history_steps must contain at least one index (e.g. [0,1,2,3]).")
        if min(self.ball_obs_history_steps) < 0:
            raise ValueError(f"ball_obs_history_steps must be non-negative, got {self.ball_obs_history_steps}.")
        self.ball_obs_buffer_size = int(max(self.ball_obs_history_steps) + 1)
        self.ball_obs_prediction_horizon_s = max(float(ball_obs_prediction_horizon_s), 0.1)
        self.approach_oracle_mix_start = float(approach_oracle_mix_start)
        self.approach_oracle_mix_end = float(approach_oracle_mix_end)
        self.approach_oracle_mix_full_progress = max(float(approach_oracle_mix_full_progress), 1.0e-6)
        self.approach_oracle_mix = self.approach_oracle_mix_start
        self.approach_contact_lead_time = max(float(approach_contact_lead_time), 0.0)
        self.approach_contact_min_t = max(float(approach_contact_min_t), 0.01)
        self.approach_contact_max_t = max(float(approach_contact_max_t), self.approach_contact_min_t + 0.01)
        self.hitting_zone_xy_radius = max(float(hitting_zone_xy_radius), 0.05)
        self.hitting_zone_z_tol = max(float(hitting_zone_z_tol), 0.05)
        self.hitting_zone_time_min = max(float(hitting_zone_time_min), 0.01)
        self.hitting_zone_time_max = max(float(hitting_zone_time_max), self.hitting_zone_time_min + 0.01)
        self.wrist_joint_patterns = tuple(str(p) for p in wrist_joint_patterns)
        self.stroke_style_min_racket_speed = float(stroke_style_min_racket_speed)
        self.stroke_style_min_forward_speed = float(stroke_style_min_forward_speed)
        self.ball_radius = float(ball_radius)
        self.ball_mass = float(ball_mass)
        self.air_density = float(air_density)
        self.air_drag_k = float(air_drag_k)
        self.drag_coef = float(drag_coef)
        self.lift_spin_scale = float(lift_spin_scale)
        self.spin_damping_coef = float(spin_damping_coef)
        self.aero_force_k = 0.5 * self.air_density * math.pi * (self.ball_radius ** 2) * self.air_drag_k
        self.bounce_restitution = float(bounce_restitution)
        self.bounce_friction = float(bounce_friction)
        self.bounce_spin_friction = float(bounce_spin_friction)
        self.bounce_spin_decay = float(bounce_spin_decay)
        self.net_height = float(net_height)
        self.net_half_thickness = float(net_half_thickness)
        self.net_clearance_reward_margin = float(net_clearance_reward_margin)
        self.miss_margin_y = float(miss_margin_y)
        self.out_margin_z = float(out_margin_z)
        self.pre_hit_dead_ball_speed_thres = float(pre_hit_dead_ball_speed_thres)
        self.pre_hit_dead_ball_height_margin = float(pre_hit_dead_ball_height_margin)
        self.pre_hit_dead_ball_patience_steps = max(1, int(pre_hit_dead_ball_patience_steps))
        self.pre_hit_dead_ball_min_steps = max(1, int(pre_hit_dead_ball_min_steps))
        self.post_hit_dead_ball_speed_thres = float(post_hit_dead_ball_speed_thres)
        self.post_hit_dead_ball_height_margin = float(post_hit_dead_ball_height_margin)
        self.post_hit_dead_ball_patience_steps = max(1, int(post_hit_dead_ball_patience_steps))
        self.post_hit_dead_ball_min_steps = max(1, int(post_hit_dead_ball_min_steps))
        self.ground_recover_enable = bool(ground_recover_enable)
        self.ground_recover_trigger_z = float(ground_recover_trigger_z)
        self.ground_recover_surface_z = float(ground_recover_surface_z)
        self.ground_recover_speed_thres = float(ground_recover_speed_thres)
        self.ground_recover_bounce_restitution = float(max(ground_recover_bounce_restitution, 0.0))
        self.ground_recover_min_upward_speed = float(max(ground_recover_min_upward_speed, 0.0))
        self.ground_recover_horizontal_damping = float(min(max(ground_recover_horizontal_damping, 0.0), 1.0))
        self.ground_recover_spin_damping = float(min(max(ground_recover_spin_damping, 0.0), 1.0))
        self.ground_recover_max_speed = float(max(ground_recover_max_speed, 0.1))
        self.court_x_limit = float(court_x_limit)
        self.court_y_limit = float(court_y_limit)
        self.court_y_min_success = float(court_y_min_success)
        self.success_requires_ball_exit = bool(success_requires_ball_exit)
        self.success_exit_x_margin = float(success_exit_x_margin)
        self.success_exit_y_margin = float(success_exit_y_margin)

        self.launcher_x_range = torch.tensor(launcher_x_range, device=self.device, dtype=torch.float32)
        self.launcher_y_range = torch.tensor(launcher_y_range, device=self.device, dtype=torch.float32)
        self.launcher_z_range = torch.tensor(launcher_z_range, device=self.device, dtype=torch.float32)
        self.strike_x_range = torch.tensor(strike_x_range, device=self.device, dtype=torch.float32)
        self.strike_y_range = torch.tensor(strike_y_range, device=self.device, dtype=torch.float32)
        self.strike_z_range = torch.tensor(strike_z_range, device=self.device, dtype=torch.float32)
        self.flight_t_range = torch.tensor(flight_t_range, device=self.device, dtype=torch.float32)
        self.launch_speed_range = torch.tensor(launch_speed_range, device=self.device, dtype=torch.float32)
        self.launch_spin_rps_range = torch.tensor(launch_spin_rps_range, device=self.device, dtype=torch.float32)
        self.launch_solver_iters = int(launch_solver_iters)
        self.launch_resample_attempts = int(launch_resample_attempts)
        self.launch_strike_tolerance = float(launch_strike_tolerance)
        self.launch_prediction_substeps = max(1, int(launch_prediction_substeps))
        base_predict_dt = float(getattr(self.env, "physics_dt", 0.005)) / float(self.launch_prediction_substeps)
        if launch_predict_dt is None:
            self.launch_predict_dt = base_predict_dt
        else:
            self.launch_predict_dt = max(float(launch_predict_dt), base_predict_dt)
        self.enforce_launch_net_clearance = bool(enforce_launch_net_clearance)
        self.launch_net_clearance_margin = float(launch_net_clearance_margin)
        self.launch_angle_deg_range = torch.tensor(launch_angle_deg_range, device=self.device, dtype=torch.float32)
        self.launch_min_vz = float(launch_min_vz)
        self.launch_min_forward_speed = float(launch_min_forward_speed)
        self.launch_clearance_correction_iters = max(0, int(launch_clearance_correction_iters))
        self.enforce_launch_incoming_bounce_in = bool(enforce_launch_incoming_bounce_in)
        self.incoming_bounce_x_range = torch.tensor(incoming_bounce_x_range, device=self.device, dtype=torch.float32)
        self.incoming_bounce_y_range = torch.tensor(incoming_bounce_y_range, device=self.device, dtype=torch.float32)
        self.launch_bank_file = str(launch_bank_file).strip() if launch_bank_file is not None else ""
        self.require_launch_bank = bool(require_launch_bank)
        self.launch_bank_shuffle = bool(launch_bank_shuffle)
        self._launch_bank_pos_local = None
        self._launch_bank_vel = None
        self._launch_bank_ang = None
        self._launch_bank_target_local = None
        self._launch_bank_size = 0
        self._launch_bank_ptr = 0
        self._launch_bank_perm = None
        self.target_x_range = torch.tensor(target_x_range, device=self.device, dtype=torch.float32)
        self.target_y_range = torch.tensor(target_y_range, device=self.device, dtype=torch.float32)
        if self.require_launch_bank and not self.launch_bank_file:
            raise ValueError(
                "HighLevelTennisCommand requires a valid launch_bank_file. "
                "Set task.command.launch_bank_file to an offline bank .npz."
            )
        if self.launch_bank_file:
            self._load_launch_bank(self.launch_bank_file)

        self.outgoing_base_speed = float(outgoing_base_speed)
        self.outgoing_speed_gain = float(outgoing_speed_gain)
        self.outgoing_speed_minmax = torch.tensor(outgoing_speed_minmax, device=self.device, dtype=torch.float32)
        self.outgoing_vz_minmax = torch.tensor(outgoing_vz_minmax, device=self.device, dtype=torch.float32)
        self.hit_spin_gain = float(hit_spin_gain)
        self.max_spin_rad_s = float(max_spin_rad_s)

        racket_body_ids, racket_names = self.asset.find_bodies(racket_body_name)
        if len(racket_body_ids) != 1:
            raise ValueError(
                f"Expected exactly one racket body from '{racket_body_name}', got {racket_names}."
            )
        self.racket_body_id = int(racket_body_ids[0])
        self.racket_center_offset = torch.tensor(racket_center_offset, device=self.device, dtype=torch.float32)
        self._racket_velocity_sensor = None
        if "robot/tennis_racket_center_global_linvel" in self.env.scene.sensors:
            self._racket_velocity_sensor = self.env.scene["robot/tennis_racket_center_global_linvel"]
        self._racket_ball_contact_sensor = None
        if "racket_ball_contact" in self.env.scene.sensors:
            self._racket_ball_contact_sensor = self.env.scene["racket_ball_contact"]
        self._ball_net_contact_sensor = None
        if "ball_net_contact" in self.env.scene.sensors:
            self._ball_net_contact_sensor = self.env.scene["ball_net_contact"]
        self._ball_court_contact_sensor = None
        if "ball_court_contact" in self.env.scene.sensors:
            self._ball_court_contact_sensor = self.env.scene["ball_court_contact"]

        ball_body_ids, _ = self.ball.find_bodies("tennis_ball")
        if len(ball_body_ids) != 1:
            raise ValueError("Tennis ball entity must contain exactly one body named 'tennis_ball'.")
        self.ball_body_ids = torch.tensor(ball_body_ids, device=self.device, dtype=torch.long)

        gravity_z = self._read_gravity_z_value(self.env.sim.model.opt.gravity)
        self.gravity = torch.full((self.num_envs, 1), gravity_z, dtype=torch.float32, device=self.device)
        self.root_default_joint_pos = self.asset.data.default_joint_pos.clone()
        self.root_default_joint_vel = self.asset.data.default_joint_vel.clone()

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
        self.racket_ball_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_net_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_court_contact = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.racket_ball_contact_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_net_contact_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.ball_court_contact_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.stroke_style_violation_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prehit_zone = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prehit_zone_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.prehit_zone_entered = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self.hit_cooldown = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.pre_hit_dead_ball_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.post_hit_dead_ball_steps = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.ground_recover_event = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
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
        self.oracle_incoming_bounce_xy_w = torch.zeros(self.num_envs, 2, device=self.device, dtype=torch.float32)
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
        self.wrist_actuator_ids = torch.zeros((0,), dtype=torch.long, device=self.device)
        self._action_layout_ready = False
        self.step_schedule(0.0, None)

    def step_schedule(self, progress: float, iters: int | None = None):
        p = float(min(max(progress, 0.0), 1.0))
        ramp = min(p / self.approach_oracle_mix_full_progress, 1.0)
        self.approach_oracle_mix = (
            (1.0 - ramp) * self.approach_oracle_mix_start
            + ramp * self.approach_oracle_mix_end
        )

    def _sample_robot_spawn(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        env_origins = self.env.scene.env_origins[env_ids]
        n = env_ids.numel()
        pos_w = env_origins + self.robot_spawn_pos.unsqueeze(0)
        if (self.robot_spawn_xy_noise.abs() > 1.0e-8).any():
            noise_xy = (torch.rand((n, 2), device=self.device) * 2.0 - 1.0) * self.robot_spawn_xy_noise.unsqueeze(0)
            pos_w[:, :2] = pos_w[:, :2] + noise_xy

        yaw = torch.full((n,), self.robot_spawn_yaw, device=self.device, dtype=torch.float32)
        if self.robot_spawn_yaw_noise_rad > 1.0e-8:
            yaw_noise = (torch.rand((n,), device=self.device) * 2.0 - 1.0) * self.robot_spawn_yaw_noise_rad
            yaw = yaw + yaw_noise
        half = 0.5 * yaw
        quat_w = torch.stack(
            [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)],
            dim=-1,
        )
        return pos_w, quat_w

    def _write_ball_launch(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        launch_pos_w, launch_vel_w, launch_ang_w, target_bounce_w = self._sample_ball_launch(env_ids)
        ball_state = torch.zeros((env_ids.numel(), 13), device=self.device, dtype=torch.float32)
        ball_state[:, :3] = launch_pos_w
        ball_state[:, 3] = 1.0
        ball_state[:, 7:10] = launch_vel_w
        ball_state[:, 10:13] = launch_ang_w
        self.ball.write_root_state_to_sim(ball_state, env_ids=env_ids)
        self.target_bounce_w[env_ids] = target_bounce_w
        gravity_z = self._get_current_gravity_z(env_ids)
        incoming_bounce_xy, _ = self._predict_first_bounce_ballistic(
            launch_pos=launch_pos_w,
            vel=launch_vel_w,
            gravity_z=gravity_z,
        )
        self.oracle_incoming_bounce_xy_w[env_ids] = incoming_bounce_xy
        self.ball_pos_w_history[env_ids] = launch_pos_w.unsqueeze(1)
        self.ball_vel_w_history[env_ids] = launch_vel_w.unsqueeze(1)
        launch_pos_l = launch_pos_w - self.env.scene.env_origins[env_ids]
        launch_net_dist = launch_pos_l[:, 1:2].abs()
        self.prev_net_dist[env_ids] = launch_net_dist
        self.net_dist[env_ids] = launch_net_dist
        self.net_dist_progress_buf[env_ids] = 0.0

    def _prime_ball_obs_history(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        ball_pos_w = self.ball.data.root_link_pos_w[env_ids]
        ball_vel_w = self.ball.data.root_link_lin_vel_w[env_ids]
        self.ball_pos_w_history[env_ids] = ball_pos_w.unsqueeze(1)
        self.ball_vel_w_history[env_ids] = ball_vel_w.unsqueeze(1)

    def _push_ball_obs_history(self) -> None:
        self.ball_pos_w_history = self.ball_pos_w_history.roll(1, dims=1)
        self.ball_vel_w_history = self.ball_vel_w_history.roll(1, dims=1)
        self.ball_pos_w_history[:, 0] = self.ball.data.root_link_pos_w
        self.ball_vel_w_history[:, 0] = self.ball.data.root_link_lin_vel_w

    def _reset_rally_state(self, env_ids: torch.Tensor, *, reset_hit_counter: bool) -> None:
        if env_ids.numel() == 0:
            return
        self.task_step[env_ids] = 0
        self.finished[env_ids] = False
        self.success[env_ids] = False
        self.success_done[env_ids] = False
        self.timeout[env_ids] = False
        self.hit_limit_reached[env_ids] = False
        self.has_hit[env_ids] = False
        self.has_bounce[env_ids] = False
        self.has_pass_net[env_ids] = False
        self.bounce_in[env_ids] = False
        self.fail_miss[env_ids] = False
        self.fail_net[env_ids] = False
        self.fail_out[env_ids] = False
        self.fail_style[env_ids] = False
        self.first_hit_step[env_ids] = self.max_task_steps
        self.first_bounce_step[env_ids] = self.max_task_steps
        self.hit_event[env_ids] = False
        self.bounce_event[env_ids] = False
        self.pass_net_event[env_ids] = False
        self.net_clearance_event[env_ids] = False
        self.stroke_style_violation_event[env_ids] = False
        self.prehit_zone[env_ids] = False
        self.prehit_zone_event[env_ids] = False
        self.prehit_zone_entered[env_ids] = False
        self.racket_ball_contact[env_ids] = False
        self.ball_net_contact[env_ids] = False
        self.ball_court_contact[env_ids] = False
        self.racket_ball_contact_event[env_ids] = False
        self.ball_net_contact_event[env_ids] = False
        self.ball_court_contact_event[env_ids] = False
        self.hit_cooldown[env_ids] = 0
        self.pre_hit_dead_ball_steps[env_ids] = 0
        self.post_hit_dead_ball_steps[env_ids] = 0
        self.ground_recover_event[env_ids] = False
        self.hit_racket_speed[env_ids] = 0.0
        if self._last_hit_mask is not None:
            self._last_hit_mask[env_ids] = False
        self.bounce_pos_w[env_ids] = 0.0
        self.oracle_incoming_bounce_xy_w[env_ids] = 0.0
        self.prev_ball_target_dist[env_ids] = 0.0
        self.ball_target_dist[env_ids] = 0.0
        self.ball_target_progress_buf[env_ids] = 0.0
        self.prev_net_dist[env_ids] = 0.0
        self.net_dist[env_ids] = 0.0
        self.net_dist_progress_buf[env_ids] = 0.0
        self._rally_relaunch_mask[env_ids] = False
        self._rally_launch_delay[env_ids] = 0
        self._rally_launch_ready[env_ids] = False
        self.racket_acc_norm[env_ids] = 0.0
        self.highlevel_action[env_ids] = 0.0
        self.correction_action[env_ids] = 0.0
        self.prev_correction_action[env_ids] = 0.0
        self.correction_action_rate[env_ids] = 0.0
        if reset_hit_counter:
            self.consecutive_return_count[env_ids] = 0

    def _sample_uniform(self, env_ids: torch.Tensor, ranges: torch.Tensor) -> torch.Tensor:
        return torch.rand((env_ids.numel(),), device=self.device) * (ranges[1] - ranges[0]) + ranges[0]

    def _sample_uniform_n(self, num_samples: int, ranges: torch.Tensor) -> torch.Tensor:
        return torch.rand((num_samples,), device=self.device) * (ranges[1] - ranges[0]) + ranges[0]

    def _resolve_action_joint_ids(self, patterns: Sequence[str]) -> torch.Tensor:
        if len(patterns) == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)
        ids: list[int] = []
        for i, name in enumerate(self.action_joint_names):
            if any(re.match(pat, name) for pat in patterns):
                ids.append(i)
        if len(ids) == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)
        return torch.tensor(sorted(set(ids)), dtype=torch.long, device=self.device)

    def _resolve_asset_joint_ids(self, patterns: Sequence[str]) -> torch.Tensor:
        if len(patterns) == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)
        ids: list[int] = []
        for pat in patterns:
            joint_ids, _ = self.asset.find_joints(pat)
            ids.extend([int(i) for i in joint_ids])
        if len(ids) == 0:
            return torch.zeros((0,), dtype=torch.long, device=self.device)
        return torch.tensor(sorted(set(ids)), dtype=torch.long, device=self.device)

    def _ensure_action_layout(self) -> None:
        if self._action_layout_ready:
            return
        if not hasattr(self.env, "action_manager"):
            return
        self.action_joint_names = list(self.env.action_manager.joint_names)
        self.lower_body_action_ids = self._resolve_action_joint_ids([r".*(waist|hip|knee|ankle).*_joint"])
        self.wrist_action_ids = self._resolve_action_joint_ids(list(self.wrist_joint_patterns))
        wrist_joint_names = [self.action_joint_names[int(i)] for i in self.wrist_action_ids.tolist()]
        name_to_act = {n: i for i, n in enumerate(self.asset.actuator_names)}
        wrist_act_ids = [name_to_act[n] for n in wrist_joint_names if n in name_to_act]
        self.wrist_actuator_ids = (
            torch.tensor(wrist_act_ids, dtype=torch.long, device=self.device)
            if len(wrist_act_ids) > 0
            else torch.zeros((0,), dtype=torch.long, device=self.device)
        )
        self._action_layout_ready = True

    def _read_gravity_z_value(self, gravity_opt) -> float:
        if hasattr(gravity_opt, "_tensor"):
            t = gravity_opt._tensor
            if isinstance(t, torch.Tensor):
                if t.ndim > 0 and t.shape[-1] >= 3:
                    return float(t[..., 2].reshape(-1)[0].detach().cpu().item())
                return float(t.reshape(-1)[0].detach().cpu().item())
        if isinstance(gravity_opt, torch.Tensor):
            g = gravity_opt
            if g.ndim == 0:
                return float(g.item())
            if g.shape[-1] >= 3:
                return float(g[..., 2].reshape(-1)[0].item())
            return float(g.reshape(-1)[0].item())
        for idx in ((Ellipsis, 2), (0, 2), (2,)):
            try:
                if len(idx) == 1:
                    val = gravity_opt[idx[0]]
                else:
                    val = gravity_opt[idx]
                if hasattr(val, "item"):
                    return float(val.item())
                return float(val)
            except Exception:
                continue
        arr = np.asarray(gravity_opt, dtype=np.float32)
        if arr.ndim > 0 and arr.shape[-1] >= 3:
            return float(arr[..., 2].reshape(-1)[0])
        return float(arr.reshape(-1)[0])

    def _get_current_gravity_z(self, env_ids: torch.Tensor) -> torch.Tensor:
        gravity_z = self._read_gravity_z_value(self.env.sim.model.opt.gravity)
        self.gravity[env_ids] = gravity_z
        return self.gravity[env_ids]

    def _sensor_contact_found(self, sensor) -> torch.Tensor:
        data = sensor.data
        found = None
        if getattr(data, "found", None) is not None:
            found = data.found > 0.0
            if found.ndim > 1:
                found = found.reshape(found.shape[0], -1).any(dim=-1)
        if getattr(data, "force_history", None) is not None:
            hist_force = data.force_history
            hist_hit = hist_force.norm(dim=-1) > 1.0e-6
            if hist_hit.ndim > 1:
                hist_hit = hist_hit.reshape(hist_hit.shape[0], -1).any(dim=-1)
            if found is None:
                found = hist_hit
            else:
                found = found | hist_hit
        if found is None:
            return torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
        return found.to(dtype=torch.bool, device=self.device)

    def _update_contact_events(self) -> None:
        prev_racket_ball = self.racket_ball_contact.clone()
        prev_ball_net = self.ball_net_contact.clone()
        prev_ball_court = self.ball_court_contact.clone()

        if self._racket_ball_contact_sensor is not None:
            self.racket_ball_contact[:] = self._sensor_contact_found(self._racket_ball_contact_sensor)
        else:
            self.racket_ball_contact[:] = False
        if self._ball_net_contact_sensor is not None:
            self.ball_net_contact[:] = self._sensor_contact_found(self._ball_net_contact_sensor)
        else:
            self.ball_net_contact[:] = False
        if self._ball_court_contact_sensor is not None:
            self.ball_court_contact[:] = self._sensor_contact_found(self._ball_court_contact_sensor)
        else:
            self.ball_court_contact[:] = False

        self.racket_ball_contact_event[:] = self.racket_ball_contact & (~prev_racket_ball)
        self.ball_net_contact_event[:] = self.ball_net_contact & (~prev_ball_net)
        self.ball_court_contact_event[:] = self.ball_court_contact & (~prev_ball_court)

    def _racket_state_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        body_pos = self.asset.data.body_link_pos_w[:, self.racket_body_id]
        body_quat = self.asset.data.body_link_quat_w[:, self.racket_body_id]
        center_offset_w = quat_apply(body_quat, self.racket_center_offset.unsqueeze(0).expand(self.num_envs, -1))
        racket_pos_w = body_pos + center_offset_w
        if self._racket_velocity_sensor is not None:
            racket_vel_w = self._racket_velocity_sensor.data
        else:
            body_lin_vel = self.asset.data.body_link_lin_vel_w[:, self.racket_body_id]
            body_ang_vel = self.asset.data.body_link_ang_vel_w[:, self.racket_body_id]
            racket_vel_w = body_lin_vel + torch.cross(body_ang_vel, center_offset_w, dim=-1)
        return racket_pos_w, racket_vel_w

    def _predict_ball_obs_features(
        self,
        *,
        ball_pos_w: torch.Tensor,
        ball_vel_w: torch.Tensor,
        racket_pos_w: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ) -> torch.Tensor:
        horizon = float(self.ball_obs_prediction_horizon_s)
        horizon_t = torch.full(
            (self.num_envs, 1),
            horizon,
            device=self.device,
            dtype=torch.float32,
        )

        # Predicted hit point: closest approach to racket center with constant-velocity extrapolation.
        rel_ball_racket = ball_pos_w - racket_pos_w
        vel_norm_sq = ball_vel_w.square().sum(dim=-1, keepdim=True).clamp_min(1.0e-6)
        t_hit = (-(rel_ball_racket * ball_vel_w).sum(dim=-1, keepdim=True) / vel_norm_sq).clamp(0.0, horizon)
        pred_hit_pos_w = ball_pos_w + ball_vel_w * t_hit
        pred_hit_pos_b = quat_apply_inverse(root_quat_w, pred_hit_pos_w - root_pos_w)
        pred_hit_dist = (pred_hit_pos_w - racket_pos_w).norm(dim=-1, keepdim=True)
        pred_hit_t_norm = t_hit / horizon_t

        # Predicted first bounce with ballistic approximation.
        gravity_z = float(self._read_gravity_z_value(self.env.sim.model.opt.gravity))
        g = torch.full((self.num_envs, 1), gravity_z, device=self.device, dtype=torch.float32)
        a = 0.5 * g
        b = ball_vel_w[:, 2:3]
        c = ball_pos_w[:, 2:3] - self.ball_radius
        disc = (b.square() - 4.0 * a * c).clamp_min(0.0)
        sqrt_disc = torch.sqrt(disc)
        denom = (2.0 * a).clamp(min=-1.0e6, max=-1.0e-6)
        t1 = (-b - sqrt_disc) / denom
        t2 = (-b + sqrt_disc) / denom
        t_candidates = torch.cat([t1, t2], dim=-1)
        valid_candidates = t_candidates > 1.0e-4
        t_pos = torch.where(valid_candidates, t_candidates, torch.full_like(t_candidates, float("inf")))
        t_bounce = t_pos.min(dim=-1, keepdim=True).values
        valid_bounce = torch.isfinite(t_bounce)
        t_bounce = torch.where(valid_bounce, t_bounce, horizon_t).clamp(0.0, horizon)
        pred_bounce_xy_w = ball_pos_w[:, :2] + ball_vel_w[:, :2] * t_bounce
        pred_bounce_pos_w = torch.cat([pred_bounce_xy_w, torch.full_like(t_bounce, self.ball_radius)], dim=-1)
        pred_bounce_pos_b = quat_apply_inverse(root_quat_w, pred_bounce_pos_w - root_pos_w)
        pred_bounce_t_norm = t_bounce / horizon_t
        pred_bounce_valid = valid_bounce.float()

        return torch.cat(
            [
                pred_hit_pos_b,
                pred_hit_t_norm,
                pred_hit_dist,
                pred_bounce_pos_b,
                pred_bounce_t_norm,
                pred_bounce_valid,
            ],
            dim=-1,
        )

    def _capture_highlevel_action(self) -> None:
        td = getattr(self.env, "input_tensordict", None)
        if td is None:
            self.highlevel_action.zero_()
            self.correction_action.zero_()
            self.correction_action_rate.zero_()
            self.prev_correction_action.zero_()
            return

        keys = td.keys(True, True)
        if "highlevel_action" not in keys:
            self.highlevel_action.zero_()
            self.correction_action.zero_()
            self.correction_action_rate.zero_()
            self.prev_correction_action.zero_()
            return

        highlevel_action = td.get("highlevel_action").detach()
        if highlevel_action.shape[-1] != self.highlevel_action.shape[-1]:
            self.highlevel_action = torch.zeros(
                (self.num_envs, highlevel_action.shape[-1]), device=self.device, dtype=torch.float32
            )
        self.highlevel_action[:] = highlevel_action

        latent_dim = min(self.highlevel_latent_dim, int(highlevel_action.shape[-1]))
        correction = highlevel_action[:, latent_dim:]
        if correction.shape[-1] == 0:
            self.correction_action = torch.zeros((self.num_envs, 1), device=self.device, dtype=torch.float32)
            self.correction_action_rate = torch.zeros_like(self.correction_action)
            self.prev_correction_action = torch.zeros_like(self.correction_action)
            return
        if correction.shape[-1] != self.correction_action.shape[-1]:
            self.correction_action = torch.zeros(
                (self.num_envs, correction.shape[-1]), device=self.device, dtype=torch.float32
            )
            self.prev_correction_action = torch.zeros_like(self.correction_action)
            self.correction_action_rate = torch.zeros_like(self.correction_action)
        self.correction_action_rate[:] = correction - self.prev_correction_action
        self.correction_action[:] = correction
        self.prev_correction_action[:] = correction

    def _compute_launch_spin(self, vel: torch.Tensor, spin_rps: torch.Tensor) -> torch.Tensor:
        gravity_dir = self.gravity_dir.unsqueeze(0).expand(vel.shape[0], -1)
        spin_axis = torch.cross(vel, gravity_dir, dim=-1)
        spin_axis_norm = spin_axis.norm(dim=-1, keepdim=True)
        fallback_axis = torch.tensor([1.0, 0.0, 0.0], device=self.device, dtype=torch.float32).view(1, 3)
        spin_axis = torch.where(spin_axis_norm > 1e-6, spin_axis / spin_axis_norm.clamp_min(1e-6), fallback_axis)
        return spin_rps.unsqueeze(-1) * (2.0 * math.pi) * spin_axis

    def _solve_ballistic_velocity(
        self,
        launch_pos: torch.Tensor,
        strike_pos: torch.Tensor,
        flight_t: torch.Tensor,
        gravity_z: torch.Tensor,
    ) -> torch.Tensor:
        vel = torch.zeros_like(launch_pos)
        vel[:, :2] = (strike_pos[:, :2] - launch_pos[:, :2]) / flight_t
        vel[:, 2:3] = (
            strike_pos[:, 2:3]
            - launch_pos[:, 2:3]
            - 0.5 * gravity_z * flight_t.square()
        ) / flight_t
        return vel

    def _solve_velocity_to_bounce(
        self,
        launch_pos: torch.Tensor,
        bounce_xy: torch.Tensor,
        bounce_t: torch.Tensor,
        gravity_z: torch.Tensor,
    ) -> torch.Tensor:
        vel = torch.zeros((launch_pos.shape[0], 3), device=self.device, dtype=torch.float32)
        vel[:, 0] = (bounce_xy[:, 0] - launch_pos[:, 0]) / bounce_t
        vel[:, 1] = (bounce_xy[:, 1] - launch_pos[:, 1]) / bounce_t
        vel[:, 2] = (
            self.ball_radius
            - launch_pos[:, 2]
            - 0.5 * gravity_z.squeeze(-1) * bounce_t.square()
        ) / bounce_t
        return vel

    def _clamp_launch_speed(self, vel: torch.Tensor) -> torch.Tensor:
        speed = vel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale_low = (self.launch_speed_range[0] / speed).clamp_min(1.0)
        scale_high = (self.launch_speed_range[1] / speed).clamp_max(1.0)
        scale = torch.where(speed < self.launch_speed_range[0], scale_low, scale_high)
        return vel * scale

    def _clamp_launch_speed_max_only(self, vel: torch.Tensor) -> torch.Tensor:
        speed = vel.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        scale_high = (self.launch_speed_range[1] / speed).clamp_max(1.0)
        return vel * scale_high

    def _launch_quality_mask(
        self,
        launch_pos: torch.Tensor,
        vel: torch.Tensor,
        pred_pos: torch.Tensor,
        net_cross_z: torch.Tensor,
        gravity_z: torch.Tensor,
    ) -> torch.Tensor:
        speed = vel.norm(dim=-1)
        horiz_speed = vel[:, :2].norm(dim=-1).clamp_min(1e-6)
        launch_angle_deg = torch.atan2(vel[:, 2], horiz_speed) * (180.0 / math.pi)
        forward_speed = -vel[:, 1]

        valid = (speed >= self.launch_speed_range[0]) & (speed <= self.launch_speed_range[1])
        valid &= pred_pos[:, 2] > (self.ball_radius + 0.02)
        valid &= forward_speed >= self.launch_min_forward_speed
        valid &= vel[:, 2] >= self.launch_min_vz
        valid &= launch_angle_deg >= self.launch_angle_deg_range[0]
        valid &= launch_angle_deg <= self.launch_angle_deg_range[1]
        if self.enforce_launch_net_clearance:
            valid &= net_cross_z > (self.net_height + self.launch_net_clearance_margin)
        if self.enforce_launch_incoming_bounce_in:
            bounce_xy, bounce_t = self._predict_first_bounce_ballistic(launch_pos, vel, gravity_z)
            valid &= bounce_t > 0.08
            valid &= bounce_xy[:, 0] >= self.incoming_bounce_x_range[0]
            valid &= bounce_xy[:, 0] <= self.incoming_bounce_x_range[1]
            valid &= bounce_xy[:, 1] >= self.incoming_bounce_y_range[0]
            valid &= bounce_xy[:, 1] <= self.incoming_bounce_y_range[1]
        return valid

    def _predict_first_bounce_ballistic(
        self,
        launch_pos: torch.Tensor,
        vel: torch.Tensor,
        gravity_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        g = gravity_z.squeeze(-1)
        g = torch.where(g.abs() > 1.0e-6, g, torch.full_like(g, -9.81))
        z0 = launch_pos[:, 2]
        vz = vel[:, 2]
        c = z0 - self.ball_radius
        disc = (vz.square() - 2.0 * g * c).clamp_min(1.0e-6)
        sqrt_disc = torch.sqrt(disc)
        t1 = (-vz - sqrt_disc) / g
        t2 = (-vz + sqrt_disc) / g
        bounce_t = torch.where(t1 > 1.0e-4, t1, t2).clamp_min(1.0e-4)
        bounce_xy = launch_pos[:, :2] + vel[:, :2] * bounce_t.unsqueeze(-1)
        return bounce_xy, bounce_t

    def _load_launch_bank(self, launch_bank_file: str) -> None:
        path = os.path.expanduser(str(launch_bank_file))
        if not os.path.isabs(path):
            path = os.path.abspath(path)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Launch bank file not found: {path}")

        with np.load(path) as data:
            keys = set(data.keys())

            def _read(*cands: str) -> np.ndarray | None:
                for cand in cands:
                    if cand in keys:
                        return np.asarray(data[cand], dtype=np.float32)
                return None

            pos_local_np = _read("launch_pos_local", "local_pos")
            vel_np = _read("launch_vel", "vel")
            ang_np = _read("launch_ang", "ang")
            target_local_np = _read("target_bounce_local", "local_tgt")

        if pos_local_np is None or vel_np is None or ang_np is None or target_local_np is None:
            raise ValueError(
                f"Invalid launch bank file {path}. "
                "Expected keys: launch_pos_local, launch_vel, launch_ang, target_bounce_local."
            )
        if (
            pos_local_np.ndim != 2
            or vel_np.ndim != 2
            or ang_np.ndim != 2
            or target_local_np.ndim != 2
            or pos_local_np.shape[1] != 3
            or vel_np.shape[1] != 3
            or ang_np.shape[1] != 3
            or target_local_np.shape[1] != 3
        ):
            raise ValueError(
                f"Invalid launch bank tensor shapes from {path}: "
                f"pos={pos_local_np.shape}, vel={vel_np.shape}, ang={ang_np.shape}, target={target_local_np.shape}"
            )
        n = int(pos_local_np.shape[0])
        if n <= 0 or vel_np.shape[0] != n or ang_np.shape[0] != n or target_local_np.shape[0] != n:
            raise ValueError(
                f"Inconsistent launch bank lengths from {path}: "
                f"pos={pos_local_np.shape[0]}, vel={vel_np.shape[0]}, ang={ang_np.shape[0]}, target={target_local_np.shape[0]}"
            )

        self._launch_bank_pos_local = torch.tensor(pos_local_np, dtype=torch.float32, device=self.device)
        self._launch_bank_vel = torch.tensor(vel_np, dtype=torch.float32, device=self.device)
        self._launch_bank_ang = torch.tensor(ang_np, dtype=torch.float32, device=self.device)
        self._launch_bank_target_local = torch.tensor(target_local_np, dtype=torch.float32, device=self.device)
        self._launch_bank_size = n
        self._launch_bank_ptr = 0
        self._launch_bank_perm = torch.randperm(n, device=self.device) if self.launch_bank_shuffle else None

    def _next_launch_bank_indices(self, num_samples: int) -> torch.Tensor:
        if self._launch_bank_size <= 0:
            raise RuntimeError("Launch bank is empty.")
        ids = torch.empty((num_samples,), dtype=torch.long, device=self.device)
        filled = 0
        while filled < num_samples:
            if self._launch_bank_ptr >= self._launch_bank_size:
                self._launch_bank_ptr = 0
                if self.launch_bank_shuffle:
                    self._launch_bank_perm = torch.randperm(self._launch_bank_size, device=self.device)
            take = min(self._launch_bank_size - self._launch_bank_ptr, num_samples - filled)
            src = slice(self._launch_bank_ptr, self._launch_bank_ptr + take)
            if self._launch_bank_perm is None:
                ids[filled : filled + take] = torch.arange(
                    self._launch_bank_ptr,
                    self._launch_bank_ptr + take,
                    dtype=torch.long,
                    device=self.device,
                )
            else:
                ids[filled : filled + take] = self._launch_bank_perm[src]
            self._launch_bank_ptr += take
            filled += take
        return ids

    def _improve_launch_clearance(
        self,
        launch_pos: torch.Tensor,
        vel: torch.Tensor,
        spin_rps: torch.Tensor,
        flight_t: torch.Tensor,
        gravity_z: torch.Tensor,
    ) -> torch.Tensor:
        if not self.enforce_launch_net_clearance or self.launch_clearance_correction_iters <= 0:
            return vel

        desired_net_z = self.net_height + self.launch_net_clearance_margin + 0.02
        for _ in range(self.launch_clearance_correction_iters):
            ang = self._compute_launch_spin(vel, spin_rps)
            _, _, net_cross_z = self._predict_ball_at_time(
                launch_pos, vel, ang, flight_t, gravity_z
            )
            low = net_cross_z <= desired_net_z
            if not low.any():
                break
            forward_speed = (-vel[low, 1]).clamp_min(1.0e-3)
            t_cross = (launch_pos[low, 1] / forward_speed).clamp_min(0.08)
            dz = (desired_net_z - net_cross_z[low]).clamp_min(0.0) + 0.04
            vel[low, 2] = vel[low, 2] + dz / t_cross
            vel[low] = self._clamp_launch_speed(vel[low])
        return vel

    def _predict_ball_at_time(
        self,
        launch_pos: torch.Tensor,
        launch_vel: torch.Tensor,
        launch_ang: torch.Tensor,
        flight_t: torch.Tensor,
        gravity_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        pos = launch_pos.clone()
        vel = launch_vel.clone()
        ang = launch_ang.clone()
        prev_pos = pos.clone()

        steps_needed = torch.ceil((flight_t.squeeze(-1) / self.launch_predict_dt).clamp_min(1.0)).to(torch.int64)
        max_steps = int(steps_needed.max().item())
        net_cross_z = torch.full((pos.shape[0],), -1.0e6, device=self.device, dtype=torch.float32)
        net_crossed = torch.zeros((pos.shape[0],), device=self.device, dtype=torch.bool)

        for i in range(max_steps):
            active = i < steps_needed
            if not active.any():
                break
            prev_pos[active] = pos[active]

            vel_a = vel[active]
            ang_a = ang[active]
            speed = vel_a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            spin_mag = ang_a.norm(dim=-1, keepdim=True)
            spin_scaled = spin_mag / (2.0 * math.pi) * self.lift_spin_scale
            vel_dir = vel_a / speed
            spin_axis = ang_a / spin_mag.clamp_min(1e-6)
            cl = 1.0 / (2.0 + torch.abs(speed / (spin_scaled + 1e-6)))
            drag_force = -self.aero_force_k * self.drag_coef * speed * vel_a
            lift_force = self.aero_force_k * cl * speed.square() * torch.cross(spin_axis, vel_dir, dim=-1)
            acc = (drag_force + lift_force) / self.ball_mass
            acc[:, 2:3] = acc[:, 2:3] + gravity_z[active]
            vel_a = vel_a + acc * self.launch_predict_dt
            pos_a = pos[active] + vel_a * self.launch_predict_dt

            prev_a = prev_pos[active]
            crossed = (~net_crossed[active]) & (prev_a[:, 1] > 0.0) & (pos_a[:, 1] <= 0.0)
            if crossed.any():
                alpha = prev_a[crossed, 1] / (prev_a[crossed, 1] - pos_a[crossed, 1] + 1e-6)
                z_cross = prev_a[crossed, 2] + alpha * (pos_a[crossed, 2] - prev_a[crossed, 2])
                active_ids = active.nonzero(as_tuple=False).squeeze(-1)
                crossed_ids = active_ids[crossed]
                net_cross_z[crossed_ids] = z_cross
                net_crossed[crossed_ids] = True

            vel[active] = vel_a
            pos[active] = pos_a
            spin_decay = max(0.85, 1.0 - self.spin_damping_coef * self.launch_predict_dt)
            ang[active] = ang_a * spin_decay

        return pos, vel, net_cross_z

    def _sample_ball_launch(
        self, env_ids: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        env_origins = self.env.scene.env_origins[env_ids]
        num_samples = env_ids.numel()
        if self._launch_bank_size > 0:
            bank_ids = self._next_launch_bank_indices(num_samples)
            return (
                self._launch_bank_pos_local[bank_ids] + env_origins,
                self._launch_bank_vel[bank_ids],
                self._launch_bank_ang[bank_ids],
                self._launch_bank_target_local[bank_ids] + env_origins,
            )

        gravity_z = self._get_current_gravity_z(env_ids)

        launch_pos_all = torch.zeros((num_samples, 3), device=self.device, dtype=torch.float32)
        launch_vel_all = torch.zeros((num_samples, 3), device=self.device, dtype=torch.float32)
        launch_ang_all = torch.zeros((num_samples, 3), device=self.device, dtype=torch.float32)
        target_all = torch.zeros((num_samples, 3), device=self.device, dtype=torch.float32)

        pending = torch.arange(num_samples, device=self.device, dtype=torch.long)
        for _ in range(self.launch_resample_attempts):
            if pending.numel() == 0:
                break
            n = pending.numel()
            launch_pos = torch.zeros((n, 3), device=self.device, dtype=torch.float32)
            launch_pos[:, 0] = self._sample_uniform_n(n, self.launcher_x_range)
            launch_pos[:, 1] = self._sample_uniform_n(n, self.launcher_y_range)
            launch_pos[:, 2] = self._sample_uniform_n(n, self.launcher_z_range)

            strike_pos = torch.zeros_like(launch_pos)
            strike_pos[:, 0] = self._sample_uniform_n(n, self.strike_x_range)
            strike_pos[:, 1] = self._sample_uniform_n(n, self.strike_y_range)
            strike_pos[:, 2] = self._sample_uniform_n(n, self.strike_z_range)

            flight_t = self._sample_uniform_n(n, self.flight_t_range).unsqueeze(-1)
            target_pos = torch.zeros_like(launch_pos)
            target_pos[:, 0] = self._sample_uniform_n(n, self.target_x_range)
            target_pos[:, 1] = self._sample_uniform_n(n, self.target_y_range)

            spin_rps = self._sample_uniform_n(n, self.launch_spin_rps_range)
            vel = self._solve_ballistic_velocity(
                launch_pos,
                strike_pos,
                flight_t,
                gravity_z[pending],
            )
            for _ in range(max(0, self.launch_solver_iters)):
                ang = self._compute_launch_spin(vel, spin_rps)
                pred_pos, _, _ = self._predict_ball_at_time(
                    launch_pos, vel, ang, flight_t, gravity_z[pending]
                )
                vel = vel + (strike_pos - pred_pos) / flight_t.clamp_min(1e-3)
                vel = self._clamp_launch_speed(vel)

            vel = self._improve_launch_clearance(
                launch_pos=launch_pos,
                vel=vel,
                spin_rps=spin_rps,
                flight_t=flight_t,
                gravity_z=gravity_z[pending],
            )

            ang = self._compute_launch_spin(vel, spin_rps)
            pred_pos, _, net_cross_z = self._predict_ball_at_time(
                launch_pos, vel, ang, flight_t, gravity_z[pending]
            )
            strike_err = (pred_pos - strike_pos).norm(dim=-1)
            valid = strike_err <= self.launch_strike_tolerance
            valid &= self._launch_quality_mask(launch_pos, vel, pred_pos, net_cross_z, gravity_z[pending])

            if valid.any():
                accepted = pending[valid]
                launch_pos_all[accepted] = launch_pos[valid]
                launch_vel_all[accepted] = vel[valid]
                launch_ang_all[accepted] = ang[valid]
                target_all[accepted] = target_pos[valid]
            pending = pending[~valid]

        if pending.numel() > 0:
            n = pending.numel()
            launch_pos = torch.zeros((n, 3), device=self.device, dtype=torch.float32)
            launch_pos[:, 0] = self._sample_uniform_n(n, self.launcher_x_range)
            launch_pos[:, 1] = self._sample_uniform_n(n, self.launcher_y_range)
            launch_pos[:, 2] = self._sample_uniform_n(n, self.launcher_z_range)
            strike_pos = torch.zeros_like(launch_pos)
            strike_pos[:, 0] = self._sample_uniform_n(n, self.strike_x_range)
            strike_pos[:, 1] = self._sample_uniform_n(n, self.strike_y_range)
            strike_z_low = max(float(self.strike_z_range[0].item()), self.net_height + 0.35)
            strike_z_high = max(strike_z_low + 0.20, float(self.strike_z_range[1].item()) + 0.35)
            strike_pos[:, 2] = self._sample_uniform_n(
                n, torch.tensor([strike_z_low, strike_z_high], device=self.device, dtype=torch.float32)
            )
            flight_t = self._sample_uniform_n(n, self.flight_t_range).unsqueeze(-1)
            target_pos = torch.zeros_like(launch_pos)
            target_pos[:, 0] = self._sample_uniform_n(n, self.target_x_range)
            target_pos[:, 1] = self._sample_uniform_n(n, self.target_y_range)
            spin_rps = self._sample_uniform_n(n, self.launch_spin_rps_range)

            vel = self._solve_ballistic_velocity(
                launch_pos,
                strike_pos,
                flight_t,
                gravity_z[pending],
            )
            for _ in range(max(1, self.launch_solver_iters)):
                ang = self._compute_launch_spin(vel, spin_rps)
                pred_pos, _, net_cross_z = self._predict_ball_at_time(
                    launch_pos, vel, ang, flight_t, gravity_z[pending]
                )
                vel = vel + (strike_pos - pred_pos) / flight_t.clamp_min(1e-3)
                vel = self._clamp_launch_speed(vel)
                if self.enforce_launch_net_clearance:
                    low = net_cross_z <= (self.net_height + self.launch_net_clearance_margin)
                    if low.any():
                        strike_pos[low, 2] = strike_pos[low, 2] + 0.15
                        vel[low] = self._solve_ballistic_velocity(
                            launch_pos[low],
                            strike_pos[low],
                            flight_t[low],
                            gravity_z[pending][low],
                        )
            vel = self._improve_launch_clearance(
                launch_pos=launch_pos,
                vel=vel,
                spin_rps=spin_rps,
                flight_t=flight_t,
                gravity_z=gravity_z[pending],
            )
            ang = self._compute_launch_spin(vel, spin_rps)
            pred_pos, _, net_cross_z = self._predict_ball_at_time(
                launch_pos, vel, ang, flight_t, gravity_z[pending]
            )
            good = self._launch_quality_mask(launch_pos, vel, pred_pos, net_cross_z, gravity_z[pending])
            if (~good).any():
                bad = ~good
                dir_xy = strike_pos[:, :2] - launch_pos[:, :2]
                dir_xy = dir_xy / dir_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
                emergency_speed = float(
                    min(
                        float(self.launch_speed_range[1].item()) - 0.5,
                        max(float(self.launch_speed_range[0].item()) + 2.0, 24.0),
                    )
                )
                emergency_angle_deg = float(
                    min(
                        float(self.launch_angle_deg_range[1].item()) - 0.5,
                        max(float(self.launch_angle_deg_range[0].item()) + 2.0, 10.0),
                    )
                )
                emergency_angle = math.radians(emergency_angle_deg)
                horiz_speed = emergency_speed * math.cos(emergency_angle)
                vz = emergency_speed * math.sin(emergency_angle)
                vel[bad, :2] = dir_xy[bad] * horiz_speed
                vel[bad, 2] = vz
                vel = self._clamp_launch_speed(vel)
                vel = self._improve_launch_clearance(
                    launch_pos=launch_pos,
                    vel=vel,
                    spin_rps=spin_rps,
                    flight_t=flight_t,
                    gravity_z=gravity_z[pending],
                )
                ang = self._compute_launch_spin(vel, spin_rps)
                pred_pos, _, net_cross_z = self._predict_ball_at_time(
                    launch_pos, vel, ang, flight_t, gravity_z[pending]
                )
                still_bad = ~self._launch_quality_mask(
                    launch_pos, vel, pred_pos, net_cross_z, gravity_z[pending]
                )
                if still_bad.any():
                    n_bad = int(still_bad.sum().item())
                    bounce_xy = torch.zeros((n_bad, 2), device=self.device, dtype=torch.float32)
                    bounce_xy[:, 0] = self._sample_uniform_n(n_bad, self.incoming_bounce_x_range)
                    bounce_xy[:, 1] = self._sample_uniform_n(n_bad, self.incoming_bounce_y_range)
                    bounce_t = self._sample_uniform_n(
                        n_bad,
                        torch.tensor([1.05, 1.35], device=self.device, dtype=torch.float32),
                    )
                    vel_safe = self._solve_velocity_to_bounce(
                        launch_pos=launch_pos[still_bad],
                        bounce_xy=bounce_xy,
                        bounce_t=bounce_t,
                        gravity_z=gravity_z[pending][still_bad],
                    )
                    vel_safe[:, 1] = torch.minimum(
                        vel_safe[:, 1],
                        torch.full_like(vel_safe[:, 1], -self.launch_min_forward_speed),
                    )
                    vel[still_bad] = vel_safe
                    ang[still_bad] = self._compute_launch_spin(vel_safe, spin_rps[still_bad])

            launch_pos_all[pending] = launch_pos
            launch_vel_all[pending] = vel
            launch_ang_all[pending] = ang
            target_all[pending] = target_pos

        if self.enforce_launch_incoming_bounce_in:
            bounce_xy, bounce_t = self._predict_first_bounce_ballistic(launch_pos_all, launch_vel_all, gravity_z)
            bad = bounce_t <= 0.08
            bad |= bounce_xy[:, 0] < self.incoming_bounce_x_range[0]
            bad |= bounce_xy[:, 0] > self.incoming_bounce_x_range[1]
            bad |= bounce_xy[:, 1] < self.incoming_bounce_y_range[0]
            bad |= bounce_xy[:, 1] > self.incoming_bounce_y_range[1]
            bad |= (-launch_vel_all[:, 1]) < self.launch_min_forward_speed
            if bad.any():
                n_bad = int(bad.sum().item())
                bounce_xy_fix = torch.zeros((n_bad, 2), device=self.device, dtype=torch.float32)
                bounce_xy_fix[:, 0] = self._sample_uniform_n(n_bad, self.incoming_bounce_x_range)
                bounce_xy_fix[:, 1] = self._sample_uniform_n(n_bad, self.incoming_bounce_y_range)
                bounce_t_fix = self._sample_uniform_n(
                    n_bad,
                    torch.tensor([1.05, 1.35], device=self.device, dtype=torch.float32),
                )
                vel_fix = self._solve_velocity_to_bounce(
                    launch_pos=launch_pos_all[bad],
                    bounce_xy=bounce_xy_fix,
                    bounce_t=bounce_t_fix,
                    gravity_z=gravity_z[bad],
                )
                vel_fix[:, 1] = torch.minimum(
                    vel_fix[:, 1],
                    torch.full_like(vel_fix[:, 1], -self.launch_min_forward_speed),
                )
                spin_rps_fix = launch_ang_all[bad].norm(dim=-1) / (2.0 * math.pi)
                launch_vel_all[bad] = vel_fix
                launch_ang_all[bad] = self._compute_launch_spin(vel_fix, spin_rps_fix)

        return (
            launch_pos_all + env_origins,
            launch_vel_all,
            launch_ang_all,
            target_all + env_origins,
        )

    def sample_init(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return None

        root_state = self.init_root_state[env_ids].clone()
        spawn_pos_w, spawn_quat_w = self._sample_robot_spawn(env_ids)
        root_state[:, :3] = spawn_pos_w
        root_state[:, 3:7] = spawn_quat_w
        root_state[:, 7:13] = 0.0

        self.asset.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.asset.write_joint_position_to_sim(self.root_default_joint_pos[env_ids], env_ids=env_ids)
        self.asset.write_joint_velocity_to_sim(self.root_default_joint_vel[env_ids], env_ids=env_ids)
        self.asset.set_joint_position_target(self.root_default_joint_pos[env_ids], env_ids=env_ids)
        self._write_ball_launch(env_ids)
        self._prev_racket_vel_w[env_ids] = 0.0
        self.prev_correction_action[env_ids] = 0.0
        self.correction_action[env_ids] = 0.0
        self.correction_action_rate[env_ids] = 0.0
        self.highlevel_action[env_ids] = 0.0

        return None

    def reset(self, env_ids: torch.Tensor):
        self._reset_rally_state(env_ids, reset_hit_counter=True)
        self._prime_ball_obs_history(env_ids)

    def step(self, substep: int):
        if substep == 0:
            self._ensure_action_layout()
            self._capture_highlevel_action()

        ball_pos_w = self.ball.data.root_link_pos_w
        ball_vel_w = self.ball.data.root_link_lin_vel_w
        ball_ang_w = self.ball.data.root_link_ang_vel_w
        env_origins = self.env.scene.env_origins
        ball_pos_l = ball_pos_w - env_origins
        self._update_contact_events()

        racket_pos_w, racket_vel_w = self._racket_state_w()
        racket_acc = (racket_vel_w - self._prev_racket_vel_w) / float(self.env.physics_dt)
        self.racket_acc_norm[:] = racket_acc.square().sum(dim=-1, keepdim=True)
        self._prev_racket_vel_w[:] = racket_vel_w

        hit_gate = (
            (~self.has_hit)
            & (~self.fail_miss)
            & (~self.fail_net)
            & (~self.fail_out)
            & (self.hit_cooldown <= 0)
        )
        hit_mask = hit_gate & (self.racket_ball_contact_event | self.racket_ball_contact)
        self._last_hit_mask = hit_mask

        if hit_mask.any():
            racket_speed = racket_vel_w[hit_mask].norm(dim=-1)
            self.has_hit[hit_mask] = True
            self.hit_event[hit_mask] = True
            self.hit_cooldown[hit_mask] = 12
            self.hit_racket_speed[hit_mask, 0] = racket_speed
            first_hit = hit_mask & (self.first_hit_step >= self.max_task_steps)
            if first_hit.any():
                self.first_hit_step[first_hit] = self.task_step[first_hit] + 1

            style_bad = hit_mask & (
                (racket_vel_w[:, 1] < self.stroke_style_min_forward_speed)
                | (racket_vel_w.norm(dim=-1) < self.stroke_style_min_racket_speed)
            )
            if style_bad.any():
                self.fail_style[style_bad] = True
                self.stroke_style_violation_event[style_bad] = True

        bounce_contact_mask = self.ball_court_contact_event | (
            self.ball_court_contact & (ball_vel_w[:, 2] > 0.0)
        )
        first_bounce = bounce_contact_mask & self.has_hit & (~self.has_bounce)
        if first_bounce.any():
            self.has_bounce[first_bounce] = True
            self.bounce_event[first_bounce] = True
            first_bounce_step = first_bounce & (self.first_bounce_step >= self.max_task_steps)
            if first_bounce_step.any():
                self.first_bounce_step[first_bounce_step] = self.task_step[first_bounce_step] + 1
            self.bounce_pos_w[first_bounce] = ball_pos_w[first_bounce]
            bounce_pos_l = self.bounce_pos_w[first_bounce] - env_origins[first_bounce]
            x_in = bounce_pos_l[:, 0].abs() <= self.court_x_limit
            y_in = (
                (bounce_pos_l[:, 1] >= self.court_y_min_success)
                & (bounce_pos_l[:, 1] <= self.court_y_limit)
            )
            self.bounce_in[first_bounce] = x_in & y_in

        net_mask = (
            self.has_hit
            & (~self.has_bounce)
            & (ball_pos_l[:, 1].abs() <= self.net_half_thickness)
            & (ball_pos_w[:, 2] <= self.net_height + 0.05)
            & (ball_vel_w[:, 1] > 0.0)
        )
        net_mask = net_mask | self.ball_net_contact_event
        if net_mask.any():
            self.fail_net[net_mask] = True

        speed = ball_vel_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        spin_mag = ball_ang_w.norm(dim=-1, keepdim=True)
        spin_scaled = spin_mag / (2.0 * math.pi) * self.lift_spin_scale
        vel_dir = ball_vel_w / speed
        spin_axis = ball_ang_w / spin_mag.clamp_min(1e-6)
        cl = 1.0 / (2.0 + torch.abs(speed / (spin_scaled + 1e-6)))
        drag_force = -self.aero_force_k * self.drag_coef * speed * ball_vel_w
        lift_dir = torch.cross(spin_axis, vel_dir, dim=-1)
        lift_force = self.aero_force_k * cl * speed.square() * lift_dir
        total_force = drag_force + lift_force
        spin_damping_torque = -self.spin_damping_coef * ball_ang_w
        total_force[self.finished] = 0.0
        spin_damping_torque[self.finished] = 0.0
        self.ball.write_external_wrench_to_sim(
            forces=total_force.unsqueeze(1),
            torques=spin_damping_torque.unsqueeze(1),
            body_ids=self.ball_body_ids,
        )
        pass_net_now = (
            self.has_hit
            & (~self.has_bounce)
            & (~self.has_pass_net)
            & (ball_vel_w[:, 1] > 0.0)
            & (ball_pos_l[:, 1] >= 0.0)
        )
        if pass_net_now.any():
            self.has_pass_net[pass_net_now] = True
            self.pass_net_event[pass_net_now] = True
            clear = ball_pos_w[pass_net_now, 2] > (self.net_height + self.net_clearance_reward_margin)
            if clear.any():
                pass_ids = pass_net_now.nonzero(as_tuple=False).squeeze(-1)
                self.net_clearance_event[pass_ids[clear]] = True
        self.hit_cooldown.sub_(1).clamp_min_(0)

    def before_update(self):
        self.task_step.add_(1)

        ball_pos_w = self.ball.data.root_link_pos_w
        ball_vel_w = self.ball.data.root_link_lin_vel_w
        root_pos_w = self.asset.data.root_link_pos_w
        ball_pos_l = ball_pos_w - self.env.scene.env_origins
        self.ground_recover_event[:] = False

        if self.ground_recover_enable:
            ball_speed = ball_vel_w.norm(dim=-1)
            within_guard = (
                (ball_pos_l[:, 0].abs() <= self.court_x_limit + 2.0)
                & (ball_pos_l[:, 1].abs() <= self.court_y_limit + 4.0)
            )
            recover_mask = (
                within_guard
                & (ball_pos_l[:, 2] < self.ground_recover_trigger_z)
                & (ball_speed <= self.ground_recover_speed_thres)
            )
            if recover_mask.any():
                recover_ids = recover_mask.nonzero(as_tuple=False).squeeze(-1)
                recover_vel = ball_vel_w[recover_ids]
                recover_ang = self.ball.data.root_link_ang_vel_w[recover_ids]
                bounce_vel = recover_vel.clone()
                bounce_vel[:, :2] = bounce_vel[:, :2] * self.ground_recover_horizontal_damping
                bounce_vz = (-recover_vel[:, 2]).clamp_min(0.0) * self.ground_recover_bounce_restitution
                bounce_vz = torch.maximum(
                    bounce_vz,
                    torch.full_like(bounce_vz, self.ground_recover_min_upward_speed),
                )
                bounce_vel[:, 2] = bounce_vz
                bounce_speed = bounce_vel.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
                clamp_scale = (self.ground_recover_max_speed / bounce_speed).clamp(max=1.0)
                bounce_vel = bounce_vel * clamp_scale
                ball_state = torch.zeros((recover_ids.numel(), 13), device=self.device, dtype=torch.float32)
                ball_state[:, :3] = ball_pos_w[recover_ids]
                ball_state[:, 2] = self.env.scene.env_origins[recover_ids, 2] + (
                    self.ground_recover_surface_z + self.ball_radius
                )
                ball_state[:, 3:7] = self.ball.data.root_link_quat_w[recover_ids]
                ball_state[:, 7:10] = bounce_vel
                ball_state[:, 10:13] = recover_ang * self.ground_recover_spin_damping
                self.ball.write_root_state_to_sim(ball_state, env_ids=recover_ids)
                self.ground_recover_event[recover_ids] = True
                ball_pos_w = self.ball.data.root_link_pos_w
                ball_vel_w = self.ball.data.root_link_lin_vel_w
                ball_pos_l = ball_pos_w - self.env.scene.env_origins

        racket_pos_w, _ = self._racket_state_w()
        contact_pos_w, contact_t, contact_valid, _, _ = self._incoming_contact_target(ball_pos_w, ball_vel_w)
        zone_xy_err = (racket_pos_w[:, :2] - contact_pos_w[:, :2]).norm(dim=-1)
        zone_z_err = (racket_pos_w[:, 2] - contact_pos_w[:, 2]).abs()
        zone_now = (
            (~self.has_hit)
            & (~self.fail_miss)
            & (~self.fail_net)
            & (~self.fail_out)
            & contact_valid
            & (contact_t >= self.hitting_zone_time_min)
            & (contact_t <= self.hitting_zone_time_max)
            & (zone_xy_err <= self.hitting_zone_xy_radius)
            & (zone_z_err <= self.hitting_zone_z_tol)
        )
        self.prehit_zone_event[:] = zone_now & (~self.prehit_zone) & (~self.prehit_zone_entered)
        self.prehit_zone_entered |= self.prehit_zone_event
        self.prehit_zone[:] = zone_now

        self.ball_target_dist[:] = (ball_pos_w[:, :2] - self.target_bounce_w[:, :2]).norm(dim=-1, keepdim=True)
        self.ball_target_progress_buf[:] = (self.prev_ball_target_dist - self.ball_target_dist).clamp(-1.0, 1.0)
        self.net_dist[:] = ball_pos_l[:, 1:2].abs()
        self.net_dist_progress_buf[:] = (self.prev_net_dist - self.net_dist).clamp(-1.0, 1.0)

        ball_speed = ball_vel_w.norm(dim=-1)
        ball_dropped = self.ball_court_contact | (
            ball_pos_l[:, 2] <= (self.ball_radius + self.pre_hit_dead_ball_height_margin)
        )
        pre_hit_dead_ball = (
            (~self.has_hit)
            & (self.task_step >= self.pre_hit_dead_ball_min_steps)
            & (ball_speed <= self.pre_hit_dead_ball_speed_thres)
            & ball_dropped
        )
        self.pre_hit_dead_ball_steps[pre_hit_dead_ball] += 1
        self.pre_hit_dead_ball_steps[~pre_hit_dead_ball] = 0
        self.fail_miss |= self.pre_hit_dead_ball_steps >= self.pre_hit_dead_ball_patience_steps

        post_hit_ball_dropped = self.ball_court_contact | (
            ball_pos_l[:, 2] <= (self.ball_radius + self.post_hit_dead_ball_height_margin)
        )
        hit_elapsed_steps = (self.task_step - self.first_hit_step).clamp_min(0)
        post_hit_dead_ball = (
            self.has_hit
            & (hit_elapsed_steps >= self.post_hit_dead_ball_min_steps)
            & (ball_speed <= self.post_hit_dead_ball_speed_thres)
            & post_hit_ball_dropped
        )
        self.post_hit_dead_ball_steps[post_hit_dead_ball] += 1
        self.post_hit_dead_ball_steps[~post_hit_dead_ball] = 0
        post_hit_dead_ball_ready = self.post_hit_dead_ball_steps >= self.post_hit_dead_ball_patience_steps

        ball_behind_root = ball_pos_w[:, 1] < (root_pos_w[:, 1] - self.miss_margin_y)
        self.fail_miss |= (~self.has_hit) & ball_behind_root & ball_dropped
        fail_out_candidate = (
            (ball_pos_l[:, 2] < self.out_margin_z)
            | (ball_pos_l[:, 0].abs() > self.court_x_limit + 2.0)
            | (ball_pos_l[:, 1].abs() > self.court_y_limit + 4.0)
        )
        self.success[:] = self.has_bounce & self.bounce_in
        self.fail_out |= (~self.success) & fail_out_candidate

        success_out_candidate = (
            (ball_pos_l[:, 2] < self.out_margin_z)
            | (ball_pos_l[:, 0].abs() > self.court_x_limit + self.success_exit_x_margin)
            | (ball_pos_l[:, 1].abs() > self.court_y_limit + self.success_exit_y_margin)
        )
        if self.success_requires_ball_exit:
            success_done = self.success & (success_out_candidate | post_hit_dead_ball_ready)
        else:
            success_done = self.success
        self.fail_miss |= self.has_hit & (~self.success) & post_hit_dead_ball_ready

        new_success_event = success_done & (~self.success_done)
        if new_success_event.any():
            self.consecutive_return_count[new_success_event] += 1

        timeout = self.task_step >= self.max_task_steps
        fail_any = self.fail_miss | self.fail_net | self.fail_out | self.fail_style
        self.hit_limit_reached[:] = False
        if self.max_consecutive_returns_before_finish > 0:
            self.hit_limit_reached[:] = (
                self.consecutive_return_count >= self.max_consecutive_returns_before_finish
            )

        if self.relaunch_on_success:
            new_relaunch = success_done & (~self.hit_limit_reached)
            newly_set = new_relaunch & (~self._rally_relaunch_mask)
            self._rally_relaunch_mask[:] = self._rally_relaunch_mask | new_relaunch
            if newly_set.any():
                self._rally_launch_delay[newly_set] = self.launch_interval_steps
            waiting = self._rally_relaunch_mask & (self._rally_launch_delay > 0)
            if waiting.any():
                self._rally_launch_delay[waiting] -= 1
            self._rally_launch_ready[:] = self._rally_relaunch_mask & (self._rally_launch_delay <= 0)
            success_finish = success_done & self.hit_limit_reached
        else:
            self._rally_relaunch_mask[:] = False
            self._rally_launch_ready[:] = False
            success_finish = success_done

        self.success_done[:] = success_done
        self.timeout[:] = timeout
        self.finished[:] = self.hit_limit_reached | success_finish | fail_any | timeout
        self._update_live_debug_metrics(
            ball_pos_w,
            ball_vel_w,
            hit_mask=self._last_hit_mask,
        )

    def _update_live_debug_metrics(
        self,
        ball_pos_w: torch.Tensor,
        ball_vel_w: torch.Tensor,
        *,
        hit_mask: torch.Tensor | None = None,
    ) -> None:
        if not hasattr(self.env, "extra"):
            return
        ball_pos_l = ball_pos_w - self.env.scene.env_origins
        ball_speed = ball_vel_w.norm(dim=-1)
        done_mask = self.finished.float()
        self.env.extra["highlevel/live_active_ratio"] = float((1.0 - done_mask).mean().item())
        self.env.extra["highlevel/live_done_ratio"] = float(done_mask.mean().item())
        self.env.extra["highlevel/live_has_hit_ratio"] = float(self.has_hit.float().mean().item())
        self.env.extra["highlevel/live_has_bounce_ratio"] = float(self.has_bounce.float().mean().item())
        self.env.extra["highlevel/live_pass_net_ratio"] = float(self.has_pass_net.float().mean().item())
        self.env.extra["highlevel/live_success_ratio"] = float(self.success.float().mean().item())
        self.env.extra["highlevel/live_success_done_ratio"] = float(self.success_done.float().mean().item())
        self.env.extra["highlevel/live_timeout_ratio"] = float(self.timeout.float().mean().item())
        self.env.extra["highlevel/live_fail_miss_ratio"] = float(self.fail_miss.float().mean().item())
        self.env.extra["highlevel/live_fail_net_ratio"] = float(self.fail_net.float().mean().item())
        self.env.extra["highlevel/live_fail_out_ratio"] = float(self.fail_out.float().mean().item())
        self.env.extra["highlevel/live_fail_style_ratio"] = float(self.fail_style.float().mean().item())
        self.env.extra["highlevel/live_hit_limit_reached_ratio"] = float(self.hit_limit_reached.float().mean().item())
        self.env.extra["highlevel/live_consecutive_return_count_mean"] = float(
            self.consecutive_return_count.float().mean().item()
        )
        self.env.extra["highlevel/live_task_step_norm_mean"] = float(
            (self.task_step.float() / float(self.max_task_steps)).mean().item()
        )
        self.env.extra["highlevel/live_ball_height_l_mean"] = float(ball_pos_l[:, 2].mean().item())
        self.env.extra["highlevel/live_ball_speed_mean"] = float(ball_speed.mean().item())
        self.env.extra["highlevel/live_pre_hit_dead_ball_ratio"] = float(
            (self.pre_hit_dead_ball_steps >= self.pre_hit_dead_ball_patience_steps).float().mean().item()
        )
        self.env.extra["highlevel/live_pre_hit_dead_ball_steps_mean"] = float(
            self.pre_hit_dead_ball_steps.float().mean().item()
        )
        self.env.extra["highlevel/live_post_hit_dead_ball_ratio"] = float(
            (self.post_hit_dead_ball_steps >= self.post_hit_dead_ball_patience_steps).float().mean().item()
        )
        self.env.extra["highlevel/live_post_hit_dead_ball_steps_mean"] = float(
            self.post_hit_dead_ball_steps.float().mean().item()
        )
        self.env.extra["highlevel/live_ground_recover_ratio"] = float(
            self.ground_recover_event.float().mean().item()
        )
        self.env.extra["highlevel/live_prehit_zone_ratio"] = float(self.prehit_zone.float().mean().item())
        self.env.extra["highlevel/live_prehit_zone_event_ratio"] = float(
            self.prehit_zone_event.float().mean().item()
        )
        self.env.extra["highlevel/live_ball_target_dist_mean"] = float(self.ball_target_dist.mean().item())
        self.env.extra["highlevel/live_approach_oracle_mix"] = float(self.approach_oracle_mix)
        self.env.extra["highlevel/live_highlevel_action_l2_mean"] = float(
            self.highlevel_action.square().mean().item() if self.highlevel_action.numel() > 0 else 0.0
        )
        self.env.extra["highlevel/live_correction_action_l2_mean"] = float(
            self.correction_action.square().mean().item() if self.correction_action.numel() > 0 else 0.0
        )
        self.env.extra["highlevel/live_racket_ball_contact_ratio"] = float(
            self.racket_ball_contact.float().mean().item()
        )
        self.env.extra["highlevel/live_ball_net_contact_ratio"] = float(
            self.ball_net_contact.float().mean().item()
        )
        self.env.extra["highlevel/live_ball_court_contact_ratio"] = float(
            self.ball_court_contact.float().mean().item()
        )
        if self.debug_draw_enabled and hit_mask is not None:
            self.env.extra["highlevel/live_hit_trigger_ratio"] = float(hit_mask.float().mean().item())

    def update(self):
        self._push_ball_obs_history()
        self.prev_ball_target_dist[:] = self.ball_target_dist
        self.prev_net_dist[:] = self.net_dist
        relaunch_ids = self._rally_launch_ready.nonzero(as_tuple=False).squeeze(-1)
        if relaunch_ids.numel() > 0:
            self._reset_rally_state(relaunch_ids, reset_hit_counter=False)
            self._write_ball_launch(relaunch_ids)
            self._rally_relaunch_mask[relaunch_ids] = False
            self._rally_launch_ready[relaunch_ids] = False
            self._rally_launch_delay[relaunch_ids] = 0
        self.hit_event[:] = False
        self.bounce_event[:] = False
        self.pass_net_event[:] = False
        self.net_clearance_event[:] = False
        self.stroke_style_violation_event[:] = False
        self.prehit_zone_event[:] = False
        self.racket_ball_contact_event[:] = False
        self.ball_net_contact_event[:] = False
        self.ball_court_contact_event[:] = False
        self.ground_recover_event[:] = False
        self.hit_racket_speed[:] = 0.0

    @observation
    def ball_task_obs(self):
        root_pos_w = self.asset.data.root_link_pos_w
        root_quat_w = self.asset.data.root_link_quat_w
        root_vel_w = self.asset.data.root_link_lin_vel_w
        env_origins = self.env.scene.env_origins
        ball_pos_w = self.ball.data.root_link_pos_w
        ball_vel_w = self.ball.data.root_link_lin_vel_w

        racket_pos_w, racket_vel_w = self._racket_state_w()
        ball_pred_obs = self._predict_ball_obs_features(
            ball_pos_w=ball_pos_w,
            ball_vel_w=ball_vel_w,
            racket_pos_w=racket_pos_w,
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
        )

        root_pos_l = root_pos_w - env_origins
        root_forward_w = quat_apply(root_quat_w, self._forward_dir_b)
        target_bounce_l = self.target_bounce_w - env_origins

        racket_pos_b = quat_apply_inverse(root_quat_w, racket_pos_w - root_pos_w)
        racket_vel_b = quat_apply_inverse(root_quat_w, racket_vel_w - root_vel_w)
        target_bounce_b = quat_apply_inverse(root_quat_w, self.target_bounce_w - root_pos_w)
        hist_ids = list(self.ball_obs_history_steps)
        ball_pos_hist_w = self.ball_pos_w_history[:, hist_ids]
        ball_vel_hist_w = self.ball_vel_w_history[:, hist_ids]
        h = len(hist_ids)
        root_quat_rep = root_quat_w.unsqueeze(1).expand(-1, h, -1).reshape(-1, 4)
        root_pos_rep = root_pos_w.unsqueeze(1).expand(-1, h, -1).reshape(-1, 3)
        root_vel_rep = root_vel_w.unsqueeze(1).expand(-1, h, -1).reshape(-1, 3)
        ball_pos_hist_b = quat_apply_inverse(
            root_quat_rep,
            (ball_pos_hist_w.reshape(-1, 3) - root_pos_rep),
        ).reshape(self.num_envs, -1)
        ball_vel_hist_b = quat_apply_inverse(
            root_quat_rep,
            (ball_vel_hist_w.reshape(-1, 3) - root_vel_rep),
        ).reshape(self.num_envs, -1)

        flags = torch.stack(
            [
                self.has_hit.float(),
                self.has_bounce.float(),
                self.bounce_in.float(),
                self.fail_miss.float(),
                self.fail_net.float(),
                self.fail_out.float(),
                self.fail_style.float(),
                self.success.float(),
                self.task_step.float() / float(self.max_task_steps),
            ],
            dim=-1,
        )
        return torch.cat(
            [
                root_pos_l[:, :2],
                root_forward_w[:, :2],
                ball_pos_hist_b,
                ball_vel_hist_b,
                target_bounce_l,
                racket_pos_b,
                racket_vel_b,
                target_bounce_b,
                ball_pred_obs,
                flags,
            ],
            dim=-1,
        )

    def ball_task_obs_sym(self):
        dim = 35 + 6 * len(self.ball_obs_history_steps)
        return sym_utils.SymmetryTransform(
            perm=torch.arange(dim, device=self.device),
            signs=[1.0] * dim,
        )

    @observation
    def ball_priv_obs(self):
        env_origins = self.env.scene.env_origins
        ball_pos_w = self.ball.data.root_link_pos_w - env_origins
        ball_vel_w = self.ball.data.root_link_lin_vel_w
        root_pos_w = self.asset.data.root_link_pos_w - env_origins
        root_quat_w = self.asset.data.root_link_quat_w
        root_forward_w = quat_apply(root_quat_w, self._forward_dir_b)
        racket_pos_w, racket_vel_w = self._racket_state_w()
        racket_pos_w = racket_pos_w - env_origins
        target_bounce_w = self.target_bounce_w - env_origins
        return torch.cat(
            [
                ball_pos_w,
                ball_vel_w,
                racket_pos_w,
                racket_vel_w,
                target_bounce_w,
                root_pos_w[:, :2],
                root_forward_w[:, :2],
            ],
            dim=-1,
        )

    def ball_priv_obs_sym(self):
        dim = 19
        return sym_utils.SymmetryTransform(
            perm=torch.arange(dim, device=self.device),
            signs=[1.0] * dim,
        )

    def _incoming_bounce_target_xy(
        self,
        ball_pos_w: torch.Tensor,
        ball_vel_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return mixed incoming-bounce target (oracle/predicted) and predicted bounce time."""
        pred_bounce_xy, pred_bounce_t = self._predict_first_bounce_ballistic(
            launch_pos=ball_pos_w,
            vel=ball_vel_w,
            gravity_z=self.gravity,
        )
        mix = float(min(max(self.approach_oracle_mix, 0.0), 1.0))
        target_bounce_xy = mix * self.oracle_incoming_bounce_xy_w + (1.0 - mix) * pred_bounce_xy
        return target_bounce_xy, pred_bounce_t

    def _incoming_contact_target(
        self,
        ball_pos_w: torch.Tensor,
        ball_vel_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict a pre-hit contact target from incoming bounce target and time-to-bounce."""
        target_bounce_xy, pred_bounce_t = self._incoming_bounce_target_xy(ball_pos_w, ball_vel_w)
        contact_t = (
            pred_bounce_t - float(self.approach_contact_lead_time)
        ).clamp(float(self.approach_contact_min_t), float(self.approach_contact_max_t))
        contact_valid = pred_bounce_t > float(self.approach_contact_min_t + 1.0e-3)

        ratio = (contact_t / pred_bounce_t.clamp_min(float(self.approach_contact_min_t))).unsqueeze(-1).clamp(0.0, 1.0)
        contact_xy = ball_pos_w[:, :2] + (target_bounce_xy - ball_pos_w[:, :2]) * ratio

        gravity_z = self.gravity.squeeze(-1)
        contact_z = (
            ball_pos_w[:, 2]
            + ball_vel_w[:, 2] * contact_t
            + 0.5 * gravity_z * contact_t.square()
        ).clamp_min(self.ball_radius + 0.02)
        contact_pos_w = torch.cat([contact_xy, contact_z.unsqueeze(-1)], dim=-1)
        return contact_pos_w, contact_t, contact_valid, target_bounce_xy, pred_bounce_t

    def _root_stance_target_xy(
        self,
        *,
        target_bounce_xy: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
        racket_pos_w: torch.Tensor,
        lateral_stance_offset: float,
    ) -> torch.Tensor:
        """Shift root target to the opposite side of the racket for handedness-aware pre-positioning."""
        offset = float(lateral_stance_offset)
        if abs(offset) <= 1.0e-6:
            return target_bounce_xy

        side_vec_xy = racket_pos_w[:, :2] - root_pos_w[:, :2]
        side_norm = side_vec_xy.norm(dim=-1, keepdim=True)
        side_dir_xy = side_vec_xy / side_norm.clamp_min(1.0e-6)

        # Fallback for degenerate side vector: use body-left direction in world frame.
        left_dir_w = quat_apply(root_quat_w, self._left_dir_b)
        left_dir_xy = left_dir_w[:, :2] / left_dir_w[:, :2].norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        side_dir_xy = torch.where(side_norm > 1.0e-4, side_dir_xy, left_dir_xy)

        # For right-handed hitting, root should stand on the opposite side of racket to expose hitting arm.
        return target_bounce_xy - offset * side_dir_xy

    @reward
    def approach_ball(
        self,
        sigma: Sequence[float] | None = (0.25, 0.55),
        root_sigma: Sequence[float] | None = (0.45, 0.95),
        min_bounce_t: float = 0.03,
        max_bounce_t: float = 1.80,
        rear_margin_y: float = 0.8,
        early_preposition_weight: float = 0.6,
        root_preposition_blend: float = 0.85,
        lateral_stance_offset: float = 0.30,
        z_weight: float = 0.6,
        z_sigma: Sequence[float] | None = (0.08, 0.18),
        z_activate_net_y: float = -0.05,
        z_activate_xy_dist: float = 0.55,
        z_activate_contact_t: float = 0.45,
    ):
        # Pre-position the racket at the final incoming bounce location (XY only).
        # During training we anneal from launch-oracle bounce to online predicted bounce.
        ball_pos_w = self.ball.data.root_link_pos_w
        ball_vel_w = self.ball.data.root_link_lin_vel_w
        root_pos_w = self.asset.data.root_link_pos_w
        root_quat_w = self.asset.data.root_link_quat_w
        racket_pos_w, _ = self._racket_state_w()
        ball_pos_l = ball_pos_w - self.env.scene.env_origins

        contact_pos_w, contact_t, contact_valid, _, _ = self._incoming_contact_target(ball_pos_w, ball_vel_w)
        root_target_xy = self._root_stance_target_xy(
            target_bounce_xy=contact_pos_w[:, :2],
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            racket_pos_w=racket_pos_w,
            lateral_stance_offset=lateral_stance_offset,
        )

        # Two-stage XY shaping:
        # - early phase (large time-to-bounce): prioritize root pre-positioning;
        # - late phase: prioritize racket XY interception.
        early_scale = (contact_t / max(float(max_bounce_t), 1.0e-3)).clamp(0.0, 1.0).unsqueeze(-1)
        preposition_alpha = (float(root_preposition_blend) * early_scale).clamp(0.0, 1.0)
        racket_xy_error = (racket_pos_w[:, :2] - contact_pos_w[:, :2]).norm(dim=-1, keepdim=True)
        root_xy_error = (root_pos_w[:, :2] - root_target_xy).norm(dim=-1, keepdim=True)
        racket_xy_rew = _exp_reward(racket_xy_error, sigma)
        root_xy_rew = _exp_reward(root_xy_error, root_sigma)
        rew = (1.0 - preposition_alpha) * racket_xy_rew + preposition_alpha * root_xy_rew
        rew = rew * (1.0 + float(early_preposition_weight) * early_scale)

        # Stage-2 shaping:
        # once the ball is on robot side of net, or XY is close enough, encourage racket height alignment.
        racket_ball_xy_dist = (racket_pos_w[:, :2] - ball_pos_w[:, :2]).norm(dim=-1, keepdim=True)
        z_active = (
            (ball_pos_l[:, 1] <= float(z_activate_net_y))
            | (racket_ball_xy_dist.squeeze(-1) <= float(z_activate_xy_dist))
            | (contact_t <= float(z_activate_contact_t))
        ).float().unsqueeze(-1)
        z_error = (racket_pos_w[:, 2:3] - contact_pos_w[:, 2:3]).abs()
        z_rew = _exp_reward(z_error, z_sigma)
        rew = rew + float(z_weight) * z_rew * z_active

        active = (
            (~self.has_hit)
            & (~self.fail_miss)
            & (~self.fail_net)
            & (~self.fail_out)
            & contact_valid
            & (contact_t >= float(min_bounce_t))
            & (contact_t <= float(max_bounce_t))
            & (ball_pos_w[:, 1] > (root_pos_w[:, 1] - float(rear_margin_y)))
        ).float().unsqueeze(-1)
        return rew * active

    @reward
    def root_preposition_xy(
        self,
        sigma: Sequence[float] | None = (0.55, 1.10),
        min_bounce_t: float = 0.03,
        max_bounce_t: float = 1.80,
        rear_margin_y: float = 0.8,
        early_preposition_weight: float = 0.5,
        lateral_stance_offset: float = 0.30,
    ):
        ball_pos_w = self.ball.data.root_link_pos_w
        ball_vel_w = self.ball.data.root_link_lin_vel_w
        root_pos_w = self.asset.data.root_link_pos_w
        root_quat_w = self.asset.data.root_link_quat_w
        racket_pos_w, _ = self._racket_state_w()

        contact_pos_w, contact_t, contact_valid, _, _ = self._incoming_contact_target(ball_pos_w, ball_vel_w)
        root_target_xy = self._root_stance_target_xy(
            target_bounce_xy=contact_pos_w[:, :2],
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            racket_pos_w=racket_pos_w,
            lateral_stance_offset=lateral_stance_offset,
        )
        root_xy_error = (root_pos_w[:, :2] - root_target_xy).norm(dim=-1, keepdim=True)
        rew = _exp_reward(root_xy_error, sigma)

        early_scale = (contact_t / max(float(max_bounce_t), 1.0e-3)).clamp(0.0, 1.0).unsqueeze(-1)
        rew = rew * (1.0 + float(early_preposition_weight) * early_scale)

        active = (
            (~self.has_hit)
            & (~self.fail_miss)
            & (~self.fail_net)
            & (~self.fail_out)
            & contact_valid
            & (contact_t >= float(min_bounce_t))
            & (contact_t <= float(max_bounce_t))
            & (ball_pos_w[:, 1] > (root_pos_w[:, 1] - float(rear_margin_y)))
        ).float().unsqueeze(-1)
        return rew * active

    @reward
    def root_towards_ball_speed(
        self,
        target_speed: float = 1.0,
        distance_norm: float = 1.8,
        min_bounce_t: float = 0.03,
        max_bounce_t: float = 1.80,
        rear_margin_y: float = 0.8,
        lateral_stance_offset: float = 0.30,
    ):
        ball_pos_w = self.ball.data.root_link_pos_w
        ball_vel_w = self.ball.data.root_link_lin_vel_w
        root_pos_w = self.asset.data.root_link_pos_w
        root_quat_w = self.asset.data.root_link_quat_w
        racket_pos_w, _ = self._racket_state_w()

        contact_pos_w, contact_t, contact_valid, _, _ = self._incoming_contact_target(ball_pos_w, ball_vel_w)
        root_target_xy = self._root_stance_target_xy(
            target_bounce_xy=contact_pos_w[:, :2],
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            racket_pos_w=racket_pos_w,
            lateral_stance_offset=lateral_stance_offset,
        )

        root_delta_xy = root_target_xy - root_pos_w[:, :2]
        root_dist_xy = root_delta_xy.norm(dim=-1, keepdim=True)
        move_dir_xy = root_delta_xy / root_dist_xy.clamp_min(1.0e-6)
        root_vel_xy = self.asset.data.root_link_lin_vel_w[:, :2]
        speed_towards = (root_vel_xy * move_dir_xy).sum(dim=-1, keepdim=True)

        dist_norm = max(float(distance_norm), 1.0e-6)
        desired_speed = float(target_speed) * (root_dist_xy / dist_norm).clamp(0.0, 1.0)
        speed_error = (speed_towards - desired_speed).abs()
        rew = torch.exp(-speed_error / 0.4) * torch.sigmoid((speed_towards + 0.05) / 0.25)

        active = (
            (~self.has_hit)
            & (~self.fail_miss)
            & (~self.fail_net)
            & (~self.fail_out)
            & contact_valid
            & (contact_t >= float(min_bounce_t))
            & (contact_t <= float(max_bounce_t))
            & (ball_pos_w[:, 1] > (root_pos_w[:, 1] - float(rear_margin_y)))
        ).float().unsqueeze(-1)
        return rew * active


    @reward
    def racket_alignment(
        self,
        dist_threshold: float = 0.8,
        activate_contact_t: float = 0.45,
        height_sigma: float = 0.14,
    ):
        """Dense shaping: reward aligning the racket face towards the opponent court when ball is nearby."""
        ball_pos_w = self.ball.data.root_link_pos_w
        racket_pos_w, _ = self._racket_state_w()
        dist = (ball_pos_w - racket_pos_w).norm(dim=-1, keepdim=True)
        proximity = (1.0 - dist / float(dist_threshold)).clamp_min(0.0)
        _, contact_t, contact_valid, _, _ = self._incoming_contact_target(ball_pos_w, self.ball.data.root_link_lin_vel_w)
        
        # Get racket normal direction (assuming Z axis of racket body is the face normal)
        body_quat_w = self.asset.data.body_link_quat_w[:, self.racket_body_id]
        racket_normal = quat_apply(body_quat_w, torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1))
        
        # Desired direction uses XY only (court plane). This avoids encouraging downward "shovel" orientation
        # caused by aiming directly at the ground bounce point in 3D.
        target_dir_xy = self.target_bounce_w[:, :2] - racket_pos_w[:, :2]
        target_dir_xy = target_dir_xy / target_dir_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        racket_normal_xy = racket_normal[:, :2]
        racket_normal_xy = racket_normal_xy / racket_normal_xy.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        alignment_xy = (racket_normal_xy * target_dir_xy).sum(dim=-1, keepdim=True).abs()

        # Keep racket center around ball height when applying orientation reward.
        h_sigma = max(float(height_sigma), 1.0e-6)
        dz = (racket_pos_w[:, 2:3] - ball_pos_w[:, 2:3]).abs()
        height_gate = torch.exp(-0.5 * (dz / h_sigma).square())
        alignment = alignment_xy * height_gate
        
        active = ((~self.has_hit) & contact_valid & (contact_t <= float(activate_contact_t))).float().unsqueeze(-1)
        return proximity * alignment * active

    @reward
    def racket_face_vertical_on_hit(self, normal_z_sigma: float = 0.22):
        """At hit moment, prefer racket face normal to stay horizontal (face plane close to vertical)."""
        body_quat_w = self.asset.data.body_link_quat_w[:, self.racket_body_id]
        racket_normal = quat_apply(
            body_quat_w,
            torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(self.num_envs, -1),
        )
        sigma = max(float(normal_z_sigma), 1.0e-6)
        vertical_score = torch.exp(-0.5 * (racket_normal[:, 2:3] / sigma).square())
        return vertical_score * self.hit_event.float().unsqueeze(-1)

    @reward
    def racket_horizontal_swing_on_hit(
        self,
        min_horizontal_speed: float = 2.8,
        horizontal_speed_scale: float = 1.0,
        vz_sigma: float = 1.1,
    ):
        """At hit moment, encourage mostly-horizontal racket velocity (avoid up/down shovel swings)."""
        _, racket_vel_w = self._racket_state_w()
        horizontal_speed = racket_vel_w[:, :2].norm(dim=-1, keepdim=True)
        horiz_term = torch.sigmoid(
            (horizontal_speed - float(min_horizontal_speed)) / max(float(horizontal_speed_scale), 1.0e-6)
        )
        vertical_term = torch.exp(
            -0.5 * (racket_vel_w[:, 2:3] / max(float(vz_sigma), 1.0e-6)).square()
        )
        return horiz_term * vertical_term * self.hit_event.float().unsqueeze(-1)

    @reward
    def hit_success(self):
        return self.hit_event.float().unsqueeze(-1)

    @reward
    def hit_contact_height(
        self,
        target_height: float = 1.05,
        sigma: float = 0.18,
        min_height: float = 0.78,
    ):
        ball_z = self.ball.data.root_link_pos_w[:, 2:3]
        sigma = max(float(sigma), 1.0e-6)
        # Prefer medium contact height and suppress very low contacts that often cause squat/fall behaviors.
        h_err = (ball_z - float(target_height)) / sigma
        band = torch.exp(-0.5 * h_err.square())
        low_gate = torch.sigmoid((ball_z - float(min_height)) / 0.05)
        return band * low_gate * self.hit_event.float().unsqueeze(-1)

    @reward
    def enter_hitting_zone(self):
        return self.prehit_zone_event.float().unsqueeze(-1)

    @reward
    def ball_target_progress(self):
        active = (self.has_hit & (~self.has_bounce)).float().unsqueeze(-1)
        return self.ball_target_progress_buf * active

    @reward
    def net_progress(self):
        active = (
            self.has_hit
            & (~self.has_pass_net)
            & (~self.has_bounce)
            & (~self.fail_net)
            & (~self.fail_out)
        ).float().unsqueeze(-1)
        return self.net_dist_progress_buf * active

    @reward
    def net_height_margin_dense(
        self,
        net_window: float = 0.8,
        height_scale: float = 0.18,
        target_clearance_over_net: float = 0.20,
    ):
        ball_pos_w = self.ball.data.root_link_pos_w
        ball_vel_w = self.ball.data.root_link_lin_vel_w
        ball_pos_l = ball_pos_w - self.env.scene.env_origins
        near_net = ball_pos_l[:, 1].abs() <= float(net_window)
        clearance = ball_pos_w[:, 2:3] - self.net_height
        target_clearance = float(target_clearance_over_net)
        scale = max(float(height_scale), 1.0e-6)
        # Encourage a moderate clearance band instead of monotonically rewarding higher arcs.
        clearance_err = (clearance - target_clearance) / scale
        rew = torch.exp(-0.5 * clearance_err.square())
        # Suppress reward if ball is below or barely above net.
        rew = rew * torch.sigmoid((clearance - 0.01) / 0.03)
        active = (
            self.has_hit
            & (~self.has_pass_net)
            & (~self.has_bounce)
            & (~self.fail_net)
            & (~self.fail_out)
            & (ball_vel_w[:, 1] > 0.0)
            & near_net
        ).float().unsqueeze(-1)
        return rew * active

    @reward
    def forward_velocity_soft(
        self,
        vy_center: float = 1.0,
        vy_scale: float = 2.0,
        vz_center: float = 0.35,
        vz_scale: float = 1.2,
    ):
        ball_vel_w = self.ball.data.root_link_lin_vel_w
        vy_term = torch.sigmoid((ball_vel_w[:, 1:2] - float(vy_center)) / max(float(vy_scale), 1.0e-6))
        vz_scale = max(float(vz_scale), 1.0e-6)
        vz_err = (ball_vel_w[:, 2:3] - float(vz_center)) / vz_scale
        # Prefer a moderate upward velocity, discouraging very high lobs.
        vz_term = torch.exp(-0.5 * vz_err.square())
        rew = vy_term * vz_term
        active = (
            self.has_hit
            & (~self.has_pass_net)
            & (~self.has_bounce)
            & (~self.fail_net)
            & (~self.fail_out)
        ).float().unsqueeze(-1)
        return rew * active

    @reward
    def outgoing_angle_band(
        self,
        target_angle_deg: float = 12.0,
        angle_sigma_deg: float = 6.0,
        min_forward_speed: float = 2.0,
    ):
        ball_vel_w = self.ball.data.root_link_lin_vel_w
        vy = ball_vel_w[:, 1]
        vz = ball_vel_w[:, 2]
        angle = torch.atan2(vz, vy.clamp_min(1.0e-4))
        target = math.radians(float(target_angle_deg))
        sigma = max(math.radians(float(angle_sigma_deg)), 1.0e-4)
        angle_err = (angle - target) / sigma
        rew = torch.exp(-0.5 * angle_err.square()).unsqueeze(-1)
        forward_gate = torch.sigmoid((vy - float(min_forward_speed)) / 0.8).unsqueeze(-1)
        # Only score this at contact time to avoid trajectory-phase exploits.
        return rew * forward_gate * self.hit_event.float().unsqueeze(-1)

    @reward
    def predicted_bounce_target(
        self,
        bounce_pos_scale: float = 0.20,
        bounce_time_scale: float = 0.35,
        max_bounce_time: float = 1.5,
        require_predicted_in: bool = True,
    ):
        # Dense post-hit shaping: estimate first bounce from current ball state,
        # then reward target-directed outgoing trajectories before true bounce happens.
        ball_pos_w = self.ball.data.root_link_pos_w
        ball_vel_w = self.ball.data.root_link_lin_vel_w

        pred_bounce_xy, pred_bounce_t = self._predict_first_bounce_ballistic(
            launch_pos=ball_pos_w,
            vel=ball_vel_w,
            gravity_z=self.gravity,
        )
        pred_bounce_t = pred_bounce_t.clamp(0.0, float(max_bounce_time))

        target_xy = self.target_bounce_w[:, :2]
        pos_err = (pred_bounce_xy - target_xy).square().sum(dim=-1, keepdim=True)
        rew = torch.exp(-float(bounce_pos_scale) * pos_err) * torch.exp(
            -float(bounce_time_scale) * pred_bounce_t.unsqueeze(-1)
        )

        if require_predicted_in:
            x_in = pred_bounce_xy[:, 0].abs() <= self.court_x_limit
            y_in = (
                (pred_bounce_xy[:, 1] >= self.court_y_min_success)
                & (pred_bounce_xy[:, 1] <= self.court_y_limit)
            )
            rew = rew * (x_in & y_in).float().unsqueeze(-1)

        active = (
            self.has_hit
            & (~self.has_bounce)
            & (~self.fail_net)
            & (~self.fail_out)
            & (ball_vel_w[:, 1] > 0.0)
        ).float().unsqueeze(-1)
        return rew * active

    @reward
    def bounce_target(self, sigma: Sequence[float] | None = (0.50, 1.0)):
        err = (self.bounce_pos_w - self.target_bounce_w).norm(dim=-1, keepdim=True)
        rew = _exp_reward(err, sigma)
        return rew * self.bounce_event.float().unsqueeze(-1) * self.bounce_in.float().unsqueeze(-1)

    @reward
    def pass_net(self):
        return self.pass_net_event.float().unsqueeze(-1)

    @reward
    def bounce_in_event(self):
        return (self.bounce_event & self.bounce_in).float().unsqueeze(-1)

    @reward
    def bounce_wrong_side_penalty(self):
        wrong_side = self.bounce_event & self.has_hit & (~self.bounce_in)
        return wrong_side.float().unsqueeze(-1)

    @reward
    def net_clearance(self):
        return self.net_clearance_event.float().unsqueeze(-1)

    @reward
    def ball_velocity_constraint(self):
        ball_vel_w = self.ball.data.root_link_lin_vel_w
        speed = ball_vel_w.norm(dim=-1)
        speed_ok = (speed >= self.outgoing_speed_minmax[0]) & (speed <= self.outgoing_speed_minmax[1])
        dir_ok = (ball_vel_w[:, 1] > 0.0) & (ball_vel_w[:, 2] > 0.0)
        valid = self.hit_event & speed_ok & dir_ok
        return valid.float().unsqueeze(-1)

    @reward
    def racket_speed_on_hit(self):
        return self.hit_racket_speed

    @reward
    def racket_velocity_constraint(self, min_racket_speed: float = 4.0):
        valid = self.hit_event & (self.hit_racket_speed[:, 0] >= float(min_racket_speed))
        return valid.float().unsqueeze(-1)

    @reward
    def highlevel_action_l2(self):
        if self.highlevel_action.numel() == 0:
            return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        return self.highlevel_action.square().mean(dim=-1, keepdim=True)

    @reward
    def correction_action_l2(self):
        if self.correction_action.numel() == 0:
            return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        return self.correction_action.square().mean(dim=-1, keepdim=True)

    @reward
    def correction_action_rate_l2(self):
        if self.correction_action_rate.numel() == 0:
            return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        return self.correction_action_rate.square().mean(dim=-1, keepdim=True)

    @reward
    def lower_body_action_rate_l2(self):
        self._ensure_action_layout()
        if self.lower_body_action_ids.numel() == 0:
            return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        action_buf = self.env.action_manager.action_buf
        action_diff = action_buf[:, 0, self.lower_body_action_ids] - action_buf[:, 1, self.lower_body_action_ids]
        return action_diff.square().mean(dim=-1, keepdim=True)

    @reward
    def whole_body_action_rate_l2(self):
        action_buf = self.env.action_manager.action_buf
        action_diff = action_buf[:, 0, :] - action_buf[:, 1, :]
        return action_diff.square().mean(dim=-1, keepdim=True)

    @reward
    def joint_vel_l2_mean(self):
        joint_vel = self.asset.data.joint_vel
        return joint_vel.square().mean(dim=-1, keepdim=True)

    @reward
    def joint_pos_limits_l1_mean(self, soft_factor: float = 0.9):
        soft_factor = min(max(float(soft_factor), 0.0), 0.999)
        jpos_limits = self.asset.data.joint_pos_limits
        jpos = self.asset.data.joint_pos
        jpos_mean = (jpos_limits[..., 0] + jpos_limits[..., 1]) * 0.5
        jpos_range = (jpos_limits[..., 1] - jpos_limits[..., 0]).clamp_min(1.0e-6)
        lower = jpos_mean - 0.5 * jpos_range * soft_factor
        upper = jpos_mean + 0.5 * jpos_range * soft_factor
        violation = (lower - jpos).clamp_min(0.0) + (jpos - upper).clamp_min(0.0)
        return violation.mean(dim=-1, keepdim=True) / max(1.0 - soft_factor, 1.0e-6)

    @reward
    def joint_vel_limits_l1_mean(self, soft_factor: float = 0.9):
        soft_factor = min(max(float(soft_factor), 0.0), 0.999)
        jvel = self.asset.data.joint_vel
        vel_limits = getattr(self.asset.data, "soft_joint_vel_limits", None)
        if vel_limits is None:
            vel_limits = getattr(self.asset.data, "joint_vel_limits", None)
        if vel_limits is None:
            return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)

        if vel_limits.ndim == jvel.ndim + 1 and vel_limits.shape[-1] == 2:
            vel_limit_abs = torch.maximum(vel_limits[..., 0].abs(), vel_limits[..., 1].abs())
        else:
            vel_limit_abs = vel_limits.abs()
            if vel_limit_abs.shape != jvel.shape:
                vel_limit_abs = vel_limit_abs.expand_as(jvel)

        soft_upper = vel_limit_abs * soft_factor
        violation = (jvel.abs() - soft_upper).clamp_min(0.0)
        denom = (vel_limit_abs * max(1.0 - soft_factor, 1.0e-6)).clamp_min(1.0e-6)
        return (violation / denom).mean(dim=-1, keepdim=True)

    @reward
    def racket_acc_l2(self):
        return self.racket_acc_norm

    @reward
    def wrist_torque_l2(self):
        self._ensure_action_layout()
        if self.wrist_actuator_ids.numel() == 0:
            return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        wrist_torque = self.asset.data.actuator_force[:, self.wrist_actuator_ids]
        return wrist_torque.square().sum(dim=-1, keepdim=True)

    @reward
    def wrist_joint_smoothness_l2(self):
        if self.wrist_joint_ids_asset.numel() == 0:
            return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        wrist_acc = self.asset.data.joint_acc[:, self.wrist_joint_ids_asset]
        return wrist_acc.square().sum(dim=-1, keepdim=True)

    @reward
    def pelvis_facing_forward(self):
        root_quat_w = self.asset.data.root_link_quat_w
        root_forward_w = quat_apply(root_quat_w, self._forward_dir_b)
        # Court forward direction is +Y.
        return root_forward_w[:, 1:2].clamp_min(0.0)

    @reward
    def episode_success(self):
        return (self.finished & self.success_done).float().unsqueeze(-1)

    @reward
    def episode_timeout(self):
        return (self.finished & self.timeout).float().unsqueeze(-1)

    @reward
    def episode_fail_miss(self):
        return (self.finished & self.fail_miss).float().unsqueeze(-1)

    @reward
    def episode_fail_net(self):
        return (self.finished & self.fail_net).float().unsqueeze(-1)

    @reward
    def episode_fail_out(self):
        return (self.finished & self.fail_out).float().unsqueeze(-1)

    @reward
    def episode_fall(self, xy_thres: float = 0.85, z_thres: float = 0.45):
        fall = (
            self.asset.data.projected_gravity_b[:, :2].norm(dim=1, keepdim=True) >= float(xy_thres)
        ) | (-self.asset.data.projected_gravity_b[:, 2:] < float(z_thres))
        return (self.finished.unsqueeze(-1) & fall).float()

    @reward
    def episode_stroke_style_violation(self):
        return (self.finished & self.fail_style).float().unsqueeze(-1)

    @reward
    def episode_has_hit(self):
        return (self.finished & self.has_hit).float().unsqueeze(-1)

    @reward
    def episode_pass_net(self):
        return (self.finished & self.has_pass_net).float().unsqueeze(-1)

    @reward
    def episode_has_bounce(self):
        return (self.finished & self.has_bounce).float().unsqueeze(-1)

    @reward
    def episode_bounce_in(self):
        return (self.finished & self.bounce_in).float().unsqueeze(-1)

    @reward
    def episode_hit_step_norm(self):
        hit_step = (self.first_hit_step.float() / float(self.max_task_steps)).unsqueeze(-1)
        valid = (self.finished & self.has_hit & (self.first_hit_step < self.max_task_steps)).float().unsqueeze(-1)
        return hit_step * valid

    @reward
    def episode_bounce_step_norm(self):
        bounce_step = (self.first_bounce_step.float() / float(self.max_task_steps)).unsqueeze(-1)
        valid = (self.finished & self.has_bounce & (self.first_bounce_step < self.max_task_steps)).float().unsqueeze(-1)
        return bounce_step * valid

    @reward
    def episode_done_step_norm(self):
        done_step = self.task_step.float() / float(self.max_task_steps)
        return done_step.unsqueeze(-1) * self.finished.float().unsqueeze(-1)

    @reward
    def episode_target_dist_at_done(self):
        return self.ball_target_dist * self.finished.float().unsqueeze(-1)

    @reward
    def episode_ball_speed_at_done(self):
        ball_speed = self.ball.data.root_link_lin_vel_w.norm(dim=-1, keepdim=True)
        return ball_speed * self.finished.float().unsqueeze(-1)

    @termination
    def miss_ball_termination(self):
        return self.fail_miss.unsqueeze(-1)

    @termination
    def net_hit_termination(self):
        return self.fail_net.unsqueeze(-1)

    @termination
    def ball_out_of_bounds_termination(self):
        return self.fail_out.unsqueeze(-1)

    @termination
    def stroke_style_violation_termination(self):
        return self.fail_style.unsqueeze(-1)

    def debug_draw(self):
        if not self.debug_draw_enabled:
            return
        ball_pos_w = self.ball.data.root_link_pos_w
        self.env.debug_draw.point(ball_pos_w, color=(1.0, 1.0, 0.1, 1.0), size=18.0)
        self.env.debug_draw.point(self.target_bounce_w, color=(0.1, 0.8, 1.0, 1.0), size=10.0)
        racket_pos_w, _ = self._racket_state_w()
        self.env.debug_draw.point(racket_pos_w, color=(1.0, 0.5, 0.1, 1.0), size=10.0)
