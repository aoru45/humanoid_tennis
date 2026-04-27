from __future__ import annotations

import math

import torch
from humanoid_tennis.utils.math import quat_apply, quat_apply_inverse


class HighLevelTennisStateMixin:
    def step_schedule(self, progress: float, iters: int | None = None):
        _ = progress
        _ = iters

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

    def _set_stroke_mode_target_from_launch(
        self,
        *,
        env_ids: torch.Tensor,
        target_bounce_w: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ) -> None:
        if env_ids.numel() == 0:
            return
        target_bounce_b = quat_apply_inverse(root_quat_w, target_bounce_w - root_pos_w)
        contact_lateral = target_bounce_b[:, 1]
        deadzone = float(self.stroke_mode_lateral_deadzone)

        target = torch.full(
            (env_ids.numel(),),
            self.STROKE_MODE_NEUTRAL,
            dtype=torch.long,
            device=self.device,
        )
        forehand_mask = contact_lateral <= -deadzone
        backhand_mask = contact_lateral >= deadzone
        target[forehand_mask] = self.STROKE_MODE_FOREHAND
        target[backhand_mask] = self.STROKE_MODE_BACKHAND

        self.stroke_mode_target[env_ids] = target
        self.stroke_mode_contact_lateral[env_ids, 0] = contact_lateral

    def _write_ball_launch(
        self,
        env_ids: torch.Tensor,
        *,
        root_pos_w: torch.Tensor | None = None,
        root_quat_w: torch.Tensor | None = None,
    ) -> None:
        if env_ids.numel() == 0:
            return
        launch_pos_w, launch_vel_w, launch_ang_w, target_bounce_w, sampled_level_ids = self._sample_ball_launch(env_ids)
        ball_state = torch.zeros((env_ids.numel(), 13), device=self.device, dtype=torch.float32)
        ball_state[:, :3] = launch_pos_w
        ball_state[:, 3] = 1.0
        ball_state[:, 7:10] = launch_vel_w
        ball_state[:, 10:13] = launch_ang_w
        self.ball.write_root_state_to_sim(ball_state, env_ids=env_ids)
        self.target_bounce_w[env_ids] = target_bounce_w
        self.launch_level_ids[env_ids] = sampled_level_ids
        self.ball_pos_w_history[env_ids] = launch_pos_w.unsqueeze(1)
        self.ball_vel_w_history[env_ids] = launch_vel_w.unsqueeze(1)
        launch_pos_l = launch_pos_w - self.env.scene.env_origins[env_ids]
        launch_net_dist = launch_pos_l[:, 1:2].abs()
        self.prev_net_dist[env_ids] = launch_net_dist
        self.net_dist[env_ids] = launch_net_dist
        self.net_dist_progress_buf[env_ids] = 0.0
        if root_pos_w is None:
            root_pos_w = self.asset.data.root_link_pos_w[env_ids]
        if root_quat_w is None:
            root_quat_w = self.asset.data.root_link_quat_w[env_ids]
        self._set_stroke_mode_target_from_launch(
            env_ids=env_ids,
            target_bounce_w=target_bounce_w,
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
        )

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
        self.fail_racket_body[env_ids] = False
        self.first_hit_step[env_ids] = self.max_task_steps
        self.first_bounce_step[env_ids] = self.max_task_steps
        self.hit_event[env_ids] = False
        self.bounce_event[env_ids] = False
        self.pass_net_event[env_ids] = False
        self.net_clearance_event[env_ids] = False
        self.hit_stroke_mode_match_event[env_ids] = False
        self.hit_stroke_mode_mismatch_event[env_ids] = False
        self.stroke_style_violation_event[env_ids] = False
        self.prehit_zone[env_ids] = False
        self.prehit_zone_event[env_ids] = False
        self.prehit_zone_entered[env_ids] = False
        self.racket_ball_contact[env_ids] = False
        self.ball_net_contact[env_ids] = False
        self.ball_court_contact[env_ids] = False
        self.racket_body_contact[env_ids] = False
        self.racket_ball_contact_event[env_ids] = False
        self.ball_net_contact_event[env_ids] = False
        self.ball_court_contact_event[env_ids] = False
        self.racket_body_contact_event[env_ids] = False
        self.hit_cooldown[env_ids] = 0
        self.pre_hit_dead_ball_steps[env_ids] = 0
        self.post_hit_dead_ball_steps[env_ids] = 0
        self.hit_racket_speed[env_ids] = 0.0
        if self._last_hit_mask is not None:
            self._last_hit_mask[env_ids] = False
        self.launch_level_ids[env_ids] = -1
        self.bounce_pos_w[env_ids] = 0.0
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
        self.spawn_root_pos_w[env_ids] = spawn_pos_w
        self.spawn_root_quat_w[env_ids] = spawn_quat_w
        spawn_forward_w = quat_apply(
            spawn_quat_w,
            self._forward_dir_b[env_ids],
        )
        self.spawn_root_forward_xy[env_ids] = spawn_forward_w[:, :2]
        self._write_ball_launch(env_ids, root_pos_w=spawn_pos_w, root_quat_w=spawn_quat_w)
        self._prev_racket_vel_w[env_ids] = 0.0
        self.prev_correction_action[env_ids] = 0.0
        self.correction_action[env_ids] = 0.0
        self.correction_action_rate[env_ids] = 0.0
        self.highlevel_action[env_ids] = 0.0

        return None

    def reset(self, env_ids: torch.Tensor):
        self._reset_rally_state(env_ids, reset_hit_counter=True)
        self._prime_ball_obs_history(env_ids)
