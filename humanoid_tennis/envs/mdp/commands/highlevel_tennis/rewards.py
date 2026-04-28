from __future__ import annotations

import math
from typing import Sequence

import torch

from humanoid_tennis.envs.mdp import reward
from humanoid_tennis.utils.math import quat_apply, quat_apply_inverse


def _exp_reward(error: torch.Tensor, sigma: Sequence[float] | None = None) -> torch.Tensor:
    if sigma is None or len(sigma) == 0:
        sigma = (0.2,)
    rewards = [torch.exp(-error / float(s)) for s in sigma]
    return sum(rewards) / float(len(rewards))


class HighLevelTennisRewardMixin:
    def _post_hit_window_mask(self, window_steps: int) -> torch.Tensor:
        steps_since_hit = (self.task_step - self.first_hit_step).clamp_min(0)
        return (
            self.has_hit
            & (self.first_hit_step < self.max_task_steps)
            & (steps_since_hit >= 0)
            & (steps_since_hit <= int(window_steps))
        )

    def _post_hit_recovery_mask(self) -> torch.Tensor:
        # Keep recovery shaping active from the first hit until the next rally is launched.
        # `_reset_rally_state()` clears `has_hit` right before the next serve is written.
        return self.has_hit & (self.first_hit_step < self.max_task_steps)

    def _incoming_bounce_target_xy(
        self,
        ball_pos_w: torch.Tensor,
        ball_vel_w: torch.Tensor,
        gravity_z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ballistic predicted incoming-bounce target and predicted bounce time."""
        if gravity_z is None:
            gravity_z = self.gravity
        if gravity_z.ndim == 1:
            gravity_z = gravity_z.unsqueeze(-1)
        if gravity_z.shape[0] != ball_pos_w.shape[0]:
            if gravity_z.numel() == 1:
                gravity_z = gravity_z.reshape(1, 1).expand(ball_pos_w.shape[0], 1)
            else:
                g_val = float(gravity_z.reshape(-1)[0].detach().cpu().item())
                gravity_z = torch.full(
                    (ball_pos_w.shape[0], 1),
                    g_val,
                    device=ball_pos_w.device,
                    dtype=ball_pos_w.dtype,
                )
        pred_bounce_xy, pred_bounce_t = self._predict_first_bounce_ballistic(
            launch_pos=ball_pos_w,
            vel=ball_vel_w,
            gravity_z=gravity_z,
        )
        return pred_bounce_xy, pred_bounce_t

    def _incoming_contact_target(
        self,
        ball_pos_w: torch.Tensor,
        ball_vel_w: torch.Tensor,
        gravity_z: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict a fixed pre-hit contact target based on the ball trajectory crossing a fixed height (e.g. z=1.0m)."""
        target_bounce_xy, pred_bounce_t = self._incoming_bounce_target_xy(
            ball_pos_w, ball_vel_w, gravity_z=gravity_z
        )
        
        # Calculate time to reach target_height (e.g., 1.0m)
        target_height = 1.0
        if gravity_z is None:
            gravity_z = self.gravity
        if gravity_z.ndim == 1:
            gravity_z = gravity_z.unsqueeze(-1)
        if gravity_z.shape[0] != ball_pos_w.shape[0]:
            if gravity_z.numel() == 1:
                gravity_z = gravity_z.reshape(1, 1).expand(ball_pos_w.shape[0], 1)
            else:
                g_val = float(gravity_z.reshape(-1)[0].detach().cpu().item())
                gravity_z = torch.full(
                    (ball_pos_w.shape[0], 1),
                    g_val,
                    device=ball_pos_w.device,
                    dtype=ball_pos_w.dtype,
                )
        gravity_z = gravity_z.squeeze(-1)
        
        # We need to solve: z0 + vz*t + 0.5*g*t^2 = target_height
        # 0.5*g*t^2 + vz*t + (z0 - target_height) = 0
        a = 0.5 * gravity_z
        b = ball_vel_w[:, 2]
        c = ball_pos_w[:, 2] - target_height
        
        disc = (b.square() - 4.0 * a * c).clamp_min(0.0)
        sqrt_disc = torch.sqrt(disc)
        denom = (2.0 * a).clamp(min=-1.0e6, max=-1.0e-6)
        t1 = (-b - sqrt_disc) / denom
        t2 = (-b + sqrt_disc) / denom
        
        # Select the valid positive time
        t_candidates = torch.cat([t1.unsqueeze(-1), t2.unsqueeze(-1)], dim=-1)
        valid_candidates = t_candidates > 1.0e-4
        t_pos = torch.where(valid_candidates, t_candidates, torch.full_like(t_candidates, float("inf")))
        contact_t = t_pos.min(dim=-1).values
        
        # If the ball doesn't reach target_height, fallback to a time slightly before bounce
        fallback_t = (pred_bounce_t - float(self.approach_contact_lead_time)).clamp(float(self.approach_contact_min_t), float(self.approach_contact_max_t))
        valid_contact = torch.isfinite(contact_t)
        contact_t = torch.where(valid_contact, contact_t, fallback_t)
        
        contact_valid = pred_bounce_t > float(self.approach_contact_min_t + 1.0e-3)

        contact_xy = ball_pos_w[:, :2] + ball_vel_w[:, :2] * contact_t.unsqueeze(-1)
        
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
        max_bounce_t: float = 1.80,
        rear_margin_y: float = 0.8,
        lateral_stance_offset: float = 0.30,
        z_weight: float = 0.6,
        z_sigma: Sequence[float] | None = (0.08, 0.18),
        z_activate_net_y: float = -0.05,
        z_activate_xy_dist: float = 0.55,
        z_activate_contact_t: float = 0.45,
    ):
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

        # Force root movement: Reward is linear, dropping to 0 at 1.0m error
        root_xy_error = (root_pos_w[:, :2] - root_target_xy).norm(dim=-1, keepdim=True)
        root_xy_rew = (1.0 - root_xy_error / 1.0).clamp_min(0.0)

        # Racket only gets rewarded if root is close enough (e.g., error < 0.6m)
        racket_xy_error = (racket_pos_w[:, :2] - contact_pos_w[:, :2]).norm(dim=-1, keepdim=True)
        racket_xy_rew = (1.0 - racket_xy_error / 0.8).clamp_min(0.0)
        racket_gate = (root_xy_error < 0.6).float()
        
        rew = root_xy_rew + racket_gate * racket_xy_rew

        # Z shaping
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
            & (contact_t <= float(max_bounce_t))
            & (ball_pos_w[:, 1] > (root_pos_w[:, 1] - float(rear_margin_y)))
        ).float().unsqueeze(-1)
        return rew * active

    @reward
    def root_preposition_xy(
        self,
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
        
        # Use linear reward to force movement, drops to 0 at 1.0m
        root_xy_error = (root_pos_w[:, :2] - root_target_xy).norm(dim=-1, keepdim=True)
        rew = (1.0 - root_xy_error / 1.0).clamp_min(0.0)

        early_scale = (contact_t / max(float(max_bounce_t), 1.0e-3)).clamp(0.0, 1.0).unsqueeze(-1)
        rew = rew * (1.0 + float(early_preposition_weight) * early_scale)

        active = (
            (~self.has_hit)
            & (~self.fail_miss)
            & (~self.fail_net)
            & (~self.fail_out)
            & contact_valid
            & (contact_t <= float(max_bounce_t))
            & (ball_pos_w[:, 1] > (root_pos_w[:, 1] - float(rear_margin_y)))
        ).float().unsqueeze(-1)
        return rew * active

    @reward
    def root_towards_ball_speed(
        self,
        target_speed: float = 1.0,
        distance_norm: float = 1.8,
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
        
        # Use the same fixed racket face axis as runtime stroke-mode logic.
        body_quat_w = self.asset.data.body_link_quat_w[:, self.racket_body_id]
        racket_face_axis = self.racket_face_axis_local.unsqueeze(0).expand(self.num_envs, -1)
        racket_normal = quat_apply(body_quat_w, racket_face_axis)
        
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
    def stroke_mode_racket_side_prehit(
        self,
        side_margin: float = 0.03,
        side_scale: float = 0.06,
        max_bounce_t: float = 1.80,
    ):
        root_pos_w = self.asset.data.root_link_pos_w
        root_quat_w = self.asset.data.root_link_quat_w
        racket_pos_w, _ = self._racket_state_w()
        ball_pos_w = self.ball.data.root_link_pos_w
        ball_vel_w = self.ball.data.root_link_lin_vel_w
        _, contact_t, contact_valid, _, _ = self._incoming_contact_target(ball_pos_w, ball_vel_w)

        racket_pos_b = quat_apply_inverse(root_quat_w, racket_pos_w - root_pos_w)
        lateral = racket_pos_b[:, 1:2]
        scale = max(float(side_scale), 1.0e-6)
        margin = float(side_margin)

        target_forehand = (self.stroke_mode_target == self.STROKE_MODE_FOREHAND).float().unsqueeze(-1)
        target_backhand = (self.stroke_mode_target == self.STROKE_MODE_BACKHAND).float().unsqueeze(-1)

        # Right-handed setup: forehand prefers racket on robot-right (negative local-Y),
        # backhand prefers robot-left (positive local-Y).
        forehand_rew = torch.sigmoid(((-lateral) - margin) / scale)
        backhand_rew = torch.sigmoid((lateral - margin) / scale)
        rew = target_forehand * forehand_rew + target_backhand * backhand_rew

        active = (
            (~self.has_hit)
            & (~self.fail_miss)
            & (~self.fail_net)
            & (~self.fail_out)
            & contact_valid
            & (contact_t <= float(max_bounce_t))
        ).float().unsqueeze(-1)
        return rew * active

    @reward
    def hit_success(self):
        return self.hit_event.float().unsqueeze(-1)

    @reward
    def stroke_mode_match_on_hit(self):
        return self.hit_stroke_mode_match_event.float().unsqueeze(-1)

    @reward
    def stroke_mode_mismatch_on_hit(self):
        return self.hit_stroke_mode_mismatch_event.float().unsqueeze(-1)

    @reward
    def post_hit_clean_bonus(self):
        active = self._post_hit_window_mask(self.post_hit_clean_bonus_window_steps)
        clean = active & (~self.fail_racket_body) & (~self.racket_body_contact) & (~self.finished)
        return clean.float().unsqueeze(-1)

    @reward
    def post_hit_recover_root_xy(
        self,
        sigma: Sequence[float] | None = (0.35, 0.70),
    ):
        active = (self._post_hit_recovery_mask() & (~self.finished)).float().unsqueeze(-1)
        root_xy_err = (
            self.asset.data.root_link_pos_w[:, :2] - self.recover_root_pos_w[:, :2]
        ).norm(dim=-1, keepdim=True)
        return _exp_reward(root_xy_err, sigma) * active

    @reward
    def post_hit_recover_root_speed(
        self,
        target_speed_max: float = 1.8,
        distance_norm: float = 1.6,
        speed_sigma: float = 0.45,
    ):
        active = (self._post_hit_recovery_mask() & (~self.finished)).float().unsqueeze(-1)
        root_pos_w = self.asset.data.root_link_pos_w
        root_vel_w = self.asset.data.root_link_lin_vel_w
        delta_xy = self.recover_root_pos_w[:, :2] - root_pos_w[:, :2]
        dist_xy = delta_xy.norm(dim=-1, keepdim=True)
        dir_xy = delta_xy / dist_xy.clamp_min(1.0e-6)
        speed_towards = (root_vel_w[:, :2] * dir_xy).sum(dim=-1, keepdim=True)
        desired_speed = float(target_speed_max) * (dist_xy / max(float(distance_norm), 1.0e-6)).clamp(0.0, 1.0)
        speed_err = (speed_towards - desired_speed).abs()
        speed_term = torch.exp(-speed_err / max(float(speed_sigma), 1.0e-6))
        moving_towards_term = torch.sigmoid((speed_towards + 0.05) / max(float(speed_sigma), 1.0e-6))
        return speed_term * moving_towards_term * active

    @reward
    def post_hit_recover_heading(
        self,
        sigma: Sequence[float] | None = (0.20, 0.45),
    ):
        active = (self._post_hit_recovery_mask() & (~self.finished)).float().unsqueeze(-1)
        root_quat_w = self.asset.data.root_link_quat_w
        root_forward_w = quat_apply(root_quat_w, self._forward_dir_b)
        root_forward_xy = root_forward_w[:, :2]
        root_forward_xy = root_forward_xy / root_forward_xy.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        target_forward_xy = self.recover_root_forward_xy / self.recover_root_forward_xy.norm(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        heading_err = (root_forward_xy - target_forward_xy).norm(dim=-1, keepdim=True)
        return _exp_reward(heading_err, sigma) * active

    @reward
    def post_hit_recover_upper_pose(
        self,
        sigma: Sequence[float] | None = (0.06, 0.14),
    ):
        if self.recovery_upper_joint_ids_asset.numel() == 0:
            return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        active = (self._post_hit_recovery_mask() & (~self.finished)).float().unsqueeze(-1)
        joint_pos = self.asset.data.joint_pos[:, self.recovery_upper_joint_ids_asset]
        joint_ref = self.root_default_joint_pos[:, self.recovery_upper_joint_ids_asset]
        pose_err = (joint_pos - joint_ref).norm(dim=-1, keepdim=True)
        return _exp_reward(pose_err, sigma) * active

    @reward
    def post_hit_alive_no_racket_body_contact(self):
        active = self._post_hit_recovery_mask()
        alive = active & (~self.racket_body_contact) & (~self.fail_racket_body) & (~self.finished)
        return alive.float().unsqueeze(-1)

    @reward
    def post_hit_recover_zone_outer_enter(self):
        return self.recover_zone_outer_enter_event.float().unsqueeze(-1)

    @reward
    def post_hit_recover_zone_inner_enter(self):
        return self.recover_zone_inner_enter_event.float().unsqueeze(-1)

    @reward
    def success_streak_bonus(self, streak_cap: int = 8, streak_power: float = 1.7):
        cap = max(1, int(streak_cap))
        power = max(1.0, float(streak_power))
        streak = self.consecutive_return_count.float().clamp(1.0, float(cap)) / float(cap)
        streak = streak.pow(power)
        return self.success_event.float().unsqueeze(-1) * streak.unsqueeze(-1)

    @reward
    def post_hit_wrist_torque_guard(self):
        self._ensure_action_layout()
        if self.wrist_actuator_ids.numel() == 0:
            return torch.zeros((self.num_envs, 1), dtype=torch.float32, device=self.device)
        active = (
            self._post_hit_window_mask(self.post_hit_stability_window_steps) & (~self.finished)
        ).float().unsqueeze(-1)
        wrist_torque = self.asset.data.actuator_force[:, self.wrist_actuator_ids]
        return wrist_torque.square().sum(dim=-1, keepdim=True) * active

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
    def racket_body_contact_penalty(self, event_only: bool = False):
        contact = self.racket_body_contact_event if bool(event_only) else self.racket_body_contact
        return contact.float().unsqueeze(-1)

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
    def episode_fail_racket_body_pre_hit(self):
        return (self.finished & self.fail_racket_body & (~self.has_hit)).float().unsqueeze(-1)

    @reward
    def episode_fail_racket_body_post_hit(self):
        return (self.finished & self.fail_racket_body & self.has_hit).float().unsqueeze(-1)

    @reward
    def episode_fail_recover_timeout(self):
        return (self.finished & self.fail_recover_timeout).float().unsqueeze(-1)

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
