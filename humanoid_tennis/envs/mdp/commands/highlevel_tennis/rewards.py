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
        """Predict pre-hit contact target; prefer lead-time before bounce, fallback to fixed-height crossing."""
        target_bounce_xy, pred_bounce_t = self._incoming_bounce_target_xy(
            ball_pos_w, ball_vel_w, gravity_z=gravity_z
        )

        # Primary target: a fixed lead time before the predicted first bounce.
        lead_t = (pred_bounce_t - float(self.approach_contact_lead_time)).clamp(
            float(self.approach_contact_min_t),
            float(self.approach_contact_max_t),
        )
        lead_valid = torch.isfinite(pred_bounce_t) & (pred_bounce_t > float(self.approach_contact_min_t + 1.0e-3))

        # Fallback: solve crossing time for a fixed contact height.
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

        a = 0.5 * gravity_z
        b = ball_vel_w[:, 2]
        c = ball_pos_w[:, 2] - target_height

        disc = (b.square() - 4.0 * a * c).clamp_min(0.0)
        sqrt_disc = torch.sqrt(disc)
        denom = (2.0 * a).clamp(min=-1.0e6, max=-1.0e-6)
        t1 = (-b - sqrt_disc) / denom
        t2 = (-b + sqrt_disc) / denom

        t_candidates = torch.cat([t1.unsqueeze(-1), t2.unsqueeze(-1)], dim=-1)
        valid_candidates = (t_candidates > float(self.approach_contact_min_t)) & (
            t_candidates <= float(self.approach_contact_max_t)
        )
        t_pos = torch.where(valid_candidates, t_candidates, torch.full_like(t_candidates, float("inf")))
        fixed_t = t_pos.min(dim=-1).values
        fixed_valid = torch.isfinite(fixed_t)

        fallback_t = torch.where(
            fixed_valid,
            fixed_t,
            torch.full_like(lead_t, float(self.approach_contact_max_t)),
        )
        contact_t = torch.where(lead_valid, lead_t, fallback_t)
        contact_valid = lead_valid | fixed_valid

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

    def _spawn_front_target_xy(
        self,
        front_offset_m: float = 1.2,
        lateral_offset_x: float = 0.0,
    ) -> torch.Tensor:
        """Opponent-court center target around mirrored spawn, slightly towards the net."""
        spawn_abs_y = abs(float(self.robot_spawn_pos[1].detach().cpu().item()))
        y_local = spawn_abs_y - float(front_offset_m)
        y_local = min(max(y_local, self.court_y_min_success), self.court_y_limit)
        env_xy = self.env.scene.env_origins[:, :2]
        target_x = env_xy[:, 0] + float(lateral_offset_x)
        target_y = env_xy[:, 1] + float(y_local)
        return torch.stack([target_x, target_y], dim=-1)

    def _solve_outgoing_velocity_with_tf_search(
        self,
        *,
        contact_pos_w: torch.Tensor,
        target_pos_w: torch.Tensor,
        tf_min: float,
        tf_max: float,
        tf_samples: int,
        max_outgoing_angle_deg: float,
        net_clearance_target: float,
        net_clearance_margin: float,
        w_speed: float,
        w_angle: float,
        w_clearance: float,
        w_time: float,
        w_out_risk: float,
        edge_soft_m: float,
        gravity_z: torch.Tensor | None = None,
        net_y_w: torch.Tensor | None = None,
        env_origins_xy: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Search flight time t_f and solve desired outgoing velocity to target bounce."""
        n = int(contact_pos_w.shape[0])
        if n <= 0:
            empty_v = torch.zeros((0, 3), device=self.device, dtype=torch.float32)
            empty_t = torch.zeros((0,), device=self.device, dtype=torch.float32)
            empty_ok = torch.zeros((0,), device=self.device, dtype=torch.bool)
            return empty_v, empty_t, empty_ok

        tf_lo = max(float(tf_min), 1.0e-3)
        tf_hi = max(float(tf_max), tf_lo + 1.0e-3)
        k = max(2, int(tf_samples))
        t_grid = torch.linspace(tf_lo, tf_hi, k, device=contact_pos_w.device, dtype=contact_pos_w.dtype)
        t = t_grid.unsqueeze(0).expand(n, -1)

        delta = (target_pos_w - contact_pos_w).to(dtype=contact_pos_w.dtype)
        dx = delta[:, 0:1]
        dy = delta[:, 1:2]
        dz = delta[:, 2:3]
        if gravity_z is None:
            gravity_z = self.gravity
        if gravity_z.ndim == 1:
            gravity_z = gravity_z.unsqueeze(-1)
        if gravity_z.shape[0] == n:
            g = gravity_z.to(dtype=contact_pos_w.dtype)
        elif gravity_z.numel() == 1:
            g = gravity_z.reshape(1, 1).to(dtype=contact_pos_w.dtype).expand(n, 1)
        else:
            g_val = float(gravity_z.reshape(-1)[0].detach().cpu().item())
            g = torch.full((n, 1), g_val, device=contact_pos_w.device, dtype=contact_pos_w.dtype)

        vx = dx / t
        vy = dy / t
        vz = (dz - 0.5 * g * t.square()) / t

        speed = torch.sqrt(vx.square() + vy.square() + vz.square()).clamp_min(1.0e-6)
        angle = torch.atan2(vz, vy.clamp_min(1.0e-4))
        angle_hi = math.radians(float(max_outgoing_angle_deg))
        angle_pen = (angle - angle_hi).clamp_min(0.0).square()

        if net_y_w is None:
            net_y_w = self.env.scene.env_origins[:, 1:2]
        if net_y_w.ndim == 1:
            net_y_w = net_y_w.unsqueeze(-1)
        if net_y_w.shape[0] == n:
            net_y = net_y_w.to(dtype=contact_pos_w.dtype)
        elif net_y_w.numel() == 1:
            net_y = net_y_w.reshape(1, 1).to(dtype=contact_pos_w.dtype).expand(n, 1)
        else:
            net_y_val = float(net_y_w.reshape(-1)[0].detach().cpu().item())
            net_y = torch.full((n, 1), net_y_val, device=contact_pos_w.device, dtype=contact_pos_w.dtype)
        t_net = (net_y - contact_pos_w[:, 1:2]) / vy.clamp_min(1.0e-4)
        z_net = contact_pos_w[:, 2:3] + vz * t_net + 0.5 * g * t_net.square()
        clearance = z_net - float(self.net_height)
        clear_pen = (float(net_clearance_target) - clearance).clamp_min(0.0).square()

        if env_origins_xy is None:
            env_origins_xy = self.env.scene.env_origins[:, :2]
        if env_origins_xy.shape[0] == n:
            origins_xy = env_origins_xy.to(dtype=target_pos_w.dtype)
        elif env_origins_xy.numel() == 2:
            origins_xy = env_origins_xy.reshape(1, 2).to(dtype=target_pos_w.dtype).expand(n, 2)
        else:
            ox = float(env_origins_xy.reshape(-1)[0].detach().cpu().item())
            oy = float(env_origins_xy.reshape(-1)[1].detach().cpu().item()) if env_origins_xy.numel() > 1 else 0.0
            origins_xy = torch.tensor([ox, oy], device=target_pos_w.device, dtype=target_pos_w.dtype).reshape(1, 2).expand(n, 2)
        target_xy_l = target_pos_w[:, :2] - origins_xy
        edge_margin_x = (self.court_x_limit - target_xy_l[:, 0].abs()).clamp_min(0.0).unsqueeze(-1)
        edge_margin_y_lo = (target_xy_l[:, 1] - self.court_y_min_success).clamp_min(0.0).unsqueeze(-1)
        edge_margin_y_hi = (self.court_y_limit - target_xy_l[:, 1]).clamp_min(0.0).unsqueeze(-1)
        edge_margin_y = torch.minimum(edge_margin_y_lo, edge_margin_y_hi)
        edge_soft = max(float(edge_soft_m), 1.0e-3)
        out_risk = speed / (edge_margin_x + edge_soft) + 0.5 * speed / (edge_margin_y + edge_soft)

        speed_min = float(self.outgoing_speed_minmax[0].detach().cpu().item())
        speed_max = float(self.outgoing_speed_minmax[1].detach().cpu().item())
        speed_pen = (speed_min - speed).clamp_min(0.0).square() + (speed - speed_max).clamp_min(0.0).square()

        obj = (
            float(w_speed) * speed
            + float(w_angle) * angle_pen
            + float(w_clearance) * clear_pen
            + float(w_time) * t
            + float(w_out_risk) * out_risk
            + 0.5 * speed_pen
        )

        feasible = (
            (vy > 0.2)
            & (t_net > 0.0)
            & (t_net < t)
            & (clearance >= float(net_clearance_margin))
        )
        infeasible_cost = 1.0e3
        obj_masked = obj + (~feasible).float() * infeasible_cost
        idx_best = obj_masked.argmin(dim=1)
        idx = idx_best.unsqueeze(-1)

        vx_best = torch.gather(vx, 1, idx)
        vy_best = torch.gather(vy, 1, idx)
        vz_best = torch.gather(vz, 1, idx)
        t_best = torch.gather(t, 1, idx).squeeze(-1)
        v_out_best = torch.cat([vx_best, vy_best, vz_best], dim=-1)
        feasible_best = torch.gather(feasible, 1, idx).squeeze(-1)
        return v_out_best, t_best, feasible_best

    def _select_receiver_friendly_bounce_target(
        self,
        *,
        env_ids: torch.Tensor,
        contact_pos_w: torch.Tensor,
        base_target_pos_w: torch.Tensor,
        tf_min: float,
        tf_max: float,
        tf_samples: int,
        max_outgoing_angle_deg: float,
        net_clearance_target: float,
        net_clearance_margin: float,
        w_speed: float,
        w_angle: float,
        w_clearance: float,
        w_time: float,
        w_out_risk: float,
        edge_soft_m: float,
        x_lim_safe: float,
        y_min_safe: float,
        y_max_safe: float,
        target_grid_x: int,
        target_grid_y: int,
        receiver_x_offset: float,
        receiver_y_offset: float,
        receiver_speed_max: float,
        receiver_reach_scale: float,
        receiver_contact_height: float,
        receiver_contact_height_sigma: float,
        receiver_contact_time_nominal: float,
        receiver_contact_time_sigma: float,
        post_bounce_vxy_damping: float,
        post_bounce_vz_restitution: float,
        w_receiver_reach: float,
        w_receiver_height: float,
        w_receiver_time: float,
        w_target_prior: float,
        target_prior_sigma: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Search candidate bounce targets and choose one that is easiest for the receiver to play."""
        n = int(env_ids.numel())
        if n <= 0:
            return base_target_pos_w, torch.zeros((0,), dtype=torch.bool, device=self.device)

        nx = max(2, int(target_grid_x))
        ny = max(2, int(target_grid_y))
        x_lin = torch.linspace(-float(x_lim_safe), float(x_lim_safe), nx, device=self.device, dtype=torch.float32)
        y_lin = torch.linspace(float(y_min_safe), float(y_max_safe), ny, device=self.device, dtype=torch.float32)
        grid_x, grid_y = torch.meshgrid(x_lin, y_lin, indexing="ij")
        cand_xy_local = torch.stack([grid_x.reshape(-1), grid_y.reshape(-1)], dim=-1)
        c = int(cand_xy_local.shape[0])

        env_origins = self.env.scene.env_origins[env_ids]
        cand_xy_world = cand_xy_local.unsqueeze(0) + env_origins[:, None, :2]
        cand_target_w = torch.zeros((n, c, 3), device=self.device, dtype=torch.float32)
        cand_target_w[:, :, :2] = cand_xy_world
        cand_target_w[:, :, 2] = float(self.ball_radius)

        contact_expand = contact_pos_w.unsqueeze(1).expand(-1, c, -1)
        gravity_flat = self.gravity[env_ids].repeat_interleave(c, dim=0)
        net_y_flat = env_origins[:, 1:2].repeat_interleave(c, dim=0)
        origins_xy_flat = env_origins[:, :2].repeat_interleave(c, dim=0)
        v_out_flat, tf_flat, feasible_flat = self._solve_outgoing_velocity_with_tf_search(
            contact_pos_w=contact_expand.reshape(-1, 3),
            target_pos_w=cand_target_w.reshape(-1, 3),
            tf_min=tf_min,
            tf_max=tf_max,
            tf_samples=tf_samples,
            max_outgoing_angle_deg=max_outgoing_angle_deg,
            net_clearance_target=net_clearance_target,
            net_clearance_margin=net_clearance_margin,
            w_speed=w_speed,
            w_angle=w_angle,
            w_clearance=w_clearance,
            w_time=w_time,
            w_out_risk=w_out_risk,
            edge_soft_m=edge_soft_m,
            gravity_z=gravity_flat,
            net_y_w=net_y_flat,
            env_origins_xy=origins_xy_flat,
        )
        v_out = v_out_flat.reshape(n, c, 3)
        tf = tf_flat.reshape(n, c)
        feasible = feasible_flat.reshape(n, c)

        spawn_abs_y = abs(float(self.robot_spawn_pos[1].detach().cpu().item()))
        receiver_x = env_origins[:, 0] + float(receiver_x_offset)
        receiver_y = env_origins[:, 1] + spawn_abs_y + float(receiver_y_offset)

        damp_xy = max(float(post_bounce_vxy_damping), 0.0)
        bounce_restitution = max(float(post_bounce_vz_restitution), 0.0)
        vx_post = v_out[:, :, 0] * damp_xy
        vy_post = v_out[:, :, 1] * damp_xy
        g = self.gravity[env_ids].squeeze(-1).unsqueeze(-1)
        vz_pre = v_out[:, :, 2] + g * tf
        vz_post = (-vz_pre).clamp_min(0.0) * bounce_restitution

        dy_recv = receiver_y.unsqueeze(-1) - cand_xy_world[:, :, 1]
        t_recv = dy_recv / vy_post.clamp_min(1.0e-4)
        valid_recv = (vy_post > 0.05) & (dy_recv > 0.05) & torch.isfinite(t_recv)

        x_recv = cand_xy_world[:, :, 0] + vx_post * t_recv
        z_recv = float(self.ball_radius) + vz_post * t_recv + 0.5 * g * t_recv.square()

        reach_scale = max(float(receiver_reach_scale), 1.0e-3)
        speed_max = max(float(receiver_speed_max), 0.1)
        reach_margin = speed_max * t_recv - (x_recv - receiver_x.unsqueeze(-1)).abs()
        reach_score = torch.sigmoid(reach_margin / reach_scale)

        h_sigma = max(float(receiver_contact_height_sigma), 1.0e-3)
        h_err = (z_recv - float(receiver_contact_height)) / h_sigma
        height_score = torch.exp(-0.5 * h_err.square())

        t_sigma = max(float(receiver_contact_time_sigma), 1.0e-3)
        t_err = (t_recv - float(receiver_contact_time_nominal)) / t_sigma
        time_score = torch.exp(-0.5 * t_err.square())

        prior_sigma = max(float(target_prior_sigma), 1.0e-3)
        prior_err = (cand_xy_world - base_target_pos_w[:, None, :2]).norm(dim=-1)
        prior_score = torch.exp(-0.5 * (prior_err / prior_sigma).square())

        score = (
            float(w_receiver_reach) * reach_score
            + float(w_receiver_height) * height_score
            + float(w_receiver_time) * time_score
            + float(w_target_prior) * prior_score
        )
        valid = feasible & valid_recv
        score = torch.where(valid, score, torch.full_like(score, -1.0e6))

        best_idx = score.argmax(dim=1)
        row_idx = torch.arange(n, device=self.device)
        best_valid = valid[row_idx, best_idx]
        best_target = cand_target_w[row_idx, best_idx]
        chosen_target = base_target_pos_w.clone()
        chosen_target[best_valid] = best_target[best_valid]
        return chosen_target, best_valid

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
        target_dir_xy = self._effective_target_bounce_w()[:, :2] - racket_pos_w[:, :2]
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
    def impact_guidance_prehit(
        self,
        tf_min: float = 0.30,
        tf_max: float = 0.65,
        tf_samples: int = 9,
        max_outgoing_angle_deg: float = 14.0,
        net_clearance_target: float = 0.14,
        net_clearance_margin: float = 0.02,
        restitution_n: float = 0.58,
        speed_sigma: float = 1.2,
        w_speed: float = 0.015,
        w_angle: float = 12.0,
        w_clearance: float = 10.0,
        w_time: float = 0.02,
        w_out_risk: float = 0.60,
        edge_soft_m: float = 0.30,
        n_align_weight: float = 0.7,
        v_track_weight: float = 0.3,
        activate_t_min: float = 0.10,
        activate_t_max: float = 0.55,
        target_inset_x: float = 0.35,
        target_inset_y: float = 0.45,
        lateral_stance_offset: float = 0.30,
        root_align_max_xy: float = 0.70,
        racket_contact_max_xy: float = 0.95,
        gate_scale_xy: float = 0.12,
        dynamic_target_search: bool = True,
        target_grid_x: int = 5,
        target_grid_y: int = 5,
        receiver_x_offset: float = 0.0,
        receiver_y_offset: float = 0.0,
        receiver_speed_max: float = 3.2,
        receiver_reach_scale: float = 0.22,
        receiver_contact_height: float = 1.02,
        receiver_contact_height_sigma: float = 0.22,
        receiver_contact_time_nominal: float = 0.55,
        receiver_contact_time_sigma: float = 0.24,
        post_bounce_vxy_damping: float = 0.80,
        post_bounce_vz_restitution: float = 0.62,
        w_receiver_reach: float = 2.0,
        w_receiver_height: float = 1.3,
        w_receiver_time: float = 1.0,
        w_target_prior: float = 0.4,
        target_prior_sigma: float = 1.4,
    ):
        # Pre-hit guidance: search a low/high-quality outgoing target via t_f,
        # then infer desired racket-face normal and normal-direction racket speed.
        ball_pos_w = self.ball.data.root_link_pos_w
        ball_vel_w = self.ball.data.root_link_lin_vel_w
        contact_pos_w, contact_t, contact_valid, _, _ = self._incoming_contact_target(ball_pos_w, ball_vel_w)

        target_pos_w = self._effective_target_bounce_w().clone()
        target_pos_w[:, 2] = self.ball_radius
        # Keep guidance target away from court edges to reduce out failures.
        target_xy_l = target_pos_w[:, :2] - self.env.scene.env_origins[:, :2]
        x_lim_safe = max(self.court_x_limit - float(target_inset_x), 0.5)
        y_min_safe = self.court_y_min_success + float(target_inset_y)
        y_max_safe = self.court_y_limit - float(target_inset_y)
        if y_min_safe > y_max_safe:
            y_mid = 0.5 * (self.court_y_min_success + self.court_y_limit)
            y_min_safe = y_mid
            y_max_safe = y_mid
        target_xy_l[:, 0] = target_xy_l[:, 0].clamp(-x_lim_safe, x_lim_safe)
        target_xy_l[:, 1] = target_xy_l[:, 1].clamp(y_min_safe, y_max_safe)
        target_pos_w[:, :2] = target_xy_l + self.env.scene.env_origins[:, :2]
        if bool(dynamic_target_search):
            dynamic_active = (
                (~self.has_hit)
                & (~self.fail_miss)
                & (~self.fail_net)
                & (~self.fail_out)
                & contact_valid
                & (contact_t >= float(activate_t_min))
                & (contact_t <= float(activate_t_max))
            )
            if bool(dynamic_active.any()):
                dyn_ids = dynamic_active.nonzero(as_tuple=False).squeeze(-1)
                dyn_base_target_w = target_pos_w[dyn_ids].clone()
                dyn_target_w, dyn_valid = self._select_receiver_friendly_bounce_target(
                    env_ids=dyn_ids,
                    contact_pos_w=contact_pos_w[dyn_ids],
                    base_target_pos_w=dyn_base_target_w,
                    tf_min=tf_min,
                    tf_max=tf_max,
                    tf_samples=tf_samples,
                    max_outgoing_angle_deg=max_outgoing_angle_deg,
                    net_clearance_target=net_clearance_target,
                    net_clearance_margin=net_clearance_margin,
                    w_speed=w_speed,
                    w_angle=w_angle,
                    w_clearance=w_clearance,
                    w_time=w_time,
                    w_out_risk=w_out_risk,
                    edge_soft_m=edge_soft_m,
                    x_lim_safe=x_lim_safe,
                    y_min_safe=y_min_safe,
                    y_max_safe=y_max_safe,
                    target_grid_x=target_grid_x,
                    target_grid_y=target_grid_y,
                    receiver_x_offset=receiver_x_offset,
                    receiver_y_offset=receiver_y_offset,
                    receiver_speed_max=receiver_speed_max,
                    receiver_reach_scale=receiver_reach_scale,
                    receiver_contact_height=receiver_contact_height,
                    receiver_contact_height_sigma=receiver_contact_height_sigma,
                    receiver_contact_time_nominal=receiver_contact_time_nominal,
                    receiver_contact_time_sigma=receiver_contact_time_sigma,
                    post_bounce_vxy_damping=post_bounce_vxy_damping,
                    post_bounce_vz_restitution=post_bounce_vz_restitution,
                    w_receiver_reach=w_receiver_reach,
                    w_receiver_height=w_receiver_height,
                    w_receiver_time=w_receiver_time,
                    w_target_prior=w_target_prior,
                    target_prior_sigma=target_prior_sigma,
                )
                target_pos_w[dyn_ids] = dyn_target_w
                self.guidance_target_bounce_w[dyn_ids] = dyn_target_w
                self.guidance_target_valid[dyn_ids] = True
                if hasattr(self.env, "extra"):
                    shift_xy = (dyn_target_w[:, :2] - dyn_base_target_w[:, :2]).norm(dim=-1)
                    self.env.extra["highlevel/dyn_target_valid_ratio"] = float(dyn_valid.float().mean().item())
                    self.env.extra["highlevel/dyn_target_shift_xy_mean"] = float(shift_xy.mean().item())
        v_out_star, _, guide_feasible = self._solve_outgoing_velocity_with_tf_search(
            contact_pos_w=contact_pos_w,
            target_pos_w=target_pos_w,
            tf_min=tf_min,
            tf_max=tf_max,
            tf_samples=tf_samples,
            max_outgoing_angle_deg=max_outgoing_angle_deg,
            net_clearance_target=net_clearance_target,
            net_clearance_margin=net_clearance_margin,
            w_speed=w_speed,
            w_angle=w_angle,
            w_clearance=w_clearance,
            w_time=w_time,
            w_out_risk=w_out_risk,
            edge_soft_m=edge_soft_m,
        )

        gravity_z = self.gravity.squeeze(-1)
        v_in_contact = ball_vel_w.clone()
        v_in_contact[:, 2] = v_in_contact[:, 2] + gravity_z * contact_t

        delta_v = v_out_star - v_in_contact
        n_star = delta_v / delta_v.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)

        e = min(max(float(restitution_n), 0.0), 0.98)
        v_in_n = (v_in_contact * n_star).sum(dim=-1, keepdim=True)
        v_out_n = (v_out_star * n_star).sum(dim=-1, keepdim=True)
        v_racket_n_star = (v_out_n + e * v_in_n) / max(1.0 + e, 1.0e-6)

        root_pos_w = self.asset.data.root_link_pos_w
        root_quat_w = self.asset.data.root_link_quat_w
        racket_pos_w, racket_vel_w = self._racket_state_w()
        root_target_xy = self._root_stance_target_xy(
            target_bounce_xy=contact_pos_w[:, :2],
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            racket_pos_w=racket_pos_w,
            lateral_stance_offset=lateral_stance_offset,
        )
        root_xy_err = (root_pos_w[:, :2] - root_target_xy).norm(dim=-1, keepdim=True)
        racket_contact_xy_err = (racket_pos_w[:, :2] - contact_pos_w[:, :2]).norm(dim=-1, keepdim=True)
        gate_scale = max(float(gate_scale_xy), 1.0e-3)
        root_near_gate = torch.sigmoid((float(root_align_max_xy) - root_xy_err) / gate_scale)
        racket_near_gate = torch.sigmoid((float(racket_contact_max_xy) - racket_contact_xy_err) / gate_scale)

        forehand_face_w, backhand_face_w = self._racket_face_dirs_w()
        target_forehand = self.stroke_mode_target == self.STROKE_MODE_FOREHAND
        target_backhand = self.stroke_mode_target == self.STROKE_MODE_BACKHAND
        target_neutral = ~(target_forehand | target_backhand)
        desired_face_w = torch.where(
            target_forehand.unsqueeze(-1),
            forehand_face_w,
            backhand_face_w,
        )
        n_align_signed = (desired_face_w * n_star).sum(dim=-1, keepdim=True).clamp_min(0.0)
        # Neutral mode fallback: choose the better face side automatically.
        fore_align = (forehand_face_w * n_star).sum(dim=-1, keepdim=True)
        back_align = (backhand_face_w * n_star).sum(dim=-1, keepdim=True)
        n_align_neutral = torch.maximum(fore_align, back_align).clamp_min(0.0)
        n_align = torch.where(target_neutral.unsqueeze(-1), n_align_neutral, n_align_signed)
        speed_err = ((racket_vel_w * n_star).sum(dim=-1, keepdim=True) - v_racket_n_star).abs()
        speed_rew = torch.exp(-speed_err / max(float(speed_sigma), 1.0e-6))
        rew = (float(n_align_weight) * n_align + float(v_track_weight) * speed_rew) * root_near_gate * racket_near_gate

        active = (
            (~self.has_hit)
            & (~self.fail_miss)
            & (~self.fail_net)
            & (~self.fail_out)
            & contact_valid
            & guide_feasible
            & (contact_t >= float(activate_t_min))
            & (contact_t <= float(activate_t_max))
        ).float().unsqueeze(-1)
        return rew * active

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

        target_xy = self._effective_target_bounce_w()[:, :2]
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
        err = (self.bounce_pos_w - self._effective_target_bounce_w()).norm(dim=-1, keepdim=True)
        rew = _exp_reward(err, sigma)
        return rew * self.bounce_event.float().unsqueeze(-1) * self.bounce_in.float().unsqueeze(-1)

    @reward
    def predicted_bounce_deep_center(
        self,
        bounce_pos_scale: float = 0.28,
        bounce_time_scale: float = 0.25,
        max_bounce_time: float = 1.5,
        front_offset_m: float = 1.2,
        lateral_offset_x: float = 0.0,
        require_predicted_in: bool = True,
    ):
        ball_pos_w = self.ball.data.root_link_pos_w
        ball_vel_w = self.ball.data.root_link_lin_vel_w

        pred_bounce_xy, pred_bounce_t = self._predict_first_bounce_ballistic(
            launch_pos=ball_pos_w,
            vel=ball_vel_w,
            gravity_z=self.gravity,
        )
        pred_bounce_t = pred_bounce_t.clamp(0.0, float(max_bounce_time))

        target_xy = self._spawn_front_target_xy(
            front_offset_m=front_offset_m,
            lateral_offset_x=lateral_offset_x,
        )
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
    def bounce_deep_center(
        self,
        sigma: Sequence[float] | None = (0.45, 0.90),
        front_offset_m: float = 1.2,
        lateral_offset_x: float = 0.0,
    ):
        target_xy = self._spawn_front_target_xy(
            front_offset_m=front_offset_m,
            lateral_offset_x=lateral_offset_x,
        )
        err = (self.bounce_pos_w[:, :2] - target_xy).norm(dim=-1, keepdim=True)
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
    def recover_timeout_penalty(self):
        # Penalty-only (non-terminal): when recovery wait times out, continue rally.
        return self.fail_recover_timeout.float().unsqueeze(-1)

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
