from __future__ import annotations

import re
from typing import Sequence

import numpy as np
import torch

from humanoid_tennis.utils.math import quat_apply, quat_apply_inverse


class HighLevelTennisActionContactMixin:
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
        prev_racket_body = self.racket_body_contact.clone()

        if self.contact_sensors.racket_ball is not None:
            self.racket_ball_contact[:] = self._sensor_contact_found(self.contact_sensors.racket_ball)
        else:
            self.racket_ball_contact[:] = False
        if self.contact_sensors.ball_net is not None:
            self.ball_net_contact[:] = self._sensor_contact_found(self.contact_sensors.ball_net)
        else:
            self.ball_net_contact[:] = False
        if self.contact_sensors.ball_court is not None:
            self.ball_court_contact[:] = self._sensor_contact_found(self.contact_sensors.ball_court)
        else:
            self.ball_court_contact[:] = False
        if self.use_racket_body_contact_sensor and self.contact_sensors.racket_body is not None:
            self.racket_body_contact[:] = self._sensor_contact_found(self.contact_sensors.racket_body)
        else:
            self.racket_body_contact[:] = False

        # Strict MuJoCo contact-pair guard (no sensor dependency):
        # detect racket geom vs robot-body collision geoms from live contact buffer.
        if (
            self.enable_racket_body_direct_contact_guard
            and self.racket_contact_geom_ids.numel() > 0
            and self.racket_body_contact_geom_ids.numel() > 0
        ):
            nacon_t = self.env.sim.data.nacon.reshape(-1)
            nacon = int(nacon_t[0].item()) if nacon_t.numel() > 0 else 0
            if nacon > 0:
                contact_geom = self.env.sim.data.contact.geom[:nacon]
                contact_world = self.env.sim.data.contact.worldid[:nacon].to(torch.long)
                contact_dist = self.env.sim.data.contact.dist[:nacon]
                g0 = contact_geom[:, 0]
                g1 = contact_geom[:, 1]
                r0 = torch.isin(g0, self.racket_contact_geom_ids)
                r1 = torch.isin(g1, self.racket_contact_geom_ids)
                b0 = torch.isin(g0, self.racket_body_contact_geom_ids)
                b1 = torch.isin(g1, self.racket_body_contact_geom_ids)
                contact_hit = (r0 & b1) | (r1 & b0)
                min_penetration = float(self.racket_body_contact_min_penetration)
                if min_penetration > 0.0:
                    # MuJoCo contact.dist < 0 means penetration depth.
                    penetration_hit = contact_dist <= (-min_penetration)
                    contact_hit = contact_hit & penetration_hit
                if contact_hit.any():
                    env_hit = torch.zeros((self.num_envs,), dtype=torch.bool, device=self.device)
                    env_ids = contact_world[contact_hit].clamp_(0, self.num_envs - 1)
                    env_hit[env_ids] = True
                    self.racket_body_contact[:] = self.racket_body_contact | env_hit

        self.racket_ball_contact_event[:] = self.racket_ball_contact & (~prev_racket_ball)
        self.ball_net_contact_event[:] = self.ball_net_contact & (~prev_ball_net)
        self.ball_court_contact_event[:] = self.ball_court_contact & (~prev_ball_court)
        self.racket_body_contact_event[:] = self.racket_body_contact & (~prev_racket_body)

    def _racket_state_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        body_pos = self.asset.data.body_link_pos_w[:, self.racket_body_id]
        body_quat = self.asset.data.body_link_quat_w[:, self.racket_body_id]
        center_offset_w = quat_apply(body_quat, self.racket_center_offset.unsqueeze(0).expand(self.num_envs, -1))
        racket_pos_w = body_pos + center_offset_w
        if self.contact_sensors.racket_velocity is not None:
            racket_vel_w = self.contact_sensors.racket_velocity.data
        else:
            body_lin_vel = self.asset.data.body_link_lin_vel_w[:, self.racket_body_id]
            body_ang_vel = self.asset.data.body_link_ang_vel_w[:, self.racket_body_id]
            racket_vel_w = body_lin_vel + torch.cross(body_ang_vel, center_offset_w, dim=-1)
        return racket_pos_w, racket_vel_w

    def _racket_face_dirs_w(self) -> tuple[torch.Tensor, torch.Tensor]:
        body_quat = self.asset.data.body_link_quat_w[:, self.racket_body_id]
        axis = self.racket_face_axis_local.unsqueeze(0).expand(self.num_envs, -1)
        face_plus = quat_apply(body_quat, axis)
        face_plus = face_plus / face_plus.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        face_minus = -face_plus
        if self.forehand_uses_negative_face_axis:
            return face_minus, face_plus
        return face_plus, face_minus

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
