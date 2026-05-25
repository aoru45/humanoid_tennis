from __future__ import annotations

import math
from dataclasses import dataclass

import torch


@dataclass
class LaunchPhysicsConfig:
    ball_radius: float
    ball_mass: float
    air_density: float
    air_drag_k: float
    drag_coef: float
    lift_spin_scale: float
    spin_damping_coef: float
    net_height: float
    gravity_z: float = -9.81


@dataclass
class LaunchSamplerConfig:
    launcher_x_range: tuple[float, float] = (-2.8, 2.8)
    launcher_y_range: tuple[float, float] = (8.5, 10.5)
    launcher_z_range: tuple[float, float] = (1.5, 2.4)
    strike_x_range: tuple[float, float] = (-1.0, 1.0)
    strike_y_range: tuple[float, float] = (-1.3, -0.2)
    strike_z_range: tuple[float, float] = (0.9, 1.3)
    flight_t_range: tuple[float, float] = (0.45, 0.75)
    launch_speed_range: tuple[float, float] = (14.0, 32.0)
    launch_spin_rps_range: tuple[float, float] = (-9.0, 9.0)
    solver_iters: int = 2
    resample_attempts: int = 5
    strike_tolerance: float = 0.22
    prediction_substeps: int = 2
    predict_dt: float | None = 0.01
    enforce_net_clearance: bool = True
    net_clearance_margin: float = 0.05
    angle_deg_range: tuple[float, float] = (7.0, 18.0)
    min_vz: float = 3.0
    min_forward_speed: float = 10.0
    extra_resample_attempts: int = 120
    enforce_aero_first_bounce_in: bool = False
    aero_bounce_max_time: float = 2.6
    aero_bounce_out_x_margin: float = 1.6
    aero_bounce_out_y_margin: float = 2.8
    aero_bounce_contact_eps: float = 0.004
    aero_bounce_range_relax_x: float = 0.6
    aero_bounce_range_relax_y: float = 0.8
    clearance_correction_iters: int = 2
    enforce_incoming_bounce_in: bool = True
    incoming_bounce_x_range: tuple[float, float] = (-3.8, 3.8)
    incoming_bounce_y_range: tuple[float, float] = (-10.8, -0.4)
    target_x_range: tuple[float, float] = (-3.5, 3.5)
    target_y_range: tuple[float, float] = (7.2, 11.2)
    physics_dt: float = 0.005


class LaunchTrajectorySampler:
    """Standalone launch-machine sampler for offline bank generation."""

    def __init__(
        self,
        *,
        device: str,
        physics: LaunchPhysicsConfig,
        config: LaunchSamplerConfig,
    ):
        self.device = torch.device(device)
        self.physics = physics
        self.config = config

        self.ball_radius = float(physics.ball_radius)
        self.ball_mass = float(physics.ball_mass)
        self.air_density = float(physics.air_density)
        self.air_drag_k = float(physics.air_drag_k)
        self.drag_coef = float(physics.drag_coef)
        self.lift_spin_scale = float(physics.lift_spin_scale)
        self.spin_damping_coef = float(physics.spin_damping_coef)
        self.net_height = float(physics.net_height)
        self.gravity_z_value = float(physics.gravity_z)

        self.aero_force_k = 0.5 * self.air_density * math.pi * (self.ball_radius**2) * self.air_drag_k

        self.launcher_x_range = self._tensor_range(config.launcher_x_range)
        self.launcher_y_range = self._tensor_range(config.launcher_y_range)
        self.launcher_z_range = self._tensor_range(config.launcher_z_range)
        self.strike_x_range = self._tensor_range(config.strike_x_range)
        self.strike_y_range = self._tensor_range(config.strike_y_range)
        self.strike_z_range = self._tensor_range(config.strike_z_range)
        self.flight_t_range = self._tensor_range(config.flight_t_range)
        self.launch_speed_range = self._tensor_range(config.launch_speed_range)
        self.launch_spin_rps_range = self._tensor_range(config.launch_spin_rps_range)
        self.launch_angle_deg_range = self._tensor_range(config.angle_deg_range)
        self.incoming_bounce_x_range = self._tensor_range(config.incoming_bounce_x_range)
        self.incoming_bounce_y_range = self._tensor_range(config.incoming_bounce_y_range)
        self.target_x_range = self._tensor_range(config.target_x_range)
        self.target_y_range = self._tensor_range(config.target_y_range)

        self.launch_solver_iters = max(0, int(config.solver_iters))
        self.launch_resample_attempts = max(1, int(config.resample_attempts))
        self.launch_strike_tolerance = float(config.strike_tolerance)
        self.launch_min_vz = float(config.min_vz)
        self.launch_min_forward_speed = float(config.min_forward_speed)
        self.launch_extra_resample_attempts = max(1, int(config.extra_resample_attempts))
        self.enforce_launch_net_clearance = bool(config.enforce_net_clearance)
        self.launch_net_clearance_margin = float(config.net_clearance_margin)
        self.enforce_launch_incoming_bounce_in = bool(config.enforce_incoming_bounce_in)
        self.enforce_aero_first_bounce_in = bool(config.enforce_aero_first_bounce_in)
        self.aero_bounce_max_time = float(config.aero_bounce_max_time)
        self.aero_bounce_out_x_margin = max(0.0, float(config.aero_bounce_out_x_margin))
        self.aero_bounce_out_y_margin = max(0.0, float(config.aero_bounce_out_y_margin))
        self.aero_bounce_contact_eps = max(1.0e-4, float(config.aero_bounce_contact_eps))
        self.aero_bounce_range_relax_x = max(0.0, float(config.aero_bounce_range_relax_x))
        self.aero_bounce_range_relax_y = max(0.0, float(config.aero_bounce_range_relax_y))
        self.launch_clearance_correction_iters = max(0, int(config.clearance_correction_iters))

        prediction_substeps = max(1, int(config.prediction_substeps))
        base_predict_dt = float(config.physics_dt) / float(prediction_substeps)
        if config.predict_dt is None:
            self.launch_predict_dt = base_predict_dt
        else:
            self.launch_predict_dt = max(float(config.predict_dt), base_predict_dt)
        self.aero_bounce_max_time = max(self.aero_bounce_max_time, self.launch_predict_dt)

        self.gravity_dir = torch.tensor([0.0, 0.0, -1.0], device=self.device, dtype=torch.float32)
        self.last_sample_diagnostics: dict[str, object] | None = None

    def _tensor_range(self, values: tuple[float, float]) -> torch.Tensor:
        lo, hi = float(values[0]), float(values[1])
        if lo >= hi:
            raise ValueError(f"Invalid range: [{lo}, {hi}]")
        return torch.tensor([lo, hi], device=self.device, dtype=torch.float32)

    def _sample_uniform_n(self, num_samples: int, ranges: torch.Tensor) -> torch.Tensor:
        return torch.rand((num_samples,), device=self.device) * (ranges[1] - ranges[0]) + ranges[0]

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

    def _predict_first_bounce_aero(
        self,
        launch_pos: torch.Tensor,
        launch_vel: torch.Tensor,
        launch_ang: torch.Tensor,
        gravity_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Predict first bounce under the same aero model used during launch prediction."""
        n = launch_pos.shape[0]
        pos = launch_pos.clone()
        vel = launch_vel.clone()
        ang = launch_ang.clone()
        active = torch.ones((n,), device=self.device, dtype=torch.bool)
        found = torch.zeros((n,), device=self.device, dtype=torch.bool)
        bounce_xy = torch.zeros((n, 2), device=self.device, dtype=torch.float32)
        bounce_t = torch.full((n,), self.aero_bounce_max_time, device=self.device, dtype=torch.float32)

        max_x = max(abs(float(self.incoming_bounce_x_range[0].item())), abs(float(self.incoming_bounce_x_range[1].item())))
        x_bound = max_x + self.aero_bounce_out_x_margin
        y_low = float(self.incoming_bounce_y_range[0].item()) - self.aero_bounce_out_y_margin
        y_high = float(self.launcher_y_range[1].item()) + self.aero_bounce_out_y_margin
        max_steps = max(1, int(math.ceil(self.aero_bounce_max_time / self.launch_predict_dt)))
        contact_z = self.ball_radius + self.aero_bounce_contact_eps

        for step_i in range(max_steps):
            if not active.any():
                break
            ids = active.nonzero(as_tuple=False).squeeze(-1)
            vel_a = vel[ids]
            ang_a = ang[ids]
            speed = vel_a.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            spin_mag = ang_a.norm(dim=-1, keepdim=True)
            spin_scaled = spin_mag / (2.0 * math.pi) * self.lift_spin_scale
            vel_dir = vel_a / speed
            spin_axis = ang_a / spin_mag.clamp_min(1e-6)
            cl = 1.0 / (2.0 + torch.abs(speed / (spin_scaled + 1e-6)))
            drag_force = -self.aero_force_k * self.drag_coef * speed * vel_a
            lift_force = self.aero_force_k * cl * speed.square() * torch.cross(spin_axis, vel_dir, dim=-1)
            acc = (drag_force + lift_force) / self.ball_mass
            acc[:, 2:3] = acc[:, 2:3] + gravity_z[ids]
            vel_next = vel_a + acc * self.launch_predict_dt
            pos_next = pos[ids] + vel_next * self.launch_predict_dt
            spin_decay = max(0.85, 1.0 - self.spin_damping_coef * self.launch_predict_dt)
            ang_next = ang_a * spin_decay

            vel[ids] = vel_next
            pos[ids] = pos_next
            ang[ids] = ang_next

            hit = (pos_next[:, 2] <= contact_z) & (vel_next[:, 2] < 0.0)
            if hit.any():
                hit_ids = ids[hit]
                found[hit_ids] = True
                bounce_xy[hit_ids] = pos_next[hit, :2]
                bounce_t[hit_ids] = (step_i + 1) * self.launch_predict_dt

            out = (
                (pos_next[:, 0].abs() > x_bound)
                | (pos_next[:, 1] < y_low)
                | (pos_next[:, 1] > y_high)
            )
            deactivate = hit | out
            if deactivate.any():
                active[ids[deactivate]] = False

        return bounce_xy, bounce_t, found

    def _launch_quality_checks(
        self,
        launch_pos: torch.Tensor,
        vel: torch.Tensor,
        ang: torch.Tensor,
        pred_pos: torch.Tensor,
        net_cross_z: torch.Tensor,
        gravity_z: torch.Tensor,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        speed = vel.norm(dim=-1)
        horiz_speed = vel[:, :2].norm(dim=-1).clamp_min(1e-6)
        launch_angle_deg = torch.atan2(vel[:, 2], horiz_speed) * (180.0 / math.pi)
        forward_speed = -vel[:, 1]

        checks: dict[str, torch.Tensor] = {}
        checks["speed_min"] = speed >= self.launch_speed_range[0]
        checks["speed_max"] = speed <= self.launch_speed_range[1]
        checks["strike_height"] = pred_pos[:, 2] > (self.ball_radius + 0.02)
        checks["forward_speed_min"] = forward_speed >= self.launch_min_forward_speed
        checks["vz_min"] = vel[:, 2] >= self.launch_min_vz
        checks["angle_min"] = launch_angle_deg >= self.launch_angle_deg_range[0]
        checks["angle_max"] = launch_angle_deg <= self.launch_angle_deg_range[1]
        if self.enforce_launch_net_clearance:
            checks["net_clearance"] = net_cross_z > (self.net_height + self.launch_net_clearance_margin)
        if self.enforce_launch_incoming_bounce_in:
            if self.enforce_aero_first_bounce_in:
                bounce_xy, bounce_t, bounce_found = self._predict_first_bounce_aero(
                    launch_pos=launch_pos,
                    launch_vel=vel,
                    launch_ang=ang,
                    gravity_z=gravity_z,
                )
                checks["bounce_found"] = bounce_found
                x_lo = self.incoming_bounce_x_range[0] - self.aero_bounce_range_relax_x
                x_hi = self.incoming_bounce_x_range[1] + self.aero_bounce_range_relax_x
                y_lo = self.incoming_bounce_y_range[0] - self.aero_bounce_range_relax_y
                y_hi = self.incoming_bounce_y_range[1] + self.aero_bounce_range_relax_y
            else:
                bounce_xy, bounce_t = self.predict_first_bounce_ballistic(launch_pos, vel, gravity_z)
                x_lo = self.incoming_bounce_x_range[0]
                x_hi = self.incoming_bounce_x_range[1]
                y_lo = self.incoming_bounce_y_range[0]
                y_hi = self.incoming_bounce_y_range[1]
            checks["bounce_time_min"] = bounce_t > 0.08
            checks["bounce_x_min"] = bounce_xy[:, 0] >= x_lo
            checks["bounce_x_max"] = bounce_xy[:, 0] <= x_hi
            checks["bounce_y_min"] = bounce_xy[:, 1] >= y_lo
            checks["bounce_y_max"] = bounce_xy[:, 1] <= y_hi

        valid = torch.ones((vel.shape[0],), device=self.device, dtype=torch.bool)
        for mask in checks.values():
            valid &= mask
        return valid, checks

    def _launch_quality_mask(
        self,
        launch_pos: torch.Tensor,
        vel: torch.Tensor,
        ang: torch.Tensor,
        pred_pos: torch.Tensor,
        net_cross_z: torch.Tensor,
        gravity_z: torch.Tensor,
    ) -> torch.Tensor:
        valid, _ = self._launch_quality_checks(
            launch_pos=launch_pos,
            vel=vel,
            ang=ang,
            pred_pos=pred_pos,
            net_cross_z=net_cross_z,
            gravity_z=gravity_z,
        )
        return valid

    def _accumulate_failure_counts(
        self,
        reason_counts: dict[str, int],
        valid: torch.Tensor,
        checks: dict[str, torch.Tensor],
    ) -> None:
        failed = ~valid
        if not failed.any():
            return
        for name, mask in checks.items():
            n_fail = int((failed & (~mask)).sum().item())
            if n_fail <= 0:
                continue
            reason_counts[name] = reason_counts.get(name, 0) + n_fail

    def _format_diag_reasons(self, reason_counts: dict[str, int], rejected: int, top_k: int = 8) -> str:
        if rejected <= 0:
            return "none"
        if not reason_counts:
            return "unknown"
        items = sorted(reason_counts.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
        return ", ".join([f"{k}={v}/{rejected}({v / max(1, rejected):.1%})" for k, v in items])

    def format_last_sample_diagnostics(self, top_k: int = 8) -> str:
        diag = self.last_sample_diagnostics
        if diag is None:
            return "no diagnostics"
        c1 = int(diag["primary_candidates"])
        a1 = int(diag["primary_accepted"])
        r1 = max(0, c1 - a1)
        c2 = int(diag["extra_candidates"])
        a2 = int(diag["extra_accepted"])
        r2 = max(0, c2 - a2)
        p = int(diag["pending_final"])
        return (
            f"primary attempts={diag['primary_attempts']}, accepted={a1}/{c1}; "
            f"extra attempts={diag['extra_attempts']}, accepted={a2}/{c2}; "
            f"pending={p}; "
            f"primary_fail=[{self._format_diag_reasons(diag['primary_reasons'], r1, top_k=top_k)}]; "
            f"extra_fail=[{self._format_diag_reasons(diag['extra_reasons'], r2, top_k=top_k)}]"
        )

    def predict_first_bounce_ballistic(
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
            _, _, net_cross_z = self._predict_ball_at_time(launch_pos, vel, ang, flight_t, gravity_z)
            low = net_cross_z <= desired_net_z
            if not low.any():
                break
            forward_speed = (-vel[low, 1]).clamp_min(1.0e-3)
            t_cross = (launch_pos[low, 1] / forward_speed).clamp_min(0.08)
            dz = (desired_net_z - net_cross_z[low]).clamp_min(0.0) + 0.04
            vel[low, 2] = vel[low, 2] + dz / t_cross
            vel[low] = self._clamp_launch_speed(vel[low])
        return vel

    def sample(
        self,
        num_samples: int,
        *,
        collect_diagnostics: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        self.last_sample_diagnostics = None
        num_samples = int(num_samples)
        if num_samples <= 0:
            empty = torch.zeros((0, 3), device=self.device, dtype=torch.float32)
            return empty, empty, empty, empty

        gravity_z = torch.full((num_samples, 1), self.gravity_z_value, device=self.device, dtype=torch.float32)

        launch_pos_all = torch.zeros((num_samples, 3), device=self.device, dtype=torch.float32)
        launch_vel_all = torch.zeros((num_samples, 3), device=self.device, dtype=torch.float32)
        launch_ang_all = torch.zeros((num_samples, 3), device=self.device, dtype=torch.float32)
        target_all = torch.zeros((num_samples, 3), device=self.device, dtype=torch.float32)
        diagnostics = None
        if collect_diagnostics:
            diagnostics = {
                "requested": num_samples,
                "primary_attempts": 0,
                "extra_attempts": 0,
                "primary_candidates": 0,
                "extra_candidates": 0,
                "primary_accepted": 0,
                "extra_accepted": 0,
                "primary_reasons": {},
                "extra_reasons": {},
                "pending_final": 0,
            }

        pending = torch.arange(num_samples, device=self.device, dtype=torch.long)
        for _ in range(self.launch_resample_attempts):
            if pending.numel() == 0:
                break
            n = pending.numel()
            if diagnostics is not None:
                diagnostics["primary_attempts"] += 1
                diagnostics["primary_candidates"] += int(n)
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
            vel = self._solve_ballistic_velocity(launch_pos, strike_pos, flight_t, gravity_z[pending])
            for _ in range(max(0, self.launch_solver_iters)):
                ang = self._compute_launch_spin(vel, spin_rps)
                pred_pos, _, _ = self._predict_ball_at_time(launch_pos, vel, ang, flight_t, gravity_z[pending])
                vel = vel + (strike_pos - pred_pos) / flight_t.clamp_min(1e-3)
                vel = self._clamp_launch_speed(vel)

            vel = self._improve_launch_clearance(
                launch_pos=launch_pos,
                vel=vel,
                spin_rps=spin_rps,
                flight_t=flight_t,
                gravity_z=gravity_z[pending],
            )
            vel = self._clamp_launch_speed(vel)

            ang = self._compute_launch_spin(vel, spin_rps)
            pred_pos, _, net_cross_z = self._predict_ball_at_time(launch_pos, vel, ang, flight_t, gravity_z[pending])
            strike_err = (pred_pos - strike_pos).norm(dim=-1)
            checks: dict[str, torch.Tensor] = {"strike_tolerance": strike_err <= self.launch_strike_tolerance}
            quality_valid, quality_checks = self._launch_quality_checks(
                launch_pos=launch_pos,
                vel=vel,
                ang=ang,
                pred_pos=pred_pos,
                net_cross_z=net_cross_z,
                gravity_z=gravity_z[pending],
            )
            checks.update(quality_checks)
            valid = checks["strike_tolerance"] & quality_valid
            if diagnostics is not None:
                diagnostics["primary_accepted"] += int(valid.sum().item())
                self._accumulate_failure_counts(diagnostics["primary_reasons"], valid, checks)

            if valid.any():
                accepted = pending[valid]
                launch_pos_all[accepted] = launch_pos[valid]
                launch_vel_all[accepted] = vel[valid]
                launch_ang_all[accepted] = ang[valid]
                target_all[accepted] = target_pos[valid]
            pending = pending[~valid]

        for _ in range(self.launch_extra_resample_attempts):
            if pending.numel() == 0:
                break
            n = pending.numel()
            if diagnostics is not None:
                diagnostics["extra_attempts"] += 1
                diagnostics["extra_candidates"] += int(n)
            launch_pos = torch.zeros((n, 3), device=self.device, dtype=torch.float32)
            launch_pos[:, 0] = self._sample_uniform_n(n, self.launcher_x_range)
            launch_pos[:, 1] = self._sample_uniform_n(n, self.launcher_y_range)
            launch_pos[:, 2] = self._sample_uniform_n(n, self.launcher_z_range)
            strike_pos = torch.zeros_like(launch_pos)
            strike_pos[:, 0] = self._sample_uniform_n(n, self.strike_x_range)
            strike_pos[:, 1] = self._sample_uniform_n(n, self.strike_y_range)
            strike_z_low = max(float(self.strike_z_range[0].item()), self.net_height + 0.30)
            strike_z_high = max(strike_z_low + 0.20, float(self.strike_z_range[1].item()) + 0.30)
            strike_pos[:, 2] = self._sample_uniform_n(
                n, torch.tensor([strike_z_low, strike_z_high], device=self.device, dtype=torch.float32)
            )
            flight_t = self._sample_uniform_n(n, self.flight_t_range).unsqueeze(-1)
            target_pos = torch.zeros_like(launch_pos)
            target_pos[:, 0] = self._sample_uniform_n(n, self.target_x_range)
            target_pos[:, 1] = self._sample_uniform_n(n, self.target_y_range)
            spin_rps = self._sample_uniform_n(n, self.launch_spin_rps_range)

            vel = self._solve_ballistic_velocity(launch_pos, strike_pos, flight_t, gravity_z[pending])
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
                vel = self._clamp_launch_speed(vel)
            vel = self._improve_launch_clearance(
                launch_pos=launch_pos,
                vel=vel,
                spin_rps=spin_rps,
                flight_t=flight_t,
                gravity_z=gravity_z[pending],
            )
            vel = self._clamp_launch_speed(vel)
            ang = self._compute_launch_spin(vel, spin_rps)
            pred_pos, _, net_cross_z = self._predict_ball_at_time(
                launch_pos, vel, ang, flight_t, gravity_z[pending]
            )
            strike_err = (pred_pos - strike_pos).norm(dim=-1)
            relaxed_strike_tol = max(self.launch_strike_tolerance * 4.0, 0.9)
            checks = {"strike_tolerance_relaxed": strike_err <= relaxed_strike_tol}
            quality_valid, quality_checks = self._launch_quality_checks(
                launch_pos=launch_pos,
                vel=vel,
                ang=ang,
                pred_pos=pred_pos,
                net_cross_z=net_cross_z,
                gravity_z=gravity_z[pending],
            )
            checks.update(quality_checks)
            valid = checks["strike_tolerance_relaxed"] & quality_valid
            if diagnostics is not None:
                diagnostics["extra_accepted"] += int(valid.sum().item())
                self._accumulate_failure_counts(diagnostics["extra_reasons"], valid, checks)

            if valid.any():
                accepted = pending[valid]
                launch_pos_all[accepted] = launch_pos[valid]
                launch_vel_all[accepted] = vel[valid]
                launch_ang_all[accepted] = ang[valid]
                target_all[accepted] = target_pos[valid]
            pending = pending[~valid]

        if diagnostics is not None:
            diagnostics["pending_final"] = int(pending.numel())
            self.last_sample_diagnostics = diagnostics

        if pending.numel() > 0:
            diag_msg = ""
            if diagnostics is not None:
                diag_msg = f" | diagnostics: {self.format_last_sample_diagnostics(top_k=8)}"
            raise RuntimeError(
                f"Failed to sample {int(pending.numel())} valid launches after "
                f"{self.launch_resample_attempts + self.launch_extra_resample_attempts} attempts. "
                f"Try relaxing launch ranges or speed constraints.{diag_msg}"
            )

        return launch_pos_all, launch_vel_all, launch_ang_all, target_all
