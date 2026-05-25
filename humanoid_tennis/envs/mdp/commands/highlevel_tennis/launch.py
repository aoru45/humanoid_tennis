from __future__ import annotations

import torch


class HighLevelTennisLaunchMixin:
    def _predict_first_bounce_ballistic(
        self,
        launch_pos: torch.Tensor,
        vel: torch.Tensor,
        gravity_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Predict first ground bounce using ballistic motion (no drag/lift)."""
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

    def _sample_replay_ball_launch(
        self,
        *,
        env_origins: torch.Tensor,
        num_samples: int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        if (not self.replay_launch_enabled) or (self.replay_launch_count <= 0):
            raise RuntimeError("Replay launch buffer is empty or disabled.")
        count = int(self.replay_launch_count)
        idx = torch.randint(0, count, (num_samples,), device=self.device)
        launch_pos_w = self.replay_launch_pos_local[idx] + env_origins
        launch_vel_w = self.replay_launch_vel[idx]
        launch_ang_w = self.replay_launch_ang[idx]
        target_bounce_w = self.replay_launch_target_local[idx] + env_origins
        return launch_pos_w, launch_vel_w, launch_ang_w, target_bounce_w

    def _record_success_replay_launches(self, env_ids: torch.Tensor) -> None:
        if (not self.replay_launch_enabled) or env_ids.numel() == 0:
            return
        valid = self.replay_pending_valid[env_ids]
        if not valid.any():
            return
        sample_ids = env_ids[valid]
        env_origins = self.env.scene.env_origins[sample_ids]

        pos_local = self.replay_pending_pos_w[sample_ids] - env_origins
        vel = self.replay_pending_vel_w[sample_ids].clone()
        ang = self.replay_pending_ang_w[sample_ids].clone()
        target_local = self.replay_pending_target_bounce_w[sample_ids] - env_origins
        strong_outgoing = vel[:, 1] > float(self.replay_capture_min_forward_speed)

        # Mirror outgoing-ball state into incoming-ball frame.
        pos_local[:, 1] = -pos_local[:, 1]
        vel[:, 1] = -vel[:, 1]
        ang[:, 0] = -ang[:, 0]
        ang[:, 2] = -ang[:, 2]

        # Success-only replay collection: keep all successful samples.
        # Only drop non-finite values to prevent buffer corruption.
        keep = (
            torch.isfinite(pos_local).all(dim=-1)
            & torch.isfinite(vel).all(dim=-1)
            & torch.isfinite(ang).all(dim=-1)
            & torch.isfinite(target_local).all(dim=-1)
            & strong_outgoing
        )

        num_total = int(sample_ids.numel())
        num_keep = int(keep.sum().item())
        self.replay_rejected_total += int(num_total - num_keep)

        # Clear pending slots for resolved rallies.
        self.replay_pending_valid[sample_ids] = False
        self.replay_pending_capture_age_substeps[sample_ids] = 0

        if num_keep <= 0:
            return

        pos_local = pos_local[keep]
        vel = vel[keep]
        ang = ang[keep]
        target_local = target_local[keep]

        cap = int(self.replay_launch_capacity)
        ptr = int(self.replay_launch_ptr)
        count = int(self.replay_launch_count)
        n = int(num_keep)

        if n >= cap:
            self.replay_launch_pos_local[:] = pos_local[-cap:]
            self.replay_launch_vel[:] = vel[-cap:]
            self.replay_launch_ang[:] = ang[-cap:]
            self.replay_launch_target_local[:] = target_local[-cap:]
            self.replay_launch_ptr = 0
            self.replay_launch_count = cap
            self.replay_added_total += n
            return

        first = min(cap - ptr, n)
        self.replay_launch_pos_local[ptr : ptr + first] = pos_local[:first]
        self.replay_launch_vel[ptr : ptr + first] = vel[:first]
        self.replay_launch_ang[ptr : ptr + first] = ang[:first]
        self.replay_launch_target_local[ptr : ptr + first] = target_local[:first]

        remain = n - first
        if remain > 0:
            self.replay_launch_pos_local[:remain] = pos_local[first:]
            self.replay_launch_vel[:remain] = vel[first:]
            self.replay_launch_ang[:remain] = ang[first:]
            self.replay_launch_target_local[:remain] = target_local[first:]

        self.replay_launch_ptr = (ptr + n) % cap
        self.replay_launch_count = min(cap, count + n)
        self.replay_added_total += n

    def _sample_ball_launch(
        self,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample launches from offline bank, with optional replay-buffer mixing."""
        env_origins = self.env.scene.env_origins[env_ids]
        launch_pos_w, launch_vel_w, launch_ang_w, target_bounce_w = self.launch_bank.sample(
            env_origins=env_origins, num_samples=env_ids.numel()
        )
        sampled_level_ids = self.launch_bank.get_last_sample_level_ids()
        if sampled_level_ids is None or sampled_level_ids.numel() != env_ids.numel():
            sampled_level_ids = torch.zeros((env_ids.numel(),), dtype=torch.long, device=self.device)

        n = int(env_ids.numel())
        self.replay_sample_requested_last = n
        self.replay_sampled_last_count = 0
        can_mix = (
            self.replay_launch_enabled
            and (self.replay_launch_count >= self.replay_launch_min_size_to_sample)
            and (self.replay_launch_mix_prob > 1.0e-6)
        )
        if can_mix and n > 0:
            mix_mask = torch.rand((n,), device=self.device) < float(self.replay_launch_mix_prob)
            if mix_mask.any():
                mix_ids = mix_mask.nonzero(as_tuple=False).squeeze(-1)
                mix_n = int(mix_ids.numel())
                mix_pos_w, mix_vel_w, mix_ang_w, mix_target_w = self._sample_replay_ball_launch(
                    env_origins=env_origins[mix_ids],
                    num_samples=mix_n,
                )
                launch_pos_w[mix_ids] = mix_pos_w
                launch_vel_w[mix_ids] = mix_vel_w
                launch_ang_w[mix_ids] = mix_ang_w
                target_bounce_w[mix_ids] = mix_target_w
                sampled_level_ids[mix_ids] = int(self.REPLAY_LAUNCH_LEVEL_ID)
                self.replay_sampled_last_count = mix_n
                self.replay_sampled_total += mix_n

        return launch_pos_w, launch_vel_w, launch_ang_w, target_bounce_w, sampled_level_ids
