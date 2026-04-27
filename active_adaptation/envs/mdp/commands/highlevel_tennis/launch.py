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

    def _sample_ball_launch(
        self,
        env_ids: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Sample launches strictly from offline launch bank."""
        env_origins = self.env.scene.env_origins[env_ids]
        launch_pos_w, launch_vel_w, launch_ang_w, target_bounce_w = self.launch_bank.sample(
            env_origins=env_origins, num_samples=env_ids.numel()
        )
        sampled_level_ids = self.launch_bank.get_last_sample_level_ids()
        if sampled_level_ids is None or sampled_level_ids.numel() != env_ids.numel():
            sampled_level_ids = torch.zeros((env_ids.numel(),), dtype=torch.long, device=self.device)
        return launch_pos_w, launch_vel_w, launch_ang_w, target_bounce_w, sampled_level_ids
