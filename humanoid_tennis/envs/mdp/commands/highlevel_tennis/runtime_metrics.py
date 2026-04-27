from __future__ import annotations

import torch


class HighLevelTennisRuntimeMetricsMixin:
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
        self.env.extra["highlevel/live_success_done_ratio"] = float(self.success_done.float().mean().item())
        self.env.extra["highlevel/live_success_ratio"] = float(self.success.float().mean().item())
        self.env.extra["highlevel/live_has_hit_ratio"] = float(self.has_hit.float().mean().item())
        self.env.extra["highlevel/live_has_bounce_ratio"] = float(self.has_bounce.float().mean().item())
        self.env.extra["highlevel/live_pass_net_ratio"] = float(self.has_pass_net.float().mean().item())
        self.env.extra["highlevel/live_fail_miss_ratio"] = float(self.fail_miss.float().mean().item())
        self.env.extra["highlevel/live_fail_net_ratio"] = float(self.fail_net.float().mean().item())
        self.env.extra["highlevel/live_fail_out_ratio"] = float(self.fail_out.float().mean().item())
        self.env.extra["highlevel/live_fail_racket_body_ratio"] = float(self.fail_racket_body.float().mean().item())
        self.env.extra["highlevel/live_racket_body_contact_ratio"] = float(
            self.racket_body_contact.float().mean().item()
        )
        self.env.extra["highlevel/live_ball_height_l_mean"] = float(ball_pos_l[:, 2].mean().item())
        self.env.extra["highlevel/live_ball_speed_mean"] = float(ball_speed.mean().item())
        curriculum_state = self.launch_bank.get_curriculum_state()
        if len(curriculum_state) > 0:
            for k, v in curriculum_state.items():
                self.env.extra[f"highlevel/curriculum_{k}"] = float(v)
        if self.debug_draw_enabled and hit_mask is not None:
            self.env.extra["highlevel/live_hit_trigger_ratio"] = float(hit_mask.float().mean().item())
