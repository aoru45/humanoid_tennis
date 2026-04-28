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
        self.env.extra["highlevel/live_fail_recover_timeout_ratio"] = float(
            self.fail_recover_timeout.float().mean().item()
        )
        self.env.extra["highlevel/live_racket_body_contact_ratio"] = float(
            self.racket_body_contact.float().mean().item()
        )
        self.env.extra["highlevel/live_recover_zone_outer_ratio"] = float(
            self.recover_zone_outer.float().mean().item()
        )
        self.env.extra["highlevel/live_recover_zone_inner_ratio"] = float(
            self.recover_zone_inner.float().mean().item()
        )
        self.env.extra["highlevel/live_recover_hold_mean"] = float(
            self.recover_zone_inner_hold_steps.float().mean().item()
        )
        self.env.extra["highlevel/live_success_event_ratio"] = float(self.success_event.float().mean().item())
        self.env.extra["highlevel/live_target_forehand_ratio"] = float(
            (self.stroke_mode_target == self.STROKE_MODE_FOREHAND).float().mean().item()
        )
        self.env.extra["highlevel/live_target_backhand_ratio"] = float(
            (self.stroke_mode_target == self.STROKE_MODE_BACKHAND).float().mean().item()
        )
        self.env.extra["highlevel/hit_used_forehand_total"] = float(int(self.hit_used_forehand_total))
        self.env.extra["highlevel/hit_used_backhand_total"] = float(int(self.hit_used_backhand_total))
        consec = self.consecutive_return_count.float()
        self.env.extra["highlevel/live_consecutive_return_mean"] = float(consec.mean().item())
        self.env.extra["highlevel/live_consecutive_return_max"] = float(consec.max().item())
        self.env.extra["highlevel/live_consecutive_return_p95"] = float(
            torch.quantile(consec, 0.95).item()
        )
        self.env.extra["highlevel/best_consecutive_return_total"] = float(
            int(self.best_consecutive_return_total)
        )
        self.env.extra["highlevel/sampling_desired_mode_forehand"] = float(
            (self.next_launch_desired_mode == self.STROKE_MODE_FOREHAND).float().mean().item()
        )
        self.env.extra["highlevel/sampling_desired_mode_backhand"] = float(
            (self.next_launch_desired_mode == self.STROKE_MODE_BACKHAND).float().mean().item()
        )
        hit_count = int(self.hit_event.sum().item())
        if hit_count > 0:
            forehand_hits = int((self.hit_event & self.hit_used_forehand_event).sum().item())
            backhand_hits = hit_count - forehand_hits
            self.env.extra["highlevel/live_hit_used_forehand_ratio"] = float(forehand_hits / max(hit_count, 1))
            self.env.extra["highlevel/live_hit_used_backhand_ratio"] = float(backhand_hits / max(hit_count, 1))
            self.env.extra["highlevel/live_hit_mode_match_ratio"] = float(
                self.hit_stroke_mode_match_event[self.hit_event].float().mean().item()
            )
            self.env.extra["highlevel/live_hit_mode_mismatch_ratio"] = float(
                self.hit_stroke_mode_mismatch_event[self.hit_event].float().mean().item()
            )
        else:
            self.env.extra["highlevel/live_hit_used_forehand_ratio"] = 0.0
            self.env.extra["highlevel/live_hit_used_backhand_ratio"] = 0.0
            self.env.extra["highlevel/live_hit_mode_match_ratio"] = 0.0
            self.env.extra["highlevel/live_hit_mode_mismatch_ratio"] = 0.0
        self.env.extra["highlevel/live_ball_height_l_mean"] = float(ball_pos_l[:, 2].mean().item())
        self.env.extra["highlevel/live_ball_speed_mean"] = float(ball_speed.mean().item())
        curriculum_state = self.launch_bank.get_curriculum_state()
        if len(curriculum_state) > 0:
            for k, v in curriculum_state.items():
                self.env.extra[f"highlevel/curriculum_{k}"] = float(v)
        if self.debug_draw_enabled and hit_mask is not None:
            self.env.extra["highlevel/live_hit_trigger_ratio"] = float(hit_mask.float().mean().item())
