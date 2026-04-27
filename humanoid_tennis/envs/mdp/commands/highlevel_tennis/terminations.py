from __future__ import annotations

from humanoid_tennis.envs.mdp import termination


class HighLevelTennisTerminationMixin:
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

    @termination
    def racket_body_contact_termination(self):
        return self.fail_racket_body.unsqueeze(-1)

    def debug_draw(self):
        if not self.debug_draw_enabled:
            return
        ball_pos_w = self.ball.data.root_link_pos_w
        self.env.debug_draw.point(ball_pos_w, color=(1.0, 1.0, 0.1, 1.0), size=18.0)
        self.env.debug_draw.point(self.target_bounce_w, color=(0.1, 0.8, 1.0, 1.0), size=10.0)
        racket_pos_w, _ = self._racket_state_w()
        self.env.debug_draw.point(racket_pos_w, color=(1.0, 0.5, 0.1, 1.0), size=10.0)
