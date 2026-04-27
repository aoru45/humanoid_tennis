from __future__ import annotations

import torch

from humanoid_tennis.envs.mdp import observation
from humanoid_tennis.utils import symmetry as sym_utils
from humanoid_tennis.utils.math import quat_apply, quat_apply_inverse


class HighLevelTennisObservationMixin:
    @observation
    def ball_task_obs(self):
        root_pos_w = self.asset.data.root_link_pos_w
        root_quat_w = self.asset.data.root_link_quat_w
        root_vel_w = self.asset.data.root_link_lin_vel_w
        env_origins = self.env.scene.env_origins
        ball_pos_w = self.ball.data.root_link_pos_w
        ball_vel_w = self.ball.data.root_link_lin_vel_w

        racket_pos_w, racket_vel_w = self._racket_state_w()
        ball_pred_obs = self._predict_ball_obs_features(
            ball_pos_w=ball_pos_w,
            ball_vel_w=ball_vel_w,
            racket_pos_w=racket_pos_w,
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
        )

        root_pos_l = root_pos_w - env_origins
        root_forward_w = quat_apply(root_quat_w, self._forward_dir_b)
        target_bounce_l = self.target_bounce_w - env_origins

        racket_pos_b = quat_apply_inverse(root_quat_w, racket_pos_w - root_pos_w)
        racket_vel_b = quat_apply_inverse(root_quat_w, racket_vel_w - root_vel_w)
        target_bounce_b = quat_apply_inverse(root_quat_w, self.target_bounce_w - root_pos_w)
        hist_ids = list(self.ball_obs_history_steps)
        ball_pos_hist_w = self.ball_pos_w_history[:, hist_ids]
        ball_vel_hist_w = self.ball_vel_w_history[:, hist_ids]
        h = len(hist_ids)
        root_quat_rep = root_quat_w.unsqueeze(1).expand(-1, h, -1).reshape(-1, 4)
        root_pos_rep = root_pos_w.unsqueeze(1).expand(-1, h, -1).reshape(-1, 3)
        root_vel_rep = root_vel_w.unsqueeze(1).expand(-1, h, -1).reshape(-1, 3)
        ball_pos_hist_b = quat_apply_inverse(
            root_quat_rep,
            (ball_pos_hist_w.reshape(-1, 3) - root_pos_rep),
        ).reshape(self.num_envs, -1)
        ball_vel_hist_b = quat_apply_inverse(
            root_quat_rep,
            (ball_vel_hist_w.reshape(-1, 3) - root_vel_rep),
        ).reshape(self.num_envs, -1)

        flags = torch.stack(
            [
                self.has_hit.float(),
                self.has_bounce.float(),
                self.bounce_in.float(),
                self.fail_miss.float(),
                self.fail_net.float(),
                self.fail_out.float(),
                self.fail_style.float(),
                self.success.float(),
                self.task_step.float() / float(self.max_task_steps),
            ],
            dim=-1,
        )
        return torch.cat(
            [
                root_pos_l[:, :2],
                root_forward_w[:, :2],
                ball_pos_hist_b,
                ball_vel_hist_b,
                target_bounce_l,
                racket_pos_b,
                racket_vel_b,
                target_bounce_b,
                ball_pred_obs,
                flags,
            ],
            dim=-1,
        )

    def ball_task_obs_sym(self):
        dim = 35 + 6 * len(self.ball_obs_history_steps)
        return sym_utils.SymmetryTransform(
            perm=torch.arange(dim, device=self.device),
            signs=[1.0] * dim,
        )

    @observation
    def ball_priv_obs(self):
        env_origins = self.env.scene.env_origins
        ball_pos_w = self.ball.data.root_link_pos_w - env_origins
        ball_vel_w = self.ball.data.root_link_lin_vel_w
        root_pos_w = self.asset.data.root_link_pos_w - env_origins
        root_quat_w = self.asset.data.root_link_quat_w
        root_forward_w = quat_apply(root_quat_w, self._forward_dir_b)
        racket_pos_w, racket_vel_w = self._racket_state_w()
        racket_pos_w = racket_pos_w - env_origins
        target_bounce_w = self.target_bounce_w - env_origins
        return torch.cat(
            [
                ball_pos_w,
                ball_vel_w,
                racket_pos_w,
                racket_vel_w,
                target_bounce_w,
                root_pos_w[:, :2],
                root_forward_w[:, :2],
            ],
            dim=-1,
        )

    def ball_priv_obs_sym(self):
        dim = 19
        return sym_utils.SymmetryTransform(
            perm=torch.arange(dim, device=self.device),
            signs=[1.0] * dim,
        )
