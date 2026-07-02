from __future__ import annotations

import math

import torch
from humanoid_tennis.utils.math import quat_apply, quat_apply_inverse


class HighLevelTennisStateMixin:
    def _effective_target_bounce_w(self) -> torch.Tensor:
        if hasattr(self, "guidance_target_bounce_w") and hasattr(self, "guidance_target_valid"):
            return torch.where(
                self.guidance_target_valid.unsqueeze(-1),
                self.guidance_target_bounce_w,
                self.target_bounce_w,
            )
        return self.target_bounce_w

    def step_schedule(self, progress: float, iters: int | None = None):
        _ = iters
        if not getattr(self, "replay_launch_enabled", False):
            self.replay_launch_mix_prob = 0.0
            return
        p = float(min(max(progress, 0.0), 1.0))
        p0 = float(self.replay_launch_mix_progress_start)
        p1 = float(self.replay_launch_mix_progress_end)
        if p <= p0:
            mix_prob = float(self.replay_launch_mix_prob_start)
        elif p >= p1:
            mix_prob = float(self.replay_launch_mix_prob_end)
        else:
            alpha = (p - p0) / max(p1 - p0, 1.0e-6)
            mix_prob = float(
                self.replay_launch_mix_prob_start
                + alpha * (self.replay_launch_mix_prob_end - self.replay_launch_mix_prob_start)
            )
        self.replay_launch_mix_prob = min(max(mix_prob, 0.0), 1.0)

    def _sample_relaunch_timing(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        n = int(env_ids.numel())

        li_min = int(self.relaunch_launch_interval_steps_min)
        li_max = int(self.relaunch_launch_interval_steps_max)
        if li_max <= li_min:
            launch_delay = torch.full((n,), li_min, dtype=torch.int32, device=self.device)
        else:
            launch_delay = torch.randint(li_min, li_max + 1, (n,), device=self.device, dtype=torch.int32)
        self._rally_launch_delay[env_ids] = launch_delay

        hold_min = int(self.relaunch_recovery_hold_steps_min)
        hold_max = int(self.relaunch_recovery_hold_steps_max)
        if hold_max <= hold_min:
            hold_steps = torch.full((n,), hold_min, dtype=torch.int32, device=self.device)
        else:
            hold_steps = torch.randint(hold_min, hold_max + 1, (n,), device=self.device, dtype=torch.int32)
        self._rally_recovery_hold_steps_target[env_ids] = hold_steps

        timeout_min = int(self.relaunch_recovery_timeout_steps_min)
        timeout_max = int(self.relaunch_recovery_timeout_steps_max)
        if timeout_max <= timeout_min:
            timeout_steps = torch.full((n,), timeout_min, dtype=torch.int32, device=self.device)
        else:
            timeout_steps = torch.randint(timeout_min, timeout_max + 1, (n,), device=self.device, dtype=torch.int32)
        self._rally_recovery_timeout_steps_target[env_ids] = torch.maximum(timeout_steps, hold_steps)

    def _sample_robot_spawn(self, env_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        env_origins = self.env.scene.env_origins[env_ids]
        n = env_ids.numel()
        pos_w = env_origins + self.robot_spawn_pos.unsqueeze(0)
        if (self.robot_spawn_xy_noise.abs() > 1.0e-8).any():
            noise_xy = (torch.rand((n, 2), device=self.device) * 2.0 - 1.0) * self.robot_spawn_xy_noise.unsqueeze(0)
            pos_w[:, :2] = pos_w[:, :2] + noise_xy

        yaw = torch.full((n,), self.robot_spawn_yaw, device=self.device, dtype=torch.float32)
        if self.robot_spawn_yaw_noise_rad > 1.0e-8:
            yaw_noise = (torch.rand((n,), device=self.device) * 2.0 - 1.0) * self.robot_spawn_yaw_noise_rad
            yaw = yaw + yaw_noise
        half = 0.5 * yaw
        quat_w = torch.stack(
            [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)],
            dim=-1,
        )
        return pos_w, quat_w

    def _set_stroke_mode_target_from_launch(
        self,
        *,
        env_ids: torch.Tensor,
        contact_ref_w: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ) -> None:
        if env_ids.numel() == 0:
            return
        target, contact_lateral = self._infer_stroke_mode_from_contact_ref(
            contact_ref_w=contact_ref_w,
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
        )
        self.stroke_mode_target[env_ids] = target
        self.stroke_mode_contact_lateral[env_ids, 0] = contact_lateral

    def _infer_stroke_mode_from_contact_ref(
        self,
        *,
        contact_ref_w: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contact_ref_b = quat_apply_inverse(root_quat_w, contact_ref_w - root_pos_w)
        contact_lateral = contact_ref_b[:, 1]
        deadzone = float(self.stroke_mode_lateral_deadzone)

        target = torch.full(
            (contact_ref_w.shape[0],),
            self.STROKE_MODE_NEUTRAL,
            dtype=torch.long,
            device=self.device,
        )
        forehand_mask = contact_lateral <= -deadzone
        backhand_mask = contact_lateral >= deadzone
        target[forehand_mask] = self.STROKE_MODE_FOREHAND
        target[backhand_mask] = self.STROKE_MODE_BACKHAND
        return target, contact_lateral

    def _infer_stroke_mode_from_launch_batch(
        self,
        *,
        launch_pos_w: torch.Tensor,
        launch_vel_w: torch.Tensor,
        target_bounce_w: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
        gravity_z: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        contact_pos_w, _, contact_valid, _, _ = self._incoming_contact_target(
            launch_pos_w,
            launch_vel_w,
            gravity_z=gravity_z,
        )
        contact_ref_w = torch.where(contact_valid.unsqueeze(-1), contact_pos_w, target_bounce_w)
        return self._infer_stroke_mode_from_contact_ref(
            contact_ref_w=contact_ref_w,
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
        )

    def _resample_launch_for_desired_mode(
        self,
        *,
        env_ids: torch.Tensor,
        launch_pos_w: torch.Tensor,
        launch_vel_w: torch.Tensor,
        launch_ang_w: torch.Tensor,
        target_bounce_w: torch.Tensor,
        sampled_level_ids: torch.Tensor,
        root_pos_w: torch.Tensor,
        root_quat_w: torch.Tensor,
        gravity_z: torch.Tensor,
        desired_mode: torch.Tensor | int,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        target_mode, contact_lateral = self._infer_stroke_mode_from_launch_batch(
            launch_pos_w=launch_pos_w,
            launch_vel_w=launch_vel_w,
            target_bounce_w=target_bounce_w,
            root_pos_w=root_pos_w,
            root_quat_w=root_quat_w,
            gravity_z=gravity_z,
        )
        if isinstance(desired_mode, torch.Tensor):
            desired_modes = desired_mode.to(device=self.device, dtype=torch.long)
        else:
            desired_modes = torch.full(
                (env_ids.numel(),),
                int(desired_mode),
                dtype=torch.long,
                device=self.device,
            )
        if desired_modes.shape[0] != env_ids.numel():
            raise ValueError(
                f"desired_mode shape mismatch: expected {env_ids.numel()}, got {desired_modes.shape}"
            )
        valid_desired = (desired_modes == self.STROKE_MODE_FOREHAND) | (
            desired_modes == self.STROKE_MODE_BACKHAND
        )
        need = valid_desired & (target_mode != desired_modes)
        rounds = int(self.balance_stroke_mode_max_resample_rounds)
        for _ in range(max(0, rounds)):
            if not need.any():
                break
            local_ids = need.nonzero(as_tuple=False).squeeze(-1)
            sub_env_ids = env_ids[local_ids]
            sub_launch_pos_w, sub_launch_vel_w, sub_launch_ang_w, sub_target_bounce_w, sub_level_ids = self._sample_ball_launch(
                sub_env_ids
            )
            sub_target_mode, sub_contact_lateral = self._infer_stroke_mode_from_launch_batch(
                launch_pos_w=sub_launch_pos_w,
                launch_vel_w=sub_launch_vel_w,
                target_bounce_w=sub_target_bounce_w,
                root_pos_w=root_pos_w[local_ids],
                root_quat_w=root_quat_w[local_ids],
                gravity_z=gravity_z[local_ids],
            )
            accept = sub_target_mode == desired_modes[local_ids]
            if not accept.any():
                continue
            accept_ids = local_ids[accept]
            launch_pos_w[accept_ids] = sub_launch_pos_w[accept]
            launch_vel_w[accept_ids] = sub_launch_vel_w[accept]
            launch_ang_w[accept_ids] = sub_launch_ang_w[accept]
            target_bounce_w[accept_ids] = sub_target_bounce_w[accept]
            sampled_level_ids[accept_ids] = sub_level_ids[accept]
            target_mode[accept_ids] = sub_target_mode[accept]
            contact_lateral[accept_ids] = sub_contact_lateral[accept]
            need = valid_desired & (target_mode != desired_modes)

        return (
            launch_pos_w,
            launch_vel_w,
            launch_ang_w,
            target_bounce_w,
            sampled_level_ids,
            target_mode,
            contact_lateral,
        )

    def _desired_stroke_mode_for_next_launch(self, env_ids: torch.Tensor) -> torch.Tensor:
        last_mode = self.last_hit_stroke_mode[env_ids]
        desired = torch.full_like(last_mode, self.STROKE_MODE_NEUTRAL)
        desired[last_mode == self.STROKE_MODE_FOREHAND] = self.STROKE_MODE_BACKHAND
        desired[last_mode == self.STROKE_MODE_BACKHAND] = self.STROKE_MODE_FOREHAND

        unknown = desired == self.STROKE_MODE_NEUTRAL
        if unknown.any():
            fallback = (
                self.STROKE_MODE_FOREHAND
                if int(self.hit_used_backhand_total) > int(self.hit_used_forehand_total)
                else self.STROKE_MODE_BACKHAND
            )
            desired[unknown] = int(fallback)
        return desired

    def _write_ball_launch(
        self,
        env_ids: torch.Tensor,
        *,
        root_pos_w: torch.Tensor | None = None,
        root_quat_w: torch.Tensor | None = None,
    ) -> None:
        if env_ids.numel() == 0:
            return
        if root_pos_w is None:
            root_pos_w = self.asset.data.root_link_pos_w[env_ids]
        if root_quat_w is None:
            root_quat_w = self.asset.data.root_link_quat_w[env_ids]
        gravity_z = self.gravity[env_ids]

        launch_pos_w, launch_vel_w, launch_ang_w, target_bounce_w, sampled_level_ids = self._sample_ball_launch(env_ids)
        desired_modes = torch.full(
            (env_ids.numel(),),
            self.STROKE_MODE_NEUTRAL,
            dtype=torch.long,
            device=self.device,
        )

        if bool(self.balance_stroke_mode_sampling):
            desired_modes = self._desired_stroke_mode_for_next_launch(env_ids)
            (
                launch_pos_w,
                launch_vel_w,
                launch_ang_w,
                target_bounce_w,
                sampled_level_ids,
                target_mode,
                contact_lateral,
            ) = self._resample_launch_for_desired_mode(
                env_ids=env_ids,
                launch_pos_w=launch_pos_w,
                launch_vel_w=launch_vel_w,
                launch_ang_w=launch_ang_w,
                target_bounce_w=target_bounce_w,
                sampled_level_ids=sampled_level_ids,
                root_pos_w=root_pos_w,
                root_quat_w=root_quat_w,
                gravity_z=gravity_z,
                desired_mode=desired_modes,
            )
        else:
            target_mode, contact_lateral = self._infer_stroke_mode_from_launch_batch(
                launch_pos_w=launch_pos_w,
                launch_vel_w=launch_vel_w,
                target_bounce_w=target_bounce_w,
                root_pos_w=root_pos_w,
                root_quat_w=root_quat_w,
                gravity_z=gravity_z,
            )

        ball_state = torch.zeros((env_ids.numel(), 13), device=self.device, dtype=torch.float32)
        ball_state[:, :3] = launch_pos_w
        ball_state[:, 3] = 1.0
        ball_state[:, 7:10] = launch_vel_w
        ball_state[:, 10:13] = launch_ang_w
        self.ball.write_root_state_to_sim(ball_state, env_ids=env_ids)
        self.target_bounce_w[env_ids] = target_bounce_w
        self.guidance_target_bounce_w[env_ids] = target_bounce_w
        self.guidance_target_valid[env_ids] = True
        self.launch_level_ids[env_ids] = sampled_level_ids
        self.ball_pos_w_history[env_ids] = launch_pos_w.unsqueeze(1)
        self.ball_vel_w_history[env_ids] = launch_vel_w.unsqueeze(1)
        launch_pos_l = launch_pos_w - self.env.scene.env_origins[env_ids]
        launch_net_dist = launch_pos_l[:, 1:2].abs()
        self.prev_net_dist[env_ids] = launch_net_dist
        self.net_dist[env_ids] = launch_net_dist
        self.net_dist_progress_buf[env_ids] = 0.0
        self.stroke_mode_target[env_ids] = target_mode
        self.stroke_mode_contact_lateral[env_ids, 0] = contact_lateral
        self.next_launch_desired_mode[env_ids] = desired_modes
        self.sampled_stroke_mode_forehand_total += int((target_mode == self.STROKE_MODE_FOREHAND).sum().item())
        self.sampled_stroke_mode_backhand_total += int((target_mode == self.STROKE_MODE_BACKHAND).sum().item())

    def _prime_ball_obs_history(self, env_ids: torch.Tensor) -> None:
        if env_ids.numel() == 0:
            return
        ball_pos_w = self.ball.data.root_link_pos_w[env_ids]
        ball_vel_w = self.ball.data.root_link_lin_vel_w[env_ids]
        self.ball_pos_w_history[env_ids] = ball_pos_w.unsqueeze(1)
        self.ball_vel_w_history[env_ids] = ball_vel_w.unsqueeze(1)

    def _push_ball_obs_history(self) -> None:
        self.ball_pos_w_history = self.ball_pos_w_history.roll(1, dims=1)
        self.ball_vel_w_history = self.ball_vel_w_history.roll(1, dims=1)
        self.ball_pos_w_history[:, 0] = self.ball.data.root_link_pos_w
        self.ball_vel_w_history[:, 0] = self.ball.data.root_link_lin_vel_w

    def _reset_rally_state(self, env_ids: torch.Tensor, *, reset_hit_counter: bool) -> None:
        if env_ids.numel() == 0:
            return
        self.task_step[env_ids] = 0
        self.finished[env_ids] = False
        self.success[env_ids] = False
        self.success_done[env_ids] = False
        self.timeout[env_ids] = False
        self.hit_limit_reached[env_ids] = False
        self.has_hit[env_ids] = False
        self.has_bounce[env_ids] = False
        self.has_pass_net[env_ids] = False
        self.pre_hit_bounce_count[env_ids] = 0
        self.pre_hit_ball_court_contact_latched[env_ids] = False
        self.bounce_in[env_ids] = False
        self.fail_miss[env_ids] = False
        self.fail_net[env_ids] = False
        self.fail_out[env_ids] = False
        self.fail_style[env_ids] = False
        self.fail_racket_body[env_ids] = False
        self.fail_second_bounce[env_ids] = False
        self.fail_recover_timeout[env_ids] = False
        self.first_hit_step[env_ids] = self.max_task_steps
        self.first_bounce_step[env_ids] = self.max_task_steps
        self.hit_event[env_ids] = False
        self.success_event[env_ids] = False
        self.bounce_event[env_ids] = False
        self.pre_hit_bounce_event[env_ids] = False
        self.pre_hit_second_bounce_event[env_ids] = False
        self.pass_net_event[env_ids] = False
        self.net_clearance_event[env_ids] = False
        self.recover_zone_outer_enter_event[env_ids] = False
        self.recover_zone_inner_enter_event[env_ids] = False
        self.recover_zone_outer[env_ids] = False
        self.recover_zone_inner[env_ids] = False
        self.recover_zone_inner_hold_steps[env_ids] = 0
        self.recover_zone_elapsed_steps[env_ids] = 0
        self.recover_zone_outer_entered[env_ids] = False
        self.recover_zone_inner_entered[env_ids] = False
        self.hit_stroke_mode_match_event[env_ids] = False
        self.hit_stroke_mode_mismatch_event[env_ids] = False
        self.hit_used_forehand_event[env_ids] = False
        self.stroke_style_violation_event[env_ids] = False
        self.prehit_zone[env_ids] = False
        self.prehit_zone_event[env_ids] = False
        self.prehit_zone_entered[env_ids] = False
        self.racket_ball_contact[env_ids] = False
        self.ball_net_contact[env_ids] = False
        self.ball_court_contact[env_ids] = False
        self.racket_body_contact[env_ids] = False
        self.racket_ball_contact_event[env_ids] = False
        self.ball_net_contact_event[env_ids] = False
        self.ball_court_contact_event[env_ids] = False
        self.racket_body_contact_event[env_ids] = False
        self.hit_cooldown[env_ids] = 0
        self.pre_hit_dead_ball_steps[env_ids] = 0
        self.post_hit_dead_ball_steps[env_ids] = 0
        self.hit_racket_speed[env_ids] = 0.0
        if self._last_hit_mask is not None:
            self._last_hit_mask[env_ids] = False
        self.launch_level_ids[env_ids] = -1
        self.bounce_pos_w[env_ids] = 0.0
        self.prev_ball_target_dist[env_ids] = 0.0
        self.ball_target_dist[env_ids] = 0.0
        self.ball_target_progress_buf[env_ids] = 0.0
        self.prev_net_dist[env_ids] = 0.0
        self.net_dist[env_ids] = 0.0
        self.net_dist_progress_buf[env_ids] = 0.0
        self._rally_relaunch_mask[env_ids] = False
        self._rally_launch_delay[env_ids] = 0
        self._rally_launch_ready[env_ids] = False
        self._rally_recovery_hold_steps_target[env_ids] = int(self.relaunch_recovery_hold_steps)
        self._rally_recovery_timeout_steps_target[env_ids] = int(self.relaunch_recovery_timeout_steps)
        self.replay_pending_valid[env_ids] = False
        self.replay_pending_pos_w[env_ids] = 0.0
        self.replay_pending_vel_w[env_ids] = 0.0
        self.replay_pending_ang_w[env_ids] = 0.0
        self.replay_pending_target_bounce_w[env_ids] = 0.0
        self.replay_pending_capture_age_substeps[env_ids] = 0
        self.guidance_target_bounce_w[env_ids] = 0.0
        self.guidance_target_valid[env_ids] = False
        self.racket_acc_norm[env_ids] = 0.0
        self.highlevel_action[env_ids] = 0.0
        self.correction_action[env_ids] = 0.0
        self.prev_correction_action[env_ids] = 0.0
        self.correction_action_rate[env_ids] = 0.0
        if reset_hit_counter:
            self.consecutive_return_count[env_ids] = 0

    def sample_init(self, env_ids: torch.Tensor):
        if env_ids.numel() == 0:
            return None

        root_state = self.init_root_state[env_ids].clone()
        spawn_pos_w, spawn_quat_w = self._sample_robot_spawn(env_ids)
        root_state[:, :3] = spawn_pos_w
        root_state[:, 3:7] = spawn_quat_w
        root_state[:, 7:13] = 0.0

        self.asset.write_root_state_to_sim(root_state, env_ids=env_ids)
        self.asset.write_joint_position_to_sim(self.root_default_joint_pos[env_ids], env_ids=env_ids)
        self.asset.write_joint_velocity_to_sim(self.root_default_joint_vel[env_ids], env_ids=env_ids)
        self.asset.set_joint_position_target(self.root_default_joint_pos[env_ids], env_ids=env_ids)
        self.spawn_root_pos_w[env_ids] = spawn_pos_w
        self.spawn_root_quat_w[env_ids] = spawn_quat_w
        spawn_forward_w = quat_apply(
            spawn_quat_w,
            self._forward_dir_b[env_ids],
        )
        self.spawn_root_forward_xy[env_ids] = spawn_forward_w[:, :2]
        # Fixed recovery target: no spawn noise.
        env_origins = self.env.scene.env_origins[env_ids]
        recover_pos_w = env_origins + self.robot_spawn_pos.unsqueeze(0)
        recover_yaw = torch.full(
            (env_ids.numel(),),
            self.robot_spawn_yaw,
            device=self.device,
            dtype=torch.float32,
        )
        half = 0.5 * recover_yaw
        recover_quat_w = torch.stack(
            [torch.cos(half), torch.zeros_like(half), torch.zeros_like(half), torch.sin(half)],
            dim=-1,
        )
        self.recover_root_pos_w[env_ids] = recover_pos_w
        self.recover_root_quat_w[env_ids] = recover_quat_w
        recover_forward_w = quat_apply(
            recover_quat_w,
            self._forward_dir_b[env_ids],
        )
        self.recover_root_forward_xy[env_ids] = recover_forward_w[:, :2]
        self._write_ball_launch(env_ids, root_pos_w=spawn_pos_w, root_quat_w=spawn_quat_w)
        self._prev_racket_vel_w[env_ids] = 0.0
        self.prev_correction_action[env_ids] = 0.0
        self.correction_action[env_ids] = 0.0
        self.correction_action_rate[env_ids] = 0.0
        self.highlevel_action[env_ids] = 0.0

        return None

    def reset(self, env_ids: torch.Tensor):
        self._reset_rally_state(env_ids, reset_hit_counter=True)
        self._prime_ball_obs_history(env_ids)
