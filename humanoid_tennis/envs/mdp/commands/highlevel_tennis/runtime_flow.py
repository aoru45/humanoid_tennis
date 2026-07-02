from __future__ import annotations

import math

import torch
from humanoid_tennis.utils.math import quat_apply


class HighLevelTennisRuntimeFlowMixin:
    def _compute_recovery_zone_state(self) -> tuple[torch.Tensor, torch.Tensor]:
        root_pos_w = self.asset.data.root_link_pos_w
        root_quat_w = self.asset.data.root_link_quat_w
        root_xy_err = (root_pos_w[:, :2] - self.recover_root_pos_w[:, :2]).norm(dim=-1)

        root_forward_w = quat_apply(root_quat_w, self._forward_dir_b)
        root_forward_xy = root_forward_w[:, :2]
        root_forward_xy = root_forward_xy / root_forward_xy.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
        target_forward_xy = self.recover_root_forward_xy / self.recover_root_forward_xy.norm(
            dim=-1, keepdim=True
        ).clamp_min(1.0e-6)
        heading_cos = (root_forward_xy * target_forward_xy).sum(dim=-1)

        outer = (root_xy_err <= self.recovery_outer_xy_radius) & (heading_cos >= self.recovery_outer_heading_cos)
        inner = (root_xy_err <= self.recovery_inner_xy_radius) & (heading_cos >= self.recovery_inner_heading_cos)
        return outer, inner

    def step(self, substep: int):
        if substep == 0:
            self._ensure_action_layout()
            self._capture_highlevel_action()

        ball_pos_w = self.ball.data.root_link_pos_w
        ball_vel_w = self.ball.data.root_link_lin_vel_w
        ball_ang_w = self.ball.data.root_link_ang_vel_w
        env_origins = self.env.scene.env_origins
        ball_pos_l = ball_pos_w - env_origins
        self._update_contact_events()
        bounce_contact_mask = self.ball_court_contact_event | (
            self.ball_court_contact & (ball_vel_w[:, 2] > 0.0) & (~self.pre_hit_ball_court_contact_latched)
        )
        pre_hit_bounce = bounce_contact_mask & (~self.has_hit) & (~self.finished)
        if pre_hit_bounce.any():
            self.pre_hit_bounce_count[pre_hit_bounce] += 1
            self.pre_hit_bounce_event[pre_hit_bounce] = True
            second_bounce = pre_hit_bounce & (self.pre_hit_bounce_count >= 2)
            if second_bounce.any():
                self.pre_hit_second_bounce_event[second_bounce] = True
                self.fail_second_bounce[second_bounce] = True
        self.pre_hit_ball_court_contact_latched[:] = self.ball_court_contact

        racket_pos_w, racket_vel_w = self._racket_state_w()
        racket_acc = (racket_vel_w - self._prev_racket_vel_w) / float(self.env.physics_dt)
        self.racket_acc_norm[:] = racket_acc.square().sum(dim=-1, keepdim=True)
        self._prev_racket_vel_w[:] = racket_vel_w

        hit_gate = (
            (~self.has_hit)
            & (~self.fail_second_bounce)
            & (~self.fail_miss)
            & (~self.fail_net)
            & (~self.fail_out)
            & (self.hit_cooldown <= 0)
        )
        hit_mask = hit_gate & (self.racket_ball_contact_event | self.racket_ball_contact)
        self._last_hit_mask = hit_mask

        if hit_mask.any():
            racket_speed = racket_vel_w[hit_mask].norm(dim=-1)
            self.has_hit[hit_mask] = True
            self.hit_event[hit_mask] = True
            self.hit_cooldown[hit_mask] = 12
            self.hit_racket_speed[hit_mask, 0] = racket_speed

            if self.replay_launch_enabled:
                effective_target_bounce_w = self._effective_target_bounce_w()
                self.replay_pending_valid[hit_mask] = True
                self.replay_pending_pos_w[hit_mask] = ball_pos_w[hit_mask]
                self.replay_pending_vel_w[hit_mask] = ball_vel_w[hit_mask]
                self.replay_pending_ang_w[hit_mask] = ball_ang_w[hit_mask]
                self.replay_pending_target_bounce_w[hit_mask] = effective_target_bounce_w[hit_mask]
                self.replay_pending_capture_age_substeps[hit_mask] = 0

            forehand_face_w, backhand_face_w = self._racket_face_dirs_w()
            incoming_dir_w = -ball_vel_w[hit_mask]
            incoming_dir_w = incoming_dir_w / incoming_dir_w.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
            fore_score = (forehand_face_w[hit_mask] * incoming_dir_w).sum(dim=-1)
            back_score = (backhand_face_w[hit_mask] * incoming_dir_w).sum(dim=-1)
            used_forehand = fore_score >= back_score
            target_mode = self.stroke_mode_target[hit_mask]
            target_forehand = target_mode == self.STROKE_MODE_FOREHAND
            target_backhand = target_mode == self.STROKE_MODE_BACKHAND
            stroke_match = (target_forehand & used_forehand) | (target_backhand & (~used_forehand))
            stroke_mismatch = (target_forehand & (~used_forehand)) | (target_backhand & used_forehand)
            self.hit_stroke_mode_match_event[hit_mask] = stroke_match
            self.hit_stroke_mode_mismatch_event[hit_mask] = stroke_mismatch
            self.hit_used_forehand_event[hit_mask] = used_forehand
            hit_env_ids = hit_mask.nonzero(as_tuple=False).squeeze(-1)
            if used_forehand.any():
                self.last_hit_stroke_mode[hit_env_ids[used_forehand]] = self.STROKE_MODE_FOREHAND
            backhand_used = ~used_forehand
            if backhand_used.any():
                self.last_hit_stroke_mode[hit_env_ids[backhand_used]] = self.STROKE_MODE_BACKHAND
            self.hit_used_forehand_total += int(used_forehand.sum().item())
            self.hit_used_backhand_total += int((~used_forehand).sum().item())
            if stroke_mismatch.any():
                # Stroke-side mismatch should never be counted as a valid scoring hit.
                mismatch_env_ids = hit_env_ids[stroke_mismatch]
                self.fail_style[mismatch_env_ids] = True
                self.stroke_style_violation_event[mismatch_env_ids] = True

            first_hit = hit_mask & (self.first_hit_step >= self.max_task_steps)
            if first_hit.any():
                self.first_hit_step[first_hit] = self.task_step[first_hit] + 1

            style_bad = hit_mask & (
                (racket_vel_w[:, 1] < self.stroke_style_min_forward_speed)
                | (racket_vel_w.norm(dim=-1) < self.stroke_style_min_racket_speed)
            )
            if style_bad.any():
                self.fail_style[style_bad] = True
                self.stroke_style_violation_event[style_bad] = True

        if self.replay_launch_enabled:
            # Lock one pre-net outgoing frame with sufficient forward speed.
            # If never reached, replay sample will be refused at commit time.
            track_mask = self.replay_pending_valid & self.has_hit & (~self.has_bounce)
            if track_mask.any():
                cur_vy = ball_vel_w[:, 1]
                v_min = float(self.replay_capture_min_forward_speed)
                min_substeps = int(self.replay_capture_min_post_hit_substeps)
                mature = self.replay_pending_capture_age_substeps >= min_substeps
                has_locked = self.replay_pending_vel_w[:, 1] > v_min
                pre_net = ball_pos_l[:, 1] < 0.0
                outgoing_mask = track_mask & mature & (~has_locked) & pre_net & (cur_vy > v_min)
                if outgoing_mask.any():
                    self.replay_pending_pos_w[outgoing_mask] = ball_pos_w[outgoing_mask]
                    self.replay_pending_vel_w[outgoing_mask] = ball_vel_w[outgoing_mask]
                    self.replay_pending_ang_w[outgoing_mask] = ball_ang_w[outgoing_mask]
                self.replay_pending_capture_age_substeps[track_mask] += 1

        post_hit_bounce_contact_mask = self.ball_court_contact_event | (
            self.ball_court_contact & (ball_vel_w[:, 2] > 0.0)
        )
        first_bounce = post_hit_bounce_contact_mask & self.has_hit & (~self.has_bounce)
        if first_bounce.any():
            self.has_bounce[first_bounce] = True
            self.bounce_event[first_bounce] = True
            first_bounce_step = first_bounce & (self.first_bounce_step >= self.max_task_steps)
            if first_bounce_step.any():
                self.first_bounce_step[first_bounce_step] = self.task_step[first_bounce_step] + 1
            self.bounce_pos_w[first_bounce] = ball_pos_w[first_bounce]
            bounce_pos_l = self.bounce_pos_w[first_bounce] - env_origins[first_bounce]
            x_in = bounce_pos_l[:, 0].abs() <= self.court_x_limit
            y_in = (
                (bounce_pos_l[:, 1] >= self.court_y_min_success)
                & (bounce_pos_l[:, 1] <= self.court_y_limit)
            )
            bounce_in = x_in & y_in
            self.bounce_in[first_bounce] = bounce_in
            if bool(self.fail_on_wrong_bounce_side):
                # First bounce must be in opponent court; otherwise fail immediately.
                wrong_bounce = first_bounce.clone()
                wrong_bounce[first_bounce] = ~bounce_in
                self.fail_out[wrong_bounce] = True

        net_mask = (
            self.has_hit
            & (~self.has_bounce)
            & (ball_pos_l[:, 1].abs() <= self.net_half_thickness)
            & (ball_pos_w[:, 2] <= self.net_height + 0.05)
            & (ball_vel_w[:, 1] > 0.0)
        )
        net_mask = net_mask | self.ball_net_contact_event
        if net_mask.any():
            self.fail_net[net_mask] = True

        speed = ball_vel_w.norm(dim=-1, keepdim=True).clamp_min(1e-6)
        spin_mag = ball_ang_w.norm(dim=-1, keepdim=True)
        spin_scaled = spin_mag / (2.0 * math.pi) * self.lift_spin_scale
        vel_dir = ball_vel_w / speed
        spin_axis = ball_ang_w / spin_mag.clamp_min(1e-6)
        cl = 1.0 / (2.0 + torch.abs(speed / (spin_scaled + 1e-6)))
        drag_coef = getattr(self, "drag_coef_env", self.drag_coef)
        drag_force = -self.aero_force_k * drag_coef * speed * ball_vel_w
        lift_dir = torch.cross(spin_axis, vel_dir, dim=-1)
        lift_force = self.aero_force_k * cl * speed.square() * lift_dir
        total_force = drag_force + lift_force
        spin_damping_torque = -self.spin_damping_coef * ball_ang_w
        total_force[self.finished] = 0.0
        spin_damping_torque[self.finished] = 0.0
        self.ball.write_external_wrench_to_sim(
            forces=total_force.unsqueeze(1),
            torques=spin_damping_torque.unsqueeze(1),
            body_ids=self.ball_body_ids,
        )
        pass_net_now = (
            self.has_hit
            & (~self.has_bounce)
            & (~self.has_pass_net)
            & (ball_vel_w[:, 1] > 0.0)
            & (ball_pos_l[:, 1] >= 0.0)
        )
        if pass_net_now.any():
            self.has_pass_net[pass_net_now] = True
            self.pass_net_event[pass_net_now] = True
            clear = ball_pos_w[pass_net_now, 2] > (self.net_height + self.net_clearance_reward_margin)
            if clear.any():
                pass_ids = pass_net_now.nonzero(as_tuple=False).squeeze(-1)
                self.net_clearance_event[pass_ids[clear]] = True
        self.hit_cooldown.sub_(1).clamp_min_(0)

    def before_update(self):
        # Refresh contact states from the latest post-step simulation buffers.
        # This avoids missing short-lived impacts that can occur during the
        # last physics substep of the previous frame.
        self._update_contact_events()
        self.task_step.add_(1)

        ball_pos_w = self.ball.data.root_link_pos_w
        ball_vel_w = self.ball.data.root_link_lin_vel_w
        root_pos_w = self.asset.data.root_link_pos_w
        ball_pos_l = ball_pos_w - self.env.scene.env_origins

        racket_pos_w, _ = self._racket_state_w()
        contact_pos_w, contact_t, contact_valid, _, _ = self._incoming_contact_target(ball_pos_w, ball_vel_w)
        zone_xy_err = (racket_pos_w[:, :2] - contact_pos_w[:, :2]).norm(dim=-1)
        zone_z_err = (racket_pos_w[:, 2] - contact_pos_w[:, 2]).abs()
        zone_now = (
            (~self.has_hit)
            & (~self.fail_second_bounce)
            & (~self.fail_miss)
            & (~self.fail_net)
            & (~self.fail_out)
            & contact_valid
            & (contact_t >= self.hitting_zone_time_min)
            & (contact_t <= self.hitting_zone_time_max)
            & (zone_xy_err <= self.hitting_zone_xy_radius)
            & (zone_z_err <= self.hitting_zone_z_tol)
        )
        self.prehit_zone_event[:] = zone_now & (~self.prehit_zone) & (~self.prehit_zone_entered)
        self.prehit_zone_entered |= self.prehit_zone_event
        self.prehit_zone[:] = zone_now

        effective_target_bounce_w = self._effective_target_bounce_w()
        self.ball_target_dist[:] = (ball_pos_w[:, :2] - effective_target_bounce_w[:, :2]).norm(dim=-1, keepdim=True)
        self.ball_target_progress_buf[:] = (self.prev_ball_target_dist - self.ball_target_dist).clamp(-1.0, 1.0)
        self.net_dist[:] = ball_pos_l[:, 1:2].abs()
        self.net_dist_progress_buf[:] = (self.prev_net_dist - self.net_dist).clamp(-1.0, 1.0)

        ball_speed = ball_vel_w.norm(dim=-1)
        ball_dropped = self.ball_court_contact | (
            ball_pos_l[:, 2] <= (self.ball_radius + self.pre_hit_dead_ball_height_margin)
        )
        pre_hit_dead_ball = (
            (~self.has_hit)
            & (self.task_step >= self.pre_hit_dead_ball_min_steps)
            & (ball_speed <= self.pre_hit_dead_ball_speed_thres)
            & ball_dropped
        )
        self.pre_hit_dead_ball_steps[pre_hit_dead_ball] += 1
        self.pre_hit_dead_ball_steps[~pre_hit_dead_ball] = 0
        self.fail_miss |= self.pre_hit_dead_ball_steps >= self.pre_hit_dead_ball_patience_steps

        post_hit_ball_dropped = self.ball_court_contact | (
            ball_pos_l[:, 2] <= (self.ball_radius + self.post_hit_dead_ball_height_margin)
        )
        hit_elapsed_steps = (self.task_step - self.first_hit_step).clamp_min(0)
        wrong_outgoing_dir = (
            self.has_hit
            & (~self.has_bounce)
            & (hit_elapsed_steps >= self.post_hit_forward_dir_grace_steps)
            & (ball_vel_w[:, 1] <= self.post_hit_min_forward_speed)
        )
        self.fail_out |= wrong_outgoing_dir
        post_hit_unresolved = self.has_hit & (~self.has_bounce) & (~self.success)
        vertical_stall_grace_steps = max(self.post_hit_forward_dir_grace_steps, self.post_hit_vertical_stall_min_steps)
        ball_xy_speed = ball_vel_w[:, :2].norm(dim=-1)
        post_hit_vertical_stall = (
            post_hit_unresolved
            & (hit_elapsed_steps >= vertical_stall_grace_steps)
            & (ball_pos_l[:, 2] >= self.post_hit_vertical_stall_min_z)
            & (ball_xy_speed <= self.post_hit_vertical_stall_max_xy_speed)
            & (ball_vel_w[:, 2] >= self.post_hit_vertical_stall_min_vz)
        )
        self.fail_out |= post_hit_vertical_stall
        post_hit_dead_ball = (
            self.has_hit
            & (hit_elapsed_steps >= self.post_hit_dead_ball_min_steps)
            & (ball_speed <= self.post_hit_dead_ball_speed_thres)
            & post_hit_ball_dropped
        )
        self.post_hit_dead_ball_steps[post_hit_dead_ball] += 1
        self.post_hit_dead_ball_steps[~post_hit_dead_ball] = 0
        post_hit_dead_ball_ready = self.post_hit_dead_ball_steps >= self.post_hit_dead_ball_patience_steps

        ball_behind_root = ball_pos_w[:, 1] < (root_pos_w[:, 1] - self.miss_margin_y)
        self.fail_miss |= (~self.has_hit) & ball_behind_root & ball_dropped
        post_hit_high_ball_out = (
            self.has_hit
            & (~self.has_bounce)
            & (hit_elapsed_steps >= self.post_hit_forward_dir_grace_steps)
            & (ball_pos_l[:, 2] > self.out_max_z)
        )
        fail_out_candidate = (
            (ball_pos_l[:, 2] < self.out_margin_z)
            | post_hit_high_ball_out
            | (ball_pos_l[:, 0].abs() > self.court_x_limit + 2.0)
            | (ball_pos_l[:, 1].abs() > self.court_y_limit + 4.0)
        )
        # Style-invalid rallies (e.g., forehand target hit with backhand face) cannot score.
        self.success[:] = self.has_bounce & self.bounce_in & (~self.fail_style)
        self.fail_out |= (~self.success) & fail_out_candidate

        success_done = self.success
        self.fail_miss |= self.has_hit & (~self.success) & post_hit_dead_ball_ready

        new_success_event = success_done & (~self.success_done)
        self.success_event[:] = new_success_event
        if new_success_event.any():
            self.consecutive_return_count[new_success_event] += 1
            if self.replay_launch_enabled:
                self._record_success_replay_launches(
                    new_success_event.nonzero(as_tuple=False).squeeze(-1)
                )
        if self.consecutive_return_count.numel() > 0:
            cur_best = int(self.consecutive_return_count.max().item())
            if cur_best > int(self.best_consecutive_return_total):
                self.best_consecutive_return_total = cur_best

        timeout = self.task_step >= self.max_task_steps
        self.fail_racket_body |= self.racket_body_contact
        self.hit_limit_reached[:] = False
        if self.max_consecutive_returns_before_finish > 0:
            self.hit_limit_reached[:] = (
                self.consecutive_return_count >= self.max_consecutive_returns_before_finish
            )

        if self.relaunch_on_success:
            new_relaunch = success_done & (~self.hit_limit_reached)
            newly_set = new_relaunch & (~self._rally_relaunch_mask)
            self._rally_relaunch_mask[:] = self._rally_relaunch_mask | new_relaunch
            if newly_set.any():
                newly_set_ids = newly_set.nonzero(as_tuple=False).squeeze(-1)
                self._sample_relaunch_timing(newly_set_ids)
                self.recover_zone_inner_hold_steps[newly_set] = 0
                self.recover_zone_elapsed_steps[newly_set] = 0
                self.recover_zone_outer_entered[newly_set] = False
                self.recover_zone_inner_entered[newly_set] = False
                self.recover_zone_outer_enter_event[newly_set] = False
                self.recover_zone_inner_enter_event[newly_set] = False
            waiting_mask = self._rally_relaunch_mask
            if waiting_mask.any():
                self.recover_zone_elapsed_steps[waiting_mask] += 1
                recover_outer_now, recover_inner_now = self._compute_recovery_zone_state()
                self.recover_zone_outer[:] = waiting_mask & recover_outer_now
                self.recover_zone_inner[:] = waiting_mask & recover_inner_now
                self.recover_zone_outer_enter_event[:] = (
                    waiting_mask & recover_outer_now & (~self.recover_zone_outer_entered)
                )
                self.recover_zone_inner_enter_event[:] = (
                    waiting_mask & recover_inner_now & (~self.recover_zone_inner_entered)
                )
                self.recover_zone_outer_entered |= self.recover_zone_outer_enter_event
                self.recover_zone_inner_entered |= self.recover_zone_inner_enter_event
                hold_now = waiting_mask & recover_inner_now
                self.recover_zone_inner_hold_steps[hold_now] += 1
                self.recover_zone_inner_hold_steps[waiting_mask & (~recover_inner_now)] = 0
            else:
                self.recover_zone_outer[:] = False
                self.recover_zone_inner[:] = False
                self.recover_zone_outer_enter_event[:] = False
                self.recover_zone_inner_enter_event[:] = False
            waiting = self._rally_relaunch_mask & (self._rally_launch_delay > 0)
            if waiting.any():
                self._rally_launch_delay[waiting] -= 1
            if self.relaunch_require_recovery:
                recover_ready = self.recover_zone_inner_hold_steps >= self._rally_recovery_hold_steps_target
                recover_timeout = self.recover_zone_elapsed_steps >= self._rally_recovery_timeout_steps_target
                timeout_fail = (
                    self._rally_relaunch_mask
                    & (self._rally_launch_delay <= 0)
                    & recover_timeout
                    & (~recover_ready)
                )
                if timeout_fail.any():
                    # Timeout only applies a penalty; do not fail/reset the rally.
                    self.fail_recover_timeout[timeout_fail] = True
                self._rally_launch_ready[:] = (
                    self._rally_relaunch_mask
                    & (self._rally_launch_delay <= 0)
                    & (recover_ready | recover_timeout)
                )
            else:
                self._rally_launch_ready[:] = self._rally_relaunch_mask & (self._rally_launch_delay <= 0)
            success_finish = success_done & self.hit_limit_reached
        else:
            self._rally_relaunch_mask[:] = False
            self._rally_launch_ready[:] = False
            self.recover_zone_outer[:] = False
            self.recover_zone_inner[:] = False
            self.recover_zone_outer_enter_event[:] = False
            self.recover_zone_inner_enter_event[:] = False
            success_finish = success_done

        success_done = self.success_done | self.success
        fail_any = (
            self.fail_miss
            | self.fail_net
            | self.fail_out
            | self.fail_style
            | self.fail_racket_body
            | self.fail_second_bounce
        )
        # Rally failure event should be counted once (before finished flips to True).
        new_fail_event = (~self.finished) & (~success_done) & (fail_any | timeout)
        if self.replay_launch_enabled and new_fail_event.any():
            self.replay_pending_valid[new_fail_event] = False
            self.replay_pending_capture_age_substeps[new_fail_event] = 0

        # Curriculum update uses resolved launch outcomes:
        # success: first legal bounce in opponent court; failure: miss/net/out/style.
        rally_resolved = new_success_event | new_fail_event
        if rally_resolved.any():
            level_ids = self.launch_level_ids[rally_resolved]
            rally_success = new_success_event[rally_resolved]
            self.launch_bank.update_curriculum(level_ids=level_ids, success=rally_success)

        self.success_done[:] = success_done
        self.timeout[:] = timeout
        self.finished[:] = self.hit_limit_reached | success_finish | fail_any | timeout
        self._update_live_debug_metrics(
            ball_pos_w,
            ball_vel_w,
            hit_mask=self._last_hit_mask,
        )

    def update(self):
        self._push_ball_obs_history()
        self.prev_ball_target_dist[:] = self.ball_target_dist
        self.prev_net_dist[:] = self.net_dist
        relaunch_ids = self._rally_launch_ready.nonzero(as_tuple=False).squeeze(-1)
        if relaunch_ids.numel() > 0:
            self._reset_rally_state(relaunch_ids, reset_hit_counter=False)
            self._write_ball_launch(relaunch_ids)
            self._rally_relaunch_mask[relaunch_ids] = False
            self._rally_launch_ready[relaunch_ids] = False
            self._rally_launch_delay[relaunch_ids] = 0
        self.hit_event[:] = False
        self.success_event[:] = False
        self.bounce_event[:] = False
        self.pre_hit_bounce_event[:] = False
        self.pre_hit_second_bounce_event[:] = False
        self.pass_net_event[:] = False
        self.net_clearance_event[:] = False
        self.recover_zone_outer_enter_event[:] = False
        self.recover_zone_inner_enter_event[:] = False
        self.stroke_style_violation_event[:] = False
        self.prehit_zone_event[:] = False
        self.racket_ball_contact_event[:] = False
        self.ball_net_contact_event[:] = False
        self.ball_court_contact_event[:] = False
        self.racket_body_contact_event[:] = False
        self.hit_stroke_mode_match_event[:] = False
        self.hit_stroke_mode_mismatch_event[:] = False
        self.hit_used_forehand_event[:] = False
        self.hit_racket_speed[:] = 0.0
