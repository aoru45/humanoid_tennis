import json
import time
from abc import ABC
from pathlib import Path
from typing import TYPE_CHECKING, Dict, Optional

import numpy as np
from scipy.spatial.transform import Rotation as R, Slerp

from common.math_utils import _linspace_rows, _slerp, _yaw_component_wxyz
from common.utils import DictToClass, MotionUDPServer
from paths import REAL_G1_ROOT

try:
    import zmq
except Exception:
    zmq = None

if TYPE_CHECKING:
    from policy import TrackingPolicyRaw


def remap_joint_array_by_names(
    data: np.ndarray,
    source_joint_names,
    target_joint_names,
) -> np.ndarray:
    data = np.asarray(data, dtype=np.float32)
    if data.ndim != 2:
        raise ValueError(f"Expected 2D joint array [T, J], got shape={data.shape}")
    if data.shape[1] != len(source_joint_names):
        raise ValueError(
            f"Joint dim mismatch: data has {data.shape[1]} dims, "
            f"but source_joint_names has {len(source_joint_names)} names."
        )

    name_to_idx = {name: i for i, name in enumerate(source_joint_names)}
    remap = np.zeros((data.shape[0], len(target_joint_names)), dtype=np.float32)
    for i, name in enumerate(target_joint_names):
        j = name_to_idx.get(name, None)
        if j is not None:
            remap[:, i] = data[:, j]
    return remap


class MotionSourceBase(ABC):
    def __init__(self, policy: "TrackingPolicyRaw", policy_cfg: DictToClass):
        self.policy = policy
        self.config = policy_cfg
        self.motions: Dict[str, Dict[str, np.ndarray]] = self._load_motions()

    def _load_motions(self) -> Dict[str, Dict[str, np.ndarray]]:
        motions: Dict[str, Dict[str, np.ndarray]] = {}

        for m in getattr(self.config, "motions", []):
            mc = DictToClass(m)
            motion_name = mc.name
            mp = Path(mc.path)
            path = str(mp if mp.is_absolute() else (REAL_G1_ROOT / mp))
            t0, t1 = int(mc.start), int(mc.end)

            data = np.load(path, allow_pickle=True)
            if not isinstance(data, np.lib.npyio.NpzFile):
                raise ValueError(f"[{self.__class__.__name__}] Only .npz is supported: {path}")

            joint_pos = data["dof_pos"][t0:t1].astype(np.float32)
            root_pos = data["root_pos"][t0:t1].astype(np.float32)
            root_rot_xyzw = data["root_rot"][t0:t1].astype(np.float32)
            root_quat = np.concatenate([root_rot_xyzw[:, 3:4], root_rot_xyzw[:, :3]], axis=-1)

            joint_names = data.get("joint_names", None)
            if joint_names is None:
                raise ValueError(
                    f"[{self.__class__.__name__}] Motion '{motion_name}' is missing 'joint_names' in npz. "
                    "Please export joint_names with the dataset."
                )
            source_joint_names = []
            for n in joint_names.tolist():
                if isinstance(n, (bytes, np.bytes_)):
                    source_joint_names.append(n.decode("utf-8"))
                else:
                    source_joint_names.append(str(n))
            joint_pos = remap_joint_array_by_names(joint_pos, source_joint_names, self.policy.obs_joint_names)

            motions[motion_name] = {
                "joint_pos": joint_pos,
                "root_quat": root_quat,
                "root_pos": root_pos,
            }

        for m in getattr(self.config, "motion_clips", []):
            mc = DictToClass(m)
            motion_name = mc.name
            joint_pos_1 = np.asarray(mc.joint_pos, dtype=np.float32).reshape(1, -1)
            if joint_pos_1.shape[1] != len(self.policy.dataset_joint_names):
                raise ValueError(
                    f"[{self.__class__.__name__}] Motion clip '{motion_name}' dim={joint_pos_1.shape[1]} "
                    f"does not match dataset_joint_names size={len(self.policy.dataset_joint_names)}."
                )
            source_joint_names = self.policy.dataset_joint_names
            joint_pos_1 = remap_joint_array_by_names(joint_pos_1, source_joint_names, self.policy.obs_joint_names)
            root_quat_1 = np.asarray(mc.root_quat, dtype=np.float32).reshape(1, 4)
            root_pos_1 = np.asarray(mc.root_pos, dtype=np.float32).reshape(1, 3)

            motions[motion_name] = {
                "joint_pos": joint_pos_1,
                "root_quat": root_quat_1,
                "root_pos": root_pos_1,
            }

        if "default" not in motions:
            raise ValueError(f"[{self.__class__.__name__}] motions must include a 'default' clip (length==1).")

        return motions

    @staticmethod
    def _empty_frames(n_joints: int) -> Dict[str, np.ndarray]:
        return {
            "joint_pos": np.zeros((0, n_joints), dtype=np.float32),
            "root_quat": np.zeros((0, 4), dtype=np.float32),
            "root_pos": np.zeros((0, 3), dtype=np.float32),
        }

    def _align_motion_to_anchor(
        self,
        motion: Dict[str, np.ndarray],
        anchor: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        p0 = motion["root_pos"][0]
        q0_yaw = _yaw_component_wxyz(motion["root_quat"][0])
        pa = anchor["root_pos"]
        qa_yaw = _yaw_component_wxyz(anchor["root_quat"])

        r0 = R.from_quat(q0_yaw, scalar_first=True)
        ra = R.from_quat(qa_yaw, scalar_first=True)
        r_delta = ra * r0.inv()

        root_pos_aligned = r_delta.apply(motion["root_pos"] - p0) + pa
        root_pos_aligned[:, 2] = motion["root_pos"][:, 2]

        root_quat_all = R.from_quat(motion["root_quat"], scalar_first=True)
        root_quat_aligned = (r_delta * root_quat_all).as_quat(scalar_first=True)

        return {
            "joint_pos": motion["joint_pos"].astype(np.float32, copy=True),
            "root_quat": root_quat_aligned.astype(np.float32),
            "root_pos": root_pos_aligned.astype(np.float32),
        }

    def _build_transition_prefix(
        self,
        anchor: Dict[str, np.ndarray],
        tgt_first: Dict[str, np.ndarray],
    ) -> Dict[str, np.ndarray]:
        t_steps = int(self.policy.transition_steps)
        if t_steps <= 0:
            return self._empty_frames(self.policy.n_joints)

        joints_tr = _linspace_rows(anchor["joint_pos"], tgt_first["joint_pos"], t_steps)
        root_pos_tr = _linspace_rows(anchor["root_pos"], tgt_first["root_pos"], t_steps)
        root_quat_tr = _slerp(anchor["root_quat"], tgt_first["root_quat"], t_steps)

        return {
            "joint_pos": joints_tr,
            "root_quat": root_quat_tr,
            "root_pos": root_pos_tr,
        }

    def append_motion_from_tail(self, name: str) -> bool:
        if name not in self.motions:
            print(f"[{self.__class__.__name__}] Unknown motion '{name}'")
            return False

        anchor = self.policy.read_ref_tail_state()
        aligned_motion = self._align_motion_to_anchor(self.motions[name], anchor)

        tgt_first = {
            "joint_pos": aligned_motion["joint_pos"][0],
            "root_quat": aligned_motion["root_quat"][0],
            "root_pos": aligned_motion["root_pos"][0],
        }
        trans_motion = self._build_transition_prefix(anchor, tgt_first)

        segment = {
            "joint_pos": np.concatenate([trans_motion["joint_pos"], aligned_motion["joint_pos"]], axis=0),
            "root_quat": np.concatenate([trans_motion["root_quat"], aligned_motion["root_quat"]], axis=0),
            "root_pos": np.concatenate([trans_motion["root_pos"], aligned_motion["root_pos"]], axis=0),
        }
        self.policy.append_ref_frames(segment)

        self.policy.current_name = name
        self.policy.current_done = (self.policy.ref_idx >= self.policy.ref_len - 1)

        print(
            f"[{self.__class__.__name__}] Append motion '{name}' | appended={segment['joint_pos'].shape[0]}, "
            f"ref_len={self.policy.ref_len}, transition={self.policy.transition_steps}"
        )
        return True

    def on_fade_in(self):
        self.append_motion_from_tail("default")

    def on_fade_out(self):
        self.append_motion_from_tail("default")

    def deactivate(self):
        return

    def post_step(self):
        return


class UDPMotionSource(MotionSourceBase):
    def __init__(self, policy: "TrackingPolicyRaw", policy_cfg: DictToClass):
        self.udp_enable = bool(getattr(policy_cfg, "udp_enable", True))
        self.udp_host = str(getattr(policy_cfg, "udp_host", "127.0.0.1"))
        self.udp_port = int(getattr(policy_cfg, "udp_port", 28562))
        self._udp_server: Optional[MotionUDPServer] = None

        super().__init__(policy, policy_cfg)

        if self.udp_enable:
            try:
                self._udp_server = MotionUDPServer(self.udp_host, self.udp_port)
                self._udp_server.start()
            except Exception as e:
                self._udp_server = None
                print(f"[UDPMotionSource] Failed to start UDP server: {e}")

    def request_motion(self, name: str) -> bool:
        if name not in self.motions:
            print(f"[UDPMotionSource] Unknown motion '{name}'")
            return False

        if (self.policy.current_name == "default" or name == "default") and self.policy.current_done:
            return self.append_motion_from_tail(name)

        print(
            f"[UDPMotionSource] Reject '{name}': "
            f"current='{self.policy.current_name}', done={self.policy.current_done}"
        )
        return False

    def post_step(self):
        if self._udp_server is None:
            return

        for cmd in self._udp_server.pop_all():
            self.request_motion("default" if cmd == "default" else cmd)

    def deactivate(self):
        if self._udp_server is not None:
            self._udp_server.stop()


class VRMotionSource(MotionSourceBase):
    def __init__(self, policy: "TrackingPolicyRaw", policy_cfg: DictToClass):
        self.vr_req_addr = str(getattr(policy_cfg, "vr_req_addr", "tcp://127.0.0.1:28701"))
        self.vr_rep_addr = str(getattr(policy_cfg, "vr_rep_addr", "tcp://127.0.0.1:28702"))
        self.vr_ctrl_addr = str(getattr(policy_cfg, "vr_ctrl_addr", "tcp://127.0.0.1:28703"))
        self.vr_low_watermark = int(getattr(policy_cfg, "vr_low_watermark", 10))
        self.vr_high_watermark = int(getattr(policy_cfg, "vr_high_watermark", 0))
        self.vr_chunk_frames = int(getattr(policy_cfg, "vr_chunk_frames", 5))
        self.vr_inflight_lifetime_steps = int(getattr(policy_cfg, "vr_inflight_lifetime_steps", 3))
        if self.vr_inflight_lifetime_steps < 0:
            raise ValueError("vr_inflight_lifetime_steps must be >= 0")
        if self.vr_high_watermark > 0 and self.vr_high_watermark < self.vr_low_watermark:
            raise ValueError("vr_high_watermark must be >= vr_low_watermark when enabled")

        self._vr_active = False
        self._vr_in_transition = False
        self._vr_transition_count = 0
        # Start-time anchor of deploy reference stream, used as transition start pose.
        self._vr_anchor_joint_pos: Optional[np.ndarray] = None
        self._vr_anchor_root_pos: Optional[np.ndarray] = None
        self._vr_anchor_root_quat: Optional[np.ndarray] = None
        self._vr_align_ready = False
        # Yaw-only alignment rotation: source(VR at start) -> target(deploy anchor at start).
        self._vr_r_delta: Optional[R] = None
        # Source VR root position at start; later VR root translation is measured relative to this origin.
        self._vr_source_root_pos0: Optional[np.ndarray] = None
        self._vr_target_anchor_pos: Optional[np.ndarray] = None

        positive_steps = [int(s) for s in np.asarray(policy.future_steps).reshape(-1).tolist() if int(s) > 0]
        self._target_future_horizon = int(max(positive_steps)) if len(positive_steps) > 0 else 0

        self._zmq_ctx = None
        self._req_sock = None
        self._rep_sock = None
        self._ctrl_sock = None
        self._req_inflight = False
        self._req_inflight_steps_left = 0
        self._pending_start_request = False
        self._vr_user_enabled = False
        self._prev_start_btn = False
        self._prev_stop_btn = False
        self._req_log_count = 0
        self._rep_log_count = 0
        self._last_req_log_monotonic: Optional[float] = None
        self._last_rep_log_monotonic: Optional[float] = None

        super().__init__(policy, policy_cfg)

        if zmq is None:
            raise ImportError("[VRMotionSource] pyzmq is required for motion_source='vr'.")
        try:
            self._zmq_ctx = zmq.Context.instance()
            self._req_sock = self._zmq_ctx.socket(zmq.PUSH)
            self._req_sock.setsockopt(zmq.LINGER, 0)
            self._req_sock.setsockopt(zmq.SNDHWM, 100)
            self._req_sock.connect(self.vr_req_addr)

            self._rep_sock = self._zmq_ctx.socket(zmq.PULL)
            self._rep_sock.setsockopt(zmq.LINGER, 0)
            self._rep_sock.setsockopt(zmq.RCVHWM, 200)
            self._rep_sock.connect(self.vr_rep_addr)

            self._ctrl_sock = self._zmq_ctx.socket(zmq.PULL)
            self._ctrl_sock.setsockopt(zmq.LINGER, 0)
            self._ctrl_sock.setsockopt(zmq.RCVHWM, 200)
            self._ctrl_sock.connect(self.vr_ctrl_addr)

            print(
                "[VRMotionSource] Connected "
                f"req->{self.vr_req_addr}, rep<-{self.vr_rep_addr}, "
                f"ctrl<-{self.vr_ctrl_addr}, low_watermark={self.vr_low_watermark}, "
                f"chunk_frames={self.vr_chunk_frames}, inflight_lifetime_steps={self.vr_inflight_lifetime_steps}"
            )
        except Exception as e:
            self._req_sock = None
            self._rep_sock = None
            self._ctrl_sock = None
            print(f"[VRMotionSource] Failed to create ZMQ sockets: {e}")

    @staticmethod
    def _extract_buttons(payload: dict) -> Optional[dict]:
        if not isinstance(payload, dict):
            return None
        buttons = payload.get("controller_buttons", None)
        if not isinstance(buttons, dict):
            return None
        return buttons

    def _drain_control(self) -> None:
        if self._ctrl_sock is None:
            return

        latest_buttons: Optional[dict] = None
        while True:
            try:
                raw = self._ctrl_sock.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as e:
                print(f"[VRMotionSource] control recv failed: {e}")
                break

            try:
                payload = json.loads(raw)
            except Exception:
                continue
            buttons = self._extract_buttons(payload)
            if buttons is not None:
                latest_buttons = buttons

        if latest_buttons is None:
            return

        start_btn = bool(latest_buttons.get("right_key_one", False))
        stop_btn = bool(latest_buttons.get("left_key_one", False))
        start_rise = start_btn and (not self._prev_start_btn)
        stop_rise = stop_btn and (not self._prev_stop_btn)
        self._prev_start_btn = start_btn
        self._prev_stop_btn = stop_btn

        if stop_rise:
            self._vr_user_enabled = False
            self._pending_start_request = False
            self._req_inflight = False
            self._req_inflight_steps_left = 0
            self._vr_active = False
            self._vr_align_ready = False
            self._vr_in_transition = False
            self._vr_transition_count = 0
            print("[VRMotionSource] VR stop from control button")

        if start_rise:
            self._vr_user_enabled = True
            self._pending_start_request = True
            self._req_inflight = False
            self._req_inflight_steps_left = 0
            self._vr_active = False
            self._vr_align_ready = False
            self._vr_in_transition = False
            self._vr_transition_count = 0
            print("[VRMotionSource] VR start requested from control button")

    def _future_horizon(self) -> int:
        if self.policy.ref_len <= 0:
            return 0
        return max(0, int(self.policy.ref_len - 1 - self.policy.ref_idx))

    @staticmethod
    def _repeat_frame(frame: Dict[str, np.ndarray], count: int) -> Dict[str, np.ndarray]:
        c = int(count)
        return {
            "joint_pos": np.repeat(frame["joint_pos"].reshape(1, -1), c, axis=0).astype(np.float32),
            "root_pos": np.repeat(frame["root_pos"].reshape(1, -1), c, axis=0).astype(np.float32),
            "root_quat": np.repeat(frame["root_quat"].reshape(1, -1), c, axis=0).astype(np.float32),
        }

    def _pad_future_once_on_start(self, frame: Dict[str, np.ndarray]) -> None:
        if self._target_future_horizon <= 0:
            return
        deficit = int(self._target_future_horizon - self._future_horizon())
        if deficit > 0:
            self.policy.append_ref_frames(self._repeat_frame(frame, deficit))

    def _pad_future_to_low_watermark(self, frame: Dict[str, np.ndarray]) -> None:
        deficit = int(self.vr_low_watermark - self._future_horizon())
        if deficit <= 0:
            return
        self.policy.append_ref_frames(self._repeat_frame(frame, deficit))
        print(
            "[VRMotionSource] Padded repeated frame "
            f"to low_watermark={self.vr_low_watermark} (added={deficit})"
        )

    def _appendable_reply_frames(self, frames: list[Dict[str, np.ndarray]]) -> list[Dict[str, np.ndarray]]:
        if self.vr_high_watermark <= 0:
            return frames
        h_now = self._future_horizon()
        if h_now >= self.vr_high_watermark:
            print(
                "[VRMotionSource][Warning] Drop reply frames because future buffer "
                f"already reached high_watermark={self.vr_high_watermark} (h={h_now})"
            )
            return []
        capacity = int(self.vr_high_watermark - h_now)
        kept = frames[:capacity]
        dropped = max(0, len(frames) - len(kept))
        if dropped > 0:
            print(
                "[VRMotionSource][Warning] Drop excess reply frames to respect "
                f"high_watermark={self.vr_high_watermark} (kept={len(kept)}, dropped={dropped}, h={h_now})"
            )
        return kept

    def _warn_horizon_if_needed(self, tag: str) -> None:
        if self._target_future_horizon <= 0:
            return
        if (not self._vr_user_enabled) and (not self._pending_start_request) and (not self._vr_active):
            return
        h = self._future_horizon()
        if h < self._target_future_horizon:
            print(
                f"[VRMotionSource][Warning] horizon={h} below required={self._target_future_horizon} ({tag})"
            )

    @staticmethod
    def _slerp_single_shortest(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
        a = float(np.clip(alpha, 0.0, 1.0))
        qq0 = np.asarray(q0, dtype=np.float64).reshape(4)
        qq1 = np.asarray(q1, dtype=np.float64).reshape(4)
        qq0 /= max(np.linalg.norm(qq0), 1e-9)
        qq1 /= max(np.linalg.norm(qq1), 1e-9)
        if float(np.dot(qq0, qq1)) < 0.0:
            qq1 = -qq1
        key = R.from_quat(np.stack([qq0, qq1], axis=0), scalar_first=True)
        interp = Slerp([0.0, 1.0], key)([a]).as_quat(scalar_first=True)[0]
        interp = interp / max(np.linalg.norm(interp), 1e-9)
        return interp.astype(np.float32)

    def _parse_frame(self, payload: dict) -> Optional[Dict[str, np.ndarray]]:
        if not isinstance(payload, dict):
            return None
        required = ["root_pos", "root_quat", "dof_pos"]
        if not all(k in payload for k in required):
            return None
        try:
            root_pos = np.asarray(payload["root_pos"], dtype=np.float32).reshape(3)
            root_quat = np.asarray(payload["root_quat"], dtype=np.float32).reshape(4)
            joint_pos = np.asarray(payload["dof_pos"], dtype=np.float32).reshape(-1)
        except Exception:
            return None
        if joint_pos.shape[0] != self.policy.n_joints:
            print(
                f"[VRMotionSource] dof dim mismatch: "
                f"got={joint_pos.shape[0]}, expected={self.policy.n_joints}"
            )
            return None
        if len(self.policy.dataset_joint_names) != joint_pos.shape[0]:
            print(
                f"[VRMotionSource] dataset_joint_names mismatch: "
                f"got={len(self.policy.dataset_joint_names)}, expected={joint_pos.shape[0]}"
            )
            return None
        joint_pos = remap_joint_array_by_names(
            joint_pos.reshape(1, -1),
            self.policy.dataset_joint_names,
            self.policy.obs_joint_names,
        )[0]
        qn = float(np.linalg.norm(root_quat))
        if not np.isfinite(qn) or qn < 1e-6:
            return None
        root_quat = (root_quat / qn).astype(np.float32)
        return {
            "joint_pos": joint_pos.astype(np.float32),
            "root_pos": root_pos.astype(np.float32),
            "root_quat": root_quat.astype(np.float32),
        }

    def _start_vr_session(self, first_frame: Dict[str, np.ndarray]) -> None:
        anchor = self.policy.read_ref_tail_state()
        self._vr_anchor_joint_pos = anchor["joint_pos"].astype(np.float32, copy=True)
        self._vr_anchor_root_pos = anchor["root_pos"].astype(np.float32, copy=True)
        self._vr_anchor_root_quat = anchor["root_quat"].astype(np.float32, copy=True)
        src_yaw = _yaw_component_wxyz(first_frame["root_quat"])
        tgt_yaw = _yaw_component_wxyz(anchor["root_quat"])
        r0 = R.from_quat(src_yaw, scalar_first=True)
        rc = R.from_quat(tgt_yaw, scalar_first=True)
        self._vr_r_delta = rc * r0.inv()
        self._vr_source_root_pos0 = first_frame["root_pos"].astype(np.float32, copy=True)
        self._vr_target_anchor_pos = anchor["root_pos"].astype(np.float32, copy=True)
        self._vr_align_ready = True
        self._vr_active = True
        self._vr_transition_count = 0
        self._vr_in_transition = int(self.policy.transition_steps) > 0
        self._pending_start_request = False
        # Bootstrap future horizon once at start using current ref-buffer tail.
        self._pad_future_once_on_start(
            {
                "joint_pos": anchor["joint_pos"].astype(np.float32, copy=True),
                "root_pos": anchor["root_pos"].astype(np.float32, copy=True),
                "root_quat": anchor["root_quat"].astype(np.float32, copy=True),
            }
        )
        print(
            "[VRMotionSource] VR start acknowledged "
            f"(transition_steps={int(self.policy.transition_steps)})"
        )

    def _apply_start_transition(self, aligned: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        if (
            not self._vr_in_transition
            or self._vr_anchor_joint_pos is None
            or self._vr_anchor_root_pos is None
            or self._vr_anchor_root_quat is None
        ):
            return aligned

        self._vr_transition_count += 1
        t_steps = max(1, int(self.policy.transition_steps))
        alpha = min(1.0, float(self._vr_transition_count) / float(t_steps))

        out_joint = (self._vr_anchor_joint_pos * (1.0 - alpha) + aligned["joint_pos"] * alpha).astype(np.float32)
        out_pos = (self._vr_anchor_root_pos * (1.0 - alpha) + aligned["root_pos"] * alpha).astype(np.float32)
        out_quat = self._slerp_single_shortest(self._vr_anchor_root_quat, aligned["root_quat"], alpha)

        if alpha >= 1.0:
            self._vr_in_transition = False
        return {
            "joint_pos": out_joint,
            "root_pos": out_pos,
            "root_quat": out_quat,
        }

    def _align_vr_frame(self, frame: Dict[str, np.ndarray]) -> Optional[Dict[str, np.ndarray]]:
        if (
            not self._vr_align_ready
            or self._vr_r_delta is None
            or self._vr_source_root_pos0 is None
            or self._vr_target_anchor_pos is None
        ):
            return None
        root_pos = frame["root_pos"].astype(np.float32)
        root_quat = frame["root_quat"].astype(np.float32)

        aligned_pos = self._vr_r_delta.apply(root_pos - self._vr_source_root_pos0) + self._vr_target_anchor_pos
        aligned_pos = aligned_pos.astype(np.float32)
        aligned_pos[2] = root_pos[2]

        aligned_quat = (self._vr_r_delta * R.from_quat(root_quat, scalar_first=True)).as_quat(scalar_first=True)
        aligned_quat = aligned_quat.astype(np.float32)
        aligned_quat /= max(np.linalg.norm(aligned_quat), 1e-6)
        return {
            "joint_pos": frame["joint_pos"].astype(np.float32, copy=True),
            "root_pos": aligned_pos,
            "root_quat": aligned_quat,
        }

    def _drain_replies(self) -> Optional[Dict[str, np.ndarray]]:
        last_aligned_frame: Optional[Dict[str, np.ndarray]] = None
        if self._rep_sock is None:
            return last_aligned_frame
        while True:
            try:
                raw = self._rep_sock.recv_string(flags=zmq.NOBLOCK)
            except zmq.Again:
                break
            except Exception as e:
                print(f"[VRMotionSource] recv failed: {e}")
                break

            try:
                payload = json.loads(raw)
            except Exception:
                print("[VRMotionSource] bad JSON reply")
                continue
            if not isinstance(payload, dict):
                continue

            raw_frames = payload.get("frames", [])
            if not isinstance(raw_frames, list):
                raw_frames = []
            parsed_frames = [f for f in (self._parse_frame(x) for x in raw_frames) if f is not None]
            if len(parsed_frames) == 0:
                continue

            recv_mono = time.monotonic()
            h_recv = self._future_horizon()
            self._rep_log_count += 1
            if self._last_rep_log_monotonic is None:
                rep_dt_msg = "first"
            else:
                rep_dt_ms = (recv_mono - self._last_rep_log_monotonic) * 1000.0
                rep_dt_msg = f"dt={rep_dt_ms:7.2f} ms"
            self._last_rep_log_monotonic = recv_mono

            start_flag = bool(payload.get("start", False))
            if self._pending_start_request and not start_flag:
                print(
                    f"[VRMotionSource][Rep] #{self._rep_log_count:06d} {rep_dt_msg} | "
                    f"h_recv={h_recv}, frames={len(parsed_frames)}, start={start_flag}, action=ignore_non_start_pending"
                )
                print("[VRMotionSource][Warning] ignore non-start reply while pending start alignment")
                continue
            if start_flag:
                if self._pending_start_request:
                    self._start_vr_session(parsed_frames[0])
                else:
                    print(
                        f"[VRMotionSource][Rep] #{self._rep_log_count:06d} {rep_dt_msg} | "
                        f"h_recv={h_recv}, frames={len(parsed_frames)}, start={start_flag}, action=drop_delayed_start"
                    )
                    print("[VRMotionSource][Warning] drop delayed start reply after alignment is ready")
                    continue

            if not self._vr_active:
                print(
                    f"[VRMotionSource][Rep] #{self._rep_log_count:06d} {rep_dt_msg} | "
                    f"h_recv={h_recv}, frames={len(parsed_frames)}, start={start_flag}, action=ignore_inactive"
                )
                continue

            out_frames = []
            for f in parsed_frames:
                aligned = self._align_vr_frame(f)
                if aligned is not None:
                    out_frames.append(self._apply_start_transition(aligned))
            out_frames = self._appendable_reply_frames(out_frames)
            if len(out_frames) == 0:
                print(
                    f"[VRMotionSource][Rep] #{self._rep_log_count:06d} {rep_dt_msg} | "
                    f"h_recv={h_recv}, frames={len(parsed_frames)}, start={start_flag}, action=ignore_no_aligned"
                )
                continue

            seg = {
                "joint_pos": np.stack([f["joint_pos"] for f in out_frames], axis=0).astype(np.float32),
                "root_pos": np.stack([f["root_pos"] for f in out_frames], axis=0).astype(np.float32),
                "root_quat": np.stack([f["root_quat"] for f in out_frames], axis=0).astype(np.float32),
            }
            self.policy.append_ref_frames(seg)
            last_aligned_frame = out_frames[-1]
            h_after = self._future_horizon()
            print(
                f"[VRMotionSource][Rep] #{self._rep_log_count:06d} {rep_dt_msg} | "
                f"h_recv={h_recv}, h_after={h_after}, frames={len(out_frames)}, start={start_flag}, action=append"
            )
        return last_aligned_frame

    def _send_request_if_needed(self) -> None:
        if self._req_sock is None:
            return
        if not self._vr_user_enabled:
            return
        if self._req_inflight:
            return
        h = self._future_horizon()
        should_request = (h <= self.vr_low_watermark) or self._pending_start_request
        if not should_request:
            return
        now = time.monotonic()
        start_flag = bool(self._pending_start_request)
        req = {
            "start": start_flag,
            "need_frames": int(max(1, self.vr_chunk_frames)),
            "future_horizon": int(h),
            "t_req_ms": int(time.time() * 1000),
        }
        try:
            self._req_sock.send_string(json.dumps(req), flags=zmq.NOBLOCK)
            self._req_inflight = True
            self._req_inflight_steps_left = int(self.vr_inflight_lifetime_steps)
            self._req_log_count += 1
            if self._last_req_log_monotonic is None:
                req_dt_msg = "first"
            else:
                req_dt_ms = (now - self._last_req_log_monotonic) * 1000.0
                req_dt_msg = f"dt={req_dt_ms:7.2f} ms"
            self._last_req_log_monotonic = now
            print(
                f"[VRMotionSource][Req] #{self._req_log_count:06d} {req_dt_msg} | "
                f"h_req={h}, start={start_flag}, need_frames={int(req['need_frames'])}"
            )
        except zmq.Again:
            return
        except Exception as e:
            print(f"[VRMotionSource] send request failed: {e}")

    def on_fade_in(self):
        self.append_motion_from_tail("default")
        self._pending_start_request = False
        self._req_inflight = False
        self._req_inflight_steps_left = 0
        self._vr_user_enabled = False
        self._prev_start_btn = False
        self._prev_stop_btn = False
        self._vr_active = False
        self._vr_in_transition = False
        self._vr_transition_count = 0
        self._vr_anchor_joint_pos = None
        self._vr_anchor_root_pos = None
        self._vr_anchor_root_quat = None
        self._vr_align_ready = False

    def on_fade_out(self):
        self._vr_user_enabled = False
        self._req_inflight = False
        self._req_inflight_steps_left = 0
        self._vr_active = False
        self._vr_in_transition = False
        self._vr_transition_count = 0
        self._vr_anchor_joint_pos = None
        self._vr_anchor_root_pos = None
        self._vr_anchor_root_quat = None
        self._vr_align_ready = False
        self._pending_start_request = False
        super().on_fade_out()

    def post_step(self):
        self._drain_control()
        last_aligned_frame = self._drain_replies()
        if last_aligned_frame is not None:
            self._req_inflight = False
            self._req_inflight_steps_left = 0
            self._pad_future_to_low_watermark(last_aligned_frame)
        elif self._req_inflight:
            self._req_inflight_steps_left -= 1
            if self._req_inflight_steps_left <= 0:
                self._req_inflight = False
                self._req_inflight_steps_left = 0
        self._send_request_if_needed()
        self._warn_horizon_if_needed("post_step")

    def deactivate(self):
        self._vr_user_enabled = False
        self._req_inflight = False
        self._req_inflight_steps_left = 0
        self._vr_active = False
        self._vr_in_transition = False
        self._vr_transition_count = 0
        self._vr_anchor_joint_pos = None
        self._vr_anchor_root_pos = None
        self._vr_anchor_root_quat = None
        self._vr_align_ready = False
        self._pending_start_request = False
        if self._req_sock is not None:
            try:
                self._req_sock.close(0)
            except Exception:
                pass
            self._req_sock = None
        if self._rep_sock is not None:
            try:
                self._rep_sock.close(0)
            except Exception:
                pass
            self._rep_sock = None
        if self._ctrl_sock is not None:
            try:
                self._ctrl_sock.close(0)
            except Exception:
                pass
            self._ctrl_sock = None
