#!/usr/bin/env python3
from __future__ import annotations

"""
Tennis contact calibration script (MuJoCo-only).

Standards used by default:
- Ball-ground rebound: ITF Rules of Tennis / ITF Ball Approval Procedures
  (drop from 2.54 m; default type2 range 1.35-1.47 m).
- Ball-racket targets: Rod Cross / Tennis Warehouse University literature:
  clamped normal COR e_y ~ 0.75, hand-held apparent COR e_A ~ 0.40,
  and contact dwell time near 5 ms (target band 3-7 ms).
"""

import argparse
import math
import re
import time
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path

import mujoco
import numpy as np


MODE_PRESETS = {
    "easy": {
        "speed_range": [12.0, 24.0],
        "launch_x_range": [-4.0, 4.0],
        "launch_y_range": [7.0, 8.8],
        "launch_z_range": [1.2, 2.8],
        "target_bounce_x_range": [-1.2, 1.6],
        "target_bounce_y_range": [-10.2, -7.4],
        "flight_time_range": [0.70, 1.00],
        "spin_rps_range": [-8.0, 8.0],
    },
    "medium": {
        "speed_range": [12.0, 24.0],
        "launch_x_range": [-4.0, 4.0],
        "launch_y_range": [7.0, 8.8],
        "launch_z_range": [1.2, 2.8],
        "target_bounce_x_range": [-2.8, 2.8],
        "target_bounce_y_range": [-11.2, -5.8],
        "flight_time_range": [0.70, 1.00],
        "spin_rps_range": [-8.0, 8.0],
    },
    "hard": {
        "speed_range": [12.0, 24.0],
        "launch_x_range": [-4.0, 4.0],
        "launch_y_range": [7.0, 8.8],
        "launch_z_range": [1.2, 2.8],
        "target_bounce_x_range": [-3.8, 3.8],
        "target_bounce_y_range": [-11.5, -3.5],
        "flight_time_range": [0.70, 1.00],
        "spin_rps_range": [-8.0, 8.0],
    },
}


@dataclass
class BounceMetrics:
    samples: int
    valid_first_bounce: int
    no_impact_ratio: float
    bounce_ge3_ratio: float
    h1_mean: float
    h1_p90: float
    h1_p99: float
    h1_max: float
    h1_gt5_ratio: float
    h1_gt7_ratio: float


@dataclass
class BounceCandidate:
    ball_solref2: float
    ground_solref2: float
    drop_rebound_h_m: float
    drop_e_eff: float
    itf_rebound_min_m: float
    itf_rebound_max_m: float
    itf_ball_type: str
    launch: BounceMetrics
    score: float


@dataclass
class RacketCandidate:
    ball_solref2: float
    racket_solref2: float
    racket_mu: float
    racket_half_thickness: float
    racket_effective_mass: float
    racket_joint_stiffness: float
    racket_joint_damping: float
    clamped_e_y: float
    handheld_e_a: float
    clamped_dwell_ms: float
    score: float
    feasible: bool


@dataclass
class ImpactObs:
    valid: bool
    pre_ball_v: np.ndarray
    post_ball_v: np.ndarray
    pre_racket_v: np.ndarray
    post_racket_v: np.ndarray
    contact_steps: int


@dataclass
class RacketSim:
    model: mujoco.MjModel
    data: mujoco.MjData
    ball_qpos: int
    ball_dof: int
    rack_dof: int | None


def _itf_rebound_range_m(ball_type: str) -> tuple[float, float]:
    key = str(ball_type).strip().lower()
    # ITF Rules of Tennis / Ball Approval Procedures (drop from 2.54 m onto rigid base).
    if key == "type1":
        return 1.38, 1.51
    if key in ("type2", "type3"):
        return 1.35, 1.47
    if key in ("high_altitude", "high-altitude", "highaltitude"):
        return 1.22, 1.35
    raise ValueError(f"Unsupported itf_ball_type: {ball_type}")


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _frange(start: float, end: float, step: float) -> list[float]:
    vals: list[float] = []
    x = float(start)
    for _ in range(10000):
        if x > float(end) + 1.0e-9:
            break
        vals.append(round(x, 6))
        x += float(step)
    return vals


def _safe_quantile(values: list[float], q: float) -> float:
    if len(values) == 0:
        return float("nan")
    return float(np.quantile(np.asarray(values, dtype=np.float64), q))


def _update_xml_geom_attrs(xml_path: Path, geom_name: str, attrs: dict[str, str]) -> None:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    geom = root.find(f".//geom[@name='{geom_name}']")
    if geom is None:
        raise KeyError(f"Cannot find geom '{geom_name}' in {xml_path}")
    for k, v in attrs.items():
        geom.set(k, v)
    tree.write(str(xml_path), encoding="utf-8", xml_declaration=False)


def _read_xml_geom_attrs(xml_path: Path, geom_name: str) -> dict[str, str]:
    tree = ET.parse(str(xml_path))
    root = tree.getroot()
    geom = root.find(f".//geom[@name='{geom_name}']")
    if geom is None:
        raise KeyError(f"Cannot find geom '{geom_name}' in {xml_path}")
    return dict(geom.attrib)


def _parse_floats(s: str, n: int) -> list[float]:
    vals = [float(x) for x in s.strip().split()]
    if len(vals) != n:
        raise ValueError(f"Expected {n} floats, got '{s}'")
    return vals


def _parse_tuple_literal(rhs: str, n: int, *, key: str) -> list[float]:
    m = re.search(r"\(([^)]*)\)", rhs)
    if m is None:
        raise ValueError(f"Cannot parse tuple literal for {key}: {rhs}")
    vals = [float(x.strip()) for x in m.group(1).split(",") if x.strip()]
    if len(vals) != n:
        raise ValueError(f"Expected {n} values for {key}, got {vals}")
    return vals


def _read_terrain_constants(tennis_py: Path) -> tuple[list[float], list[float]]:
    text = tennis_py.read_text(encoding="utf-8")
    m_fric = re.search(r"^TERRAIN_BALL_BOUNCE_FRICTION\s*=\s*(.+)$", text, flags=re.MULTILINE)
    m_solref = re.search(r"^TERRAIN_BALL_BOUNCE_SOLREF\s*=\s*(.+)$", text, flags=re.MULTILINE)
    if m_fric is None or m_solref is None:
        raise KeyError(f"Cannot find TERRAIN_BALL_BOUNCE_* in {tennis_py}")
    fric = _parse_tuple_literal(m_fric.group(1), 3, key="TERRAIN_BALL_BOUNCE_FRICTION")
    solref = _parse_tuple_literal(m_solref.group(1), 2, key="TERRAIN_BALL_BOUNCE_SOLREF")
    return fric, solref


def _update_terrain_constants(tennis_py: Path, *, friction: tuple[float, float, float], solref: tuple[float, float]) -> None:
    text = tennis_py.read_text(encoding="utf-8")
    text_new = re.sub(
        r"^TERRAIN_BALL_BOUNCE_FRICTION\s*=.*$",
        f"TERRAIN_BALL_BOUNCE_FRICTION = ({friction[0]:.3f}, {friction[1]:.3f}, {friction[2]:.3f})",
        text,
        flags=re.MULTILINE,
    )
    text_new = re.sub(
        r"^TERRAIN_BALL_BOUNCE_SOLREF\s*=.*$",
        f"TERRAIN_BALL_BOUNCE_SOLREF = ({solref[0]:.3f}, {solref[1]:.3f})",
        text_new,
        flags=re.MULTILINE,
    )
    tennis_py.write_text(text_new, encoding="utf-8")


def _apply_training_sim_defaults(args: argparse.Namespace) -> None:
    from omegaconf import OmegaConf

    cfg_path = _repo_root() / "cfg/task/G1/G1_tennis_highlevel.yaml"
    cfg = OmegaConf.load(str(cfg_path))
    sim = cfg.get("sim", {})

    args.physics_dt = float(sim.get("mujoco_physics_dt", args.physics_dt))
    args.iterations = int(sim.get("mujoco_iterations", args.iterations))
    args.ls_iterations = int(sim.get("mujoco_ls_iterations", args.ls_iterations))
    args.ccd_iterations = int(sim.get("mujoco_ccd_iterations", args.ccd_iterations))
    args.nconmax = int(sim.get("nconmax", args.nconmax))
    args.njmax = int(sim.get("njmax", args.njmax))


class MujocoBounceCalibrator:
    def __init__(
        self,
        *,
        physics_dt: float,
        iterations: int,
        ls_iterations: int,
        ccd_iterations: int,
        nconmax: int,
        njmax: int,
        max_episode_steps: int,
    ) -> None:
        root = _repo_root()
        ball_attr = _read_xml_geom_attrs(root / "active_adaptation/assets/tennis/tennis_ball.xml", "tennis_ball_geom")
        terrain_friction, terrain_solref = _read_terrain_constants(root / "active_adaptation/assets/tennis.py")

        b_mu = _parse_floats(ball_attr.get("friction", "0.45 0.05 0.02"), 3)
        b_sr = _parse_floats(ball_attr.get("solref", "0.010 0.050"), 2)
        b_si = _parse_floats(ball_attr.get("solimp", "0.95 0.995 0.001 0.5 2"), 5)
        c_mu = terrain_friction
        c_sr = terrain_solref
        c_si = [0.95, 0.995, 0.001, 0.5, 2.0]

        ball_radius = float(ball_attr.get("size", "0.0335"))
        ball_mass = float(ball_attr.get("mass", "0.0577"))
        self.contact_center_z = float(ball_radius)

        xml = f"""
<mujoco model="tennis_bounce_ident_fast">
  <size nconmax="{int(nconmax)}" njmax="{int(njmax)}"/>
  <option timestep="{float(physics_dt):.8f}" gravity="0 0 -9.81"/>
  <worldbody>
    <geom name="ground" type="plane"
      pos="0 0 0"
      size="0 0 1"
      condim="3"
      friction="{c_mu[0]} {c_mu[1]} {c_mu[2]}"
      solref="{c_sr[0]} {c_sr[1]}"
      solimp="{c_si[0]} {c_si[1]} {c_si[2]} {c_si[3]} {c_si[4]}"
      contype="1"
      conaffinity="1" />
    <body name="ball" pos="0 -4 1.5">
      <freejoint/>
      <geom name="ball_geom" type="sphere"
        size="{ball_radius}" mass="{ball_mass}"
        condim="{ball_attr.get('condim', '3')}"
        friction="{b_mu[0]} {b_mu[1]} {b_mu[2]}"
        solref="{b_sr[0]} {b_sr[1]}"
        solimp="{b_si[0]} {b_si[1]} {b_si[2]} {b_si[3]} {b_si[4]}"
        contype="1"
        conaffinity="1" />
    </body>
  </worldbody>
</mujoco>
"""
        self.model = mujoco.MjModel.from_xml_string(xml)
        self.model.opt.iterations = int(iterations)
        self.model.opt.ls_iterations = int(ls_iterations)
        if hasattr(self.model.opt, "ccd_iterations"):
            self.model.opt.ccd_iterations = int(ccd_iterations)
        self.data = mujoco.MjData(self.model)
        self.ball_gid = self.model.geom("ball_geom").id
        self.ground_gid = self.model.geom("ground").id
        self.physics_dt = float(self.model.opt.timestep)
        self.gravity_z = float(self.model.opt.gravity[2])
        self.max_episode_steps = max(0, int(max_episode_steps))
        self.contact_eps = float(self.contact_center_z + 1.0e-3)
        self.liftoff_eps = float(self.contact_center_z + 1.0e-2)

    def _reset_ball_state(self, pos: np.ndarray, vel: np.ndarray, ang: np.ndarray) -> None:
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[0:3] = pos
        self.data.qpos[3:7] = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
        self.data.qvel[0:3] = vel
        self.data.qvel[3:6] = ang
        mujoco.mj_forward(self.model, self.data)

    def _max_steps(self, max_time_s: float) -> int:
        steps = max(1, int(math.ceil(float(max_time_s) / self.physics_dt)))
        if self.max_episode_steps > 0:
            steps = min(steps, self.max_episode_steps)
        return steps

    def set_contact_solref2(self, ball_solref2: float, ground_solref2: float) -> None:
        self.model.geom_solref[self.ball_gid, 1] = float(ball_solref2)
        self.model.geom_solref[self.ground_gid, 1] = float(ground_solref2)

    def _rollout_first_rebound(
        self,
        pos: np.ndarray,
        vel: np.ndarray,
        ang: np.ndarray,
        *,
        max_time_s: float,
    ) -> tuple[bool, int, float]:
        self._reset_ball_state(pos, vel, ang)
        impact = False
        liftoff = False
        rebound_peak = float("-inf")
        bounce_count = 0
        prev_contact = False
        steps = self._max_steps(max_time_s)

        for _ in range(steps):
            mujoco.mj_step(self.model, self.data)
            z = float(self.data.qpos[2])
            in_contact = bool(z <= self.contact_eps)
            if in_contact and (not prev_contact):
                bounce_count += 1
                if liftoff:
                    break
                impact = True
            if impact and (not liftoff) and (z > self.liftoff_eps):
                liftoff = True
                rebound_peak = z
            if liftoff and (z > rebound_peak):
                rebound_peak = z
            prev_contact = in_contact

        if (not impact) or (not np.isfinite(rebound_peak)):
            return impact, bounce_count, float("nan")
        rebound_h = max(0.0, rebound_peak - self.contact_center_z)
        return True, bounce_count, float(rebound_h)

    def sample_launch_case(
        self,
        rng: np.random.Generator,
        cfg: dict[str, list[float]],
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        for _ in range(400):
            x = rng.uniform(cfg["launch_x_range"][0], cfg["launch_x_range"][1])
            y = rng.uniform(cfg["launch_y_range"][0], cfg["launch_y_range"][1])
            z = rng.uniform(cfg["launch_z_range"][0], cfg["launch_z_range"][1])
            tx = rng.uniform(cfg["target_bounce_x_range"][0], cfg["target_bounce_x_range"][1])
            ty = rng.uniform(cfg["target_bounce_y_range"][0], cfg["target_bounce_y_range"][1])
            t = rng.uniform(cfg["flight_time_range"][0], cfg["flight_time_range"][1])
            if t <= 1.0e-4:
                continue
            vx = (tx - x) / t
            vy = (ty - y) / t
            vz = (self.contact_center_z - z - 0.5 * self.gravity_z * (t * t)) / t
            if vy >= 0.0:
                continue
            speed = float(np.linalg.norm(np.array([vx, vy, vz], dtype=np.float64)))
            if speed < float(cfg["speed_range"][0]) or speed > float(cfg["speed_range"][1]):
                continue
            spin_rps = rng.uniform(cfg["spin_rps_range"][0], cfg["spin_rps_range"][1], size=3)
            spin_rad = spin_rps * (2.0 * math.pi)
            return (
                np.array([x, y, z], dtype=np.float64),
                np.array([vx, vy, vz], dtype=np.float64),
                spin_rad.astype(np.float64),
            )
        raise RuntimeError("Failed to sample launch case after 400 attempts.")


class MujocoRacketCalibrator:
    def __init__(
        self,
        *,
        dt: float,
        iterations: int,
        ls_iterations: int,
        ccd_iterations: int,
        nconmax: int,
        njmax: int,
        max_episode_steps: int,
    ):
        self.dt = float(dt)
        self.iterations = int(iterations)
        self.ls_iterations = int(ls_iterations)
        self.ccd_iterations = int(ccd_iterations)
        self.nconmax = int(nconmax)
        self.njmax = int(njmax)
        self.max_episode_steps = max(0, int(max_episode_steps))

    def _build_sim(
        self,
        *,
        ball_solref2: float,
        racket_solref2: float,
        racket_mu: float,
        racket_half_thickness: float,
        racket_mass: float,
        handheld: bool,
        joint_stiffness: float,
        joint_damping: float,
    ) -> RacketSim:
        joint_xml = ""
        if handheld:
            joint_xml = (
                f'<joint name="racket_slide_y" type="slide" axis="0 1 0" '
                f'stiffness="{float(joint_stiffness):.6f}" '
                f'damping="{float(joint_damping):.6f}"/>'
            )

        xml = f"""
<mujoco model="tennis_racket_ident_mujoco">
  <size nconmax="{self.nconmax}" njmax="{self.njmax}"/>
  <option timestep="{self.dt:.6f}" gravity="0 0 0" iterations="{self.iterations}" ls_iterations="{self.ls_iterations}"/>
  <worldbody>
    <body name="racket_body" pos="0 0 1.2">
      {joint_xml}
      <geom
        name="racket"
        type="box"
        pos="0 0 0"
        quat="0.7071068 -0.7071068 0 0"
        size="0.25 0.25 {float(racket_half_thickness):.6f}"
        mass="{float(racket_mass):.6f}"
        condim="3"
        friction="{float(racket_mu):.6f} 0.05 0.01"
        solref="0.010 {float(racket_solref2):.6f}"
        solimp="0.95 0.995 0.001 0.5 2"
      />
    </body>
    <body name="ball" pos="0 0.9 1.2">
      <freejoint/>
      <geom
        name="ball_geom"
        type="sphere"
        size="0.0335"
        mass="0.0577"
        condim="3"
        friction="0.20 0.01 0.001"
        solref="0.010 {float(ball_solref2):.6f}"
        solimp="0.95 0.995 0.001 0.5 2"
      />
    </body>
  </worldbody>
</mujoco>
"""
        model = mujoco.MjModel.from_xml_string(xml)
        if hasattr(model.opt, "ccd_iterations"):
            model.opt.ccd_iterations = int(self.ccd_iterations)
        data = mujoco.MjData(model)

        ball_body = model.body("ball").id
        ball_joint = int(model.body_jntadr[ball_body])
        ball_qpos = int(model.jnt_qposadr[ball_joint])
        ball_dof = int(model.jnt_dofadr[ball_joint])

        rack_dof = None
        if handheld:
            rack_joint = model.joint("racket_slide_y").id
            rack_dof = int(model.jnt_dofadr[rack_joint])

        return RacketSim(model=model, data=data, ball_qpos=ball_qpos, ball_dof=ball_dof, rack_dof=rack_dof)

    def _run_impact(
        self,
        sim: RacketSim,
        *,
        v_in_n: float,
        v_in_t: float,
        max_time_s: float,
        center_oblique_hit: bool,
    ) -> ImpactObs:
        mujoco.mj_resetData(sim.model, sim.data)

        if center_oblique_hit and abs(float(v_in_t)) > 1.0e-8 and abs(float(v_in_n)) > 1.0e-8:
            y0 = float(sim.data.qpos[sim.ball_qpos + 1])
            t_hit = max((y0 - 0.05) / abs(float(v_in_n)), 0.0)
            sim.data.qpos[sim.ball_qpos + 0] = -float(v_in_t) * t_hit

        sim.data.qvel[sim.ball_dof + 0] = float(v_in_t)
        sim.data.qvel[sim.ball_dof + 1] = -abs(float(v_in_n))
        sim.data.qvel[sim.ball_dof + 2] = 0.0
        sim.data.qvel[sim.ball_dof + 3 : sim.ball_dof + 6] = 0.0
        if sim.rack_dof is not None:
            sim.data.qvel[sim.rack_dof] = 0.0

        mujoco.mj_forward(sim.model, sim.data)

        max_steps = max(1, int(math.ceil(float(max_time_s) / self.dt)))
        if self.max_episode_steps > 0:
            max_steps = min(max_steps, self.max_episode_steps)

        pre_ball_v = None
        post_ball_v = None
        pre_racket_v = None
        post_racket_v = None
        in_contact = False
        contact_steps = 0

        for _ in range(max_steps):
            ball_v = np.asarray(sim.data.qvel[sim.ball_dof : sim.ball_dof + 3], dtype=np.float64).copy()
            racket_v = np.zeros((3,), dtype=np.float64)
            if sim.rack_dof is not None:
                racket_v[1] = float(sim.data.qvel[sim.rack_dof])

            ncon = int(sim.data.ncon)
            if (not in_contact) and ncon > 0:
                pre_ball_v = ball_v.copy()
                pre_racket_v = racket_v.copy()
                in_contact = True

            if in_contact and ncon > 0:
                contact_steps += 1

            if in_contact and ncon == 0:
                post_ball_v = ball_v.copy()
                post_racket_v = racket_v.copy()
                break

            mujoco.mj_step(sim.model, sim.data)

        if pre_ball_v is None or post_ball_v is None or pre_racket_v is None or post_racket_v is None:
            return ImpactObs(
                valid=False,
                pre_ball_v=np.zeros((3,), dtype=np.float64),
                post_ball_v=np.zeros((3,), dtype=np.float64),
                pre_racket_v=np.zeros((3,), dtype=np.float64),
                post_racket_v=np.zeros((3,), dtype=np.float64),
                contact_steps=0,
            )

        return ImpactObs(
            valid=True,
            pre_ball_v=pre_ball_v,
            post_ball_v=post_ball_v,
            pre_racket_v=pre_racket_v,
            post_racket_v=post_racket_v,
            contact_steps=contact_steps,
        )

    @staticmethod
    def _metrics(obs: ImpactObs) -> tuple[float, float]:
        if not obs.valid:
            return float("nan"), float("nan")

        rel_in_n = float(obs.pre_ball_v[1] - obs.pre_racket_v[1])
        rel_out_n = float(obs.post_ball_v[1] - obs.post_racket_v[1])
        if abs(rel_in_n) < 1.0e-8:
            e_y = float("nan")
        else:
            e_y = -rel_out_n / rel_in_n

        vin_mag = abs(float(obs.pre_ball_v[1]))
        if vin_mag < 1.0e-8:
            e_a = float("nan")
        else:
            e_a = max(float(obs.post_ball_v[1]), 0.0) / vin_mag

        return float(e_y), float(e_a)

    def evaluate_candidate(
        self,
        *,
        ball_solref2: float,
        racket_solref2: float,
        racket_mu: float,
        racket_half_thickness: float,
        racket_effective_mass: float,
        racket_joint_stiffness: float,
        racket_joint_damping: float,
        incoming_speed: float,
        incoming_tangent_speed: float,
        max_time_s: float,
    ) -> RacketCandidate:
        sim_clamped = self._build_sim(
            ball_solref2=ball_solref2,
            racket_solref2=racket_solref2,
            racket_mu=racket_mu,
            racket_half_thickness=racket_half_thickness,
            racket_mass=racket_effective_mass,
            handheld=False,
            joint_stiffness=0.0,
            joint_damping=0.0,
        )
        obs_clamped = self._run_impact(
            sim_clamped,
            v_in_n=incoming_speed,
            v_in_t=incoming_tangent_speed,
            max_time_s=max_time_s,
            center_oblique_hit=True,
        )
        clamped_e_y, _ = self._metrics(obs_clamped)
        clamped_dwell_ms = float(obs_clamped.contact_steps) * self.dt * 1000.0

        sim_handheld = self._build_sim(
            ball_solref2=ball_solref2,
            racket_solref2=racket_solref2,
            racket_mu=racket_mu,
            racket_half_thickness=racket_half_thickness,
            racket_mass=racket_effective_mass,
            handheld=True,
            joint_stiffness=racket_joint_stiffness,
            joint_damping=racket_joint_damping,
        )
        obs_handheld = self._run_impact(
            sim_handheld,
            v_in_n=incoming_speed,
            v_in_t=incoming_tangent_speed,
            max_time_s=max_time_s,
            center_oblique_hit=True,
        )
        _, handheld_e_a = self._metrics(obs_handheld)

        vals = [clamped_e_y, handheld_e_a, clamped_dwell_ms]
        finite = all(np.isfinite(v) for v in vals)
        feasible = finite and (0.70 <= clamped_e_y <= 0.78) and (0.38 <= handheld_e_a <= 0.43) and (3.0 <= clamped_dwell_ms <= 7.0)

        if not finite:
            score = 1.0e9
        else:
            score = 0.0
            score += 4.0 * ((clamped_e_y - 0.75) / 0.05) ** 2
            score += 4.0 * ((handheld_e_a - 0.40) / 0.05) ** 2
            score += 2.0 * ((clamped_dwell_ms - 5.0) / 2.0) ** 2
            if not feasible:
                if clamped_e_y < 0.70:
                    score += 400.0 * (0.70 - clamped_e_y)
                if clamped_e_y > 0.78:
                    score += 400.0 * (clamped_e_y - 0.78)
                if handheld_e_a < 0.38:
                    score += 500.0 * (0.38 - handheld_e_a)
                if handheld_e_a > 0.43:
                    score += 500.0 * (handheld_e_a - 0.43)
                if clamped_dwell_ms < 3.0:
                    score += 80.0 * (3.0 - clamped_dwell_ms)
                if clamped_dwell_ms > 7.0:
                    score += 80.0 * (clamped_dwell_ms - 7.0)

        return RacketCandidate(
            ball_solref2=float(ball_solref2),
            racket_solref2=float(racket_solref2),
            racket_mu=float(racket_mu),
            racket_half_thickness=float(racket_half_thickness),
            racket_effective_mass=float(racket_effective_mass),
            racket_joint_stiffness=float(racket_joint_stiffness),
            racket_joint_damping=float(racket_joint_damping),
            clamped_e_y=float(clamped_e_y),
            handheld_e_a=float(handheld_e_a),
            clamped_dwell_ms=float(clamped_dwell_ms),
            score=float(score),
            feasible=bool(feasible),
        )


def run_bounce(args) -> BounceCandidate:
    ball_vals = _frange(args.ball_solref2_min, args.ball_solref2_max, args.ball_solref2_step)
    ground_vals = _frange(args.ground_solref2_min, args.ground_solref2_max, args.ground_solref2_step)
    pairs = [(float(b), float(g)) for b in ball_vals for g in ground_vals]
    if len(pairs) == 0:
        raise RuntimeError("No bounce candidates generated. Check search range and step.")

    itf_min_h, itf_max_h = _itf_rebound_range_m(args.itf_ball_type)
    target_h2 = 0.5 * (itf_min_h + itf_max_h)
    if args.itf_rebound_min_m is not None:
        itf_min_h = float(args.itf_rebound_min_m)
    if args.itf_rebound_max_m is not None:
        itf_max_h = float(args.itf_rebound_max_m)
    target_h2 = 0.5 * (itf_min_h + itf_max_h)
    cal = MujocoBounceCalibrator(
        physics_dt=args.physics_dt,
        iterations=args.iterations,
        ls_iterations=args.ls_iterations,
        ccd_iterations=args.ccd_iterations,
        nconmax=args.nconmax,
        njmax=args.njmax,
        max_episode_steps=args.max_episode_steps,
    )
    cfg = MODE_PRESETS[str(args.mode)]

    def _build_launch_cases(*, samples: int, seed: int) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
        rng = np.random.default_rng(int(seed))
        out: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
        for _ in range(max(1, int(samples))):
            out.append(cal.sample_launch_case(rng, cfg))
        return out

    def _evaluate_candidate(
        ball_solref2: float,
        ground_solref2: float,
        launch_cases: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
    ) -> BounceCandidate:
        cal.set_contact_solref2(ball_solref2, ground_solref2)

        drop_pos = np.array([0.0, -4.0, cal.contact_center_z + float(args.drop_height_m)], dtype=np.float64)
        drop_vel = np.zeros((3,), dtype=np.float64)
        drop_ang = np.zeros((3,), dtype=np.float64)
        _, _, rebound_h = cal._rollout_first_rebound(drop_pos, drop_vel, drop_ang, max_time_s=float(args.max_time_s))
        rebound_h = float(0.0 if (not np.isfinite(rebound_h)) else rebound_h)
        e_eff = float(np.sqrt(max(rebound_h, 0.0) / max(float(args.drop_height_m), 1.0e-9)))

        h1_vals: list[float] = []
        no_impact = 0
        bounce_ge3 = 0
        gt5 = 0
        gt7 = 0
        for pos, vel, ang in launch_cases:
            impact, bounce_count, h1 = cal._rollout_first_rebound(pos, vel, ang, max_time_s=float(args.max_time_s))
            if not impact:
                no_impact += 1
            if bounce_count >= 3:
                bounce_ge3 += 1
            if np.isfinite(h1):
                h1_vals.append(float(h1))
                if h1 > 5.0:
                    gt5 += 1
                if h1 > 7.0:
                    gt7 += 1

        samples = max(1, len(launch_cases))
        valid = len(h1_vals)
        h1_arr = np.asarray(h1_vals, dtype=np.float64) if valid > 0 else np.asarray([], dtype=np.float64)
        h1_mean = float(np.mean(h1_arr)) if valid > 0 else float("nan")
        h1_p90 = _safe_quantile(h1_vals, 0.90)
        h1_p99 = _safe_quantile(h1_vals, 0.99)
        h1_max = float(np.max(h1_arr)) if valid > 0 else float("nan")
        no_impact_ratio = float(no_impact) / float(samples)
        bounce_ge3_ratio = float(bounce_ge3) / float(samples)
        gt5_ratio = float(gt5) / float(max(valid, 1))
        gt7_ratio = float(gt7) / float(max(valid, 1))

        score = abs(rebound_h - target_h2)
        if rebound_h < itf_min_h:
            score += 2.0 * (itf_min_h - rebound_h)
        if rebound_h > itf_max_h:
            score += 2.0 * (rebound_h - itf_max_h)
        score += 0.5 * gt5_ratio
        score += 1.0 * gt7_ratio
        score += 0.5 * bounce_ge3_ratio
        score += 0.5 * no_impact_ratio

        return BounceCandidate(
            ball_solref2=float(ball_solref2),
            ground_solref2=float(ground_solref2),
            drop_rebound_h_m=float(rebound_h),
            drop_e_eff=float(e_eff),
            itf_rebound_min_m=float(itf_min_h),
            itf_rebound_max_m=float(itf_max_h),
            itf_ball_type=str(args.itf_ball_type),
            launch=BounceMetrics(
                samples=int(samples),
                valid_first_bounce=int(valid),
                no_impact_ratio=float(no_impact_ratio),
                bounce_ge3_ratio=float(bounce_ge3_ratio),
                h1_mean=float(h1_mean),
                h1_p90=float(h1_p90),
                h1_p99=float(h1_p99),
                h1_max=float(h1_max),
                h1_gt5_ratio=float(gt5_ratio),
                h1_gt7_ratio=float(gt7_ratio),
            ),
            score=float(score),
        )

    def _evaluate_pairs(
        search_pairs: list[tuple[float, float]],
        *,
        launch_samples: int,
        seed: int,
        stage_name: str,
    ) -> list[BounceCandidate]:
        if len(search_pairs) == 0:
            return []
        launch_cases = _build_launch_cases(samples=launch_samples, seed=seed)
        out: list[BounceCandidate] = []
        total = len(search_pairs)
        t0 = time.perf_counter()
        print(
            f"[INFO][bounce:{stage_name}] backend=mujoco candidates={total} launch_samples={len(launch_cases)} "
            f"dt={cal.physics_dt:.6f} iterations={args.iterations} ls_iter={args.ls_iterations} ccd_iter={args.ccd_iterations}",
            flush=True,
        )
        for i, (ball_solref2, ground_solref2) in enumerate(search_pairs, start=1):
            cand = _evaluate_candidate(ball_solref2, ground_solref2, launch_cases)
            out.append(cand)
            if args.verbose:
                print(
                    f"[CAND] ball={cand.ball_solref2:.3f} ground={cand.ground_solref2:.3f} "
                    f"drop_rebound_h={cand.drop_rebound_h_m:.3f} e={cand.drop_e_eff:.3f} "
                    f"h1_p99={cand.launch.h1_p99:.3f} gt5={cand.launch.h1_gt5_ratio:.3f} "
                    f"gt7={cand.launch.h1_gt7_ratio:.3f} score={cand.score:.4f}",
                    flush=True,
                )
            if (i % max(1, int(args.progress_every)) == 0) or (i == total):
                elapsed = time.perf_counter() - t0
                rate = i / max(elapsed, 1.0e-6)
                eta = (total - i) / max(rate, 1.0e-6)
                print(
                    f"[PROGRESS][bounce:{stage_name}] {i}/{total} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                    flush=True,
                )
        return out

    print(
        f"[INFO] bounce search candidates={len(pairs)} (backend=mujoco, launch_samples={args.launch_samples}) "
        f"itf_ball_type={args.itf_ball_type} rebound_range_m=[{itf_min_h:.3f}, {itf_max_h:.3f}]",
        flush=True,
    )
    if args.prefilter_samples > 0 and args.prefilter_samples < args.launch_samples and args.prefilter_topk < len(pairs):
        print(f"[INFO] stage-1 coarse filter: samples={args.prefilter_samples}, keep_topk={args.prefilter_topk}")
        coarse = _evaluate_pairs(
            pairs,
            launch_samples=args.prefilter_samples,
            seed=args.seed,
            stage_name="coarse",
        )
        coarse.sort(key=lambda x: x.score)
        fine_pairs = [(c.ball_solref2, c.ground_solref2) for c in coarse[: args.prefilter_topk]]
        if len(fine_pairs) == 0:
            fine_pairs = pairs
        print(f"[INFO] stage-2 fine eval candidates={len(fine_pairs)}")
        candidates = _evaluate_pairs(
            fine_pairs,
            launch_samples=args.launch_samples,
            seed=args.seed + 1,
            stage_name="fine",
        )
    else:
        candidates = _evaluate_pairs(
            pairs,
            launch_samples=args.launch_samples,
            seed=args.seed,
            stage_name="single",
        )

    candidates.sort(key=lambda x: x.score)
    topk = min(max(1, args.topk), len(candidates))
    print("\n[RESULT] bounce top candidates")
    for i in range(topk):
        c = candidates[i]
        print(
            f"#{i+1}: ball={c.ball_solref2:.3f}, ground={c.ground_solref2:.3f}, "
            f"drop_rebound_h={c.drop_rebound_h_m:.3f}, e={c.drop_e_eff:.3f}, "
            f"h1_p99={c.launch.h1_p99:.3f}, gt5={c.launch.h1_gt5_ratio:.3f}, gt7={c.launch.h1_gt7_ratio:.3f}, "
            f"no_impact={c.launch.no_impact_ratio:.3f}, bounce>=3={c.launch.bounce_ge3_ratio:.3f}, score={c.score:.4f}"
        )

    best = candidates[0]
    print(
        f"[BEST][bounce] ball_solref2={best.ball_solref2:.6f} ground_solref2={best.ground_solref2:.6f} "
        f"drop_rebound_h={best.drop_rebound_h_m:.6f} e={best.drop_e_eff:.6f}"
    )
    return best


def run_racket(args, *, ball_solref2_for_racket: float | None = None) -> RacketCandidate:
    cal = MujocoRacketCalibrator(
        dt=args.physics_dt,
        iterations=args.iterations,
        ls_iterations=args.ls_iterations,
        ccd_iterations=args.ccd_iterations,
        nconmax=args.nconmax,
        njmax=args.njmax,
        max_episode_steps=args.max_episode_steps,
    )

    if ball_solref2_for_racket is not None:
        ball_solref2 = float(ball_solref2_for_racket)
    elif args.racket_ball_solref2 is not None:
        ball_solref2 = float(args.racket_ball_solref2)
    else:
        root = _repo_root()
        ball_attr = _read_xml_geom_attrs(root / "active_adaptation/assets/tennis/tennis_ball.xml", "tennis_ball_geom")
        ball_solref2 = _parse_floats(ball_attr.get("solref", "0.010 0.050"), 2)[1]
    speed_values = [float(v) for v in args.racket_incoming_speed_values]
    if len(speed_values) == 0:
        raise ValueError("racket_incoming_speed_values must contain at least one value.")

    candidates: list[RacketCandidate] = []
    total = (
        len(args.racket_solref2_values)
        * len(args.racket_mu_values)
        * len(args.racket_effective_mass_values)
        * len(args.racket_joint_stiffness_values)
        * len(args.racket_joint_damping_values)
    )
    print(
        f"[INFO] racket search candidates={total} incoming_speed_values={speed_values} "
        f"target_clamped_e_y~0.75 target_handheld_e_A~0.40 dwell_ms~5",
        flush=True,
    )
    done = 0
    t0 = time.perf_counter()
    for solref2 in args.racket_solref2_values:
        for mu in args.racket_mu_values:
            for mass in args.racket_effective_mass_values:
                for k in args.racket_joint_stiffness_values:
                    for d in args.racket_joint_damping_values:
                        speed_cands: list[RacketCandidate] = []
                        for speed in speed_values:
                            speed_cands.append(
                                cal.evaluate_candidate(
                                    ball_solref2=ball_solref2,
                                    racket_solref2=float(solref2),
                                    racket_mu=float(mu),
                                    racket_half_thickness=float(args.racket_half_thickness),
                                    racket_effective_mass=float(mass),
                                    racket_joint_stiffness=float(k),
                                    racket_joint_damping=float(d),
                                    incoming_speed=float(speed),
                                    incoming_tangent_speed=float(args.racket_incoming_tangent_speed),
                                    max_time_s=float(args.racket_max_time_s),
                                )
                            )
                        mean_e_y = float(np.nanmean([c.clamped_e_y for c in speed_cands]))
                        mean_e_a = float(np.nanmean([c.handheld_e_a for c in speed_cands]))
                        mean_dwell = float(np.nanmean([c.clamped_dwell_ms for c in speed_cands]))
                        mean_score = float(np.nanmean([c.score for c in speed_cands]))
                        if not np.isfinite(mean_score):
                            mean_score = 1.0e9
                        cand = RacketCandidate(
                            ball_solref2=float(ball_solref2),
                            racket_solref2=float(solref2),
                            racket_mu=float(mu),
                            racket_half_thickness=float(args.racket_half_thickness),
                            racket_effective_mass=float(mass),
                            racket_joint_stiffness=float(k),
                            racket_joint_damping=float(d),
                            clamped_e_y=mean_e_y,
                            handheld_e_a=mean_e_a,
                            clamped_dwell_ms=mean_dwell,
                            score=mean_score,
                            feasible=bool(all(c.feasible for c in speed_cands)),
                        )
                        candidates.append(cand)
                        if args.verbose:
                            print(
                                f"[CAND] solref2={cand.racket_solref2:.3f} mu={cand.racket_mu:.3f} M={cand.racket_effective_mass:.3f} "
                                f"k={cand.racket_joint_stiffness:.1f} d={cand.racket_joint_damping:.2f} "
                                f"e_y={cand.clamped_e_y:.3f} e_A={cand.handheld_e_a:.3f} dwell_ms={cand.clamped_dwell_ms:.3f} "
                                f"feasible={cand.feasible} score={cand.score:.4f} "
                                f"(avg over speeds={speed_values})"
                            )
                        done += 1
                        if (done % args.progress_every == 0) or (done == total):
                            elapsed = time.perf_counter() - t0
                            rate = done / max(elapsed, 1.0e-6)
                            eta = (total - done) / max(rate, 1.0e-6)
                            print(
                                f"[PROGRESS][racket] {done}/{total} elapsed={elapsed:.1f}s eta={eta:.1f}s",
                                flush=True,
                            )

    candidates.sort(key=lambda x: x.score)
    topk = min(max(1, args.topk), len(candidates))
    print("\n[RESULT] racket top candidates")
    for i in range(topk):
        c = candidates[i]
        print(
            f"#{i+1}: solref2={c.racket_solref2:.3f}, mu={c.racket_mu:.3f}, M={c.racket_effective_mass:.3f}, "
            f"k={c.racket_joint_stiffness:.1f}, d={c.racket_joint_damping:.2f}, "
            f"e_y={c.clamped_e_y:.3f}, e_A={c.handheld_e_a:.3f}, dwell_ms={c.clamped_dwell_ms:.3f}, "
            f"feasible={c.feasible}, score={c.score:.4f}"
        )

    best = candidates[0]
    print(
        f"[BEST][racket] solref2={best.racket_solref2:.6f} mu={best.racket_mu:.6f} "
        f"half_thickness={best.racket_half_thickness:.6f} "
        f"(fit: M={best.racket_effective_mass:.6f}, k={best.racket_joint_stiffness:.2f}, d={best.racket_joint_damping:.2f}) "
        f"e_y={best.clamped_e_y:.6f} e_A={best.handheld_e_a:.6f} dwell_ms={best.clamped_dwell_ms:.6f}"
    )
    return best


def apply_xml_from_best(best_bounce: BounceCandidate | None, best_racket: RacketCandidate | None) -> None:
    root = _repo_root()
    if best_bounce is not None:
        ball_xml = root / "active_adaptation/assets/tennis/tennis_ball.xml"
        tennis_py = root / "active_adaptation/assets/tennis.py"
        _update_xml_geom_attrs(
            ball_xml,
            "tennis_ball_geom",
            {"solref": f"0.010 {best_bounce.ball_solref2:.3f}"},
        )
        terrain_friction, _ = _read_terrain_constants(tennis_py)
        _update_terrain_constants(
            tennis_py,
            friction=(float(terrain_friction[0]), float(terrain_friction[1]), float(terrain_friction[2])),
            solref=(0.010, float(best_bounce.ground_solref2)),
        )
        print(
            f"[APPLY] updated ball XML + terrain ground solref2 -> "
            f"ball={best_bounce.ball_solref2:.3f}, ground={best_bounce.ground_solref2:.3f}"
        )

    if best_racket is not None:
        racket_xml = root / "active_adaptation/assets/G1/g1_racket.xml"
        _update_xml_geom_attrs(
            racket_xml,
            "tennis_racket_collision",
            {
                "solref": f"0.010 {best_racket.racket_solref2:.3f}",
                "friction": f"{best_racket.racket_mu:.3f} 0.05 0.01",
                "size": f"0.115 0.185 {best_racket.racket_half_thickness:.3f}",
            },
        )
        print(
            f"[APPLY] updated XML racket solref2/mu/thickness -> "
            f"{best_racket.racket_solref2:.3f}, {best_racket.racket_mu:.3f}, {best_racket.racket_half_thickness:.3f}"
        )


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="Unified tennis contact calibration (pure MuJoCo, ball-ground then ball-racket).")
    p.add_argument("--target", choices=["bounce", "racket", "all"], default="all")
    p.add_argument("--mode", choices=["easy", "medium", "hard"], default="hard")
    p.add_argument(
        "--use-training-cfg",
        action="store_true",
        help="Force sim params from cfg/task/G1/G1_tennis_highlevel.yaml.",
    )
    p.add_argument("--seed", type=int, default=1234)
    p.add_argument("--topk", type=int, default=8)
    p.add_argument("--progress-every", type=int, default=1)
    p.add_argument("--verbose", action="store_true")
    p.add_argument("--apply-xml", action="store_true")

    p.add_argument("--physics-dt", type=float, default=0.0005)
    p.add_argument("--iterations", type=int, default=24)
    p.add_argument("--ls-iterations", type=int, default=48)
    p.add_argument("--ccd-iterations", type=int, default=50)
    p.add_argument("--nconmax", type=int, default=600)
    p.add_argument("--njmax", type=int, default=2000)

    p.add_argument("--drop-height-m", type=float, default=2.54)
    p.add_argument("--itf-ball-type", choices=["type1", "type2", "type3", "high_altitude"], default="type2")
    p.add_argument("--itf-rebound-min-m", type=float, default=None)
    p.add_argument("--itf-rebound-max-m", type=float, default=None)
    p.add_argument("--launch-samples", type=int, default=64)
    p.add_argument("--max-time-s", type=float, default=2.5)
    p.add_argument("--max-episode-steps", type=int, default=0)
    p.add_argument("--prefilter-samples", type=int, default=8)
    p.add_argument("--prefilter-topk", type=int, default=24)
    p.add_argument("--ball-solref2-min", type=float, default=0.005)
    p.add_argument("--ball-solref2-max", type=float, default=0.300)
    p.add_argument("--ball-solref2-step", type=float, default=0.03)
    p.add_argument("--ground-solref2-min", type=float, default=0.02)
    p.add_argument("--ground-solref2-max", type=float, default=0.30)
    p.add_argument("--ground-solref2-step", type=float, default=0.02)

    p.add_argument("--racket-ball-solref2", type=float, default=None)
    p.add_argument("--racket-solref2-values", nargs="+", type=float, default=[0.29, 0.30, 0.31, 0.32])
    p.add_argument("--racket-mu-values", nargs="+", type=float, default=[2.5, 3.0, 3.5])
    p.add_argument("--racket-effective-mass-values", nargs="+", type=float, default=[0.30, 0.35])
    p.add_argument("--racket-joint-stiffness-values", nargs="+", type=float, default=[0.0])
    p.add_argument("--racket-joint-damping-values", nargs="+", type=float, default=[0.2, 0.4])
    p.add_argument("--racket-half-thickness", type=float, default=0.015)
    p.add_argument("--racket-incoming-speed-values", nargs="+", type=float, default=[25.0, 30.0, 35.0])
    p.add_argument("--racket-incoming-tangent-speed", type=float, default=0.0)
    p.add_argument("--racket-max-time-s", type=float, default=0.25)
    return p


def main() -> None:
    args = build_parser().parse_args()
    if bool(args.use_training_cfg):
        _apply_training_sim_defaults(args)

    best_bounce = None
    best_racket = None

    if args.target in ("bounce", "all"):
        best_bounce = run_bounce(args)

    if args.target in ("racket", "all"):
        best_racket = run_racket(
            args,
            ball_solref2_for_racket=(best_bounce.ball_solref2 if best_bounce is not None else None),
        )

    if args.apply_xml:
        apply_xml_from_best(best_bounce, best_racket)


if __name__ == "__main__":
    main()
