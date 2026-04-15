from __future__ import annotations

import os
from pathlib import Path

import mujoco
from mjlab.actuator import BuiltinPositionActuatorCfg
from mjlab.entity import EntityArticulationInfoCfg, EntityCfg
from mjlab.utils.spec_config import CollisionCfg
from mjlab.utils.os import update_assets

from mjlab.asset_zoo.robots.unitree_g1.g1_constants import FULL_COLLISION

ASSET_PATH = os.path.dirname(__file__)
G1_XML = Path(ASSET_PATH) / "G1" / "g1.xml"
G1_RACKET_MESH = Path(ASSET_PATH) / "G1_racket" / "unitree_g1" / "assets" / "tennis" / "entire_visual.STL"
# Keep racket mass-neutral for controlled comparisons.
RACKET_MASS = 0.345643
# RACKET_MASS = 0.0  # kg


def _get_assets(meshdir: str) -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    update_assets(assets, G1_XML.parent / "assets", meshdir)
    return assets


def _get_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    spec.assets = _get_assets(spec.meshdir)
    return spec


def _get_racket_assets(meshdir: str) -> dict[str, bytes]:
    assets = _get_assets(meshdir)
    if G1_RACKET_MESH.exists():
        mesh_key = f"{meshdir}/tennis/entire_visual.STL" if meshdir else "tennis/entire_visual.STL"
        assets[mesh_key] = G1_RACKET_MESH.read_bytes()
    return assets


def _has_name(items, name: str) -> bool:
    return any(getattr(item, "name", None) == name for item in items)


def _delete_geoms_by_mesh(spec: mujoco.MjSpec, mesh_name: str) -> int:
    removed = 0
    for geom in list(spec.geoms):
        if getattr(geom, "meshname", None) == mesh_name:
            spec.delete(geom)
            removed += 1
    return removed


def _add_racket_to_spec(spec: mujoco.MjSpec) -> None:
    # Remove the original right fake-hand visual mesh so the racket is directly attached
    # to the arm end-effector in rendering.
    _delete_geoms_by_mesh(spec, "right_rubber_hand")

    if not _has_name(spec.meshes, "tennis_racket"):
        spec.add_mesh(name="tennis_racket", file="tennis/entire_visual.STL")

    wrist_body = spec.body("right_wrist_yaw_link")
    if _has_name(spec.bodies, "tennis_racket_mount"):
        racket_body = spec.body("tennis_racket_mount")
    else:
        racket_body = wrist_body.add_body(name="tennis_racket_mount")
        racket_body.pos[:] = [0.0, 0.0, 0.0]

    if not _has_name(spec.geoms, "tennis_racket_visual"):
        racket_visual = racket_body.add_geom(
            name="tennis_racket_visual",
            type=mujoco.mjtGeom.mjGEOM_MESH,
            meshname="tennis_racket",
        )
        racket_visual.pos[:] = [0.0415, -0.003, 0.0]
        racket_visual.quat[:] = [0.5, -0.5, 0.5, -0.5]
        racket_visual.density = 0.0
        racket_visual.group = 2
        racket_visual.contype = 0
        racket_visual.conaffinity = 0

    if not _has_name(spec.geoms, "tennis_racket_collision"):
        racket_collision = racket_body.add_geom(
            name="tennis_racket_collision",
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
        )
        racket_collision.pos[:] = [0.1025, -0.004, 0.4]
        racket_collision.quat[:] = [0.0, 0.0, -0.7071068, 0.7071068]
        racket_collision.size[:] = [0.12, 0.005, 0.0]
        racket_collision.mass = RACKET_MASS
        racket_collision.condim = 1
        racket_collision.contype = 0
        racket_collision.conaffinity = 0

    if not _has_name(spec.sites, "tennis_racket_center"):
        racket_center = racket_body.add_site(name="tennis_racket_center")
        racket_center.pos[:] = [0.1025, -0.004, 0.4]
        racket_center.quat[:] = [0.0, 0.0, 0.7071068, 0.7071068]
        racket_center.size[:] = [0.01, 0.01, 0.01]

    if not _has_name(spec.sensors, "tennis_racket_center_global_linvel"):
        spec.add_sensor(
            name="tennis_racket_center_global_linvel",
            type=mujoco.mjtSensor.mjSENS_FRAMELINVEL,
            objtype=mujoco.mjtObj.mjOBJ_SITE,
            objname="tennis_racket_center",
        )


def _get_racket_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(G1_XML))
    _add_racket_to_spec(spec)
    spec.assets = _get_racket_assets(spec.meshdir)
    return spec


# Manual symmetry maps (explicit control).
JOINT_SYMMETRY_MAP = {
    "left_hip_pitch_joint": (1, "right_hip_pitch_joint"),
    "right_hip_pitch_joint": (1, "left_hip_pitch_joint"),
    "left_hip_roll_joint": (-1, "right_hip_roll_joint"),
    "right_hip_roll_joint": (-1, "left_hip_roll_joint"),
    "left_hip_yaw_joint": (-1, "right_hip_yaw_joint"),
    "right_hip_yaw_joint": (-1, "left_hip_yaw_joint"),
    "left_knee_joint": (1, "right_knee_joint"),
    "right_knee_joint": (1, "left_knee_joint"),
    "left_ankle_pitch_joint": (1, "right_ankle_pitch_joint"),
    "right_ankle_pitch_joint": (1, "left_ankle_pitch_joint"),
    "left_ankle_roll_joint": (-1, "right_ankle_roll_joint"),
    "right_ankle_roll_joint": (-1, "left_ankle_roll_joint"),
    "waist_yaw_joint": (-1, "waist_yaw_joint"),
    "waist_roll_joint": (-1, "waist_roll_joint"),
    "waist_pitch_joint": (1, "waist_pitch_joint"),
    "left_shoulder_pitch_joint": (1, "right_shoulder_pitch_joint"),
    "right_shoulder_pitch_joint": (1, "left_shoulder_pitch_joint"),
    "left_shoulder_roll_joint": (-1, "right_shoulder_roll_joint"),
    "right_shoulder_roll_joint": (-1, "left_shoulder_roll_joint"),
    "left_shoulder_yaw_joint": (-1, "right_shoulder_yaw_joint"),
    "right_shoulder_yaw_joint": (-1, "left_shoulder_yaw_joint"),
    "left_elbow_joint": (1, "right_elbow_joint"),
    "right_elbow_joint": (1, "left_elbow_joint"),
    "left_wrist_roll_joint": (-1, "right_wrist_roll_joint"),
    "right_wrist_roll_joint": (-1, "left_wrist_roll_joint"),
    "left_wrist_pitch_joint": (1, "right_wrist_pitch_joint"),
    "right_wrist_pitch_joint": (1, "left_wrist_pitch_joint"),
    "left_wrist_yaw_joint": (-1, "right_wrist_yaw_joint"),
    "right_wrist_yaw_joint": (-1, "left_wrist_yaw_joint"),
}

SPATIAL_SYMMETRY_MAP = {
    "left_hip_pitch_link": "right_hip_pitch_link",
    "right_hip_pitch_link": "left_hip_pitch_link",
    "left_hip_roll_link": "right_hip_roll_link",
    "right_hip_roll_link": "left_hip_roll_link",
    "left_hip_yaw_link": "right_hip_yaw_link",
    "right_hip_yaw_link": "left_hip_yaw_link",
    "left_knee_link": "right_knee_link",
    "right_knee_link": "left_knee_link",
    "left_ankle_pitch_link": "right_ankle_pitch_link",
    "right_ankle_pitch_link": "left_ankle_pitch_link",
    "left_ankle_roll_link": "right_ankle_roll_link",
    "right_ankle_roll_link": "left_ankle_roll_link",
    "pelvis": "pelvis",
    "torso_link": "torso_link",
    "waist_yaw_link": "waist_yaw_link",
    "waist_roll_link": "waist_roll_link",
    "left_shoulder_pitch_link": "right_shoulder_pitch_link",
    "right_shoulder_pitch_link": "left_shoulder_pitch_link",
    "left_shoulder_roll_link": "right_shoulder_roll_link",
    "right_shoulder_roll_link": "left_shoulder_roll_link",
    "left_shoulder_yaw_link": "right_shoulder_yaw_link",
    "right_shoulder_yaw_link": "left_shoulder_yaw_link",
    "left_elbow_link": "right_elbow_link",
    "right_elbow_link": "left_elbow_link",
    "left_wrist_roll_link": "right_wrist_roll_link",
    "right_wrist_roll_link": "left_wrist_roll_link",
    "left_wrist_pitch_link": "right_wrist_pitch_link",
    "right_wrist_pitch_link": "left_wrist_pitch_link",
    "left_wrist_yaw_link": "right_wrist_yaw_link",
    "right_wrist_yaw_link": "left_wrist_yaw_link",
    "left_hand_mimic": "right_hand_mimic",
    "right_hand_mimic": "left_hand_mimic",
    "head_mimic": "head_mimic",
    "right_ankle_roll_toe_link": "left_ankle_roll_toe_link",
    "left_ankle_roll_toe_link": "right_ankle_roll_toe_link",
}

SPATIAL_SYMMETRY_MAP_RACKET = {
    **SPATIAL_SYMMETRY_MAP,
    "tennis_racket_mount": "tennis_racket_mount",
}

G1_INIT_STATE = EntityCfg.InitialStateCfg(
    pos=(0.0, 0.0, 0.74),
    joint_pos={
        ".*_hip_pitch_joint": -0.28,
        ".*_knee_joint": 0.5,
        ".*_ankle_pitch_joint": -0.23,
        ".*_elbow_joint": 0.87,
        "left_shoulder_roll_joint": 0.16,
        "left_shoulder_pitch_joint": 0.35,
        "right_shoulder_roll_joint": -0.16,
        "right_shoulder_pitch_joint": 0.35,
        ".*_wrist_roll_joint": 0.0,
        ".*_wrist_pitch_joint": 0.0,
        ".*_wrist_yaw_joint": 0.0,
        ".*": 0.0,
    },
    joint_vel={".*": 0.0},
)

G1_ACTUATOR_UPPER = BuiltinPositionActuatorCfg(
    target_names_expr=(
        ".*_elbow_joint",
        ".*_shoulder_pitch_joint",
        ".*_shoulder_roll_joint",
        ".*_shoulder_yaw_joint",
        ".*_wrist_roll_joint",
    ),
    armature=0.003609725,
    stiffness=14.25062309787429,
    damping=0.907222843292423,
    effort_limit=25.0,
)
G1_ACTUATOR_HIP_YAW_WAIST_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_yaw_joint", "waist_yaw_joint"),
    armature=0.010177520,
    stiffness=40.17923847137318,
    damping=2.5578897650279457,
    effort_limit=88.0,
)
G1_ACTUATOR_HIP_PITCH_ROLL_KNEE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_hip_pitch_joint", ".*_hip_roll_joint", ".*_knee_joint"),
    armature=0.025101925,
    stiffness=99.09842777666113,
    damping=6.3088018534966395,
    effort_limit=139.0,
)
G1_ACTUATOR_WRIST_PITCH_YAW = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_wrist_pitch_joint", ".*_wrist_yaw_joint"),
    armature=0.0021812,
    stiffness=8.611032447370201,
    damping=0.548195351665136,
    effort_limit=13.4,
)
G1_ACTUATOR_WAIST = BuiltinPositionActuatorCfg(
    target_names_expr=("waist_pitch_joint", "waist_roll_joint"),
    armature=0.00721945,
    stiffness=28.50124619574858,
    damping=1.814445686584846,
    effort_limit=35.0,
)
G1_ACTUATOR_ANKLE = BuiltinPositionActuatorCfg(
    target_names_expr=(".*_ankle_pitch_joint", ".*_ankle_roll_joint"),
    armature=0.00721945,
    stiffness=28.50124619574858,
    damping=1.814445686584846,
    effort_limit=35.0,
)

# Keep racket visual for training, but disable its collision geometry so it does not
# interact with terrain (e.g. lying motions touching the floor).
G1_RACKET_NO_COLLISION = CollisionCfg(
    geom_names_expr=("tennis_racket_collision",),
    contype=0,
    conaffinity=0,
    disable_other_geoms=False,
)

G1_ARTICULATION = EntityArticulationInfoCfg(
    actuators=(
        G1_ACTUATOR_UPPER,
        G1_ACTUATOR_HIP_PITCH_ROLL_KNEE,
        G1_ACTUATOR_HIP_YAW_WAIST_YAW,
        G1_ACTUATOR_WRIST_PITCH_YAW,
        G1_ACTUATOR_WAIST,
        G1_ACTUATOR_ANKLE,
    ),
    soft_joint_pos_limit_factor=0.9,
)

G1_JOINT_ORDER = ('left_hip_pitch_joint', 'right_hip_pitch_joint', 'waist_yaw_joint', 'left_hip_roll_joint', 'right_hip_roll_joint', 'waist_roll_joint', 'left_hip_yaw_joint', 'right_hip_yaw_joint', 'waist_pitch_joint', 'left_knee_joint', 'right_knee_joint', 'left_shoulder_pitch_joint', 'right_shoulder_pitch_joint', 'left_ankle_pitch_joint', 'right_ankle_pitch_joint', 'left_shoulder_roll_joint', 'right_shoulder_roll_joint', 'left_ankle_roll_joint', 'right_ankle_roll_joint', 'left_shoulder_yaw_joint', 'right_shoulder_yaw_joint', 'left_elbow_joint', 'right_elbow_joint', 'left_wrist_roll_joint', 'right_wrist_roll_joint', 'left_wrist_pitch_joint', 'right_wrist_pitch_joint', 'left_wrist_yaw_joint', 'right_wrist_yaw_joint')

G1_CFG = EntityCfg(
    init_state=G1_INIT_STATE,
    collisions=(FULL_COLLISION,),
    spec_fn=_get_spec,
    articulation=G1_ARTICULATION,
)

G1_CFG.joint_symmetry_mapping = JOINT_SYMMETRY_MAP
G1_CFG.spatial_symmetry_mapping = SPATIAL_SYMMETRY_MAP
G1_CFG.joint_name_order = G1_JOINT_ORDER

G1_RACKET_CFG = EntityCfg(
    init_state=G1_INIT_STATE,
    collisions=(FULL_COLLISION, G1_RACKET_NO_COLLISION),
    spec_fn=_get_racket_spec,
    articulation=G1_ARTICULATION,
)

G1_RACKET_CFG.joint_symmetry_mapping = JOINT_SYMMETRY_MAP
G1_RACKET_CFG.spatial_symmetry_mapping = SPATIAL_SYMMETRY_MAP_RACKET
G1_RACKET_CFG.joint_name_order = G1_JOINT_ORDER

G1_COL_FULL = G1_CFG
G1_COL_FULL_SELF = G1_CFG
G1_COL_FULL_SELF_RACKET = G1_RACKET_CFG
