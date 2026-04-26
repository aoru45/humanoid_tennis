from __future__ import annotations

import copy
from pathlib import Path

import mujoco
from mjlab.entity import EntityCfg

BALL_RADIUS = 0.0335
BALL_MASS = 0.057
COURT_HALF_WIDTH = 4.11
COURT_HALF_LENGTH = 11.89
NET_HEIGHT = 1.07
# Ball uses receive-only mask: other geoms' contype must match this conaffinity.
# Always enable collisions with racket(2), net(4), dedicated court bounce
# surface(8), and default terrain(1) through conaffinity=14 + terrain
# conaffinity extension in the scene.
BALL_COLLISION_CONTYPE = 16
BALL_COLLISION_CONAFFINITY = 14
NET_COLLISION_CONTYPE = 4
NET_COLLISION_CONAFFINITY = 0  # ball-only: avoid robot/net contacts
# Dedicated court bounce collider that only interacts with the ball.
COURT_BALL_COLLISION_CONTYPE = 8
COURT_BALL_COLLISION_CONAFFINITY = 0  # ball-only: avoid robot/court contacts
# Optional court-surface collider that only interacts with racket(contype=2).
COURT_RACKET_COLLISION_CONTYPE = 1
COURT_RACKET_COLLISION_CONAFFINITY = 2
ASSET_DIR = Path(__file__).resolve().parent / "tennis"
BALL_XML = ASSET_DIR / "tennis_ball.xml"
COURT_XML = ASSET_DIR / "tennis_court.xml"


def _get_tennis_assets() -> dict[str, bytes]:
    assets: dict[str, bytes] = {}
    for filename in ("tennis_court_green.png", "tennis_court_red_blue.png"):
        path = ASSET_DIR / filename
        if path.exists():
            assets[f"tennis/{filename}"] = path.read_bytes()
    return assets


def _find_named(items, name: str):
    for item in items:
        if getattr(item, "name", None) == name:
            return item
    raise KeyError(f"Cannot find named item: {name}")


def _resolve_court_texture(texture: str) -> str:
    name = str(texture).strip().lower()
    if name in ("green", "tennis_court_green", "tennis_court_green.png"):
        return "tennis/tennis_court_green.png"
    if name in ("red_blue", "red-blue", "tennis_court_red_blue", "tennis_court_red_blue.png"):
        return "tennis/tennis_court_red_blue.png"
    raise ValueError(
        f"Unknown tennis court texture '{texture}'. Expected one of: "
        "'green', 'red_blue'."
    )


def _build_tennis_ball_spec() -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(BALL_XML))
    ball_geom = _find_named(spec.geoms, "tennis_ball_geom")
    ball_geom.contype = BALL_COLLISION_CONTYPE
    ball_geom.conaffinity = BALL_COLLISION_CONAFFINITY
    return spec


def _build_tennis_court_spec(
    texture_file: str = "tennis/tennis_court_green.png",
    net_height: float = NET_HEIGHT,
    net_collision_half_thickness: float = 0.06,
    enable_racket_court_collision: bool = False,
) -> mujoco.MjSpec:
    spec = mujoco.MjSpec.from_file(str(COURT_XML))

    # Patch small runtime variants while keeping the court structure in static XML.
    texture = _find_named(spec.textures, "tennis_court_tex")
    texture.file = texture_file

    net_hh = float(net_height) * 0.5
    net_visual = _find_named(spec.geoms, "tennis_net_visual")
    net_visual.pos[:] = [0.0, 0.0, net_hh]
    net_visual.size[:] = [COURT_HALF_WIDTH, 0.02, net_hh]

    net_collision = _find_named(spec.geoms, "tennis_net_collision")
    net_collision.pos[:] = [0.0, 0.0, net_hh]
    net_collision.size[:] = [COURT_HALF_WIDTH, float(net_collision_half_thickness), net_hh]
    net_collision.contype = NET_COLLISION_CONTYPE
    net_collision.conaffinity = NET_COLLISION_CONAFFINITY

    court_ball_collision = _find_named(spec.geoms, "tennis_court_ball_collision")
    court_ball_collision.contype = COURT_BALL_COLLISION_CONTYPE
    court_ball_collision.conaffinity = COURT_BALL_COLLISION_CONAFFINITY

    court_racket_collision = _find_named(spec.geoms, "tennis_court_racket_collision")
    if enable_racket_court_collision:
        court_racket_collision.contype = COURT_RACKET_COLLISION_CONTYPE
        court_racket_collision.conaffinity = COURT_RACKET_COLLISION_CONAFFINITY
    else:
        court_racket_collision.contype = 0
        court_racket_collision.conaffinity = 0

    spec.assets = _get_tennis_assets()
    return spec


TENNIS_BALL_CFG = EntityCfg(spec_fn=_build_tennis_ball_spec)
TENNIS_COURT_CFG = EntityCfg(spec_fn=_build_tennis_court_spec)


def get_tennis_ball_cfg() -> EntityCfg:
    return copy.deepcopy(TENNIS_BALL_CFG)


def get_tennis_court_cfg(
    texture: str = "green",
    net_height: float = NET_HEIGHT,
    net_collision_half_thickness: float = 0.06,
    enable_racket_court_collision: bool = False,
) -> EntityCfg:
    texture_file = _resolve_court_texture(texture)
    return EntityCfg(
        spec_fn=lambda: _build_tennis_court_spec(
            texture_file=texture_file,
            net_height=float(net_height),
            net_collision_half_thickness=float(net_collision_half_thickness),
            enable_racket_court_collision=bool(enable_racket_court_collision),
        )
    )
