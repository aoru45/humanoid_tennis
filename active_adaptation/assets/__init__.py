import os
from .humanoid import G1_CFG, G1_COL_FULL, G1_COL_FULL_SELF, G1_COL_FULL_SELF_RACKET
from .tennis import get_tennis_ball_cfg, get_tennis_court_cfg

ASSET_PATH = os.path.dirname(__file__)

ROBOTS = {
    "g1": G1_CFG,
    "g1_col_full": G1_COL_FULL,
    "g1_col_full_self": G1_COL_FULL_SELF,
    "g1_col_full_self_racket": G1_COL_FULL_SELF_RACKET,
    "g1_racket": G1_COL_FULL_SELF_RACKET,
}


def get_robot_cfg(name: str):
    if name not in ROBOTS:
        raise ValueError(f"Unknown robot name: {name}")
    return ROBOTS[name]


__all__ = [
    "ASSET_PATH",
    "ROBOTS",
    "get_robot_cfg",
    "get_tennis_ball_cfg",
    "get_tennis_court_cfg",
]
