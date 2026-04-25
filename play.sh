#uv run python scripts/data_process/replay_motion_npz.py data/tennis/adorozco_Derecha.npz --device cpu

uv run python scripts/data_process/replay_motion_npz.py data/tennis_tracking_npz --device cuda:1
