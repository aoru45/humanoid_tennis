#uv run python scripts/data_process/replay_motion_npz.py data/tennis/adorozco_Derecha.npz --device cpu

uv run python scripts/data_process/replay_motion_npz.py data/seed_g1_tracking_npz/Turn_Start_Walk_0360_001__A020_M.npz --device cuda:1
