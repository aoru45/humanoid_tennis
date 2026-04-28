uv run python scripts/data_process/convert_tennis_to_tracking_dataset.py \
    --input-dir data/tennis \
    --output-dir data/tennis_tracking_npz \
    --build-mem-path dataset/tennis \
    --overwrite
