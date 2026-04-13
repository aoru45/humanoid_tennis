uv run torchrun --nproc_per_node=4 scripts/train.py \
    task=G1/G1_tracking +exp=train \
    'task.command.dataset.mem_paths=[tennis]' \
    'task.command.dataset.path_weights=[1.0]' \
    'task.num_envs=4096' \
    wandb.mode=online \
    +wandb.entity=aoru45 \
    wandb.project=gentle_humanoid \
    wandb.name=tennis-stage1
