import humanoid_tennis

if humanoid_tennis.get_backend() != "mjlab":
    raise NotImplementedError("Only the mjlab backend is supported.")


def create_mjlab_scene(*args, **kwargs):
    raise NotImplementedError("MJLab scene creation is handled in envs/locomotion.py.")
