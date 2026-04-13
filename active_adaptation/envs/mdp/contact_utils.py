from __future__ import annotations

from typing import Sequence, Tuple


def resolve_contact_indices(contact_sensor, asset, name_keys) -> Tuple[list[int], list[str]]:
    """Resolve contact sensor indices for given body name patterns.

    MJLab ContactSensor does not expose find_bodies; we mirror its primary
    matching order using the entity's body list.
    """
    if hasattr(contact_sensor, "find_bodies"):
        return contact_sensor.find_bodies(name_keys)

    primary_names = None
    try:
        primary = contact_sensor.cfg.primary
        mode = getattr(primary, "mode", None)
        pattern = getattr(primary, "pattern", None)
        if mode in ("subtree", "body"):
            _, primary_names = asset.find_bodies(pattern)
        elif mode == "geom":
            _, primary_names = asset.find_geoms(pattern)
        else:
            _, primary_names = asset.find_bodies(pattern)
    except Exception:
        primary_names = None

    if primary_names is None:
        primary_names = list(asset.body_names)

    _, target_names = asset.find_bodies(name_keys)
    name_to_index = {n: i for i, n in enumerate(primary_names)}
    indices: list[int] = []
    missing = []
    for name in target_names:
        if name not in name_to_index:
            missing.append(name)
            continue
        indices.append(name_to_index[name])

    if missing:
        raise ValueError(
            f"Contact sensor primaries do not include bodies: {missing}. "
            f"Primary set size={len(primary_names)}."
        )

    return indices, target_names
