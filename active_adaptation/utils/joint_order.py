from __future__ import annotations

import re
from typing import Any, Sequence

import mjlab.utils.lab_api.string as string_utils


def _normalize_patterns(joint_names: str | Sequence[str]) -> list[str]:
    if isinstance(joint_names, str):
        return [joint_names]
    return list(joint_names)


def get_joint_name_order(asset) -> list[str]:
    """Return the canonical joint order for an asset, falling back to asset order."""
    order = getattr(asset, "joint_name_order", None)
    if order is None and hasattr(asset, "cfg"):
        order = getattr(asset.cfg, "joint_name_order", None)
    if order is None:
        return list(asset.joint_names)
    order = list(order)
    if set(order) != set(asset.joint_names):
        missing = sorted(set(asset.joint_names) - set(order))
        extra = sorted(set(order) - set(asset.joint_names))
        raise ValueError(
            "Canonical joint order must include all asset joints. "
            f"Missing: {missing} Extra: {extra}"
        )
    return order


def _filter_order(order: Sequence[str], joint_names: str | Sequence[str]) -> list[str]:
    patterns = _normalize_patterns(joint_names)
    filtered: list[str] = []
    for name in order:
        for pat in patterns:
            if re.fullmatch(pat, name):
                filtered.append(name)
                break
    if not filtered:
        raise ValueError(f"No joints matched patterns {patterns} in canonical order.")
    return filtered


def resolve_joint_order(asset, joint_names: str | Sequence[str] = ".*") -> tuple[list[int], list[str]]:
    """Resolve joint ids/names using a canonical joint order."""
    order = get_joint_name_order(asset)
    filtered = _filter_order(order, joint_names)
    name_to_id = {name: idx for idx, name in enumerate(asset.joint_names)}
    missing = [name for name in filtered if name not in name_to_id]
    if missing:
        raise ValueError(f"Canonical joints missing in asset: {missing}")
    ids = [name_to_id[name] for name in filtered]
    return ids, filtered


def resolve_joint_order_with_values(
    asset,
    values_map: dict[str, Any],
    joint_names: str | Sequence[str] = ".*",
    preserve_order: bool = False,
) -> tuple[list[int], list[str], list[Any]]:
    """Resolve joint ids/names/values using canonical joint order."""
    order = get_joint_name_order(asset)
    filtered = _filter_order(order, joint_names)
    _, names, values = string_utils.resolve_matching_names_values(
        dict(values_map), filtered, preserve_order=preserve_order
    )
    name_to_id = {name: idx for idx, name in enumerate(asset.joint_names)}
    missing = [name for name in names if name not in name_to_id]
    if missing:
        raise ValueError(f"Canonical joints missing in asset: {missing}")
    ids = [name_to_id[name] for name in names]
    return ids, names, values
