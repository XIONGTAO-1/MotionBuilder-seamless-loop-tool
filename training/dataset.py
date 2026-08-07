"""Load capped AMASS/BABEL windows and compute shared descriptors."""

from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np

from core.motion_features import (
    SMPL_22_JOINT_INDICES,
    SMPL_22_PARENTS,
    WINDOW_FRAMES,
    extract_motion_descriptor,
    global_rotations_from_local,
    rotation_6d_to_matrix,
)


CATEGORY_TO_CLASS = {
    "walk": "walk",
    "run": "run",
    "combat": "other",
    "idle": "other",
    "dance": "other",
    "jump_acrobatic": "other",
    "unknown_mixed": "other",
}


def coarse_label(category: str) -> str:
    try:
        return CATEGORY_TO_CLASS[category]
    except KeyError as exc:
        raise ValueError(f"Unsupported category: {category}") from exc


def load_window_manifest(dataset_root: Path, split: str | None = None) -> list[dict]:
    manifest_path = Path(dataset_root) / "windows.jsonl"
    rows: list[dict] = []
    with manifest_path.open("r", encoding="utf-8") as stream:
        for line_number, line in enumerate(stream, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            if split is not None and row.get("split") != split:
                continue
            try:
                row["coarse_class"] = coarse_label(row["category"])
            except (KeyError, ValueError) as exc:
                raise ValueError(f"Invalid manifest row {line_number}: {exc}") from exc
            rows.append(row)
    return rows


def _even_indices(size: int, count: int) -> np.ndarray:
    if count >= size:
        return np.arange(size, dtype=np.int64)
    indices = np.rint(np.linspace(0, size - 1, count)).astype(np.int64)
    if len(np.unique(indices)) != count:
        raise RuntimeError("Even source sampling produced duplicate indices")
    return indices


def select_training_windows(
    rows: Iterable[dict],
    max_per_source_class: int = 10,
) -> list[dict]:
    if max_per_source_class <= 0:
        raise ValueError("max_per_source_class must be positive")
    groups: dict[tuple[str, str], list[dict]] = defaultdict(list)
    for original in rows:
        row = dict(original)
        row["coarse_class"] = coarse_label(row["category"])
        groups[(row["source_path"], row["coarse_class"])].append(row)

    selected: list[dict] = []
    for key in sorted(groups):
        group = sorted(
            groups[key],
            key=lambda row: (int(row["window_start_frame"]), str(row["window_id"])),
        )
        count = min(max_per_source_class, len(group))
        selected.extend(group[index] for index in _even_indices(len(group), count))
    return selected


def descriptor_from_window(path: Path) -> np.ndarray:
    with np.load(Path(path), allow_pickle=False) as data:
        features = np.asarray(data["features"], dtype=np.float64)
        fps = float(data["fps"])
    if features.shape != (WINDOW_FRAMES, 135):
        raise ValueError(f"Expected a ({WINDOW_FRAMES}, 135) feature window, got {features.shape}")
    if not np.isfinite(features).all():
        raise ValueError(f"Window contains non-finite features: {path}")

    local_rotations = rotation_6d_to_matrix(features[:, :132].reshape(WINDOW_FRAMES, 22, 6))
    global_smpl = global_rotations_from_local(local_rotations, SMPL_22_PARENTS)
    global_rotations = global_smpl[:, SMPL_22_JOINT_INDICES]

    root_velocity = features[:, 132:135]
    root_positions = np.zeros((WINDOW_FRAMES, 3), dtype=np.float64)
    root_positions[1:] = np.cumsum(root_velocity[1:] / fps, axis=0)
    return extract_motion_descriptor(global_rotations, root_positions, fps=fps)


def load_descriptor_dataset(
    dataset_root: Path,
    rows: Iterable[dict],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    root = Path(dataset_root)
    metadata: list[dict] = []
    descriptors: list[np.ndarray] = []
    labels: list[str] = []
    for original in rows:
        row = dict(original)
        row["coarse_class"] = coarse_label(row["category"])
        descriptors.append(descriptor_from_window(root / row["output_path"]))
        labels.append(row["coarse_class"])
        metadata.append(row)
    if not descriptors:
        raise ValueError("No windows were selected")
    return np.stack(descriptors), np.asarray(labels), metadata


def assert_group_disjoint(
    train_indices: np.ndarray,
    validation_indices: np.ndarray,
    groups: np.ndarray,
) -> None:
    overlap = set(groups[train_indices]).intersection(groups[validation_indices])
    if overlap:
        raise ValueError(f"Cross-validation source leakage: {sorted(overlap)[:3]}")
