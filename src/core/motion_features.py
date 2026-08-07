"""Versioned geometric motion descriptor shared by training and runtime."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Sequence

import numpy as np


SCHEMA_VERSION = 1
WINDOW_FRAMES = 45
TARGET_FPS = 30.0
DESCRIPTOR_SIZE = 175

JOINT_NAMES = (
    "Hips",
    "LeftUpLeg",
    "RightUpLeg",
    "Spine",
    "LeftLeg",
    "RightLeg",
    "LeftFoot",
    "RightFoot",
    "LeftToeBase",
    "RightToeBase",
    "Neck",
    "Head",
    "LeftArm",
    "RightArm",
    "LeftForeArm",
    "RightForeArm",
    "LeftHand",
    "RightHand",
)

SMPL_22_JOINT_INDICES = (0, 1, 2, 3, 4, 5, 7, 8, 10, 11, 12, 15, 16, 17, 18, 19, 20, 21)
SMPL_22_PARENTS = (-1, 0, 0, 0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 9, 9, 12, 13, 14, 16, 17, 18, 19)

_JOINT_GROUPS = {
    "lower_body": (1, 2, 4, 5, 6, 7, 8, 9),
    "trunk": (0, 3, 10, 11),
    "upper_body": (12, 13, 14, 15, 16, 17),
    "full_body": tuple(range(18)),
}
_LEFT_RIGHT_PAIRS = (
    ("hip", 1, 2),
    ("knee", 4, 5),
    ("ankle", 6, 7),
    ("toe", 8, 9),
    ("shoulder", 12, 13),
    ("elbow", 14, 15),
    ("wrist", 16, 17),
)


def _feature_names() -> list[str]:
    names: list[str] = []
    for joint in JOINT_NAMES:
        for statistic in ("mean", "std", "p90", "max"):
            names.append(f"angular_speed.{joint}.{statistic}")
    for joint in JOINT_NAMES:
        for statistic in ("mean", "std", "max"):
            names.append(f"angular_excursion.{joint}.{statistic}")
    names.extend(
        [
            "root.horizontal_speed.mean",
            "root.horizontal_speed.std",
            "root.horizontal_speed.p25",
            "root.horizontal_speed.p50",
            "root.horizontal_speed.p75",
            "root.horizontal_speed.p90",
            "root.horizontal_speed.max",
            "root.vertical_velocity.mean",
            "root.vertical_velocity.std",
            "root.vertical_velocity_abs.p90",
            "root.vertical_velocity_abs.max",
            "root.horizontal_acceleration.mean",
            "root.horizontal_acceleration.std",
            "root.horizontal_acceleration.p90",
            "root.horizontal_acceleration.max",
            "root.vertical_range",
            "root.horizontal_path_length",
            "root.horizontal_net_displacement",
            "root.horizontal_straightness",
        ]
    )
    for group in _JOINT_GROUPS:
        for statistic in ("mean", "std", "max"):
            names.append(f"energy.{group}.{statistic}")
    for pair, _, _ in _LEFT_RIGHT_PAIRS:
        names.extend((f"symmetry.{pair}.correlation_peak", f"symmetry.{pair}.normalized_lag"))
    names.extend(
        (
            "periodicity.lower_body.autocorrelation_peak",
            "periodicity.lower_body.normalized_period",
            "periodicity.full_body.autocorrelation_peak",
            "periodicity.full_body.normalized_period",
        )
    )
    if len(names) != DESCRIPTOR_SIZE:
        raise RuntimeError(f"Feature schema has {len(names)} names, expected {DESCRIPTOR_SIZE}")
    return names


FEATURE_NAMES = tuple(_feature_names())


def build_feature_schema() -> dict:
    """Return the canonical schema and a digest of all compatibility fields."""
    schema = {
        "schema_version": SCHEMA_VERSION,
        "descriptor_size": DESCRIPTOR_SIZE,
        "window_frames": WINDOW_FRAMES,
        "fps": TARGET_FPS,
        "joint_order": list(JOINT_NAMES),
        "smpl_22_joint_indices": list(SMPL_22_JOINT_INDICES),
        "feature_names": list(FEATURE_NAMES),
    }
    canonical = json.dumps(schema, sort_keys=True, separators=(",", ":")).encode("utf-8")
    schema["sha256"] = hashlib.sha256(canonical).hexdigest()
    return schema


FEATURE_SCHEMA_HASH = build_feature_schema()["sha256"]


def write_feature_schema(path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(build_feature_schema(), indent=2) + "\n", encoding="utf-8")


def rotation_6d_to_matrix(rotation_6d: np.ndarray) -> np.ndarray:
    """Convert the first two rotation-matrix columns to orthonormal matrices."""
    values = np.asarray(rotation_6d, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != 6:
        raise ValueError("rotation_6d must have shape (..., 6)")
    if not np.isfinite(values).all():
        raise ValueError("rotation_6d must contain only finite values")
    first = values[..., :3]
    second = values[..., 3:]
    first_norm = np.linalg.norm(first, axis=-1, keepdims=True)
    if np.any(first_norm < 1e-12):
        raise ValueError("rotation_6d contains a degenerate first axis")
    first = first / first_norm
    second = second - np.sum(first * second, axis=-1, keepdims=True) * first
    second_norm = np.linalg.norm(second, axis=-1, keepdims=True)
    if np.any(second_norm < 1e-12):
        raise ValueError("rotation_6d contains collinear axes")
    second = second / second_norm
    third = np.cross(first, second)
    return np.stack((first, second, third), axis=-1)


def global_rotations_from_local(
    local_rotations: np.ndarray,
    parents: Sequence[int] = SMPL_22_PARENTS,
) -> np.ndarray:
    """Accumulate local joint rotations along a parent hierarchy."""
    local = np.asarray(local_rotations, dtype=np.float64)
    if local.ndim != 4 or local.shape[-2:] != (3, 3):
        raise ValueError("local_rotations must have shape (frames, joints, 3, 3)")
    if local.shape[1] != len(parents):
        raise ValueError("parents must contain one entry per joint")
    if not np.isfinite(local).all():
        raise ValueError("local_rotations must contain only finite values")
    result = np.empty_like(local)
    for joint, parent in enumerate(parents):
        if parent < 0:
            result[:, joint] = local[:, joint]
        else:
            if parent >= joint:
                raise ValueError("parents must be topologically ordered")
            result[:, joint] = result[:, parent] @ local[:, joint]
    return result


def _geodesic_angles(first: np.ndarray, second: np.ndarray) -> np.ndarray:
    relative = np.swapaxes(first, -1, -2) @ second
    trace = np.trace(relative, axis1=-2, axis2=-1)
    cosine = np.clip((trace - 1.0) * 0.5, -1.0, 1.0)
    return np.arccos(cosine)


def _summaries(values: np.ndarray, statistics: tuple[str, ...]) -> list[float]:
    output: list[float] = []
    for column in range(values.shape[1]):
        data = values[:, column]
        for statistic in statistics:
            if statistic == "mean":
                output.append(float(np.mean(data)))
            elif statistic == "std":
                output.append(float(np.std(data)))
            elif statistic == "p90":
                output.append(float(np.percentile(data, 90)))
            elif statistic == "max":
                output.append(float(np.max(data)))
            else:
                raise ValueError(f"Unknown statistic: {statistic}")
    return output


def _correlation_peak(first: np.ndarray, second: np.ndarray, minimum_lag: int = 0) -> tuple[float, float]:
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    maximum_lag = min(22, len(first) - 2)
    if maximum_lag < minimum_lag or len(first) < 3:
        return 0.0, 0.0
    best_correlation = -1.0
    best_lag = 0
    for lag in range(minimum_lag, maximum_lag + 1):
        if lag == 0:
            left, right = first, second
        else:
            left, right = first[:-lag], second[lag:]
        left = left - np.mean(left)
        right = right - np.mean(right)
        denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
        correlation = float(np.dot(left, right) / denominator) if denominator > 1e-12 else 0.0
        if correlation > best_correlation:
            best_correlation = correlation
            best_lag = lag
    return max(0.0, best_correlation), best_lag / float(len(first))


def extract_motion_descriptor(
    global_rotations: np.ndarray,
    root_positions_m: np.ndarray,
    fps: float = TARGET_FPS,
) -> np.ndarray:
    """Extract the version-1 175-dimensional motion descriptor."""
    rotations = np.asarray(global_rotations, dtype=np.float64)
    positions = np.asarray(root_positions_m, dtype=np.float64)
    fps_value = float(fps)
    expected_shape = (WINDOW_FRAMES, len(JOINT_NAMES), 3, 3)
    if rotations.shape != expected_shape:
        raise ValueError(f"global_rotations must have shape {expected_shape}")
    if positions.shape != (WINDOW_FRAMES, 3):
        raise ValueError(f"root_positions_m must have shape ({WINDOW_FRAMES}, 3)")
    if not math.isfinite(fps_value) or fps_value <= 0:
        raise ValueError("fps must be finite and positive")
    if not np.isfinite(rotations).all() or not np.isfinite(positions).all():
        raise ValueError("Motion inputs must contain only finite values")

    angular_speed = _geodesic_angles(rotations[:-1], rotations[1:]) * fps_value
    first_rotations = np.broadcast_to(rotations[0], rotations.shape)
    angular_excursion = _geodesic_angles(first_rotations, rotations)

    descriptor: list[float] = []
    descriptor.extend(_summaries(angular_speed, ("mean", "std", "p90", "max")))
    descriptor.extend(_summaries(angular_excursion, ("mean", "std", "max")))

    velocity = np.diff(positions, axis=0) * fps_value
    horizontal_velocity = velocity[:, (0, 2)]
    horizontal_speed = np.linalg.norm(horizontal_velocity, axis=1)
    vertical_velocity = velocity[:, 1]
    horizontal_acceleration = np.linalg.norm(np.diff(horizontal_velocity, axis=0) * fps_value, axis=1)
    horizontal_steps = np.diff(positions[:, (0, 2)], axis=0)
    horizontal_path = float(np.sum(np.linalg.norm(horizontal_steps, axis=1)))
    horizontal_net = float(np.linalg.norm(positions[-1, (0, 2)] - positions[0, (0, 2)]))
    descriptor.extend(
        (
            float(np.mean(horizontal_speed)),
            float(np.std(horizontal_speed)),
            float(np.percentile(horizontal_speed, 25)),
            float(np.percentile(horizontal_speed, 50)),
            float(np.percentile(horizontal_speed, 75)),
            float(np.percentile(horizontal_speed, 90)),
            float(np.max(horizontal_speed)),
            float(np.mean(vertical_velocity)),
            float(np.std(vertical_velocity)),
            float(np.percentile(np.abs(vertical_velocity), 90)),
            float(np.max(np.abs(vertical_velocity))),
            float(np.mean(horizontal_acceleration)),
            float(np.std(horizontal_acceleration)),
            float(np.percentile(horizontal_acceleration, 90)),
            float(np.max(horizontal_acceleration)),
            float(np.ptp(positions[:, 1])),
            horizontal_path,
            horizontal_net,
            horizontal_net / horizontal_path if horizontal_path > 1e-12 else 0.0,
        )
    )

    energy_signals: dict[str, np.ndarray] = {}
    for group, indices in _JOINT_GROUPS.items():
        energy = np.mean(np.square(angular_speed[:, indices]), axis=1)
        energy_signals[group] = energy
        descriptor.extend((float(np.mean(energy)), float(np.std(energy)), float(np.max(energy))))

    for _, left, right in _LEFT_RIGHT_PAIRS:
        peak, lag = _correlation_peak(angular_speed[:, left], angular_speed[:, right])
        descriptor.extend((peak, lag))

    for group in ("lower_body", "full_body"):
        signal = energy_signals[group]
        peak, period = _correlation_peak(signal, signal, minimum_lag=6)
        descriptor.extend((peak, period))

    result = np.asarray(descriptor, dtype=np.float32)
    if result.shape != (DESCRIPTOR_SIZE,) or not np.isfinite(result).all():
        raise ValueError("Feature extraction produced an invalid descriptor")
    return result
