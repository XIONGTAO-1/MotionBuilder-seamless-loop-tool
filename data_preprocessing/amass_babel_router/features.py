from __future__ import annotations

import math

import numpy as np


def axis_angle_to_matrix(axis_angle: np.ndarray) -> np.ndarray:
    values = np.asarray(axis_angle, dtype=np.float64)
    if values.ndim == 0 or values.shape[-1] != 3:
        raise ValueError("axis_angle must have shape (..., 3)")
    if not np.isfinite(values).all():
        raise ValueError("axis_angle contains non-finite values")

    x, y, z = np.moveaxis(values, -1, 0)
    zeros = np.zeros_like(x)
    skew = np.stack(
        (
            zeros,
            -z,
            y,
            z,
            zeros,
            -x,
            -y,
            x,
            zeros,
        ),
        axis=-1,
    ).reshape(values.shape[:-1] + (3, 3))

    theta_squared = np.sum(values * values, axis=-1)
    theta = np.sqrt(theta_squared)
    small = theta_squared < 1e-12
    coefficient_a = np.empty_like(theta)
    coefficient_b = np.empty_like(theta)
    coefficient_a[small] = 1.0 - theta_squared[small] / 6.0
    coefficient_b[small] = 0.5 - theta_squared[small] / 24.0
    coefficient_a[~small] = np.sin(theta[~small]) / theta[~small]
    coefficient_b[~small] = (1.0 - np.cos(theta[~small])) / theta_squared[~small]

    identity = np.broadcast_to(np.eye(3), values.shape[:-1] + (3, 3))
    skew_squared = skew @ skew
    return (
        identity
        + coefficient_a[..., None, None] * skew
        + coefficient_b[..., None, None] * skew_squared
    )


def matrix_to_rotation_6d(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix)
    if values.ndim < 2 or values.shape[-2:] != (3, 3):
        raise ValueError("matrix must have shape (..., 3, 3)")
    return np.concatenate((values[..., :, 0], values[..., :, 1]), axis=-1)


def _validate_pose_trans(
    poses: np.ndarray,
    trans: np.ndarray,
    fps: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    pose_values = np.asarray(poses, dtype=np.float64)
    trans_values = np.asarray(trans, dtype=np.float64)
    fps_value = float(fps)
    if pose_values.ndim != 2 or pose_values.shape[1] < 66 or pose_values.shape[0] == 0:
        raise ValueError("poses must have shape (frames, >=66)")
    if trans_values.shape != (pose_values.shape[0], 3):
        raise ValueError("trans must have shape (frames, 3)")
    if not math.isfinite(fps_value) or fps_value <= 0:
        raise ValueError("fps must be finite and positive")
    if not np.isfinite(pose_values).all() or not np.isfinite(trans_values).all():
        raise ValueError("poses and trans must be finite")
    return pose_values, trans_values, fps_value


def resample_pose_trans(
    poses: np.ndarray,
    trans: np.ndarray,
    source_fps: float,
    target_fps: float,
) -> tuple[np.ndarray, np.ndarray]:
    pose_values, trans_values, source_fps_value = _validate_pose_trans(
        poses, trans, source_fps
    )
    target_fps_value = float(target_fps)
    if not math.isfinite(target_fps_value) or target_fps_value <= 0:
        raise ValueError("target_fps must be finite and positive")

    duration = (pose_values.shape[0] - 1) / source_fps_value
    target_count = int(math.floor(duration * target_fps_value + 1e-9)) + 1
    target_times = np.arange(target_count, dtype=np.float64) / target_fps_value
    source_indices = np.rint(target_times * source_fps_value).astype(np.int64)
    source_indices = np.clip(source_indices, 0, pose_values.shape[0] - 1)
    sampled_poses = pose_values[source_indices].copy()

    source_times = np.arange(pose_values.shape[0], dtype=np.float64) / source_fps_value
    sampled_trans = np.column_stack(
        [np.interp(target_times, source_times, trans_values[:, axis]) for axis in range(3)]
    )
    return sampled_poses, sampled_trans


def _yaw_rotation(angle: float) -> np.ndarray:
    cosine = math.cos(angle)
    sine = math.sin(angle)
    return np.array(
        [[cosine, 0.0, sine], [0.0, 1.0, 0.0], [-sine, 0.0, cosine]],
        dtype=np.float64,
    )


def build_router_features(
    poses: np.ndarray,
    trans: np.ndarray,
    fps: float,
) -> np.ndarray:
    pose_values, trans_values, fps_value = _validate_pose_trans(poses, trans, fps)
    rotations = axis_angle_to_matrix(pose_values[:, :66].reshape(-1, 22, 3))

    initial_forward = rotations[0, 0] @ np.array([0.0, 0.0, 1.0])
    initial_yaw = math.atan2(float(initial_forward[0]), float(initial_forward[2]))
    yaw_inverse = _yaw_rotation(-initial_yaw)
    rotations[:, 0] = yaw_inverse @ rotations[:, 0]

    local_translation = (trans_values - trans_values[0]) @ yaw_inverse.T
    velocity = np.zeros_like(local_translation)
    if len(local_translation) > 1:
        velocity[1:] = np.diff(local_translation, axis=0) * fps_value
        velocity[0] = velocity[1]

    rotation_features = matrix_to_rotation_6d(rotations).reshape(len(rotations), 132)
    features = np.concatenate((rotation_features, velocity), axis=1).astype(np.float32)
    if features.shape[1] != 135 or not np.isfinite(features).all():
        raise ValueError("Feature extraction produced an invalid array")
    return features


def window_starts(
    n_frames: int,
    window_frames: int,
    stride_frames: int,
) -> list[int]:
    if n_frames < 0 or window_frames <= 0 or stride_frames <= 0:
        raise ValueError("Frame counts and stride must be valid positive values")
    if n_frames < window_frames:
        return []
    last_start = n_frames - window_frames
    starts = list(range(0, last_start + 1, stride_frames))
    if starts[-1] != last_start:
        starts.append(last_start)
    return starts
