import hashlib
import json

import numpy as np
import pytest

from core.motion_features import (
    DESCRIPTOR_SIZE,
    JOINT_NAMES,
    SMPL_22_PARENTS,
    build_feature_schema,
    extract_motion_descriptor,
    global_rotations_from_local,
    rotation_6d_to_matrix,
)


def _rotation_y(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]])


def _motion() -> tuple[np.ndarray, np.ndarray]:
    frames = 45
    rotations = np.empty((frames, len(JOINT_NAMES), 3, 3), dtype=np.float64)
    for frame in range(frames):
        for joint in range(len(JOINT_NAMES)):
            rotations[frame, joint] = _rotation_y(0.01 * frame * (joint + 1))
    positions = np.column_stack(
        (
            np.linspace(0.0, 1.2, frames),
            0.04 * np.sin(np.linspace(0.0, 4.0 * np.pi, frames)),
            np.linspace(0.0, 0.3, frames),
        )
    )
    return rotations, positions


def test_descriptor_schema_is_stable_and_complete():
    rotations, positions = _motion()
    descriptor = extract_motion_descriptor(rotations, positions)
    schema = build_feature_schema()

    assert descriptor.shape == (DESCRIPTOR_SIZE,) == (175,)
    assert descriptor.dtype == np.float32
    assert np.isfinite(descriptor).all()
    assert len(schema["feature_names"]) == DESCRIPTOR_SIZE
    assert len(set(schema["feature_names"])) == DESCRIPTOR_SIZE
    assert schema["joint_order"] == list(JOINT_NAMES)
    assert schema["fps"] == 30.0
    assert schema["window_frames"] == 45

    payload = dict(schema)
    digest = payload.pop("sha256")
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    assert digest == hashlib.sha256(canonical).hexdigest()


def test_descriptor_is_invariant_to_translation_initial_heading_and_basis():
    rotations, positions = _motion()
    expected = extract_motion_descriptor(rotations, positions)

    heading = _rotation_y(0.73)
    transformed_rotations = heading @ rotations
    transformed_positions = positions @ heading.T + np.array([12.0, -3.0, 5.0])
    actual = extract_motion_descriptor(transformed_rotations, transformed_positions)
    np.testing.assert_allclose(actual, expected, rtol=2e-5, atol=2e-5)

    basis = _rotation_y(-0.41)
    conjugated_rotations = basis @ rotations @ basis.T
    conjugated_positions = positions @ basis.T
    conjugated = extract_motion_descriptor(conjugated_rotations, conjugated_positions)
    np.testing.assert_allclose(conjugated, expected, rtol=2e-5, atol=2e-5)


def test_descriptor_is_invariant_to_fixed_joint_axis_offsets():
    rotations, positions = _motion()
    offsets = np.stack([_rotation_y(0.03 * joint) for joint in range(len(JOINT_NAMES))])
    adjusted = rotations @ offsets[None, ...]
    np.testing.assert_allclose(
        extract_motion_descriptor(adjusted, positions),
        extract_motion_descriptor(rotations, positions),
        rtol=2e-5,
        atol=2e-5,
    )


def test_descriptor_rejects_wrong_shape_and_non_finite_data():
    rotations, positions = _motion()
    with pytest.raises(ValueError, match="global_rotations"):
        extract_motion_descriptor(rotations[:-1], positions)
    positions[0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        extract_motion_descriptor(rotations, positions)


def test_rotation_6d_and_smpl_global_conversion():
    identity_6d = np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0])
    local = rotation_6d_to_matrix(np.tile(identity_6d, (2, 22, 1)))
    global_rotations = global_rotations_from_local(local, SMPL_22_PARENTS)
    assert global_rotations.shape == (2, 22, 3, 3)
    expected = np.broadcast_to(np.eye(3), global_rotations.shape)
    np.testing.assert_allclose(global_rotations, expected, atol=1e-12)
