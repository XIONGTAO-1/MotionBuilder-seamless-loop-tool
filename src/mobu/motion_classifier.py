"""MotionBuilder sampling and safe walk/run/other classification."""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from core.motion_classifier import (
    ClassificationResult,
    NumpyRandomForest,
    classify_feature_windows,
)
from core.motion_features import (
    JOINT_NAMES,
    TARGET_FPS,
    WINDOW_FRAMES,
    extract_motion_descriptor,
)
from . import adapter as adapter_module


@dataclass(frozen=True)
class CharacterMotion:
    global_rotations: np.ndarray
    root_positions_m: np.ndarray
    fps: float


class MotionSamplingError(RuntimeError):
    def __init__(self, reason: str):
        super().__init__(reason)
        self.reason = reason


_BODY_NODE_ATTRIBUTES = (
    "kFBHipsNodeId",
    "kFBLeftHipNodeId",
    "kFBRightHipNodeId",
    "kFBWaistNodeId",
    "kFBLeftKneeNodeId",
    "kFBRightKneeNodeId",
    "kFBLeftAnkleNodeId",
    "kFBRightAnkleNodeId",
    "kFBLeftFootNodeId",
    "kFBRightFootNodeId",
    "kFBNeckNodeId",
    "kFBHeadNodeId",
    "kFBLeftShoulderNodeId",
    "kFBRightShoulderNodeId",
    "kFBLeftElbowNodeId",
    "kFBRightElbowNodeId",
    "kFBLeftWristNodeId",
    "kFBRightWristNodeId",
)


def _matrix_to_quaternion(matrix: np.ndarray) -> np.ndarray:
    values = np.asarray(matrix, dtype=np.float64)
    flat = values.reshape(-1, 3, 3)
    output = np.empty((len(flat), 4), dtype=np.float64)
    for index, rotation in enumerate(flat):
        trace = float(np.trace(rotation))
        if trace > 0.0:
            scale = math.sqrt(trace + 1.0) * 2.0
            quaternion = np.array(
                [
                    0.25 * scale,
                    (rotation[2, 1] - rotation[1, 2]) / scale,
                    (rotation[0, 2] - rotation[2, 0]) / scale,
                    (rotation[1, 0] - rotation[0, 1]) / scale,
                ]
            )
        else:
            diagonal = np.diag(rotation)
            axis = int(np.argmax(diagonal))
            first, second = (axis + 1) % 3, (axis + 2) % 3
            scale = math.sqrt(max(1e-15, 1.0 + diagonal[axis] - diagonal[first] - diagonal[second])) * 2.0
            quaternion = np.empty(4, dtype=np.float64)
            quaternion[axis + 1] = 0.25 * scale
            quaternion[0] = (rotation[second, first] - rotation[first, second]) / scale
            quaternion[first + 1] = (rotation[first, axis] + rotation[axis, first]) / scale
            quaternion[second + 1] = (rotation[second, axis] + rotation[axis, second]) / scale
        output[index] = quaternion / np.linalg.norm(quaternion)
    return output.reshape(values.shape[:-2] + (4,))


def _quaternion_to_matrix(quaternion: np.ndarray) -> np.ndarray:
    values = np.asarray(quaternion, dtype=np.float64)
    values = values / np.linalg.norm(values, axis=-1, keepdims=True)
    w, x, y, z = np.moveaxis(values, -1, 0)
    return np.stack(
        (
            1.0 - 2.0 * (y * y + z * z),
            2.0 * (x * y - z * w),
            2.0 * (x * z + y * w),
            2.0 * (x * y + z * w),
            1.0 - 2.0 * (x * x + z * z),
            2.0 * (y * z - x * w),
            2.0 * (x * z - y * w),
            2.0 * (y * z + x * w),
            1.0 - 2.0 * (x * x + y * y),
        ),
        axis=-1,
    ).reshape(values.shape[:-1] + (3, 3))


def _slerp(first: np.ndarray, second: np.ndarray, fraction: np.ndarray) -> np.ndarray:
    dot = np.sum(first * second, axis=-1, keepdims=True)
    second = np.where(dot < 0.0, -second, second)
    dot = np.clip(np.abs(dot), 0.0, 1.0)
    angle = np.arccos(dot)
    sine = np.sin(angle)
    linear = sine < 1e-8
    safe_sine = np.where(linear, 1.0, sine)
    first_weight = np.where(
        linear, 1.0 - fraction, np.sin((1.0 - fraction) * angle) / safe_sine
    )
    second_weight = np.where(
        linear, fraction, np.sin(fraction * angle) / safe_sine
    )
    result = first_weight * first + second_weight * second
    return result / np.linalg.norm(result, axis=-1, keepdims=True)


def resample_character_motion(
    global_rotations: np.ndarray,
    root_positions_m: np.ndarray,
    source_fps: float,
    target_fps: float = TARGET_FPS,
) -> CharacterMotion:
    rotations = np.asarray(global_rotations, dtype=np.float64)
    positions = np.asarray(root_positions_m, dtype=np.float64)
    source = float(source_fps)
    target = float(target_fps)
    if rotations.ndim != 4 or rotations.shape[1:] != (len(JOINT_NAMES), 3, 3):
        raise MotionSamplingError("invalid_rotation_shape")
    if positions.shape != (len(rotations), 3):
        raise MotionSamplingError("invalid_root_position_shape")
    if not math.isfinite(source) or source <= 0 or not math.isfinite(target) or target <= 0:
        raise MotionSamplingError("invalid_frame_rate")
    if not np.isfinite(rotations).all() or not np.isfinite(positions).all():
        raise MotionSamplingError("non_finite_motion")
    if len(rotations) == 0:
        raise MotionSamplingError("empty_frame_range")
    if source == target or len(rotations) == 1:
        return CharacterMotion(rotations.copy(), positions.copy(), target)

    duration = (len(rotations) - 1) / source
    target_count = int(math.floor(duration * target + 1e-9)) + 1
    source_times = np.arange(len(rotations), dtype=np.float64) / source
    target_times = np.arange(target_count, dtype=np.float64) / target
    upper = np.searchsorted(source_times, target_times, side="right")
    upper = np.clip(upper, 1, len(source_times) - 1)
    lower = upper - 1
    interval = source_times[upper] - source_times[lower]
    fraction = ((target_times - source_times[lower]) / interval)[:, None, None]
    quaternions = _matrix_to_quaternion(rotations)
    sampled_rotations = _quaternion_to_matrix(
        _slerp(quaternions[lower], quaternions[upper], fraction)
    )
    sampled_positions = np.column_stack(
        [np.interp(target_times, source_times, positions[:, axis]) for axis in range(3)]
    )
    return CharacterMotion(sampled_rotations, sampled_positions, target)


def _orthonormal_rotation(matrix: np.ndarray) -> np.ndarray:
    rotation = np.asarray(matrix, dtype=np.float64)[:3, :3]
    left, _, right = np.linalg.svd(rotation)
    result = left @ right
    if np.linalg.det(result) < 0:
        left[:, -1] *= -1.0
        result = left @ right
    return result


class MotionBuilderCharacterSampler:
    """Sample the 18 canonical characterized body nodes in world space."""

    def __init__(
        self,
        adapter=None,
        body_node_type=None,
        time_factory: Callable[[int], object] | None = None,
    ):
        self._adapter = adapter
        self._body_node_type = body_node_type
        self._time_factory = time_factory

    def _runtime_dependencies(self):
        adapter = self._adapter
        if adapter is None:
            adapter = adapter_module.MoBuAdapter()
            self._adapter = adapter
        body_node_type = self._body_node_type or adapter_module.FBBodyNodeId
        if body_node_type is None:
            raise MotionSamplingError("motionbuilder_body_node_api_unavailable")
        time_factory = self._time_factory
        if time_factory is None:
            if adapter_module.FBTime is None:
                raise MotionSamplingError("motionbuilder_time_api_unavailable")
            time_factory = lambda frame: adapter_module.FBTime(0, 0, 0, frame)
        return adapter, body_node_type, time_factory

    def sample(
        self,
        character,
        frame_range: tuple[int, int] | None = None,
    ) -> CharacterMotion:
        if character is None or not callable(getattr(character, "GetCharacterize", None)):
            raise MotionSamplingError("character_not_characterized")
        try:
            characterized = bool(character.GetCharacterize())
        except Exception as exc:
            raise MotionSamplingError("characterization_state_unavailable") from exc
        if not characterized:
            raise MotionSamplingError("character_not_characterized")

        adapter, body_node_type, time_factory = self._runtime_dependencies()
        models = []
        for joint_name, attribute in zip(JOINT_NAMES, _BODY_NODE_ATTRIBUTES):
            try:
                body_node_id = getattr(body_node_type, attribute)
                model = character.GetModel(body_node_id)
            except Exception as exc:
                raise MotionSamplingError(f"body_node_lookup_failed:{joint_name}") from exc
            if model is None:
                raise MotionSamplingError(f"missing_required_joint:{joint_name}")
            models.append(model)

        start_frame, end_frame = frame_range or adapter.get_frame_range()
        start_frame, end_frame = int(start_frame), int(end_frame)
        if end_frame < start_frame:
            raise MotionSamplingError("invalid_frame_range")
        frame_count = end_frame - start_frame + 1
        rotations = np.empty((frame_count, len(models), 3, 3), dtype=np.float64)
        positions_cm = np.empty((frame_count, 3), dtype=np.float64)

        previous_time = adapter._get_local_time()
        try:
            for output_index, frame in enumerate(range(start_frame, end_frame + 1)):
                time = time_factory(frame)
                if not adapter._set_local_time(time):
                    raise MotionSamplingError("timeline_seek_failed")
                adapter._evaluate_scene()
                for joint_index, model in enumerate(models):
                    try:
                        matrix = adapter._get_model_matrix(model)
                        array = adapter._matrix_to_array(matrix)
                    except Exception as exc:
                        raise MotionSamplingError(
                            f"matrix_sampling_failed:{JOINT_NAMES[joint_index]}"
                        ) from exc
                    if array is None or np.asarray(array).shape != (4, 4):
                        raise MotionSamplingError(
                            f"invalid_world_matrix:{JOINT_NAMES[joint_index]}"
                        )
                    rotations[output_index, joint_index] = _orthonormal_rotation(array)
                    if joint_index == 0:
                        translation = adapter._matrix_translation(matrix)
                        if translation is None:
                            raise MotionSamplingError("invalid_hips_translation")
                        positions_cm[output_index] = translation
        finally:
            if previous_time is not None:
                adapter._set_local_time(previous_time)
                adapter._evaluate_scene()

        source_fps = float(adapter.get_current_fps())
        return resample_character_motion(rotations, positions_cm * 0.01, source_fps)


def _fallback(reason: str) -> ClassificationResult:
    return ClassificationResult(
        label="other",
        confidence=1.0,
        probabilities={"walk": 0.0, "run": 0.0, "other": 1.0},
        window_count=0,
        reason=reason,
    )


def _window_starts(frame_count: int) -> list[int]:
    if frame_count < WINDOW_FRAMES:
        return []
    final_start = frame_count - WINDOW_FRAMES
    starts = list(range(0, final_start + 1, 15))
    if starts[-1] != final_start:
        starts.append(final_start)
    return starts


class MotionClassifier:
    def __init__(
        self,
        model: Path | NumpyRandomForest,
        *,
        sampler=None,
        allow_experimental: bool = False,
    ):
        self._sampler = sampler
        self._model_error: str | None = None
        if isinstance(model, (str, Path)):
            try:
                self._forest = NumpyRandomForest.load(
                    Path(model), allow_experimental=allow_experimental
                )
            except Exception as exc:
                self._forest = None
                self._model_error = f"model_load_failed:{exc}"
        else:
            self._forest = model

    def classify_character(
        self,
        character,
        frame_range: tuple[int, int] | None = None,
    ) -> ClassificationResult:
        if self._forest is None:
            return _fallback(self._model_error or "model_unavailable")
        sampler = self._sampler
        if sampler is None:
            sampler = MotionBuilderCharacterSampler()
            self._sampler = sampler
        try:
            motion = sampler.sample(character, frame_range=frame_range)
            if motion.fps != TARGET_FPS:
                motion = resample_character_motion(
                    motion.global_rotations,
                    motion.root_positions_m,
                    motion.fps,
                )
            frame_count = len(motion.global_rotations)
            starts = _window_starts(frame_count)
            if not starts:
                return _fallback(f"clip_too_short:{frame_count}_frames")
            descriptors = np.stack(
                [
                    extract_motion_descriptor(
                        motion.global_rotations[start : start + WINDOW_FRAMES],
                        motion.root_positions_m[start : start + WINDOW_FRAMES],
                        fps=TARGET_FPS,
                    )
                    for start in starts
                ]
            )
            return classify_feature_windows(self._forest, descriptors)
        except MotionSamplingError as exc:
            return _fallback(exc.reason)
        except (ValueError, FloatingPointError) as exc:
            return _fallback(f"invalid_motion:{exc}")
        except Exception as exc:
            return _fallback(f"classification_failed:{type(exc).__name__}")
