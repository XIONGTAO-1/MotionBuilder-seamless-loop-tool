import numpy as np

from core.motion_features import JOINT_NAMES
from mobu.motion_classifier import (
    CharacterMotion,
    MotionBuilderCharacterSampler,
    MotionClassifier,
    MotionSamplingError,
    resample_character_motion,
)


class _BodyNodes:
    kFBHipsNodeId = "hips"
    kFBLeftHipNodeId = "left_hip"
    kFBRightHipNodeId = "right_hip"
    kFBWaistNodeId = "spine"
    kFBLeftKneeNodeId = "left_knee"
    kFBRightKneeNodeId = "right_knee"
    kFBLeftAnkleNodeId = "left_ankle"
    kFBRightAnkleNodeId = "right_ankle"
    kFBLeftFootNodeId = "left_toe"
    kFBRightFootNodeId = "right_toe"
    kFBNeckNodeId = "neck"
    kFBHeadNodeId = "head"
    kFBLeftShoulderNodeId = "left_shoulder"
    kFBRightShoulderNodeId = "right_shoulder"
    kFBLeftElbowNodeId = "left_elbow"
    kFBRightElbowNodeId = "right_elbow"
    kFBLeftWristNodeId = "left_wrist"
    kFBRightWristNodeId = "right_wrist"


class _Character:
    def __init__(self, missing=None, characterized=True):
        self.missing = missing
        self.characterized = characterized
        self.calls = []

    def GetCharacterize(self):
        return self.characterized

    def GetModel(self, body_node_id):
        self.calls.append(body_node_id)
        return None if body_node_id == self.missing else body_node_id


class _Adapter:
    def __init__(self):
        self.current_time = 0

    def get_frame_range(self):
        return 0, 44

    def get_current_fps(self):
        return 30.0

    def _get_local_time(self):
        return self.current_time

    def _set_local_time(self, time):
        self.current_time = time
        return True

    def _evaluate_scene(self):
        pass

    def _get_model_matrix(self, model, time=None):
        matrix = np.eye(4)
        if model == "hips":
            matrix[:3, 3] = [100.0 + self.current_time, 50.0, -25.0]
        return matrix

    def _matrix_to_array(self, matrix):
        return np.asarray(matrix)

    def _matrix_translation(self, matrix):
        return tuple(np.asarray(matrix)[:3, 3])


def test_sampler_uses_standard_body_node_order_and_converts_centimeters():
    character = _Character()
    sampler = MotionBuilderCharacterSampler(
        adapter=_Adapter(), body_node_type=_BodyNodes, time_factory=lambda frame: frame
    )
    motion = sampler.sample(character)

    assert motion.global_rotations.shape == (45, 18, 3, 3)
    assert motion.root_positions_m.shape == (45, 3)
    assert character.calls == [
        "hips",
        "left_hip",
        "right_hip",
        "spine",
        "left_knee",
        "right_knee",
        "left_ankle",
        "right_ankle",
        "left_toe",
        "right_toe",
        "neck",
        "head",
        "left_shoulder",
        "right_shoulder",
        "left_elbow",
        "right_elbow",
        "left_wrist",
        "right_wrist",
    ]
    np.testing.assert_allclose(motion.root_positions_m[0], [1.0, 0.5, -0.25])
    np.testing.assert_allclose(motion.root_positions_m[-1], [1.44, 0.5, -0.25])


def test_sampler_restores_timeline_and_never_passes_time_as_evaluate_info():
    class TimelineAdapter:
        def __init__(self):
            self.current_time = 17
            self.evaluated_times = []

        def get_frame_range(self):
            return 0, 44

        def get_current_fps(self):
            return 30.0

        def _get_local_time(self):
            return self.current_time

        def _set_local_time(self, time):
            self.current_time = time
            return True

        def _evaluate_scene(self):
            self.evaluated_times.append(self.current_time)

        def _get_model_matrix(self, model, time=None):
            if time is not None:
                raise AssertionError("FBTime must not be passed as FBEvaluateInfo")
            matrix = np.eye(4)
            if model == "hips":
                matrix[:3, 3] = [100.0 + self.current_time, 50.0, -25.0]
            return matrix

        def _matrix_to_array(self, matrix):
            return np.asarray(matrix)

        def _matrix_translation(self, matrix):
            return tuple(np.asarray(matrix)[:3, 3])

    adapter = TimelineAdapter()
    sampler = MotionBuilderCharacterSampler(
        adapter=adapter,
        body_node_type=_BodyNodes,
        time_factory=lambda frame: frame,
    )

    motion = sampler.sample(_Character())

    assert adapter.current_time == 17
    assert adapter.evaluated_times[-1] == 17
    np.testing.assert_allclose(motion.root_positions_m[0], [1.0, 0.5, -0.25])
    np.testing.assert_allclose(motion.root_positions_m[-1], [1.44, 0.5, -0.25])


def test_sampler_rejects_uncharacterized_or_missing_required_joint():
    sampler = MotionBuilderCharacterSampler(
        adapter=_Adapter(), body_node_type=_BodyNodes, time_factory=lambda frame: frame
    )
    try:
        sampler.sample(_Character(characterized=False))
    except MotionSamplingError as exc:
        assert exc.reason == "character_not_characterized"
    else:
        raise AssertionError("Expected uncharacterized character failure")

    try:
        sampler.sample(_Character(missing="left_toe"))
    except MotionSamplingError as exc:
        assert exc.reason == "missing_required_joint:LeftToeBase"
    else:
        raise AssertionError("Expected missing joint failure")


def test_resampling_to_30_fps_preserves_duration_and_position():
    rotations = np.broadcast_to(np.eye(3), (89, 18, 3, 3)).copy()
    positions = np.zeros((89, 3))
    positions[:, 0] = np.arange(89) / 60.0
    motion = resample_character_motion(rotations, positions, source_fps=60.0)
    assert motion.global_rotations.shape == (45, 18, 3, 3)
    np.testing.assert_allclose(motion.root_positions_m[:, 0], np.arange(45) / 30.0)


def test_rotation_resampling_uses_geodesic_interpolation():
    rotations = np.broadcast_to(np.eye(3), (2, 18, 3, 3)).copy()
    rotations[1, :, 0, 0] = 0.0
    rotations[1, :, 0, 2] = 1.0
    rotations[1, :, 2, 0] = -1.0
    rotations[1, :, 2, 2] = 0.0
    positions = np.zeros((2, 3))
    motion = resample_character_motion(
        rotations, positions, source_fps=1.0, target_fps=2.0
    )
    expected_midpoint = np.array(
        [
            [np.sqrt(0.5), 0.0, np.sqrt(0.5)],
            [0.0, 1.0, 0.0],
            [-np.sqrt(0.5), 0.0, np.sqrt(0.5)],
        ]
    )
    np.testing.assert_allclose(motion.global_rotations[1, 0], expected_midpoint, atol=1e-7)


class _Forest:
    classes = ("other", "run", "walk")
    thresholds = {"walk": 0.6, "run": 0.7}

    def predict_proba(self, features):
        return np.tile([0.8, 0.1, 0.1], (len(features), 1))


class _ShortSampler:
    def sample(self, character, frame_range=None):
        return CharacterMotion(
            np.broadcast_to(np.eye(3), (44, 18, 3, 3)).copy(),
            np.zeros((44, 3)),
            30.0,
        )


class _FailingSampler:
    def sample(self, character, frame_range=None):
        raise MotionSamplingError("missing_required_joint:Head")


class _NonFiniteSampler:
    def sample(self, character, frame_range=None):
        positions = np.zeros((45, 3))
        positions[3, 1] = np.nan
        return CharacterMotion(
            np.broadcast_to(np.eye(3), (45, 18, 3, 3)).copy(),
            positions,
            30.0,
        )


def test_classifier_returns_safe_other_for_short_or_invalid_character():
    short = MotionClassifier(_Forest(), sampler=_ShortSampler()).classify_character(object())
    assert short.label == "other"
    assert short.window_count == 0
    assert short.reason == "clip_too_short:44_frames"

    missing = MotionClassifier(_Forest(), sampler=_FailingSampler()).classify_character(object())
    assert missing.label == "other"
    assert missing.reason == "missing_required_joint:Head"

    non_finite = MotionClassifier(_Forest(), sampler=_NonFiniteSampler()).classify_character(object())
    assert non_finite.label == "other"
    assert "finite" in non_finite.reason
