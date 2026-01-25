"""
Tests for mobu/adapter module.

Following TDD: We test MockMoBuAdapter which can run outside MotionBuilder.
The real MoBuAdapter shares the same interface and is tested manually in MoBu.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mobu.adapter import MockMoBuAdapter
import mobu.adapter as adapter_module


class TestMockMoBuAdapter:
    """Test suite for MockMoBuAdapter (allows testing outside MoBu)."""

    def test_get_frame_range_returns_tuple(self):
        """get_frame_range should return start and end frames."""
        adapter = MockMoBuAdapter(mock_frame_range=(10, 200))
        
        start, end = adapter.get_frame_range()
        
        assert start == 10
        assert end == 200

    def test_get_root_trajectory_returns_correct_shape(self):
        """get_root_trajectory should return (num_frames, 6) array."""
        mock_data = np.random.rand(50, 6)
        adapter = MockMoBuAdapter(mock_trajectory=mock_data)
        
        trajectory = adapter.get_root_trajectory("Hips")
        
        assert trajectory.shape == (50, 6)
        np.testing.assert_array_almost_equal(trajectory, mock_data)

    def test_get_root_trajectory_returns_copy(self):
        """get_root_trajectory should return a copy, not the original."""
        mock_data = np.random.rand(10, 6)
        adapter = MockMoBuAdapter(mock_trajectory=mock_data)
        
        trajectory = adapter.get_root_trajectory("Hips")
        trajectory[0, 0] = 999.0  # Modify the returned array
        
        # Original should be unchanged
        original = adapter.get_root_trajectory("Hips")
        assert original[0, 0] != 999.0

    def test_set_root_trajectory_updates_data(self):
        """set_root_trajectory should update the stored trajectory."""
        adapter = MockMoBuAdapter()
        new_trajectory = np.array([
            [1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
            [4.0, 5.0, 6.0, 0.0, 0.0, 0.0],
        ])
        
        adapter.set_root_trajectory("Hips", new_trajectory)
        result = adapter.get_root_trajectory("Hips")
        
        np.testing.assert_array_almost_equal(result, new_trajectory)

    def test_get_all_bone_poses_returns_identity_by_default(self):
        """get_all_bone_poses should return identity matrix by default."""
        adapter = MockMoBuAdapter()
        
        poses = adapter.get_all_bone_poses(frame=0)
        
        assert "Hips" in poses
        np.testing.assert_array_almost_equal(poses["Hips"], np.eye(4))

    def test_set_mock_poses_allows_custom_data(self):
        """set_mock_poses should allow setting custom pose data for testing."""
        adapter = MockMoBuAdapter()
        custom_matrix = np.array([
            [1, 0, 0, 5],
            [0, 1, 0, 10],
            [0, 0, 1, 15],
            [0, 0, 0, 1],
        ], dtype=float)
        
        adapter.set_mock_poses(frame=10, poses={"Hips": custom_matrix})
        result = adapter.get_all_bone_poses(frame=10)
        
        np.testing.assert_array_almost_equal(result["Hips"], custom_matrix)

    def test_generate_unique_take_name_appends_suffix(self):
        """generate_unique_take_name should append numeric suffix if needed."""
        name = adapter_module.generate_unique_take_name(
            "Walk",
            ["Walk_inplace", "Walk_inplace_1"],
            suffix="_inplace",
        )

        assert name == "Walk_inplace_2"


class TestMoBuAdapterSetRootTrajectory:
    """Tests for MoBuAdapter with stubbed MotionBuilder objects."""

    def test_set_root_trajectory_writes_rotation_keys(self, monkeypatch):
        """set_root_trajectory should write rotation keys even if nodes were static."""
        class FakeTime:
            def __init__(self, _h=0, _m=0, _s=0, frame=0):
                self._frame = frame

            def GetFrame(self):
                return self._frame

        class FakeTimeSpan:
            def __init__(self, start, stop):
                self._start = start
                self._stop = stop

            def GetStart(self):
                return self._start

            def GetStop(self):
                return self._stop

            def SetStart(self, time):
                self._start = time

            def SetStop(self, time):
                self._stop = time

        class FakeFCurve:
            def __init__(self):
                self.key_add_calls = []
                self._keys = []
                self.interpolation_calls = []
                self.tangent_calls = []

            def KeyAdd(self, time, value):
                frame = time.GetFrame()
                self.key_add_calls.append((frame, value))
                self._keys.append((frame, value))
                return len(self._keys) - 1

            def KeySetInterpolation(self, index, mode):
                self.interpolation_calls.append((index, mode))

            def KeySetTangentMode(self, index, mode, *args):
                self.tangent_calls.append((index, mode))

            def KeyGetCount(self):
                return len(self._keys)

            def KeyGetTime(self, index):
                return FakeTime(frame=self._keys[index][0])

            def KeyRemove(self, index):
                self._keys.pop(index)

        class FakeNode:
            def __init__(self):
                self.FCurve = FakeFCurve()

        class FakeAnimNode:
            def __init__(self):
                self.Nodes = [FakeNode(), FakeNode(), FakeNode()]

        class FakeProperty:
            def __init__(self):
                self._node = None

            def SetAnimated(self, value):
                if value and self._node is None:
                    self._node = FakeAnimNode()

            def GetAnimationNode(self):
                return self._node

        class FakeModel:
            def __init__(self):
                self.Translation = FakeProperty()
                self.Rotation = FakeProperty()

        class FakeTake:
            def __init__(self):
                self.LocalTimeSpan = FakeTimeSpan(FakeTime(frame=0), FakeTime(frame=0))

        class FakeSystem:
            def __init__(self):
                self.CurrentTake = FakeTake()

        class FakePlayer:
            def __init__(self):
                self.LoopStart = FakeTime(frame=0)
                self.LoopStop = FakeTime(frame=0)

        fake_model = FakeModel()

        class FakeInterpolation:
            kFBInterpolationCubic = "cubic"

        class FakeTangentMode:
            kFBTangentModeClampProgressive = "clamp"

        monkeypatch.setattr(adapter_module, "IN_MOTIONBUILDER", True)
        monkeypatch.setattr(adapter_module, "FBSystem", FakeSystem)
        monkeypatch.setattr(adapter_module, "FBPlayerControl", FakePlayer)
        monkeypatch.setattr(adapter_module, "FBFindModelByLabelName", lambda name: fake_model)
        monkeypatch.setattr(adapter_module, "FBFindModelByName", lambda name: fake_model)
        monkeypatch.setattr(adapter_module, "FBTime", FakeTime)
        monkeypatch.setattr(adapter_module, "FBTimeSpan", FakeTimeSpan)
        monkeypatch.setattr(adapter_module, "FBInterpolation", FakeInterpolation)
        monkeypatch.setattr(adapter_module, "FBTangentMode", FakeTangentMode)

        adapter = adapter_module.MoBuAdapter()

        trajectory = np.array([
            [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            [4.0, 5.0, 6.0, 40.0, 50.0, 60.0],
        ])

        # NOTE: After Reset & Inject refactor, set_root_trajectory always writes from Frame 0
        adapter.set_root_trajectory("Hips", trajectory, start_frame=10)  # start_frame is now ignored

        rotation_node = fake_model.Rotation.GetAnimationNode()
        rot_x = rotation_node.Nodes[0].FCurve.key_add_calls
        rot_y = rotation_node.Nodes[1].FCurve.key_add_calls
        rot_z = rotation_node.Nodes[2].FCurve.key_add_calls

        # After refactor: always writes to Frame 0, 1, 2...
        assert rot_x == [(0, 10.0), (1, 40.0)]
        assert rot_y == [(0, 20.0), (1, 50.0)]
        assert rot_z == [(0, 30.0), (1, 60.0)]

        assert all(isinstance(frame, int) for frame, _ in rot_x)
        assert all(isinstance(value, float) for _, value in rot_x)

        interp_calls = rotation_node.Nodes[0].FCurve.interpolation_calls
        tangent_calls = rotation_node.Nodes[0].FCurve.tangent_calls
        assert interp_calls
        assert tangent_calls


class TestMoBuAdapterWorldTranslations:
    """Tests for world-translation sampling with GetMatrix API differences."""

    def test_get_world_translations_falls_back_without_time_arg(self, monkeypatch):
        class FakeTime:
            def __init__(self, _h=0, _m=0, _s=0, frame=0, *_args):
                self._frame = frame

            def GetFrame(self):
                return self._frame

        class FakeMatrix:
            def __init__(self):
                self._data = [[0.0] * 4 for _ in range(4)]

            def __getitem__(self, idx):
                return self._data[idx]

        class FakeSystem:
            last_instance = None

            def __init__(self):
                self._local_time = FakeTime(frame=0)
                FakeSystem.last_instance = self

            @property
            def LocalTime(self):
                return self._local_time

            @LocalTime.setter
            def LocalTime(self, value):
                self._local_time = value

        class FakePlayer:
            def Goto(self, time):
                FakeSystem.last_instance.LocalTime = time

        class FakeModelTransformationType:
            kModelTransformation = "xform"

        class FakeModel:
            def __init__(self):
                self.Translation = object()

            def GetMatrix(self, matrix, _xform_type=None, _world=True):
                frame = FakeSystem.last_instance.LocalTime.GetFrame()
                matrix[0][3] = float(frame)
                matrix[1][3] = float(frame + 10)
                matrix[2][3] = float(frame + 20)

        fake_model = FakeModel()

        monkeypatch.setattr(adapter_module, "IN_MOTIONBUILDER", True)
        monkeypatch.setattr(adapter_module, "FBSystem", FakeSystem)
        monkeypatch.setattr(adapter_module, "FBPlayerControl", FakePlayer)
        monkeypatch.setattr(adapter_module, "FBFindModelByLabelName", lambda name: fake_model)
        monkeypatch.setattr(adapter_module, "FBFindModelByName", lambda name: fake_model)
        monkeypatch.setattr(adapter_module, "FBTime", FakeTime)
        monkeypatch.setattr(adapter_module, "FBMatrix", FakeMatrix)
        monkeypatch.setattr(
            adapter_module,
            "FBModelTransformationType",
            FakeModelTransformationType,
        )

        adapter = adapter_module.MoBuAdapter()

        positions = adapter.get_world_translations("Hips", start_frame=0, end_frame=2)

        assert positions.shape == (3, 3)
        assert positions[:, 0].tolist() == [0.0, 1.0, 2.0]
        assert positions[:, 1].tolist() == [10.0, 11.0, 12.0]
        assert positions[:, 2].tolist() == [20.0, 21.0, 22.0]

    def test_get_world_translations_handles_flat_matrix(self, monkeypatch):
        class FakeTime:
            def __init__(self, _h=0, _m=0, _s=0, frame=0, *_args):
                self._frame = frame

            def GetFrame(self):
                return self._frame

        class FakeMatrix:
            def __init__(self):
                self._data = [0.0] * 16

            def __getitem__(self, idx):
                return self._data[idx]

        class FakeSystem:
            last_instance = None

            def __init__(self):
                self._local_time = FakeTime(frame=0)
                FakeSystem.last_instance = self

            @property
            def LocalTime(self):
                return self._local_time

            @LocalTime.setter
            def LocalTime(self, value):
                self._local_time = value

        class FakePlayer:
            def Goto(self, time):
                FakeSystem.last_instance.LocalTime = time

        class FakeModelTransformationType:
            kModelTransformation = "xform"

        class FakeModel:
            def __init__(self):
                self.Translation = object()

            def GetMatrix(self, matrix, _xform_type=None, _world=True):
                frame = FakeSystem.last_instance.LocalTime.GetFrame()
                matrix._data[3] = float(frame)
                matrix._data[7] = float(frame + 10)
                matrix._data[11] = float(frame + 20)

        fake_model = FakeModel()

        monkeypatch.setattr(adapter_module, "IN_MOTIONBUILDER", True)
        monkeypatch.setattr(adapter_module, "FBSystem", FakeSystem)
        monkeypatch.setattr(adapter_module, "FBPlayerControl", FakePlayer)
        monkeypatch.setattr(adapter_module, "FBFindModelByLabelName", lambda name: fake_model)
        monkeypatch.setattr(adapter_module, "FBFindModelByName", lambda name: fake_model)
        monkeypatch.setattr(adapter_module, "FBTime", FakeTime)
        monkeypatch.setattr(adapter_module, "FBMatrix", FakeMatrix)
        monkeypatch.setattr(
            adapter_module,
            "FBModelTransformationType",
            FakeModelTransformationType,
        )

        adapter = adapter_module.MoBuAdapter()

        positions = adapter.get_world_translations("Hips", start_frame=0, end_frame=2)

        assert positions.shape == (3, 3)
        assert positions[:, 0].tolist() == [0.0, 1.0, 2.0]
        assert positions[:, 1].tolist() == [10.0, 11.0, 12.0]
        assert positions[:, 2].tolist() == [20.0, 21.0, 22.0]

    def test_plot_animation_uses_custom_time_mode_without_fps_arg(self, monkeypatch):
        """plot_animation_on_skeleton should not pass an fps value to FBTime."""
        calls = []

        class FakeTime:
            def __init__(self, *args):
                calls.append(args)
                if len(args) == 7:
                    raise TypeError("FBTime received fps arg")

        class FakePlotOptions:
            def __init__(self):
                self.PlotAllTakes = False
                self.PlotOnFrame = False
                self.UseConstantKeyReducer = False
                self.PlotPeriod = None

        class FakeScene:
            Characters = []

        class FakeSystem:
            def __init__(self):
                self.Scene = FakeScene()

        class FakePlayer:
            pass

        class FakeTimeMode:
            kFBTimeModeCustom = "custom"

        monkeypatch.setattr(adapter_module, "IN_MOTIONBUILDER", True)
        monkeypatch.setattr(adapter_module, "FBSystem", FakeSystem)
        monkeypatch.setattr(adapter_module, "FBPlayerControl", FakePlayer)
        monkeypatch.setattr(adapter_module, "FBTime", FakeTime)
        monkeypatch.setattr(adapter_module, "FBPlotOptions", FakePlotOptions)
        monkeypatch.setattr(adapter_module, "FBTimeMode", FakeTimeMode)
        monkeypatch.setattr(adapter_module, "FBCharacterPlotWhere", None)

        adapter = adapter_module.MoBuAdapter()
        adapter.plot_animation_on_skeleton(60.0)

        assert calls
        assert len(calls[-1]) == 6
