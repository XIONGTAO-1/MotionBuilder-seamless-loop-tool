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


def test_get_current_character_uses_application_selection(monkeypatch):
    selected_character = object()

    class FakeApplication:
        CurrentCharacter = selected_character

    monkeypatch.setattr(adapter_module, "FBApplication", FakeApplication)
    adapter = adapter_module.MoBuAdapter.__new__(adapter_module.MoBuAdapter)

    assert adapter.get_current_character() is selected_character


def test_get_current_character_returns_none_without_application_api(monkeypatch):
    monkeypatch.setattr(adapter_module, "FBApplication", None)
    adapter = adapter_module.MoBuAdapter.__new__(adapter_module.MoBuAdapter)

    assert adapter.get_current_character() is None


def test_get_current_character_returns_none_when_application_lookup_fails(monkeypatch):
    def failing_application():
        raise RuntimeError("application unavailable")

    monkeypatch.setattr(adapter_module, "FBApplication", failing_application)
    adapter = adapter_module.MoBuAdapter.__new__(adapter_module.MoBuAdapter)

    assert adapter.get_current_character() is None


class TestMoBuAdapterModelResolution:
    class FakeTime:
        def __init__(self, _h=0, _m=0, _s=0, frame=0, *_args):
            self._frame = frame

        def GetFrame(self):
            return self._frame

    class FakeFCurve:
        def __init__(self, value=0.0):
            self.value = float(value)
            self.key_add_calls = []

        def Evaluate(self, _time):
            return self.value

        def KeyAdd(self, time, value):
            self.key_add_calls.append((time.GetFrame(), value))
            return len(self.key_add_calls) - 1

        def EditClear(self):
            self.key_add_calls.clear()

    class FakeNode:
        def __init__(self, value=0.0):
            self.FCurve = TestMoBuAdapterModelResolution.FakeFCurve(value)

    class FakeAnimationNode:
        def __init__(self, values):
            self.Nodes = [
                TestMoBuAdapterModelResolution.FakeNode(value)
                for value in values
            ]

    class FakeProperty:
        def __init__(self, values):
            self.node = TestMoBuAdapterModelResolution.FakeAnimationNode(values)

        def SetAnimated(self, _value):
            pass

        def GetAnimationNode(self):
            return self.node

    class FakeModel:
        def __init__(self, name, long_name, rotation_values=(0.0, 0.0, 0.0)):
            self.Name = name
            self.LongName = long_name
            self.LabelName = name
            self.Children = []
            self.Translation = TestMoBuAdapterModelResolution.FakeProperty(
                (0.0, 0.0, 0.0)
            )
            self.Rotation = TestMoBuAdapterModelResolution.FakeProperty(
                rotation_values
            )

    def make_adapter(self, monkeypatch, roots):
        scene_root = self.FakeModel("Scene", "Scene")
        scene_root.Children = list(roots)
        scene = type("FakeScene", (), {"RootModel": scene_root})()
        system = type("FakeSystem", (), {"Scene": scene, "CurrentTake": object()})()
        models = []

        def collect(model):
            models.append(model)
            for child in model.Children:
                collect(child)

        for root in roots:
            collect(root)
        by_long_name = {model.LongName: model for model in models}

        monkeypatch.setattr(
            adapter_module,
            "FBFindModelByLabelName",
            lambda name: by_long_name.get(name),
        )
        monkeypatch.setattr(adapter_module, "FBFindModelByName", lambda _name: None)
        monkeypatch.setattr(adapter_module, "FBTime", self.FakeTime)
        adapter = adapter_module.MoBuAdapter.__new__(adapter_module.MoBuAdapter)
        adapter._system = system
        adapter._player = None
        return adapter

    def make_mixamo_hierarchy(self):
        hips = self.FakeModel(
            "Hips",
            "mixamorig:Hips",
            rotation_values=(100.0, 101.0, 102.0),
        )
        upper_leg = self.FakeModel(
            "LeftUpLeg",
            "mixamorig:LeftUpLeg",
            rotation_values=(10.0, 11.0, 12.0),
        )
        lower_leg = self.FakeModel(
            "LeftLeg",
            "mixamorig:LeftLeg",
            rotation_values=(20.0, 21.0, 22.0),
        )
        hips.Children = [upper_leg]
        upper_leg.Children = [lower_leg]
        return hips, upper_leg, lower_leg

    def test_hierarchy_ids_preserve_namespace(self, monkeypatch):
        hips, _upper_leg, _lower_leg = self.make_mixamo_hierarchy()
        adapter = self.make_adapter(monkeypatch, [hips])

        assert adapter.get_hierarchy_nodes("mixamorig:Hips") == [
            "mixamorig:Hips",
            "mixamorig:LeftUpLeg",
            "mixamorig:LeftLeg",
        ]

    def test_namespaced_hierarchy_samples_the_requested_child(self, monkeypatch):
        hips, _upper_leg, _lower_leg = self.make_mixamo_hierarchy()
        adapter = self.make_adapter(monkeypatch, [hips])

        child_name = adapter.get_hierarchy_nodes("mixamorig:Hips")[-1]
        trajectory = adapter.get_node_trajectory(child_name, 0, 0)

        assert trajectory[0, 3:6].tolist() == [20.0, 21.0, 22.0]

    def test_missing_child_never_writes_to_hips(self, monkeypatch):
        hips, _upper_leg, _lower_leg = self.make_mixamo_hierarchy()
        adapter = self.make_adapter(monkeypatch, [hips])
        trajectory = np.array([[0.0, 0.0, 0.0, 1.0, 2.0, 3.0]])

        with pytest.raises(ValueError, match="not found"):
            adapter.set_node_trajectory(
                "mixamorig:MissingBone",
                trajectory,
                include_translation=False,
            )

        assert all(
            not node.FCurve.key_add_calls
            for node in hips.Rotation.GetAnimationNode().Nodes
        )

    def test_unqualified_duplicate_name_is_rejected(self, monkeypatch):
        first = self.FakeModel("LeftLeg", "hero:LeftLeg")
        second = self.FakeModel("LeftLeg", "enemy:LeftLeg")
        adapter = self.make_adapter(monkeypatch, [first, second])

        with pytest.raises(ValueError, match="Ambiguous"):
            adapter.resolve_model("LeftLeg")

        assert adapter.resolve_model("hero:LeftLeg") is first
        assert adapter.resolve_model("enemy:LeftLeg") is second

    def test_validation_rejects_two_names_for_the_same_model(self, monkeypatch):
        leg = self.FakeModel("LeftLeg", "mixamorig:LeftLeg")
        adapter = self.make_adapter(monkeypatch, [leg])

        with pytest.raises(ValueError, match="same model"):
            adapter.validate_node_names(["mixamorig:LeftLeg", "LeftLeg"])

    def test_unqualified_root_is_rejected_when_two_characters_match(self, monkeypatch):
        first = self.FakeModel("Hips", "hero:Hips")
        second = self.FakeModel("Hips", "enemy:Hips")
        adapter = self.make_adapter(monkeypatch, [first, second])

        with pytest.raises(ValueError, match="Ambiguous"):
            adapter.find_root_bone("Hips")

    def test_same_long_name_from_two_python_wrappers_is_one_model(self, monkeypatch):
        scene_model = self.FakeModel("LeftLeg", "mixamorig:LeftLeg")
        lookup_wrapper = self.FakeModel("LeftLeg", "mixamorig:LeftLeg")
        adapter = self.make_adapter(monkeypatch, [scene_model])
        monkeypatch.setattr(
            adapter_module,
            "FBFindModelByLabelName",
            lambda _name: lookup_wrapper,
        )

        resolved = adapter.resolve_model("mixamorig:LeftLeg")

        assert resolved.LongName == "mixamorig:LeftLeg"


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
                self.Name = "Hips"
                self.LongName = "Hips"
                self.LabelName = "Hips"
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

    def test_set_node_trajectory_can_leave_child_translation_untouched(self, monkeypatch):
        class FakeTime:
            def __init__(self, _h=0, _m=0, _s=0, frame=0):
                self._frame = frame

            def GetFrame(self):
                return self._frame

        class FakeFCurve:
            def __init__(self):
                self.key_add_calls = []
                self.edit_clear_calls = 0

            def KeyAdd(self, time, value):
                self.key_add_calls.append((time.GetFrame(), value))
                return len(self.key_add_calls) - 1

            def EditClear(self):
                self.edit_clear_calls += 1

        class FakeNode:
            def __init__(self):
                self.FCurve = FakeFCurve()

        class FakeAnimNode:
            def __init__(self):
                self.Nodes = [FakeNode(), FakeNode(), FakeNode()]

        class FakeProperty:
            def __init__(self):
                self.node = FakeAnimNode()
                self.set_animated_calls = []

            def SetAnimated(self, value):
                self.set_animated_calls.append(value)

            def GetAnimationNode(self):
                return self.node

        class FakeModel:
            def __init__(self):
                self.Translation = FakeProperty()
                self.Rotation = FakeProperty()

        class FakeSystem:
            def __init__(self):
                self.CurrentTake = object()

        model = FakeModel()
        adapter = object.__new__(adapter_module.MoBuAdapter)
        adapter._system = FakeSystem()
        adapter.resolve_model = lambda _name: model

        monkeypatch.setattr(adapter_module, "FBTime", FakeTime)

        trajectory = np.array([
            [0.1, 0.2, 0.3, 10.0, 20.0, 30.0],
            [0.4, 0.5, 0.6, 40.0, 50.0, 60.0],
        ])

        adapter.set_node_trajectory(
            "LeftLeg",
            trajectory,
            include_translation=False,
        )

        assert model.Translation.set_animated_calls == []
        assert all(not node.FCurve.key_add_calls for node in model.Translation.node.Nodes)
        assert all(node.FCurve.edit_clear_calls == 0 for node in model.Translation.node.Nodes)
        assert model.Rotation.node.Nodes[0].FCurve.key_add_calls == [
            (0, 10.0),
            (1, 40.0),
        ]


class TestMoBuAdapterGetNodeTrajectory:
    def test_samples_fcurves_without_seeking_or_evaluating(self, monkeypatch):
        """Hierarchy processing should not evaluate the whole scene per frame."""

        class FakeTime:
            def __init__(self, _h=0, _m=0, _s=0, frame=0, *_args):
                self._frame = frame

            def GetFrame(self):
                return self._frame

        class FakeFCurve:
            def __init__(self, offset):
                self.offset = offset

            def Evaluate(self, time):
                return self.offset + time.GetFrame()

        class FakeNode:
            def __init__(self, offset):
                self.FCurve = FakeFCurve(offset)

        class FakeAnimationNode:
            def __init__(self, offsets):
                self.Nodes = [FakeNode(offset) for offset in offsets]

        class FakeProperty:
            def __init__(self, offsets):
                self.node = FakeAnimationNode(offsets)

            def GetAnimationNode(self):
                return self.node

        class FakeModel:
            Name = "LeftLeg"
            LongName = "mixamorig:LeftLeg"
            LabelName = "mixamorig:LeftLeg"
            Translation = FakeProperty((1.0, 10.0, 20.0))
            Rotation = FakeProperty((30.0, 40.0, 50.0))

            def GetVector(self, *_args):
                pytest.fail("hierarchy sampling must read FCurves")

        fake_model = FakeModel()
        monkeypatch.setattr(adapter_module, "FBTime", FakeTime)

        adapter = object.__new__(adapter_module.MoBuAdapter)
        adapter.resolve_model = lambda _name: fake_model
        adapter._set_local_time = lambda _time: pytest.fail(
            "hierarchy sampling must not seek the timeline"
        )
        adapter._evaluate_scene = lambda: pytest.fail(
            "hierarchy sampling must not evaluate the scene"
        )
        trajectory = adapter.get_node_trajectory("mixamorig:LeftLeg", 0, 2)

        np.testing.assert_allclose(
            trajectory,
            [
                [1.0, 10.0, 20.0, 30.0, 40.0, 50.0],
                [2.0, 11.0, 21.0, 31.0, 41.0, 51.0],
                [3.0, 12.0, 22.0, 32.0, 42.0, 52.0],
            ],
        )


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
                self.Name = "Hips"
                self.LongName = "Hips"
                self.LabelName = "Hips"
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
                self.Name = "Hips"
                self.LongName = "Hips"
                self.LabelName = "Hips"
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
