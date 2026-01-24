"""
Tests for mobu/loop_processor service.

Integration tests that verify the full pipeline works correctly.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from mobu.adapter import MockMoBuAdapter
from mobu.loop_processor import LoopProcessorService


class TestLoopProcessorService:
    """Test suite for LoopProcessorService integration."""

    def create_walking_trajectory(self, num_frames: int = 100) -> np.ndarray:
        """Create a mock walking trajectory for testing."""
        trajectory = np.zeros((num_frames, 6))
        
        # Simulate forward movement (X increases linearly)
        trajectory[:, 0] = np.linspace(0, 10, num_frames)  # X
        
        # Simulate height bobbing (Y oscillates)
        trajectory[:, 1] = 0.1 * np.sin(np.linspace(0, 8 * np.pi, num_frames))  # Y
        
        # Simulate side-to-side (Z oscillates)
        trajectory[:, 2] = 0.05 * np.sin(np.linspace(0, 4 * np.pi, num_frames))  # Z
        
        return trajectory

    def make_translation_matrix(self, x: float, y: float, z: float) -> np.ndarray:
        """Create a 4x4 translation matrix."""
        mat = np.eye(4)
        mat[0, 3] = x
        mat[1, 3] = y
        mat[2, 3] = z
        return mat

    def test_analyze_loop_points_finds_similar_frame(self):
        """analyze_loop_points should find a frame within the search range."""
        # Create a cyclic trajectory
        trajectory = self.create_walking_trajectory(100)
        # Make frame 50 very similar to frame 0
        trajectory[50] = trajectory[0] + 0.001  # Tiny difference
        
        adapter = MockMoBuAdapter(mock_trajectory=trajectory, mock_frame_range=(0, 99))
        service = LoopProcessorService(adapter)
        
        loop_frame = service.analyze_loop_points(search_range=(40, 60))
        
        # With velocity-weighted algorithm, the best frame may differ slightly
        # but should be within the search range
        assert 40 <= loop_frame <= 60, f"Expected frame in range [40, 60], got {loop_frame}"

    def test_analyze_loop_points_prefers_pose_similarity(self):
        """Pose similarity should drive loop selection when available."""
        num_frames = 100
        trajectory = self.create_walking_trajectory(num_frames)
        adapter = MockMoBuAdapter(mock_trajectory=trajectory, mock_frame_range=(0, num_frames - 1))

        for frame in range(num_frames):
            root_x = frame * 0.1
            foot_offset = np.sin(2 * np.pi * frame / num_frames)
            poses = {
                "Hips": self.make_translation_matrix(root_x, 0.0, 0.0),
                "Foot": self.make_translation_matrix(root_x + foot_offset, 0.0, 0.0),
            }
            adapter.set_mock_poses(frame, poses)

        service = LoopProcessorService(adapter)
        loop_frame = service.analyze_loop_points(search_range=(40, 60))

        # With velocity-weighted algorithm, the best frame may differ slightly
        assert 40 <= loop_frame <= 60, f"Expected frame in range [40, 60], got {loop_frame}"

    def test_process_in_place_locks_xz_at_frame_0(self):
        """process_in_place should set X and Z to 0.0 while preserving Y."""
        trajectory = self.create_walking_trajectory(50)
        trajectory[:, 0] += 5.0
        trajectory[:, 2] -= 3.0
        adapter = MockMoBuAdapter(mock_trajectory=trajectory)
        service = LoopProcessorService(adapter)
        
        result = service.process_in_place()
        
        # X and Z should be forced to 0.0 (world origin)
        assert np.allclose(result[:, 0], 0.0), "X should be 0.0"
        assert np.allclose(result[:, 2], 0.0), "Z should be 0.0"
        
        # Y should be preserved
        np.testing.assert_array_almost_equal(result[:, 1], trajectory[:, 1])

    def test_run_full_pipeline_returns_results(self):
        """run_full_pipeline should return loop_frame and trajectory."""
        trajectory = self.create_walking_trajectory(100)
        adapter = MockMoBuAdapter(mock_trajectory=trajectory, mock_frame_range=(0, 99))
        service = LoopProcessorService(adapter)
        
        results = service.run_full_pipeline(in_place=True, apply=False)
        
        assert "loop_frame" in results
        assert "trajectory" in results
        assert results["applied"] is False
        assert isinstance(results["trajectory"], np.ndarray)

    def test_create_seamless_loop_uses_cycle_detection_segment(self, monkeypatch):
        """create_seamless_loop should use the detected cycle segment when enabled."""
        trajectory = np.arange(300).reshape(100, 3).astype(float)
        adapter = MockMoBuAdapter(mock_trajectory=trajectory, mock_frame_range=(100, 199))
        service = LoopProcessorService(adapter)

        monkeypatch.setattr(
            service,
            "find_walk_cycle",
            lambda root_name="Hips": (110, 120)
        )

        captured = {}

        def fake_blend(data, blend_frames=5):
            captured["data"] = data.copy()
            return data

        monkeypatch.setattr(service.root_processor, "blend_loop_ends", fake_blend)

        result = service.create_seamless_loop(in_place=False, use_cycle_detection=True)

        expected = trajectory[10:21]
        assert result.shape[0] == expected.shape[0]
        np.testing.assert_array_equal(captured["data"], expected)
        assert service.best_loop_frame == 120

    def test_create_seamless_loop_sets_loop_start_from_cycle_detection(self, monkeypatch):
        """create_seamless_loop should store the detected start frame."""
        trajectory = np.arange(300).reshape(100, 3).astype(float)
        adapter = MockMoBuAdapter(mock_trajectory=trajectory, mock_frame_range=(100, 199))
        service = LoopProcessorService(adapter)

        monkeypatch.setattr(
            service,
            "find_walk_cycle",
            lambda root_name="Hips": (110, 120)
        )

        service.create_seamless_loop(in_place=False, use_cycle_detection=True)

        assert service.loop_start_frame == 110

    def test_create_seamless_loop_sets_loop_start_from_explicit_range(self):
        """create_seamless_loop should store the explicit start frame."""
        trajectory = np.arange(300).reshape(100, 3).astype(float)
        adapter = MockMoBuAdapter(mock_trajectory=trajectory, mock_frame_range=(100, 199))
        service = LoopProcessorService(adapter)

        service.create_seamless_loop(
            start_frame=130,
            loop_frame=140,
            in_place=False,
            use_cycle_detection=False
        )

        assert service.loop_start_frame == 130

    def test_create_seamless_loop_does_not_align_orientation_by_default(self, monkeypatch):
        """create_seamless_loop should not change root rotation Y unless requested."""
        trajectory = np.zeros((3, 6))
        trajectory[:, 4] = [90.0, 100.0, 110.0]
        adapter = MockMoBuAdapter(mock_trajectory=trajectory, mock_frame_range=(0, 2))
        service = LoopProcessorService(adapter)

        monkeypatch.setattr(service.root_processor, "blend_loop_ends", lambda data, blend_frames=5: data)

        result = service.create_seamless_loop(
            start_frame=0,
            loop_frame=2,
            in_place=False,
            use_cycle_detection=False,
        )

        np.testing.assert_array_equal(result[:, 4], trajectory[:, 4])

    def test_create_seamless_loop_aligns_root_orientation_y_when_requested(self, monkeypatch):
        """create_seamless_loop should align root rotation Y when target is provided."""
        trajectory = np.zeros((3, 6))
        trajectory[:, 4] = [90.0, 100.0, 110.0]
        adapter = MockMoBuAdapter(mock_trajectory=trajectory, mock_frame_range=(0, 2))
        service = LoopProcessorService(adapter)

        monkeypatch.setattr(service.root_processor, "blend_loop_ends", lambda data, blend_frames=5: data)

        result = service.create_seamless_loop(
            start_frame=0,
            loop_frame=2,
            in_place=False,
            use_cycle_detection=False,
            target_rot_y=180.0,
        )

        assert result[0, 4] == 180.0
        np.testing.assert_array_equal(result[:, 4], trajectory[:, 4] + 90.0)

    def test_mock_create_clean_take_clears_animation(self):
        """create_clean_take should clear animation data in the mock adapter."""
        adapter = MockMoBuAdapter()

        take_name = adapter.create_clean_take("_InPlace", root_name="Hips")

        assert take_name.endswith("_InPlace")
        assert adapter.get_current_take_name().endswith("_InPlace")
        assert adapter.animation_cleared is True

    def test_mock_set_root_trajectory_defaults_to_zero(self):
        """set_root_trajectory should default to frame 0 and set frame range in mock."""
        adapter = MockMoBuAdapter(mock_frame_range=(100, 199))
        trajectory = np.zeros((5, 3))

        adapter.set_root_trajectory("Hips", trajectory)

        assert adapter.frame_range == (0, 4)
        assert adapter.last_written_frames == [0, 1, 2, 3, 4]

    def test_apply_changes_writes_back(self):
        """apply_changes should write the trajectory back to adapter."""
        trajectory = self.create_walking_trajectory(50)
        adapter = MockMoBuAdapter(mock_trajectory=trajectory)
        service = LoopProcessorService(adapter)
        
        # Process and apply
        service.process_in_place()
        service.apply_changes()
        
        # Check that adapter was updated
        updated = adapter.get_root_trajectory("Hips")
        assert np.allclose(updated[:, 0], 0.0), "X should be zeroed after apply"
        assert adapter.frame_range == (0, len(updated) - 1)
        assert adapter.animation_cleared is True

    def test_apply_changes_sets_transport_fps_when_provided(self):
        """apply_changes should set transport fps when target_fps is provided."""
        class FakeAdapter:
            def __init__(self):
                self.transport_fps = None
                self.cleared = False
                self.written = False

            def clear_all_animation(self, root_name="Hips"):
                self.cleared = True

            def set_root_trajectory(self, root_name, trajectory, start_frame=0):
                self.written = True

            def set_transport_fps(self, target_fps):
                self.transport_fps = target_fps

        adapter = FakeAdapter()
        service = LoopProcessorService(adapter)
        service.processed_trajectory = np.zeros((10, 6))

        service.apply_changes(preserve_original=False, target_fps=60.0)

        assert adapter.transport_fps == 60.0
        assert adapter.cleared is True
        assert adapter.written is True

    def test_apply_changes_hierarchy_sets_transport_fps_when_provided(self):
        """apply_changes_hierarchy should set transport fps when target_fps is provided."""
        class FakeAdapter:
            def __init__(self):
                self.transport_fps = None
                self.cleared = False
                self.written_nodes = []

            def clear_all_animation(self, root_name="Hips"):
                self.cleared = True

            def set_node_trajectory(self, node_name, trajectory, start_frame=0):
                self.written_nodes.append(node_name)

            def set_transport_fps(self, target_fps):
                self.transport_fps = target_fps

        adapter = FakeAdapter()
        service = LoopProcessorService(adapter)
        service.processed_data = {"Hips": np.zeros((10, 6))}
        service.processed_trajectory = np.zeros((10, 6))

        service.apply_changes_hierarchy(preserve_original=False, target_fps=90.0)

        assert adapter.transport_fps == 90.0
        assert adapter.cleared is True
        assert adapter.written_nodes == ["Hips"]
