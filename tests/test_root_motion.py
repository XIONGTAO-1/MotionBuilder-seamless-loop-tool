"""
Tests for root_motion module.

Following TDD: Write tests FIRST, watch them FAIL, then implement.
"""

import pytest
import numpy as np
import sys
import os

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.root_motion import RootProcessor


class TestRootProcessor:
    """Test suite for RootProcessor class."""

    def test_process_in_place_locks_xz_at_frame_0(self):
        """
        process_in_place should set X and Z to exactly 0.0 (world origin).
        This allows game engines to control character movement.
        
        Y translation (height) should be preserved.
        """
        processor = RootProcessor()
        
        # Create a trajectory starting at a non-zero position
        # Shape: (num_frames, 3) for X, Y, Z positions
        trajectory = np.array([
            [5.0, 0.0, 10.0],   # Frame 0: starts at (5, 0, 10)
            [6.0, 0.1, 10.5],   # Frame 1: moved forward
            [7.0, 0.15, 11.0],  # Frame 2: moved more
            [8.0, 0.2, 11.5],   # Frame 3: etc.
        ])
        
        result = processor.process_in_place(trajectory)
        
        # X and Z should be forced to 0.0 (world origin for game engine control)
        assert np.allclose(result[:, 0], 0.0), "X translation should be 0.0"
        assert np.allclose(result[:, 2], 0.0), "Z translation should be 0.0"
        
        # Y (height) should be preserved (changing values)
        np.testing.assert_array_almost_equal(
            result[:, 1], 
            trajectory[:, 1],
            err_msg="Y translation should be preserved"
        )

    def test_process_in_place_preserves_rotation_data(self):
        """
        process_in_place should not modify rotation data if present.
        
        If the input has 6 columns (XYZ translation + XYZ rotation),
        rotations should be unchanged.
        """
        processor = RootProcessor()
        
        # 6 columns: X, Y, Z, RotX, RotY, RotZ
        trajectory = np.array([
            [0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [1.0, 0.1, 0.5, 10.0, 20.0, 30.0],
            [2.0, 0.2, 1.0, 15.0, 25.0, 35.0],
        ])
        
        result = processor.process_in_place(trajectory)
        
        # Rotations (columns 3-5) should be unchanged
        np.testing.assert_array_almost_equal(
            result[:, 3:6],
            trajectory[:, 3:6],
            err_msg="Rotations should not be modified"
        )

    def test_process_in_place_preserves_rotation_values(self):
        """
        process_in_place should preserve Y translation and rotations.
        """
        processor = RootProcessor()

        trajectory = np.array([
            [1.0, 2.0, 3.0, 10.0, 20.0, 30.0],
            [4.0, 5.0, 6.0, 40.0, 50.0, 60.0],
        ])

        result = processor.process_in_place(trajectory)

        np.testing.assert_array_equal(result[:, 1], trajectory[:, 1])
        np.testing.assert_array_equal(result[:, 3:6], trajectory[:, 3:6])

    def test_reset_origin_moves_to_zero(self):
        """
        reset_origin should translate so frame 0 is at origin.
        """
        processor = RootProcessor()
        
        # Character starts at (5, 1, 3), not at origin
        trajectory = np.array([
            [5.0, 1.0, 3.0],
            [6.0, 1.1, 4.0],
            [7.0, 1.2, 5.0],
        ])
        
        result = processor.reset_origin(trajectory)
        
        # First frame should now be at origin
        np.testing.assert_array_almost_equal(
            result[0],
            [0.0, 0.0, 0.0],
            err_msg="First frame should be at origin after reset"
        )
        
        # Relative movement should be preserved
        expected_delta = trajectory[1] - trajectory[0]
        actual_delta = result[1] - result[0]
        np.testing.assert_array_almost_equal(
            actual_delta,
            expected_delta,
            err_msg="Relative movement should be preserved"
        )

    def test_trim_to_loop_cuts_at_end_frame(self):
        """
        trim_to_loop should truncate trajectory at the loop frame.
        """
        processor = RootProcessor()
        
        # 10 frames, want to trim to frame 5
        trajectory = np.arange(30).reshape(10, 3).astype(float)
        
        result = processor.trim_to_loop(trajectory, loop_frame=5)
        
        # Should have 6 frames (0, 1, 2, 3, 4, 5)
        assert result.shape[0] == 6, f"Expected 6 frames, got {result.shape[0]}"
        np.testing.assert_array_equal(result, trajectory[:6])

    def test_extract_segment_returns_inclusive_slice(self):
        """
        extract_segment should return a copy of the inclusive segment.
        """
        processor = RootProcessor()

        trajectory = np.arange(30).reshape(10, 3).astype(float)

        result = processor.extract_segment(trajectory, start=2, end=5)

        assert result.shape[0] == 4, f"Expected 4 frames, got {result.shape[0]}"
        np.testing.assert_array_equal(result, trajectory[2:6])

        result[0, 0] = -1.0
        assert trajectory[2, 0] != -1.0

    def test_crop_sequence_returns_inclusive_segment(self):
        """
        crop_sequence should return a copied inclusive segment.
        """
        processor = RootProcessor()

        trajectory = np.arange(30).reshape(10, 3).astype(float)

        result = processor.crop_sequence(trajectory, start_idx=3, end_idx=6)

        np.testing.assert_array_equal(result, trajectory[3:7])
        result[0, 0] = -2.0
        assert trajectory[3, 0] != -2.0

    def test_blend_loop_ends_makes_first_last_similar(self):
        """
        blend_loop_ends should blend the last N frames to match the start.
        
        After blending, the last frame should be very similar to the first.
        """
        processor = RootProcessor()
        
        # Create a trajectory where first and last frames are different
        trajectory = np.array([
            [0.0, 0.0, 0.0],   # Start pose
            [1.0, 0.1, 0.5],
            [2.0, 0.2, 1.0],
            [3.0, 0.3, 1.5],
            [4.0, 0.4, 2.0],
            [5.0, 0.5, 2.5],   # End pose (different from start)
        ])
        
        result = processor.blend_loop_ends(trajectory, blend_frames=3)
        
        # The last frame should now be much closer to the first
        first_frame = result[0]
        last_frame = result[-1]
        
        # They should be approximately equal (within tolerance)
        np.testing.assert_array_almost_equal(
            last_frame, first_frame, decimal=1,
            err_msg="Last frame should blend towards first frame"
        )

    def test_blend_loop_ends_linear_offset_preserves_first_frame(self):
        """
        Linear Offset Compensation should preserve the first frame unchanged.
        The last frame should exactly equal the first frame.
        """
        processor = RootProcessor()
        
        trajectory = np.arange(30).reshape(10, 3).astype(float)
        
        result = processor.blend_loop_ends(trajectory, blend_frames=3)
        
        # First frame should be unchanged
        np.testing.assert_array_equal(
            result[0], trajectory[0],
            err_msg="First frame should be unchanged"
        )
        
        # Last frame should now equal first frame
        np.testing.assert_array_almost_equal(
            result[-1], result[0],
            err_msg="Last frame should equal first frame after linear offset"
        )

    def test_align_orientation_offsets_rotation_y(self):
        """
        align_orientation should offset rotation Y so frame 0 equals target.
        """
        processor = RootProcessor()

        trajectory = np.array([
            [0.0, 0.0, 0.0, 0.0, 170.0, 0.0],
            [1.0, 0.0, 0.0, 0.0, 175.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 190.0, 0.0],
        ])

        result = processor.align_orientation(trajectory, target_rot_y=180.0)

        assert result[0, 4] == 180.0
        np.testing.assert_array_equal(result[:, 4], trajectory[:, 4] + 10.0)
        np.testing.assert_array_equal(
            result[:, [0, 1, 2, 3, 5]],
            trajectory[:, [0, 1, 2, 3, 5]],
        )
