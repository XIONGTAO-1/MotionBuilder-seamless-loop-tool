"""
Tests for loop_analysis module.

Following TDD: Write tests FIRST, watch them FAIL, then implement.
"""

import pytest
import numpy as np

import sys
import os
# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from core.loop_analysis import FrameAnalyzer


class TestFrameAnalyzer:
    """Test suite for FrameAnalyzer class."""

    def test_similarity_score_identical_frames(self):
        """
        Two identical pose matrices should have a similarity score of 0.0.
        
        This is the base case: if the poses are the same, there's no difference.
        """
        analyzer = FrameAnalyzer()
        
        # Create a simple 4x4 identity matrix as a mock "pose"
        pose_a = np.eye(4)
        pose_b = np.eye(4)
        
        score = analyzer.calculate_similarity_score(pose_a, pose_b)
        
        assert score == pytest.approx(0.0), \
            "Identical poses should have 0.0 difference"

    def test_similarity_score_different_frames(self):
        """
        Two different pose matrices should have a positive similarity score.
        
        The score should be proportional to the difference.
        """
        analyzer = FrameAnalyzer()
        
        pose_a = np.eye(4)
        pose_b = np.eye(4)
        pose_b[0, 3] = 10.0  # Translate X by 10 units
        
        score = analyzer.calculate_similarity_score(pose_a, pose_b)
        
        assert score > 0.0, "Different poses should have positive difference"

    def test_find_best_loop_frame_returns_minimum(self):
        """
        find_best_loop_frame should return the frame with the lowest score.
        
        Given mock data where frame 5 is most similar to frame 0,
        the function should return 5.
        """
        analyzer = FrameAnalyzer()
        
        # Mock similarity scores: frame 5 has the minimum
        mock_scores = [10.0, 8.0, 6.0, 4.0, 2.0, 0.5, 3.0, 5.0, 7.0, 9.0]
        
        # Use dependency injection / mock to provide these scores
        # For now, we'll test the logic directly
        best_frame = analyzer.find_best_loop_frame(
            reference_frame=0,
            search_range=(1, 10),
            score_function=lambda f: mock_scores[f]
        )
        
        assert best_frame == 5, "Should return frame with minimum score"

    def test_find_best_loop_frame_respects_range(self):
        """
        find_best_loop_frame should only search within the given range.
        """
        analyzer = FrameAnalyzer()
        
        # Frame 2 has the global minimum, but it's outside the search range
        mock_scores = [10.0, 8.0, 0.1, 4.0, 2.0, 0.5, 3.0, 5.0, 7.0, 9.0]
        
        best_frame = analyzer.find_best_loop_frame(
            reference_frame=0,
            search_range=(4, 10),  # Excludes frame 2
            score_function=lambda f: mock_scores[f]
        )
        
        assert best_frame == 5, "Should find minimum within range, not global minimum"

    def test_pose_similarity_root_relative_ignores_root_translation(self):
        """Root-relative pose similarity should ignore root translation offsets."""
        analyzer = FrameAnalyzer()

        def make_translation_matrix(x: float, y: float, z: float) -> np.ndarray:
            mat = np.eye(4)
            mat[0, 3] = x
            mat[1, 3] = y
            mat[2, 3] = z
            return mat

        reference_poses = {
            "Hips": make_translation_matrix(0.0, 0.0, 0.0),
            "Foot": make_translation_matrix(1.0, 0.0, 0.0),
        }
        candidate_poses = {
            "Hips": make_translation_matrix(5.0, 0.0, 0.0),
            "Foot": make_translation_matrix(6.0, 0.0, 0.0),
        }

        score = analyzer.calculate_pose_similarity(
            reference_poses,
            candidate_poses,
            root_name="Hips"
        )

        assert score == pytest.approx(0.0), "Root-relative poses should match"

    def test_evaluate_loop_pair_disqualifies_static_segment(self):
        """
        evaluate_loop_pair should return infinity if the segment's velocity is below threshold.
        """
        analyzer = FrameAnalyzer()
        threshold = 5.0
        
        # 1. Create a STATIC trajectory (T-Pose)
        # 10 frames, no movement (X,Y,Z = 0)
        static_traj = np.zeros((10, 6)) 
        
        score_static = analyzer.evaluate_loop_pair(
            static_traj,
            start_idx=0,
            end_idx=9,
            min_average_velocity=threshold
        )
        assert score_static == float('inf'), "Static segment should be disqualified (score=inf)"
        
        # 2. Create a MOVING trajectory (Walking)
        # 11 frames to safely evaluate up to index 9 (needs index 10 for velocity)
        moving_traj = np.zeros((11, 6))
        moving_traj[:, 0] = np.arange(11) * 1.0 # X axis movement
        
        score_moving = analyzer.evaluate_loop_pair(
            moving_traj,
            start_idx=0,
            end_idx=9,
            min_average_velocity=threshold
        )
        assert score_moving < float('inf'), "Moving segment should be valid (score < inf)"

    def test_evaluate_loop_pair_disqualifies_low_vertical_bounce(self):
        """
        evaluate_loop_pair should return infinity if vertical bounce is below threshold.
        """
        analyzer = FrameAnalyzer()

        low_bounce = np.zeros((6, 6))
        low_bounce[:, 0] = np.arange(6) * 1.0
        low_bounce[:, 1] = 0.05

        score_low = analyzer.evaluate_loop_pair(
            low_bounce,
            start_idx=0,
            end_idx=4,
            min_average_velocity=0.0,
            min_vertical_bounce=0.1
        )
        assert score_low == float('inf')

        ok_bounce = low_bounce.copy()
        ok_bounce[:, 1] = np.array([0.0, 0.2, 0.0, 0.2, 0.0, 0.2])

        score_ok = analyzer.evaluate_loop_pair(
            ok_bounce,
            start_idx=0,
            end_idx=4,
            min_average_velocity=0.0,
            min_vertical_bounce=0.1
        )
        assert score_ok < float('inf')
