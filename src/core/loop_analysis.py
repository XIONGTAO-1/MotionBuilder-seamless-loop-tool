"""
Loop Analysis Module.

Provides algorithms for detecting optimal loop points in motion capture data.
"""

import numpy as np
from typing import Tuple, Callable, Optional, Dict


class FrameAnalyzer:
    """
    Analyzes animation frames to find the best loop points.
    
    Uses pose similarity scoring to identify frames where the animation
    can seamlessly loop back to the start.
    """

    def calculate_similarity_score(
        self, 
        pose_a: np.ndarray, 
        pose_b: np.ndarray
    ) -> float:
        """
        Calculate the similarity score between two poses.
        
        Uses the Frobenius norm (sum of squared differences) to measure
        how different two pose matrices are.
        
        Args:
            pose_a: First pose matrix (4x4 or Nx4x4 for multiple joints)
            pose_b: Second pose matrix
            
        Returns:
            A non-negative float where 0.0 means identical poses.
            Higher values indicate more difference.
        """
        # Compute the difference matrix
        diff = pose_a - pose_b

        # Frobenius norm over any shape: sqrt(sum of squared elements)
        score = float(np.sqrt(np.sum(diff * diff)))

        return score

    def calculate_pose_similarity(
        self,
        reference_poses: Dict[str, np.ndarray],
        candidate_poses: Dict[str, np.ndarray],
        root_name: Optional[str] = None
    ) -> float:
        """
        Calculate similarity between two pose dictionaries.

        Uses root-relative transforms when a root bone is available in both.

        Args:
            reference_poses: Dict of bone name to 4x4 transform at reference frame
            candidate_poses: Dict of bone name to 4x4 transform at candidate frame
            root_name: Optional root bone name for root-relative comparison

        Returns:
            A non-negative float where 0.0 means identical poses.
        """
        if not reference_poses or not candidate_poses:
            raise ValueError("Pose dictionaries must not be empty")

        common_bones = sorted(set(reference_poses) & set(candidate_poses))
        if not common_bones:
            raise ValueError("No common bones to compare")

        ref_root = reference_poses.get(root_name) if root_name else None
        cand_root = candidate_poses.get(root_name) if root_name else None

        ref_root_inv = np.linalg.inv(ref_root) if ref_root is not None else None
        cand_root_inv = np.linalg.inv(cand_root) if cand_root is not None else None

        total = 0.0
        for bone in common_bones:
            ref_mat = reference_poses[bone]
            cand_mat = candidate_poses[bone]
            if ref_root_inv is not None and cand_root_inv is not None:
                ref_mat = ref_root_inv @ ref_mat
                cand_mat = cand_root_inv @ cand_mat
            diff = ref_mat - cand_mat
            total += float(np.sum(diff * diff))

        return float(np.sqrt(total))

    def find_best_loop_frame(
        self,
        reference_frame: int,
        search_range: Tuple[int, int],
        score_function: Optional[Callable[[int], float]] = None
    ) -> int:
        """
        Find the frame with the best loop potential within a search range.
        
        Iterates through the search range and finds the frame with the
        lowest similarity score (most similar to reference).
        
        Args:
            reference_frame: The frame to compare against (usually frame 0)
            search_range: Tuple of (start_frame, end_frame) to search
            score_function: Function that returns similarity score for a frame
            
        Returns:
            The frame index with the lowest similarity score (best match)
            
        Raises:
            ValueError: If search_range is invalid or score_function is None
        """
        if score_function is None:
            raise ValueError("score_function is required")
        
        start, end = search_range
        
        if start >= end:
            raise ValueError(f"Invalid search range: {search_range}")
        
        best_frame = start
        best_score = float('inf')
        
        for frame in range(start, end):
            score = score_function(frame)
            if score < best_score:
                best_score = score
                best_frame = frame
        
        return best_frame

    def calculate_loop_cost(
        self,
        trajectory: np.ndarray,
        candidate_index: int,
        reference_index: int = 0,
        position_weight: float = 1.0,
        velocity_weight: float = 5.0
    ) -> float:
        """
        Calculate weighted cost for loop candidacy using position AND velocity.
        
        Based on technical report recommendations:
        - Position similarity alone is not enough
        - Velocity consistency is critical (weight 5x higher)
        - Uses Hip Y-translation as primary feature
        
        Formula: Cost = w_p * ||Pose_diff|| + w_v * ||Velocity_diff||
        
        Args:
            trajectory: Array of shape (num_frames, 6) [X,Y,Z,RotX,RotY,RotZ]
            candidate_index: Frame index to evaluate as loop point
            reference_index: Reference frame (usually 0)
            position_weight: Weight for position difference (default 1.0)
            velocity_weight: Weight for velocity difference (default 5.0)
            
        Returns:
            Weighted cost score (lower is better)
        """
        num_frames = len(trajectory)
        
        if candidate_index < 1 or candidate_index >= num_frames - 1:
            return float('inf')
        if reference_index < 0 or reference_index >= num_frames - 1:
            return float('inf')
        
        # Position difference (focus on Y - the hip height, index 1)
        pos_ref = trajectory[reference_index]
        pos_cand = trajectory[candidate_index]
        
        # Position cost: primarily Y (hip height), secondarily rotations
        pos_diff = pos_ref - pos_cand
        # Weight Y more heavily (the characteristic bobbing motion)
        pos_cost = abs(pos_diff[1]) * 2.0 + np.sqrt(np.sum(pos_diff[3:6] ** 2))
        
        # Velocity difference (direction of motion must match)
        vel_ref = trajectory[reference_index + 1] - trajectory[reference_index]
        vel_cand = trajectory[candidate_index + 1] - trajectory[candidate_index]
        vel_diff = vel_ref - vel_cand
        vel_cost = np.sqrt(np.sum(vel_diff ** 2))
        
        # Weighted total cost
        total_cost = position_weight * pos_cost + velocity_weight * vel_cost
        
        return float(total_cost)

    def find_loop_frame_with_velocity(
        self,
        trajectory: np.ndarray,
        search_range: Tuple[int, int],
        reference_index: int = 0
    ) -> int:
        """
        Find best loop frame using the weighted cost function with velocity.
        
        This is the recommended algorithm from the technical report.
        
        Args:
            trajectory: Root trajectory array (num_frames, 6)
            search_range: (start_index, end_index) relative indices
            reference_index: Reference frame index (usually 0)
            
        Returns:
            Best loop frame index (relative to trajectory array)
        """
        start, end = search_range
        
        if start >= end:
            raise ValueError(f"Invalid search range: {search_range}")
        
        best_frame = start
        best_cost = float('inf')
        
        for idx in range(start, min(end, len(trajectory) - 1)):
            cost = self.calculate_loop_cost(
                trajectory, 
                candidate_index=idx,
                reference_index=reference_index
            )
            if cost < best_cost:
                best_cost = cost
                best_frame = idx
        
        return best_frame

    # =========================================================================
    # Walk Cycle Detection Algorithm (Peak/Valley-based)
    # =========================================================================

    def find_valleys(self, signal: np.ndarray, min_distance: int = 5) -> np.ndarray:
        """
        Find local minima (valleys) in a 1D signal.
        
        Valleys in Hip Y correspond to foot landing moments (double support phase).
        
        Args:
            signal: 1D numpy array of values
            min_distance: Minimum frames between valleys
            
        Returns:
            Array of indices where valleys occur
        """
        valleys = []
        n = len(signal)
        
        for i in range(1, n - 1):
            # Local minimum: lower than neighbors
            if signal[i - 1] > signal[i] and signal[i] < signal[i + 1]:
                # Check minimum distance from last valley
                if not valleys or (i - valleys[-1]) >= min_distance:
                    valleys.append(i)
        
        return np.array(valleys)

    def find_peaks(self, signal: np.ndarray, min_distance: int = 5) -> np.ndarray:
        """
        Find local maxima (peaks) in a 1D signal.
        
        Args:
            signal: 1D numpy array of values
            min_distance: Minimum frames between peaks
            
        Returns:
            Array of indices where peaks occur
        """
        peaks = []
        n = len(signal)
        
        for i in range(1, n - 1):
            # Local maximum: higher than neighbors
            if signal[i - 1] < signal[i] and signal[i] > signal[i + 1]:
                if not peaks or (i - peaks[-1]) >= min_distance:
                    peaks.append(i)
        
        return np.array(peaks)

    def estimate_cycle_period(self, signal: np.ndarray, min_period: int = 15, max_period: int = 60) -> int:
        """
        Estimate the gait cycle period using autocorrelation.
        
        Args:
            signal: Hip Y-axis displacement curve
            min_period: Minimum expected cycle length (frames)
            max_period: Maximum expected cycle length (frames)
            
        Returns:
            Estimated period in frames
        """
        n = len(signal)
        if n < max_period:
            return min_period
        
        # Normalize signal
        normalized = signal - np.mean(signal)
        
        # Compute autocorrelation for lags in range
        best_lag = min_period
        best_corr = -float('inf')
        
        for lag in range(min_period, min(max_period, n // 2)):
            corr = np.sum(normalized[:n - lag] * normalized[lag:])
            if corr > best_corr:
                best_corr = corr
                best_lag = lag
        
        return best_lag

    def evaluate_loop_pair(
        self,
        trajectory: np.ndarray,
        start_idx: int,
        end_idx: int,
        position_weight: float = 1.0,
        velocity_weight: float = 5.0,
        min_average_velocity: float = 5.0,
        min_vertical_bounce: float = 0.0
    ) -> float:
        """
        Evaluate the quality of a loop between two frames.
        
        Based on technical report: Velocity consistency is 5x more important than position.
        
        Args:
            trajectory: Root trajectory (num_frames, 6)
            start_idx: Start frame index
            end_idx: End frame index
            position_weight: Weight for position difference
            velocity_weight: Weight for velocity difference (default 5x)
            min_average_velocity: Minimum avg velocity required to consider this loop (prevents T-pose/Idle selection)
            min_vertical_bounce: Minimum required Y-axis range (max - min) for the segment
            
        Returns:
            Cost score (lower = better loop quality). Returns inf if velocity is too low.
        """
        num_frames = len(trajectory)
        
        if start_idx < 0 or end_idx >= num_frames - 1:
            return float('inf')
        if start_idx >= end_idx:
            return float('inf')

        segment = trajectory[start_idx:end_idx+1]
        vertical_range = float(np.max(segment[:, 1]) - np.min(segment[:, 1]))
        if vertical_range < min_vertical_bounce:
            return float('inf')

        # Check average velocity to filter out static poses (T-Pose, Idle)
        # We look at horizontal velocity (X, Z) magnitude
        if len(segment) > 1:
            # Calculate per-frame displacements
            displacements = segment[1:, [0, 2]] - segment[:-1, [0, 2]] # Only X, Z
            dists = np.sqrt(np.sum(displacements**2, axis=1))
            avg_vel = np.mean(dists) * 30.0 # Approximate cm/s assuming 30fps, or just units/sec
            # Note: Since we don't know FPS, we treat this as units-per-frame * 30. 
            # Or simpler: just units-per-frame. 
            # Let's say the threshold is passed in units-per-frame.
            # If user passes 5.0 (units/sec) and we are 30fps, threshold per frame is 5/30 = 0.16
            # Let's assume the user passes a raw value suitable for the data scale.
            # If Mobu units are cm, 5cm/s is slow. 
            # Let's calculate total distance / duration.
            total_dist = np.sum(dists)
            duration = end_idx - start_idx
            if duration > 0:
                 avg_speed_per_frame = total_dist / duration
                 if avg_speed_per_frame * 30.0 < min_average_velocity:
                     return float('inf')
        
        # Position difference (focus on Y - hip height)
        pos_start = trajectory[start_idx]
        pos_end = trajectory[end_idx]
        pos_diff = pos_start - pos_end
        
        # Y (height) is most important, rotations secondary
        pos_cost = abs(pos_diff[1]) * 2.0 + np.sqrt(np.sum(pos_diff[3:6] ** 2))
        
        # Velocity difference (CRITICAL: direction must match)
        vel_start = trajectory[start_idx + 1] - trajectory[start_idx]
        vel_end = trajectory[end_idx + 1] - trajectory[end_idx]
        vel_diff = vel_start - vel_end
        vel_cost = np.sqrt(np.sum(vel_diff ** 2))
        
        return float(position_weight * pos_cost + velocity_weight * vel_cost)

    def find_best_walk_cycle(
        self,
        trajectory: np.ndarray,
        min_cycle_frames: int = 20,
        max_cycle_frames: int = 60,
        up_axis: int = 1,
        use_peaks: bool = True,
        min_average_velocity: float = 5.0,
        min_vertical_bounce: float = 0.0
    ) -> Tuple[int, int]:
        """
        Find the best walk cycle using Hip Y-axis peak/valley detection.
        
        Updated algorithm per notebook recommendations:
        1. Use peaks (local maxima) as candidates - corresponds to single-leg support
        2. Prefer double-step cycles (2x period) for left-right symmetry
        3. Weight velocity match heavily (5x)
        4. Enforce minimum velocity to ignore T-poses
        
        Args:
            trajectory: Root trajectory array (num_frames, 6) [X,Y,Z,RotX,RotY,RotZ]
            min_cycle_frames: Minimum expected cycle length
            max_cycle_frames: Maximum expected cycle length
            up_axis: Index of the vertical (up) axis (default 1 = Y)
            use_peaks: If True, use peaks (recommended). If False, use valleys.
            min_average_velocity: Minimum speed (units/sec) to consider as valid motion.
            min_vertical_bounce: Minimum required Y-axis range (max - min) for the segment.
            
        Returns:
            Tuple of (start_frame_idx, end_frame_idx) for best loop
        """
        num_frames = len(trajectory)
        
        if num_frames < min_cycle_frames:
            raise ValueError(f"Trajectory too short: {num_frames} < {min_cycle_frames}")
        
        # 1. Extract Hip Y (vertical) signal
        hip_y = trajectory[:, up_axis]
        
        # 2. Find candidates: peaks (single-leg support) or valleys (double support)
        if use_peaks:
            candidates = self.find_peaks(hip_y, min_distance=min_cycle_frames // 2)
            print(f"[SeamlessLoopTool] Using PEAKS for candidates (recommended)")
        else:
            candidates = self.find_valleys(hip_y, min_distance=min_cycle_frames // 2)
            print(f"[SeamlessLoopTool] Using valleys for candidates")
        
        if len(candidates) < 2:
            # Fallback: use simple frame-based search
            print("[SeamlessLoopTool] No candidates found, using fallback search")
            mid = num_frames // 2
            return (0, mid)
        
        # 3. Estimate cycle period (half-step)
        estimated_half_period = self.estimate_cycle_period(
            hip_y, 
            min_period=min_cycle_frames // 2, 
            max_period=max_cycle_frames
        )
        # Double-step period (full left-right cycle)
        estimated_full_period = estimated_half_period * 2
        
        print(f"[SeamlessLoopTool] Estimated half-period: {estimated_half_period}, full-period: {estimated_full_period}")
        print(f"[SeamlessLoopTool] Found {len(candidates)} candidates at: {candidates[:10].tolist()}...")
        
        # 4. Evaluate candidate pairs - PREFER double-step cycles for symmetry
        best_score = float('inf')
        best_pair = (candidates[0], candidates[-1])
        
        for i in range(len(candidates)):
            for j in range(i + 1, len(candidates)):
                start_idx = candidates[i]
                end_idx = candidates[j]
                duration = end_idx - start_idx
                
                # Filter: must be at least min_cycle and at most 2.5x max
                if duration < min_cycle_frames or duration > max_cycle_frames * 2.5:
                    continue
                
                # Base score from loop quality
                score = self.evaluate_loop_pair(
                    trajectory, 
                    start_idx, 
                    end_idx, 
                    min_average_velocity=min_average_velocity,
                    min_vertical_bounce=min_vertical_bounce
                )
                
                # BONUS for double-step cycle (2x period) - provides left-right symmetry
                if abs(duration - estimated_full_period) < 10:
                    score -= 0.3  # Strong bonus for full stride
                elif abs(duration - estimated_half_period) < 5:
                    score -= 0.1  # Smaller bonus for half stride
                
                if score < best_score:
                    best_score = score
                    best_pair = (start_idx, end_idx)
        
        print(f"[SeamlessLoopTool] Best cycle: frames {best_pair[0]}-{best_pair[1]} "
              f"(duration: {best_pair[1] - best_pair[0]}, score: {best_score:.3f})")
        
        return best_pair
