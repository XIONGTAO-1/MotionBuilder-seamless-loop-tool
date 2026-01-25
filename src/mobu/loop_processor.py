"""
Loop Processor Service.

Orchestrates the full loop processing pipeline by connecting
the MoBu adapter with the core algorithms.
"""

import numpy as np
import logging
from typing import Tuple, Optional, List

from core.loop_analysis import FrameAnalyzer
from core.root_motion import RootProcessor

logger = logging.getLogger(__name__)


class LoopProcessorService:
    """
    High-level service that orchestrates the seamless loop workflow.
    
    Connects:
    - MoBuAdapter (or MockMoBuAdapter) for data I/O
    - FrameAnalyzer for finding loop points
    - RootProcessor for in-place conversion
    """

    def __init__(self, adapter):
        """
        Initialize with an adapter.
        
        Args:
            adapter: MoBuAdapter or MockMoBuAdapter instance
        """
        self.adapter = adapter
        self.frame_analyzer = FrameAnalyzer()
        self.root_processor = RootProcessor()
        
        # Results storage
        self.best_loop_frame: Optional[int] = None
        self.loop_start_frame: Optional[int] = None
        self.processed_trajectory: Optional[np.ndarray] = None
        self.processed_root_pre_inplace: Optional[np.ndarray] = None
        self.processed_in_place: bool = False
        self.processed_contacts_abs: dict = {}
        self.processed_contact_source_fps: Optional[float] = None

    def analyze_loop_points(
        self, 
        root_name: str = "Hips",
        search_range: Optional[Tuple[int, int]] = None,
        use_pose_similarity: bool = True,
        min_pose_count: int = 2,
        use_velocity_weighting: bool = True
    ) -> int:
        """
        Find the best loop point in the current animation.
        
        Uses velocity-weighted cost function as recommended:
        - Position similarity alone is not enough
        - Velocity consistency is critical (weight 5x higher)
        - Focus on Hip Y-translation
        
        Args:
            root_name: Name of root bone to analyze
            search_range: Optional (start, end) frame range to search.
                         If None, uses second half of animation.
            use_pose_similarity: Prefer full-pose matching when available
            min_pose_count: Minimum bones required to enable pose matching
            use_velocity_weighting: Use the improved velocity-weighted algorithm (recommended)
                         
        Returns:
            The frame number with the best loop potential (absolute frame)
        """
        trajectory = self.adapter.get_root_trajectory(root_name)
        start_frame, end_frame = self.adapter.get_frame_range()
        num_frames = len(trajectory)
        
        print(f"[SeamlessLoopTool] Trajectory has {num_frames} frames")
        print(f"[SeamlessLoopTool] Frame range: {start_frame} - {end_frame}")
        
        # Convert search range to relative indices
        if search_range is None:
            # Default: search in the last 50% of the animation (relative indices)
            mid_idx = num_frames // 2
            search_start_idx = mid_idx
            search_end_idx = num_frames
        else:
            # Convert absolute frames to relative indices
            search_start_idx = max(0, search_range[0] - start_frame)
            search_end_idx = min(num_frames, search_range[1] - start_frame + 1)
        
        if search_start_idx >= search_end_idx:
            raise ValueError(f"Invalid search range indices: ({search_start_idx}, {search_end_idx})")
        
        print(f"[SeamlessLoopTool] Searching indices {search_start_idx} - {search_end_idx}")
        
        # Use the new velocity-weighted algorithm (recommended by NotebookLM)
        if use_velocity_weighting:
            best_idx = self.frame_analyzer.find_loop_frame_with_velocity(
                trajectory,
                search_range=(search_start_idx, search_end_idx),
                reference_index=0
            )
            # Convert back to absolute frame number
            self.best_loop_frame = start_frame + best_idx
            print(f"[SeamlessLoopTool] Best loop index={best_idx}, frame={self.best_loop_frame}")
            return self.best_loop_frame
        
        # Fallback to old method (for compatibility)
        reference_pose = trajectory[0]
        
        def score_frame(idx: int) -> float:
            if idx < 0 or idx >= len(trajectory):
                return float("inf")
            current_pose = trajectory[idx]
            return self.frame_analyzer.calculate_similarity_score(
                reference_pose.reshape(1, -1),
                current_pose.reshape(1, -1)
            )
        
        best_idx = search_start_idx
        best_score = float('inf')
        for idx in range(search_start_idx, search_end_idx):
            score = score_frame(idx)
            if score < best_score:
                best_score = score
                best_idx = idx
        
        self.best_loop_frame = start_frame + best_idx
        return self.best_loop_frame

    def find_walk_cycle(
        self,
        root_name: str = "Hips",
        min_cycle_frames: int = 20,
        max_cycle_frames: int = 60,
        up_axis: int = 1,
        min_average_velocity: float = 5.0,
        min_vertical_bounce: float = 0.0
    ) -> Tuple[int, int]:
        """
        Find the best walk cycle using peak/valley detection.
        
        This algorithm:
        1. Uses Hip Y-axis valleys to detect foot landings
        2. Estimates gait period via autocorrelation
        3. Evaluates all candidate loop pairs with velocity+pose scoring
        4. Filters out static poses using min_average_velocity
        
        Args:
            root_name: Name of root bone
            min_cycle_frames: Minimum expected cycle length
            max_cycle_frames: Maximum expected cycle length
            up_axis: Index of vertical axis (1=Y for Maya/MoBu, 2=Z for 3ds Max)
            min_average_velocity: Minimum avg velocity (units/sec)
            min_vertical_bounce: Minimum required Y-axis range (max - min) for the segment
            
        Returns:
            Tuple of (start_frame, end_frame) as absolute frame numbers
        """
        trajectory = self.adapter.get_root_trajectory(root_name)
        start_frame, _ = self.adapter.get_frame_range()
        
        # Use the new walk cycle detection algorithm
        start_idx, end_idx = self.frame_analyzer.find_best_walk_cycle(
            trajectory,
            min_cycle_frames=min_cycle_frames,
            max_cycle_frames=max_cycle_frames,
            up_axis=up_axis,
            min_average_velocity=min_average_velocity,
            min_vertical_bounce=min_vertical_bounce
        )
        
        # Store the end frame as the best loop frame (for compatibility)
        self.best_loop_frame = start_frame + end_idx
        
        # Return absolute frame numbers
        return (start_frame + start_idx, start_frame + end_idx)

    def process_in_place(self, root_name: str = "Hips") -> np.ndarray:
        """
        Convert the root motion to in-place animation.
        
        Args:
            root_name: Name of root bone to process
            
        Returns:
            The processed trajectory
        """
        trajectory = self.adapter.get_root_trajectory(root_name)
        self.processed_root_pre_inplace = trajectory.copy()
        self.processed_in_place = True
        
        # Do NOT reset origin! 
        # process_in_place locks to frame 0, which preserves the initial position.
        # Resetting here would force the character to world origin (0,0,0).
        # trajectory = self.root_processor.reset_origin(trajectory)
        
        self.processed_trajectory = self.root_processor.process_in_place(trajectory)
        
        return self.processed_trajectory

    def _ensure_take_copy(self, root_name: str, preserve_original: bool) -> None:
        if not preserve_original:
            if hasattr(self.adapter, "clear_all_animation"):
                try:
                    self.adapter.clear_all_animation(root_name)
                except Exception as e:
                    print(f"[SeamlessLoopTool] Clear animation failed: {e}, continuing")
            return

        try:
            take_name = None
            if hasattr(self.adapter, "get_current_take_name"):
                take_name = self.adapter.get_current_take_name()
            needs_new_take = not take_name or not take_name.endswith("_InPlace")

            if needs_new_take:
                if hasattr(self.adapter, "create_clean_take"):
                    self.adapter.create_clean_take("_InPlace", root_name=root_name)
                    return
                if hasattr(self.adapter, "create_take_copy"):
                    self.adapter.create_take_copy("_InPlace", root_name=root_name)
                if hasattr(self.adapter, "clear_all_animation"):
                    self.adapter.clear_all_animation(root_name)
            else:
                if hasattr(self.adapter, "clear_all_animation"):
                    self.adapter.clear_all_animation(root_name)
        except Exception as e:
            print(f"[SeamlessLoopTool] Take copy failed: {e}, continuing with current take")

    def apply_changes(
        self,
        root_name: str = "Hips",
        preserve_original: bool = True,
        target_fps: Optional[float] = None,
    ) -> None:
        """
        Write the processed trajectory back to MotionBuilder.
        
        Args:
            root_name: Name of root bone to update
            preserve_original: If True, create a clean take before injecting
        """
        if self.processed_trajectory is None:
            raise RuntimeError("No processed trajectory. Call process_in_place first.")

        trajectory = self.processed_trajectory
        if target_fps is not None and hasattr(self.adapter, "get_current_fps"):
            source_fps = self.adapter.get_current_fps()
            trajectory = self.root_processor.resample_trajectory_to_fps(
                trajectory,
                source_fps=source_fps,
                target_fps=target_fps,
            )

        self._ensure_take_copy(root_name, preserve_original)
        if target_fps is not None and hasattr(self.adapter, "set_transport_fps"):
            self.adapter.set_transport_fps(target_fps)
        self.adapter.set_root_trajectory(
            root_name,
            trajectory,
            start_frame=0
        )
        if hasattr(self.adapter, "_set_take_time_span"):
            self.adapter._set_take_time_span(0, len(trajectory) - 1)

    def _map_contacts_abs_to_local(
        self,
        intervals_abs: List[Tuple[int, int]],
        loop_start_frame_abs: Optional[int],
        source_fps: Optional[float],
        target_fps: Optional[float],
        target_frame_count: Optional[int],
    ) -> List[Tuple[int, int]]:
        if not intervals_abs:
            return []
        if loop_start_frame_abs is None:
            return []
        if not target_frame_count or target_frame_count <= 0:
            return []

        if source_fps is None or target_fps is None:
            mapped = [
                (start - loop_start_frame_abs, end - loop_start_frame_abs)
                for start, end in intervals_abs
            ]
        else:
            def map_frame(frame: int) -> int:
                time_sec = (frame - loop_start_frame_abs) / source_fps
                return int(round(time_sec * target_fps))

            mapped = [(map_frame(start), map_frame(end)) for start, end in intervals_abs]

        result = []
        last_frame = target_frame_count - 1
        for start, end in mapped:
            if start > end:
                start, end = end, start
            start = max(0, min(last_frame, start))
            end = max(0, min(last_frame, end))
            if end >= start:
                result.append((start, end))

        return result

    def apply_changes_hierarchy(
        self, 
        root_name: str = "Hips", 
        preserve_original: bool = True,
        target_fps: Optional[float] = None,
        left_foot: Optional[str] = None,
        right_foot: Optional[str] = None,
        left_toe: Optional[str] = None,
        right_toe: Optional[str] = None,
        ground_height: float = 0.0,
        enable_foot_fix: bool = True,
        contact_height_threshold: float = 2.0,
        contact_speed_threshold: float = 0.5,
        contact_min_span: int = 3,
    ) -> None:
        """
        Write all processed bone data back to MotionBuilder.
        
        This is the hierarchy-aware version that writes ALL bones collected
        during create_seamless_loop_hierarchy.
        
        Args:
            root_name: Name of root bone (used for take creation)
            preserve_original: If True, create a clean take before injecting
            left_foot/right_foot: Foot bone names for contact correction
            left_toe/right_toe: Toe bone names for contact correction
            ground_height: Ground plane height (default 0.0)
            enable_foot_fix: If True, apply foot contact correction
            contact_height_threshold: Max height to consider contact
            contact_speed_threshold: Max speed to consider contact
            contact_min_span: Minimum frame span for contact
        """
        if not hasattr(self, 'processed_data') or not self.processed_data:
            raise RuntimeError("No processed data. Call create_seamless_loop_hierarchy first.")

        current_fps = None
        if target_fps is not None and hasattr(self.adapter, "get_current_fps"):
            try:
                current_fps = self.adapter.get_current_fps()
            except Exception:
                current_fps = None
        if target_fps is not None and current_fps is None:
            current_fps = self.processed_contact_source_fps

        self._ensure_take_copy(root_name, preserve_original)
        if target_fps is not None and hasattr(self.adapter, "set_transport_fps"):
            self.adapter.set_transport_fps(target_fps)
        
        bone_count = 0
        resampled_root = None
        for bone_name, trajectory in self.processed_data.items():
            try:
                if target_fps is not None and current_fps is not None:
                    trajectory = self.root_processor.resample_trajectory_to_fps(
                        trajectory,
                        source_fps=current_fps,
                        target_fps=target_fps,
                    )
                if bone_name == root_name:
                    resampled_root = trajectory
                if hasattr(self.adapter, 'set_node_trajectory'):
                    self.adapter.set_node_trajectory(bone_name, trajectory, start_frame=0)
                else:
                    self.adapter.set_root_trajectory(bone_name, trajectory, start_frame=0)
                bone_count += 1
            except Exception as e:
                print(f"[SeamlessLoopTool] Warning: Failed to write bone '{bone_name}': {e}")
        
        # Set final time span
        span_source = resampled_root if resampled_root is not None else self.processed_trajectory
        target_frame_count = 0
        if span_source is not None:
            num_frames = len(span_source)
            target_frame_count = num_frames
            if hasattr(self.adapter, '_set_take_time_span'):
                self.adapter._set_take_time_span(0, num_frames - 1)

        target_fps_value = target_fps
        if target_fps_value is None and hasattr(self.adapter, "get_current_fps"):
            try:
                target_fps_value = self.adapter.get_current_fps()
            except Exception:
                target_fps_value = None

        logger.info(
            "FootFix: enable=%s, stored_contacts=%s, source_fps=%s, target_fps=%s, left=%s/%s, right=%s/%s, "
            "ground_height=%.3f, height_thres=%.3f, metric_thres=%.3f, min_span=%d",
            enable_foot_fix,
            bool(self.processed_contacts_abs),
            self.processed_contact_source_fps if self.processed_contact_source_fps is not None else "no",
            target_fps_value if target_fps_value is not None else "no",
            left_foot,
            left_toe,
            right_foot,
            right_toe,
            float(ground_height),
            float(contact_height_threshold),
            float(contact_speed_threshold),
            int(contact_min_span),
        )

        if enable_foot_fix:
            if not self.processed_contacts_abs:
                logger.info("FootFix: no stored contacts; skipping clamp")
            elif self.loop_start_frame is None:
                logger.info("FootFix: missing loop_start_frame; skipping clamp")
            elif target_frame_count <= 0:
                logger.info("FootFix: missing target frame count; skipping clamp")
            elif not hasattr(self.adapter, "clamp_node_to_ground"):
                logger.info("FootFix: adapter lacks clamp_node_to_ground; skipping clamp")
            else:
                logger.info("FootFix: using stored contacts from Process; thresholds ignored in Apply")
                for foot_name, toe_name in ((left_foot, left_toe), (right_foot, right_toe)):
                    if not foot_name:
                        continue
                    intervals_abs = self.processed_contacts_abs.get(foot_name, [])
                    intervals_local = self._map_contacts_abs_to_local(
                        intervals_abs,
                        loop_start_frame_abs=self.loop_start_frame,
                        source_fps=self.processed_contact_source_fps,
                        target_fps=target_fps_value,
                        target_frame_count=target_frame_count,
                    )
                    if not intervals_local:
                        continue
                    self.adapter.clamp_node_to_ground(
                        foot_name,
                        intervals_local,
                        ground_height=ground_height,
                    )
                    if toe_name:
                        self.adapter.clamp_node_to_ground(
                            toe_name,
                            intervals_local,
                            ground_height=ground_height,
                        )
        
        logger.info("Wrote %d bones to new Take", bone_count)

    def detect_contact_intervals(
        self,
        heights: List[float],
        speeds: List[float],
        height_threshold: float = 2.0,
        speed_threshold: float = 0.5,
        min_span: int = 3,
    ) -> List[Tuple[int, int]]:
        if len(heights) != len(speeds):
            raise ValueError("Heights and speeds length mismatch")

        contact_mask = [
            (height <= height_threshold) and (speed <= speed_threshold)
            for height, speed in zip(heights, speeds)
        ]
        return self._extract_intervals(contact_mask, min_span)

    def _compute_contact_intervals_for_foot(
        self,
        foot_name: str,
        toe_name: Optional[str],
        start_frame: int,
        end_frame: int,
        ground_height: float,
        height_threshold: float,
        speed_threshold: float,
        min_span: int,
        root_velocity: Optional[np.ndarray],
    ) -> List[Tuple[int, int]]:
        foot_positions = self._get_world_positions(foot_name, start_frame, end_frame)
        if foot_positions.size == 0:
            return []

        toe_positions = None
        if toe_name:
            toe_positions = self._get_world_positions(toe_name, start_frame, end_frame)

        up_axis = self.root_processor.up_axis
        heights = []
        for idx, foot_pos in enumerate(foot_positions):
            foot_height = foot_pos[up_axis]
            if toe_positions is not None and idx < len(toe_positions):
                toe_height = toe_positions[idx][up_axis]
                height = min(foot_height, toe_height)
            else:
                height = foot_height
            heights.append(height - ground_height)

        metric = self._compute_contact_metric(foot_positions, root_velocity)
        return self.detect_contact_intervals(
            heights,
            metric,
            height_threshold=height_threshold,
            speed_threshold=speed_threshold,
            min_span=min_span,
        )

    def apply_stance_correction(
        self,
        take_name: str,
        node_name: str,
        contact_intervals: List[Tuple[int, int]],
        ground_height: float = 0.0,
    ) -> None:
        if not contact_intervals:
            return

        start_frame, end_frame = self.adapter.get_frame_range()

        if hasattr(self.adapter, "get_node_trajectory"):
            trajectory = self.adapter.get_node_trajectory(
                node_name, start_frame=start_frame, end_frame=end_frame
            )
        else:
            trajectory = self.adapter.get_root_trajectory(node_name)

        corrected = trajectory.copy()
        up_axis = self.root_processor.up_axis

        for start, end in contact_intervals:
            corrected[start : end + 1, up_axis] = ground_height

        if hasattr(self.adapter, "set_node_trajectory"):
            self.adapter.set_node_trajectory(node_name, corrected, start_frame=start_frame)
        else:
            self.adapter.set_root_trajectory(node_name, corrected, start_frame=start_frame)

    def apply_foot_contact_fix(
        self,
        take_name: str,
        foot_name: str,
        toe_name: Optional[str] = None,
        ground_height: float = 0.0,
        height_threshold: float = 2.0,
        speed_threshold: float = 0.5,
        min_span: int = 3,
        root_velocity: Optional[np.ndarray] = None,
    ) -> List[Tuple[int, int]]:
        start_frame, end_frame = self.adapter.get_frame_range()
        foot_positions = self._get_world_positions(foot_name, start_frame, end_frame)

        if foot_positions.size == 0:
            return []

        toe_positions = None
        if toe_name:
            toe_positions = self._get_world_positions(toe_name, start_frame, end_frame)

        up_axis = self.root_processor.up_axis
        heights = []
        for idx, foot_pos in enumerate(foot_positions):
            foot_height = foot_pos[up_axis]
            if toe_positions is not None and idx < len(toe_positions):
                toe_height = toe_positions[idx][up_axis]
                height = min(foot_height, toe_height)
            else:
                height = foot_height
            heights.append(height - ground_height)

        metric = self._compute_contact_metric(foot_positions, root_velocity)
        contacts = self.detect_contact_intervals(
            heights,
            metric,
            height_threshold=height_threshold,
            speed_threshold=speed_threshold,
            min_span=min_span,
        )
        metric_label = "speed"
        if root_velocity is not None:
            metric_label = "rel_accel"

        if heights and metric:
            logger.info(
                "FootFix '%s': contacts=%d, height[min,max]=(%.3f,%.3f), %s[min,max]=(%.3f,%.3f), "
                "thres(height<=%.3f, %s<=%.3f)",
                foot_name,
                len(contacts),
                float(min(heights)),
                float(max(heights)),
                metric_label,
                float(min(metric)),
                float(max(metric)),
                float(height_threshold),
                metric_label,
                float(speed_threshold),
            )

        if contacts:
            abs_intervals = [(start_frame + s, start_frame + e) for (s, e) in contacts[:5]]
            extra = ""
            if len(contacts) > 5:
                extra = f" (+{len(contacts) - 5} more)"
            logger.info("FootFix '%s': intervals=%s%s", foot_name, abs_intervals, extra)

            right_axis, forward_axis = self.root_processor.get_horizontal_axes()
            pre_drift = []
            for s, e in contacts:
                lock = foot_positions[s]
                segment = foot_positions[s : e + 1]
                dx = segment[:, right_axis] - lock[right_axis]
                dz = segment[:, forward_axis] - lock[forward_axis]
                pre_drift.append(float(np.max(np.sqrt(dx * dx + dz * dz))))
            logger.info(
                "FootFix '%s': pre_clamp horizontal drift max=%.3f",
                foot_name,
                float(max(pre_drift)) if pre_drift else 0.0,
            )

        # Use world-space clamping to avoid bone hierarchy collapse
        if hasattr(self.adapter, 'clamp_node_to_ground'):
            self.adapter.clamp_node_to_ground(foot_name, contacts, ground_height=ground_height)
            if toe_name:
                self.adapter.clamp_node_to_ground(toe_name, contacts, ground_height=ground_height)
        else:
            # Fallback to legacy method (may cause overlap issues)
            self.apply_stance_correction(
                take_name, foot_name, contacts, ground_height=ground_height
            )
            if toe_name:
                self.apply_stance_correction(
                    take_name, toe_name, contacts, ground_height=ground_height
                )

        if contacts:
            post_positions = self._get_world_positions(foot_name, start_frame, end_frame)
            if len(post_positions) == len(foot_positions):
                right_axis, forward_axis = self.root_processor.get_horizontal_axes()
                post_drift = []
                for s, e in contacts:
                    lock = post_positions[s]
                    segment = post_positions[s : e + 1]
                    dx = segment[:, right_axis] - lock[right_axis]
                    dz = segment[:, forward_axis] - lock[forward_axis]
                    post_drift.append(float(np.max(np.sqrt(dx * dx + dz * dz))))
                logger.info(
                    "FootFix '%s': post_clamp horizontal drift max=%.3f",
                    foot_name,
                    float(max(post_drift)) if post_drift else 0.0,
                )

        return contacts

    def _get_world_positions(
        self, node_name: str, start_frame: int, end_frame: int
    ) -> np.ndarray:
        if not node_name:
            return np.array([], dtype=float)

        if hasattr(self.adapter, "get_world_translations"):
            try:
                positions = self.adapter.get_world_translations(
                    node_name, start_frame=start_frame, end_frame=end_frame
                )
            except TypeError:
                positions = self.adapter.get_world_translations(node_name)
        elif hasattr(self.adapter, "get_node_trajectory"):
            positions = self.adapter.get_node_trajectory(
                node_name, start_frame=start_frame, end_frame=end_frame
            )[:, :3]
        else:
            positions = self.adapter.get_root_trajectory(node_name)[:, :3]

        return np.array(positions, dtype=float)

    @staticmethod
    def _compute_speeds(positions: np.ndarray) -> List[float]:
        if len(positions) == 0:
            return []

        speeds = [0.0]
        for idx in range(1, len(positions)):
            delta = positions[idx] - positions[idx - 1]
            speeds.append(float(np.linalg.norm(delta)))
        return speeds

    def _compute_contact_metric(
        self,
        positions: np.ndarray,
        root_velocity: Optional[np.ndarray] = None,
    ) -> List[float]:
        """
        For root-motion (root_velocity None): use world speed magnitude.
        For in-place: use relative horizontal acceleration (sliding can be constant speed).
        """
        if len(positions) == 0:
            return []
        if root_velocity is None or root_velocity.shape[1] < 3:
            return self._compute_speeds(positions)

        velocity = self._compute_relative_horizontal_velocity(positions, root_velocity)
        if len(velocity) == 0:
            return []

        accelerations = [0.0]
        for idx in range(1, len(velocity)):
            delta = velocity[idx] - velocity[idx - 1]
            accelerations.append(float(np.linalg.norm(delta)))
        return accelerations

    @staticmethod
    def _resample_vectors(vectors: Optional[np.ndarray], target_len: int) -> Optional[np.ndarray]:
        if vectors is None or target_len <= 0:
            return None
        if len(vectors) == target_len:
            return vectors
        src_t = np.linspace(0.0, 1.0, len(vectors))
        dst_t = np.linspace(0.0, 1.0, target_len)
        resampled = np.zeros((target_len, vectors.shape[1]))
        for col in range(vectors.shape[1]):
            resampled[:, col] = np.interp(dst_t, src_t, vectors[:, col])
        return resampled

    def _compute_root_velocity(self, trajectory: Optional[np.ndarray]) -> Optional[np.ndarray]:
        if trajectory is None or len(trajectory) < 2 or trajectory.shape[1] < 3:
            return None
        positions = trajectory[:, :3]
        velocity = np.zeros_like(positions)
        velocity[1:] = positions[1:] - positions[:-1]
        velocity[0] = velocity[1]
        return velocity

    def _compute_relative_horizontal_velocity(
        self,
        positions: np.ndarray,
        root_velocity: np.ndarray,
    ) -> np.ndarray:
        root_velocity = self._resample_vectors(root_velocity, len(positions))
        if root_velocity is None:
            return np.zeros((len(positions), 2), dtype=float)

        right_axis, forward_axis = self.root_processor.get_horizontal_axes()
        velocity = np.zeros((len(positions), 2), dtype=float)
        for idx in range(1, len(positions)):
            delta = positions[idx] - positions[idx - 1]
            delta[right_axis] += root_velocity[idx, right_axis]
            delta[forward_axis] += root_velocity[idx, forward_axis]
            velocity[idx, 0] = float(delta[right_axis])
            velocity[idx, 1] = float(delta[forward_axis])
        return velocity

    def _compute_relative_speeds(
        self,
        positions: np.ndarray,
        root_velocity: Optional[np.ndarray] = None,
    ) -> List[float]:
        if len(positions) == 0:
            return []
        if root_velocity is None or root_velocity.shape[1] < 3:
            return self._compute_speeds(positions)

        velocity = self._compute_relative_horizontal_velocity(positions, root_velocity)
        speeds = [0.0]
        for idx in range(1, len(velocity)):
            speed = float(np.linalg.norm(velocity[idx]))
            speeds.append(speed)
        return speeds

    @staticmethod
    def _extract_intervals(
        contact_mask: List[bool], min_span: int
    ) -> List[Tuple[int, int]]:
        intervals = []
        start = None
        for idx, is_contact in enumerate(contact_mask):
            if is_contact and start is None:
                start = idx
                continue
            if not is_contact and start is not None:
                end = idx - 1
                if (end - start + 1) >= min_span:
                    intervals.append((start, end))
                start = None
        if start is not None:
            end = len(contact_mask) - 1
            if (end - start + 1) >= min_span:
                intervals.append((start, end))
        return intervals

    def create_seamless_loop(
        self,
        root_name: str = "Hips",
        loop_frame: Optional[int] = None,
        start_frame: Optional[int] = None,
        blend_frames: int = 5,
        in_place: bool = True,
        use_cycle_detection: bool = True,
        target_rot_y: Optional[float] = None,
    ) -> np.ndarray:
        """
        Create a seamless looping animation from the current take.
        
        This is the main workflow method that:
        1. Finds the best loop segment (or uses provided start_frame/loop_frame)
        2. Crops the trajectory to that segment
        3. Blends the end frames to match the start
        4. Optionally converts to in-place animation
        
        Args:
            root_name: Name of root bone
            loop_frame: End frame of loop (absolute), or None to auto-detect
            start_frame: Start frame of loop (absolute), or None to auto-detect
            blend_frames: Number of frames to crossfade at the end
            in_place: Whether to lock XZ at frame 0's position
            use_cycle_detection: If True and frames are None, detect mid-animation cycle
            target_rot_y: If set, offset root rotation Y so frame 0 equals this value
            
        Returns:
            The processed trajectory ready for export
        """
        trajectory = self.adapter.get_root_trajectory(root_name)
        anim_start, anim_end = self.adapter.get_frame_range()
        
        print(f"[SeamlessLoopTool] Animation range: {anim_start}-{anim_end}, {len(trajectory)} frames")
        
        if start_frame is None and loop_frame is None:
            if not use_cycle_detection:
                raise ValueError("start_frame/loop_frame required when cycle detection is disabled")
            cycle_start, cycle_end = self.find_walk_cycle(root_name)
            start_frame = cycle_start
            loop_frame = cycle_end
            print(f"[SeamlessLoopTool] Detected cycle: {cycle_start}-{cycle_end}")
        elif loop_frame is None:
            raise ValueError("loop_frame is required when start_frame is provided")
        elif start_frame is None:
            start_frame = anim_start
            print(f"[SeamlessLoopTool] Using legacy start at {start_frame} for loop_frame {loop_frame}")

        # Determine crop range (start_idx, end_idx as relative indices)
        start_idx = start_frame - anim_start
        end_idx = loop_frame - anim_start
        self.best_loop_frame = loop_frame
        print(f"[SeamlessLoopTool] Using range: {start_frame}-{loop_frame} (idx {start_idx}-{end_idx})")
        
        # Validate and clamp indices
        start_idx = max(0, start_idx)
        end_idx = min(len(trajectory) - 1, end_idx)
        if start_idx > end_idx:
            print(f"[SeamlessLoopTool] Warning: invalid range, using full animation")
            start_idx = 0
            end_idx = len(trajectory) - 1
        self.loop_start_frame = anim_start + start_idx
        
        # Step 1: Crop to the segment
        trajectory = self.root_processor.crop_sequence(trajectory, start_idx, end_idx)
        print(f"[SeamlessLoopTool] Cropped to {len(trajectory)} frames")
        
        # Step 2: Blend the end frames to match the start (Linear Offset Compensation)
        trajectory = self.root_processor.blend_loop_ends(trajectory, blend_frames)
        
        # Step 3: Convert to in-place if requested
        # NOTE: process_in_place now LOCKS at frame 0, not resets to (0,0)
        #       DO NOT call reset_origin before this!
        if in_place:
            self.processed_root_pre_inplace = trajectory.copy()
            self.processed_in_place = True
            trajectory = self.root_processor.process_in_place(trajectory)
        else:
            self.processed_root_pre_inplace = None
            self.processed_in_place = False

        if target_rot_y is not None:
            trajectory = self.root_processor.align_orientation(trajectory, target_rot_y=target_rot_y)
        
        self.processed_trajectory = trajectory
        return trajectory

    def create_seamless_loop_hierarchy(
        self,
        root_name: str = "Hips",
        loop_frame: Optional[int] = None,
        start_frame: Optional[int] = None,
        blend_frames: int = 5,
        in_place: bool = True,
        use_cycle_detection: bool = True,
        target_rot_y: Optional[float] = None,
        left_foot: Optional[str] = None,
        right_foot: Optional[str] = None,
        left_toe: Optional[str] = None,
        right_toe: Optional[str] = None,
        enable_foot_fix: bool = True,
        ground_height: float = 0.0,
        contact_height_threshold: float = 2.0,
        contact_speed_threshold: float = 0.5,
        contact_min_span: int = 3,
    ) -> dict:
        """
        Create a seamless looping animation for THE ENTIRE HIERARCHY.
        
        This is the whole-hierarchy version that:
        1. Discovers all bones in the hierarchy
        2. For each bone: Resample -> Crop -> Blend
        3. For root only: Apply in-place fix
        4. Stores all processed data for bulk injection
        
        Args:
            root_name: Name of root bone
            loop_frame: End frame of loop (absolute), or None to auto-detect
            start_frame: Start frame of loop (absolute), or None to auto-detect
            blend_frames: Number of frames to crossfade
            in_place: Whether to lock root XZ
            use_cycle_detection: If True and frames are None, detect cycle
            target_rot_y: If set, offset root rotation Y so frame 0 equals this value
            left_foot/right_foot: Foot bone names for contact detection
            left_toe/right_toe: Toe bone names for contact detection
            enable_foot_fix: If True, compute and store contact intervals
            ground_height: Ground plane height for contact detection
            contact_height_threshold: Max height to consider contact
            contact_speed_threshold: Max speed/accel to consider contact
            contact_min_span: Minimum frame span for contact
            
        Returns:
            Dictionary mapping bone names to processed trajectories
        """
        anim_start, anim_end = self.adapter.get_frame_range()
        
        # Phase 1: Detect loop range using root
        if start_frame is None and loop_frame is None:
            if not use_cycle_detection:
                raise ValueError("start_frame/loop_frame required when cycle detection is disabled")
            cycle_start, cycle_end = self.find_walk_cycle(root_name)
            start_frame = cycle_start
            loop_frame = cycle_end
            print(f"[SeamlessLoopTool] Detected cycle: {cycle_start}-{cycle_end}")
        elif loop_frame is None:
            raise ValueError("loop_frame is required when start_frame is provided")
        elif start_frame is None:
            start_frame = anim_start
        
        self.best_loop_frame = loop_frame
        self.loop_start_frame = start_frame

        self.processed_contacts_abs = {}
        self.processed_contact_source_fps = None

        source_fps = None
        if hasattr(self.adapter, "get_current_fps"):
            try:
                source_fps = self.adapter.get_current_fps()
            except Exception:
                source_fps = None
        self.processed_contact_source_fps = source_fps

        if enable_foot_fix:
            root_segment = None
            if hasattr(self.adapter, "get_node_trajectory"):
                try:
                    root_segment = self.adapter.get_node_trajectory(
                        root_name, start_frame=start_frame, end_frame=loop_frame
                    )
                except Exception:
                    root_segment = None

            if root_segment is None:
                full_traj = self.adapter.get_root_trajectory(root_name)
                start_idx = max(0, start_frame - anim_start)
                end_idx = min(len(full_traj) - 1, loop_frame - anim_start)
                root_segment = full_traj[start_idx : end_idx + 1]

            root_velocity = self._compute_root_velocity(root_segment)

            for foot_name, toe_name in ((left_foot, left_toe), (right_foot, right_toe)):
                if not foot_name:
                    continue
                contacts = self._compute_contact_intervals_for_foot(
                    foot_name,
                    toe_name,
                    start_frame,
                    loop_frame,
                    ground_height,
                    contact_height_threshold,
                    contact_speed_threshold,
                    contact_min_span,
                    root_velocity,
                )
                self.processed_contacts_abs[foot_name] = [
                    (start_frame + s, start_frame + e) for (s, e) in contacts
                ]
        
        # Phase 2: Get all bones in hierarchy
        if hasattr(self.adapter, 'get_hierarchy_nodes'):
            bone_names = self.adapter.get_hierarchy_nodes(root_name)
        else:
            bone_names = [root_name]
        
        print(f"[SeamlessLoopTool] Processing {len(bone_names)} bones...")
        
        # Phase 3: Process each bone
        self.processed_data = {}
        self.processed_root_pre_inplace = None
        self.processed_in_place = in_place
        
        for i, bone_name in enumerate(bone_names):
            try:
                # Get trajectory using resampling
                if hasattr(self.adapter, 'get_node_trajectory'):
                    trajectory = self.adapter.get_node_trajectory(
                        bone_name, 
                        start_frame=start_frame, 
                        end_frame=loop_frame
                    )
                else:
                    # Fallback (less accurate)
                    full_traj = self.adapter.get_root_trajectory(bone_name)
                    start_idx = start_frame - anim_start
                    end_idx = loop_frame - anim_start
                    trajectory = self.root_processor.crop_sequence(full_traj, start_idx, end_idx)
                
                # Apply linear offset compensation to ALL bones
                trajectory = self.root_processor.blend_loop_ends(trajectory, blend_frames)
                
                # Apply in-place ONLY to root
                is_root = (bone_name == bone_names[0])
                if is_root:
                    self.processed_root_pre_inplace = trajectory.copy()

                if in_place and is_root:
                    trajectory = self.root_processor.process_in_place(trajectory)

                if target_rot_y is not None and is_root:
                    trajectory = self.root_processor.align_orientation(trajectory, target_rot_y=target_rot_y)
                
                self.processed_data[bone_name] = trajectory
                
                if i == 0 or (i + 1) % 10 == 0:
                    print(f"[SeamlessLoopTool] Processed {i + 1}/{len(bone_names)}: {bone_name}")
                    
            except Exception as e:
                print(f"[SeamlessLoopTool] Warning: Skipping bone '{bone_name}': {e}")
        
        # Store root trajectory for compatibility
        self.processed_trajectory = self.processed_data.get(root_name)
        
        print(f"[SeamlessLoopTool] Hierarchy processing complete: {len(self.processed_data)} bones")
        return self.processed_data

    def run_full_pipeline(
        self, 
        root_name: str = "Hips",
        in_place: bool = True,
        blend_frames: int = 5,
        apply: bool = False
    ) -> dict:
        """
        Run the complete loop processing pipeline.
        
        Args:
            root_name: Name of root bone
            in_place: Whether to convert to in-place animation
            blend_frames: Number of frames to blend at the loop point
            apply: Whether to apply changes back to MoBu
            
        Returns:
            Dictionary with results: loop_frame, trajectory, applied
        """
        results = {}
        
        # Use the new seamless loop workflow
        trajectory = self.create_seamless_loop(
            root_name=root_name,
            loop_frame=None,  # Auto-detect
            blend_frames=blend_frames,
            in_place=in_place
        )
        
        results["loop_frame"] = self.best_loop_frame
        results["trajectory"] = trajectory
        results["num_frames"] = len(trajectory)
        
        # Apply changes if requested
        if apply:
            self.apply_changes(root_name)
            results["applied"] = True
        else:
            results["applied"] = False
        
        return results
