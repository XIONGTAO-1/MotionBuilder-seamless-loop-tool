"""
Loop Processor Service.

Orchestrates the full loop processing pipeline by connecting
the MoBu adapter with the core algorithms.
"""

import numpy as np
from typing import Tuple, Optional

from core.loop_analysis import FrameAnalyzer
from core.root_motion import RootProcessor


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

        self._ensure_take_copy(root_name, preserve_original)
        if target_fps is not None and hasattr(self.adapter, "set_transport_fps"):
            self.adapter.set_transport_fps(target_fps)
        self.adapter.set_root_trajectory(
            root_name,
            self.processed_trajectory,
            start_frame=0
        )

    def apply_changes_hierarchy(
        self, 
        root_name: str = "Hips", 
        preserve_original: bool = True,
        target_fps: Optional[float] = None,
    ) -> None:
        """
        Write all processed bone data back to MotionBuilder.
        
        This is the hierarchy-aware version that writes ALL bones collected
        during create_seamless_loop_hierarchy.
        
        Args:
            root_name: Name of root bone (used for take creation)
            preserve_original: If True, create a clean take before injecting
        """
        if not hasattr(self, 'processed_data') or not self.processed_data:
            raise RuntimeError("No processed data. Call create_seamless_loop_hierarchy first.")

        self._ensure_take_copy(root_name, preserve_original)
        if target_fps is not None and hasattr(self.adapter, "set_transport_fps"):
            self.adapter.set_transport_fps(target_fps)
        
        bone_count = 0
        for bone_name, trajectory in self.processed_data.items():
            try:
                if hasattr(self.adapter, 'set_node_trajectory'):
                    self.adapter.set_node_trajectory(bone_name, trajectory, start_frame=0)
                else:
                    self.adapter.set_root_trajectory(bone_name, trajectory, start_frame=0)
                bone_count += 1
            except Exception as e:
                print(f"[SeamlessLoopTool] Warning: Failed to write bone '{bone_name}': {e}")
        
        # Set final time span
        if self.processed_trajectory is not None:
            num_frames = len(self.processed_trajectory)
            if hasattr(self.adapter, '_set_take_time_span'):
                self.adapter._set_take_time_span(0, num_frames - 1)
        
        print(f"[SeamlessLoopTool] Wrote {bone_count} bones to new Take")

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
            trajectory = self.root_processor.process_in_place(trajectory)

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
        
        # Phase 2: Get all bones in hierarchy
        if hasattr(self.adapter, 'get_hierarchy_nodes'):
            bone_names = self.adapter.get_hierarchy_nodes(root_name)
        else:
            bone_names = [root_name]
        
        print(f"[SeamlessLoopTool] Processing {len(bone_names)} bones...")
        
        # Phase 3: Process each bone
        self.processed_data = {}
        
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
