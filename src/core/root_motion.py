"""
Root Motion Processing Module.

Handles conversion between Root Motion and In-Place animation modes.
"""

import numpy as np
from typing import Optional, Tuple


# Axis mapping presets for common coordinate systems
AXIS_PRESETS = {
    "Y_UP": {"up": 1, "forward": 2, "right": 0},      # MotionBuilder, Maya (default)
    "Z_UP": {"up": 2, "forward": 1, "right": 0},      # 3ds Max, some Mocap systems
    "Z_UP_NEG_Y": {"up": 2, "forward": 0, "right": 1}, # Blender
}


class RootProcessor:
    """
    Processes root bone trajectory for loop animation.
    
    Supports:
    - In-Place mode: Remove horizontal translation, keep height
    - Reset Origin: Move character to world origin at frame 0
    - Configurable axis mapping for different coordinate systems
    """
    
    def __init__(self, up_axis: int = 1, forward_axis: int = 2, right_axis: int = 0):
        """
        Initialize with axis configuration.
        
        Args:
            up_axis: Index for vertical (height) axis (default: 1 = Y)
            forward_axis: Index for forward movement axis (default: 2 = Z)
            right_axis: Index for lateral movement axis (default: 0 = X)
        """
        self.up_axis = up_axis
        self.forward_axis = forward_axis
        self.right_axis = right_axis
    
    @classmethod
    def from_preset(cls, preset: str) -> "RootProcessor":
        """
        Create RootProcessor from a preset name.
        
        Args:
            preset: One of "Y_UP", "Z_UP", or "Z_UP_NEG_Y"
            
        Returns:
            Configured RootProcessor instance
        """
        if preset not in AXIS_PRESETS:
            raise ValueError(f"Unknown preset: {preset}. Available: {list(AXIS_PRESETS.keys())}")
        config = AXIS_PRESETS[preset]
        return cls(up_axis=config["up"], forward_axis=config["forward"], right_axis=config["right"])
    
    def get_horizontal_axes(self) -> Tuple[int, int]:
        """Get the two horizontal (ground plane) axes."""
        return (self.right_axis, self.forward_axis)

    def process_in_place(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Convert trajectory to in-place animation.
        
        FORCES horizontal position to world origin (0, 0).
        X (left/right) and Z (forward/back) are set to exactly 0.0.
        Y (height) bounce and all rotation data are preserved.
        
        This allows game engines to control character movement programmatically.
        
        Args:
            trajectory: Array of shape (num_frames, 3) for XYZ 
                       or (num_frames, 6) for XYZ + rotation
                       
        Returns:
            Modified trajectory with X=0, Z=0, Y preserved
        """
        # Make a copy to avoid modifying original
        result = trajectory.copy()
        
        # Force X and Z to exactly 0.0 (world origin)
        # This allows game engine to control character position
        result[:, 0] = 0.0  # X = 0
        result[:, 2] = 0.0  # Z = 0
        
        # Y (column 1) is preserved - keeps the vertical bounce
        # Columns 3+ (rotations if present) are untouched
        return result

    def align_orientation(self, trajectory: np.ndarray, target_rot_y: Optional[float] = None) -> np.ndarray:
        """
        Align root orientation by offsetting rotation Y (Euler) by a constant amount.

        If target_rot_y is None, this is a no-op that returns a copy.
        """
        if target_rot_y is None or len(trajectory) == 0 or trajectory.shape[1] < 5:
            return trajectory.copy()

        result = trajectory.copy()
        offset = target_rot_y - float(result[0, 4])
        result[:, 4] = result[:, 4] + offset
        return result

    def reset_origin(self, trajectory: np.ndarray) -> np.ndarray:
        """
        Translate trajectory so frame 0 is at world origin.
        
        Args:
            trajectory: Array of shape (num_frames, 3+)
            
        Returns:
            Translated trajectory with frame 0 at origin
        """
        # Get the offset from origin (first frame's position)
        offset = trajectory[0].copy()
        
        # Subtract offset from all frames
        result = trajectory - offset
        
        return result

    def trim_to_loop(self, trajectory: np.ndarray, loop_frame: int) -> np.ndarray:
        """
        Trim trajectory to end at the specified loop frame.
        
        Args:
            trajectory: Array of shape (num_frames, N)
            loop_frame: Last frame to include (inclusive)
            
        Returns:
            Trimmed trajectory with frames [0, loop_frame]
        """
        if loop_frame < 0 or loop_frame >= len(trajectory):
            raise ValueError(f"loop_frame {loop_frame} out of range [0, {len(trajectory)-1}]")
        
        return trajectory[:loop_frame + 1].copy()

    def extract_segment(self, trajectory: np.ndarray, start: int, end: int) -> np.ndarray:
        """
        Extract a trajectory segment [start, end] (inclusive).
        """
        return trajectory[start:end + 1].copy()

    def crop_sequence(
        self, 
        trajectory: np.ndarray, 
        start_idx: int, 
        end_idx: int
    ) -> np.ndarray:
        """
        Crop trajectory to a specific segment.
        
        This is used to extract a walking segment from the middle of a 
        longer animation (e.g., extract frames 100-140 from a T-pose->walk->T-pose).
        
        Args:
            trajectory: Full trajectory array
            start_idx: First frame to include (relative index, 0-based)
            end_idx: Last frame to include (inclusive, relative index)
            
        Returns:
            Cropped trajectory with frame 0 at original start_frame position
        """
        # Extract the segment
        cropped = self.extract_segment(trajectory, start_idx, end_idx)
        
        # DON'T reset origin here - that's a separate optional step
        # Just return the raw cropped segment
        return cropped

    def blend_loop_ends(
        self, 
        trajectory: np.ndarray, 
        blend_frames: int = 5,
        skip_forward_axis: bool = False
    ) -> np.ndarray:
        """
        Make trajectory seamless using Linear Offset Compensation.
        
        This is the recommended algorithm from the technical report:
        Instead of blending/fading frames (which causes foot sliding),
        distribute the first-last frame error evenly across ALL frames.
        
        Formula: V'[i] = V[i] + delta * (i / N)
        Where delta = V[start] - V[end]
        
        IMPORTANT: For root-motion loops (in_place=False), do NOT close
        the forward axis to avoid velocity snapping.
        
        Args:
            trajectory: Array of shape (num_frames, N)
            blend_frames: Ignored - kept for API compatibility.
                         Linear offset uses all frames.
            skip_forward_axis: If True, don't apply offset compensation
                              to the forward axis (for root-motion loops).
            
        Returns:
            Seamless trajectory with last frame matching first frame
        """
        num_frames = len(trajectory)
        if num_frames < 2:
            return trajectory.copy()
        
        result = trajectory.copy()
        
        # Calculate the gap between first and last frame
        first_frame = trajectory[0]
        last_frame = trajectory[-1]
        delta = first_frame - last_frame
        
        # For root-motion loops: skip forward axis to preserve motion
        if skip_forward_axis:
            # forward_axis is typically Z (index 2) for Maya/MoBu
            delta[self.forward_axis] = 0.0
            print(f"[SeamlessLoopTool] Skipping forward axis {self.forward_axis} in blend")
        
        # Distribute the error linearly across all frames
        # At frame 0: add 0% of delta (keep original)
        # At frame N: add 100% of delta (exactly match first frame)
        for i in range(num_frames):
            ratio = i / (num_frames - 1)
            result[i] = trajectory[i] + delta * ratio
        
        return result

    def make_seamless_linear(
        self, 
        trajectory: np.ndarray,
        forward_axis: int = 2
    ) -> np.ndarray:
        """
        Make trajectory seamless, handling forward axis specially.
        
        For the forward axis (usually Z), we DON'T close the loop
        because that would teleport the character back to start.
        For all other axes, we apply linear offset compensation.
        
        Args:
            trajectory: Array of shape (num_frames, 6) [X,Y,Z,RotX,RotY,RotZ]
            forward_axis: Column index for forward movement (default: 2 = Z)
            
        Returns:
            Seamless trajectory
        """
        num_frames = len(trajectory)
        if num_frames < 2:
            return trajectory.copy()
        
        result = trajectory.copy()
        num_cols = trajectory.shape[1]
        
        for col in range(num_cols):
            if col == forward_axis:
                # Skip forward axis - don't close the position loop
                continue
            
            # Apply linear offset compensation to this column
            val_start = trajectory[0, col]
            val_end = trajectory[-1, col]
            delta = val_start - val_end
            
            for i in range(num_frames):
                ratio = i / (num_frames - 1)
                result[i, col] = trajectory[i, col] + delta * ratio
        
        return result
