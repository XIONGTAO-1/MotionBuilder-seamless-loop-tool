"""
MotionBuilder Adapter Module.

Provides an abstraction layer between the core algorithms and pyfbsdk.
This allows the core logic to be tested independently of MotionBuilder.
"""

import numpy as np
from typing import Optional, Tuple, Dict

# Try to import pyfbsdk - will only work inside MotionBuilder
IN_MOTIONBUILDER = False
FBSystem = None
FBPlayerControl = None
FBApplication = None
FBFindModelByLabelName = None
FBFindModelByName = None
FBTime = None
FBTimeSpan = None
FBTimeMode = None
FBPlotOptions = None
FBCharacterPlotWhere = None
FBFbxOptions = None
FBElementAction = None
FBInterpolation = None
FBTangentMode = None
FBModelList = None
FBGetSelectedModels = None
FBModelTransformationType = None
FBMatrix = None
FBModel = None

# Robust import: try each separately to handle missing symbols
try:
    import pyfbsdk
    IN_MOTIONBUILDER = True
    print("[SeamlessLoopTool] pyfbsdk module found")
    
    # Import core classes
    FBSystem = getattr(pyfbsdk, 'FBSystem', None)
    FBPlayerControl = getattr(pyfbsdk, 'FBPlayerControl', None)
    FBApplication = getattr(pyfbsdk, 'FBApplication', None)
    FBTime = getattr(pyfbsdk, 'FBTime', None)
    FBTimeSpan = getattr(pyfbsdk, 'FBTimeSpan', None)
    FBTimeMode = getattr(pyfbsdk, 'FBTimeMode', None)
    FBPlotOptions = getattr(pyfbsdk, 'FBPlotOptions', None)
    FBCharacterPlotWhere = getattr(pyfbsdk, 'FBCharacterPlotWhere', None)
    FBFbxOptions = getattr(pyfbsdk, 'FBFbxOptions', None)
    FBElementAction = getattr(pyfbsdk, 'FBElementAction', None)
    FBInterpolation = getattr(pyfbsdk, 'FBInterpolation', None)
    FBTangentMode = getattr(pyfbsdk, 'FBTangentMode', None)
    FBModel = getattr(pyfbsdk, 'FBModel', None)
    FBMatrix = getattr(pyfbsdk, 'FBMatrix', None)
    
    # Import find functions
    FBFindModelByLabelName = getattr(pyfbsdk, 'FBFindModelByLabelName', None)
    FBFindModelByName = getattr(pyfbsdk, 'FBFindModelByName', None)
    
    # Import selection functions
    FBModelList = getattr(pyfbsdk, 'FBModelList', None)
    FBGetSelectedModels = getattr(pyfbsdk, 'FBGetSelectedModels', None)
    
    # Import enums
    FBModelTransformationType = getattr(pyfbsdk, 'FBModelTransformationType', None)
    
    print(f"[SeamlessLoopTool] FBSystem: {FBSystem is not None}")
    print(f"[SeamlessLoopTool] FBFindModelByLabelName: {FBFindModelByLabelName is not None}")
    
except ImportError as e:
    print(f"[SeamlessLoopTool] pyfbsdk not available: {e}")
    IN_MOTIONBUILDER = False
except Exception as e:
    print(f"[SeamlessLoopTool] Unexpected error importing pyfbsdk: {e}")
    IN_MOTIONBUILDER = False


def generate_unique_take_name(base_name: str, existing: list, suffix: str = "_inplace") -> str:
    """
    Generate a unique take name using a suffix and numeric increment.
    """
    candidate = f"{base_name}{suffix}"
    if candidate not in existing:
        return candidate
    index = 1
    while f"{candidate}_{index}" in existing:
        index += 1
    return f"{candidate}_{index}"


class MoBuAdapter:
    """
    Adapter for reading and writing animation data from MotionBuilder.
    
    Abstracts pyfbsdk calls into simple numpy arrays that can be processed
    by the core algorithms.
    """

    def __init__(self):
        """Initialize the adapter."""
        if not IN_MOTIONBUILDER:
            raise RuntimeError(
                "MoBuAdapter requires MotionBuilder environment. "
                "Use MockMoBuAdapter for testing outside MoBu."
            )
        self._system = FBSystem()
        self._player = FBPlayerControl()
    
    def get_current_take_name(self) -> str:
        """Get the name of the current take."""
        return self._system.CurrentTake.Name

    def get_take_names(self) -> list:
        """Return all take names in the current scene."""
        return [take.Name for take in self._system.Scene.Takes]

    def set_current_take(self, take_name: str) -> None:
        """Switch the current take by name."""
        for take in self._system.Scene.Takes:
            if take.Name == take_name:
                self._system.CurrentTake = take
                return
        raise ValueError(f"Take '{take_name}' not found")

    def create_sandbox_take(self, suffix: str = "_inplace") -> str:
        """
        Create a sandbox take for export and switch to it.
        """
        original_take = self._system.CurrentTake
        existing = self.get_take_names()
        new_name = generate_unique_take_name(original_take.Name, existing, suffix=suffix)
        sandbox_take = original_take.CopyTake(new_name)
        self._system.CurrentTake = sandbox_take
        return new_name

    def get_current_fps(self) -> float:
        """
        Get the current FPS from the transport time mode.
        """
        if self._player is None or FBTime is None:
            return 30.0

        mode = None
        if hasattr(self._player, "GetTransportFps"):
            mode = self._player.GetTransportFps()

        if FBTimeMode is not None and mode is not None:
            if mode == FBTimeMode.kFBTimeModeCustom:
                try:
                    seconds = FBTime(0, 0, 0, 1).GetSecondDouble()
                    return 1.0 / seconds if seconds > 0 else 30.0
                except Exception:
                    return 30.0

            try:
                seconds = FBTime(0, 0, 0, 1, 0, mode).GetSecondDouble()
                return 1.0 / seconds if seconds > 0 else 30.0
            except Exception:
                return 30.0

        return 30.0

    def set_transport_fps(self, target_fps: float) -> None:
        """Set the transport FPS to a custom value."""
        if self._player is None:
            return
        if FBTimeMode is not None:
            self._player.SetTransportFps(FBTimeMode.kFBTimeModeCustom, float(target_fps))
        else:
            self._player.SetTransportFps(float(target_fps))

    def plot_animation_on_skeleton(self, target_fps: float) -> None:
        """Plot animation to skeleton using plot options at target FPS."""
        if FBPlotOptions is None or FBTime is None:
            print("[SeamlessLoopTool] FBPlotOptions/FBTime not available; skipping plot")
            return

        plot_options = FBPlotOptions()
        plot_options.PlotAllTakes = False
        plot_options.PlotOnFrame = True
        plot_options.UseConstantKeyReducer = False

        if FBTimeMode is not None:
            period = FBTime(0, 0, 0, 1, 0, FBTimeMode.kFBTimeModeCustom)
        else:
            period = FBTime(0, 0, 0, 1)
        plot_options.PlotPeriod = period

        scene = self._system.Scene
        character = None
        if hasattr(scene, "Characters") and len(scene.Characters) > 0:
            character = scene.Characters[0]

        if character is not None and FBCharacterPlotWhere is not None:
            character.PlotAnimation(FBCharacterPlotWhere.kFBCharacterPlotOnSkeleton, plot_options)
            return

        print("[SeamlessLoopTool] No character found for plotting; skipping plot")

    def export_take_to_fbx(self, path: str, take_name: str) -> None:
        """Export only the specified take to FBX."""
        if FBApplication is None or FBFbxOptions is None or FBElementAction is None:
            raise RuntimeError("FBX export requires MotionBuilder environment")

        options = FBFbxOptions(False)
        options.SetAll(FBElementAction.kFBElementActionDiscard, False)

        take_count = options.GetTakeCount()
        for i in range(take_count):
            if options.GetTakeName(i) == take_name:
                options.SetTakeSelect(i, True)
                break

        FBApplication().FileSave(path, options)

    def create_clean_take(self, suffix: str = "_InPlace", root_name: str = "Hips") -> str:
        """
        Create a copy of the current take and clear animation keys.

        Args:
            suffix: Suffix to add to the take name (default: "_InPlace")
            root_name: Root bone name used to clear animation data

        Returns:
            Name of the new take
        """
        original_take = self._system.CurrentTake
        original_name = original_take.Name
        new_name = f"{original_name}{suffix}"

        original_take.CopyTake(new_name)

        for take in self._system.Scene.Takes:
            if take.Name == new_name:
                self._system.CurrentTake = take
                print(f"[SeamlessLoopTool] Created clean take: {new_name}")
                try:
                    self.clear_all_animation(root_name)
                except Exception as e:
                    print(f"[SeamlessLoopTool] Clear animation failed: {e}")
                return new_name

        raise RuntimeError(f"Failed to create clean take: {new_name}")
    
    def create_take_copy(self, suffix: str = "_InPlace", root_name: str = "Hips") -> str:
        """
        Create a copy of the current take and clear animation keys.

        Args:
            suffix: Suffix to add to the take name (default: "_InPlace")
            root_name: Root bone name used to clear animation data

        Returns:
            Name of the new take
        """
        return self.create_clean_take(suffix=suffix, root_name=root_name)

    def get_selected_model_name(self) -> Optional[str]:
        """
        Get the name of the currently selected model in Navigator.
        
        Returns:
            Name of selected model, or None if nothing selected
        """
        if FBModelList is None or FBGetSelectedModels is None:
            return None
        
        models = FBModelList()
        FBGetSelectedModels(models)
        
        if len(models) > 0:
            model = models[0]
            # Try LabelName first (more readable), then Name
            label = getattr(model, "LongName", "") or getattr(model, "LabelName", "")
            if label:
                return label
            return model.Name
        return None

    def get_frame_range(self) -> Tuple[int, int]:
        """
        Get the current animation frame range.
        
        Tries to get the take's actual time span first, falls back to loop range.
        
        Returns:
            Tuple of (start_frame, end_frame)
        """
        # Try to get the actual take time span first
        try:
            take = self._system.CurrentTake
            if hasattr(take, 'LocalTimeSpan'):
                start = take.LocalTimeSpan.GetStart().GetFrame()
                end = take.LocalTimeSpan.GetStop().GetFrame()
                if end > start:
                    return (int(start), int(end))
        except Exception:
            pass

        try:
            scene = self._system.Scene
            if hasattr(scene, 'LocalTimeSpan'):
                start = scene.LocalTimeSpan.GetStart().GetFrame()
                end = scene.LocalTimeSpan.GetStop().GetFrame()
                if end > start:
                    return (int(start), int(end))
        except Exception:
            pass
        
        # Fallback to player loop range
        start = self._player.LoopStart.GetFrame()
        end = self._player.LoopStop.GetFrame()
        return (int(start), int(end))


    # Common root bone name patterns to search for
    ROOT_BONE_PATTERNS = [
        "Hips", "Hip", "Pelvis", "Root", "Bip01", "Bip001",
        "mixamorig:Hips", "mixamorig:Hip",
        "CC_Base_Hip", "CC_Base_Pelvis",
        "root", "hips", "hip", "pelvis",
    ]
    
    def find_root_bone(self, root_name: str = "Hips") -> Optional[FBModel]:
        """
        Find the root bone in the scene.
        
        First tries the exact name, then searches for common root bone patterns.
        
        Args:
            root_name: Preferred name to look for first
            
        Returns:
            The actual bone model found, or None if not found
        """
        # Helper to check if model exists (relaxed check)
        def is_valid_model(model):
            try:
                return model is not None and hasattr(model, "Translation")
            except Exception:
                return False

        def model_display_name(model) -> str:
            label = getattr(model, "LabelName", "")
            if label:
                return label
            return getattr(model, "Name", "")

        def find_model_exact(name: str):
            name = (name or "").strip()
            if not name:
                return None
            model = FBFindModelByLabelName(name)
            if is_valid_model(model):
                return model
            if FBFindModelByName is not None:
                model = FBFindModelByName(name)
                if is_valid_model(model):
                    return model
            return None
        
        # Try exact match first - trust the user's selection
        model = find_model_exact(root_name)
        if is_valid_model(model):
            print(f"[SeamlessLoopTool] Found exact match: {model_display_name(model)}")
            return model
        
        # Build search list with user's name first
        root_name = (root_name or "").strip()
        search_list = [root_name] if root_name else []
        for pattern in self.ROOT_BONE_PATTERNS:
            if pattern.lower() != root_name.lower() and pattern not in search_list:
                search_list.append(pattern)
        
        # Try each pattern with exact label/name lookup
        for pattern in search_list:
            model = find_model_exact(pattern)
            if model is not None:
                print(f"[SeamlessLoopTool] Found pattern match: {model_display_name(model)}")
                return model
        
        # Search all models in scene for matching name (fuzzy search)
        scene = self._system.Scene
        found_models = []
        patterns_lower = [pattern.lower() for pattern in search_list if pattern]
        
        # Use recursive helper to iterate all descendants
        def collect_models(parent):
            children = getattr(parent, "Children", None)
            if children is None:
                return
            for child in children:
                if is_valid_model(child):
                    name = getattr(child, "Name", "")
                    label = getattr(child, "LabelName", "")
                    name_lower = name.lower()
                    label_lower = label.lower()
                    for pattern_lower in patterns_lower:
                        if pattern_lower in name_lower or pattern_lower in label_lower:
                            found_models.append(child)
                            break
                collect_models(child)
        
        collect_models(scene.RootModel)
        
        if found_models:
            print(f"[SeamlessLoopTool] Fuzzy matched: {model_display_name(found_models[0])} from {len(found_models)} candidates")
            return found_models[0]
        
        print(f"[SeamlessLoopTool] No bone found matching: {search_list[:5]}...")
        return None

    def _collect_child_models(self, parent, result) -> None:
        children = getattr(parent, "Children", None)
        if not children:
            return
        for child in children:
            result.append(child)
            self._collect_child_models(child, result)

    def clear_all_animation(self, root_name: str = "Hips") -> None:
        """
        Clear all animation keys for the root hierarchy in the current take.
        """
        model = self.find_root_bone(root_name)
        if model is None:
            raise ValueError(f"Root bone '{root_name}' not found for clearing")

        targets = [model]
        self._collect_child_models(model, targets)

        clear_start = -10**9
        clear_end = 10**9
        for target in targets:
            self._clear_keys_all_layers(target, clear_start, clear_end, True)

    def get_hierarchy_nodes(self, root_name: str = "Hips") -> list:
        """
        Get a list of all bone names in the hierarchy starting from the root.
        
        Args:
            root_name: Name of the root bone
            
        Returns:
            List of bone names (strings) including root and all descendants
        """
        model = self.find_root_bone(root_name)
        if model is None:
            raise ValueError(f"Root bone '{root_name}' not found")
        
        targets = [model]
        self._collect_child_models(model, targets)
        
        # Extract names
        names = []
        for m in targets:
            label = getattr(m, "LabelName", "") or getattr(m, "Name", "")
            if label:
                names.append(label)
        
        print(f"[SeamlessLoopTool] Found {len(names)} bones in hierarchy")
        return names

    def get_node_trajectory(
        self, 
        node_name: str, 
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None
    ) -> np.ndarray:
        """
        Extract a bone's trajectory over a specified frame range using resampling.
        
        This method uses FCurve.Evaluate() to sample at integer frames,
        ensuring clean data regardless of source keyframe positions.
        
        Args:
            node_name: Name of the bone to sample
            start_frame: Start frame (absolute), defaults to take start
            end_frame: End frame (absolute), defaults to take end
            
        Returns:
            Array of shape (num_frames, 6) with [X, Y, Z, RotX, RotY, RotZ]
        """
        model = self.find_root_bone(node_name)  # Reuses fuzzy matching
        if model is None:
            raise ValueError(f"Bone '{node_name}' not found")
        
        if start_frame is None or end_frame is None:
            anim_start, anim_end = self.get_frame_range()
            start_frame = start_frame if start_frame is not None else anim_start
            end_frame = end_frame if end_frame is not None else anim_end
        
        num_frames = end_frame - start_frame + 1
        trajectory = np.zeros((num_frames, 6))
        
        for i, frame in enumerate(range(start_frame, end_frame + 1)):
            time = FBTime(0, 0, 0, frame)
            
            # Get translation
            translation = model.Translation.GetAnimationNode()
            if translation and hasattr(translation, "Nodes") and len(translation.Nodes) >= 3:
                trajectory[i, 0] = translation.Nodes[0].FCurve.Evaluate(time)
                trajectory[i, 1] = translation.Nodes[1].FCurve.Evaluate(time)
                trajectory[i, 2] = translation.Nodes[2].FCurve.Evaluate(time)
            
            # Get rotation
            rotation = model.Rotation.GetAnimationNode()
            if rotation and hasattr(rotation, "Nodes") and len(rotation.Nodes) >= 3:
                trajectory[i, 3] = rotation.Nodes[0].FCurve.Evaluate(time)
                trajectory[i, 4] = rotation.Nodes[1].FCurve.Evaluate(time)
                trajectory[i, 5] = rotation.Nodes[2].FCurve.Evaluate(time)
        
        return trajectory

    def get_world_translations(
        self,
        node_name: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> np.ndarray:
        """
        Sample world-space translations for a node over a frame range.
        """
        model = self.find_root_bone(node_name)
        if model is None:
            raise ValueError(f"Bone '{node_name}' not found")

        if start_frame is None or end_frame is None:
            anim_start, anim_end = self.get_frame_range()
            start_frame = start_frame if start_frame is not None else anim_start
            end_frame = end_frame if end_frame is not None else anim_end

        num_frames = end_frame - start_frame + 1
        positions = np.zeros((num_frames, 3))

        for i, frame in enumerate(range(start_frame, end_frame + 1)):
            time = FBTime(0, 0, 0, frame)

            if FBMatrix is not None and FBModelTransformationType is not None:
                fb_matrix = FBMatrix()
                model.GetMatrix(
                    fb_matrix,
                    FBModelTransformationType.kModelTransformation_Transformation,
                    True,
                    time,
                )
                positions[i, 0] = fb_matrix[0][3]
                positions[i, 1] = fb_matrix[1][3]
                positions[i, 2] = fb_matrix[2][3]
            else:
                translation = model.Translation.GetAnimationNode()
                if translation and hasattr(translation, "Nodes") and len(translation.Nodes) >= 3:
                    positions[i, 0] = translation.Nodes[0].FCurve.Evaluate(time)
                    positions[i, 1] = translation.Nodes[1].FCurve.Evaluate(time)
                    positions[i, 2] = translation.Nodes[2].FCurve.Evaluate(time)

        return positions

    def set_node_trajectory(
        self,
        node_name: str,
        trajectory: np.ndarray,
        start_frame: int = 0
    ) -> None:
        """
        Write a trajectory to a specific bone, starting at given frame.
        
        Args:
            node_name: Name of the bone
            trajectory: Array of shape (num_frames, 3 or 6)
            start_frame: Frame to start writing from (default 0)
        """
        model = self.find_root_bone(node_name)
        if model is None:
            raise ValueError(f"Bone '{node_name}' not found in scene")

        end_frame = start_frame + int(len(trajectory)) - 1
        has_rotation = trajectory.shape[1] >= 6

        self._clear_keys_all_layers(model, start_frame, end_frame, True)

        translation_node, rotation_node = self._ensure_animation_nodes(model)
        translation_nodes = None
        rotation_nodes = None

        if translation_node is not None and hasattr(translation_node, "Nodes"):
            if len(translation_node.Nodes) >= 3:
                translation_nodes = translation_node.Nodes

        if has_rotation and rotation_node is not None and hasattr(rotation_node, "Nodes"):
            if len(rotation_node.Nodes) >= 3:
                rotation_nodes = rotation_node.Nodes

        for i, row in enumerate(trajectory):
            time = FBTime(0, 0, 0, int(start_frame + i))

            if translation_nodes is not None:
                key = translation_nodes[0].FCurve.KeyAdd(time, float(row[0]))
                self._apply_key_tangents(translation_nodes[0].FCurve, key)
                key = translation_nodes[1].FCurve.KeyAdd(time, float(row[1]))
                self._apply_key_tangents(translation_nodes[1].FCurve, key)
                key = translation_nodes[2].FCurve.KeyAdd(time, float(row[2]))
                self._apply_key_tangents(translation_nodes[2].FCurve, key)

            if rotation_nodes is not None:
                key = rotation_nodes[0].FCurve.KeyAdd(time, float(row[3]))
                self._apply_key_tangents(rotation_nodes[0].FCurve, key)
                key = rotation_nodes[1].FCurve.KeyAdd(time, float(row[4]))
                self._apply_key_tangents(rotation_nodes[1].FCurve, key)
                key = rotation_nodes[2].FCurve.KeyAdd(time, float(row[5]))
                self._apply_key_tangents(rotation_nodes[2].FCurve, key)

    def get_root_trajectory(self, root_name: str = "Hips") -> np.ndarray:
        """
        Extract the root bone's trajectory over time.
        
        Args:
            root_name: Name of the root bone (default: "Hips")
            
        Returns:
            Array of shape (num_frames, 6) with [X, Y, Z, RotX, RotY, RotZ]
        """
        # Try to find the root bone with fuzzy matching
        model = self.find_root_bone(root_name)
        if model is None:
            raise ValueError(
                f"Root bone not found. Searched for '{root_name}' and common patterns: "
                f"{', '.join(self.ROOT_BONE_PATTERNS[:5])}..."
            )
        
        label = getattr(model, "LabelName", "")
        name = getattr(model, "Name", "")
        display_name = label or name
        if display_name and display_name != root_name:
            print(f"[SeamlessLoopTool] Note: Using '{display_name}' as root bone")
        
        start, end = self.get_frame_range()
        num_frames = end - start + 1
        trajectory = np.zeros((num_frames, 6))
        
        for i, frame in enumerate(range(start, end + 1)):
            time = FBTime(0, 0, 0, frame)
            
            # Get translation
            translation = model.Translation.GetAnimationNode()
            if translation:
                trajectory[i, 0] = translation.Nodes[0].FCurve.Evaluate(time)  # X
                trajectory[i, 1] = translation.Nodes[1].FCurve.Evaluate(time)  # Y
                trajectory[i, 2] = translation.Nodes[2].FCurve.Evaluate(time)  # Z
            
            # Get rotation
            rotation = model.Rotation.GetAnimationNode()
            if rotation:
                trajectory[i, 3] = rotation.Nodes[0].FCurve.Evaluate(time)  # RotX
                trajectory[i, 4] = rotation.Nodes[1].FCurve.Evaluate(time)  # RotY
                trajectory[i, 5] = rotation.Nodes[2].FCurve.Evaluate(time)  # RotZ
        
        return trajectory

    def get_all_bone_poses(self, frame: int) -> Dict[str, np.ndarray]:
        """
        Get the pose of all bones at a specific frame.
        
        Args:
            frame: Frame number to sample
            
        Returns:
            Dictionary mapping bone names to their 4x4 transform matrices
        """
        time = FBTime(0, 0, 0, frame)
        poses = {}
        
        # Recursively collect all models in the scene
        def collect_all_models(parent, result):
            children = getattr(parent, "Children", None)
            if children is None:
                return
            for child in children:
                result.append(child)
                collect_all_models(child, result)
        
        all_models = []
        collect_all_models(FBSystem().Scene.RootModel, all_models)
        
        for model in all_models:
            if model.GetAnimationNode():
                if FBMatrix is not None:
                    fb_matrix = FBMatrix()
                    model.GetMatrix(
                        fb_matrix,
                        FBModelTransformationType.kModelTransformation_Transformation,
                        True,
                        time
                    )
                    matrix = np.array(
                        [[fb_matrix[r][c] for c in range(4)] for r in range(4)],
                        dtype=float
                    )
                else:
                    matrix = np.eye(4)
                poses[model.Name] = matrix
        
        return poses

    def _ensure_animation_nodes(self, model) -> Tuple[Optional[object], Optional[object]]:
        translation_node = None
        rotation_node = None
        if model is None:
            return translation_node, rotation_node

        translation = getattr(model, "Translation", None)
        if translation is not None:
            try:
                translation.SetAnimated(True)
            except Exception:
                pass
            try:
                translation_node = translation.GetAnimationNode()
            except Exception:
                translation_node = None

        rotation = getattr(model, "Rotation", None)
        if rotation is not None:
            try:
                rotation.SetAnimated(True)
            except Exception:
                pass
            try:
                rotation_node = rotation.GetAnimationNode()
            except Exception:
                rotation_node = None

        return translation_node, rotation_node

    def _get_current_layer(self, take):
        get_current = getattr(take, "GetCurrentLayer", None)
        if callable(get_current):
            try:
                return get_current()
            except Exception:
                return None
        return getattr(take, "CurrentLayer", None)

    def _set_current_layer(self, take, layer) -> None:
        if take is None or layer is None:
            return
        setter = getattr(take, "SetCurrentLayer", None)
        if callable(setter):
            try:
                setter(layer)
                return
            except Exception:
                pass
        try:
            take.CurrentLayer = layer
        except Exception:
            pass

    def _iter_take_layers(self, take):
        get_count = getattr(take, "GetLayerCount", None)
        get_layer = getattr(take, "GetLayer", None)
        if callable(get_count) and callable(get_layer):
            try:
                count = int(get_count())
            except Exception:
                count = 0
            if count > 0:
                for idx in range(count):
                    try:
                        layer = get_layer(idx)
                    except Exception:
                        layer = None
                    if layer is not None:
                        yield layer
                return
        yield None

    def _apply_key_tangents(self, fcurve, key_index: Optional[int]) -> None:
        if fcurve is None:
            return

        if key_index is None:
            get_count = getattr(fcurve, "KeyGetCount", None)
            if callable(get_count):
                try:
                    count = get_count()
                    if count > 0:
                        key_index = count - 1
                except Exception:
                    key_index = None

        if key_index is None:
            return

        if FBInterpolation is not None and hasattr(fcurve, "KeySetInterpolation"):
            mode = getattr(FBInterpolation, "kFBInterpolationCubic", None)
            try:
                if mode is not None:
                    fcurve.KeySetInterpolation(int(key_index), mode)
            except Exception:
                pass

        if FBTangentMode is not None and hasattr(fcurve, "KeySetTangentMode"):
            mode = getattr(FBTangentMode, "kFBTangentModeClampProgressive", None)
            try:
                if mode is not None:
                    fcurve.KeySetTangentMode(int(key_index), mode)
            except TypeError:
                try:
                    if mode is not None:
                        fcurve.KeySetTangentMode(
                            int(key_index),
                            mode,
                            mode
                        )
                except Exception:
                    pass
            except Exception:
                pass

    def _clear_fcurve_with_edit_clear(self, fcurve) -> bool:
        """
        Clear all keys using FCurve.EditClear().
        
        This is the recommended MotionBuilder approach - clears all keys
        from negative infinity to positive infinity instantly.
        
        Args:
            fcurve: The FCurve to clear
            
        Returns:
            True if EditClear succeeded, False otherwise
        """
        if fcurve is None:
            return False
        
        edit_clear = getattr(fcurve, "EditClear", None)
        if callable(edit_clear):
            try:
                edit_clear()
                return True
            except Exception:
                pass
        return False

    def _clear_keys_all_layers(
        self,
        model,
        start_frame: int,
        end_frame: int,
        has_rotation: bool
    ) -> None:
        """
        Clear all animation keys for a model across all layers.
        
        Uses FCurve.EditClear() for instant clearing, with fallback to
        key-by-key deletion if EditClear is not available.
        """
        take = getattr(self._system, "CurrentTake", None)
        if take is None:
            return

        original_layer = self._get_current_layer(take)

        for layer in self._iter_take_layers(take):
            if layer is not None:
                self._set_current_layer(take, layer)

            translation_node, rotation_node = self._ensure_animation_nodes(model)

            if translation_node is not None and hasattr(translation_node, "Nodes"):
                if len(translation_node.Nodes) >= 3:
                    for i in range(3):
                        fcurve = translation_node.Nodes[i].FCurve
                        # Try EditClear first, fallback to range-based deletion
                        if not self._clear_fcurve_with_edit_clear(fcurve):
                            self._clear_fcurve_keys_in_range(fcurve, start_frame, end_frame)

            if has_rotation and rotation_node is not None and hasattr(rotation_node, "Nodes"):
                if len(rotation_node.Nodes) >= 3:
                    for i in range(3):
                        fcurve = rotation_node.Nodes[i].FCurve
                        # Try EditClear first, fallback to range-based deletion
                        if not self._clear_fcurve_with_edit_clear(fcurve):
                            self._clear_fcurve_keys_in_range(fcurve, start_frame, end_frame)

        if original_layer is not None:
            self._set_current_layer(take, original_layer)

    def _clear_fcurve_keys_in_range(self, fcurve, start_frame: int, end_frame: int) -> None:
        if fcurve is None:
            return

        get_count = getattr(fcurve, "KeyGetCount", None)
        get_time = getattr(fcurve, "KeyGetTime", None)
        remover = getattr(fcurve, "KeyRemove", None) or getattr(fcurve, "KeyDelete", None)
        if not (callable(get_count) and callable(get_time) and callable(remover)):
            return

        for idx in range(get_count() - 1, -1, -1):
            key_time = get_time(idx)
            if key_time is None:
                continue
            try:
                frame = key_time.GetFrame()
            except Exception:
                continue
            if start_frame <= frame <= end_frame:
                remover(idx)

    def _set_take_time_span(self, start_frame: int, end_frame: int) -> None:
        if FBTime is None:
            return

        start_time = FBTime(0, 0, 0, int(start_frame))
        end_time = FBTime(0, 0, 0, int(end_frame))

        take = getattr(self._system, "CurrentTake", None)
        if take is not None:
            if FBTimeSpan is not None:
                try:
                    take.LocalTimeSpan = FBTimeSpan(start_time, end_time)
                except Exception:
                    pass
            time_span = getattr(take, "LocalTimeSpan", None)
            if time_span is not None:
                try:
                    time_span.SetStart(start_time)
                    time_span.SetStop(end_time)
                except Exception:
                    try:
                        time_span.Set(start_time, end_time)
                    except Exception:
                        pass

        if self._player is not None:
            try:
                self._player.LoopStart = start_time
                self._player.LoopStop = end_time
            except Exception:
                try:
                    self._player.LoopStart.Set(start_time)
                    self._player.LoopStop.Set(end_time)
                except Exception:
                    pass

    def set_root_trajectory(
        self,
        root_name: str,
        trajectory: np.ndarray,
        start_frame: Optional[int] = None
    ) -> None:
        """
        Write a modified trajectory back to the root bone.
        
        Args:
            root_name: Name of the root bone
            trajectory: Array of shape (num_frames, 3 or 6) with [X, Y, Z, RotX, RotY, RotZ]
            start_frame: Optional start frame (ignored; keys are written from frame 0)
        """
        root_name = (root_name or "").strip()
        model = FBFindModelByLabelName(root_name)
        if model is None and FBFindModelByName is not None:
            model = FBFindModelByName(root_name)
        if model is None:
            raise ValueError(f"Bone '{root_name}' not found in scene")

        start_frame = 0
        end_frame = int(len(trajectory)) - 1

        has_rotation = trajectory.shape[1] >= 6

        self._clear_keys_all_layers(model, start_frame, end_frame, True)

        translation_node, rotation_node = self._ensure_animation_nodes(model)
        translation_nodes = None
        rotation_nodes = None

        if translation_node is not None and hasattr(translation_node, "Nodes"):
            if len(translation_node.Nodes) >= 3:
                translation_nodes = translation_node.Nodes

        if has_rotation and rotation_node is not None and hasattr(rotation_node, "Nodes"):
            if len(rotation_node.Nodes) >= 3:
                rotation_nodes = rotation_node.Nodes

        for i, row in enumerate(trajectory):
            time = FBTime(0, 0, 0, int(i))

            if translation_nodes is not None:
                key = translation_nodes[0].FCurve.KeyAdd(time, float(row[0]))  # X
                self._apply_key_tangents(translation_nodes[0].FCurve, key)
                key = translation_nodes[1].FCurve.KeyAdd(time, float(row[1]))  # Y
                self._apply_key_tangents(translation_nodes[1].FCurve, key)
                key = translation_nodes[2].FCurve.KeyAdd(time, float(row[2]))  # Z
                self._apply_key_tangents(translation_nodes[2].FCurve, key)

            if rotation_nodes is not None:
                key = rotation_nodes[0].FCurve.KeyAdd(time, float(row[3]))  # RotX
                self._apply_key_tangents(rotation_nodes[0].FCurve, key)
                key = rotation_nodes[1].FCurve.KeyAdd(time, float(row[4]))  # RotY
                self._apply_key_tangents(rotation_nodes[1].FCurve, key)
                key = rotation_nodes[2].FCurve.KeyAdd(time, float(row[5]))  # RotZ
                self._apply_key_tangents(rotation_nodes[2].FCurve, key)

        self._set_take_time_span(start_frame, end_frame)
        self._clear_keys_all_layers(model, end_frame + 1, 10**9, True)


class MockMoBuAdapter:
    """
    Mock adapter for testing outside MotionBuilder.
    
    Provides the same interface as MoBuAdapter but uses in-memory data.
    """

    def __init__(self, mock_trajectory: np.ndarray = None, mock_frame_range: Tuple[int, int] = (0, 100)):
        """
        Initialize with mock data.
        
        Args:
            mock_trajectory: Pre-defined trajectory data
            mock_frame_range: Frame range to return
        """
        self._frame_range = mock_frame_range
        self._trajectory = mock_trajectory if mock_trajectory is not None else np.zeros((101, 6))
        self._poses = {}
        self._current_take_name = "Take1"
        self._last_written_frames = []
        self._animation_cleared = False
        self._mock_frames = {}
        self._world_positions = {}

    @property
    def frame_range(self) -> Tuple[int, int]:
        return self._frame_range

    @property
    def last_written_frames(self) -> list:
        return self._last_written_frames

    @property
    def animation_cleared(self) -> bool:
        return self._animation_cleared

    def get_frame_range(self) -> Tuple[int, int]:
        """Return mock frame range."""
        return self._frame_range

    def get_current_take_name(self) -> str:
        return self._current_take_name

    def get_root_trajectory(self, root_name: str = "Hips") -> np.ndarray:
        """Return mock trajectory data."""
        return self._trajectory.copy()

    def get_all_bone_poses(self, frame: int) -> Dict[str, np.ndarray]:
        """Return mock pose data."""
        if frame in self._poses:
            return self._poses[frame]
        return {"Hips": np.eye(4)}

    def clear_all_animation(self, root_name: str = "Hips") -> None:
        self._animation_cleared = True

    def create_clean_take(self, suffix: str = "_InPlace", root_name: str = "Hips") -> str:
        self._current_take_name = f"{self._current_take_name}{suffix}"
        self.clear_all_animation(root_name)
        return self._current_take_name

    def create_take_copy(self, suffix: str = "_InPlace", root_name: str = "Hips") -> str:
        return self.create_clean_take(suffix=suffix, root_name=root_name)

    def set_root_trajectory(
        self,
        root_name: str,
        trajectory: np.ndarray,
        start_frame: Optional[int] = None
    ) -> None:
        """Store trajectory in memory."""
        start_frame = 0
        self._trajectory = trajectory.copy()
        self._last_written_frames = list(range(start_frame, start_frame + len(trajectory)))
        self._frame_range = (start_frame, start_frame + len(trajectory) - 1)

    def set_mock_poses(self, frame: int, poses: Dict[str, np.ndarray]) -> None:
        """Set mock pose data for testing."""
        self._poses[frame] = poses

    # =========================================================================
    # Hierarchy-aware methods for testing
    # =========================================================================
    
    def get_hierarchy_nodes(self, root_name: str = "Hips") -> list:
        """Return a mock hierarchy list."""
        # Return a simple mock hierarchy for testing
        if not hasattr(self, '_mock_hierarchy'):
            self._mock_hierarchy = [root_name, f"{root_name}_Child1", f"{root_name}_Child2"]
        return self._mock_hierarchy
    
    def set_mock_hierarchy(self, hierarchy: list) -> None:
        """Set the mock hierarchy for testing."""
        self._mock_hierarchy = hierarchy
        # Initialize bone data storage
        if not hasattr(self, '_bone_data'):
            self._bone_data = {}
    
    def get_node_trajectory(
        self, 
        node_name: str, 
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None
    ) -> np.ndarray:
        """Return trajectory for a specific node."""
        if hasattr(self, '_bone_data') and node_name in self._bone_data:
            return self._bone_data[node_name].copy()
        # Default: return the main trajectory
        return self._trajectory.copy()
    
    def set_node_trajectory(
        self,
        node_name: str,
        trajectory: np.ndarray,
        start_frame: int = 0
    ) -> None:
        """Store trajectory for a specific node."""
        if not hasattr(self, '_bone_data'):
            self._bone_data = {}
        self._bone_data[node_name] = trajectory.copy()
        # Update frame range based on what was written
        self._last_written_frames = list(range(start_frame, start_frame + len(trajectory)))
        self._frame_range = (start_frame, start_frame + len(trajectory) - 1)
    
    def set_mock_bone_trajectory(self, bone_name: str, trajectory: np.ndarray) -> None:
        """Set mock trajectory for a specific bone (for testing input)."""
        if not hasattr(self, '_bone_data'):
            self._bone_data = {}
        self._bone_data[bone_name] = trajectory.copy()

    def set_mock_trajectory(
        self,
        take_name: str,
        node_name: str,
        frames: list,
        translations: list,
        rotations: list,
    ) -> None:
        """Set mock trajectory data using explicit frames."""
        if not hasattr(self, "_bone_data"):
            self._bone_data = {}

        num_frames = len(frames)
        trajectory = np.zeros((num_frames, 6))

        for i, translation in enumerate(translations):
            trajectory[i, 0] = translation[0]
            trajectory[i, 1] = translation[1]
            trajectory[i, 2] = translation[2]

        if rotations:
            for i, rotation in enumerate(rotations):
                trajectory[i, 3] = rotation[0]
                trajectory[i, 4] = rotation[1]
                trajectory[i, 5] = rotation[2]

        self._bone_data[node_name] = trajectory.copy()
        self._mock_frames[(take_name, node_name)] = list(frames)
        if frames:
            self._frame_range = (frames[0], frames[-1])

    def get_mock_trajectory(self, take_name: str, node_name: str) -> tuple:
        """Return mock trajectory data for tests."""
        frames = self._mock_frames.get((take_name, node_name))
        trajectory = None
        if hasattr(self, "_bone_data"):
            trajectory = self._bone_data.get(node_name)

        if trajectory is None:
            trajectory = self._trajectory.copy()

        if frames is None:
            frames = list(range(len(trajectory)))

        translations = [(row[0], row[1], row[2]) for row in trajectory]
        rotations = [(row[3], row[4], row[5]) for row in trajectory]
        return frames, translations, rotations

    def set_mock_world_positions(self, take_name: str, node_name: str, positions: list) -> None:
        """Set mock world-space positions for a node."""
        self._world_positions[(take_name, node_name)] = list(positions)
        self._world_positions[node_name] = list(positions)

    def get_world_translations(
        self,
        node_name: str,
        start_frame: Optional[int] = None,
        end_frame: Optional[int] = None,
    ) -> np.ndarray:
        """Return mock world-space translations for a node."""
        positions = self._world_positions.get(node_name)
        if positions is None:
            positions = []
            if hasattr(self, "_bone_data") and node_name in self._bone_data:
                for row in self._bone_data[node_name]:
                    positions.append((row[0], row[1], row[2]))
            else:
                for row in self._trajectory:
                    positions.append((row[0], row[1], row[2]))

        if start_frame is not None or end_frame is not None:
            start_frame = start_frame or 0
            end_frame = end_frame if end_frame is not None else len(positions) - 1
            positions = positions[start_frame : end_frame + 1]

        return np.array(positions, dtype=float)
