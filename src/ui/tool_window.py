"""
Seamless Loop Tool Window for MotionBuilder.

Uses PySide2 (Qt) for reliable UI display.
"""

import logging

logger = logging.getLogger(__name__)

from ui.bone_namespace import (
    extract_namespace,
    normalize_namespace,
    qualify_bone_name,
)
from ui.export_fps import get_export_fps_choices, get_default_export_fps

# Try to import PySide2 (available in MotionBuilder 2024)
try:
    from PySide2 import QtWidgets, QtCore
    from PySide2.QtWidgets import QSpinBox, QDoubleSpinBox, QLabel, QFormLayout, QGroupBox
    IN_MOTIONBUILDER = True
except ImportError:
    try:
        from PySide6 import QtWidgets, QtCore
        from PySide6.QtWidgets import QSpinBox, QDoubleSpinBox, QLabel, QFormLayout, QGroupBox
        IN_MOTIONBUILDER = False
    except ImportError:
        QtWidgets = None
        QtCore = None
        QSpinBox = None
        QDoubleSpinBox = None
        QLabel = None
        QFormLayout = None
        QGroupBox = None
        IN_MOTIONBUILDER = False
        logger.warning("PySide2/PySide6 not available")

class _QtFallbackWidget:
    def __init__(self, *args, **kwargs):
        pass


QtBaseWidget = QtWidgets.QWidget if QtWidgets is not None else _QtFallbackWidget


class SeamlessLoopToolWindow(QtBaseWidget):
    """
    Qt-based Tool Window for creating seamless animation loops.
    
    Workflow:
    1. Click "Analyze" to find best loop frame
    2. Adjust parameters if needed
    3. Click "Process" to create seamless loop
    4. Click "Apply" to write changes back
    """

    WINDOW_TITLE = "Seamless Loop Tool v2.1"
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.service = None
        self.motion_router = None
        self.analysis_context = None
        
        # State
        self.root_name = "Hips"
        self.left_foot_name = "LeftFoot"
        self.right_foot_name = "RightFoot"
        self.left_toe_name = "LeftToeBase"
        self.right_toe_name = "RightToeBase"
        self.blend_frames = 5
        self.target_rot_y = 180.0
        self.loop_frame = None
        self.processed = False
        
        self._setup_ui()
        self._init_service()

    def _create_foot_fix_checkbox(self):
        checkbox = QtWidgets.QCheckBox("Enable Foot Contact Fix")
        checkbox.setChecked(False)
        checkbox.setToolTip("Apply ground contact correction to foot bones")
        checkbox.stateChanged.connect(self._on_foot_fix_toggled)
        return checkbox
    
    def _setup_ui(self):
        """Create the UI layout."""
        self.setWindowTitle(self.WINDOW_TITLE)
        self.setMinimumSize(350, 280)
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Root Bone Input
        root_layout = QtWidgets.QHBoxLayout()
        root_layout.addWidget(QLabel("Root Bone:"))
        self.edit_root = QtWidgets.QLineEdit(self.root_name)
        root_layout.addWidget(self.edit_root)
        self.btn_get_selected = QtWidgets.QPushButton("← Get Selected")
        self.btn_get_selected.setToolTip("Get bone name from Navigator selection")
        self.btn_get_selected.clicked.connect(self._on_get_selected_clicked)
        root_layout.addWidget(self.btn_get_selected)
        layout.addLayout(root_layout)

        namespace_layout = QtWidgets.QHBoxLayout()
        namespace_layout.addWidget(QLabel("Namespace:"))
        self.edit_namespace = QtWidgets.QLineEdit("")
        self.edit_namespace.setPlaceholderText("e.g. mixamorig:")
        self.edit_namespace.setToolTip(
            "Enter a namespace and press Enter, or capture it from a selected bone"
        )
        self.edit_namespace.editingFinished.connect(
            self._on_namespace_edit_finished
        )
        namespace_layout.addWidget(self.edit_namespace)
        self.btn_get_namespace = QtWidgets.QPushButton("← Get from Selected")
        self.btn_get_namespace.setToolTip(
            "Extract the namespace from the selected bone's complete name"
        )
        self.btn_get_namespace.clicked.connect(self._on_get_namespace_clicked)
        namespace_layout.addWidget(self.btn_get_namespace)
        layout.addLayout(namespace_layout)

        # Foot/Toe Bone Inputs
        self.edit_left_foot = QtWidgets.QLineEdit(self.left_foot_name)
        self.edit_right_foot = QtWidgets.QLineEdit(self.right_foot_name)
        self.edit_left_toe = QtWidgets.QLineEdit(self.left_toe_name)
        self.edit_right_toe = QtWidgets.QLineEdit(self.right_toe_name)

        left_foot_layout = QtWidgets.QHBoxLayout()
        left_foot_layout.addWidget(QLabel("Left Foot:"))
        left_foot_layout.addWidget(self.edit_left_foot)
        btn_left_foot = QtWidgets.QPushButton("← Get Selected")
        btn_left_foot.clicked.connect(lambda: self._set_selected_to_edit(self.edit_left_foot, "Left Foot"))
        left_foot_layout.addWidget(btn_left_foot)
        layout.addLayout(left_foot_layout)

        right_foot_layout = QtWidgets.QHBoxLayout()
        right_foot_layout.addWidget(QLabel("Right Foot:"))
        right_foot_layout.addWidget(self.edit_right_foot)
        btn_right_foot = QtWidgets.QPushButton("← Get Selected")
        btn_right_foot.clicked.connect(lambda: self._set_selected_to_edit(self.edit_right_foot, "Right Foot"))
        right_foot_layout.addWidget(btn_right_foot)
        layout.addLayout(right_foot_layout)

        left_toe_layout = QtWidgets.QHBoxLayout()
        left_toe_layout.addWidget(QLabel("Left Toe:"))
        left_toe_layout.addWidget(self.edit_left_toe)
        btn_left_toe = QtWidgets.QPushButton("← Get Selected")
        btn_left_toe.clicked.connect(lambda: self._set_selected_to_edit(self.edit_left_toe, "Left Toe"))
        left_toe_layout.addWidget(btn_left_toe)
        layout.addLayout(left_toe_layout)

        right_toe_layout = QtWidgets.QHBoxLayout()
        right_toe_layout.addWidget(QLabel("Right Toe:"))
        right_toe_layout.addWidget(self.edit_right_toe)
        btn_right_toe = QtWidgets.QPushButton("← Get Selected")
        btn_right_toe.clicked.connect(lambda: self._set_selected_to_edit(self.edit_right_toe, "Right Toe"))
        right_toe_layout.addWidget(btn_right_toe)
        layout.addLayout(right_toe_layout)
        
        # Blend Frames Input
        blend_layout = QtWidgets.QHBoxLayout()
        blend_layout.addWidget(QLabel("Blend Frames:"))
        self.edit_blend = QSpinBox()
        self.edit_blend.setRange(1, 30)
        self.edit_blend.setValue(self.blend_frames)
        blend_layout.addWidget(self.edit_blend)
        blend_layout.addStretch()
        layout.addLayout(blend_layout)
        
        # Coordinate System Selection
        coord_layout = QtWidgets.QHBoxLayout()
        coord_layout.addWidget(QLabel("Up Axis:"))
        self.combo_upaxis = QtWidgets.QComboBox()
        self.combo_upaxis.addItems(["Y-Up (Maya/MoBu)", "Z-Up (3ds Max/Mocap)"])
        self.combo_upaxis.setCurrentIndex(0)  # Default Y-Up
        coord_layout.addWidget(self.combo_upaxis)
        coord_layout.addStretch()
        layout.addLayout(coord_layout)
        
        # Preserve Original Checkbox
        self.chk_preserve = QtWidgets.QCheckBox("Create new Take (preserve original)")
        self.chk_preserve.setChecked(True)  # Default to non-destructive
        self.chk_preserve.setToolTip("Creates a copy of the current Take before processing")
        layout.addWidget(self.chk_preserve)

        # Enable Foot Fix Checkbox
        self.chk_enable_foot_fix = self._create_foot_fix_checkbox()
        layout.addWidget(self.chk_enable_foot_fix)

        # Advanced Settings
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QFormLayout(advanced_group)

        self.spin_min_cycle_frames = QSpinBox()
        self.spin_min_cycle_frames.setRange(5, 100)
        self.spin_min_cycle_frames.setValue(20)

        self.spin_max_cycle_frames = QSpinBox()
        self.spin_max_cycle_frames.setRange(10, 200)
        self.spin_max_cycle_frames.setValue(60)

        self.spin_min_vertical_bounce = QDoubleSpinBox()
        self.spin_min_vertical_bounce.setRange(0.0, 100.0)
        self.spin_min_vertical_bounce.setSingleStep(0.1)
        self.spin_min_vertical_bounce.setValue(0.0)

        self.spin_target_rot_y = QDoubleSpinBox()
        self.spin_target_rot_y.setRange(-360.0, 360.0)
        self.spin_target_rot_y.setSingleStep(1.0)
        self.spin_target_rot_y.setValue(self.target_rot_y)

        self.combo_export_fps = QtWidgets.QComboBox()
        export_choices = get_export_fps_choices()
        for fps in export_choices:
            self.combo_export_fps.addItem(str(fps), fps)
        default_export_fps = get_default_export_fps()
        if default_export_fps in export_choices:
            self.combo_export_fps.setCurrentIndex(export_choices.index(default_export_fps))

        advanced_layout.addRow(QLabel("Min Cycle Frames"), self.spin_min_cycle_frames)
        advanced_layout.addRow(QLabel("Max Cycle Frames"), self.spin_max_cycle_frames)
        advanced_layout.addRow(QLabel("Min Vertical Bounce"), self.spin_min_vertical_bounce)
        advanced_layout.addRow(QLabel("Hips RotY Target"), self.spin_target_rot_y)
        advanced_layout.addRow(QLabel("Export FPS"), self.combo_export_fps)
        layout.addWidget(advanced_group)

        motion_group = QGroupBox("Motion Type")
        motion_layout = QtWidgets.QVBoxLayout(motion_group)
        self.lbl_motion_type = QLabel("Not classified")
        self.lbl_motion_type.setStyleSheet("font-weight: bold; color: gray;")
        motion_layout.addWidget(self.lbl_motion_type)
        self.lbl_motion_probabilities = QLabel(
            "Walk -- | Run -- | Other -- | Windows 0"
        )
        motion_layout.addWidget(self.lbl_motion_probabilities)
        self.lbl_motion_diagnostic = QLabel("")
        self.lbl_motion_diagnostic.setWordWrap(True)
        self.lbl_motion_diagnostic.setStyleSheet("color: #b00020;")
        self.lbl_motion_diagnostic.hide()
        motion_layout.addWidget(self.lbl_motion_diagnostic)
        layout.addWidget(motion_group)
        
        # Analyze Button
        self.btn_analyze = QtWidgets.QPushButton("1. Analyze Loop Point")
        self.btn_analyze.clicked.connect(self._on_analyze_clicked)
        layout.addWidget(self.btn_analyze)
        
        # Loop Frame Display
        self.lbl_loop_frame = QLabel("Best Loop Frame: (click Analyze)")
        layout.addWidget(self.lbl_loop_frame)
        
        # Process Button
        self.btn_process = QtWidgets.QPushButton("2. Process (Trim + Blend + In-Place)")
        self.btn_process.clicked.connect(self._on_process_clicked)
        self.btn_process.setEnabled(False)
        layout.addWidget(self.btn_process)
        
        # Apply Button
        self.btn_apply = QtWidgets.QPushButton("3. Apply Changes to Scene")
        self.btn_apply.clicked.connect(self._on_apply_clicked)
        self.btn_apply.setEnabled(False)
        layout.addWidget(self.btn_apply)
        
        layout.addStretch()
        
        # Status Label
        self.lbl_status = QLabel("Ready")
        self.lbl_status.setStyleSheet("color: gray;")
        layout.addWidget(self.lbl_status)
        
        # Author Info
        # Author Info
        lbl_author = QLabel("Author: niexiongtao")
        lbl_author.setStyleSheet("font-size: 10px; color: gray;")
        lbl_author.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(lbl_author)
        
        lbl_contact = QLabel("Contact: niexiongtao@gmail.com")
        lbl_contact.setStyleSheet("font-size: 10px; color: gray;")
        lbl_contact.setAlignment(QtCore.Qt.AlignCenter)
        layout.addWidget(lbl_contact)
    
    def _init_service(self):
        """Initialize the processing service."""
        self.service = None
        self.motion_router = None
        try:
            from mobu.adapter import MoBuAdapter, IN_MOTIONBUILDER
            from mobu.loop_processor import LoopProcessorService
            from mobu.motion_classifier import (
                MotionBuilderCharacterSampler,
                MotionClassifier,
            )
            from ui.motion_routing import (
                MotionRoutingController,
                default_motion_model_path,
            )
            
            if not IN_MOTIONBUILDER:
                self._set_status("Error: Not in MotionBuilder environment")
                return
            
            adapter = MoBuAdapter()
            self.service = LoopProcessorService(adapter)
            classifier = MotionClassifier(
                default_motion_model_path(),
                sampler=MotionBuilderCharacterSampler(adapter=adapter),
            )
            self.motion_router = MotionRoutingController(classifier, adapter)
            self._set_status("Service initialized - Ready")
        except ImportError as e:
            self._set_status(f"Import error: {e}")
            print(f"[SeamlessLoopTool] Import failed: {e}")
        except Exception as e:
            self._set_status(f"Error: {e}")
            print(f"[SeamlessLoopTool] Init failed: {e}")
            import traceback
            traceback.print_exc()
    
    def _set_status(self, msg: str):
        """Update the status label."""
        self.lbl_status.setText(msg)
        logger.info(msg)
    
    def _get_params(self):
        """Read current parameters from UI."""
        self.root_name = self.edit_root.text().strip() or "Hips"
        self.left_foot_name = self.edit_left_foot.text().strip() or "LeftFoot"
        self.right_foot_name = self.edit_right_foot.text().strip() or "RightFoot"
        self.left_toe_name = self.edit_left_toe.text().strip() or "LeftToeBase"
        self.right_toe_name = self.edit_right_toe.text().strip() or "RightToeBase"
        self.blend_frames = self.edit_blend.value()
        self.target_rot_y = self.spin_target_rot_y.value()
        self.enable_foot_fix = self.chk_enable_foot_fix.isChecked()

    def _get_export_fps(self) -> float:
        data = self.combo_export_fps.currentData()
        if data is not None:
            return float(data)
        try:
            return float(self.combo_export_fps.currentText())
        except ValueError:
            return float(get_default_export_fps())
    
    def _check_service(self) -> bool:
        """Check if service is ready."""
        if self.service is None:
            self._set_status("Error: Service not initialized. Restart tool.")
            return False
        return True

    def _check_motion_router(self) -> bool:
        if self.motion_router is None:
            self._set_status("Error: Motion classifier not initialized. Restart tool.")
            return False
        return True

    def _reset_motion_display(self):
        self.lbl_motion_type.setText("Not classified")
        self.lbl_motion_type.setStyleSheet("font-weight: bold; color: gray;")
        self.lbl_motion_probabilities.setText(
            "Walk -- | Run -- | Other -- | Windows 0"
        )
        self.lbl_motion_diagnostic.clear()
        self.lbl_motion_diagnostic.hide()

    def _reset_analysis_state(self, clear_motion=False):
        for name in ("cycle_start", "cycle_end", "start_frame", "end_frame"):
            if hasattr(self, name):
                delattr(self, name)
        self.loop_frame = None
        self.processed = False
        self.analysis_context = None
        self.btn_process.setEnabled(False)
        self.btn_apply.setEnabled(False)
        self.lbl_loop_frame.setText("Best Loop Frame: (click Analyze)")
        if clear_motion:
            self._reset_motion_display()

    def _display_motion_decision(self, decision):
        from ui.motion_routing import format_probability_summary

        colors = {
            "walk": "#2e7d32",
            "run": "#1565c0",
            "other": "#a15c00",
        }
        color = "#b00020" if decision.is_failure else colors[decision.result.label]
        label = decision.display_label
        if not decision.is_failure:
            label = f"{label} — {decision.result.confidence:.1%} confidence"
        self.lbl_motion_type.setText(label)
        self.lbl_motion_type.setStyleSheet(f"font-weight: bold; color: {color};")
        self.lbl_motion_probabilities.setText(format_probability_summary(decision))
        self.lbl_motion_diagnostic.setText(decision.diagnostic)
        self.lbl_motion_diagnostic.setVisible(decision.is_failure)

    def _confirm_analyze_anyway(self, decision) -> bool:
        from ui.motion_routing import confirmation_message

        dialog = QtWidgets.QMessageBox(self)
        dialog.setIcon(QtWidgets.QMessageBox.Warning)
        dialog.setWindowTitle("Motion Classification")
        dialog.setText(confirmation_message(decision))
        cancel_button = dialog.addButton(
            "Cancel", QtWidgets.QMessageBox.RejectRole
        )
        continue_button = dialog.addButton(
            "Analyze Anyway", QtWidgets.QMessageBox.AcceptRole
        )
        dialog.setDefaultButton(cancel_button)
        dialog.setEscapeButton(cancel_button)
        execute = getattr(dialog, "exec", None) or dialog.exec_
        execute()
        return dialog.clickedButton() is continue_button

    def _classify_current_take(self):
        self._set_status("Classifying current take...")
        self.lbl_motion_type.setText("Classifying...")
        self.lbl_motion_type.setStyleSheet("font-weight: bold; color: gray;")
        QtWidgets.QApplication.setOverrideCursor(QtCore.Qt.WaitCursor)
        try:
            QtWidgets.QApplication.processEvents()
            return self.motion_router.classify_current_take()
        finally:
            QtWidgets.QApplication.restoreOverrideCursor()

    def _validate_analysis_context(self) -> bool:
        if self.analysis_context is not None and self.motion_router.context_is_current(
            self.analysis_context
        ):
            return True
        self._reset_analysis_state(clear_motion=True)
        self._set_status("Character, Take, or frame range changed. Analyze again.")
        return False
    
    def _set_selected_to_edit(self, edit_field, label: str):
        """Get selected bone from Navigator and fill in a field."""
        if not self._check_service():
            return
        try:
            selected_name = self.service.adapter.get_selected_model_name()
            if selected_name:
                edit_field.setText(selected_name)
                self._set_status(f"{label} set to: {selected_name}")
            else:
                self._set_status("No model selected in Navigator")
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _on_get_selected_clicked(self):
        """Get selected bone from Navigator and fill in Root Bone field."""
        self._set_selected_to_edit(self.edit_root, "Root Bone")

    def _apply_bone_namespace(self, namespace: str) -> None:
        normalized = normalize_namespace(namespace)
        fields = (
            (self.edit_root, "Hips"),
            (self.edit_left_foot, "LeftFoot"),
            (self.edit_right_foot, "RightFoot"),
            (self.edit_left_toe, "LeftToeBase"),
            (self.edit_right_toe, "RightToeBase"),
        )
        updates = [
            (field, qualify_bone_name(field.text(), normalized, default_name))
            for field, default_name in fields
        ]
        changed = self.edit_namespace.text() != normalized or any(
            field.text() != value for field, value in updates
        )

        self.edit_namespace.setText(normalized)
        for field, value in updates:
            field.setText(value)

        if changed:
            self._reset_analysis_state(clear_motion=True)

    def _on_namespace_edit_finished(self) -> None:
        self._apply_bone_namespace(self.edit_namespace.text())
        namespace = self.edit_namespace.text() or "(none)"
        self._set_status(f"Namespace set to: {namespace}")

    def _on_get_namespace_clicked(self) -> None:
        if not self._check_service():
            return
        try:
            selected_name = self.service.adapter.get_selected_model_name()
            if not selected_name:
                self._set_status("No model selected in Navigator")
                return
            self._apply_bone_namespace(extract_namespace(selected_name))
            namespace = self.edit_namespace.text() or "(none)"
            self._set_status(f"Namespace set to: {namespace}")
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _find_loop_cycle(self, context):
        start, end = self.service.adapter.get_frame_range()
        print(
            f"[SeamlessLoopTool] Frame range: {start} - {end} "
            f"({end - start + 1} frames)"
        )
        self.cycle_start, self.cycle_end = self.service.find_walk_cycle(
            root_name=self.root_name,
            min_cycle_frames=self.spin_min_cycle_frames.value(),
            max_cycle_frames=self.spin_max_cycle_frames.value(),
            min_vertical_bounce=self.spin_min_vertical_bounce.value(),
        )
        self.start_frame = self.cycle_start
        self.end_frame = self.cycle_end
        self.loop_frame = self.cycle_end
        self.analysis_context = context
        cycle_length = self.cycle_end - self.cycle_start
        self.lbl_loop_frame.setText(
            f"Cycle: {self.cycle_start} - {self.cycle_end} ({cycle_length} frames)"
        )
        self.btn_process.setEnabled(True)
        self._set_status(f"Found cycle: frames {self.cycle_start}-{self.cycle_end}")

    def _on_analyze_clicked(self):
        """Classify the current Take, then run gait-cycle detection when allowed."""
        if not self._check_service() or not self._check_motion_router():
            return
        self._reset_analysis_state()
        self._get_params()

        try:
            decision = self._classify_current_take()
            self._display_motion_decision(decision)
            if decision.requires_confirmation and not self._confirm_analyze_anyway(
                decision
            ):
                self._set_status("Loop analysis cancelled by motion classification gate.")
                return
            route = "override" if decision.requires_confirmation else decision.result.label
            self._set_status(f"Motion route: {route}. Analyzing loop...")
            self._find_loop_cycle(decision.context)
        except Exception as e:
            self._reset_analysis_state()
            self._set_status(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_process_clicked(self):
        """Handle Process button click - processes the entire hierarchy."""
        if not self._check_service():
            return
        self._get_params()
        
        # Check if we have analyzed cycle first
        if not hasattr(self, 'start_frame') or not hasattr(self, 'end_frame'):
            self._set_status("Please Analyze first!")
            return
        if not self._validate_analysis_context():
            return
        
        self._set_status("Processing hierarchy...")
        
        try:
            # Use hierarchy-aware processing for the entire skeleton
            if hasattr(self.service, 'create_seamless_loop_hierarchy'):
                processed_data = self.service.create_seamless_loop_hierarchy(
                    root_name=self.root_name,
                    start_frame=self.start_frame,
                    loop_frame=self.end_frame,
                    blend_frames=self.blend_frames,
                    in_place=True,
                    target_rot_y=self.target_rot_y,
                    left_foot=self.left_foot_name,
                    right_foot=self.right_foot_name,
                    left_toe=self.left_toe_name,
                    right_toe=self.right_toe_name,
                    enable_foot_fix=bool(self.enable_foot_fix),
                )
                self.processed = True
                self.btn_apply.setEnabled(True)
                bone_count = len(processed_data)
                frame_count = len(next(iter(processed_data.values()))) if processed_data else 0
                self._set_status(f"Processed: {bone_count} bones, {frame_count} frames")
            else:
                # Fallback to root-only processing
                trajectory = self.service.create_seamless_loop(
                    root_name=self.root_name,
                    start_frame=self.start_frame,
                    loop_frame=self.end_frame,
                    blend_frames=self.blend_frames,
                    in_place=True,
                    target_rot_y=self.target_rot_y,
                )
                self.processed = True
                self.btn_apply.setEnabled(True)
                self._set_status(f"Processed: {len(trajectory)} frames (root only)")
        except Exception as e:
            self.processed = False
            self.btn_apply.setEnabled(False)
            self._set_status(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_apply_clicked(self):
        """Handle Apply button click - writes the entire hierarchy."""
        if not self.processed:
            self._set_status("Please Process first!")
            return
        if not self._validate_analysis_context():
            return
        self._get_params()
        self._set_status("Applying changes to hierarchy...")

        try:
            # Log the live checkbox state to debug cases where UI looks checked but evaluates false.
            try:
                chk_state = int(self.chk_enable_foot_fix.checkState())
            except Exception:
                chk_state = None
            logger.info(
                "UI: Enable Foot Fix isChecked=%s checkState=%s",
                bool(self.chk_enable_foot_fix.isChecked()),
                chk_state,
            )

            # Use hierarchy-aware apply if available
            if hasattr(self.service, 'apply_changes_hierarchy') and hasattr(self.service, 'processed_data') and self.service.processed_data:
                self.service.apply_changes_hierarchy(
                    root_name=self.root_name,
                    preserve_original=self.chk_preserve.isChecked(),
                    target_fps=self._get_export_fps(),
                    left_foot=self.left_foot_name,
                    right_foot=self.right_foot_name,
                    left_toe=self.left_toe_name,
                    right_toe=self.right_toe_name,
                    ground_height=0.0,
                    enable_foot_fix=bool(self.chk_enable_foot_fix.isChecked()),
                )
                self._set_status("Hierarchy applied to scene!")
            else:
                # Fallback to root-only apply
                self.service.apply_changes(
                    root_name=self.root_name,
                    preserve_original=self.chk_preserve.isChecked(),
                    target_fps=self._get_export_fps(),
                )
                self._set_status("Changes applied to scene (root only)!")
        except Exception as e:
            self._set_status(f"Error: {e}")

    def _on_foot_fix_toggled(self, state: int) -> None:
        logger.info("UI: Enable Foot Fix toggled state=%s isChecked=%s", state, bool(self.chk_enable_foot_fix.isChecked()))


# Global reference to keep window alive
_window_instance = None


def create_tool():
    """Factory function to create and show the tool window."""
    global _window_instance
    
    if QtWidgets is None:
        logger.error("Qt not available!")
        return None
    
    # Close existing window if any
    if _window_instance is not None:
        try:
            _window_instance.close()
        except Exception:
            pass
    
    # Create new window
    _window_instance = SeamlessLoopToolWindow()
    _window_instance.show()
    _window_instance.raise_()
    _window_instance.activateWindow()
    
    logger.info("Tool window created and shown!")
    return _window_instance
