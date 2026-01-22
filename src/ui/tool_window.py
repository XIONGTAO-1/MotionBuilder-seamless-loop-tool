"""
Seamless Loop Tool Window for MotionBuilder.

Uses PySide2 (Qt) for reliable UI display.
"""

import logging

logger = logging.getLogger(__name__)

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


class SeamlessLoopToolWindow(QtWidgets.QWidget):
    """
    Qt-based Tool Window for creating seamless animation loops.
    
    Workflow:
    1. Click "Analyze" to find best loop frame
    2. Adjust parameters if needed
    3. Click "Process" to create seamless loop
    4. Click "Apply" to write changes back
    """

    WINDOW_TITLE = "Seamless Loop Tool v1.8"
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.service = None
        
        # State
        self.root_name = "Hips"
        self.blend_frames = 5
        self.loop_frame = None
        self.processed = False
        
        self._setup_ui()
        self._init_service()
    
    def _setup_ui(self):
        """Create the UI layout."""
        self.setWindowTitle(self.WINDOW_TITLE)
        self.setMinimumSize(350, 280)
        self.setWindowFlags(QtCore.Qt.Window | QtCore.Qt.WindowStaysOnTopHint)
        
        layout = QtWidgets.QVBoxLayout(self)
        layout.setSpacing(10)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Title
        title = QLabel(self.WINDOW_TITLE)
        title.setStyleSheet("font-size: 14px; font-weight: bold;")
        layout.addWidget(title)
        
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

        # Advanced Settings
        advanced_group = QGroupBox("Advanced Settings")
        advanced_layout = QFormLayout(advanced_group)

        self.spin_min_cycle_frames = QSpinBox()
        self.spin_min_cycle_frames.setRange(5, 100)
        self.spin_min_cycle_frames.setValue(20)

        self.spin_max_cycle_frames = QSpinBox()
        self.spin_max_cycle_frames.setRange(10, 200)
        self.spin_max_cycle_frames.setValue(60)

        self.spin_min_avg_velocity = QDoubleSpinBox()
        self.spin_min_avg_velocity.setRange(0.0, 1000.0)
        self.spin_min_avg_velocity.setSingleStep(1.0)
        self.spin_min_avg_velocity.setValue(5.0)

        self.spin_min_vertical_bounce = QDoubleSpinBox()
        self.spin_min_vertical_bounce.setRange(0.0, 100.0)
        self.spin_min_vertical_bounce.setSingleStep(0.1)
        self.spin_min_vertical_bounce.setValue(0.0)

        advanced_layout.addRow(QLabel("Min Cycle Frames"), self.spin_min_cycle_frames)
        advanced_layout.addRow(QLabel("Max Cycle Frames"), self.spin_max_cycle_frames)
        advanced_layout.addRow(QLabel("Min Avg Velocity"), self.spin_min_avg_velocity)
        advanced_layout.addRow(QLabel("Min Vertical Bounce"), self.spin_min_vertical_bounce)
        layout.addWidget(advanced_group)
        
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
        layout.addWidget(self.btn_process)
        
        # Apply Button
        self.btn_apply = QtWidgets.QPushButton("3. Apply Changes to Scene")
        self.btn_apply.clicked.connect(self._on_apply_clicked)
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
        try:
            from mobu.adapter import MoBuAdapter, IN_MOTIONBUILDER
            from mobu.loop_processor import LoopProcessorService
            
            if not IN_MOTIONBUILDER:
                self._set_status("Error: Not in MotionBuilder environment")
                return
            
            adapter = MoBuAdapter()
            self.service = LoopProcessorService(adapter)
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
        self.blend_frames = self.edit_blend.value()
    
    def _check_service(self) -> bool:
        """Check if service is ready."""
        if self.service is None:
            self._set_status("Error: Service not initialized. Restart tool.")
            return False
        return True
    
    def _on_get_selected_clicked(self):
        """Get selected bone from Navigator and fill in Root Bone field."""
        if not self._check_service():
            return
        try:
            selected_name = self.service.adapter.get_selected_model_name()
            if selected_name:
                self.edit_root.setText(selected_name)
                self._set_status(f"Selected: {selected_name}")
            else:
                self._set_status("No model selected in Navigator")
        except Exception as e:
            self._set_status(f"Error: {e}")
    
    def _on_analyze_clicked(self):
        """Handle Analyze button click - uses cycle detection to find mid-animation loop."""
        if not self._check_service():
            return
        self._get_params()
        self._set_status(f"Analyzing... (root: {self.root_name})")
        
        try:
            # Get frame range for debugging
            start, end = self.service.adapter.get_frame_range()
            print(f"[SeamlessLoopTool] Frame range: {start} - {end} ({end - start + 1} frames)")
            
            # Use find_walk_cycle to detect a cycle segment (not starting from frame 0)
            min_cycle_frames = self.spin_min_cycle_frames.value()
            max_cycle_frames = self.spin_max_cycle_frames.value()
            min_avg_velocity = self.spin_min_avg_velocity.value()
            min_vertical_bounce = self.spin_min_vertical_bounce.value()

            self.cycle_start, self.cycle_end = self.service.find_walk_cycle(
                root_name=self.root_name,
                min_cycle_frames=min_cycle_frames,
                max_cycle_frames=max_cycle_frames,
                min_average_velocity=min_avg_velocity,
                min_vertical_bounce=min_vertical_bounce
            )
            self.start_frame = self.cycle_start
            self.end_frame = self.cycle_end
            # For compatibility
            self.loop_frame = self.cycle_end
            
            cycle_length = self.cycle_end - self.cycle_start
            self.lbl_loop_frame.setText(
                f"Cycle: {self.cycle_start} - {self.cycle_end} ({cycle_length} frames)"
            )
            self._set_status(f"Found cycle: frames {self.cycle_start}-{self.cycle_end}")
        except Exception as e:
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
        
        self._set_status("Processing hierarchy...")
        
        try:
            # Use hierarchy-aware processing for the entire skeleton
            if hasattr(self.service, 'create_seamless_loop_hierarchy'):
                processed_data = self.service.create_seamless_loop_hierarchy(
                    root_name=self.root_name,
                    start_frame=self.start_frame,
                    loop_frame=self.end_frame,
                    blend_frames=self.blend_frames,
                    in_place=True
                )
                self.processed = True
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
                    in_place=True
                )
                self.processed = True
                self._set_status(f"Processed: {len(trajectory)} frames (root only)")
        except Exception as e:
            self._set_status(f"Error: {e}")
            import traceback
            traceback.print_exc()
    
    def _on_apply_clicked(self):
        """Handle Apply button click - writes the entire hierarchy."""
        if not self.processed:
            self._set_status("Please Process first!")
            return
        
        self._set_status("Applying changes to hierarchy...")
        
        try:
            # Use hierarchy-aware apply if available
            if hasattr(self.service, 'apply_changes_hierarchy') and hasattr(self.service, 'processed_data') and self.service.processed_data:
                self.service.apply_changes_hierarchy(
                    root_name=self.root_name,
                    preserve_original=self.chk_preserve.isChecked()
                )
                self._set_status("Hierarchy applied to scene!")
            else:
                # Fallback to root-only apply
                self.service.apply_changes(
                    root_name=self.root_name,
                    preserve_original=self.chk_preserve.isChecked()
                )
                self._set_status("Changes applied to scene (root only)!")
        except Exception as e:
            self._set_status(f"Error: {e}")


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
