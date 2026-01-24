"""
Tests for export FPS helper.
"""

import os
import sys

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from ui.export_fps import get_export_fps_choices, get_default_export_fps


class TestExportFps:
    def test_export_fps_choices(self):
        assert get_export_fps_choices() == [30, 60, 90, 120]

    def test_default_export_fps_is_in_choices(self):
        default_fps = get_default_export_fps()
        assert default_fps in get_export_fps_choices()
        assert default_fps == 30
