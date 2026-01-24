"""
Helpers for export FPS choices (non-Qt).
"""

from typing import List

EXPORT_FPS_CHOICES = (30, 60, 90, 120)
DEFAULT_EXPORT_FPS = 30


def get_export_fps_choices() -> List[int]:
    return list(EXPORT_FPS_CHOICES)


def get_default_export_fps() -> int:
    return DEFAULT_EXPORT_FPS
