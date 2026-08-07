from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ui.bone_namespace import (
    extract_namespace,
    normalize_namespace,
    qualify_bone_name,
)


def test_normalizes_optional_separator_and_nested_namespace():
    assert normalize_namespace("mixamorig") == "mixamorig:"
    assert normalize_namespace(" hero:rig: ") == "hero:rig:"
    assert normalize_namespace("") == ""


def test_extracts_everything_before_final_colon():
    assert extract_namespace("hero:rig:LeftLeg") == "hero:rig:"
    assert extract_namespace("mixamorig:Hips") == "mixamorig:"
    assert extract_namespace("LeftLeg") == ""


def test_replaces_existing_namespace_without_stacking():
    assert (
        qualify_bone_name("mixamorig:LeftFoot", "hero", "LeftFoot")
        == "hero:LeftFoot"
    )
    assert qualify_bone_name("", "hero:", "RightFoot") == "hero:RightFoot"
    assert (
        qualify_bone_name("hero:LeftToeBase", "", "LeftToeBase")
        == "LeftToeBase"
    )


def test_preserves_custom_local_bone_name():
    assert (
        qualify_bone_name("old:ankle_l", "new:rig", "LeftFoot")
        == "new:rig:ankle_l"
    )
