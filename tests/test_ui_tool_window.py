from pathlib import Path
import sys
from types import SimpleNamespace

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from ui.tool_window import SeamlessLoopToolWindow


def make_window(decision, *, confirmation=True):
    window = SeamlessLoopToolWindow.__new__(SeamlessLoopToolWindow)
    events = []
    window._check_service = lambda: True
    window._check_motion_router = lambda: True
    window._reset_analysis_state = lambda *args, **kwargs: events.append("reset")
    window._get_params = lambda: events.append("params")
    window._classify_current_take = lambda: decision
    window._display_motion_decision = lambda value: events.append(
        ("display", value)
    )
    window._confirm_analyze_anyway = lambda value: confirmation
    window._find_loop_cycle = lambda context: events.append(("find", context))
    window._set_status = lambda message: events.append(("status", message))
    return window, events


def test_analyze_routes_walk_directly_to_cycle_detection():
    decision = SimpleNamespace(
        requires_confirmation=False,
        result=SimpleNamespace(label="walk"),
        context="walk-context",
    )
    window, events = make_window(decision)

    window._on_analyze_clicked()

    assert events[:3] == ["reset", "params", ("display", decision)]
    assert ("find", "walk-context") in events


def test_analyze_stops_when_other_override_is_cancelled():
    decision = SimpleNamespace(
        requires_confirmation=True,
        result=SimpleNamespace(label="other"),
        context="other-context",
    )
    window, events = make_window(decision, confirmation=False)

    window._on_analyze_clicked()

    assert not any(event[0] == "find" for event in events if isinstance(event, tuple))
    assert events[-1] == (
        "status",
        "Loop analysis cancelled by motion classification gate.",
    )


def test_analyze_uses_existing_detector_after_other_override():
    decision = SimpleNamespace(
        requires_confirmation=True,
        result=SimpleNamespace(label="other"),
        context="other-context",
    )
    window, events = make_window(decision, confirmation=True)

    window._on_analyze_clicked()

    assert ("status", "Motion route: override. Analyzing loop...") in events
    assert ("find", "other-context") in events


def test_process_stops_when_analysis_context_is_stale():
    window = SeamlessLoopToolWindow.__new__(SeamlessLoopToolWindow)
    window.start_frame = 10
    window.end_frame = 50
    window._check_service = lambda: True
    window._get_params = lambda: None
    window._validate_analysis_context = lambda: False
    window._set_status = lambda message: None
    window.service = SimpleNamespace(
        create_seamless_loop_hierarchy=lambda **kwargs: (_ for _ in ()).throw(
            AssertionError("stale analysis must not be processed")
        )
    )

    window._on_process_clicked()


def test_apply_stops_when_analysis_context_is_stale():
    window = SeamlessLoopToolWindow.__new__(SeamlessLoopToolWindow)
    window.processed = True
    window._validate_analysis_context = lambda: False
    window._get_params = lambda: (_ for _ in ()).throw(
        AssertionError("stale analysis must not be applied")
    )
    window._set_status = lambda message: None

    window._on_apply_clicked()


class FakeEdit:
    def __init__(self, value=""):
        self.value = value

    def text(self):
        return self.value

    def setText(self, value):
        self.value = value


def make_namespace_window(selected="mixamorig:LeftLeg"):
    window = SeamlessLoopToolWindow.__new__(SeamlessLoopToolWindow)
    window.edit_namespace = FakeEdit("")
    window.edit_root = FakeEdit("Hips")
    window.edit_left_foot = FakeEdit("LeftFoot")
    window.edit_right_foot = FakeEdit("RightFoot")
    window.edit_left_toe = FakeEdit("LeftToeBase")
    window.edit_right_toe = FakeEdit("RightToeBase")
    window.reset_calls = []
    window.status_messages = []
    window._reset_analysis_state = lambda clear_motion=False: window.reset_calls.append(
        clear_motion
    )
    window._set_status = lambda message: window.status_messages.append(message)
    window._check_service = lambda: True
    window.service = SimpleNamespace(
        adapter=SimpleNamespace(get_selected_model_name=lambda: selected)
    )
    return window


def test_apply_namespace_updates_all_bone_fields_and_invalidates_state():
    window = make_namespace_window()

    window._apply_bone_namespace("hero")

    assert window.edit_namespace.text() == "hero:"
    assert window.edit_root.text() == "hero:Hips"
    assert window.edit_left_foot.text() == "hero:LeftFoot"
    assert window.edit_right_foot.text() == "hero:RightFoot"
    assert window.edit_left_toe.text() == "hero:LeftToeBase"
    assert window.edit_right_toe.text() == "hero:RightToeBase"
    assert window.reset_calls == [True]


def test_manual_namespace_edit_normalizes_and_applies_value():
    window = make_namespace_window()
    window.edit_namespace.setText(" studio:hero ")

    window._on_namespace_edit_finished()

    assert window.edit_namespace.text() == "studio:hero:"
    assert window.edit_root.text() == "studio:hero:Hips"


def test_capture_namespace_uses_selected_complete_name():
    window = make_namespace_window(selected="studio:hero:LeftLeg")

    window._on_get_namespace_clicked()

    assert window.edit_namespace.text() == "studio:hero:"
    assert window.edit_root.text() == "studio:hero:Hips"
    assert window.status_messages[-1] == "Namespace set to: studio:hero:"


def test_capture_unqualified_name_clears_existing_namespace():
    window = make_namespace_window(selected="LeftLeg")
    window._apply_bone_namespace("mixamorig")
    window.reset_calls.clear()

    window._on_get_namespace_clicked()

    assert window.edit_namespace.text() == ""
    assert window.edit_root.text() == "Hips"
    assert window.edit_left_foot.text() == "LeftFoot"
    assert window.reset_calls == [True]


def test_capture_without_selection_leaves_fields_unchanged():
    window = make_namespace_window(selected=None)
    before = window.edit_root.text()

    window._on_get_namespace_clicked()

    assert window.edit_root.text() == before
    assert window.reset_calls == []
    assert window.status_messages[-1] == "No model selected in Navigator"


def test_reapplying_same_namespace_keeps_analysis_state():
    window = make_namespace_window()
    window._apply_bone_namespace("mixamorig")
    window.reset_calls.clear()

    window._apply_bone_namespace("mixamorig:")

    assert window.reset_calls == []
