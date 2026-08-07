import importlib
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from core.motion_classifier import ClassificationResult


class FakeCharacter:
    def __init__(self, name="Hero"):
        self.LongName = name


class FakeAdapter:
    def __init__(self):
        self.character = FakeCharacter()
        self.take_name = "WalkTake"
        self.frame_range = (10, 100)

    def get_current_character(self):
        return self.character

    def get_current_take_name(self):
        return self.take_name

    def get_frame_range(self):
        return self.frame_range


class FakeClassifier:
    def __init__(self, result):
        self.result = result
        self.calls = []

    def classify_character(self, character, frame_range=None):
        self.calls.append((character, frame_range))
        return self.result


def load_routing_module():
    return importlib.import_module("ui.motion_routing")


def make_result(label, *, windows=3, reason=None):
    probabilities = {
        "walk": 0.82 if label == "walk" else 0.08,
        "run": 0.82 if label == "run" else 0.10,
        "other": 0.82 if label == "other" else 0.08,
    }
    return ClassificationResult(
        label=label,
        confidence=probabilities[label],
        probabilities=probabilities,
        window_count=windows,
        reason=reason or f"{label}_threshold_passed",
    )


def test_walk_decision_uses_current_character_and_take_range():
    routing = load_routing_module()
    adapter = FakeAdapter()
    classifier = FakeClassifier(make_result("walk"))

    decision = routing.MotionRoutingController(classifier, adapter).classify_current_take()

    assert classifier.calls == [(adapter.character, (10, 100))]
    assert decision.context == routing.MotionContext("Hero", "WalkTake", (10, 100))
    assert decision.display_label == "Walk"
    assert decision.diagnostic == ""
    assert decision.is_failure is False
    assert decision.requires_confirmation is False


def test_valid_other_requires_confirmation_and_has_readable_diagnostic():
    routing = load_routing_module()
    result = make_result("other", reason="specialized_threshold_not_met")

    decision = routing.MotionRoutingController(
        FakeClassifier(result), FakeAdapter()
    ).classify_current_take()

    assert decision.display_label == "Other"
    assert decision.diagnostic == "No confident walk/run match."
    assert decision.is_failure is False
    assert decision.requires_confirmation is True


def test_safe_fallback_is_displayed_as_failure_instead_of_other_100_percent():
    routing = load_routing_module()
    result = ClassificationResult(
        label="other",
        confidence=1.0,
        probabilities={"walk": 0.0, "run": 0.0, "other": 1.0},
        window_count=0,
        reason="missing_required_joint:Head",
    )

    decision = routing.MotionRoutingController(
        FakeClassifier(result), FakeAdapter()
    ).classify_current_take()

    assert decision.display_label == "Unable to classify"
    assert decision.diagnostic == "Required characterized joint is missing: Head."
    assert decision.is_failure is True
    assert decision.requires_confirmation is True


def test_context_becomes_stale_when_take_or_frame_range_changes():
    routing = load_routing_module()
    adapter = FakeAdapter()
    controller = routing.MotionRoutingController(
        FakeClassifier(make_result("run")), adapter
    )
    context = controller.classify_current_take().context

    assert controller.context_is_current(context) is True

    adapter.take_name = "RunTake"
    assert controller.context_is_current(context) is False

    adapter.take_name = "WalkTake"
    adapter.frame_range = (20, 110)
    assert controller.context_is_current(context) is False


def test_default_model_path_is_repository_relative(monkeypatch, tmp_path):
    routing = load_routing_module()
    monkeypatch.chdir(tmp_path)

    model_path = routing.default_motion_model_path()

    expected = Path(routing.__file__).resolve().parents[2] / "models" / "motion_router_v1.json"
    assert model_path == expected
    assert model_path.is_file()


def test_probability_summary_hides_safe_fallback_probabilities():
    routing = load_routing_module()
    adapter = FakeAdapter()
    failure = ClassificationResult(
        label="other",
        confidence=1.0,
        probabilities={"walk": 0.0, "run": 0.0, "other": 1.0},
        window_count=0,
        reason="model_unavailable",
    )
    decision = routing.MotionRoutingController(
        FakeClassifier(failure), adapter
    ).classify_current_take()

    assert routing.format_probability_summary(decision) == (
        "Walk -- | Run -- | Other -- | Windows 0"
    )


def test_confirmation_message_distinguishes_other_from_failure():
    routing = load_routing_module()
    adapter = FakeAdapter()
    other = routing.MotionRoutingController(
        FakeClassifier(make_result("other", reason="specialized_threshold_not_met")),
        adapter,
    ).classify_current_take()
    failure = routing.MotionRoutingController(
        FakeClassifier(
            ClassificationResult(
                label="other",
                confidence=1.0,
                probabilities={"walk": 0.0, "run": 0.0, "other": 1.0},
                window_count=0,
                reason="character_not_characterized",
            )
        ),
        adapter,
    ).classify_current_take()

    assert "Other motion detected (82.0% confidence)." in routing.confirmation_message(other)
    assert "Motion classification is unavailable." in routing.confirmation_message(failure)
    assert "Select and characterize" in routing.confirmation_message(failure)
