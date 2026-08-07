"""UI-facing motion classification decisions without Qt dependencies."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from core.motion_classifier import ClassificationResult


@dataclass(frozen=True)
class MotionContext:
    character_name: str
    take_name: str
    frame_range: tuple[int, int]


@dataclass(frozen=True)
class MotionRouteDecision:
    result: ClassificationResult
    context: MotionContext
    display_label: str
    diagnostic: str
    is_failure: bool
    requires_confirmation: bool


def default_motion_model_path() -> Path:
    return Path(__file__).resolve().parents[2] / "models" / "motion_router_v1.json"


def format_probability_summary(decision: MotionRouteDecision) -> str:
    if decision.is_failure:
        return "Walk -- | Run -- | Other -- | Windows 0"
    probabilities = decision.result.probabilities
    return (
        f"Walk {probabilities['walk']:.1%} | "
        f"Run {probabilities['run']:.1%} | "
        f"Other {probabilities['other']:.1%} | "
        f"Windows {decision.result.window_count}"
    )


def confirmation_message(decision: MotionRouteDecision) -> str:
    if decision.is_failure:
        summary = "Motion classification is unavailable."
    else:
        summary = (
            f"Other motion detected ({decision.result.confidence:.1%} confidence)."
        )
    return (
        f"{summary}\n{decision.diagnostic}\n\n"
        "The current loop detector is gait-specific. Analyze anyway?"
    )


def _character_name(character) -> str:
    if character is None:
        return ""
    return str(
        getattr(character, "LongName", "")
        or getattr(character, "Name", "")
    )


def _diagnostic(reason: str) -> str:
    if reason in {"walk_threshold_passed", "run_threshold_passed"}:
        return ""
    if reason == "specialized_threshold_not_met":
        return "No confident walk/run match."
    if reason == "character_not_characterized":
        return "Select and characterize a Character in Character Controls."
    if reason == "characterization_state_unavailable":
        return "Could not read the current Character's characterization state."
    if reason.startswith("missing_required_joint:"):
        joint = reason.split(":", 1)[1]
        return f"Required characterized joint is missing: {joint}."
    if reason.startswith("clip_too_short:"):
        frame_count = reason.split(":", 1)[1].removesuffix("_frames")
        return f"The current Take is too short after 30 FPS resampling ({frame_count}/45 frames)."
    if reason.startswith("model_load_failed:") or reason == "model_unavailable":
        return "The motion classifier model could not be loaded."
    if "finite" in reason or reason.startswith("invalid_motion:"):
        return "The current Take contains invalid motion data."
    return "Motion classification failed."


class MotionRoutingController:
    def __init__(self, classifier, adapter):
        self._classifier = classifier
        self._adapter = adapter

    def _current_context(self) -> tuple[object, MotionContext]:
        character = self._adapter.get_current_character()
        frame_range = tuple(map(int, self._adapter.get_frame_range()))
        context = MotionContext(
            character_name=_character_name(character),
            take_name=str(self._adapter.get_current_take_name()),
            frame_range=frame_range,
        )
        return character, context

    def classify_current_take(self) -> MotionRouteDecision:
        character, context = self._current_context()
        result = self._classifier.classify_character(
            character,
            frame_range=context.frame_range,
        )
        is_failure = result.window_count == 0
        display_label = "Unable to classify" if is_failure else result.label.title()
        return MotionRouteDecision(
            result=result,
            context=context,
            display_label=display_label,
            diagnostic=_diagnostic(result.reason),
            is_failure=is_failure,
            requires_confirmation=is_failure or result.label == "other",
        )

    def context_is_current(self, context: MotionContext) -> bool:
        try:
            _, current = self._current_context()
        except Exception:
            return False
        return current == context
