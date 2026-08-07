"""Conservative walk/run threshold selection from out-of-fold probabilities."""

from __future__ import annotations

from typing import Sequence

import numpy as np


THRESHOLD_GRID = np.round(np.arange(0.40, 0.951, 0.01), 2)


def aggregate_clip_probabilities(
    labels: Sequence[str],
    probabilities: np.ndarray,
    clip_ids: Sequence[str],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Average window probabilities for each consistently labelled source clip."""
    y_true = np.asarray(labels)
    values = np.asarray(probabilities, dtype=np.float64)
    raw_ids = list(clip_ids)
    if any(value is None for value in raw_ids):
        raise ValueError("clip_ids must contain non-empty values")
    ids = np.asarray([str(value) for value in raw_ids])
    if (
        y_true.ndim != 1
        or values.ndim != 2
        or values.shape[0] != len(y_true)
        or len(ids) != len(y_true)
        or len(y_true) == 0
    ):
        raise ValueError("labels, probabilities, and clip_ids must contain the same rows")
    if any(not clip_id.strip() for clip_id in ids.tolist()):
        raise ValueError("clip_ids must contain non-empty values")

    ordered_ids = sorted(set(ids.tolist()))
    aggregated_labels: list[str] = []
    aggregated_probabilities: list[np.ndarray] = []
    for clip_id in ordered_ids:
        indices = np.flatnonzero(ids == clip_id)
        clip_labels = set(y_true[indices].tolist())
        if len(clip_labels) != 1:
            raise ValueError(f"Clip {clip_id} contains multiple labels")
        aggregated_labels.append(str(next(iter(clip_labels))))
        aggregated_probabilities.append(values[indices].mean(axis=0))

    return (
        np.asarray(aggregated_labels),
        np.stack(aggregated_probabilities),
        ordered_ids,
    )


def _candidate_predictions(
    probabilities: np.ndarray,
    classes: np.ndarray,
    target_class: str,
    threshold: float,
) -> np.ndarray:
    target_index = int(np.flatnonzero(classes == target_class)[0])
    competitor = "run" if target_class == "walk" else "walk"
    competitor_index = int(np.flatnonzero(classes == competitor)[0])
    return (probabilities[:, target_index] >= threshold) & (
        probabilities[:, target_index] > probabilities[:, competitor_index]
    )


def select_specialized_threshold(
    labels: Sequence[str],
    probabilities: np.ndarray,
    classes: Sequence[str],
    target_class: str,
    required_precision: float = 0.90,
    minimum_predictions: int = 20,
) -> dict:
    if target_class not in {"walk", "run"}:
        raise ValueError("target_class must be walk or run")
    y_true = np.asarray(labels)
    values = np.asarray(probabilities, dtype=np.float64)
    class_values = np.asarray(classes)
    if values.shape != (len(y_true), len(class_values)):
        raise ValueError("probabilities shape does not match labels and classes")
    if set(class_values.tolist()) != {"walk", "run", "other"}:
        raise ValueError("classes must contain walk, run, and other")

    positive_count = int(np.sum(y_true == target_class))
    candidates: list[dict] = []
    for threshold in THRESHOLD_GRID:
        predicted = _candidate_predictions(values, class_values, target_class, float(threshold))
        prediction_count = int(np.sum(predicted))
        true_positives = int(np.sum(predicted & (y_true == target_class)))
        precision = true_positives / prediction_count if prediction_count else 0.0
        recall = true_positives / positive_count if positive_count else 0.0
        beta_squared = 0.25
        denominator = beta_squared * precision + recall
        f05 = (1.0 + beta_squared) * precision * recall / denominator if denominator else 0.0
        candidates.append(
            {
                "threshold": float(threshold),
                "precision": float(precision),
                "recall": float(recall),
                "f0_5": float(f05),
                "prediction_count": prediction_count,
            }
        )

    for candidate in candidates:
        if (
            candidate["precision"] >= required_precision
            and candidate["prediction_count"] >= minimum_predictions
        ):
            candidate["quality_warning"] = None
            return candidate

    best = max(candidates, key=lambda item: (item["f0_5"], item["precision"], -item["threshold"]))
    best = dict(best)
    best["quality_warning"] = (
        f"{target_class} did not reach precision >= {required_precision:.2f} "
        f"with at least {minimum_predictions} out-of-fold predictions; selected maximum F0.5"
    )
    return best


def tune_walk_run_thresholds(
    labels: Sequence[str],
    probabilities: np.ndarray,
    classes: Sequence[str],
) -> tuple[dict[str, float], dict[str, dict], list[str]]:
    details = {
        target: select_specialized_threshold(labels, probabilities, classes, target)
        for target in ("walk", "run")
    }
    thresholds = {target: details[target]["threshold"] for target in details}
    warnings = [
        details[target]["quality_warning"]
        for target in details
        if details[target]["quality_warning"] is not None
    ]
    return thresholds, details, warnings


def tune_clip_walk_run_thresholds(
    labels: Sequence[str],
    probabilities: np.ndarray,
    classes: Sequence[str],
    clip_ids: Sequence[str],
) -> tuple[dict[str, float], dict[str, dict], list[str], dict]:
    """Tune conservative thresholds from clip-averaged training OOF probabilities."""
    clip_labels, clip_probabilities, aggregated_ids = aggregate_clip_probabilities(
        labels,
        probabilities,
        clip_ids,
    )
    thresholds, details, warnings = tune_walk_run_thresholds(
        clip_labels,
        clip_probabilities,
        classes,
    )
    summary = {
        "level": "clip",
        "probability_source": "training_oof",
        "clip_count": len(aggregated_ids),
        "class_counts": {
            str(name): int(np.sum(clip_labels == name)) for name in sorted(classes)
        },
    }
    return thresholds, details, warnings, summary
