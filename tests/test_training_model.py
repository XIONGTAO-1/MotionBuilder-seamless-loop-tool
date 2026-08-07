import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from core.motion_classifier import NumpyRandomForest
from training.model_export import export_random_forest_json
from training.thresholds import (
    aggregate_clip_probabilities,
    select_specialized_threshold,
    tune_clip_walk_run_thresholds,
)
from training.train_motion_router import _training_configuration, _write_csv


def test_clip_probability_aggregation_averages_windows_in_stable_order():
    labels = np.array(["walk", "run", "walk"])
    probabilities = np.array(
        [
            [0.1, 0.1, 0.8],
            [0.1, 0.8, 0.1],
            [0.3, 0.1, 0.6],
        ]
    )

    clip_labels, clip_probabilities, clip_ids = aggregate_clip_probabilities(
        labels,
        probabilities,
        ["walk-b", "run-a", "walk-b"],
    )

    assert clip_ids == ["run-a", "walk-b"]
    assert clip_labels.tolist() == ["run", "walk"]
    np.testing.assert_allclose(
        clip_probabilities,
        [[0.1, 0.8, 0.1], [0.2, 0.1, 0.7]],
    )


def test_clip_probability_aggregation_rejects_mixed_labels():
    with pytest.raises(ValueError, match="multiple labels"):
        aggregate_clip_probabilities(
            ["walk", "run"],
            np.array([[0.1, 0.1, 0.8], [0.1, 0.8, 0.1]]),
            ["same", "same"],
        )


def test_clip_probability_aggregation_rejects_mismatched_rows():
    with pytest.raises(ValueError, match="same rows"):
        aggregate_clip_probabilities(
            ["walk", "walk"],
            np.array([[0.1, 0.1, 0.8]]),
            ["walk-a", "walk-a"],
        )


def test_clip_probability_aggregation_rejects_empty_clip_id():
    with pytest.raises(ValueError, match="non-empty"):
        aggregate_clip_probabilities(
            ["walk"],
            np.array([[0.1, 0.1, 0.8]]),
            [""],
        )


def test_clip_probability_aggregation_rejects_null_clip_id():
    with pytest.raises(ValueError, match="non-empty"):
        aggregate_clip_probabilities(
            ["walk"],
            np.array([[0.1, 0.1, 0.8]]),
            [None],
        )


def test_clip_threshold_tuning_counts_clips_and_reports_calibration_level():
    classes = np.array(["other", "run", "walk"])
    labels = []
    probabilities = []
    clip_ids = []
    for target_class, probability in (
        ("walk", [0.1, 0.1, 0.8]),
        ("run", [0.1, 0.8, 0.1]),
    ):
        for clip_index in range(20):
            for _ in range(2):
                labels.append(target_class)
                probabilities.append(probability)
                clip_ids.append(f"{target_class}-{clip_index:02d}")

    thresholds, details, warnings, summary = tune_clip_walk_run_thresholds(
        labels,
        np.asarray(probabilities),
        classes,
        clip_ids,
    )

    assert thresholds == {"walk": 0.4, "run": 0.4}
    assert details["walk"]["prediction_count"] == 20
    assert details["run"]["prediction_count"] == 20
    assert summary == {
        "level": "clip",
        "probability_source": "training_oof",
        "clip_count": 40,
        "class_counts": {"other": 0, "run": 20, "walk": 20},
    }
    assert warnings == []


def test_training_configuration_records_clip_threshold_calibration():
    calibration = {
        "level": "clip",
        "probability_source": "training_oof",
        "clip_count": 40,
        "class_counts": {"other": 0, "run": 20, "walk": 20},
    }
    arguments = SimpleNamespace(
        random_state=42,
        search_iterations=30,
        max_windows_per_source_class=10,
        dataset_root=Path("dataset"),
    )

    config = _training_configuration(arguments, calibration)

    assert config["threshold_calibration"] == calibration


def test_threshold_selects_lowest_value_meeting_precision_and_count():
    classes = np.array(["other", "run", "walk"])
    labels = np.array(["walk"] * 25 + ["other"] * 3)
    probabilities = np.zeros((28, 3), dtype=np.float64)
    probabilities[:25] = [0.1, 0.1, 0.8]
    probabilities[25:] = [0.35, 0.05, 0.60]

    selection = select_specialized_threshold(labels, probabilities, classes, "walk")
    assert selection["threshold"] == 0.61
    assert selection["precision"] == 1.0
    assert selection["prediction_count"] == 25
    assert selection["quality_warning"] is None


def test_threshold_falls_back_to_f05_with_warning():
    classes = np.array(["other", "run", "walk"])
    labels = np.array(["walk"] * 4 + ["other"] * 4)
    probabilities = np.array(
        [[0.1, 0.1, 0.8]] * 4 + [[0.2, 0.1, 0.7]] * 4,
        dtype=np.float64,
    )
    selection = select_specialized_threshold(labels, probabilities, classes, "walk")
    assert 0.4 <= selection["threshold"] <= 0.95
    assert selection["quality_warning"]


class _FakeTree:
    children_left = np.array([1, -1, -1], dtype=np.int64)
    children_right = np.array([2, -1, -1], dtype=np.int64)
    feature = np.array([0, -2, -2], dtype=np.int64)
    threshold = np.array([0.5, -2.0, -2.0], dtype=np.float64)
    value = np.array([[[5.0, 1.0, 4.0]], [[5.0, 0.0, 0.0]], [[0.0, 1.0, 4.0]]])


class _FakeEstimator:
    tree_ = _FakeTree()


class _FakeForest:
    classes_ = np.array(["other", "run", "walk"])
    estimators_ = [_FakeEstimator()]

    def get_params(self, deep=False):
        return {"n_estimators": 1, "random_state": 42}


def test_json_export_is_hashable_and_loadable_by_numpy_runtime(tmp_path):
    path = tmp_path / "motion_router_v1.json"
    payload = export_random_forest_json(
        _FakeForest(),
        path,
        thresholds={"walk": 0.6, "run": 0.7},
        training_config={"seed": 42},
        sklearn_version="test",
        deployment_status="production",
        quality_warnings=[],
    )

    raw = path.read_bytes()
    assert (tmp_path / "motion_router_v1.sha256").read_text().strip() == hashlib.sha256(raw).hexdigest()
    assert payload["trees"][0]["probabilities"][2] == [0.0, 0.2, 0.8]

    forest = NumpyRandomForest.load(path)
    features = np.zeros((2, 175), dtype=np.float64)
    features[1, 0] = 1.0
    np.testing.assert_allclose(forest.predict_proba(features), [[1.0, 0.0, 0.0], [0.0, 0.2, 0.8]])


def test_audit_csv_uses_repository_safe_lf_line_endings(tmp_path):
    path = tmp_path / "audit.csv"
    _write_csv(path, [{"value": "ok"}], ("value",))
    assert b"\r" not in path.read_bytes()
