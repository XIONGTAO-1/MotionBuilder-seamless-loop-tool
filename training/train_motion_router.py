#!/usr/bin/env python3
"""Train, evaluate, and export the AMASS/BABEL walk/run/other router."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Sequence

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPOSITORY_ROOT))
sys.path.insert(0, str(REPOSITORY_ROOT / "src"))

import numpy as np

from core.motion_classifier import NumpyRandomForest, route_probabilities
from core.motion_features import write_feature_schema
from training.dataset import (
    assert_group_disjoint,
    load_descriptor_dataset,
    load_window_manifest,
    select_training_windows,
)
from training.model_export import export_random_forest_json
from training.thresholds import tune_clip_walk_run_thresholds


CLASS_REPORT_ORDER = ("walk", "run", "other")
SEARCH_SPACE = {
    "n_estimators": [200, 400, 600],
    "max_depth": [8, 12, 16],
    "min_samples_leaf": [1, 2, 4, 8],
    "min_samples_split": [2, 5, 10],
    "max_features": ["sqrt", 0.3, 0.5],
    "max_samples": [0.6, 0.8, 1.0],
}


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Cannot serialize {type(value).__name__}")


def _write_csv(path: Path, rows: Sequence[dict], fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=fieldnames,
            extrasaction="ignore",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)


def _route_rows(probabilities: np.ndarray, classes: Sequence[str], thresholds: dict) -> list[str]:
    return [route_probabilities(row, classes, thresholds)[0] for row in probabilities]


def _classification_metrics(y_true, y_pred) -> dict:
    from sklearn.metrics import precision_recall_fscore_support

    precision, recall, f1, support = precision_recall_fscore_support(
        y_true,
        y_pred,
        labels=CLASS_REPORT_ORDER,
        zero_division=0,
    )
    per_class = {
        name: {
            "precision": float(precision[index]),
            "recall": float(recall[index]),
            "f1": float(f1[index]),
            "support": int(support[index]),
        }
        for index, name in enumerate(CLASS_REPORT_ORDER)
    }
    return {"per_class": per_class, "macro_f1": float(np.mean(f1))}


def _aggregate_clips(
    probabilities: np.ndarray,
    classes: Sequence[str],
    thresholds: dict,
    metadata: Sequence[dict],
) -> tuple[np.ndarray, np.ndarray, list[dict]]:
    groups: dict[str, list[int]] = defaultdict(list)
    for index, row in enumerate(metadata):
        groups[str(row["source_clip_id"])].append(index)

    labels: list[str] = []
    predictions: list[str] = []
    rows: list[dict] = []
    for clip_id in sorted(groups):
        indices = groups[clip_id]
        clip_labels = {metadata[index]["coarse_class"] for index in indices}
        if len(clip_labels) != 1:
            raise ValueError(f"Clip {clip_id} contains multiple coarse classes")
        mean_probability = probabilities[indices].mean(axis=0)
        prediction, reason = route_probabilities(mean_probability, classes, thresholds)
        label = next(iter(clip_labels))
        probability_map = {
            name: float(mean_probability[index]) for index, name in enumerate(classes)
        }
        labels.append(label)
        predictions.append(prediction)
        rows.append(
            {
                "source_clip_id": clip_id,
                "source_path": metadata[indices[0]]["source_path"],
                "category": metadata[indices[0]]["category"],
                "true_class": label,
                "predicted_class": prediction,
                "probability_walk": probability_map["walk"],
                "probability_run": probability_map["run"],
                "probability_other": probability_map["other"],
                "window_count": len(indices),
                "reason": reason,
            }
        )
    return np.asarray(labels), np.asarray(predictions), rows


def _confusion_rows(y_true: Sequence[str], y_pred: Sequence[str]) -> list[dict]:
    from sklearn.metrics import confusion_matrix

    matrix = confusion_matrix(y_true, y_pred, labels=CLASS_REPORT_ORDER)
    rows: list[dict] = []
    for row_index, true_class in enumerate(CLASS_REPORT_ORDER):
        rows.append(
            {
                "true_class": true_class,
                **{
                    f"predicted_{predicted_class}": int(matrix[row_index, column_index])
                    for column_index, predicted_class in enumerate(CLASS_REPORT_ORDER)
                },
            }
        )
    return rows


def _other_subcategory_recall(metadata: Sequence[dict], predictions: Sequence[str]) -> dict[str, dict]:
    totals: dict[str, int] = defaultdict(int)
    correct: dict[str, int] = defaultdict(int)
    for row, prediction in zip(metadata, predictions):
        if row["coarse_class"] != "other":
            continue
        category = str(row["category"])
        totals[category] += 1
        correct[category] += int(prediction == "other")
    return {
        category: {
            "recall": correct[category] / totals[category],
            "support": totals[category],
        }
        for category in sorted(totals)
    }


def _training_configuration(
    arguments: argparse.Namespace,
    threshold_calibration: dict,
) -> dict:
    return {
        "python_version": ".".join(map(str, sys.version_info[:3])),
        "random_state": arguments.random_state,
        "cv_splits": 5,
        "search_iterations": arguments.search_iterations,
        "scoring": "f1_macro",
        "max_windows_per_source_class": arguments.max_windows_per_source_class,
        "target_fps": 30.0,
        "window_frames": 45,
        "window_stride_frames": 15,
        "search_space": SEARCH_SPACE,
        "dataset_root": str(arguments.dataset_root.resolve()),
        "threshold_calibration": threshold_calibration,
    }


def train(arguments: argparse.Namespace) -> dict:
    if sys.version_info[:2] != (3, 11):
        raise RuntimeError(f"Training requires Python 3.11, found {sys.version.split()[0]}")

    import sklearn
    from sklearn.base import clone
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import RandomizedSearchCV, StratifiedGroupKFold, cross_val_predict

    if sklearn.__version__ != "1.9.0":
        raise RuntimeError(f"Training requires scikit-learn 1.9.0, found {sklearn.__version__}")

    output_root = arguments.output_root.resolve()
    models_dir = output_root / "models"
    reports_dir = output_root / "reports"
    models_dir.mkdir(parents=True, exist_ok=True)
    reports_dir.mkdir(parents=True, exist_ok=True)
    write_feature_schema(models_dir / "feature_schema_v1.json")

    all_train_rows = load_window_manifest(arguments.dataset_root, split="train")
    selected_rows = select_training_windows(
        all_train_rows,
        max_per_source_class=arguments.max_windows_per_source_class,
    )
    validation_rows = load_window_manifest(arguments.dataset_root, split="val")
    X_train, y_train, train_metadata = load_descriptor_dataset(arguments.dataset_root, selected_rows)
    X_validation, y_validation, validation_metadata = load_descriptor_dataset(
        arguments.dataset_root, validation_rows
    )
    missing_train = sorted(set(CLASS_REPORT_ORDER).difference(y_train.tolist()))
    if missing_train:
        raise ValueError(f"Training data is missing classes: {missing_train}")

    groups = np.asarray([row["source_path"] for row in train_metadata])
    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=arguments.random_state)
    folds = list(splitter.split(X_train, y_train, groups))
    for train_indices, validation_indices in folds:
        assert_group_disjoint(train_indices, validation_indices, groups)

    base_forest = RandomForestClassifier(
        bootstrap=True,
        class_weight="balanced_subsample",
        random_state=arguments.random_state,
        n_jobs=arguments.n_jobs,
    )
    search = RandomizedSearchCV(
        base_forest,
        SEARCH_SPACE,
        n_iter=arguments.search_iterations,
        scoring="f1_macro",
        cv=folds,
        random_state=arguments.random_state,
        n_jobs=arguments.n_jobs,
        refit=True,
        verbose=arguments.verbose,
    )
    search.fit(X_train, y_train)

    best_forest = search.best_estimator_
    oof_probabilities = cross_val_predict(
        clone(best_forest),
        X_train,
        y_train,
        cv=folds,
        method="predict_proba",
        n_jobs=arguments.n_jobs,
    )
    classes = best_forest.classes_
    thresholds, threshold_details, quality_warnings, threshold_calibration = (
        tune_clip_walk_run_thresholds(
            y_train,
            oof_probabilities,
            classes,
            [row["source_clip_id"] for row in train_metadata],
        )
    )

    validation_probabilities = best_forest.predict_proba(X_validation)
    window_predictions = np.asarray(_route_rows(validation_probabilities, classes, thresholds))
    clip_labels, clip_predictions, clip_rows = _aggregate_clips(
        validation_probabilities,
        classes,
        thresholds,
        validation_metadata,
    )
    window_metrics = _classification_metrics(y_validation, window_predictions)
    clip_metrics = _classification_metrics(clip_labels, clip_predictions)
    missing_validation = sorted(set(CLASS_REPORT_ORDER).difference(clip_labels.tolist()))
    if missing_validation:
        quality_warnings.append(f"Validation clips are missing classes: {missing_validation}")

    gates = {
        "walk_precision_at_least_0_90": clip_metrics["per_class"]["walk"]["precision"] >= 0.90,
        "run_precision_at_least_0_85": clip_metrics["per_class"]["run"]["precision"] >= 0.85,
        "macro_f1_at_least_0_70": clip_metrics["macro_f1"] >= 0.70,
        "validation_has_all_classes": not missing_validation,
    }
    deployment_status = "production" if all(gates.values()) and not quality_warnings else "experimental"
    if deployment_status == "experimental" and not quality_warnings:
        quality_warnings.append("One or more deployment quality gates failed")

    training_config = _training_configuration(arguments, threshold_calibration)
    model_path = models_dir / "motion_router_v1.json"
    export_random_forest_json(
        best_forest,
        model_path,
        thresholds=thresholds,
        training_config=training_config,
        sklearn_version=sklearn.__version__,
        deployment_status=deployment_status,
        quality_warnings=quality_warnings,
    )

    numpy_forest = NumpyRandomForest.load(model_path, allow_experimental=True)
    numpy_probabilities = numpy_forest.predict_proba(X_validation)
    maximum_probability_error = float(np.max(np.abs(numpy_probabilities - validation_probabilities)))
    numpy_labels = np.asarray(_route_rows(numpy_probabilities, numpy_forest.classes, thresholds))
    label_match = bool(np.array_equal(numpy_labels, window_predictions))
    if maximum_probability_error > 1e-7 or not label_match:
        raise RuntimeError(
            "Exported NumPy model failed parity validation: "
            f"max_error={maximum_probability_error}, label_match={label_match}"
        )

    selected_fields = (
        "window_id",
        "source_clip_id",
        "source_path",
        "category",
        "coarse_class",
        "subject",
        "window_start_frame",
        "window_end_frame",
        "output_path",
    )
    _write_csv(reports_dir / "selected_training_windows.csv", train_metadata, selected_fields)
    _write_csv(
        reports_dir / "validation_predictions.csv",
        clip_rows,
        (
            "source_clip_id",
            "source_path",
            "category",
            "true_class",
            "predicted_class",
            "probability_walk",
            "probability_run",
            "probability_other",
            "window_count",
            "reason",
        ),
    )
    confusion_rows = _confusion_rows(y_validation, window_predictions)
    _write_csv(
        reports_dir / "confusion_matrix.csv",
        confusion_rows,
        ("true_class", "predicted_walk", "predicted_run", "predicted_other"),
    )

    metrics = {
        "deployment_status": deployment_status,
        "quality_warnings": quality_warnings,
        "quality_gates": gates,
        "training_window_count": len(X_train),
        "validation_window_count": len(X_validation),
        "validation_clip_count": len(clip_rows),
        "selected_class_counts": {
            name: int(np.sum(y_train == name)) for name in CLASS_REPORT_ORDER
        },
        "best_parameters": search.best_params_,
        "best_cross_validation_f1_macro": float(search.best_score_),
        "thresholds": thresholds,
        "threshold_selection": threshold_details,
        "threshold_calibration": threshold_calibration,
        "clip_level": clip_metrics,
        "window_level": window_metrics,
        "other_subcategory_recall": _other_subcategory_recall(
            validation_metadata, window_predictions
        ),
        "numpy_json_parity": {
            "maximum_probability_error": maximum_probability_error,
            "labels_identical": label_match,
        },
        "training_config": training_config,
        "scikit_learn_version": sklearn.__version__,
        "numpy_version": np.__version__,
    }
    (reports_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, sort_keys=True, default=_json_default) + "\n",
        encoding="utf-8",
    )
    return metrics


def parse_arguments(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, default=REPOSITORY_ROOT)
    parser.add_argument("--max-windows-per-source-class", type=int, default=10)
    parser.add_argument("--search-iterations", type=int, default=30)
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument("--n-jobs", type=int, default=-1)
    parser.add_argument("--verbose", type=int, default=1)
    return parser.parse_args(argv)


def main(argv: Sequence[str] | None = None) -> int:
    metrics = train(parse_arguments(argv))
    print(json.dumps({
        "deployment_status": metrics["deployment_status"],
        "clip_level": metrics["clip_level"],
        "quality_warnings": metrics["quality_warnings"],
    }, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
