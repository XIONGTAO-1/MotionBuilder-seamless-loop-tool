"""Versioned, portable JSON export for scikit-learn RandomForestClassifier."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from core.motion_features import DESCRIPTOR_SIZE, FEATURE_SCHEMA_HASH


def _json_default(value: Any) -> Any:
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        return float(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"Object of type {type(value).__name__} is not JSON serializable")


def _export_tree(estimator: Any, class_count: int) -> dict:
    tree = estimator.tree_
    values = np.asarray(tree.value, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 1 or values.shape[2] != class_count:
        raise ValueError(f"Unexpected scikit-learn tree value shape: {values.shape}")
    values = values[:, 0, :]
    totals = values.sum(axis=1, keepdims=True)
    if np.any(totals <= 0.0):
        raise ValueError("A tree node has no class samples")
    probabilities = values / totals
    return {
        "children_left": np.asarray(tree.children_left, dtype=np.int64).tolist(),
        "children_right": np.asarray(tree.children_right, dtype=np.int64).tolist(),
        "feature": np.asarray(tree.feature, dtype=np.int64).tolist(),
        "threshold": np.asarray(tree.threshold, dtype=np.float64).tolist(),
        "probabilities": probabilities.tolist(),
    }


def export_random_forest_json(
    model: Any,
    path: Path,
    *,
    thresholds: Mapping[str, float],
    training_config: Mapping[str, Any],
    sklearn_version: str,
    deployment_status: str,
    quality_warnings: Sequence[str],
) -> dict:
    classes = [str(value) for value in model.classes_]
    if set(classes) != {"walk", "run", "other"} or len(classes) != 3:
        raise ValueError("The fitted forest must contain walk, run, and other")
    if deployment_status not in {"production", "experimental"}:
        raise ValueError("deployment_status must be production or experimental")
    trees = [_export_tree(estimator, len(classes)) for estimator in model.estimators_]
    payload = {
        "model_format_version": 1,
        "classes": classes,
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "descriptor_size": DESCRIPTOR_SIZE,
        "thresholds": {"walk": float(thresholds["walk"]), "run": float(thresholds["run"])},
        "deployment_status": deployment_status,
        "quality_warnings": list(quality_warnings),
        "scikit_learn_version": str(sklearn_version),
        "training_config": dict(training_config),
        "forest_parameters": model.get_params(deep=False),
        "trees": trees,
    }

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    raw = (json.dumps(payload, indent=2, sort_keys=True, default=_json_default) + "\n").encode("utf-8")
    output_path.write_bytes(raw)
    digest = hashlib.sha256(raw).hexdigest()
    output_path.with_suffix(".sha256").write_text(digest + "\n", encoding="utf-8")
    return payload
