import json
import hashlib

import numpy as np
import pytest

from core.motion_classifier import NumpyRandomForest, classify_feature_windows, route_probabilities
from core.motion_features import FEATURE_SCHEMA_HASH


def _model_payload(status: str = "production") -> dict:
    leaf_other = {
        "children_left": [-1],
        "children_right": [-1],
        "feature": [-2],
        "threshold": [-2.0],
        "probabilities": [[1.0, 0.0, 0.0]],
    }
    split_tree = {
        "children_left": [1, -1, -1],
        "children_right": [2, -1, -1],
        "feature": [0, -2, -2],
        "threshold": [0.5, -2.0, -2.0],
        "probabilities": [[0.5, 0.1, 0.4], [1.0, 0.0, 0.0], [0.0, 0.2, 0.8]],
    }
    return {
        "model_format_version": 1,
        "classes": ["other", "run", "walk"],
        "feature_schema_hash": FEATURE_SCHEMA_HASH,
        "descriptor_size": 175,
        "thresholds": {"walk": 0.6, "run": 0.7},
        "deployment_status": status,
        "trees": [leaf_other, split_tree],
    }


def test_numpy_forest_traverses_trees_and_averages_probabilities(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(_model_payload()), encoding="utf-8")
    forest = NumpyRandomForest.load(path)

    features = np.zeros((2, 175), dtype=np.float32)
    features[1, 0] = 1.0
    probabilities = forest.predict_proba(features)
    np.testing.assert_allclose(probabilities[0], [1.0, 0.0, 0.0])
    np.testing.assert_allclose(probabilities[1], [0.5, 0.1, 0.4])


def test_experimental_model_requires_explicit_opt_in(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(_model_payload("experimental")), encoding="utf-8")
    with pytest.raises(ValueError, match="experimental"):
        NumpyRandomForest.load(path)
    assert NumpyRandomForest.load(path, allow_experimental=True).deployment_status == "experimental"


def test_schema_mismatch_and_malformed_tree_are_rejected(tmp_path):
    payload = _model_payload()
    payload["feature_schema_hash"] = "wrong"
    path = tmp_path / "model.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="schema"):
        NumpyRandomForest.load(path)

    payload = _model_payload()
    payload["trees"][0]["probabilities"] = [[0.2, 0.3]]
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(ValueError, match="tree"):
        NumpyRandomForest.load(path)


def test_existing_checksum_and_cyclic_tree_are_rejected(tmp_path):
    path = tmp_path / "model.json"
    raw = json.dumps(_model_payload()).encode()
    path.write_bytes(raw)
    path.with_suffix(".sha256").write_text("0" * 64 + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match="checksum"):
        NumpyRandomForest.load(path)

    path.with_suffix(".sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n")
    payload = _model_payload()
    payload["trees"][0] = {
        "children_left": [0],
        "children_right": [0],
        "feature": [0],
        "threshold": [0.5],
        "probabilities": [[1.0, 0.0, 0.0]],
    }
    raw = json.dumps(payload).encode()
    path.write_bytes(raw)
    path.with_suffix(".sha256").write_text(hashlib.sha256(raw).hexdigest() + "\n")
    with pytest.raises(ValueError, match="tree"):
        NumpyRandomForest.load(path)


def test_conservative_routing_and_window_aggregation(tmp_path):
    path = tmp_path / "model.json"
    path.write_text(json.dumps(_model_payload()), encoding="utf-8")
    forest = NumpyRandomForest.load(path)

    assert route_probabilities([0.10, 0.20, 0.70], forest.classes, forest.thresholds)[0] == "walk"
    assert route_probabilities([0.10, 0.72, 0.18], forest.classes, forest.thresholds)[0] == "run"
    assert route_probabilities([0.15, 0.44, 0.41], forest.classes, forest.thresholds)[0] == "other"
    assert route_probabilities([0.05, 0.48, 0.47], forest.classes, {"walk": 0.4, "run": 0.4})[0] == "run"

    features = np.ones((3, 175), dtype=np.float32)
    result = classify_feature_windows(forest, features)
    assert result.label == "other"
    assert result.window_count == 3
    assert set(result.probabilities) == {"other", "run", "walk"}
